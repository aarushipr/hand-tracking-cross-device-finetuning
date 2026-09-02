"""
HOT3D full (VRS-based) detection loader -- reads Meta's own pre-computed,
occlusion-aware 2D hand bounding boxes from box2d_hands.csv directly,
instead of re-deriving boxes from 3D keypoint projection the way
HOT3DDetectionDataset.py (built against the lightweight HOT3D-Clips /
WebDataset format) has to.

Why this exists (see prior HOT3DDetectionDataset.py visual verification):
projecting 3D hand keypoints through the camera model only tells you where
the hand *geometrically* is relative to the camera frustum -- it can't know
the hand is occluded by the body, an object, or the headset frame. That was
the irreducible remaining error after fixing camera-stream selection and
adding a behind-camera depth guard. The full VRS-based HOT3D dataset ships
box2d_hands.csv, which is Meta's own ground-truth 2D box per hand per
camera stream per timestamp, INCLUDING a visibility_ratio field -- i.e.
already occlusion-aware, no projection math needed at all here. NOTE:
visibility_ratio is a 0.0-1.0 fraction (fully-visible hands read 1.0), NOT
a 0-100 percent as an earlier version of this docstring assumed -- that
mismatch made min_visibility_ratio's old default of 20.0 an unreachable
threshold that silently dropped every box (see verify_hot3d_vrs_visual.py
run history). Confirmed via debug_boxes.py against a real sample.

Source of truth for the CSV schema / reader API (Apache 2.0, read not
vendored -- imported directly from the cloned hot3d repo, same repo already
cloned on the cluster to build the `hot3d` conda env for downloading):
    hot3d/hot3d/data_loaders/HandBox2dDataProvider.py
    hot3d/hot3d/data_loaders/PathProvider.py
    hot3d/hot3d/data_loaders/AriaDataProvider.py

Deliberately does NOT build a full Hot3dDataProvider (dataset_api.py) --
that requires an ObjectLibrary (object models + eval sets), which would mean
downloading the whole Hot3DAssets bundle just to read hand boxes we don't
need object data for at all. AriaDataProvider (image reads from
recording.vrs) + HandBox2dProvider (box2d_hands.csv) are self-contained and
don't need it.

hand_index -> left/right slot mapping is ASSUMED (0=left, 1=right, matching
UmeTrack's HandSide convention used everywhere else in this project) but
NOT yet confirmed against this specific CSV -- verify visually with
verify_hot3d_vrs_visual.py the same way EgoHands' class-index assumption
was checked (verify_egohands_classes.py) before trusting it for real
training.

SAMPLE INDEX, CACHING, AND FRAME STRIDE
---------------------------------------
The index holds only plain data -- sequence, image stream, timestamp,
headset -- and never a live provider object. An earlier version stored the
open AriaDataProvider in every sample tuple, which had three consequences,
all of which become severe once the mixed Aria+Quest split raises the
sequence count:

1. Every sequence's .vrs had to be opened before training could begin, over
   network storage, on every run, with no way to cache the result.
2. DataLoader workers are forked, so they inherited the parent's already-open
   VRS handles. Two processes reading the same C++ file handle is what
   produced garbled timestamps and JPEG decode failures in the keypoint
   pipeline before HOT3DKeypointDataset was restructured the same way. This
   loader had the same defect and, unlike that one, no worker_init to fix it.
3. The index could not be serialised at all, so the cost in (1) was
   unavoidable.

The structure here mirrors HOT3DKeypointDataset's, for the same reasons:

1. The per-sequence index is cached to disk (see index_cache_dir). The cache
   key includes frame_stride and INDEX_FORMAT_VERSION, so changing either --
   or changing the sampling logic here and bumping the version -- invalidates
   stale caches automatically. min_visibility_ratio is deliberately NOT part
   of the key: it filters boxes inside __getitem__ and has no effect on which
   (frame, stream) pairs the index contains. Writes go to a temporary file and
   are then os.replace()d into position, so a job killed mid-write cannot
   leave a half-written index behind.

2. Providers open lazily, on first access to a sample from that sequence, and
   are then kept open. They are deliberately NOT evicted: DataLoader(
   shuffle=True) draws consecutive samples from unrelated sequences, so any
   LRU policy would thrash.

   With num_workers>0, each worker MUST clear this cache in its
   worker_init_fn -- see worker_init() below.

3. frame_stride subsamples timestamps. HOT3D's cameras run at 30 Hz, so
   consecutive frames are near-duplicates. Set frame_stride=1 to reproduce the
   exhaustive sampling this loader used previously.

Frames with no visible hand are kept as genuine exists=0 negatives -- the
index iterates every image stream's own timestamps rather than only the
box2d CSV's, the same role EgoHands fills for DetNet upstream.

Usage:
    ds = HOT3DVRSDetectionDataset(
        sequence_dirs=["/storage/user/praa/hot3d_full_setup/hot3d_repo/hot3d/dataset/P0003_c701bd11"],
        hot3d_repo_root="/storage/user/praa/hot3d_full_setup/hot3d_repo/hot3d",
        min_visibility_ratio=0.2,
        frame_stride=5,
        index_cache_dir="/storage/user/praa/scratch/hot3d_detection_index",
    )
"""
import os
import sys

import numpy as np
import torch

import augmentation
from a_structs import ImageWithBoundingBoxes, bbox


# Bump whenever the meaning of a cached index entry changes, so stale caches
# are ignored rather than silently reused.
INDEX_FORMAT_VERSION = 1


def worker_init(worker_id):
    """
    DataLoader worker_init_fn. Pass this as worker_init_fn whenever
    num_workers > 0.

    Workers are forked processes, so they inherit whatever providers the
    parent already had open. Clearing the cache here forces each worker to
    open its own providers on first access, so no VRS handle is ever shared
    between processes.
    """
    info = torch.utils.data.get_worker_info()
    if info is not None:
        info.dataset._open_sequences = {}


class _SequenceProviders:
    """Everything held open for one HOT3D sequence."""

    def __init__(self, aria_provider, box2d_provider, mono_stream_ids):
        self.aria_provider = aria_provider
        self.box2d_provider = box2d_provider
        self.stream_id_by_str = {str(s): s for s in mono_stream_ids}


class HOT3DVRSDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, sequence_dirs: list, hot3d_repo_root: str,
                 min_visibility_ratio: float = 0.2, margin: float = 0.15,
                 orientation: int = None, frame_stride: int = 1,
                 index_cache_dir: str = None, augment: bool = True):
        """
        orientation: camera mount rotation applied to every frame (and to its
        boxes) BEFORE augmentation, so the hand reaches the network upright.

        This is not cosmetic. `augmentation.augment_image` only ever applies a
        small random rotation of +/- 0.1 pi about the frame centre; it has no
        notion of a camera mount. The upstream corpora DetNet was trained on
        (synthetic renders, EgoHands, EPIC-KITCHENS) are all natively upright,
        so the weights encode upright hands. HOT3D's SLAM cameras are mounted
        sideways and need 270 degrees. Without this, fine-tuning would see
        sideways hands while evaluation (eval_detnet.py, which reproduces
        Monado's own letterbox) sees upright ones, and the fine-tuned model
        would be scored off-distribution -- the exact mismatch the evaluation
        convention exists to prevent.

        Default None resolves per sequence from
        preprocess_baseline.DEVICE_ORIENTATION, which is where the verified
        per-device values live. Pass 0 to disable, for an ablation.

        augment: apply augmentation.augment_image to every sample. True for
        training, and it must be False for the validation set. Augmentation is
        a random draw per sample per epoch, so an augmented validation split
        re-measures a different distribution every epoch; the resulting
        epoch-to-epoch noise is then what early stopping reacts to, rather
        than genuine convergence. This mirrors the determinism argument
        HOT3DKeypointDataset's eval_mode implements for KeyNet.
        """
        # hot3d's data_loaders package uses bare `from data_loaders.X import Y`
        # (relative to hot3d/hot3d), so that directory has to be on sys.path
        # before these imports work -- same pattern as this project's own
        # `import py.training.common.X` needing mercury_train root on
        # sys.path (see augmentation.py / trainer_detection.py __main__).
        if hot3d_repo_root not in sys.path:
            sys.path.insert(0, hot3d_repo_root)

        # Quest 3 HOT3D recordings have no TimeCode reference -- see
        # py/training/common/hot3d_timecode_compat.py's docstring for the
        # full investigation (verified 2026-08-09 against real Quest
        # sequences) and why falling back to DEVICE_TIME here is safe
        # rather than just quieting the crash. No effect on Aria sequences.
        # Defensive sys.path insert: this class is usually imported from a
        # caller that already put mercury_train root on sys.path (e.g.
        # trainer_detection.py's __main__), but don't assume that here.
        _mercury_train_root = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
        if _mercury_train_root not in sys.path:
            sys.path.insert(0, _mercury_train_root)
        from py.training.common.hot3d_timecode_compat import patch as _patch_quest_timecode
        _patch_quest_timecode()

        from data_loaders.PathProvider import Hot3dDataPathProvider
        from data_loaders.HandBox2dDataProvider import load_box2d_trajectory_from_csv
        from data_loaders.AriaDataProvider import AriaDataProvider
        from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
        from projectaria_tools.core.stream_id import StreamId

        self._TimeDomain = TimeDomain
        self._TimeQueryOptions = TimeQueryOptions
        self._StreamId = StreamId
        self._AriaDataProvider = AriaDataProvider
        self._Hot3dDataPathProvider = Hot3dDataPathProvider
        self._load_box2d = load_box2d_trajectory_from_csv

        if frame_stride < 1:
            raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")

        self.sequence_dirs = list(sequence_dirs)
        self.min_visibility_ratio = min_visibility_ratio
        self.margin = margin
        self.orientation_override = orientation
        self.augment = augment
        self.frame_stride = frame_stride
        self.index_cache_dir = index_cache_dir

        # seq_dir -> _SequenceProviders, lazily populated, never evicted.
        # Cleared per worker process by worker_init() above.
        self._open_sequences = {}

        if self.index_cache_dir:
            os.makedirs(self.index_cache_dir, exist_ok=True)

        # Plain-data index only: no provider objects, so it serialises and so
        # forked workers share nothing. See the module docstring.
        seq_idx, stream_str, ts, headset = [], [], [], []
        for i, seq_dir in enumerate(self.sequence_dirs):
            for entry in self._index_for_sequence(seq_dir):
                seq_idx.append(i)
                stream_str.append(entry[0])
                ts.append(entry[1])
                headset.append(entry[2])

        self._seq_idx = np.asarray(seq_idx, dtype=np.int32)
        self._stream_str = np.asarray(stream_str, dtype=np.str_)
        self._ts = np.asarray(ts, dtype=np.int64)
        self._headset = np.asarray(headset, dtype=np.str_)

        print(f"[HOT3DVRSDetectionDataset] {len(self._ts)} samples from "
              f"{len(self.sequence_dirs)} sequences (frame_stride="
              f"{self.frame_stride}, augment={self.augment})")

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    def _cache_path(self, seq_dir):
        if not self.index_cache_dir:
            return None
        name = os.path.basename(os.path.normpath(seq_dir))
        key = f"{name}__stride{self.frame_stride}__v{INDEX_FORMAT_VERSION}"
        return os.path.join(self.index_cache_dir, key + ".npz")

    def _index_for_sequence(self, seq_dir):
        """Entries of (stream_id_str, timestamp_ns, headset), cached to disk."""
        cache_path = self._cache_path(seq_dir)

        if cache_path and os.path.exists(cache_path):
            try:
                with np.load(cache_path, allow_pickle=False) as z:
                    return list(zip(z["stream_str"].tolist(),
                                    z["ts"].tolist(),
                                    z["headset"].tolist()))
            except (OSError, ValueError, KeyError) as e:
                print(f"[HOT3DVRSDetectionDataset] ignoring unreadable index "
                      f"cache {cache_path}: {e}", file=sys.stderr)

        entries = self._build_index_for_sequence(seq_dir)

        if cache_path:
            self._save_index(cache_path, entries)

        return entries

    def _save_index(self, cache_path, entries):
        """
        Write atomically: a job killed mid-write must not leave a half-written
        index that a later run would happily load.
        """
        tmp_path = f"{cache_path}.tmp{os.getpid()}"
        try:
            np.savez(
                tmp_path,
                stream_str=np.asarray([e[0] for e in entries], dtype=np.str_),
                ts=np.asarray([e[1] for e in entries], dtype=np.int64),
                headset=np.asarray([e[2] for e in entries], dtype=np.str_),
            )
            os.replace(f"{tmp_path}.npz", cache_path)
        except OSError as e:
            print(f"[HOT3DVRSDetectionDataset] could not write index cache "
                  f"{cache_path}: {e}", file=sys.stderr)
            for leftover in (tmp_path, f"{tmp_path}.npz"):
                if os.path.exists(leftover):
                    try:
                        os.remove(leftover)
                    except OSError:
                        pass

    def _build_index_for_sequence(self, seq_dir):
        """
        Read one sequence's stream timestamps. Opens the .vrs, so this is the
        expensive path the cache exists to avoid paying twice.
        """
        import py.training.common.hot3d_split as hot3d_split

        paths = self._Hot3dDataPathProvider.fromRecordingFolder(seq_dir)
        if not paths.is_valid():
            print(f"WARNING: {seq_dir} missing required files, skipping")
            return []

        if self._load_box2d(paths.box2d_hands_filepath) is None:
            print(f"WARNING: {seq_dir} has no box2d_hands.csv, skipping")
            return []

        # Read from the sequence's own metadata.json; costs one small JSON
        # read instead of opening the multi-gigabyte .vrs.
        headset = hot3d_split.headset_of(seq_dir) or "Aria"

        # mps_folder_path intentionally omitted -- get_image()/
        # get_image_stream_ids() don't touch MPS data, and pulling in
        # eye-gaze/point-cloud calibration would be dead weight for a
        # detection-only loader.
        aria_provider = self._AriaDataProvider(paths.vrs_filepath,
                                               mps_folder_path=None)

        entries = []
        for stream_id in aria_provider.get_image_stream_ids():
            # Skip the RGB camera (Aria RecordableTypeId 214, "camera-rgb")
            # -- this project's camera model is 2 monochrome cameras, matching
            # the target headset hardware. Keeps only camera-slam-left/-right
            # (type 1201), which are mono and also give the stereo pair the
            # 2-cam model expects. Mixing in RGB frames would crash
            # augmentation/heatmap conversion downstream, which assumes
            # single-channel input. (Quest recordings carry no RGB stream at
            # all, so this is a no-op there.)
            if str(stream_id).startswith("214-"):
                continue

            timestamps = aria_provider.get_sequence_timestamps(
                stream_id, self._TimeDomain.TIME_CODE)
            for ts in timestamps[::self.frame_stride]:
                entries.append((str(stream_id), int(ts), headset))

        return entries

    # ------------------------------------------------------------------
    # Providers
    # ------------------------------------------------------------------

    def _providers_for(self, seq_dir):
        bundle = self._open_sequences.get(seq_dir)
        if bundle is not None:
            return bundle

        paths = self._Hot3dDataPathProvider.fromRecordingFolder(seq_dir)
        aria_provider = self._AriaDataProvider(paths.vrs_filepath,
                                               mps_folder_path=None)
        box2d_provider = self._load_box2d(paths.box2d_hands_filepath)
        mono_stream_ids = [s for s in aria_provider.get_image_stream_ids()
                           if not str(s).startswith("214-")]

        bundle = _SequenceProviders(aria_provider, box2d_provider,
                                    mono_stream_ids)
        self._open_sequences[seq_dir] = bundle
        return bundle

    def __len__(self):
        return len(self._ts)

    def __getitem__(self, idx):
        seq_dir = self.sequence_dirs[int(self._seq_idx[idx])]
        stream_str = str(self._stream_str[idx])
        ts = int(self._ts[idx])
        headset = str(self._headset[idx])

        bundle = self._providers_for(seq_dir)
        stream_id = bundle.stream_id_by_str.get(stream_str)
        if stream_id is None:
            # The cached index names a stream this recording no longer
            # exposes. Rare, and a stale cache rather than bad data, but
            # returning a labelled-empty sample is safer than raising inside
            # a DataLoader worker.
            return self._empty_sample()
        box2d_provider = bundle.box2d_provider

        image = bundle.aria_provider.get_image(ts, stream_id)
        if image is None:
            return self._empty_sample()

        from py.evaluation import preprocess_baseline as _pp
        orientation = (self.orientation_override if self.orientation_override is not None
                       else _pp.DEVICE_ORIENTATION.get(headset, 270))
        raw_h, raw_w = image.shape[:2]
        image = _pp.rotate_upright(image, orientation)

        bbox_list = [None, None]

        if self._StreamId(str(stream_id)) in box2d_provider.stream_ids:
            result = box2d_provider.get_bbox_at_timestamp(
                stream_id=stream_id,
                timestamp_ns=ts,
                time_query_options=self._TimeQueryOptions.CLOSEST,
                time_domain=self._TimeDomain.TIME_CODE,
            )
            if result is not None:
                for hand_index, hand_box in result.box2d_collection.box2ds.items():
                    if hand_box.box2d is None:
                        continue
                    if hand_box.visibility_ratio is None:
                        continue
                    if hand_box.visibility_ratio < self.min_visibility_ratio:
                        continue

                    b2d = hand_box.box2d
                    # Rotate the box with the image, then re-derive an
                    # axis-aligned box, before applying the margin.
                    corners = _pp.rotate_points_upright(
                        [[b2d.left, b2d.top], [b2d.right, b2d.bottom]],
                        orientation, raw_w, raw_h)
                    left, top = float(corners[:, 0].min()), float(corners[:, 1].min())
                    right, bottom = float(corners[:, 0].max()), float(corners[:, 1].max())

                    w, h = right - left, bottom - top
                    x0 = left - w * self.margin
                    x1 = right + w * self.margin
                    y0 = top - h * self.margin
                    y1 = bottom + h * self.margin

                    b = bbox((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)
                    # ASSUMED 0=left, 1=right -- not yet visually confirmed
                    # for this CSV, see module docstring.
                    slot = 0 if hand_index == 0 else 1
                    bbox_list[slot] = b

        e = ImageWithBoundingBoxes(image=image, bboxes=bbox_list)
        if self.augment:
            e = augmentation.augment_image(e)
        e = augmentation.imgwithboundingboxes320_to_heatmaps_2hand(e)
        return e

    def _empty_sample(self):
        import header
        blank = np.zeros((header.model_input_height, header.model_input_width), dtype=np.uint8)
        e = ImageWithBoundingBoxes(image=blank, bboxes=[None, None])
        return augmentation.imgwithboundingboxes320_to_heatmaps_2hand(e)


if __name__ == "__main__":
    import argparse

    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))

    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-dirs", required=True, nargs="+")
    parser.add_argument("--hot3d-repo-root", required=True)
    parser.add_argument("--min-visibility-ratio", type=float, default=0.2)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--index-cache-dir", default=None)
    args = parser.parse_args()

    ds = HOT3DVRSDetectionDataset(
        sequence_dirs=args.sequence_dirs,
        hot3d_repo_root=args.hot3d_repo_root,
        min_visibility_ratio=args.min_visibility_ratio,
        frame_stride=args.frame_stride,
        index_cache_dir=args.index_cache_dir,
    )
    print(f"{len(ds)} samples")
    samp = ds[0]
    print({k: (v.shape if hasattr(v, "shape") else v) for k, v in samp.items()})
