"""
HOT3D detection loader for the full VRS dataset. Reads Meta's occlusion-aware 2D
boxes from box2d_hands.csv instead of projecting 3D keypoints, which cannot know a
hand is hidden. visibility_ratio is a 0.0-1.0 fraction, not a percentage, and
hand_index is assumed 0=left, 1=right (confirm with verify_hot3d_vrs_visual.py).
The index holds plain data, never a live provider, and is cached per sequence;
with num_workers>0 each worker must clear it, see worker_init(). Hand-free frames
are kept as genuine exists=0 negatives.
"""
import os
import sys

import numpy as np
import torch

import augmentation
from a_structs import ImageWithBoundingBoxes, bbox


# Bump when a cached index entry's meaning changes, so stale caches are ignored.
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
        would be scored off-distribution; the exact mismatch the evaluation
        convention exists to prevent.

        Default None resolves per sequence from
        preprocess_baseline.DEVICE_ORIENTATION, which is where the verified
        per-device values live. Pass 0 to disable, for an ablation.

        augment: randomise the framing. True for training, and it must be
        False for the validation set. Augmentation is a random draw per sample
        per epoch, so an augmented validation split re-measures a different
        distribution every epoch; the resulting epoch-to-epoch noise is then
        what early stopping reacts to, rather than genuine convergence. This
        mirrors the determinism argument HOT3DKeypointDataset's eval_mode
        implements for KeyNet.

        augment=False does NOT skip augmentation.augment_image. That function
        also performs the resize to the network's input size, so skipping it
        feeds raw full-resolution frames to a fully-connected head and raises a
        shape error. It is called with deterministic=True instead.
        """
        # hot3d's data_loaders uses bare imports, so hot3d/hot3d must be on sys.path first.
        if hot3d_repo_root not in sys.path:
            sys.path.insert(0, hot3d_repo_root)

        # Quest 3 has no TimeCode reference; see hot3d_timecode_compat.py. No-op for Aria.
        # Defensive sys.path insert: callers usually do this already, but don't assume it.
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

        # seq_dir -> providers, lazy and never evicted; cleared per worker by worker_init().
        self._open_sequences = {}

        if self.index_cache_dir:
            os.makedirs(self.index_cache_dir, exist_ok=True)

        # Plain-data index only: it serialises, and forked workers share nothing.
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

    # --- Index -----------------------------------------------------------------

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

        # From the sequence's own metadata.json: one small read instead of opening the .vrs.
        headset = hot3d_split.headset_of(seq_dir) or "Aria"

        # mps_folder_path omitted: nothing here touches MPS data.
        aria_provider = self._AriaDataProvider(paths.vrs_filepath,
                                               mps_folder_path=None)

        entries = []
        for stream_id in aria_provider.get_image_stream_ids():
            # Skip the RGB camera (type 214); keep camera-slam-left/-right (1201), mono only.
            if str(stream_id).startswith("214-"):
                continue

            timestamps = aria_provider.get_sequence_timestamps(
                stream_id, self._TimeDomain.TIME_CODE)
            for ts in timestamps[::self.frame_stride]:
                entries.append((str(stream_id), int(ts), headset))

        return entries

    # --- Providers -------------------------------------------------------------

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
            # Stale cache naming a missing stream; an empty sample beats raising in a worker.
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
                    # Rotate the box with the image, re-derive it axis-aligned, then add the margin.
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
                    # ASSUMED 0=left, 1=right; not confirmed for this CSV, see docstring.
                    slot = 0 if hand_index == 0 else 1
                    bbox_list[slot] = b

        e = ImageWithBoundingBoxes(image=image, bboxes=bbox_list)
        # Always call augment_image: it also warps to input size and carries the boxes.
        e = augmentation.augment_image(e, deterministic=not self.augment)
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
