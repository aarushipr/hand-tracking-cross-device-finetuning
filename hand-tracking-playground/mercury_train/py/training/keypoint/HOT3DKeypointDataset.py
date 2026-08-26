"""
HOT3DKeypointDataset -- live-loading 3D keypoint + depth ground truth for
KeyNet, from the full HOT3D dataset (via Hot3dDataProvider).

One sample = one hand crop (matching RandoDataset's contract in
RandoData.py), not one frame -- a frame with both hands visible
contributes two samples.

Keypoints come from umetrack_hand_data_provider.get_hand_landmarks(),
reordered via hot3d_keypoint_mapping.hot3d_landmarks_to_project_keypoints
(see that module's docstring for the thumb-joint approximation caveat).
2D pixel coordinates come from projecting the mapped 3D world-space
landmarks through the camera's real calibration (verified against real
data in smoke_test_hot3d.py: both hands' wrists landed inside the actual
image bounds).

Depth is NOT raw metric distance in meters. It matches ArtificialData's
native convention exactly, traced from cpp/dataloader/dataloader_pybind.cpp's
add_rel_depth()/hand_length(): each joint's full 3D distance from the
camera, minus the middle-proximal joint's own distance from the camera,
divided by the wrist-to-middle-pxm "hand size" scale. This is why
maker_of_augmentations.py's depth formula expects a small value roughly
in [-1.5, 1.5] rather than a metric depth in meters.

Cropping reuses RandoDataset's own crop()/add_2d_noise_to_keypoints()/
rotate_hand() directly from RandoData.py, rather than reimplementing
that logic, so crops match what the network already expects. The one
deliberate departure: RandoData.py's rotate_hand() only ever operates on
(21,2) arrays, silently discarding the z/depth column -- a known,
already-documented bug in that file. Here we use a small
depth-preserving replacement, _rotate_hand_keep_depth, since depth (in
this project's relative-depth convention) is unaffected by a 2D affine
crop/rotation, so it can simply be carried through untouched instead of
being dropped.

elbow/curls are unconditionally zeroed -- HOT3D has no body-pose data to
derive them from.


SAMPLE INDEX, CACHING, AND FRAME STRIDE
---------------------------------------
The sample index deliberately holds only plain data --
(sequence, image stream, timestamp, hand) -- and never a live provider
object. An earlier version stored the open Hot3dDataProvider in every
sample tuple, which forced every sequence's .vrs recording to be opened
before training could begin (110 simultaneous file handles for the Aria
training split) and made the index impossible to serialise. Over the
cluster's network storage that cost ~44 minutes of wall-clock at only
~73 seconds of CPU -- almost pure I/O latency, and it was paid again in
full on every single run.

Three consequences of the current design:

1. The per-sequence index is cached to disk (see index_cache_dir). The
   cache key includes min_visibility_ratio, frame_stride and
   INDEX_FORMAT_VERSION, so changing any of them -- or changing the
   sampling logic here and bumping the version -- invalidates stale
   caches automatically instead of silently reusing them. Cache writes
   go to a temporary file and are then os.replace()d into position, so a
   job killed mid-write cannot leave a half-written index behind.

2. Providers are opened lazily, on first access to a sample from that
   sequence, and then kept open. They are deliberately NOT evicted:
   DataLoader(shuffle=True) draws consecutive samples from unrelated
   sequences, so any LRU policy would thrash, reopening .vrs files
   constantly. Construction therefore does no I/O at all once the cache
   is warm.

3. frame_stride subsamples timestamps. HOT3D's cameras run at 30 Hz, so
   consecutive frames are near-duplicates; the full Aria training split
   is roughly 2.4M hand crops, far more temporal redundancy than a
   frozen-backbone KeyNet head (~986K trainable parameters) needs. The
   default of 5 keeps every fifth frame. Set frame_stride=1 to reproduce
   the original exhaustive behaviour -- that is also how the
   P0001_10a27bf7 == 11,353 samples regression check is run.

Usage:
    ds = HOT3DKeypointDataset(
        sequence_dirs=["/storage/user/praa/hot3d_full_setup/hot3d/hot3d/dataset/P0003_c701bd11"],
        hot3d_repo_root="/storage/user/praa/hot3d_full_setup/hot3d_repo/hot3d",
        object_library_path="/storage/user/praa/hot3d_full_setup/hot3d/hot3d/dataset/assets",
        min_visibility_ratio=0.2,
        frame_stride=5,
        index_cache_dir="/storage/user/praa/scratch/hot3d_keypoint_index",
    )
"""
import os
import random
import sys

import cv2
import numpy as np
import torch

_mercury_train_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../')
if _mercury_train_root not in sys.path:
    sys.path.insert(0, _mercury_train_root)

import py.training.common.a_geometry as geo
from py.training.common.hot3d_keypoint_mapping import hot3d_landmarks_to_project_keypoints
from RandoData import crop, rotate_hand, add_2d_noise_to_keypoints
from maker_of_augmentations import AugmentationMaker
from a_aug_config import aug_config_validatoor


# Bump this whenever the sampling logic below changes in a way that would
# produce a different index for the same inputs. Cached indices built by
# an older version are then ignored rather than silently reused.
INDEX_FORMAT_VERSION = 1


def world_point_to_camera_frame(world_point, T_world_device, T_device_camera):
    """
    Transforms a single 3D point from the shared world/scene frame (same
    frame as HOT3D's hand landmarks and device trajectory) into this
    camera's own local coordinate frame.
    """
    T_world_camera = T_world_device.to_matrix() @ T_device_camera.to_matrix()
    T_camera_world = np.linalg.inv(T_world_camera)
    p_world_h = np.array([world_point[0], world_point[1], world_point[2], 1.0])
    return (T_camera_world @ p_world_h)[:3]


def _rotate_hand_keep_depth(kps_with_depth, mat):
    """
    Same 2D affine transform as RandoData.py's own rotate_hand(), but
    carries the 3rd (depth) column through untouched instead of
    discarding it.
    """
    out = np.zeros((21, 3), dtype=np.float32)
    for i in range(21):
        out[i, :2] = geo.transformVecBy2x3(kps_with_depth[i, :2], mat)
    out[:, 2] = kps_with_depth[:, 2]
    return out


class _SequenceProviders:
    """
    Everything that has to be held open for one HOT3D sequence, plus the
    per-stream values that are fixed for the whole recording and would
    otherwise be recomputed on every single sample.
    """

    def __init__(self, hot3d_data_provider, mono_stream_ids, calibrations):
        self.hot3d_data_provider = hot3d_data_provider
        self.device_data_provider = hot3d_data_provider.device_data_provider
        self.umetrack_provider = hot3d_data_provider.umetrack_hand_data_provider
        self.hand_box2d_provider = hot3d_data_provider.hand_box2d_data_provider
        self.device_pose_provider = hot3d_data_provider.device_pose_data_provider

        self.mono_stream_ids = mono_stream_ids
        self.stream_id_by_str = {str(s): s for s in mono_stream_ids}
        # str(stream_id) -> (T_device_camera, camera_calibration)
        self.calibrations = calibrations


class HOT3DKeypointDataset(torch.utils.data.Dataset):
    def __init__(self, sequence_dirs: list, hot3d_repo_root: str,
                 object_library_path: str, min_visibility_ratio: float = 0.2,
                 frame_stride: int = 5, index_cache_dir: str = None):
        if hot3d_repo_root not in sys.path:
            sys.path.insert(0, hot3d_repo_root)

        from dataset_api import Hot3dDataProvider
        from data_loaders.loader_object_library import load_object_library
        from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions

        self._Hot3dDataProvider = Hot3dDataProvider
        self._TimeDomain = TimeDomain
        self._TimeQueryOptions = TimeQueryOptions

        if frame_stride < 1:
            raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")

        self.sequence_dirs = list(sequence_dirs)
        self.hot3d_repo_root = hot3d_repo_root
        self.min_visibility_ratio = min_visibility_ratio
        self.frame_stride = frame_stride
        self.index_cache_dir = index_cache_dir
        self.augmaker = AugmentationMaker(aug_config_validatoor)

        self._object_library = load_object_library(
            object_library_folderpath=object_library_path)

        # seq_dir -> _SequenceProviders, populated lazily, never evicted.
        self._open_sequences = {}

        if self.index_cache_dir:
            os.makedirs(self.index_cache_dir, exist_ok=True)

        # Parallel arrays, one entry per sample. Kept as numpy rather than a
        # list of tuples purely for size: the full-rate Aria split is a few
        # million samples, where Python tuple overhead alone runs to
        # hundreds of megabytes.
        seq_idx, stream_str, ts, hand_key, hand_index = [], [], [], [], []

        for i, seq_dir in enumerate(self.sequence_dirs):
            entries = self._index_for_sequence(seq_dir)
            for e_stream_str, e_ts, e_hand_key, e_hand_index in entries:
                seq_idx.append(i)
                stream_str.append(e_stream_str)
                ts.append(e_ts)
                hand_key.append(e_hand_key)
                hand_index.append(e_hand_index)

        self._seq_idx = np.asarray(seq_idx, dtype=np.int32)
        self._stream_str = np.asarray(stream_str, dtype=np.str_)
        self._ts = np.asarray(ts, dtype=np.int64)
        self._hand_key = np.asarray(hand_key, dtype=np.str_)
        self._hand_index = np.asarray(hand_index, dtype=np.int8)

        self.actual_size = len(self._ts)
        self.num_times_to_repeat = 1

    # ------------------------------------------------------------------
    # Sample index: build, cache, load
    # ------------------------------------------------------------------

    def _cache_path(self, seq_dir):
        if not self.index_cache_dir:
            return None
        name = os.path.basename(os.path.normpath(seq_dir))
        key = (f"{name}__vis{self.min_visibility_ratio}"
               f"__stride{self.frame_stride}__v{INDEX_FORMAT_VERSION}")
        return os.path.join(self.index_cache_dir, key + ".npz")

    def _index_for_sequence(self, seq_dir):
        """
        Return the sample index for one sequence as a list of
        (stream_id_str, timestamp_ns, hand_key_str, hand_index) tuples,
        loading it from the on-disk cache when possible.
        """
        cache_path = self._cache_path(seq_dir)

        if cache_path and os.path.exists(cache_path):
            try:
                with np.load(cache_path, allow_pickle=False) as z:
                    return list(zip(
                        z["stream_str"].tolist(),
                        z["ts"].tolist(),
                        z["hand_key"].tolist(),
                        z["hand_index"].tolist(),
                    ))
            except (OSError, ValueError, KeyError) as e:
                # A truncated or otherwise unreadable cache file should cost
                # a rebuild, not the whole run.
                print(f"[HOT3DKeypointDataset] ignoring unreadable index cache "
                      f"{cache_path}: {e}", file=sys.stderr)

        entries = self._build_index_for_sequence(seq_dir)

        if cache_path:
            self._save_index(cache_path, entries)

        return entries

    def _save_index(self, cache_path, entries):
        """
        Write atomically: a job killed mid-write must not leave a
        half-written index that a later run would happily load.
        """
        tmp_path = f"{cache_path}.tmp{os.getpid()}"
        try:
            np.savez(
                tmp_path,
                stream_str=np.asarray([e[0] for e in entries], dtype=np.str_),
                ts=np.asarray([e[1] for e in entries], dtype=np.int64),
                hand_key=np.asarray([e[2] for e in entries], dtype=np.str_),
                hand_index=np.asarray([e[3] for e in entries], dtype=np.int8),
            )
            os.replace(f"{tmp_path}.npz", cache_path)
        except OSError as e:
            print(f"[HOT3DKeypointDataset] could not write index cache "
                  f"{cache_path}: {e}", file=sys.stderr)
            for leftover in (tmp_path, f"{tmp_path}.npz"):
                if os.path.exists(leftover):
                    try:
                        os.remove(leftover)
                    except OSError:
                        pass

    def _build_index_for_sequence(self, seq_dir):
        TimeDomain = self._TimeDomain
        TimeQueryOptions = self._TimeQueryOptions

        bundle = self._providers_for(seq_dir)
        entries = []

        for stream_id in bundle.mono_stream_ids:
            timestamps = self._sequence_timestamps(bundle, stream_id)

            for ts in timestamps[::self.frame_stride]:
                device_pose = bundle.device_pose_provider.get_pose_at_timestamp(
                    timestamp_ns=ts,
                    time_query_options=TimeQueryOptions.CLOSEST,
                    time_domain=TimeDomain.TIME_CODE,
                )
                if device_pose is None:
                    continue

                hand_poses_with_dt = bundle.umetrack_provider.get_pose_at_timestamp(
                    timestamp_ns=ts,
                    time_query_options=TimeQueryOptions.CLOSEST,
                    time_domain=TimeDomain.TIME_CODE,
                )
                if hand_poses_with_dt is None:
                    continue

                # Fetched once per (stream, timestamp): it carries both
                # hands, so re-fetching it inside the per-hand loop --
                # as an earlier version did -- was pure waste.
                box_result = bundle.hand_box2d_provider.get_bbox_at_timestamp(
                    stream_id=stream_id,
                    timestamp_ns=ts,
                    time_query_options=TimeQueryOptions.CLOSEST,
                    time_domain=TimeDomain.TIME_CODE,
                )
                if box_result is None:
                    continue

                for key, hand_pose_data in hand_poses_with_dt.pose3d_collection.poses.items():
                    # Unlike HOT3DVRSDetectionDataset (which keeps every
                    # frame, including ones with no visible hand, as
                    # negative "exists=0" examples for DetNet's detection
                    # task), KeyNet has no "hand exists" signal to train --
                    # it only ever consumes an already-cropped hand image.
                    # A hand that isn't visible enough simply has no valid
                    # crop, so we skip it entirely.
                    #
                    # ASSUMED 0=left, 1=right, matching
                    # HOT3DVRSDetectionDataset's own (also unverified)
                    # assumption for this same box2d data.
                    hand_index = 0 if hand_pose_data.is_left_hand else 1
                    hand_box = box_result.box2d_collection.box2ds.get(hand_index)
                    if hand_box is None or hand_box.box2d is None:
                        continue
                    if hand_box.visibility_ratio is None or \
                            hand_box.visibility_ratio < self.min_visibility_ratio:
                        continue

                    # The pose collection's own key identifies the hand
                    # unambiguously. Storing it (rather than a derived
                    # left/right flag) means __getitem__ re-selects exactly
                    # the same pose object this index entry was built from.
                    entries.append((str(stream_id), int(ts), str(key), hand_index))

        return entries

    # ------------------------------------------------------------------
    # Providers
    # ------------------------------------------------------------------

    def _providers_for(self, seq_dir):
        bundle = self._open_sequences.get(seq_dir)
        if bundle is not None:
            return bundle

        hot3d_data_provider = self._Hot3dDataProvider(
            sequence_folder=seq_dir,
            object_library=self._object_library,
            mano_hand_model=None,
        )
        device_data_provider = hot3d_data_provider.device_data_provider

        # Skip the RGB camera -- same reasoning as
        # HOT3DVRSDetectionDataset: this project's camera model is 2
        # monochrome cameras, matching the target headset hardware. (Quest
        # recordings have no RGB stream at all, so this is a no-op there.)
        mono_stream_ids = [s for s in device_data_provider.get_image_stream_ids()
                           if not str(s).startswith("214-")]

        calibrations = {str(s): device_data_provider.get_camera_calibration(s)
                        for s in mono_stream_ids}

        bundle = _SequenceProviders(hot3d_data_provider, mono_stream_ids, calibrations)
        self._open_sequences[seq_dir] = bundle
        return bundle

    def _sequence_timestamps(self, bundle, stream_id):
        """
        Per-stream capture timestamps for one image stream.

        Aria and Quest expose genuinely different APIs here, not just
        different argument counts:

            AriaDataProvider.get_sequence_timestamps(stream_id, time_domain)
                -> that one stream's timestamps
            QuestDataProvider.get_sequence_timestamps()
                -> the merged, de-duplicated set of every image timestamp
                   in the recording, in the recording's own DEVICE_TIME
                   domain, because Quest 3 HOT3D recordings carry no
                   TimeCode track at all (see
                   py/training/common/hot3d_timecode_compat.py)

        Feeding Quest's merged list straight into a per-stream lookup
        could silently return a frame from the wrong capture instant, so
        this raises instead. Quest support belongs in the evaluation
        path, where get_frameset_from_timestamp() can map a reference
        timestamp onto each stream's own nearest capture time under an
        explicit tolerance. Training is Aria-only by design (see
        py/training/common/hot3d_split.py), so nothing in the training
        path reaches this.
        """
        headset = bundle.hot3d_data_provider.get_device_type()
        if getattr(headset, "name", str(headset)) != "Aria":
            raise NotImplementedError(
                f"HOT3DKeypointDataset does not yet support {headset} recordings. "
                f"Training is Aria-only by design; Quest support is needed only "
                f"for the cross-device test split and is not wired up yet.")

        return bundle.device_data_provider.get_sequence_timestamps(
            stream_id, self._TimeDomain.TIME_CODE)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self):
        return self.actual_size * self.num_times_to_repeat

    def __getitem__(self, idx):
        idx = idx % self.actual_size
        TimeDomain = self._TimeDomain
        TimeQueryOptions = self._TimeQueryOptions

        seq_dir = self.sequence_dirs[int(self._seq_idx[idx])]
        stream_str = str(self._stream_str[idx])
        ts = int(self._ts[idx])
        hand_key = str(self._hand_key[idx])

        bundle = self._providers_for(seq_dir)
        stream_id = bundle.stream_id_by_str[stream_str]
        T_device_camera, camera_calibration = bundle.calibrations[stream_str]

        device_pose = bundle.device_pose_provider.get_pose_at_timestamp(
            timestamp_ns=ts,
            time_query_options=TimeQueryOptions.CLOSEST,
            time_domain=TimeDomain.TIME_CODE,
        )
        if device_pose is None:
            return self._empty_sample()
        T_world_device = device_pose.pose3d.T_world_device

        hand_poses_with_dt = bundle.umetrack_provider.get_pose_at_timestamp(
            timestamp_ns=ts,
            time_query_options=TimeQueryOptions.CLOSEST,
            time_domain=TimeDomain.TIME_CODE,
        )
        if hand_poses_with_dt is None:
            return self._empty_sample()

        hand_pose_data = None
        for key, pose in hand_poses_with_dt.pose3d_collection.poses.items():
            if str(key) == hand_key:
                hand_pose_data = pose
                break
        if hand_pose_data is None:
            return self._empty_sample()

        image = bundle.device_data_provider.get_image(ts, stream_id)
        if image is None:
            return self._empty_sample()

        hot3d_landmarks = bundle.umetrack_provider.get_hand_landmarks(hand_pose_data)
        hot3d_landmarks = hot3d_landmarks.detach().cpu().numpy()
        world_keypoints = hot3d_landmarks_to_project_keypoints(hot3d_landmarks)  # (21, 3), world space

        camera_points = np.zeros((21, 3), dtype=np.float32)
        for i in range(21):
            camera_points[i] = world_point_to_camera_frame(
                world_keypoints[i], T_world_device, T_device_camera)

        if np.any(camera_points[:, 2] <= 0):
            return self._empty_sample()  # a keypoint landed behind the camera

        keypoints_px_and_depth = np.zeros((21, 3), dtype=np.float32)
        for i in range(21):
            pixel = camera_calibration.project(camera_points[i])
            if pixel is None:
                return self._empty_sample()
            keypoints_px_and_depth[i, :2] = pixel

        # Relative depth, matching ArtificialData's native convention
        # exactly (see module docstring): each joint's full 3D distance
        # from the camera, minus the middle-proximal joint's own distance
        # from the camera, divided by the wrist-to-middle-pxm "hand size".
        hand_size = np.linalg.norm(world_keypoints[0] - world_keypoints[9])  # wrist -> middle_pxm
        midpxm_depth = np.linalg.norm(camera_points[9])
        joint_distances = np.linalg.norm(camera_points, axis=1)
        keypoints_px_and_depth[:, 2] = (joint_distances - midpxm_depth) / hand_size

        is_right = hand_pose_data.is_right_hand

        # Same cropping approach as RandoDataset: derive the crop from a
        # NOISED version of the keypoints (simulating a previous frame's
        # imperfect prediction, not oracle-perfect current-frame GT).
        noisy_keypoints = add_2d_noise_to_keypoints(keypoints_px_and_depth[:, :2])
        trans = crop(image, noisy_keypoints, is_right)

        img_cropped = cv2.warpAffine(image, trans, (128, 128))
        keypoints_cropped = _rotate_hand_keep_depth(keypoints_px_and_depth, trans)

        # Matches RandoDataset's own "30% chance of no predicted input"
        # convention, so this data source behaves consistently with the
        # rest of KeyNet's training data.
        predicted_px = None
        if random.uniform(0, 1) >= 0.3:
            predicted_px = rotate_hand(noisy_keypoints, trans)

        ret = self.augmaker.do_one_augmentation(
            img_cropped,
            keypoints_cropped,
            predicted_px=predicted_px,
            mask=None,
            img_alpha_premultiplied=False,
            is_right=is_right)
        ret["elbow"] = torch.zeros(3).float()
        ret["curls"] = torch.zeros(5).float()
        return ret

    def _empty_sample(self):
        blank = np.zeros((128, 128), dtype=np.uint8)
        zeros_kps = np.zeros((21, 3), dtype=np.float32)
        ret = self.augmaker.do_one_augmentation(
            blank, zeros_kps, mask=None, img_alpha_premultiplied=False, is_right=False)
        ret["elbow"] = torch.zeros(3).float()
        ret["curls"] = torch.zeros(5).float()
        return ret


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-dirs", required=True, nargs="+")
    parser.add_argument("--hot3d-repo-root", required=True)
    parser.add_argument("--object-library-path", required=True)
    parser.add_argument("--min-visibility-ratio", type=float, default=0.2)
    parser.add_argument("--frame-stride", type=int, default=5,
                        help="keep every Nth frame; 1 reproduces the original "
                             "exhaustive sampling (used for the 11,353-sample "
                             "regression check on P0001_10a27bf7)")
    parser.add_argument("--index-cache-dir", default=None,
                        help="directory for the on-disk sample index cache, "
                             "e.g. /storage/user/praa/scratch/hot3d_keypoint_index")
    args = parser.parse_args()

    ds = HOT3DKeypointDataset(
        sequence_dirs=args.sequence_dirs,
        hot3d_repo_root=args.hot3d_repo_root,
        object_library_path=args.object_library_path,
        min_visibility_ratio=args.min_visibility_ratio,
        frame_stride=args.frame_stride,
        index_cache_dir=args.index_cache_dir,
    )
    print(f"{len(ds)} samples")
    samp = ds[0]
    print({k: (v.shape if hasattr(v, "shape") else v) for k, v in samp.items()})
