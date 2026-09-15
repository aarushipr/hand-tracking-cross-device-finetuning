"""
Live-loading keypoint and depth ground truth for KeyNet from the full HOT3D
dataset. One sample is one hand crop. Depth is not metric: it is each joint's
camera distance minus the middle-proximal joint's, over the wrist-to-middle-pxm
hand size, so it lands in about [-1.5, 1.5]. Validity is per joint, not per hand,
and a hand is dropped only when no joint is usable. The index holds plain data and
is cached per sequence; with num_workers>0 see worker_init().
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


# Bump when the sampling logic changes, so stale caches are ignored not reused.
# 1 box-visibility only; 2 projection gate at index time; 3 per-joint validity.
INDEX_FORMAT_VERSION = 3


# Guards against the spurious 0ns sentinel in QuestDataProvider's merged timestamps.
MAX_POSE_TIME_DELTA_NS = 50_000_000  # 50ms; inter-frame gap is ~33ms


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


def worker_init(worker_id):
    """
    DataLoader worker_init_fn. Pass this as worker_init_fn whenever
    num_workers > 0.

    Workers are forked processes, so they inherit whatever providers the
    parent already had open, and two processes reading the same C++ VRS
    file handle is exactly what produced garbled timestamps and JPEG decode
    failures previously. Clearing the cache here forces each worker to open
    its own providers on first access, so no handle is ever shared.
    """
    info = torch.utils.data.get_worker_info()
    if info is not None:
        info.dataset._open_sequences = {}


class _SequenceProviders:
    """
    Everything held open for one HOT3D sequence, plus the per-stream values
    that are fixed for the whole recording.
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
                 frame_stride: int = 10, index_cache_dir: str = None,
                 eval_mode: bool = False):
        """
        eval_mode: when True, every per-sample random draw is removed, so a
        sample is a deterministic function of its index alone.

        Three things are randomised during training and must not be during
        evaluation, because they are properties of the *measurement* rather
        than of the model, and two models compared under different draws are
        not being compared:
          - the crop rotation, drawn uniformly over the full circle
          - the crop radius multiplier, drawn over roughly [0.81, 1.18]
          - the keypoint noise that decides where the crop is centred
        In eval_mode the crop matrix comes from
        py.evaluation.preprocess_baseline.keynet_crop_matrix at the centre of
        those distributions, and it is centred on the ground truth itself.
        The predicted-keypoint side input is also withheld, matching
        validatoor.validation_loop's use_prediction=False.

        Defaults to False, so the training path is bit-for-bit unchanged.
        """
        if hot3d_repo_root not in sys.path:
            sys.path.insert(0, hot3d_repo_root)

        # Quest 3 has no TimeCode track, so anything reaching AriaDataProvider raises without this.
        # Verified for detection GT only; run check_quest_keypoints.py before trusting Quest labels.
        # Defensive sys.path insert: callers usually do this already, but don't assume it.
        _mercury_train_root = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
        if _mercury_train_root not in sys.path:
            sys.path.insert(0, _mercury_train_root)
        from py.training.common.hot3d_timecode_compat import patch as _patch_quest_timecode
        _patch_quest_timecode()

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
        self.eval_mode = eval_mode
        self.augmaker = AugmentationMaker(aug_config_validatoor)

        self._object_library = load_object_library(
            object_library_folderpath=object_library_path)

        # seq_dir -> providers, lazy and never evicted; cleared per worker by worker_init().
        self._open_sequences = {}

        if self.index_cache_dir:
            os.makedirs(self.index_cache_dir, exist_ok=True)

        seq_idx, stream_str, ts, is_right, kps, valid = [], [], [], [], [], []
        n_candidates = 0

        for i, seq_dir in enumerate(self.sequence_dirs):
            entries, candidates = self._index_for_sequence(seq_dir)
            n_candidates += candidates
            for e in entries:
                seq_idx.append(i)
                stream_str.append(e[0])
                ts.append(e[1])
                is_right.append(e[2])
                kps.append(e[3])
                valid.append(e[4])

        self._seq_idx = np.asarray(seq_idx, dtype=np.int32)
        self._stream_str = np.asarray(stream_str, dtype=np.str_)
        self._ts = np.asarray(ts, dtype=np.int64)
        self._is_right = np.asarray(is_right, dtype=bool)
        self._kps = (np.asarray(kps, dtype=np.float32).reshape(-1, 21, 3)
                     if kps else np.zeros((0, 21, 3), dtype=np.float32))
        self._valid = (np.asarray(valid, dtype=bool).reshape(-1, 21, 2)
                       if valid else np.zeros((0, 21, 2), dtype=bool))

        self.actual_size = len(self._ts)
        self.num_times_to_repeat = 1
        self.n_rejected_by_projection = n_candidates - self.actual_size

        if n_candidates:
            pct = 100.0 * self.n_rejected_by_projection / n_candidates
            print(f"[HOT3DKeypointDataset] {self.actual_size} samples from "
                  f"{len(self.sequence_dirs)} sequences "
                  f"({self.n_rejected_by_projection} of {n_candidates} candidates "
                  f"rejected, {pct:.1f}%, no joint at all usable; remaining "
                  f"samples may still have individual joints masked out)")

    # --- Sample index: build, cache, load --------------------------------------

    def _cache_path(self, seq_dir):
        if not self.index_cache_dir:
            return None
        name = os.path.basename(os.path.normpath(seq_dir))
        key = (f"{name}__vis{self.min_visibility_ratio}"
               f"__stride{self.frame_stride}__v{INDEX_FORMAT_VERSION}")
        return os.path.join(self.index_cache_dir, key + ".npz")

    def _index_for_sequence(self, seq_dir):
        """
        Returns (entries, n_candidates), where each entry is
        (stream_id_str, timestamp_ns, is_right, keypoints_px_and_depth,
        per_joint_valid), and per_joint_valid is a (21,2) bool array of
        (xy_valid, depth_valid) for that hand's 21 joints.
        n_candidates counts hands that passed the box-visibility gate,
        including those later rejected for having no usable joint at all.
        """
        cache_path = self._cache_path(seq_dir)

        if cache_path and os.path.exists(cache_path):
            try:
                with np.load(cache_path, allow_pickle=False) as z:
                    entries = list(zip(
                        z["stream_str"].tolist(),
                        z["ts"].tolist(),
                        z["is_right"].tolist(),
                        list(z["kps"]),
                        list(z["valid"]),
                    ))
                    return entries, int(z["n_candidates"])
            except (OSError, ValueError, KeyError) as e:
                print(f"[HOT3DKeypointDataset] ignoring unreadable index cache "
                      f"{cache_path}: {e}", file=sys.stderr)

        entries, n_candidates = self._build_index_for_sequence(seq_dir)

        if cache_path:
            self._save_index(cache_path, entries, n_candidates)

        return entries, n_candidates

    def _save_index(self, cache_path, entries, n_candidates):
        """
        Write atomically: a job killed mid-write must not leave a
        half-written index that a later run would happily load.
        """
        tmp_path = f"{cache_path}.tmp{os.getpid()}"
        try:
            kps = (np.asarray([e[3] for e in entries], dtype=np.float32)
                   if entries else np.zeros((0, 21, 3), dtype=np.float32))
            valid = (np.asarray([e[4] for e in entries], dtype=bool)
                     if entries else np.zeros((0, 21, 2), dtype=bool))
            np.savez(
                tmp_path,
                stream_str=np.asarray([e[0] for e in entries], dtype=np.str_),
                ts=np.asarray([e[1] for e in entries], dtype=np.int64),
                is_right=np.asarray([e[2] for e in entries], dtype=bool),
                kps=kps.reshape(-1, 21, 3),
                valid=valid.reshape(-1, 21, 2),
                n_candidates=np.asarray(n_candidates, dtype=np.int64),
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

    def _project_hand(self, world_keypoints, T_world_device,
                      T_device_camera, camera_calibration):
        """
        World-space (21,3) landmarks -> (21,3) of [x_px, y_px, relative_depth],
        plus two (21,) validity masks (xy_valid, depth_valid), or None if not
        even one joint is usable.

        A joint is xy-invalid if it falls behind the camera or outside the
        camera model's valid region; its x_px/y_px are left at 0 here and
        must not be trained on directly; see __getitem__, which substitutes
        a geometry-safe placeholder before cropping, and xy_valid_per_joint,
        which keeps it out of the loss regardless of that placeholder value.

        Depth is relative to the middle-proximal joint (index 9) and scaled
        by the wrist-to-middle-pxm "hand size" (see module docstring), so
        every joint's depth additionally requires BOTH the wrist (index 0)
        and the middle-proximal joint (index 9) to themselves be xy-valid.
        If either of those two fails, no joint's depth is computable that
        frame, not only the joints that individually failed.
        """
        camera_points = np.zeros((21, 3), dtype=np.float32)
        for i in range(21):
            camera_points[i] = world_point_to_camera_frame(
                world_keypoints[i], T_world_device, T_device_camera)

        xy_valid = camera_points[:, 2] > 0
        out = np.zeros((21, 3), dtype=np.float32)
        for i in range(21):
            if not xy_valid[i]:
                continue
            pixel = camera_calibration.project(camera_points[i])
            if pixel is None:
                xy_valid[i] = False
                continue
            out[i, :2] = pixel

        if not xy_valid.any():
            return None  # nothing usable in this hand at all

        # Relative depth (see module docstring); wrist and middle-pxm must both be xy-valid.
        depth_valid = np.zeros(21, dtype=bool)
        if xy_valid[0] and xy_valid[9]:
            hand_size = np.linalg.norm(world_keypoints[0] - world_keypoints[9])
            if hand_size > 0:
                midpxm_depth = np.linalg.norm(camera_points[9])
                joint_distances = np.linalg.norm(camera_points, axis=1)
                out[:, 2] = (joint_distances - midpxm_depth) / hand_size
                depth_valid = xy_valid.copy()

        return out, xy_valid, depth_valid

    def _build_index_for_sequence(self, seq_dir):
        TimeDomain = self._TimeDomain
        TimeQueryOptions = self._TimeQueryOptions

        bundle = self._providers_for(seq_dir)
        entries = []
        n_candidates = 0

        for stream_id in bundle.mono_stream_ids:
            T_device_camera, camera_calibration = bundle.calibrations[str(stream_id)]
            timestamps = self._sequence_timestamps(bundle, stream_id)

            for ts in timestamps[::self.frame_stride]:
                device_pose = bundle.device_pose_provider.get_pose_at_timestamp(
                    timestamp_ns=ts,
                    time_query_options=TimeQueryOptions.CLOSEST,
                    time_domain=TimeDomain.TIME_CODE,
                    acceptable_time_delta=MAX_POSE_TIME_DELTA_NS,
                )
                if device_pose is None:
                    continue
                T_world_device = device_pose.pose3d.T_world_device

                hand_poses_with_dt = bundle.umetrack_provider.get_pose_at_timestamp(
                    timestamp_ns=ts,
                    time_query_options=TimeQueryOptions.CLOSEST,
                    time_domain=TimeDomain.TIME_CODE,
                    acceptable_time_delta=MAX_POSE_TIME_DELTA_NS,
                )
                if hand_poses_with_dt is None:
                    continue

                # Fetched once per (stream, timestamp): it carries both hands.
                box_result = bundle.hand_box2d_provider.get_bbox_at_timestamp(
                    stream_id=stream_id,
                    timestamp_ns=ts,
                    time_query_options=TimeQueryOptions.CLOSEST,
                    time_domain=TimeDomain.TIME_CODE,
                )
                if box_result is None:
                    continue

                for hand_pose_data in hand_poses_with_dt.pose3d_collection.poses.values():
                    # KeyNet has no hand-presence signal to train; an unusable hand is just skipped.
                    # ASSUMED 0=left, 1=right, as in HOT3DVRSDetectionDataset.
                    hand_index = 0 if hand_pose_data.is_left_hand else 1
                    hand_box = box_result.box2d_collection.box2ds.get(hand_index)
                    if hand_box is None or hand_box.box2d is None:
                        continue
                    if hand_box.visibility_ratio is None or \
                            hand_box.visibility_ratio < self.min_visibility_ratio:
                        continue

                    n_candidates += 1

                    landmarks = bundle.umetrack_provider.get_hand_landmarks(hand_pose_data)
                    if landmarks is None:
                        continue
                    world_keypoints = hot3d_landmarks_to_project_keypoints(
                        landmarks.detach().cpu().numpy())

                    projection = self._project_hand(
                        world_keypoints, T_world_device,
                        T_device_camera, camera_calibration)
                    if projection is None:
                        continue
                    proj_kps, xy_valid, depth_valid = projection
                    per_joint_valid = np.stack([xy_valid, depth_valid], axis=1)

                    entries.append((str(stream_id), int(ts),
                                    bool(hand_pose_data.is_right_hand),
                                    proj_kps, per_joint_valid))

        return entries, n_candidates

    # --- Providers -------------------------------------------------------------

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

        # Skip the RGB camera, same reasoning as HOT3DVRSDetectionDataset:
        # this project's camera model is 2 monochrome cameras, matching the
        # target headset hardware. (Quest recordings have no RGB stream at
        # all, so this is a no-op there.)
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

        Aria and Quest expose genuinely different APIs here, not merely
        different argument counts:

            AriaDataProvider.get_sequence_timestamps(stream_id, time_domain)
                -> that one stream's timestamps
            QuestDataProvider.get_sequence_timestamps()
                -> the merged, de-duplicated set of every image timestamp in
                   the recording, in the recording's own DEVICE_TIME domain,
                   because Quest 3 HOT3D recordings carry no TimeCode track
                   at all (see py/training/common/hot3d_timecode_compat.py)

        QuestDataProvider.__init__ assigns this exact merged list to every
        stream_id anyway (see its _stream_timestamps_sorted), so using it
        directly per-stream is equivalent to what get_frameset_from_timestamp
        would resolve to here; no separate frameset step needed.

        The Quest branch below is on the live training path: the mixed split
        (see py/training/common/hot3d_split.py) puts Quest recordings into
        both training and evaluation.
        """
        headset = bundle.hot3d_data_provider.get_device_type()
        if getattr(headset, "name", str(headset)) == "Aria":
            return bundle.device_data_provider.get_sequence_timestamps(
                stream_id, self._TimeDomain.TIME_CODE)

        # time_domain is a guard clause only, no conversion; verified against real Quest data.
        return bundle.device_data_provider.get_sequence_timestamps()

    # --- Dataset protocol ------------------------------------------------------

    def __len__(self):
        return self.actual_size * self.num_times_to_repeat

    def __getitem__(self, idx):
        idx = idx % self.actual_size

        seq_dir = self.sequence_dirs[int(self._seq_idx[idx])]
        stream_str = str(self._stream_str[idx])
        ts = int(self._ts[idx])
        is_right = bool(self._is_right[idx])
        keypoints_px_and_depth = self._kps[idx].copy()
        xy_valid_per_joint = self._valid[idx, :, 0].copy()
        depth_valid_per_joint = self._valid[idx, :, 1].copy()

        # Failed joints take this hand's valid-joint mean so crop geometry isn't skewed.
        # Never a training target: the per-joint masks drop it from the loss.
        if not xy_valid_per_joint.all():
            valid_mean_xy = keypoints_px_and_depth[xy_valid_per_joint, :2].mean(axis=0)
            keypoints_px_and_depth[~xy_valid_per_joint, :2] = valid_mean_xy

        bundle = self._providers_for(seq_dir)
        stream_id = bundle.stream_id_by_str[stream_str]

        image = bundle.device_data_provider.get_image(ts, stream_id)
        if image is None:
            # Rare; every other failure mode is now caught at index time.
            return self._empty_sample()

        if self.eval_mode:
            # Deterministic: crop centred on GT, at the centre of both distributions.
            from py.evaluation import preprocess_baseline as _pp
            trans = _pp.keynet_crop_matrix(keypoints_px_and_depth[:, :2], is_right)
            predicted_px = None
        else:
            # Crop from a NOISED copy of the keypoints, as RandoDataset does.
            # All terms are hand-relative, so Aria and Quest get the same augmentation.
            noisy_keypoints = add_2d_noise_to_keypoints(keypoints_px_and_depth[:, :2])
            trans = crop(image, noisy_keypoints, is_right)

        img_cropped = cv2.warpAffine(image, trans, (128, 128))
        keypoints_cropped = _rotate_hand_keep_depth(keypoints_px_and_depth, trans)

        if not self.eval_mode:
            # Matches RandoDataset's 30%-chance-of-no-predicted-input convention.
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
        ret["xy_valid_per_joint"] = xy_valid_per_joint.astype(np.float32)
        ret["depth_valid_per_joint"] = depth_valid_per_joint.astype(np.float32)
        return ret

    def _empty_sample(self):
        """
        Placeholder for an unreadable image.

        Flagged is_hand=0 / has_xy=0 / has_depth=0, with every per-joint
        validity flag also 0, so every loss term masks it out and it
        contributes no gradient. Version 1 of this class left is_hand=1,
        which made each placeholder a mislabelled POSITIVE; a black image
        asserting a hand was present at 21 coincident points. That is not a
        negative training example, it is a wrong one.

        Note this is still not a useful negative even so: a genuine negative
        would be a crop of a real frame containing no hand. Hand presence is
        DetNet's task (HOT3DVRSDetectionDataset does keep hand-free frames as
        exists=0 examples), and settings.existence_loss_mul is 0 for this
        fine-tuning, so KeyNet's existence head is deliberately left at its
        Monado initialisation.
        """
        blank = np.zeros((128, 128), dtype=np.uint8)
        zeros_kps = np.zeros((21, 3), dtype=np.float32)
        ret = self.augmaker.do_one_augmentation(
            blank, zeros_kps, mask=None, img_alpha_premultiplied=False, is_right=False)
        ret["is_hand"] = np.float32(0)
        ret["has_xy"] = np.float32(0)
        ret["has_depth"] = np.float32(0)
        ret["elbow"] = torch.zeros(3).float()
        ret["curls"] = torch.zeros(5).float()
        ret["xy_valid_per_joint"] = np.zeros(21, dtype=np.float32)
        ret["depth_valid_per_joint"] = np.zeros(21, dtype=np.float32)
        return ret


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-dirs", required=True, nargs="+")
    parser.add_argument("--hot3d-repo-root", required=True)
    parser.add_argument("--object-library-path", required=True)
    parser.add_argument("--min-visibility-ratio", type=float, default=0.2)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--index-cache-dir", default=None)
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
