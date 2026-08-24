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

Usage:
    ds = HOT3DKeypointDataset(
        sequence_dirs=["/storage/user/praa/hot3d_full_setup/hot3d/hot3d/dataset/P0003_c701bd11"],
        hot3d_repo_root="/storage/user/praa/hot3d_full_setup/hot3d_repo/hot3d",
        object_library_path="/storage/user/praa/hot3d_full_setup/hot3d/hot3d/dataset/assets",
        min_visibility_ratio=0.2,
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


class HOT3DKeypointDataset(torch.utils.data.Dataset):
    def __init__(self, sequence_dirs: list, hot3d_repo_root: str,
                 object_library_path: str, min_visibility_ratio: float = 0.2):
        if hot3d_repo_root not in sys.path:
            sys.path.insert(0, hot3d_repo_root)

        from dataset_api import Hot3dDataProvider
        from data_loaders.loader_object_library import load_object_library
        from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions

        self._TimeDomain = TimeDomain
        self._TimeQueryOptions = TimeQueryOptions
        self.min_visibility_ratio = min_visibility_ratio
        self.augmaker = AugmentationMaker(aug_config_validatoor)

        object_library = load_object_library(object_library_folderpath=object_library_path)

        self.samples = []
        for seq_dir in sequence_dirs:
            hot3d_data_provider = Hot3dDataProvider(
                sequence_folder=seq_dir,
                object_library=object_library,
                mano_hand_model=None,
            )
            device_data_provider = hot3d_data_provider.device_data_provider
            umetrack_provider = hot3d_data_provider.umetrack_hand_data_provider
            hand_box2d_provider = hot3d_data_provider.hand_box2d_data_provider
            device_pose_provider = hot3d_data_provider.device_pose_data_provider

            for stream_id in device_data_provider.get_image_stream_ids():
                # Skip the RGB camera -- same reasoning as
                # HOT3DVRSDetectionDataset: this project's camera model is
                # 2 monochrome cameras, matching the target headset hardware.
                if str(stream_id).startswith("214-"):
                    continue

                # Fixed per stream -- computed once, reused for every
                # timestamp/hand sampled from this stream.
                T_device_camera, camera_calibration = device_data_provider.get_camera_calibration(stream_id)

                timestamps = device_data_provider.get_sequence_timestamps(
                    stream_id, TimeDomain.TIME_CODE)

                for ts in timestamps:
                    device_pose = device_pose_provider.get_pose_at_timestamp(
                        timestamp_ns=ts,
                        time_query_options=TimeQueryOptions.CLOSEST,
                        time_domain=TimeDomain.TIME_CODE,
                    )
                    if device_pose is None:
                        continue
                    T_world_device = device_pose.pose3d.T_world_device

                    hand_poses_with_dt = umetrack_provider.get_pose_at_timestamp(
                        timestamp_ns=ts,
                        time_query_options=TimeQueryOptions.CLOSEST,
                        time_domain=TimeDomain.TIME_CODE,
                    )
                    if hand_poses_with_dt is None:
                        continue

                    for hand_pose_data in hand_poses_with_dt.pose3d_collection.poses.values():
                        # Unlike HOT3DVRSDetectionDataset (which keeps
                        # every frame, including ones with no visible
                        # hand, as negative "exists=0" examples for
                        # DetNet's detection task), KeyNet has no "hand
                        # exists" signal to train -- it only ever
                        # consumes an already-cropped hand image. A hand
                        # that's not visible enough simply has no valid
                        # crop, so we skip it entirely.
                        box_result = hand_box2d_provider.get_bbox_at_timestamp(
                            stream_id=stream_id,
                            timestamp_ns=ts,
                            time_query_options=TimeQueryOptions.CLOSEST,
                            time_domain=TimeDomain.TIME_CODE,
                        )
                        if box_result is None:
                            continue

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

                        self.samples.append((
                            device_data_provider, umetrack_provider,
                            T_device_camera, camera_calibration,
                            T_world_device, hand_pose_data,
                            stream_id, ts,
                        ))
            
        self.actual_size = len(self.samples)
        self.num_times_to_repeat = 1

    def __len__(self):
        return self.actual_size * self.num_times_to_repeat

    def __getitem__(self, idx):
        idx = idx % self.actual_size
        (device_data_provider, umetrack_provider, T_device_camera,
         camera_calibration, T_world_device, hand_pose_data,
         stream_id, ts) = self.samples[idx]

        image = device_data_provider.get_image(ts, stream_id)
        if image is None:
            return self._empty_sample()

        hot3d_landmarks = umetrack_provider.get_hand_landmarks(hand_pose_data)
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
    args = parser.parse_args()

    ds = HOT3DKeypointDataset(
        sequence_dirs=args.sequence_dirs,
        hot3d_repo_root=args.hot3d_repo_root,
        object_library_path=args.object_library_path,
        min_visibility_ratio=args.min_visibility_ratio,
    )
    print(f"{len(ds)} samples")
    samp = ds[0]
    print({k: (v.shape if hasattr(v, "shape") else v) for k, v in samp.items()})