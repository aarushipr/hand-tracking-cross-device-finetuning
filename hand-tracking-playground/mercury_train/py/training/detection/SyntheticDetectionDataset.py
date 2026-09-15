import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation as R

import augmentation
import local_config
from a_structs import ImageWithBoundingBoxes, bbox


class SyntheticDetectionDataset(torch.utils.data.Dataset):
    """
    Reuses the same Blender-rendered synthetic sequences as the keypoint
    pipeline's ArtificialDataset (raw frames + hand_poses.csv +
    camera_info.csv under local_config.artificial_dataset_path), but derives
    a detection bounding box instead of calling into the KeyNet-specific
    ad4_stereographic_projection crop.

    Why this exists: as originally written, DetNet trained on 100% real
    data (HMDHandRects + EgoHands + EpicKitchens) with zero synthetic
    exposure, meaning DetNet's ability to generalize to a new XR device
    had nothing to do with the thesis's actual method (camera-randomized
    synthetic training). This gives DetNet the same synthetic training
    path KeyNet already has, so real sources can move to val/test instead
    of being required just to get a training signal at all.

    Method: rotate each frame's 3D hand joints into camera-local space
    using camera_info.csv's pose + quaternion, project with a standard
    pinhole model using camera_info.csv's per-frame fx/fy/cx/cy, and take
    the min/max extent (+ margin) as the bbox. Verified against real
    rendered frames by overlaying the projected points and bbox on the
    actual image before wiring this in, projection lines up tightly with
    the visible hand.

    Handedness: the generator only ever renders one hand per frame with no
    handedness label, so the projected bbox is always placed in slot 0.
    This is fine because augmentation.augment_image() already does a
    random horizontal flip that swaps the left/right bbox slots; see its
    `if flip:` block; so this still produces a left/right-balanced
    training signal without needing to invent a labeling scheme here.
    """

    def __init__(self, margin: float = 0.15):
        self.superroot = local_config.artificial_dataset_path
        self.margin = margin

        self.num_sequences = len(os.listdir(self.superroot))
        self.camera_poses_seq_array = np.zeros((self.num_sequences, 200, 7 + 4))
        self.hand_poses_seq_array = np.zeros((self.num_sequences, 200, 26 * 7))

        for i in range(self.num_sequences):
            seqname = f"seq{i}"
            self.camera_poses_seq_array[i] = pd.read_csv(
                os.path.join(self.superroot, seqname, "camera_info.csv"))
            self.hand_poses_seq_array[i] = pd.read_csv(
                os.path.join(self.superroot, seqname, "hand_poses.csv"))

        self.len = self.num_sequences * 200

    def __len__(self):
        return self.len

    def __getitem__(self, idx, retries_left=5):
        seq_idx = idx // 200
        frame_idx = idx % 200
        seqname = f"seq{seq_idx}"

        numstr = str(frame_idx).zfill(4)
        img_path = os.path.join(self.superroot, seqname, "imgs_color", f"file_name{numstr}.png")

        im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if im is None:
            # One corrupt frame shouldn't kill an unattended multi-hour job.
            print(f"[SyntheticDetectionDataset] WARNING: failed to load {img_path}. "
                  f"Substituting a random sample.")
            if retries_left <= 0:
                raise RuntimeError(
                    f"[SyntheticDetectionDataset] Too many consecutive failed "
                    f"samples -- last attempted {img_path}. This looks like a "
                    f"systemic data problem, not one bad frame."
                )
            return self.__getitem__(random.randrange(self.len), retries_left - 1)

        cam = self.camera_poses_seq_array[seq_idx, frame_idx]
        cam_pos = cam[0:3]
        cam_quat = cam[3:7]  # x, y, z, w; matches camera_info.csv column order
        fx, fy, cx, cy = cam[7], cam[8], cam[9], cam[10]

        joints = self.hand_poses_seq_array[seq_idx, frame_idx].reshape(26, 7)
        # Joints 0-24 are the hand; joint 25 is the elbow and is excluded.
        # Including it would blow the bbox out over empty background.
        joints_global = joints[0:25, 0:3]

        cam_rot_inv = R.from_quat(cam_quat).inv()
        joints_local = cam_rot_inv.apply(joints_global - cam_pos)

        # Camera looks down -Z (OpenXR/Monado convention), so forward distance is -z.
        depth = -joints_local[:, 2]
        depth = np.where(np.abs(depth) < 1e-6, 1e-6, depth)  # guard divide-by-zero
        u = cx + fx * (joints_local[:, 0] / depth)
        v = cy - fy * (joints_local[:, 1] / depth)

        x0, x1 = float(u.min()), float(u.max())
        y0, y1 = float(v.min()), float(v.max())
        w, h = x1 - x0, y1 - y0
        x0 -= w * self.margin
        x1 += w * self.margin
        y0 -= h * self.margin
        y1 += h * self.margin

        b = bbox((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)
        # Always slot 0 (left); see class docstring re: handedness.
        bbox_list = [b, None]

        e = ImageWithBoundingBoxes(image=im, bboxes=bbox_list)
        e = augmentation.augment_image(e)
        e = augmentation.imgwithboundingboxes320_to_heatmaps_2hand(e)
        return e


if __name__ == "__main__":
    d = SyntheticDetectionDataset()
    print(len(d))
    samp = d[50]
    print(samp)
