"""
Loads Phanesim's synthetic keypoint data (21 2D joints from joints_2d.csv) for
KeyNet's phase 2. Depth is invalid for every sample: the target needs per-frame
camera pose and joints_3d.csv gives only world-space positions, so has_depth is 0
throughout. This is not a "no hand" placeholder; is_hand and xy stay valid. No
joint permutation is needed, the column order already matches. Occlusion is per
joint, taken straight from NaN in joints_2d.csv.
"""
import csv
import glob
import os
import random
import sys

import cv2
import numpy as np
import torch

_mercury_train_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../')
if _mercury_train_root not in sys.path:
    sys.path.insert(0, _mercury_train_root)

from RandoData import crop, rotate_hand, add_2d_noise_to_keypoints
from maker_of_augmentations import AugmentationMaker
from a_aug_config import aug_config_validatoor


# Index-for-index identical to this project's canonical 21-joint order.
PHANESIM_JOINT_NAMES = [
    "Wrist",
    "ThumbMetacarpal", "ThumbProximal", "ThumbDistal", "ThumbTip",
    "IndexProximal", "IndexIntermediate", "IndexDistal", "IndexTip",
    "MiddleProximal", "MiddleIntermediate", "MiddleDistal", "MiddleTip",
    "RingProximal", "RingIntermediate", "RingDistal", "RingTip",
    "LittleProximal", "LittleIntermediate", "LittleDistal", "LittleTip",
]
assert len(PHANESIM_JOINT_NAMES) == 21


def discover_clip_dirs(dataset_roots: list) -> list:
    """Sorted list of clip directories (across all given roots) that have
    both _done.json and cam_head0/joints_2d.csv."""
    clip_dirs = []
    for root in dataset_roots:
        for clip_dir in sorted(glob.glob(os.path.join(root, "clip_*"))):
            done_path = os.path.join(clip_dir, "_done.json")
            joints_path = os.path.join(clip_dir, "cam_head0", "joints_2d.csv")
            if os.path.exists(done_path) and os.path.exists(joints_path):
                clip_dirs.append(clip_dir)
    return clip_dirs


def _clip_hand_frames(clip_dir):
    """
    Yields (frame_idx, is_right, keypoints_px (21,2) float32,
    xy_valid_per_joint (21,) float32) for every hand-frame in this clip
    with at least one non-NaN joint. Reads only the CSV; no image
    decoding, matching HOT3DKeypointDataset's design of doing index
    construction with no image I/O.
    """
    joints_path = os.path.join(clip_dir, "cam_head0", "joints_2d.csv")
    with open(joints_path) as f:
        reader = csv.DictReader(f)
        for frame_idx, row in enumerate(reader):
            for is_right, prefix in ((False, "left"), (True, "right")):
                kps = np.zeros((21, 2), dtype=np.float32)
                valid = np.zeros(21, dtype=np.float32)
                for j, name in enumerate(PHANESIM_JOINT_NAMES):
                    try:
                        u = float(row[f"{prefix}_{name}_u"])
                        v = float(row[f"{prefix}_{name}_v"])
                    except (KeyError, ValueError):
                        u = v = float("nan")
                    if not (np.isnan(u) or np.isnan(v)):
                        kps[j] = (u, v)
                        valid[j] = 1.0
                if not valid.any():
                    continue
                if not valid.all():
                    # Invalid joints take this hand's valid-joint mean so the crop isn't skewed.
                    # Masked out of the loss anyway. Mirrors HOT3DKeypointDataset.
                    valid_mean = kps[valid.astype(bool)].mean(axis=0)
                    kps[~valid.astype(bool)] = valid_mean
                yield frame_idx, is_right, kps, valid


class PhanesimKeypointDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_roots: list = None, clip_dirs: list = None,
                 eval_mode: bool = False):
        self.eval_mode = eval_mode
        self.augmaker = AugmentationMaker(aug_config_validatoor)

        if clip_dirs is None:
            if not dataset_roots:
                raise ValueError("Pass either dataset_roots or clip_dirs.")
            clip_dirs = discover_clip_dirs(dataset_roots)

        self._samples = []  # (clip_dir, frame_idx, is_right, kps, valid)
        for clip_dir in clip_dirs:
            for frame_idx, is_right, kps, valid in _clip_hand_frames(clip_dir):
                self._samples.append((clip_dir, frame_idx, is_right, kps, valid))

        if not self._samples:
            raise RuntimeError(
                f"[PhanesimKeypointDataset] Found 0 usable hand-frames "
                f"across {len(clip_dirs)} clip(s). Check the paths and "
                f"that joints_2d.csv exists and has non-NaN columns.")

        print(f"[PhanesimKeypointDataset] {len(self._samples)} samples "
              f"from {len(clip_dirs)} clip(s) (eval_mode={self.eval_mode})")

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx, retries_left=5):
        clip_dir, frame_idx, is_right, keypoints_px, xy_valid_per_joint = \
            self._samples[idx]

        img_path = os.path.join(clip_dir, "cam_head0", f"frame_{frame_idx:06d}.png")
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            # One bad frame shouldn't kill a long job; many in a row is systemic.
            print(f"[PhanesimKeypointDataset] WARNING: failed to load "
                  f"{img_path}. Substituting a random sample.")
            if retries_left <= 0:
                raise RuntimeError(
                    f"[PhanesimKeypointDataset] Too many consecutive failed "
                    f"samples -- last attempted {img_path}. This looks like "
                    f"a systemic data problem, not one bad frame.")
            return self.__getitem__(random.randrange(len(self)), retries_left - 1)

        if self.eval_mode:
            # Deterministic crop on the GT, as in HOT3DKeypointDataset's eval_mode.
            from py.evaluation import preprocess_baseline as _pp
            trans = _pp.keynet_crop_matrix(keypoints_px, is_right)
            predicted_px = None
        else:
            noisy_keypoints = add_2d_noise_to_keypoints(keypoints_px)
            trans = crop(image, noisy_keypoints, is_right)

        img_cropped = cv2.warpAffine(image, trans, (128, 128))
        keypoints_cropped = rotate_hand(keypoints_px, trans)

        if not self.eval_mode:
            # Matches RandoDataset's "30% chance of no predicted input" convention.
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

        # Set explicitly as well as via the (21,2) shape, so intent doesn't rest on that.
        ret["has_depth"] = np.float32(0)
        ret["depth_valid_per_joint"] = np.zeros(21, dtype=np.float32)
        ret["elbow"] = torch.zeros(3).float()
        ret["curls"] = torch.zeros(5).float()
        ret["xy_valid_per_joint"] = xy_valid_per_joint.astype(np.float32)
        return ret


if __name__ == "__main__":
    import local_config
    roots = getattr(local_config, "phanesim_dataset_roots", None)
    if not roots:
        raise RuntimeError(
            "Set local_config.phanesim_dataset_roots to a list of dataset "
            "root paths (e.g. ['/storage/user/praa/phanesim_dataset/dataset', "
            "'/storage/user/praa/phanesim_dataset/dataset2']) before running "
            "this as a smoke test.")
    d = PhanesimKeypointDataset(dataset_roots=roots)
    print(len(d))
    samp = d[0]
    print({k: (v.shape if hasattr(v, "shape") else v) for k, v in samp.items()})
