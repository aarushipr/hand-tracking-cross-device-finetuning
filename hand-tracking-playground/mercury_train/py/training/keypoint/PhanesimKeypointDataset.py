"""
PhanesimKeypointDataset -- loads Phanesim's synthetic hand-keypoint data
(21 2D joints per hand, from joints_2d.csv) for KeyNet's second fine-tuning
phase, on top of the HOT3D fine-tuning already done.

Phanesim was built by wany (another student of the same supervisor). See
PhanesimDetectionDataset.py's docstring for how the data got onto this
cluster and its dataset_roots convention (`dataset` + `dataset2`, pass
both to pool them).

WHY DEPTH IS INVALID FOR EVERY SAMPLE HERE (as of 2026-09-09)
---------------------------------------------------------------
KeyNet's depth target is not raw camera distance -- it's each joint's
camera-space depth relative to the wrist/middle-proximal scale (see
HOT3DKeypointDataset.py's docstring for the exact formula). That requires
the camera's pose for the frame, and joints_3d.csv only gives world-space
positions, not camera-relative ones. sequence.json's head_camera block
(rest_position/rest_forward) was investigated as a way to recover this,
including a DIY numerical fit (scipy.optimize.least_squares against real
3D/2D correspondences) -- it did not converge reliably across frames and
was abandoned. This is now a question queued for wany (camera motion
within a clip + the exact lens distortion formula); until it's answered,
every sample here is built with has_depth=0 and an all-zero
depth_valid_per_joint, exactly like HOT3DKeypointDataset's own
_empty_sample() placeholder path -- NOT a "no hand" placeholder (is_hand
stays 1, xy stays valid), just "no depth supervision available".
Concretely: gt_px passed into do_one_augmentation below is built as
(21,2), not (21,3), so it infers has_depth=False purely from that shape
(see its own `has_depth = gt_px.shape == (21,3)` check) -- has_depth and
depth_valid_per_joint are then also set explicitly below so this intent
doesn't silently depend on that shape trick alone if do_one_augmentation
is ever refactored. elbow/curls are zeroed for the same reason
HOT3DKeypointDataset zeroes them (no supervision available), and
settings.elbow_loss_mul / curls_loss_mul are already 0 so this doesn't
change anything currently trained.

JOINT NAME MAPPING -- verified directly, not from memory
---------------------------------------------------------
Retrieved dataset/clip_00000/cam_head0/joints_2d.csv's real header on
2026-09-09 (via scp to the laptop) rather than trusting an earlier
recollection. Its per-hand columns are, in this exact order:
    Wrist, ThumbMetacarpal, ThumbProximal, ThumbDistal, ThumbTip,
    IndexProximal, IndexIntermediate, IndexDistal, IndexTip,
    MiddleProximal, MiddleIntermediate, MiddleDistal, MiddleTip,
    RingProximal, RingIntermediate, RingDistal, RingTip,
    LittleProximal, LittleIntermediate, LittleDistal, LittleTip
This is index-for-index IDENTICAL (21 entries, same order, Little==pinky)
to this project's own canonical joint order -- see
py/training/common/hot3d_keypoint_mapping.py's _HOT3D_LANDMARK_PERMUTATION
comment and py/evaluation/eval_keynet.py's JOINT_NAMES, both:
    wrist, thumb_mcp, thumb_pxm, thumb_dst, thumb_tip,
    index_pxm, index_int, index_dst, index_tip,
    middle_pxm, middle_int, middle_dst, middle_tip,
    ring_pxm, ring_int, ring_dst, ring_tip,
    pinky_pxm, pinky_int, pinky_dst, pinky_tip
So no permutation is needed here at all -- unlike HOT3D, which had to
permute AND approximate thumb_mcp as a midpoint (no such joint exists in
HOT3D's own landmark set). Phanesim actually ships a real ThumbMetacarpal
joint, so this mapping is more direct than HOT3D's.

PER-JOINT OCCLUSION IS REAL HERE, NOT HYPOTHETICAL
----------------------------------------------------
Spot-checked dataset/clip_00000/cam_head0/joints_2d.csv: of its 15 rows,
3 have a hand with SOME (not all) of its 21 joints NaN -- occlusion
within an otherwise-tracked hand, same situation HOT3DKeypointDataset
handles with per-joint validity rather than an all-or-nothing gate. This
loader mirrors that: xy_valid_per_joint is computed per joint per hand
per frame directly from NaN in joints_2d.csv. A hand-frame enters the
index at all only if at least one of its 21 joints is non-NaN; if every
joint is NaN there is nothing to train on and it's skipped. (NOT gated
off hand_rect.csv's presence flag -- joints_2d.csv's own NaN pattern is
the more direct source of truth for what THIS network needs, and
PhanesimDetectionDataset.py already owns hand_rect.csv separately.)

One sample = one hand crop (matches HOT3DKeypointDataset's and
RandoDataset's contract) -- a frame with both hands tracked contributes
two samples.

NOT yet verified, flag if wrong: whether every clip in both `dataset` and
`dataset2` shares joints_2d.csv's exact column set/order as clip_00000 --
only that one clip was spot-checked.
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


# Index-for-index identical order to this project's canonical 21-joint
# convention -- see the module docstring's "JOINT NAME MAPPING" section.
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
    with at least one non-NaN joint. Reads only the CSV -- no image
    decoding -- matching HOT3DKeypointDataset's design of doing index
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
                    # Substitute the mean of this hand's own valid joints
                    # for invalid ones -- purely so crop()/
                    # add_2d_noise_to_keypoints (which look at all 21
                    # points together) aren't thrown off by a meaningless
                    # (0, 0) outlier. Never used as a training target:
                    # xy_valid_per_joint masks it out of the loss
                    # regardless. Mirrors HOT3DKeypointDataset.__getitem__.
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
            # Mirrors PhanesimDetectionDataset's resilience approach -- one
            # bad frame shouldn't kill an unattended multi-hour job, but too
            # many in a row means something systemic rather than one-off.
            print(f"[PhanesimKeypointDataset] WARNING: failed to load "
                  f"{img_path}. Substituting a random sample.")
            if retries_left <= 0:
                raise RuntimeError(
                    f"[PhanesimKeypointDataset] Too many consecutive failed "
                    f"samples -- last attempted {img_path}. This looks like "
                    f"a systemic data problem, not one bad frame.")
            return self.__getitem__(random.randrange(len(self)), retries_left - 1)

        if self.eval_mode:
            # Deterministic crop centred on the ground truth -- same
            # convention as HOT3DKeypointDataset's eval_mode.
            from py.evaluation import preprocess_baseline as _pp
            trans = _pp.keynet_crop_matrix(keypoints_px, is_right)
            predicted_px = None
        else:
            noisy_keypoints = add_2d_noise_to_keypoints(keypoints_px)
            trans = crop(image, noisy_keypoints, is_right)

        img_cropped = cv2.warpAffine(image, trans, (128, 128))
        keypoints_cropped = rotate_hand(keypoints_px, trans)

        if not self.eval_mode:
            # Matches RandoDataset's/HOT3DKeypointDataset's own "30% chance
            # of no predicted input" convention.
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

        # Depth-invalid fallback -- see module docstring. has_depth/gt_depth
        # are already zero from the (21,2) shape above; set explicitly too
        # so this doesn't silently depend on that alone.
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
