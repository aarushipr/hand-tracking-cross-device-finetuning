"""
PhanesimDetectionDataset -- loads Phanesim's synthetic hand-detection data
(per-frame 2D box + presence, from hand_rect.csv) for DetNet's second
fine-tuning phase, on top of the HOT3D fine-tuning already done.

Phanesim was built by wany (another student of the same supervisor) with a
Blender-based renderer at /home/stud/wany/phanesim/. Two batches were mirrored
to /storage/group/dataset_mirrors/01_incoming/phanesim20260908/ and copied
into /storage/user/praa/phanesim_dataset/ as `dataset` (3601 clips) and
`dataset2` (1400 clips); pass both roots in to pool them.

Why this dataset only needs hand_rect.csv, unlike HOT3DVRSDetectionDataset
(which projects 3D joints through a camera pose): Phanesim already ships a
ready-made 2D box per frame, so no camera geometry is needed here at all.
That also means this loader is unaffected by the still-open question of how
to recover per-frame camera pose for KeyNet's depth target (see
PhanesimKeypointDataset.py) -- detection only ever needed pixel coordinates,
which we already have directly.

Verified against real data before writing this (see thesis-work chat log,
2026-09-08):
  - Box convention: (x, y) is the TOP-LEFT corner, not the center -- cross-
    checked against joints_2d.csv's pixel ranges for the same frame, and
    later confirmed visually by drawing the box on frame_000002.png of
    dataset/clip_00000 (both the left- and right-hand boxes wrap the visible
    hand tightly, including a right-hand box that correctly captured a
    single finger clipped by the frame edge).
  - Orientation: dataset/clip_00000 frame 0 and dataset2/clip_00000 frame 0
    both render upright (vertical walls/trees, no cocked horizon), unlike
    HOT3D's raw device frames which need preprocess_baseline.rotate_upright.
    Hence orientation=0 by default here, not per-device 270.
  - Margin: the drawn boxes show visible padding beyond the hand silhouette,
    not a pixel-tight box -- so no additional margin is applied by default
    (margin=0.0). If a future check finds the boxes too tight after all,
    pass margin > 0.
  - Left/right: hand_rect.csv has separate left_/right_ prefixed columns, so
    handedness is unambiguous here (unlike HOT3DVRSDetectionDataset's
    hand_index 0/1 assumption for HOT3D).

Confirmed working end to end on the cluster 2026-09-09: 50185 samples loaded
across both roots, sample contents (160x160 crop, exists/center/size all in
sane ranges) look correct.

NOT yet verified, flag if wrong: whether every clip in both `dataset` and
`dataset2` shares the same camera intrinsics/orientation as clip_00000 --
only a handful of clips were spot-checked. If a future fine-tuning run's
loss looks wrong, re-check a few frames from clips other than clip_00000.
"""

import csv
import glob
import os
import random

import cv2
import numpy as np
import torch

import augmentation
import header
from a_structs import ImageWithBoundingBoxes, bbox
from py.evaluation import preprocess_baseline as _pp


def discover_clip_dirs(dataset_roots: list) -> list:
    """Sorted list of clip directories across dataset_roots that have both
    _done.json and cam_head0/hand_rect.csv. Factored out of __init__ so
    callers that need to split by CLIP rather than by frame (e.g. carving
    out a validation set without letting near-duplicate frames from the
    same clip leak across the split -- see trainer_detection_phanesim.py)
    use the exact same filtering the dataset itself does, rather than a
    second copy that could quietly drift out of sync.
    """
    clip_dirs = []
    for root in dataset_roots:
        for clip_dir in sorted(glob.glob(os.path.join(root, "clip_*"))):
            done_path = os.path.join(clip_dir, "_done.json")
            rect_path = os.path.join(clip_dir, "cam_head0", "hand_rect.csv")
            if os.path.exists(done_path) and os.path.exists(rect_path):
                clip_dirs.append(clip_dir)
    return clip_dirs


class PhanesimDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_roots: list = None, clip_dirs: list = None,
                 orientation: int = 0, margin: float = 0.0, augment: bool = True):
        """Either pass dataset_roots (globs every clip_* under each root --
        the normal case) or clip_dirs (an explicit, already-filtered list
        of clip directories -- used by trainer_detection_phanesim.py to
        build separate train/val datasets from a single clip-level split).
        """
        if clip_dirs is None:
            if not dataset_roots:
                raise ValueError(
                    "[PhanesimDetectionDataset] Provide either dataset_roots "
                    "or clip_dirs.")
            clip_dirs = discover_clip_dirs(dataset_roots)

        self.orientation = orientation
        self.margin = margin
        self.augment = augment

        self._samples = []  # list of (clip_dir, frame_idx)
        for clip_dir in clip_dirs:
            rect_path = os.path.join(clip_dir, "cam_head0", "hand_rect.csv")
            with open(rect_path) as f:
                n_frames = sum(1 for _ in f) - 1  # minus header row
            for i in range(n_frames):
                self._samples.append((clip_dir, i))

        if not self._samples:
            raise RuntimeError(
                f"[PhanesimDetectionDataset] Found 0 samples across "
                f"{len(clip_dirs)} clip dir(s). Check the paths and that "
                f"clips have both _done.json and cam_head0/hand_rect.csv.")

        print(f"[PhanesimDetectionDataset] {len(self._samples)} samples from "
              f"{len(clip_dirs)} clip(s) (orientation={self.orientation}, "
              f"margin={self.margin}, augment={self.augment})")

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx, retries_left=5):
        clip_dir, frame_idx = self._samples[idx]
        img_path = os.path.join(clip_dir, "cam_head0", f"frame_{frame_idx:06d}.png")

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            # Mirrors SyntheticDetectionDataset's resilience approach -- one
            # bad frame shouldn't kill an unattended multi-hour job, but too
            # many in a row means something systemic (bad path, corrupt
            # mirror) rather than one-off.
            print(f"[PhanesimDetectionDataset] WARNING: failed to load "
                  f"{img_path}. Substituting a random sample.")
            if retries_left <= 0:
                raise RuntimeError(
                    f"[PhanesimDetectionDataset] Too many consecutive failed "
                    f"samples -- last attempted {img_path}. This looks like a "
                    f"systemic data problem, not one bad frame.")
            return self.__getitem__(random.randrange(len(self)), retries_left - 1)

        if self.orientation:
            image = _pp.rotate_upright(image, self.orientation)

        rect_path = os.path.join(clip_dir, "cam_head0", "hand_rect.csv")
        with open(rect_path) as f:
            row = list(csv.DictReader(f))[frame_idx]

        # Slot 0 = left, slot 1 = right -- matches KeyNet's JOINT_NAMES-style
        # convention and, unlike HOT3D, is unambiguous here since
        # hand_rect.csv names left_/right_ explicitly rather than an
        # untyped hand index.
        bbox_list = [None, None]
        for slot, prefix in ((0, "left"), (1, "right")):
            if row.get(f"{prefix}_present") != "1":
                continue
            x = float(row[f"{prefix}_x"])
            y = float(row[f"{prefix}_y"])
            w = float(row[f"{prefix}_w"])
            h = float(row[f"{prefix}_h"])
            cx, cy = x + w / 2.0, y + h / 2.0
            if self.margin:
                w *= (1 + 2 * self.margin)
                h *= (1 + 2 * self.margin)
            bbox_list[slot] = bbox(cx, cy, w, h)

        e = ImageWithBoundingBoxes(image=image, bboxes=bbox_list)
        e = augmentation.augment_image(e, deterministic=not self.augment)
        e = augmentation.imgwithboundingboxes320_to_heatmaps_2hand(e)
        return e


if __name__ == "__main__":
    import py.training.detection.local_config as local_config
    roots = getattr(local_config, "phanesim_dataset_roots", None)
    if not roots:
        raise RuntimeError(
            "Set local_config.phanesim_dataset_roots to a list of dataset "
            "root paths (e.g. ['/storage/user/praa/phanesim_dataset/dataset', "
            "'/storage/user/praa/phanesim_dataset/dataset2']) before running "
            "this as a smoke test.")
    d = PhanesimDetectionDataset(dataset_roots=roots)
    print(len(d))
    samp = d[0]
    print(samp)
