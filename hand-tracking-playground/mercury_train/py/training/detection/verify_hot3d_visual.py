"""
Visual sanity check for HOT3DDetectionDataset -- draws the derived
bounding box(es) on top of the post-augmentation crop, same approach used
to verify SyntheticDetectionDataset and the UmeTrack keypoint converter
before trusting them.

Usage:
    python verify_hot3d_visual.py \
        --root /storage/user/praa/hot3d_sample/train_aria \
        --sequences clip-001849 \
        --out-dir /storage/user/praa/hot3d_sample/viz \
        --num-samples 8
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from HOT3DDetectionDataset import HOT3DDetectionDataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--sequences", required=True, nargs="+")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ds = HOT3DDetectionDataset(root=args.root, sequence_names=args.sequences)
    print(f"{len(ds)} samples total")

    n = min(args.num_samples, len(ds))
    indices = np.linspace(0, len(ds) - 1, n, dtype=int)

    n_with_hand = 0
    for i, idx in enumerate(indices):
        samp = ds[int(idx)]
        raw = samp["image"][0]
        disp = ((raw - raw.min()) / (raw.max() - raw.min() + 1e-6) * 255).astype(np.uint8)
        img = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]

        any_hand = False
        for slot, label in enumerate(["left", "right"]):
            if samp["exists"][slot] < 0.5:
                continue
            any_hand = True
            cx = (samp["center_x"][slot] + 1) / 2 * w
            cy = (samp["center_y"][slot] + 1) / 2 * h
            size = samp["size"][slot] * w
            color = (0, 255, 0) if slot == 0 else (0, 255, 255)
            cv2.rectangle(img, (int(cx - size / 2), int(cy - size / 2)),
                          (int(cx + size / 2), int(cy + size / 2)), color, 1)
            cv2.putText(img, label, (int(cx - size / 2), int(cy - size / 2) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        if any_hand:
            n_with_hand += 1

        img = cv2.resize(img, (w * 3, h * 3), interpolation=cv2.INTER_NEAREST)
        out_path = os.path.join(args.out_dir, f"check_{i:02d}_idx{idx}.png")
        cv2.imwrite(out_path, img)
        print(f"wrote {out_path} (exists={samp['exists']})")

    print(f"\n{n_with_hand}/{n} samples had at least one visible hand box.")
    print("Look at the images -- do the boxes actually land on the hand(s), "
          "not empty space or the wrong object? Unlike the synthetic data, "
          "these are real photos so it should be easy to tell at a glance.")


if __name__ == "__main__":
    main()
