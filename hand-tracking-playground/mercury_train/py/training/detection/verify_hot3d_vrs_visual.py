"""
Visual sanity check for HOT3DVRSDetectionDataset: does the drawn box land on
an actual visible hand, not empty space? Supersedes the earlier
verify_hot3d_visual.py, which targeted a dataset class no longer in use.
This one additionally needs to confirm the
hand_index -> left/right slot assumption (0=left, 1=right) that
HOT3DVRSDetectionDataset.py flags as unverified, since box2d_hands.csv
doesn't self-document that.

Usage:
    python verify_hot3d_vrs_visual.py \
        --sequence-dirs /storage/user/praa/hot3d_full_setup/hot3d_repo/hot3d/dataset/P0003_c701bd11 \
        --hot3d-repo-root /storage/user/praa/hot3d_full_setup/hot3d_repo/hot3d \
        --out-dir /storage/user/praa/hot3d_full_setup/viz_vrs \
        --num-samples 12
"""
import argparse
import os
import sys

import cv2
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(_THIS_DIR, "../../../"))
from HOT3DVRSDetectionDataset import HOT3DVRSDetectionDataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-dirs", required=True, nargs="+")
    parser.add_argument("--hot3d-repo-root", required=True)
    parser.add_argument("--min-visibility-ratio", type=float, default=0.2)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=12)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ds = HOT3DVRSDetectionDataset(
        sequence_dirs=args.sequence_dirs,
        hot3d_repo_root=args.hot3d_repo_root,
        min_visibility_ratio=args.min_visibility_ratio,
    )
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
        for slot, label in enumerate(["slot0=left?", "slot1=right?"]):
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
    print("Two things to check by eye: (1) do boxes land on the actual hand, "
          "not empty space -- should be much more reliable now than the "
          "keypoint-projection version, since these come from Meta's own "
          "occlusion-aware ground truth; (2) is slot0(green) consistently "
          "the LEFT hand and slot1(yellow) consistently the RIGHT hand, or "
          "is it flipped/mixed -- if so, HOT3DVRSDetectionDataset.py's "
          "hand_index==0 -> slot0 assumption needs to change.")


if __name__ == "__main__":
    main()
