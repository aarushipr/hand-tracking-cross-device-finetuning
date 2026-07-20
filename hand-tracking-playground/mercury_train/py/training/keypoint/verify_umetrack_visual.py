"""
Visual sanity check for convert_umetrack_to_rando_csv.py's output.

The converter's landmark ordering (UmeTrack's 20 native landmarks -> this
project's 21-keypoint wrist-first convention) is flagged UNVERIFIED in that
script's docstring. This draws the resulting keypoints on top of a handful
of converted crop images so a human can eyeball whether fingertips land on
fingertips, rather than trusting the guessed ordering blindly.

Usage (run on the machine that has the converted output, e.g. the cluster):
    python verify_umetrack_visual.py \
        --csv /storage/user/praa/umetrack_test/umetrack.csv \
        --img-dir /storage/user/praa/umetrack_test/umetrack \
        --out-dir /storage/user/praa/umetrack_test/viz \
        --num-samples 8

Then copy the --out-dir folder back to a machine with a display (or into
this project's mounted thesis folder) to actually look at them:
    scp -r praa@atcremers41:/storage/user/praa/umetrack_test/viz ./umetrack_viz_check
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd

# Wrist (kp0) + 4 joints x 5 fingers, in this project's convention
# (thumb, index, middle, ring, pinky), each finger MCP -> PIP -> DIP -> TIP.
FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]
FINGER_COLORS = [
    (255, 0, 0),    # thumb: blue (BGR)
    (0, 255, 0),    # index: green
    (0, 255, 255),  # middle: yellow
    (0, 128, 255),  # ring: orange
    (255, 0, 255),  # pinky: magenta
]
WRIST_COLOR = (255, 255, 255)  # white


def draw_sample(img_path, row, out_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"  WARNING: couldn't load {img_path}")
        return

    img = cv2.resize(img, (img.shape[1] * 3, img.shape[0] * 3), interpolation=cv2.INTER_NEAREST)
    scale = 3

    # kp0 = wrist
    wx, wy = row["kp0_x"] * scale, row["kp0_y"] * scale
    cv2.circle(img, (int(wx), int(wy)), 5, WRIST_COLOR, -1)
    cv2.putText(img, "wrist", (int(wx) + 6, int(wy)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WRIST_COLOR, 1)

    # kp1..kp20 = 5 fingers x 4 joints (mcp, pip, dip, tip), per the
    # converter's assumed ordering -- this is exactly what we're checking.
    kp_idx = 1
    for finger_i, finger_name in enumerate(FINGER_NAMES):
        color = FINGER_COLORS[finger_i]
        pts = []
        for joint_i in range(4):
            x = row[f"kp{kp_idx}_x"] * scale
            y = row[f"kp{kp_idx}_y"] * scale
            pts.append((int(x), int(y)))
            cv2.circle(img, (int(x), int(y)), 4, color, -1)
            kp_idx += 1
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, color, 1)
        cv2.line(img, (int(wx), int(wy)), pts[0], color, 1)
        cv2.putText(img, finger_name, pts[-1], cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.imwrite(out_path, img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--img-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.csv, delimiter=" ", quotechar="|")
    print(f"Loaded {len(df)} rows from {args.csv}")

    n = min(args.num_samples, len(df))
    # Spread samples across the file rather than just the first N, in case
    # the first sequence/frames are atypical.
    indices = np.linspace(0, len(df) - 1, n, dtype=int)

    for i, idx in enumerate(indices):
        row = df.iloc[idx]
        img_path = os.path.join(args.img_dir, row["filename"])
        out_path = os.path.join(args.out_dir, f"check_{i:02d}_{row['filename']}")
        draw_sample(img_path, row, out_path)
        print(f"  wrote {out_path}")

    print(f"\nDone. Look at the images in {args.out_dir} -- for each finger, "
          f"the 4 dots (mcp->pip->dip->tip) should trace along that actual "
          f"finger in the image, ending at its tip. If e.g. the 'index' "
          f"dots trace the thumb instead, the landmark ordering in "
          f"convert_umetrack_to_rando_csv.py needs a permutation fix.")


if __name__ == "__main__":
    main()
