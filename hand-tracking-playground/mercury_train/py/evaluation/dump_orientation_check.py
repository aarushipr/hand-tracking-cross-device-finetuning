"""
Reusable version of the ad-hoc visual check that confirmed orientation=270
for Aria's SLAM cameras (see hot3d_baseline_detection_eval.py's module
docstring and git history: d9686ae, 20b3919). That check was done by hand,
one-off, for Aria only -- this script is the same idea, packaged so it can
be re-run against Quest (or re-verified against Aria) without redoing the
setup from scratch.

For a handful of real frames from a given sequence, dumps the 160x160
blackbar-letterboxed crop at all four camera orientations (0/90/180/270)
side by side, with the ground-truth hand box (if present for that frame)
drawn on top using the same affine transform applied to the image. Look at
the four crops and pick the one where:
  1. the hand/scene appears upright to a human viewer, AND
  2. the drawn GT box actually lands on the visible hand, not empty space.
That orientation value is what should be passed to
hot3d_baseline_detection_eval.py's --orientation flag for this device.

Usage (run on the cluster, where the real HOT3D data lives -- this can't
be run from a sandbox without the dataset and projectaria_tools installed):
    python dump_orientation_check.py \\
        --sequence-dir /storage/user/praa/hot3d_full_setup/hot3d_repo/hot3d/dataset/P0013_XXXXXXXX \\
        --hot3d-repo-root /storage/user/praa/hot3d_full_setup/hot3d_repo/hot3d \\
        --device Quest \\
        --out-dir /storage/user/praa/hot3d_full_setup/orientation_check_quest \\
        --num-frames 4

Pick a --sequence-dir for the device you're checking. For Quest, use any
Quest-only participant (e.g. one from CROSS_DEVICE_HELD_OUT_PARTICIPANTS's
complement) so this doesn't accidentally consume an eval-relevant sequence.
"""
import argparse
import os
import sys

import cv2
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from hot3d_baseline_detection_eval import (  # noqa: E402
    Hot3dRawFrameSource, compute_blackbar_transform, DETECTION_INPUT_SIZE,
)

ORIENTATIONS = [0, 90, 180, 270]


def transform_box_corners(box, go):
    """box = (left, top, right, bottom) in original image coords. Returns an
    axis-aligned (left, top, right, bottom) in letterboxed-crop coords, by
    transforming all four corners and taking their bounding box -- exact for
    0/180, a reasonable visual approximation for 90/270 (fine for eyeballing
    whether the box lands on the hand, which is all this script is for)."""
    left, top, right, bottom = box
    corners = np.array([
        [left, top, 1.0], [right, top, 1.0],
        [right, bottom, 1.0], [left, bottom, 1.0],
    ])
    transformed = corners @ go.T  # (4, 2)
    xs, ys = transformed[:, 0], transformed[:, 1]
    return xs.min(), ys.min(), xs.max(), ys.max()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequence-dir", required=True)
    parser.add_argument("--hot3d-repo-root", required=True)
    parser.add_argument("--device", required=True, choices=["Aria", "Quest"],
                         help="label only, used in the output filenames/printout")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-frames", type=int, default=4,
                         help="how many distinct frames (with a GT hand box) to check -- more than 1 "
                              "is worth doing since a single frame could coincidentally look plausible "
                              "at the wrong orientation")
    parser.add_argument("--min-visibility-ratio", type=float, default=0.2)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    source = Hot3dRawFrameSource([args.sequence_dir], args.hot3d_repo_root, args.min_visibility_ratio)
    print(f"{len(source)} (frame, camera-stream) samples in {args.sequence_dir}")

    # Only bother with frames that actually have a GT box -- otherwise there's
    # nothing to check the box-alignment half of the visual test against.
    checked = 0
    idx = 0
    while checked < args.num_frames and idx < len(source):
        seq_name, stream_id, ts, image, gt_boxes = source.get(idx)
        idx += 1
        if image is None or not gt_boxes:
            continue

        h, w = image.shape[:2]
        panels = []
        for orientation in ORIENTATIONS:
            go = compute_blackbar_transform(w, h, DETECTION_INPUT_SIZE, DETECTION_INPUT_SIZE, orientation)
            crop = cv2.warpAffine(image, go, (DETECTION_INPUT_SIZE, DETECTION_INPUT_SIZE))
            panel = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)

            for gt in gt_boxes:
                box = (gt["left"], gt["top"], gt["right"], gt["bottom"])
                l, t, r, b = transform_box_corners(box, go)
                color = (0, 255, 0) if gt["slot"] == 0 else (0, 255, 255)
                cv2.rectangle(panel, (int(l), int(t)), (int(r), int(b)), color, 1)

            panel = cv2.resize(panel, (DETECTION_INPUT_SIZE * 2, DETECTION_INPUT_SIZE * 2),
                                interpolation=cv2.INTER_NEAREST)
            cv2.putText(panel, f"orientation={orientation}", (6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            panels.append(panel)

        top_row = np.hstack(panels[0:2])
        bottom_row = np.hstack(panels[2:4])
        grid = np.vstack([top_row, bottom_row])

        out_path = os.path.join(args.out_dir, f"{args.device.lower()}_orientation_check_frame{checked:02d}.png")
        cv2.imwrite(out_path, grid)
        print(f"wrote {out_path}  (sequence={seq_name}, stream={stream_id}, ts={ts})")
        checked += 1

    if checked == 0:
        print("No frames with a GT hand box were found in this sequence -- try a different --sequence-dir.")
        return

    print(f"\nWrote {checked} orientation-check grids to {args.out_dir}.")
    print("For each grid (2x2: orientation 0 top-left, 90 top-right, 180 bottom-left, 270 bottom-right):")
    print("  1. Which panel shows the hand/scene upright, the way a person would naturally view it?")
    print("  2. In that same panel, does the drawn box (green=slot0, yellow=slot1) land on the actual")
    print("     visible hand, not empty space?")
    print("If the same orientation wins on both counts across all checked frames, that's your confirmed")
    print(f"value for --device {args.device} -- pass it to hot3d_baseline_detection_eval.py's --orientation.")
    print("If it's inconsistent across frames, or no orientation looks right, something else is off")
    print("(e.g. the hand_index left/right slot assumption) -- don't just pick the closest-looking one.")


if __name__ == "__main__":
    main()
