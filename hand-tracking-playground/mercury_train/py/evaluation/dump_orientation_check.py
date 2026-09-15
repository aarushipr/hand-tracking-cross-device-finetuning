"""
Dumps the 160x160 letterboxed crop of a few frames at all four camera orientations
with the ground-truth box drawn on each. Pick the one where the scene is upright
AND the box lands on the hand; that is the device's value in
preprocess_baseline.DEVICE_ORIENTATION. Run it on the cluster, where the HOT3D
data and projectaria_tools live. See --help for arguments.
"""
import argparse
import os
import sys

import cv2
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from eval_detnet import Hot3dRawFrameSource  # noqa: E402
from preprocess_baseline import (  # noqa: E402
    DETECTION_INPUT_SIZE, compute_blackbar_transform, rotate_points_upright,
    rotate_upright,
)

ORIENTATIONS = [0, 90, 180, 270]


def transform_box_corners(box, orientation, in_w, in_h, go):
    """box = (left, top, right, bottom) in raw sensor coords. Returns an
    axis-aligned (left, top, right, bottom) in letterboxed-crop coords: the
    four corners are rotated upright the same way the image is, then run
    through the blackbar affine, and their bounding box is taken. Exact for
    0/180 and close enough at 90/270 for eyeballing box alignment."""
    left, top, right, bottom = box
    corners = np.array([
        [left, top], [right, top], [right, bottom], [left, bottom],
    ], dtype=np.float64)
    rotated = rotate_points_upright(corners, orientation, in_w, in_h)
    homogeneous = np.concatenate([rotated, np.ones((4, 1))], axis=1)
    transformed = homogeneous @ go.T  # (4, 2)
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

    # Only frames with a GT box; without one there is no box-alignment to check.
    checked = 0
    idx = 0
    while checked < args.num_frames and idx < len(source):
        seq_name, headset, stream_id, ts, image, gt_boxes = source.get(idx)
        idx += 1
        if image is None or not gt_boxes:
            continue

        h, w = image.shape[:2]
        panels = []
        for orientation in ORIENTATIONS:
            upright = rotate_upright(image, orientation)
            uh, uw = upright.shape[:2]
            go = compute_blackbar_transform(uw, uh, DETECTION_INPUT_SIZE, DETECTION_INPUT_SIZE)
            crop = cv2.warpAffine(upright, go, (DETECTION_INPUT_SIZE, DETECTION_INPUT_SIZE))
            panel = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)

            for gt in gt_boxes:
                l, t, r, b = transform_box_corners(gt["box"], orientation, w, h, go)
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
    print(f"value for --device {args.device}; put it in preprocess_baseline.DEVICE_ORIENTATION.")
    print("If it's inconsistent across frames, or no orientation looks right, something else is off")
    print("(e.g. the hand_index left/right slot assumption) -- don't just pick the closest-looking one.")


if __name__ == "__main__":
    main()
