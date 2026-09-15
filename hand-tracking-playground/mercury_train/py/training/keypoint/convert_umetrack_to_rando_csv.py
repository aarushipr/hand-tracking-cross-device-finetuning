"""
Converts Meta's UmeTrack / HOT3D data into the flat CSV plus image folder that
RandoData.py already reads, so no new loader path is needed. build_hand_dataset
with output_crops=True already gives a 128x128 crop per hand through a
distortion-free crop camera in the same frame as the pose, so no reprojection
happens here. Superseded by HOT3DKeypointDataset for the thesis. Note that
RandoData's rotate_hand() drops the depth column, so depth from here is lost.
"""

import argparse
import os
import sys

import cv2
import numpy as np


# --- Landmark order mapping ------------------------------------------------
# VERIFIED against subject_000_separate_hand_000000, not guessed from the layout.
# Thumb found by rest-pose angular deviation, 39.9 deg against 4.7-15.4 for the rest.
# Landmarks 0-4 are the fingertips; 5-19 are mcp/pip/dip per finger:
#   thumb 5,6,7,0   index 8,9,10,1   middle 11,12,13,2   ring 14,15,16,3   pinky 17,18,19,4
NUM_UMETRACK_LANDMARKS = 20
NUM_PROJECT_KEYPOINTS = 21  # wrist + 4 joints x 5 fingers, per ArtificialData._25_to_21

# Raw 20-landmark index per project slot 1-20; slot 0 is the wrist, filled separately.
# mcp/pip/dip/tip per finger, thumb to pinky, as ArtificialData._25_to_21 produces.
_UMETRACK_LANDMARK_PERMUTATION = [
    5, 6, 7, 0,      # thumb: mcp, pip, dip, tip
    8, 9, 10, 1,     # index: mcp, pip, dip, tip
    11, 12, 13, 2,   # middle: mcp, pip, dip, tip
    14, 15, 16, 3,   # ring: mcp, pip, dip, tip
    17, 18, 19, 4,   # pinky: mcp, pip, dip, tip
]


def umetrack_landmarks_to_project_keypoints(
    wrist_world_pos: np.ndarray, landmarks_world: np.ndarray
) -> np.ndarray:
    """
    wrist_world_pos: (3,) wrist translation, from UmeTrackHandPose.wrist_xform
    landmarks_world: (20, 3) output of umetrack_hand_model.forward_kinematics

    Returns (21, 3) in this project's wrist-first joint order. See the
    _UMETRACK_LANDMARK_PERMUTATION comment above for how this was derived
    and verified.
    """
    assert landmarks_world.shape == (NUM_UMETRACK_LANDMARKS, 3), landmarks_world.shape
    out = np.zeros((NUM_PROJECT_KEYPOINTS, 3), dtype=np.float64)
    out[0] = wrist_world_pos
    out[1:21] = landmarks_world[_UMETRACK_LANDMARK_PERMUTATION]
    return out


def build_rando_csv_row(
    filename: str,
    kps_px_depth_21: np.ndarray,  # (21, 3): px, py, depth in the crop image
    is_right: bool,
) -> str:
    """
    Reproduces, in reverse, the exact column layout that
    RandoData.py's RandoDataset.__getitem__ reads:

        b = row
        acc_idx = 0
        filename = b[acc_idx]; acc_idx += 1

        kps = np.zeros((22, 3))
        for i in range(22):
            for j in range(3):
                kps[i][j] = b[acc_idx]; acc_idx += 1

        for i in range(22):
            gt_xy_valid[i] = b[acc_idx]; acc_idx += 2   # <- consumes 2 columns
            gt_depth_valid[i] = b[acc_idx]; acc_idx += 1  # <- consumes 1 column

        is_right = bool(b[acc_idx]); acc_idx += 1

        # The optional trailing mask column isn't written; UmeTrack crops have no mask.

    So per row: 1 (filename) + 22*3 (kps) + 22*3 (validity, 3 cols/joint even
    though only 2 of the 3 are actually read) + 1 (is_right) = 134 fields.
    (Verified against RandoDataset's actual parsing logic with a
    round-trip test.)

    We only have real values for the project's 21 keypoints; row/joint 21
    (the "22nd" slot) is written as zeros, RandoDataset's own
    `rotate_hand` helper only ever looks at the first 21 rows of `kps`
    anyway (hardcoded `np.zeros((21, 2))` / `range(21)`), so slot 21 is
    dead weight kept only to match the column count of the existing
    nikitha.csv / panoptic_*.csv / tom.csv files.
    """
    assert kps_px_depth_21.shape == (NUM_PROJECT_KEYPOINTS, 3)

    kps22 = np.zeros((22, 3), dtype=np.float64)
    kps22[:21] = kps_px_depth_21

    fields = [f"|{filename}|"]

    for i in range(22):
        for j in range(3):
            fields.append(f"{kps22[i, j]:.6f}")

    # Mocap GT, so every real joint is valid; the padding 22nd is marked invalid.
    for i in range(22):
        valid = 1.0 if i < 21 else 0.0
        fields.append(f"{valid:.1f}")  # gt_xy_valid[i]
        fields.append("0.0")  # unused padding column consumed by acc_idx += 2
        fields.append(f"{valid:.1f}")  # gt_depth_valid[i]

    fields.append("1" if is_right else "0")

    return " ".join(fields)


CSV_HEADER = " ".join(
    ["filename"]
    + [f"kp{i}_{c}" for i in range(22) for c in ("x", "y", "z")]
    + [f"kp{i}_{c}" for i in range(22) for c in ("xy_valid", "_unused", "depth_valid")]
    + ["is_right"]
)


def convert(root: str, sequence_names: list, out_dir: str, out_name: str, max_frames: int = None):
    # Deferred import: third-party packages installed separately, see module docstring.
    from hand_tracking_toolkit.dataset import build_hand_dataset
    from hand_tracking_toolkit.dataset import HandSide
    from hand_tracking_toolkit.hand_models.umetrack_hand_model import forward_kinematics

    img_out_dir = os.path.join(out_dir, out_name)
    os.makedirs(img_out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{out_name}.csv")

    dataset = build_hand_dataset(
        root,
        sequence_names,
        load_monochrome=True,
        load_rgb=False,
        output_crops=True,
        crop_size=128,
    )

    n_written = 0
    n_skipped_no_pose = 0

    with open(csv_path, "w", newline="\n") as csv_file:
        csv_file.write(CSV_HEADER + "\n")

        for sample_crops in dataset:
            # output_crops=True makes each item a list of HandCropData, one per visible hand.
            for crop in sample_crops:
                if crop.hand_pose is None or crop.hand_pose.umetrack is None:
                    n_skipped_no_pose += 1
                    continue
                if crop.hand_shape is None or crop.hand_shape.umetrack is None:
                    n_skipped_no_pose += 1
                    continue

                stream_ids = sorted(crop.images.keys())
                if not stream_ids:
                    continue
                stream_id = stream_ids[0]  # see caveats: only first stream used for now

                image = crop.images[stream_id]
                camera = crop.cameras[stream_id]

                landmarks, _, _ = forward_kinematics(
                    crop.hand_pose.umetrack, crop.hand_shape.umetrack, requires_mesh=False
                )
                landmarks = landmarks.detach().cpu().numpy()

                wrist_world_pos = (
                    crop.hand_pose.umetrack.wrist_xform.detach().cpu().numpy()[:3, 3]
                )

                kps_world = umetrack_landmarks_to_project_keypoints(
                    wrist_world_pos, landmarks
                )

                kps_px_depth = camera.world_to_window3(kps_world)

                is_right = crop.hand_pose.hand_side == HandSide.RIGHT

                seq_tag = os.path.basename(crop.url).replace(".tar", "")
                filename = f"{seq_tag}_{crop.frame_id:06d}_{'r' if is_right else 'l'}_{stream_id}.jpg"

                cv2.imwrite(os.path.join(img_out_dir, filename), image)
                csv_file.write(
                    build_rando_csv_row(filename, kps_px_depth, is_right) + "\n"
                )

                n_written += 1
                if max_frames is not None and n_written >= max_frames:
                    print(f"Reached --max-frames={max_frames}, stopping early.")
                    print(f"Wrote {n_written} rows, skipped {n_skipped_no_pose} (no pose/shape).")
                    return

    print(f"Wrote {n_written} rows to {csv_path}")
    print(f"Skipped {n_skipped_no_pose} crops with no usable pose/shape annotation.")
    print()
    print("Next step: add this to CombinedDataset.py's AllOfTheDatasetsCombined.__init__:")
    print(
        f'    b(RandoDataset(f"{{datasets_basepath}}/", "{out_name}.csv"), 0.6)  '
        "# weight is a starting guess, tune like the others"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Directory containing the .tar sequence files")
    parser.add_argument("--sequences", required=True, nargs="+", help="Sequence names, without .tar extension")
    parser.add_argument("--out-dir", required=True, help="Should match local_config.real_datasets_basepath")
    parser.add_argument("--out-name", default="umetrack", help="Base name for the .csv and image subfolder")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after writing this many rows (for a quick smoke test)")
    args = parser.parse_args()

    try:
        convert(args.root, args.sequences, args.out_dir, args.out_name, args.max_frames)
    except ImportError as e:
        print(
            "Missing dependency. Install with:\n"
            "  pip install --break-system-packages git+https://github.com/facebookresearch/hand_tracking_toolkit webdataset opencv-python",
            file=sys.stderr,
        )
        raise


if __name__ == "__main__":
    main()
