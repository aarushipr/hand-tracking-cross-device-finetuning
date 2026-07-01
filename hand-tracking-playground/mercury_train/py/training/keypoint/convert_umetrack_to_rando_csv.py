"""
Convert Meta's UmeTrack / HOT3D hand-tracking-challenge data (loaded via
facebookresearch/hand_tracking_toolkit) into the flat CSV + image-folder
format that RandoData.py's RandoDataset already knows how to read.

WHY THIS SHAPE OF CONVERTER
----------------------------
mercury_train already has an established pattern for plugging in "real"
(non-Blender) hand datasets: RandoDataset reads a space-delimited CSV
(filename, 21 3D keypoints, per-keypoint validity flags, is_right, optional
mask filename) plus a folder of grayscale crop images, and feeds them
through the same AugmentationMaker used by ArtificialDataset. Panoptic,
FreiHand, Tom OpenHands and "nikitha" are already wired in this way (see
CombinedDataset.py / KEYPOINT_PIPELINE.md, which explicitly calls out
"Future work: replace Tom with HOT3D or UmeTrack").

We deliberately do NOT try to shoehorn UmeTrack through the synthetic
ArtificialData.py / camera_info.csv / hand_poses.csv path. That path
expects a full 26-joint rig (position + orientation per bone, matching
this project's own Blender armature convention) produced by procedural
generation, and reprojects it with a from-scratch pinhole camera via the
`ad4_stereographic_projection` pybind module. Real motion-capture datasets
don't give you rig-space bone orientations for an arbitrary third-party
skeleton -- you'd need to solve a full retargeting problem (map UmeTrack's
own kinematic tree onto this project's rig, bone-length and axis
differences included) which cannot be verified without the dataset, a
working Blender + pybind build, and a lot of visual debugging. Reusing
RandoDataset instead means: no new loader code path, no retargeting, and
it slots into CombinedDataset.py's existing weighting/held-out-split logic
for free.

WHAT UMETRACK ACTUALLY GIVES US
--------------------------------
hand_tracking_toolkit's `build_hand_dataset(..., output_crops=True,
crop_size=128)` already does the hard part: it returns, per hand per
frame, a 128x128 image warped by a synthetic "crop camera" that's already
pointed at and centered on that hand (see hand_tracking_toolkit/
dataset.py: SampleDecoder / make_hand_crops / decode_hand_crop_params).
That crop camera is a `PinholePlaneCameraModel` with no lens distortion,
in the coordinate frame in which the hand's ground-truth pose is also
expressed. This means:

  1. We do not need to do any of our own reprojection math or camera
     placement -- their crop camera already solved that (this is the same
     "look at the hand" problem our Blender miniball-based camera placement
     solves for synthetic data).
  2. Ground truth keypoints are obtained by running UmeTrack forward
     kinematics (`umetrack_hand_model.forward_kinematics`) on the
     per-frame `joint_angles` + `wrist_xform`, using the per-sequence
     `hand_shape.umetrack` bone lengths, then projecting the resulting
     world-space landmarks through the crop camera with `world_to_window3`
     (returns pixel x, pixel y, and camera-space depth in one call).

CAVEATS THAT COULD NOT BE VERIFIED WITHOUT THE ACTUAL DATASET
---------------------------------------------------------------
- Landmark order: `forward_kinematics` returns 20 landmarks
  (`UMETRACK_TO_CANONICAL_LANDMARK_MAPPING = range(20)`, i.e. the model's
  own native order -- the toolkit source does not document what that
  order is beyond "canonical landmark mapping"). This project's convention
  is 21 keypoints: wrist + 4 joints x 5 fingers (see `_25_to_21` in
  ArtificialData.py and `palm_length_2d`'s indexing of 0/5/9/17 in
  RandoData.py). We assume UmeTrack's 20 landmarks are the 4-per-finger x
  5-finger set (no separate wrist entry, since wrist is already known via
  `wrist_xform`), and put the wrist translation from `wrist_xform` at
  index 0. THIS ORDERING IS UNVERIFIED. Before trusting this for real
  training, generate a handful of crops, draw the resulting `kps` array on
  top of the crop image (e.g. with RandoData.py's own
  `geo.draw_hand_rainbow_pts`/`geo.draw_21_hand_lines`, or
  hand_tracking_toolkit.visualization.visualize_hand_crop_data with
  pose_type="umetrack"), and eyeball whether fingertips land on
  fingertips. If the order is wrong, it is very likely a fixed
  permutation (5 fingers x 4 joints, possibly ordered
  thumb/index/middle/ring/pinky vs. this project's
  thumb/index/middle/ring/pinky x [mcp,pip,dip,tip] -- adjust
  `umetrack_landmarks_to_project_keypoints` below once confirmed).
- Multiple camera streams: each UmeTrack frame has 2-4 synchronized
  monochrome streams (plus optional RGB). This script only takes the
  first available stream per crop to keep the first version simple; using
  every stream would multiply the amount of usable training data for
  free, and is a natural follow-up once this path is confirmed to work.
- has_depth / 3D supervision: unlike Panoptic/FreiHand/Tom (2D-only, per
  KEYPOINT_PIPELINE.md), UmeTrack is a proper mocap dataset, so we *do*
  have accurate depth. RandoDataset's `do_one_augmentation` infers
  `has_depth` purely from whether the keypoints array passed in is shape
  (21,3) or (21,2) -- but note RandoDataset's own `rotate_hand` helper
  currently discards the 3rd (depth) column before calling
  `do_one_augmentation` (see RandoData.py's `crop`/`rotate_hand`, which
  hardcode `np.zeros((21,2))`). That means today, depth from this
  converter would be silently dropped unless RandoData.py is also updated
  to preserve it -- flagged here rather than silently "fixed" as part of
  this converter, since that's a behavior change to code this converter
  doesn't own.

LICENSING
---------
See UMETRACK_LICENSE_NOTES.md next to this script before running it on
real downloaded data.

USAGE
-----
    pip install --break-system-packages git+https://github.com/facebookresearch/hand_tracking_toolkit
    pip install --break-system-packages webdataset opencv-python

    python convert_umetrack_to_rando_csv.py \
        --root /path/to/downloaded/umetrack/train \
        --sequences subject_000_separate_hand_000000 subject_000_hand_hand_000001 \
        --out-dir /media/moses/traindata-jakob/keypoint_real_data/munge_april26 \
        --out-name umetrack

This writes:
    {out-dir}/{out-name}/*.jpg          (128x128 grayscale crops)
    {out-dir}/{out-name}.csv            (RandoDataset-compatible CSV)

Then add one line to CombinedDataset.py (see bottom of this file for the
exact snippet) to fold it into training.
"""

import argparse
import os
import sys

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Landmark order mapping. UNVERIFIED -- see caveats above. This is a
# best-effort placeholder (identity-ish: wrist, then the 20 UmeTrack
# landmarks in their native order) so the pipeline is at least plumbed
# end-to-end; treat the exact permutation as a TODO once real data/images
# are available to check against.
# ---------------------------------------------------------------------------
NUM_UMETRACK_LANDMARKS = 20
NUM_PROJECT_KEYPOINTS = 21  # wrist + 4 joints x 5 fingers, per ArtificialData._25_to_21


def umetrack_landmarks_to_project_keypoints(
    wrist_world_pos: np.ndarray, landmarks_world: np.ndarray
) -> np.ndarray:
    """
    wrist_world_pos: (3,) wrist translation, from UmeTrackHandPose.wrist_xform
    landmarks_world: (20, 3) output of umetrack_hand_model.forward_kinematics

    Returns (21, 3) in this project's wrist-first joint order.
    UNVERIFIED mapping -- see module docstring.
    """
    assert landmarks_world.shape == (NUM_UMETRACK_LANDMARKS, 3), landmarks_world.shape
    out = np.zeros((NUM_PROJECT_KEYPOINTS, 3), dtype=np.float64)
    out[0] = wrist_world_pos
    out[1:21] = landmarks_world
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

        # optional trailing mask filename column, only if present, is not
        # written by this converter (UmeTrack crops have no matting mask).

    So per row: 1 (filename) + 22*3 (kps) + 22*3 (validity, 3 cols/joint even
    though only 2 of the 3 are actually read) + 1 (is_right) = 134 fields.
    (Verified against RandoDataset's actual parsing logic -- see the
    round-trip test in verify_umetrack_csv_format.py next to this script.)

    We only have real values for the project's 21 keypoints; row/joint 21
    (the "22nd" slot) is written as zeros -- RandoDataset's own
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

    # Validity columns: UmeTrack ground truth comes from a mocap rig and is
    # considered fully valid for every real joint; the padding 22nd joint
    # (unused downstream) is marked invalid for honesty even though nothing
    # currently reads it.
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
    # Deferred import: these are third-party packages the user needs to
    # install separately (see module docstring), not part of this repo.
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
            # output_crops=True makes each dataset item a *list* of
            # HandCropData, one per hand visible in that frame (0, 1, or 2).
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
