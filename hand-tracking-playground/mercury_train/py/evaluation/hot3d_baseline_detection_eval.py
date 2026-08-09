"""
Baseline (zero-shot) evaluation of Monado's shipped Mercury DetNet on HOT3D.

Runs `grayscale_detection_160x160.onnx` -- confirmed as the real production
detection model via TWO independent sources, not a guess:
  1. original-code/packaging-scripts/monado-data/CMakeLists.txt
     (`set(DETECTION_MODEL "hand/grayscale_detection_160x160.onnx")`)
  2. monado/src/xrt/tracking/hand/mercury/hg_model.cpp `init_hand_detection()`
     (`path /= "grayscale_detection_160x160.onnx";`)
-- against HOT3D's occlusion-aware ground truth (box2d_hands.csv), and reports
IoU / center-error / existence-accuracy, split by device (Aria vs Quest) and
stratified by `visibility_ratio`.

This is a BASELINE run only: no training, no fine-tuning. It answers "how
good is what already ships today," which is the number the improved model
(trained/fine-tuned on HOT3D, see trainer_detection.py) needs to beat.

=== Preprocessing: ported faithfully from hg_model.cpp, not reimplemented from
=== memory. Two functions matter and are the actual source of correctness risk
=== here -- read the docstrings before trusting any number this script prints.

1. `blackbar_letterbox()` -- ports `blackbar()` (hg_model.cpp lines ~24-100):
   scale-to-fit + center-pad (letterbox) into a square, with a camera-mount
   ROTATION applied via `orientation`. CONFIRMED 2026-08-08 for HOT3D Aria
   SLAM cameras: default is 270, verified by dumping the actual 160x160
   letterboxed crop at all four rotation values for a real frame and
   visually checking which one shows an upright hand/scene (see
   git history / thesis notes for the four sample crops). Also confirmed
   independently for BOTH Aria SLAM cameras -- camera-slam-left and
   camera-slam-right were checked separately and both are correct at 270,
   so this is not a per-camera-mount difference.
   CONFIRMED 2026-08-09 for Quest 3 as well, using the same method
   (dump_orientation_check.py, run against P0013_0ec32d10) -- 270 is also
   correct for Quest's SLAM cameras. Getting Quest readable at all first
   required a separate fix (see py/training/common/hot3d_timecode_compat.py
   -- Quest recordings have no TimeCode reference, unlike Aria).
   Even with the correct rotation, initial small-sample runs still showed
   low model confidence and oversized predicted boxes -- see the `size`
   decode math below and re-verify it before trusting IoU at scale.

2. `normalize_grayscale()` -- ports `normalizeGrayscaleImage()` (hg_model.cpp
   lines ~234-255): NOT simple /255 or ImageNet mean/std. It's a two-pass
   per-image contrast normalization -- rescale so stddev=0.25, then shift so
   mean=0.5, recomputing mean/std between the two steps (matches the C++
   exactly, including its "recompute even though technically redundant"
   quirk). Getting this wrong produces a *plausible-looking* run with quietly
   wrong numbers, not a crash -- there is nothing else that will tell you.

=== Output tensor shape: NOT independently verified against a running ONNX
=== session by the author of this script (no onnxruntime environment
=== available at write time). The script prints the raw output shapes for
=== the FIRST sample and asserts nothing beyond that -- read that printout
=== before trusting a full run, and adjust `run_detection()`'s indexing if
=== the 2-hand batching doesn't come out the way DetNet.py's per-hand
=== (left, right) convention would suggest.

Usage:
    python hot3d_baseline_detection_eval.py \\
        --sequence-dirs /path/to/dataset/P0001_10a27bf7 /path/to/dataset/P0002_1464cbdc ... \\
        --hot3d-repo-root /path/to/cloned/hot3d/hot3d \\
        --models-dir /path/to/hand-tracking-models \\
        --output results.csv

    # Or point at a dataset root and let the script split by the plan's
    # participant design (excludes no-GT test participants automatically):
    python hot3d_baseline_detection_eval.py \\
        --dataset-root /path/to/dataset --split train \\
        --hot3d-repo-root /path/to/cloned/hot3d/hot3d \\
        --models-dir /path/to/hand-tracking-models \\
        --output results_train.csv

See FOUR_WEEK_SUBMISSION_PLAN.md for where these numbers slot into the thesis
plan, and THESIS_STRUCTURE_v3.md Chapter 5 for the split design this script's
--split flag implements.
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "../training/detection"))
sys.path.insert(0, os.path.join(_THIS_DIR, "../../"))  # mercury_train root, for py.training.common

DETECTION_MODEL_FILENAME = "grayscale_detection_160x160.onnx"
DETECTION_INPUT_SIZE = 160

# Both confirmed from hg_model.cpp / hg_sync.cpp, not guessed:
# - hg_sync.cpp: DEBUG_GET_ONCE_FLOAT_OPTION(mercury_min_detection_confidence,
#   "MERCURY_MIN_DETECTION_CONFIDENCE", 0.3) -- default is 0.3, NOT 0.5.
# - hg_model.cpp run_hand_detection_unsafe(): output.found = hand_exists[i] >
#   min_detection_confidence.val -- strict greater-than, not >=.
MIN_DETECTION_CONFIDENCE = 0.3

# Confirmed via participant-overlap check against the actual downloaded
# manifests (Hot3DAria_download_urls.json / Hot3DQuest_download_urls.json),
# not the HOT3D repo's own docs alone -- see FOUR_WEEK_SUBMISSION_PLAN.md.
NO_GT_TEST_PARTICIPANTS = {"P0004", "P0005", "P0006", "P0008", "P0016", "P0020"}
CROSS_DEVICE_HELD_OUT_PARTICIPANTS = {"P0002", "P0003", "P0010"}  # captured on both Aria + Quest, has GT


# ---------------------------------------------------------------------------
# Preprocessing -- ported from hg_model.cpp. See module docstring for the
# verification caveats on both functions below.
# ---------------------------------------------------------------------------

def compute_blackbar_transform(in_w, in_h, out_w, out_h, orientation=0):
    """Port of hg_model.cpp's `blackbar()` affine-matrix construction.

    Returns a 2x3 float32 matrix in the same convention cv2.warpAffine
    expects: maps a point in the ORIGINAL image to its location in the
    out_w x out_h letterboxed image.

    orientation: 0/90/180/270 degrees, matching Monado's
    t_camera_orientation enum (CAMERA_ORIENTATION_0/90/180/270). Verify this
    against HOT3D's actual SLAM camera mounting before trusting results --
    see module docstring.
    """
    swapped_wh = orientation in (90, 270)
    w = in_h if swapped_wh else in_w
    h = in_w if swapped_wh else in_h

    scale_down = min(out_w / w, out_h / h)

    if swapped_wh:
        width_inside = in_h * scale_down
        height_inside = in_w * scale_down
    else:
        width_inside = in_w * scale_down
        height_inside = in_h * scale_down

    translate_x = (out_w - width_inside) / 2.0
    translate_y = (out_h - height_inside) / 2.0

    go = np.zeros((2, 3), dtype=np.float32)
    if orientation == 0:
        go[0] = [scale_down, 0.0, translate_x]
        go[1] = [0.0, scale_down, translate_y]
    elif orientation == 90:
        go[0] = [0.0, scale_down, translate_x]
        go[1] = [-scale_down, 0.0, translate_y + out_h - 1]
    elif orientation == 180:
        go[0] = [-scale_down, 0.0, translate_x + out_w - 1]
        go[1] = [0.0, -scale_down, -translate_y + out_h - 1]
    elif orientation == 270:
        go[0] = [0.0, -scale_down, -translate_x + out_w - 1]
        go[1] = [scale_down, 0.0, translate_y]
    else:
        raise ValueError(f"orientation must be 0/90/180/270, got {orientation}")
    return go


def blackbar_letterbox(image, orientation=0, out_size=DETECTION_INPUT_SIZE):
    """Letterbox `image` (grayscale, HxW uint8) into an out_size x out_size
    square, matching hg_model.cpp's `blackbar()`. Returns (letterboxed_uint8,
    go_transform) -- keep `go_transform` to map predictions back to original
    image coordinates via cv2.invertAffineTransform(go_transform).
    """
    h, w = image.shape[:2]
    go = compute_blackbar_transform(w, h, out_size, out_size, orientation)
    out = cv2.warpAffine(image, go, (out_size, out_size))
    return out, go


def normalize_grayscale(img_uint8):
    """Port of hg_model.cpp's `normalizeGrayscaleImage()`. Returns float32
    array or None if the input has zero variance (matches the C++'s
    zero-stddev bailout)."""
    data = img_uint8.astype(np.float32) / 255.0
    std = float(data.std())
    if std == 0:
        return None
    data = data * (0.25 / std)
    mean = float(data.mean())
    data = data + (0.5 - mean)
    return data


# ---------------------------------------------------------------------------
# HOT3D ground-truth access -- reuses the same data_loaders import pattern
# already verified in HOT3DVRSDetectionDataset.py, but returns RAW images +
# RAW (unmargined) boxes instead of the training-augmented heatmap format,
# since this script needs to reproduce production preprocessing exactly,
# not training-time augmentation.
# ---------------------------------------------------------------------------

class Hot3dRawFrameSource:
    def __init__(self, sequence_dirs, hot3d_repo_root, min_visibility_ratio=0.2, max_samples_per_sequence=None):
        if hot3d_repo_root not in sys.path:
            sys.path.insert(0, hot3d_repo_root)

        # Quest 3 HOT3D recordings have no TimeCode reference -- verified
        # 2026-08-09 against real Quest sequences (P0013_0ec32d10,
        # P0013_3f269bab), both raise "Timedomain TimeCode not supported"
        # from AriaDataProvider.__init__ otherwise. See
        # hot3d_timecode_compat.py's docstring for why the fallback to
        # DEVICE_TIME is safe (not just a silencer): box2d_hands.csv's
        # timestamps are themselves DEVICE_TIME-domain for Quest, confirmed
        # by exact nanosecond match against a real VRS reading, and
        # HandBox2dDataProvider.get_bbox_at_timestamp's time_domain
        # parameter is unused beyond its guard clause. Must patch AFTER
        # hot3d_repo_root is on sys.path (this import needs `data_loaders`
        # importable) and BEFORE any AriaDataProvider is constructed.
        from py.training.common.hot3d_timecode_compat import patch as _patch_quest_timecode
        _patch_quest_timecode()

        from data_loaders.PathProvider import Hot3dDataPathProvider
        from data_loaders.HandBox2dDataProvider import load_box2d_trajectory_from_csv
        from data_loaders.AriaDataProvider import AriaDataProvider
        from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
        from projectaria_tools.core.stream_id import StreamId

        self._TimeDomain = TimeDomain
        self._TimeQueryOptions = TimeQueryOptions
        self._StreamId = StreamId
        self.min_visibility_ratio = min_visibility_ratio

        self.samples = []  # (seq_name, aria_provider, box2d_provider, stream_id, ts)
        for seq_dir in sequence_dirs:
            seq_name = os.path.basename(os.path.normpath(seq_dir))
            paths = Hot3dDataPathProvider.fromRecordingFolder(seq_dir)
            if not paths.is_valid():
                print(f"WARNING: {seq_dir} missing required files, skipping")
                continue

            box2d_provider = load_box2d_trajectory_from_csv(paths.box2d_hands_filepath)
            if box2d_provider is None:
                print(f"WARNING: {seq_dir} has no box2d_hands.csv, skipping")
                continue

            aria_provider = AriaDataProvider(paths.vrs_filepath, mps_folder_path=None)

            seq_samples = []
            for stream_id in aria_provider.get_image_stream_ids():
                if str(stream_id).startswith("214-"):  # RGB stream, skip -- mono-only model
                    continue
                timestamps = aria_provider.get_sequence_timestamps(stream_id, TimeDomain.TIME_CODE)
                for ts in timestamps:
                    seq_samples.append((seq_name, aria_provider, box2d_provider, stream_id, ts))

            if max_samples_per_sequence is not None and len(seq_samples) > max_samples_per_sequence:
                # Stride evenly across the whole sequence instead of taking the
                # first N -- consecutive frames are near-duplicates, so only
                # taking the start would badly under-represent each sequence's
                # actual variety (motion, lighting, hand pose/position).
                idxs = np.linspace(0, len(seq_samples) - 1, max_samples_per_sequence, dtype=int)
                seq_samples = [seq_samples[i] for i in idxs]

            self.samples.extend(seq_samples)

    def __len__(self):
        return len(self.samples)

    def get(self, idx):
        """Returns (seq_name, stream_id, ts, image_uint8_HxW, gt_boxes) where
        gt_boxes is a list of up to 2 dicts: {slot, left, top, right, bottom,
        visibility_ratio}. slot 0=left hand, 1=right hand -- SAME UNVERIFIED
        ASSUMPTION flagged in HOT3DVRSDetectionDataset.py, carried over here
        rather than re-derived, so both scripts stay consistent."""
        seq_name, aria_provider, box2d_provider, stream_id, ts = self.samples[idx]
        image = aria_provider.get_image(ts, stream_id)
        gt_boxes = []
        if image is not None and self._StreamId(str(stream_id)) in box2d_provider.stream_ids:
            result = box2d_provider.get_bbox_at_timestamp(
                stream_id=stream_id,
                timestamp_ns=ts,
                time_query_options=self._TimeQueryOptions.CLOSEST,
                time_domain=self._TimeDomain.TIME_CODE,
            )
            if result is not None:
                for hand_index, hand_box in result.box2d_collection.box2ds.items():
                    if hand_box.box2d is None or hand_box.visibility_ratio is None:
                        continue
                    if hand_box.visibility_ratio < self.min_visibility_ratio:
                        continue
                    b2d = hand_box.box2d
                    gt_boxes.append({
                        "slot": 0 if hand_index == 0 else 1,
                        "left": b2d.left, "top": b2d.top,
                        "right": b2d.right, "bottom": b2d.bottom,
                        "visibility_ratio": hand_box.visibility_ratio,
                    })
        return seq_name, str(stream_id), ts, image, gt_boxes


# ---------------------------------------------------------------------------
# Split helpers -- matches FOUR_WEEK_SUBMISSION_PLAN.md's participant design.
# ---------------------------------------------------------------------------

def participant_id_of(seq_dir):
    return os.path.basename(os.path.normpath(seq_dir)).split("_")[0]


def filter_sequence_dirs(all_seq_dirs, split):
    """split in {'train', 'cross_device_test', 'all_labeled'}. Always drops
    the no-GT official HOT3D test participants -- they have no annotations
    to evaluate against regardless of split."""
    usable = [d for d in all_seq_dirs if participant_id_of(d) not in NO_GT_TEST_PARTICIPANTS]
    if split == "all_labeled":
        return usable
    if split == "cross_device_test":
        return [d for d in usable if participant_id_of(d) in CROSS_DEVICE_HELD_OUT_PARTICIPANTS]
    if split == "train":
        return [d for d in usable if participant_id_of(d) not in CROSS_DEVICE_HELD_OUT_PARTICIPANTS]
    raise ValueError(f"unknown split {split!r}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def box_iou(box_a, box_b):
    """Both boxes as (left, top, right, bottom)."""
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    inter_x0, inter_y0 = max(ax0, bx0), max(ay0, by0)
    inter_x1, inter_y1 = min(ax1, bx1), min(ay1, by1)
    inter_w, inter_h = max(0.0, inter_x1 - inter_x0), max(0.0, inter_y1 - inter_y0)
    inter = inter_w * inter_h
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Detection inference
# ---------------------------------------------------------------------------

def load_detection_session(models_dir):
    import onnxruntime as ort
    path = os.path.join(models_dir, DETECTION_MODEL_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- confirm --models-dir points at the folder containing "
            f"the real shipped {DETECTION_MODEL_FILENAME} (not one of the other .onnx "
            f"variants sitting alongside it, e.g. plain grayscale_detection.onnx)."
        )
    return ort.InferenceSession(path)


_debug_print_count = [0]  # caps the side-by-side pred-vs-GT debug prints in main()


def run_detection(session, letterboxed_float, _printed_shapes=[False]):
    inp = letterboxed_float.reshape(1, 1, DETECTION_INPUT_SIZE, DETECTION_INPUT_SIZE).astype(np.float32)
    outputs = session.run(["hand_exists", "cx", "cy", "size"], {"inputImg": inp})
    if not _printed_shapes[0]:
        print("First-sample raw output shapes (VERIFY these match a 2-hand "
              "[left, right] convention before trusting the rest of the run):")
        for name, val in zip(["hand_exists", "cx", "cy", "size"], outputs):
            print(f"  {name}: shape={np.asarray(val).shape} value={val}")
        _printed_shapes[0] = True
    return dict(zip(["hand_exists", "cx", "cy", "size"], outputs))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequence-dirs", nargs="+", default=None,
                         help="explicit list of sequence folders to evaluate")
    parser.add_argument("--dataset-root", default=None,
                         help="alternative to --sequence-dirs: evaluate every P00xx_* folder under this root")
    parser.add_argument("--split", choices=["train", "cross_device_test", "all_labeled"], default="all_labeled",
                         help="only used with --dataset-root; see FOUR_WEEK_SUBMISSION_PLAN.md for the split design")
    parser.add_argument("--hot3d-repo-root", required=True,
                         help="path to the cloned facebookresearch/hot3d repo's hot3d/hot3d subfolder")
    parser.add_argument("--models-dir", required=True,
                         help="folder containing grayscale_detection_160x160.onnx")
    parser.add_argument("--min-visibility-ratio", type=float, default=0.2)
    parser.add_argument("--orientation", type=int, choices=[0, 90, 180, 270], default=270,
                         help="camera mount rotation for blackbar letterboxing -- 270 confirmed for both Aria "
                              "(2026-08-08) and Quest (2026-08-09) SLAM cameras by visually inspecting the "
                              "letterboxed crop at all 4 rotations against real frames. See module docstring.")
    parser.add_argument("--device", required=True, choices=["Aria", "Quest"],
                         help="which physical device these --sequence-dirs/--dataset-root sequences were captured "
                              "on. Required (not inferred) so results are never silently mislabeled -- an earlier "
                              "version of this script guessed device from --hot3d-repo-root's path string via a "
                              "broken `or True` condition that always evaluated to 'Aria' regardless of the actual "
                              "data; this argument replaces that.")
    parser.add_argument("--output", required=True, help="CSV path for per-sample results")
    parser.add_argument("--limit", type=int, default=None, help="cap total samples evaluated, for a quick smoke test")
    parser.add_argument("--max-samples-per-sequence", type=int, default=None,
                         help="cap samples taken from EACH sequence (strided evenly across it), so a run spans many "
                              "different sequences/participants instead of exhausting --limit on just the first one")
    args = parser.parse_args()

    if args.sequence_dirs:
        seq_dirs = args.sequence_dirs
    elif args.dataset_root:
        all_dirs = sorted(
            os.path.join(args.dataset_root, d) for d in os.listdir(args.dataset_root)
            if os.path.isdir(os.path.join(args.dataset_root, d)) and d.startswith("P0")
        )
        seq_dirs = filter_sequence_dirs(all_dirs, args.split)
        print(f"Split '{args.split}': {len(seq_dirs)} / {len(all_dirs)} sequences selected "
              f"(excludes no-GT test participants {sorted(NO_GT_TEST_PARTICIPANTS)})")
    else:
        parser.error("must pass either --sequence-dirs or --dataset-root")

    source = Hot3dRawFrameSource(seq_dirs, args.hot3d_repo_root, args.min_visibility_ratio,
                                  max_samples_per_sequence=args.max_samples_per_sequence)
    print(f"{len(source)} (frame, camera-stream) samples across {len(seq_dirs)} sequences")

    session = load_detection_session(args.models_dir)

    rows = []
    n = len(source) if args.limit is None else min(args.limit, len(source))
    for idx in range(n):
        seq_name, stream_id, ts, image, gt_boxes = source.get(idx)
        if image is None:
            continue

        letterboxed, go = blackbar_letterbox(image, orientation=args.orientation)
        normalized = normalize_grayscale(letterboxed)
        if normalized is None:
            continue

        pred = run_detection(session, normalized)
        inv = cv2.invertAffineTransform(go)
        scale_recovered = 1.0 / np.hypot(go[0, 0], go[1, 0])  # linear part only, for `size`

        # NOTE: indexing below assumes pred["cx"] etc. are shape (1, 2) for
        # [left, right] -- confirm against the printed shapes on your first
        # run (see run_detection()) and adjust if the model batches hands
        # differently.
        for slot in (0, 1):
            exists = float(np.asarray(pred["hand_exists"]).reshape(-1)[slot])
            cx_raw = float(np.asarray(pred["cx"]).reshape(-1)[slot])
            cy_raw = float(np.asarray(pred["cy"]).reshape(-1)[slot])
            size_raw = float(np.asarray(pred["size"]).reshape(-1)[slot])

            # Raw cx/cy/size are NOT pixel coordinates -- confirmed via
            # hg_model.cpp run_hand_detection_unsafe():
            #   _pt.x = math_map_ranges(cx[i], -1, 1, 0, kDetectionInputSize)
            #   size *= kDetectionInputSize * 2.0f; size *= ||go_back row0||
            # i.e. cx/cy are normalized to [-1, 1] over the 160x160 letterboxed
            # frame, and size is a normalized fraction scaled by 2x the input
            # size before being converted to original-image pixels via the
            # inverse letterbox scale. Earlier version of this script treated
            # all three as already being 160-space pixels -- that produced
            # 0.0000 mean IoU, silently wrong, not a crash.
            cx_160 = (cx_raw + 1.0) / 2.0 * DETECTION_INPUT_SIZE
            cy_160 = (cy_raw + 1.0) / 2.0 * DETECTION_INPUT_SIZE

            orig_pt = inv @ np.array([cx_160, cy_160, 1.0])
            pred_cx, pred_cy = float(orig_pt[0]), float(orig_pt[1])
            pred_size = size_raw * DETECTION_INPUT_SIZE * 2.0 * scale_recovered
            pred_box = (pred_cx - pred_size / 2, pred_cy - pred_size / 2,
                        pred_cx + pred_size / 2, pred_cy + pred_size / 2)

            gt = next((g for g in gt_boxes if g["slot"] == slot), None)
            row = {
                "sequence": seq_name, "stream_id": stream_id, "timestamp_ns": ts, "hand_slot": slot,
                "device": args.device,
                "pred_exists": exists, "pred_cx": pred_cx, "pred_cy": pred_cy, "pred_size": pred_size,
                "gt_exists": gt is not None,
                "visibility_ratio": gt["visibility_ratio"] if gt else None,
                "iou": None, "center_error_px": None,
            }
            if gt is not None:
                gt_box = (gt["left"], gt["top"], gt["right"], gt["bottom"])
                row["iou"] = box_iou(pred_box, gt_box)
                gt_cx, gt_cy = (gt["left"] + gt["right"]) / 2, (gt["top"] + gt["bottom"]) / 2
                row["center_error_px"] = float(np.hypot(pred_cx - gt_cx, pred_cy - gt_cy))

                # DEBUG: side-by-side comparison for the first few GT-present
                # samples, to diagnose the orientation/scale issue numerically
                # instead of guessing. Remove once IoU is confirmed sane.
                if _debug_print_count[0] < 5:
                    _debug_print_count[0] += 1
                    print(f"DEBUG sample: image shape (h,w)={image.shape}, orientation={args.orientation}")
                    print(f"  PRED center=({pred_cx:.1f}, {pred_cy:.1f}) size={pred_size:.1f}")
                    print(f"  GT   center=({gt_cx:.1f}, {gt_cy:.1f}) box=(l={gt['left']:.1f}, t={gt['top']:.1f}, "
                          f"r={gt['right']:.1f}, b={gt['bottom']:.1f})")
                    print(f"  IoU={row['iou']:.4f}")
            rows.append(row)

        if idx % 500 == 0:
            print(f"  {idx}/{n}")

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")

    ious = [r["iou"] for r in rows if r["iou"] is not None]
    if ious:
        print(f"Mean IoU (GT-present hands only): {np.mean(ious):.4f}  (n={len(ious)})")
    exist_correct = [r["pred_exists"] > MIN_DETECTION_CONFIDENCE for r in rows]
    gt_present = [r["gt_exists"] for r in rows]
    if rows:
        acc = np.mean([p == g for p, g in zip(exist_correct, gt_present)])
        print(f"Existence accuracy: {acc:.4f}  (n={len(rows)})")


if __name__ == "__main__":
    main()
