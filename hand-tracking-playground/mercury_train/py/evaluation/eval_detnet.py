"""
eval_detnet.py -- score a set of DetNet weights against a HOT3D split.

Replaces hot3d_baseline_detection_eval.py, which could only run the shipped
ONNX baseline and decoded its `size` output twice as large as the training
target defines it (see preprocess_baseline.SIZE_DECODE). Its numbers should
not be carried forward.

WHAT IS MEASURED
----------------
Per (frame, camera stream, hand slot):
  - IoU between the predicted box and HOT3D's own occlusion-aware ground
    truth from box2d_hands.csv, for hands the ground truth says are present
  - centre error in pixels
  - whether the existence head agreed with the ground truth, at Monado's own
    production threshold of 0.3 (strict >)
aggregated overall and stratified by HOT3D's `visibility_ratio`, because a
detector's behaviour on a hand that is 30 % visible is a different question
from its behaviour on one in full view, and a single mean hides that.

ONE CODE PATH FOR BOTH MODELS
-----------------------------
--weights takes either `monado` (the shipped grayscale_detection_160x160.onnx,
zero-shot) or a path to a fine-tuned checkpoint. Both are preprocessed,
decoded and scored by the identical code below, so a difference between the
two numbers is a difference between the two models. The previous script
could only run the ONNX, which meant the fine-tuned model would have had to
be scored by some second script -- exactly the arrangement that let the two
conventions drift apart in the first place.

PREPROCESSING
-------------
Imported from preprocess_baseline, not restated here. Read that module's
docstring for what the convention is and why the training code rather than
the C++ runtime defines it.

Usage:
    python py/evaluation/eval_detnet.py --weights monado --split test_aria \
        --out results/detnet_baseline_aria.json

    python py/evaluation/eval_detnet.py \
        --weights py/training/detection/checkpoints/checkpoint_best.pth \
        --split test_aria --out results/detnet_finetuned_aria.json
"""
import argparse
import csv
import json
import os
import random
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MERCURY_TRAIN_ROOT = os.path.join(_THIS_DIR, "..", "..")
for _p in (_MERCURY_TRAIN_ROOT, os.path.join(_THIS_DIR, "..", "training", "detection")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import preprocess_baseline as pp
import py.training.common.hot3d_split as hot3d_split

DETECTION_MODEL_FILENAME = "grayscale_detection_160x160.onnx"
VISIBILITY_BANDS = [(0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

class Hot3dRawFrameSource:
    """Raw frames plus unmargined ground-truth boxes.

    Deliberately NOT the training dataset. HOT3DVRSDetectionDataset applies a
    15 % box margin and a randomised affine, both of which are training-time
    choices; evaluation needs the frame as the sensor produced it and the box
    as Meta annotated it, so the only transformation between them and the
    network is the convention itself.
    """

    def __init__(self, sequence_dirs, hot3d_repo_root, min_visibility_ratio=0.2,
                 max_samples_per_sequence=None):
        if hot3d_repo_root not in sys.path:
            sys.path.insert(0, hot3d_repo_root)

        # Quest 3 recordings carry no TimeCode reference and raise from
        # AriaDataProvider.__init__ without this. See
        # py/training/common/hot3d_timecode_compat.py for why DEVICE_TIME is
        # the correct fallback rather than a way of silencing the error.
        from py.training.common.hot3d_timecode_compat import patch as _patch
        _patch()

        from data_loaders.PathProvider import Hot3dDataPathProvider
        from data_loaders.HandBox2dDataProvider import load_box2d_trajectory_from_csv
        from data_loaders.AriaDataProvider import AriaDataProvider
        from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
        from projectaria_tools.core.stream_id import StreamId

        self._TimeDomain = TimeDomain
        self._TimeQueryOptions = TimeQueryOptions
        self._StreamId = StreamId
        self.min_visibility_ratio = min_visibility_ratio
        self.samples = []

        for seq_dir in sequence_dirs:
            seq_name = os.path.basename(os.path.normpath(seq_dir))
            paths = Hot3dDataPathProvider.fromRecordingFolder(seq_dir)
            if not paths.is_valid():
                print(f"WARNING: {seq_dir} missing required files, skipping")
                continue
            box2d = load_box2d_trajectory_from_csv(paths.box2d_hands_filepath)
            if box2d is None:
                print(f"WARNING: {seq_dir} has no box2d_hands.csv, skipping")
                continue

            provider = AriaDataProvider(paths.vrs_filepath, mps_folder_path=None)
            headset = hot3d_split.headset_of(seq_dir) or "Aria"

            seq_samples = []
            for stream_id in provider.get_image_stream_ids():
                if str(stream_id).startswith("214-"):    # RGB, mono-only model
                    continue
                for ts in provider.get_sequence_timestamps(stream_id, TimeDomain.TIME_CODE):
                    seq_samples.append((seq_name, headset, provider, box2d, stream_id, ts))

            if max_samples_per_sequence and len(seq_samples) > max_samples_per_sequence:
                # Strided, not the first N: consecutive frames are near
                # duplicates, so taking a prefix would sample one moment of
                # the sequence rather than the sequence.
                idx = np.linspace(0, len(seq_samples) - 1, max_samples_per_sequence, dtype=int)
                seq_samples = [seq_samples[i] for i in idx]
            self.samples.extend(seq_samples)

    def __len__(self):
        return len(self.samples)

    def get(self, idx):
        seq_name, headset, provider, box2d, stream_id, ts = self.samples[idx]
        image = provider.get_image(ts, stream_id)
        boxes = []
        if image is not None and self._StreamId(str(stream_id)) in box2d.stream_ids:
            result = box2d.get_bbox_at_timestamp(
                stream_id=stream_id, timestamp_ns=ts,
                time_query_options=self._TimeQueryOptions.CLOSEST,
                time_domain=self._TimeDomain.TIME_CODE)
            if result is not None:
                for hand_index, hand_box in result.box2d_collection.box2ds.items():
                    if hand_box.box2d is None or hand_box.visibility_ratio is None:
                        continue
                    if hand_box.visibility_ratio < self.min_visibility_ratio:
                        continue
                    b = hand_box.box2d
                    boxes.append({"slot": 0 if hand_index == 0 else 1,
                                  "box": (b.left, b.top, b.right, b.bottom),
                                  "visibility_ratio": float(hand_box.visibility_ratio)})
        return seq_name, headset, str(stream_id), ts, image, boxes


# ---------------------------------------------------------------------------
# Models. Both expose the same call signature so scoring never branches.
# ---------------------------------------------------------------------------

class OnnxDetector:
    name = "monado (zero-shot, no fine-tuning)"

    def __init__(self, models_dir):
        import onnxruntime as ort
        path = os.path.join(models_dir, DETECTION_MODEL_FILENAME)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. --models-dir must contain the shipped "
                f"{DETECTION_MODEL_FILENAME}, not one of the other .onnx "
                f"variants beside it (e.g. plain grayscale_detection.onnx).")
        self.sess = ort.InferenceSession(path)
        self.out_names = [o.name for o in self.sess.get_outputs()]
        self.in_name = self.sess.get_inputs()[0].name

        # Asserted, not printed and eyeballed. The old script printed the
        # shapes for the first sample and asked the reader to check them.
        in_shape = list(self.sess.get_inputs()[0].shape)
        assert in_shape == [1, 1, pp.DETECTION_INPUT_SIZE, pp.DETECTION_INPUT_SIZE], \
            f"unexpected input shape {in_shape}"
        assert self.out_names[:1] == ["hand_exists"] or "hand_exists" in self.out_names, \
            f"unexpected outputs {self.out_names}"
        for o in self.sess.get_outputs():
            assert list(o.shape) == [1, 2], \
                (f"output {o.name} has shape {o.shape}, expected [1, 2] for the "
                 f"two hand slots. Every decode below indexes [left, right].")

    def __call__(self, net_input):
        outs = self.sess.run(["hand_exists", "cx", "cy", "size"], {self.in_name: net_input})
        return dict(zip(["hand_exists", "cx", "cy", "size"], outs))


class TorchDetector:
    def __init__(self, checkpoint_path, device=None):
        import torch
        import DetNet
        from load_weights import load_detnet_weights

        self.torch = torch
        model = DetNet.DetNet()
        # Gives the bias-free convs real bias parameters, matching what
        # trainer_detection.py does before it checkpoints anything.
        load_detnet_weights(model)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state)
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        epoch = ckpt.get("epoch")
        self.name = checkpoint_path + (f" (epoch {epoch})" if epoch is not None else "")

    def __call__(self, net_input):
        with self.torch.no_grad():
            t = self.torch.from_numpy(net_input).to(self.device)
            exists, cx, cy, size = self.model(t)
        return {"hand_exists": exists.cpu().numpy(), "cx": cx.cpu().numpy(),
                "cy": cy.cpu().numpy(), "size": size.cpu().numpy()}


def build_model(weights, models_dir):
    if weights == "monado":
        m = OnnxDetector(models_dir)
    else:
        m = TorchDetector(weights)
    return m, m.name


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def summarise(rows):
    def mean(xs):
        return float(np.mean(xs)) if len(xs) else None

    matched = [r for r in rows if r["iou"] is not None]
    ious = [r["iou"] for r in matched]
    result = {
        "n_rows": len(rows),
        "n_gt_present": len(matched),
        "mean_iou": mean(ious),
        "median_iou": float(np.median(ious)) if ious else None,
        "iou_at_0.5": mean([i >= 0.5 for i in ious]),
        "iou_at_0.75": mean([i >= 0.75 for i in ious]),
        "mean_centre_error_px": mean([r["centre_error_px"] for r in matched]),
        "existence_accuracy": mean([
            (r["pred_exists"] > pp.MIN_DETECTION_CONFIDENCE) == r["gt_exists"] for r in rows]),
        "existence_recall": mean([
            r["pred_exists"] > pp.MIN_DETECTION_CONFIDENCE for r in rows if r["gt_exists"]]),
        "existence_precision": mean([
            r["gt_exists"] for r in rows if r["pred_exists"] > pp.MIN_DETECTION_CONFIDENCE]),
        "by_visibility": {},
        "by_device": {},
    }
    for lo, hi in VISIBILITY_BANDS:
        band = [r["iou"] for r in matched if lo <= r["visibility_ratio"] < hi]
        result["by_visibility"][f"[{lo:.1f},{hi:.1f})"] = {
            "n": len(band), "mean_iou": mean(band)}
    for dev in sorted({r["device"] for r in rows}):
        d = [r for r in matched if r["device"] == dev]
        result["by_device"][dev] = {"n": len(d), "mean_iou": mean([r["iou"] for r in d])}
    return result


def print_report(split, source, orientation, r):
    print("\n" + "=" * 66)
    print(f"  split            {split}")
    print(f"  weights          {source}")
    print(f"  orientation      {orientation} (camera mount rotation)")
    print(f"  size decode      size * {pp.DETECTION_INPUT_SIZE} "
          f"* {pp.SIZE_DECODE_FACTOR} (training-target convention)")
    print("-" * 66)
    print(f"  hand slots scored               {r['n_rows']}")
    print(f"  of those, ground truth present  {r['n_gt_present']}")
    print("-" * 66)
    print(f"  MEAN IoU              {r['mean_iou']:8.4f}" if r["mean_iou"] is not None else "  MEAN IoU   --")
    print(f"  median IoU            {r['median_iou']:8.4f}" if r["median_iou"] is not None else "")
    print(f"  IoU >= 0.50           {r['iou_at_0.5'] * 100:8.2f} %" if r["iou_at_0.5"] is not None else "")
    print(f"  IoU >= 0.75           {r['iou_at_0.75'] * 100:8.2f} %" if r["iou_at_0.75"] is not None else "")
    print(f"  mean centre error     {r['mean_centre_error_px']:8.2f} px" if r["mean_centre_error_px"] is not None else "")
    print("-" * 66)
    print(f"  existence accuracy    {r['existence_accuracy'] * 100:8.2f} %")
    if r["existence_recall"] is not None:
        print(f"  existence recall      {r['existence_recall'] * 100:8.2f} %")
    if r["existence_precision"] is not None:
        print(f"  existence precision   {r['existence_precision'] * 100:8.2f} %")
    print("-" * 66)
    print("  mean IoU by ground-truth visibility_ratio:")
    for band, v in r["by_visibility"].items():
        s = f"{v['mean_iou']:.4f}" if v["mean_iou"] is not None else "  --  "
        print(f"    {band:<12} n={v['n']:<7} {s}")
    if len(r["by_device"]) > 1:
        print("  mean IoU by device:")
        for dev, v in r["by_device"].items():
            s = f"{v['mean_iou']:.4f}" if v["mean_iou"] is not None else "  --  "
            print(f"    {dev:<12} n={v['n']:<7} {s}")
    print("=" * 66 + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True,
                        help="'monado' for the shipped zero-shot ONNX, or a path to a .pth")
    parser.add_argument("--split", required=True,
                        choices=["train", "val", "test_aria", "test_quest",
                                 "cross_device_test", "all_labeled"])
    parser.add_argument("--models-dir", default=None,
                        help="folder holding grayscale_detection_160x160.onnx "
                             "(default: <repo>/hand-tracking-models)")
    parser.add_argument("--dataset-root", default=None,
                        help="default: local_config_cluster.hot3d_dataset_root")
    parser.add_argument("--hot3d-repo-root", default=None,
                        help="default: local_config_cluster.hot3d_repo_root")
    parser.add_argument("--min-visibility-ratio", type=float, default=0.2)
    parser.add_argument("--orientation", type=int, choices=[0, 90, 180, 270], default=None,
                        help="override the per-device camera mount rotation; "
                             "normally leave unset so preprocess_baseline decides")
    parser.add_argument("--max-samples-per-sequence", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="cap total frames, for a smoke test")
    parser.add_argument("--seed", type=int, default=0,
                        help="recorded in the output even though this evaluation is "
                             "deterministic, so a future stochastic option cannot "
                             "quietly produce unreproducible numbers")
    parser.add_argument("--out", default=None, help="write metrics as JSON here")
    parser.add_argument("--out-csv", default=None, help="write per-slot rows as CSV here")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    pp.self_check(verbose=False)

    import local_config_cluster as lc
    dataset_root = args.dataset_root or lc.hot3d_dataset_root
    repo_root = args.hot3d_repo_root or lc.hot3d_repo_root
    models_dir = args.models_dir or os.path.join(
        _MERCURY_TRAIN_ROOT, "..", "..", "hand-tracking-models")

    seq_dirs = hot3d_split.filter_sequence_dirs(
        sorted(os.path.join(dataset_root, d) for d in os.listdir(dataset_root)
               if os.path.isdir(os.path.join(dataset_root, d)) and d.startswith("P0")),
        args.split)
    if not seq_dirs:
        raise SystemExit(f"No sequences for split {args.split!r} under {dataset_root}")
    print(f"[eval_detnet] {args.split}: {len(seq_dirs)} sequences")

    source_data = Hot3dRawFrameSource(seq_dirs, repo_root, args.min_visibility_ratio,
                                      args.max_samples_per_sequence)
    print(f"[eval_detnet] {len(source_data)} (frame, camera-stream) samples")

    model, source = build_model(args.weights, models_dir)

    rows = []
    n = len(source_data) if args.limit is None else min(args.limit, len(source_data))
    orientation_used = set()

    for idx in range(n):
        seq_name, headset, stream_id, ts, image, gt_boxes = source_data.get(idx)
        if image is None:
            continue

        orientation = args.orientation if args.orientation is not None \
            else pp.DEVICE_ORIENTATION.get(headset, 270)
        orientation_used.add(orientation)

        net_input, upright, go = pp.detnet_preprocess(image, orientation)
        if net_input is None:      # zero-variance frame; the C++ bails too
            continue
        pred = model(net_input)

        h, w = image.shape[:2]
        for gt in gt_boxes:
            x0, y0, x1, y1 = gt["box"]
            corners = pp.rotate_points_upright([[x0, y0], [x1, y1]], orientation, w, h)
            gt["box_upright"] = (float(min(corners[:, 0])), float(min(corners[:, 1])),
                                 float(max(corners[:, 0])), float(max(corners[:, 1])))

        for slot in (0, 1):
            conf, box = pp.decode_detection(pred["hand_exists"], pred["cx"], pred["cy"],
                                            pred["size"], go, slot)
            gt = next((g for g in gt_boxes if g["slot"] == slot), None)
            row = {"sequence": seq_name, "device": headset, "stream_id": stream_id,
                   "timestamp_ns": ts, "hand_slot": slot, "orientation": orientation,
                   "pred_exists": conf,
                   "pred_left": box[0], "pred_top": box[1],
                   "pred_right": box[2], "pred_bottom": box[3],
                   "gt_exists": gt is not None,
                   "visibility_ratio": gt["visibility_ratio"] if gt else None,
                   "iou": None, "centre_error_px": None}
            if gt is not None:
                g = gt["box_upright"]
                row["iou"] = pp.box_iou(box, g)
                row["centre_error_px"] = float(np.hypot(
                    (box[0] + box[2]) / 2 - (g[0] + g[2]) / 2,
                    (box[1] + box[3]) / 2 - (g[1] + g[3]) / 2))
            rows.append(row)

        if idx % 500 == 0:
            print(f"  {idx}/{n}", flush=True)

    if not rows:
        raise SystemExit("No usable samples in this split.")

    result = summarise(rows)
    print_report(args.split, source, sorted(orientation_used), result)

    if args.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[eval_detnet] wrote {args.out_csv}")

    if args.out:
        payload = dict(result)
        payload.update({
            "model": "DetNet", "split": args.split, "weights": args.weights,
            "weights_source": source, "n_sequences": len(seq_dirs),
            "orientation": sorted(orientation_used), "seed": args.seed,
            "min_visibility_ratio": args.min_visibility_ratio,
            "size_decode_factor": pp.SIZE_DECODE_FACTOR,
        })
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[eval_detnet] wrote {args.out}")


if __name__ == "__main__":
    main()
