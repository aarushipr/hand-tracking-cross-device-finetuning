"""
eval_sam3_boxes.py -- how good would SAM 3 be at annotating hand bounding boxes?

Prompts SAM 3 with the text "hand" on HOT3D frames, derives an axis-aligned box
from each returned mask, and scores those boxes against HOT3D's motion-capture
ground truth using the same IoU definition eval_detnet.py uses. The question is
not whether SAM is a good hand detector, it is whether a future custom corpus
could be annotated automatically instead of by hand.

WHY THIS DOES NOT REUSE THE MERCURY PREPROCESSING PATH
------------------------------------------------------
py/evaluation/preprocess_baseline.py exists to reproduce the input format the
shipped DetNet and KeyNet weights were trained on -- letterbox to 160x160, and a
contrast normalisation to stddev 0.25 / mean 0.5. None of that applies to SAM,
which has its own internal resizing and normalisation. Feeding SAM a
Mercury-normalised 160x160 crop would measure format mismatch rather than SAM.

Exactly two preprocessing steps are applied here:

  1. rotate_upright, using the same per-device orientation the rest of the
     pipeline uses. SAM was trained on upright photographs, and this also puts
     the image in the same coordinate frame as the ground-truth boxes.
  2. grayscale replicated to three channels, because SAM's encoder expects RGB.

Full sensor resolution is passed through untouched. Fisheye distortion is left
in deliberately: that is what an annotator would actually be labelling.

--clahe adds contrast-limited histogram equalisation as a second condition,
since a number of the Quest 3 SLAM frames are very dark.

THE CAVEAT THAT MATTERS MOST WHEN READING THE NUMBERS
-----------------------------------------------------
HOT3D's ground truth comes from motion capture and draws the hand/forearm
boundary one specific way. SAM may reasonably segment the whole visible arm. If
it does, IoU will be poor for a reason that has nothing to do with segmentation
quality. ALWAYS look at the overlays written by --save-overlays before trusting
any aggregate number from this script.

Sampling is random across the split with a fixed seed, never a prefix.
eval_detnet.py's --limit truncates its sample list instead, which on this split
yields six sequences from a single device.

Usage:
  python py/evaluation/eval_sam3_boxes.py --split test_mixed --limit 300 \
      --sam-weights /path/to/sam3.pt --save-overlays results/sam3_overlays \
      --out results/sam3_boxes.json
"""

import argparse
import json
import os
import random
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_MERCURY_TRAIN_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
for _p in (_MERCURY_TRAIN_ROOT, os.path.join(_MERCURY_TRAIN_ROOT, "py", "training", "detection")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from py.evaluation import preprocess_baseline as _pp
from py.training.detection.HOT3DVRSDetectionDataset import HOT3DVRSDetectionDataset
from py.training.common.hot3d_split import list_sequence_dirs
import py.training.detection.local_config as local_config


# ---------------------------------------------------------------------------
# Ground truth, reproduced exactly as HOT3DVRSDetectionDataset builds it
# ---------------------------------------------------------------------------

def raw_sample(ds, idx):
    """Return (upright image, [gt boxes], headset) for one dataset index.

    Deliberately duplicates the body of HOT3DVRSDetectionDataset.__getitem__ up
    to but NOT including augment_image, because this script needs the full-
    resolution frame and boxes in image coordinates rather than the network's
    160x160 heatmap targets. Reaching into the dataset's private index arrays is
    acceptable for an analysis script; the alternative is a second copy of the
    VRS indexing logic that could drift out of sync with the real one.
    """
    seq_dir = ds.sequence_dirs[int(ds._seq_idx[idx])]
    stream_str = str(ds._stream_str[idx])
    ts = int(ds._ts[idx])
    headset = str(ds._headset[idx])

    bundle = ds._providers_for(seq_dir)
    stream_id = bundle.stream_id_by_str.get(stream_str)
    if stream_id is None:
        return None

    image = bundle.aria_provider.get_image(ts, stream_id)
    if image is None:
        return None

    orientation = (ds.orientation_override if ds.orientation_override is not None
                   else _pp.DEVICE_ORIENTATION.get(headset, 270))
    raw_h, raw_w = image.shape[:2]
    image = _pp.rotate_upright(image, orientation)

    gt = []
    box2d_provider = bundle.box2d_provider
    if ds._StreamId(str(stream_id)) in box2d_provider.stream_ids:
        result = box2d_provider.get_bbox_at_timestamp(
            stream_id=stream_id, timestamp_ns=ts,
            time_query_options=ds._TimeQueryOptions.CLOSEST,
            time_domain=ds._TimeDomain.TIME_CODE)
        if result is not None:
            for hand_index, hand_box in result.box2d_collection.box2ds.items():
                if hand_box.box2d is None or hand_box.visibility_ratio is None:
                    continue
                if hand_box.visibility_ratio < ds.min_visibility_ratio:
                    continue
                b2d = hand_box.box2d
                corners = _pp.rotate_points_upright(
                    [[b2d.left, b2d.top], [b2d.right, b2d.bottom]],
                    orientation, raw_w, raw_h)
                l, t = float(corners[:, 0].min()), float(corners[:, 1].min())
                r, b = float(corners[:, 0].max()), float(corners[:, 1].max())
                w, h = r - l, b - t
                gt.append((l - w * ds.margin, t - h * ds.margin,
                           r + w * ds.margin, b + h * ds.margin))
    return image, gt, headset


# ---------------------------------------------------------------------------
# SAM
# ---------------------------------------------------------------------------

def masks_from_result(res, verbose=False):
    """Pull binary masks out of whatever Ultralytics returned.

    Written defensively on purpose. The exact shape of a SAM3SemanticPredictor
    result is not something this script should assume, so the first call runs
    with verbose=True and prints what it actually got.
    """
    items = res if isinstance(res, (list, tuple)) else [res]
    out = []
    for r in items:
        if verbose:
            print(f"[sam] result type={type(r)} attrs={[a for a in dir(r) if not a.startswith('_')][:25]}")
        m = getattr(r, "masks", None)
        if m is None:
            if verbose:
                print("[sam] no .masks attribute on this result")
            continue
        data = getattr(m, "data", m)
        try:
            arr = data.cpu().numpy()
        except AttributeError:
            arr = np.asarray(data)
        if verbose:
            print(f"[sam] mask array shape={arr.shape} dtype={arr.dtype}")
        if arr.ndim == 2:
            arr = arr[None]
        for i in range(arr.shape[0]):
            out.append(arr[i] > 0.5)
    return out


def mask_to_box(mask, target_hw):
    """Axis-aligned extent of a binary mask, resized to the image if needed."""
    if mask.shape[:2] != target_hw:
        mask = cv2.resize(mask.astype(np.uint8), (target_hw[1], target_hw[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match_greedy(gt_boxes, pred_boxes):
    """Greedy IoU matching. Returns (pairs, unmatched_gt, unmatched_pred)."""
    cand = sorted(((iou(g, p), gi, pi)
                   for gi, g in enumerate(gt_boxes)
                   for pi, p in enumerate(pred_boxes)),
                  key=lambda t: -t[0])
    used_g, used_p, pairs = set(), set(), []
    for v, gi, pi in cand:
        if v <= 0.0 or gi in used_g or pi in used_p:
            continue
        pairs.append((gi, pi, v))
        used_g.add(gi)
        used_p.add(pi)
    return (pairs,
            [i for i in range(len(gt_boxes)) if i not in used_g],
            [i for i in range(len(pred_boxes)) if i not in used_p])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--split", default="test_mixed")
    ap.add_argument("--limit", type=int, default=300, help="frames to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sam-weights", default="sam3.pt")
    ap.add_argument("--prompt", default="hand")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou-thresh", type=float, default=0.5,
                    help="IoU at which a match counts as a correct annotation")
    ap.add_argument("--clahe", action="store_true",
                    help="apply CLAHE before SAM, as a second condition")
    ap.add_argument("--frame-stride", type=int, default=5)
    ap.add_argument("--save-overlays", default=None)
    ap.add_argument("--max-overlays", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seq_dirs = list_sequence_dirs(local_config.hot3d_dataset_root, args.split)
    if not seq_dirs:
        raise SystemExit(f"no sequences for split {args.split}")
    ds = HOT3DVRSDetectionDataset(
        sequence_dirs=seq_dirs,
        hot3d_repo_root=local_config.hot3d_repo_root,
        frame_stride=args.frame_stride,
        index_cache_dir=getattr(local_config, "hot3d_index_cache_dir", None),
        augment=False)
    print(f"[sam] {len(ds)} candidate samples across {len(seq_dirs)} sequences")

    rng = random.Random(args.seed)
    idxs = rng.sample(range(len(ds)), min(args.limit, len(ds)))

    from ultralytics.models.sam import SAM3SemanticPredictor
    predictor = SAM3SemanticPredictor(overrides={
        "model": args.sam_weights, "task": "segment", "mode": "predict",
        "conf": args.conf, "save": False, "verbose": False})

    if args.save_overlays:
        os.makedirs(args.save_overlays, exist_ok=True)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if args.clahe else None

    per_device = {}
    n_gt = n_pred = n_hit = 0
    ious, saved = [], 0

    for k, idx in enumerate(idxs):
        s = raw_sample(ds, idx)
        if s is None:
            continue
        img, gt_boxes, headset = s
        if not gt_boxes:
            continue

        sam_in = clahe.apply(img) if clahe is not None else img
        rgb = cv2.cvtColor(sam_in, cv2.COLOR_GRAY2RGB)

        predictor.set_image(rgb)
        res = predictor(text=[args.prompt])
        masks = masks_from_result(res, verbose=(k == 0))

        pred_boxes = [b for b in (mask_to_box(m, img.shape[:2]) for m in masks) if b]
        pairs, miss_gt, extra = match_greedy(gt_boxes, pred_boxes)

        d = per_device.setdefault(headset, {"gt": 0, "pred": 0, "hit": 0, "ious": []})
        d["gt"] += len(gt_boxes)
        d["pred"] += len(pred_boxes)
        n_gt += len(gt_boxes)
        n_pred += len(pred_boxes)
        for _, _, v in pairs:
            ious.append(v)
            d["ious"].append(v)
            if v >= args.iou_thresh:
                n_hit += 1
                d["hit"] += 1

        if args.save_overlays and saved < args.max_overlays:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for b in gt_boxes:
                cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
            for b in pred_boxes:
                cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)
            cv2.putText(vis, f"{headset} green=GT red=SAM", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(args.save_overlays, f"sam_{saved:03d}_{headset}.png"), vis)
            saved += 1

        if (k + 1) % 25 == 0:
            print(f"[sam] {k+1}/{len(idxs)} frames, {n_gt} gt, {n_pred} pred, {n_hit} hits")

    def block(tag, gt, pred, hit, vals):
        rec = 100.0 * hit / gt if gt else 0.0
        prec = 100.0 * hit / pred if pred else 0.0
        mi = float(np.mean(vals)) if vals else 0.0
        md = float(np.median(vals)) if vals else 0.0
        print(f"  {tag:10s} gt={gt:5d} pred={pred:5d}  recall {rec:6.2f}%  "
              f"precision {prec:6.2f}%  mean IoU {mi:.4f}  median {md:.4f}")
        return {"gt": gt, "pred": pred, "hits": hit, "recall_pct": rec,
                "precision_pct": prec, "mean_iou": mi, "median_iou": md}

    print("\n" + "=" * 66)
    print(f"SAM 3 auto-annotation, prompt={args.prompt!r}, "
          f"match at IoU >= {args.iou_thresh}, clahe={bool(args.clahe)}")
    print("=" * 66)
    summary = {"overall": block("overall", n_gt, n_pred, n_hit, ious), "by_device": {}}
    for dev, d in sorted(per_device.items()):
        summary["by_device"][dev] = block(dev, d["gt"], d["pred"], d["hit"], d["ious"])
    print("=" * 66)
    print("Look at the overlays before trusting these numbers. If SAM is "
          "segmenting whole arms rather than hands, low IoU says nothing about "
          "segmentation quality.")

    if args.out:
        summary["config"] = vars(args)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[sam] wrote {args.out}")


if __name__ == "__main__":
    main()
