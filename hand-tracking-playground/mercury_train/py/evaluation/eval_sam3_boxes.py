"""
How good would SAM 3 be at annotating hand boxes? Prompts it with "hand" on HOT3D
frames and scores the results with eval_detnet.py's IoU. Only rotate_upright and
grayscale-to-RGB are applied; Mercury's normalisation would measure format
mismatch. SAM cuts at the wrist while HOT3D's box covers the whole hand plus 15%,
so --scale-sweep measures how much of that gap a fixed expansion closes.
"""

import argparse
import json
import os
import random
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_MERCURY_TRAIN_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_MERCURY_TRAIN_ROOT, os.path.join(_MERCURY_TRAIN_ROOT, "py", "training", "detection")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from py.evaluation import preprocess_baseline as _pp
from py.training.detection.HOT3DVRSDetectionDataset import HOT3DVRSDetectionDataset
from py.training.common.hot3d_split import list_sequence_dirs
import py.training.detection.local_config as local_config


# --- Ground truth, exactly as HOT3DVRSDetectionDataset builds it ---------------

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


# --- SAM -----------------------------------------------------------------------

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
    ap.add_argument("--gt-margin", type=float, default=None,
                    help="override the ground-truth box margin (Section 4.2 uses "
                         "0.15). Pass 0.0 to score against the unpadded mocap box, "
                         "which separates SAM's disagreement with HOT3D from this "
                         "project's own padding choice.")
    ap.add_argument("--scale-sweep", default="1.0,1.1,1.2,1.3,1.4,1.5,1.6",
                    help="comma-separated box expansion factors, applied about "
                         "each predicted box's centre, re-matched from the same "
                         "masks so no SAM inference is repeated")
    ap.add_argument("--out-csv", default=None,
                    help="per-matched-pair rows, for characterising the offset "
                         "offline without re-running SAM")
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
    if args.gt_margin is not None:
        print(f"[sam] overriding ground-truth margin {ds.margin} -> {args.gt_margin}")
        ds.margin = args.gt_margin
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

    frames = []          # (headset, gt_boxes, pred_boxes) per frame
    saved = 0

    for k, idx in enumerate(idxs):
        s_ = raw_sample(ds, idx)
        if s_ is None:
            continue
        img, gt_boxes, headset = s_
        if not gt_boxes:
            continue

        sam_in = clahe.apply(img) if clahe is not None else img
        rgb = cv2.cvtColor(sam_in, cv2.COLOR_GRAY2RGB)

        predictor.set_image(rgb)
        res = predictor(text=[args.prompt])
        masks = masks_from_result(res, verbose=(k == 0))

        pred_boxes = [b for b in (mask_to_box(m, img.shape[:2]) for m in masks) if b]
        frames.append((headset, gt_boxes, pred_boxes))

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
            print(f"[sam] {k+1}/{len(idxs)} frames")

    # --- Scoring ---------------------------------------------------------------
    # Scaling happens here, not in the loop, so the sweep reuses one pass of inference.

    def scaled(b, f):
        cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
        hw, hh = (b[2] - b[0]) / 2.0 * f, (b[3] - b[1]) / 2.0 * f
        return (cx - hw, cy - hh, cx + hw, cy + hh)

    def evaluate(scale):
        acc = {}
        pairs_out = []
        for headset, gt_boxes, pred_boxes in frames:
            pb = [scaled(b, scale) for b in pred_boxes]
            pairs, _, _ = match_greedy(gt_boxes, pb)
            d = acc.setdefault(headset, {"gt": 0, "pred": 0, "hit": 0, "ious": []})
            d["gt"] += len(gt_boxes)
            d["pred"] += len(pb)
            for gi, pi, v in pairs:
                d["ious"].append(v)
                if v >= args.iou_thresh:
                    d["hit"] += 1
                pairs_out.append((headset, gt_boxes[gi], pb[pi], v))
        return acc, pairs_out

    def block(tag, gt, pred, hit, vals):
        rec = 100.0 * hit / gt if gt else 0.0
        prec = 100.0 * hit / pred if pred else 0.0
        mi = float(np.mean(vals)) if vals else 0.0
        md = float(np.median(vals)) if vals else 0.0
        # Matched-pairs mean IoU answers "when SAM finds a hand, how good is the box".
        # Not comparable to eval_detnet.py's mean IoU; the second figure scores misses as zero.
        mi_all = float(sum(vals) / gt) if gt else 0.0
        print(f"  {tag:10s} gt={gt:5d} pred={pred:5d}  recall {rec:6.2f}%  "
              f"precision {prec:6.2f}%  IoU(matched) {mi:.4f}  median {md:.4f}  "
              f"IoU(all GT) {mi_all:.4f}")
        return {"gt": gt, "pred": pred, "hits": hit, "recall_pct": rec,
                "precision_pct": prec, "mean_iou_matched": mi, "median_iou": md,
                "mean_iou_over_all_gt": mi_all}

    print("\n" + "=" * 78)
    print(f"SAM 3 auto-annotation, prompt={args.prompt!r}, match at IoU >= "
          f"{args.iou_thresh}, clahe={bool(args.clahe)}, gt_margin={ds.margin}")
    print("=" * 78)

    acc, pairs_out = evaluate(1.0)
    n_gt = sum(d["gt"] for d in acc.values())
    n_pred = sum(d["pred"] for d in acc.values())
    n_hit = sum(d["hit"] for d in acc.values())
    all_ious = [v for d in acc.values() for v in d["ious"]]
    summary = {"uncorrected": {"overall": block("overall", n_gt, n_pred, n_hit, all_ious),
                               "by_device": {}}}
    for dev, d in sorted(acc.items()):
        summary["uncorrected"]["by_device"][dev] = block(dev, d["gt"], d["pred"], d["hit"], d["ious"])

    # How much smaller is a SAM box than the ground-truth box it matched?
    if pairs_out:
        wr = [ (p[2] - p[0]) / (g[2] - g[0]) for _, g, p, _ in pairs_out if g[2] > g[0] ]
        hr = [ (p[3] - p[1]) / (g[3] - g[1]) for _, g, p, _ in pairs_out if g[3] > g[1] ]
        print(f"\n  SAM box size relative to ground truth, matched pairs only")
        print(f"    width  ratio  mean {np.mean(wr):.3f}  median {np.median(wr):.3f}")
        print(f"    height ratio  mean {np.mean(hr):.3f}  median {np.median(hr):.3f}")
        summary["size_ratio"] = {"width_mean": float(np.mean(wr)),
                                 "width_median": float(np.median(wr)),
                                 "height_mean": float(np.mean(hr)),
                                 "height_median": float(np.median(hr))}

    print(f"\n  Fixed-expansion sweep (same masks, boxes scaled about their centre)")
    print(f"  {'scale':>6}  {'recall%':>8}  {'precision%':>10}  {'IoU(matched)':>12}  {'IoU(all GT)':>11}")
    sweep = {}
    for f in [float(x) for x in args.scale_sweep.split(",")]:
        a, _ = evaluate(f)
        g = sum(d["gt"] for d in a.values())
        p_ = sum(d["pred"] for d in a.values())
        h = sum(d["hit"] for d in a.values())
        v = [x for d in a.values() for x in d["ious"]]
        rec = 100.0 * h / g if g else 0.0
        prec = 100.0 * h / p_ if p_ else 0.0
        mi = float(np.mean(v)) if v else 0.0
        mi_all = float(sum(v) / g) if g else 0.0
        print(f"  {f:6.2f}  {rec:8.2f}  {prec:10.2f}  {mi:12.4f}  {mi_all:11.4f}")
        sweep[f"{f:.2f}"] = {"recall_pct": rec, "precision_pct": prec,
                             "mean_iou_matched": mi, "mean_iou_over_all_gt": mi_all}
    summary["scale_sweep"] = sweep
    print("=" * 78)

    if args.out_csv:
        import csv as _csv
        with open(args.out_csv, "w", newline="") as f_:
            w = _csv.writer(f_)
            w.writerow(["device", "gt_x0", "gt_y0", "gt_x1", "gt_y1",
                        "sam_x0", "sam_y0", "sam_x1", "sam_y1", "iou"])
            for dev, g, p_, v in pairs_out:
                w.writerow([dev, *[f"{x:.2f}" for x in g], *[f"{x:.2f}" for x in p_], f"{v:.4f}"])
        print(f"[sam] wrote {args.out_csv}")

    if args.out:
        summary["config"] = vars(args)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[sam] wrote {args.out}")


if __name__ == "__main__":
    main()
