"""
gt_box_aspect_ratio.py -- how much IoU can DetNet reach in principle?

DetNet emits a single `size` value per hand slot, so every box it can produce is
a SQUARE. HOT3D's ground-truth boxes are rectangles: eval_detnet.py builds
gt["box_upright"] from the min/max of the rotated corners with no squaring, and
box_iou compares a square prediction against that rectangle.

For a ground-truth box of aspect ratio r = long/short, the best IoU any square
can achieve, at the optimal side length sqrt(w*h), is

    IoU_max = 1 / (2*sqrt(r) - 1)

which is 1.000 at r=1, 0.547 at r=2 and 0.406 at r=3. This script measures the
actual distribution of r across the evaluation split and reports the resulting
per-box ceiling, so Chapter 6 can say whether DetNet's near-zero rate at
IoU >= 0.75 reflects a training shortfall or an architectural bound.

The ceiling is an upper bound and assumes a perfectly centred, optimally sized
square. Real predictions also carry centring error, so measured IoU will sit
below it.

Reads nothing but HOT3D and modifies no existing code. Ground-truth boxes are
constructed by eval_sam3_boxes.raw_sample, which reproduces
HOT3DVRSDetectionDataset.__getitem__ exactly, including the Section 4.2 margin.

Usage:
  python py/evaluation/gt_box_aspect_ratio.py --split test_mixed --limit 2000
"""

import argparse
import json
import os
import random
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_ROOT, os.path.join(_ROOT, "py", "training", "detection")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from py.evaluation.eval_sam3_boxes import raw_sample
from py.training.detection.HOT3DVRSDetectionDataset import HOT3DVRSDetectionDataset
from py.training.common.hot3d_split import list_sequence_dirs
import py.training.detection.local_config as local_config


def ceiling_for(r):
    """Best IoU a square can reach against a rectangle of aspect ratio r."""
    return 1.0 / (2.0 * np.sqrt(r) - 1.0)


def report(tag, ratios):
    if not len(ratios):
        print(f"  {tag:10s} no boxes")
        return None
    c = ceiling_for(np.asarray(ratios))
    out = {
        "n": int(len(ratios)),
        "aspect_mean": float(np.mean(ratios)),
        "aspect_median": float(np.median(ratios)),
        "aspect_p90": float(np.percentile(ratios, 90)),
        "ceiling_mean": float(np.mean(c)),
        "ceiling_median": float(np.median(c)),
        "pct_where_0.75_reachable": float(100.0 * np.mean(c >= 0.75)),
        "pct_where_0.50_reachable": float(100.0 * np.mean(c >= 0.50)),
    }
    print(f"  {tag:10s} n={out['n']:6d}  aspect mean {out['aspect_mean']:.2f} "
          f"median {out['aspect_median']:.2f} p90 {out['aspect_p90']:.2f}  |  "
          f"IoU ceiling mean {out['ceiling_mean']:.3f} median {out['ceiling_median']:.3f}  |  "
          f"0.75 reachable for {out['pct_where_0.75_reachable']:5.1f}%  "
          f"0.50 for {out['pct_where_0.50_reachable']:5.1f}%")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--split", default="test_mixed")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frame-stride", type=int, default=5)
    ap.add_argument("--gt-margin", type=float, default=None,
                    help="override the Section 4.2 margin, e.g. 0.0 for the raw mocap box")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seq_dirs = list_sequence_dirs(local_config.hot3d_dataset_root, args.split)
    ds = HOT3DVRSDetectionDataset(
        sequence_dirs=seq_dirs,
        hot3d_repo_root=local_config.hot3d_repo_root,
        frame_stride=args.frame_stride,
        index_cache_dir=getattr(local_config, "hot3d_index_cache_dir", None),
        augment=False)
    if args.gt_margin is not None:
        print(f"[aspect] margin {ds.margin} -> {args.gt_margin}")
        ds.margin = args.gt_margin

    rng = random.Random(args.seed)
    idxs = rng.sample(range(len(ds)), min(args.limit, len(ds)))

    by_dev, all_r = {}, []
    for k, idx in enumerate(idxs):
        s = raw_sample(ds, idx)
        if s is None:
            continue
        _, gt_boxes, headset = s
        for b in gt_boxes:
            w, h = b[2] - b[0], b[3] - b[1]
            if w <= 0 or h <= 0:
                continue
            r = max(w, h) / min(w, h)
            all_r.append(r)
            by_dev.setdefault(headset, []).append(r)
        if (k + 1) % 250 == 0:
            print(f"[aspect] {k+1}/{len(idxs)} frames, {len(all_r)} boxes")

    print("\n" + "=" * 96)
    print(f"Ground-truth box aspect ratios, {args.split}, margin={ds.margin}")
    print("A square predictor's best possible IoU is 1 / (2*sqrt(r) - 1)")
    print("=" * 96)
    summary = {"overall": report("overall", all_r), "by_device": {}}
    for dev, r in sorted(by_dev.items()):
        summary["by_device"][dev] = report(dev, r)
    print("=" * 96)

    if args.out:
        summary["config"] = vars(args)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[aspect] wrote {args.out}")


if __name__ == "__main__":
    main()
