"""
evaluate_keypoint.py -- score a set of KeyNet weights against a HOT3D split.

WHY THIS IS A SEPARATE SCRIPT
-----------------------------
Training reports a heatmap MSE loss. That number is fine for watching
convergence and useless for a results table: it is the mean squared
difference between two blurred 22x22 grids, in no meaningful unit, and it
cannot be compared against the zero-shot Monado baseline without running a
training job that does not need to exist.

This script answers the question the thesis actually asks -- "how far, in
pixels, is each predicted hand joint from where it really is?" -- and it
answers it identically for any set of weights. The zero-shot baseline and
the fine-tuned model therefore go through exactly the same code path, and
the metric can be changed later without retraining anything.

    # zero-shot Monado baseline, needs no training at all
    python py/training/keypoint/evaluate_keypoint.py --weights monado --split test_aria

    # the fine-tuned model
    python py/training/keypoint/evaluate_keypoint.py \
        --weights py/training/keypoint/checkpoints/checkpoint_best.pth \
        --split test_aria

DECODING HEATMAPS TO COORDINATES
--------------------------------
KeyNet does not output coordinates. Per joint it outputs a 22x22 grid whose
brightest region marks the predicted location within the 128x128 crop. One
grid cell therefore spans 128/22 = 5.82 pixels, so taking the brightest cell
alone would quantise every measurement to ~5.8px -- potentially coarser than
the effect being measured, which would hide a real improvement inside
rounding error.

So the peak cell is located first, then refined: a weighted centroid is taken
over a 5x5 window centred on that peak, using clamped (non-negative) heatmap
values as weights. That gives a sub-cell estimate while staying local, unlike
a global soft-argmax, which is pulled off-target by stray activation
elsewhere in the map.

The +0.5 offsets below match maker_of_augmentations.make_heatmap_output()'s
own convention: heatmap_1d builds its grid as `arange(size) + 0.5 - center`,
so grid cell i peaks when the target coordinate is i + 0.5 in heatmap units,
and heatmap units are pixels * 22/128.

Depth is decoded the same way in 1D, inverting
`depth_value = ((z / 1.5 / 2) + 0.5) * 22`.

MEASURING THE MEASUREMENT
-------------------------
Before scoring the model, the ground-truth heatmaps are pushed through the
identical decoder and compared against the exact ground-truth coordinates in
`gt_joint_locs`. A perfect model would score exactly that, so it is the noise
floor of the whole experiment. Any difference between two models that is not
comfortably larger than this floor is not a result. It is printed alongside
every evaluation for exactly that reason.

FAIRNESS
--------
HOT3DKeypointDataset supplies `input_predicted_keypoints` -- a deliberately
noised copy of the ground truth, standing in for what a real tracker would
know from the previous video frame. Feeding that during evaluation would
leak ground truth into the model's input, so it is zeroed by default (the
same thing validatoor.validation_loop does with use_prediction=False). Both
models are handicapped identically, so the comparison stays clean.
--use-predicted-input restores it, as an ablation only.
"""
import argparse
import json
import os
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../'))

import numpy as np
import torch
from torch.utils.data import DataLoader

import local_config
import py.training.common.hot3d_split as hot3d_split
from HOT3DKeypointDataset import HOT3DKeypointDataset, worker_init as hot3d_worker_init
import KeyNet
from load_weights import load_keynet_weights


# Must match maker_of_augmentations.make_heatmap_output().
HEATMAP_SIDE = 22
CROP_SIDE_PX = 128
PX_PER_CELL = CROP_SIDE_PX / HEATMAP_SIDE          # 5.818...
DEPTH_HALF_RANGE = 1.5                             # z is expected in [-1.5, 1.5]

# Half-width of the refinement window, in cells. 2 gives a 5x5 window.
REFINE_RADIUS = 2

# PCK thresholds, in pixels of the 128x128 crop.
PCK_THRESHOLDS_PX = (5.0, 10.0)


def _refine_1d(values, peak_idx, radius):
    """
    Weighted-centroid refinement of a 1D peak.

    values:    (..., N) non-negative weights
    peak_idx:  (...)    integer index of the peak along the last axis
    Returns    (...)    float cell coordinate.
    """
    n = values.shape[-1]
    offsets = torch.arange(-radius, radius + 1, device=values.device)
    idx = peak_idx.unsqueeze(-1) + offsets                      # (..., k)
    valid = (idx >= 0) & (idx < n)
    idx_clamped = idx.clamp(0, n - 1)

    w = torch.gather(values, -1, idx_clamped) * valid
    total = w.sum(-1)

    centroid = (w * idx.clamp(0, n - 1)).sum(-1) / total.clamp(min=1e-12)
    # A window with no positive mass at all falls back to the raw peak.
    return torch.where(total > 0, centroid, peak_idx.to(centroid.dtype))


def decode_xy(hmaps):
    """
    hmaps: (B, 21, 22, 22) predicted or ground-truth xy heatmaps.
    Returns (B, 21, 2) x,y coordinates in 128x128 crop pixels.
    """
    b, j, h, w = hmaps.shape
    clamped = hmaps.clamp(min=0)

    flat = clamped.reshape(b, j, h * w)
    peak = flat.argmax(dim=-1)
    peak_y = torch.div(peak, w, rounding_mode="floor")
    peak_x = peak % w

    # Refine each axis against that axis's marginal within the window. The
    # heatmap is a rank-1 outer product of two 1D Gaussians by construction
    # (heatmap_1d.two_heatmaps_to_2d), so the marginals are the right thing
    # to take a centroid over.
    marg_x = clamped.sum(dim=2)     # (B, J, W)
    marg_y = clamped.sum(dim=3)     # (B, J, H)

    cx = _refine_1d(marg_x, peak_x, REFINE_RADIUS)
    cy = _refine_1d(marg_y, peak_y, REFINE_RADIUS)

    return torch.stack([(cx + 0.5) * PX_PER_CELL,
                        (cy + 0.5) * PX_PER_CELL], dim=-1)


def decode_depth(hmaps):
    """
    hmaps: (B, 21, 22) predicted or ground-truth depth heatmaps.
    Returns (B, 21) relative-depth values, inverting
    depth_value = ((z / 1.5 / 2) + 0.5) * 22.
    """
    clamped = hmaps.clamp(min=0)
    peak = clamped.argmax(dim=-1)
    d = _refine_1d(clamped, peak, REFINE_RADIUS) + 0.5
    return (d / HEATMAP_SIDE - 0.5) * 2.0 * DEPTH_HALF_RANGE


def build_model(weights, device):
    model = KeyNet.KeyNet()

    if weights == "monado":
        # The unmodified upstream Monado weights -- the zero-shot baseline.
        load_keynet_weights(model)
        source = "monado (zero-shot, no fine-tuning)"
    else:
        checkpoint = torch.load(weights, map_location="cpu")
        # Checkpoints are written from model.module.state_dict(), so the keys
        # carry no "module." prefix.
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict)
        epoch = checkpoint.get("epoch")
        source = f"{weights}" + (f" (epoch {epoch})" if epoch is not None else "")

    model = model.to(device)
    model.eval()
    return model, source


def sequence_dirs_for_split(split):
    root = local_config.hot3d_dataset_path
    if split == "val":
        # The same carve-out the trainer uses, so "val" here means exactly
        # the sequences the trainer validated on.
        _, val_dirs = hot3d_split.split_train_val(
            hot3d_split.list_sequence_dirs(root, "train"))
        return val_dirs
    return hot3d_split.list_sequence_dirs(root, split)


def evaluate(model, dataloader, device, use_predicted_input, limit_batches=None):
    """
    Returns a dict of metrics. Joints whose ground truth falls outside the
    128x128 crop are excluded: their Gaussian would sit off the heatmap grid
    entirely, so no prediction could match them and scoring against them
    measures nothing.
    """
    errors = []          # per-joint 2D error, pixels
    depth_errors = []    # per-joint depth error, relative-depth units
    floor_errors = []    # per-joint decoder error on ground-truth heatmaps
    joint_masks = []

    n_samples = 0
    n_degenerate = 0

    total_batches = len(dataloader)
    if limit_batches:
        total_batches = min(total_batches, limit_batches)
    with torch.no_grad():
        for batch_idx, doct in enumerate(dataloader):
            if limit_batches and batch_idx >= limit_batches:
                break
            if batch_idx % 20 == 0:
                print(f"  batch {batch_idx}/{total_batches}", flush=True)

            image = doct["input_image"].to(device)
            gt_locs = doct["gt_joint_locs"].to(device)          # (B, 21, 3)
            gt_xy_hmap = doct["gt_xy"].to(device).float()
            gt_depth_hmap = doct["gt_depth"].to(device).float()

            pred_kp = doct["input_predicted_keypoints"].to(device).float()
            pred_valid = doct["input_predicted_keypoints_valid"].to(device).float()
            if not use_predicted_input:
                pred_kp = torch.zeros_like(pred_kp)
                pred_valid = torch.zeros_like(pred_valid)

            model_xy, model_depth, _extras, _curls = model(
                image, torch.flatten(pred_kp, start_dim=1), pred_valid)

            gt_px = gt_locs[..., :2]
            gt_z = gt_locs[..., 2]

            # A sample produced by HOT3DKeypointDataset._empty_sample() has
            # all 21 joints at the same point. An affine crop maps identical
            # points to identical points, so this signature survives
            # augmentation and identifies those samples reliably.
            spread = (gt_px.max(dim=1).values - gt_px.min(dim=1).values).max(dim=1).values
            sample_ok = spread > 1.0
            n_degenerate += int((~sample_ok).sum())
            n_samples += int(sample_ok.sum())
            if not bool(sample_ok.any()):
                continue

            in_crop = ((gt_px >= 0) & (gt_px < CROP_SIDE_PX)).all(dim=-1)
            mask = in_crop & sample_ok.unsqueeze(-1)           # (B, 21)

            pred_px = decode_xy(model_xy)
            pred_z = decode_depth(model_depth)
            floor_px = decode_xy(gt_xy_hmap)

            errors.append(torch.linalg.norm(pred_px - gt_px, dim=-1).cpu())
            floor_errors.append(torch.linalg.norm(floor_px - gt_px, dim=-1).cpu())
            depth_errors.append((pred_z - gt_z).abs().cpu())
            joint_masks.append(mask.cpu())

    if not errors:
        raise RuntimeError("No usable samples in this split.")

    err = torch.cat(errors).numpy()
    floor = torch.cat(floor_errors).numpy()
    derr = torch.cat(depth_errors).numpy()
    mask = torch.cat(joint_masks).numpy()

    m = mask.reshape(-1)
    e = err.reshape(-1)[m]
    f = floor.reshape(-1)[m]
    d = derr.reshape(-1)[m]

    per_joint = []
    for joint in range(err.shape[1]):
        jm = mask[:, joint]
        per_joint.append(float(err[:, joint][jm].mean()) if jm.any() else None)

    return {
        "n_samples": n_samples,
        "n_degenerate_samples_excluded": n_degenerate,
        "n_joints_scored": int(m.sum()),
        "n_joints_excluded_outside_crop": int((~m).sum()),
        "mean_joint_error_px": float(e.mean()),
        "median_joint_error_px": float(np.median(e)),
        "decoder_error_floor_px": float(f.mean()),
        "depth_mae": float(d.mean()),
        "pck": {f"@{t}px": float((e < t).mean()) for t in PCK_THRESHOLDS_PX},
        "per_joint_mean_error_px": per_joint,
    }


JOINT_NAMES = [
    "wrist", "thumb_mcp", "thumb_pxm", "thumb_dst", "thumb_tip",
    "index_pxm", "index_int", "index_dst", "index_tip",
    "middle_pxm", "middle_int", "middle_dst", "middle_tip",
    "ring_pxm", "ring_int", "ring_dst", "ring_tip",
    "pinky_pxm", "pinky_int", "pinky_dst", "pinky_tip",
]


def print_report(split, source, frame_stride, use_predicted_input, r):
    print("\n" + "=" * 66)
    print(f"  split            {split}")
    print(f"  weights          {source}")
    print(f"  frame_stride     {frame_stride}")
    print(f"  predicted input  {'ENABLED (ablation)' if use_predicted_input else 'zeroed (default)'}")
    print("-" * 66)
    print(f"  samples scored                  {r['n_samples']}")
    print(f"  joints scored                   {r['n_joints_scored']}")
    print(f"  joints excluded (outside crop)  {r['n_joints_excluded_outside_crop']}")
    print(f"  samples excluded (degenerate)   {r['n_degenerate_samples_excluded']}")
    print("-" * 66)
    print(f"  MEAN JOINT ERROR      {r['mean_joint_error_px']:8.3f} px")
    print(f"  median joint error    {r['median_joint_error_px']:8.3f} px")
    for k, v in r["pck"].items():
        print(f"  PCK {k:<8}          {v * 100:8.2f} %")
    print(f"  depth MAE             {r['depth_mae']:8.4f} (relative-depth units)")
    print("-" * 66)
    print(f"  decoder error floor   {r['decoder_error_floor_px']:8.3f} px")
    print( "    ^ what a PERFECT model would score. Any difference between")
    print( "      two models smaller than this is measurement noise.")
    print("-" * 66)
    print("  per-joint mean error (px):")
    for name, v in zip(JOINT_NAMES, r["per_joint_mean_error_px"]):
        print(f"    {name:<12} {v:7.3f}" if v is not None else f"    {name:<12}      --")
    print("=" * 66 + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--weights", required=True,
                        help="'monado' for the zero-shot upstream weights, or a "
                             "path to a checkpoint .pth")
    parser.add_argument("--split", required=True,
                        choices=["val", "test_aria", "test_quest",
                                 "device_shift_quest", "cross_device_test"])
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--use-predicted-input", action="store_true",
                        help="ABLATION ONLY: feed the noised ground-truth "
                             "previous-frame keypoints. Leaks ground truth "
                             "into the input; not the headline number.")
    parser.add_argument("--limit", type=int, default=None,
                        help="score only the first N batches. For quick sanity "
                             "checks -- a full split takes hours, 20 batches "
                             "takes a minute.")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers. Safe above 0 thanks to "
                             "HOT3DKeypointDataset.worker_init; use it with "
                             "--limit to verify workers agree with the "
                             "single-process path before trusting a long run.")
    parser.add_argument("--out", default=None, help="write metrics as JSON here")
    args = parser.parse_args()

    sequence_dirs = sequence_dirs_for_split(args.split)
    if not sequence_dirs:
        raise SystemExit(f"No sequences found for split {args.split!r} under "
                         f"{local_config.hot3d_dataset_path}")
    print(f"[evaluate_keypoint] {args.split}: {len(sequence_dirs)} sequences")

    try:
        dataset = HOT3DKeypointDataset(
            sequence_dirs=sequence_dirs,
            hot3d_repo_root=local_config.hot3d_repo_root,
            object_library_path=local_config.hot3d_object_library_path,
            frame_stride=args.frame_stride,
            index_cache_dir=getattr(local_config, "hot3d_index_cache_dir", None),
        )
    except NotImplementedError as e:
        raise SystemExit(
            f"\n[evaluate_keypoint] This split contains Quest recordings, which "
            f"HOT3DKeypointDataset cannot load yet.\n  {e}\n"
            f"Aria splits (val, test_aria) work today.")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=hot3d_worker_init,
        timeout=0,
        persistent_workers=False,
        drop_last=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, source = build_model(args.weights, device)

    result = evaluate(model, dataloader, device, args.use_predicted_input, args.limit)
    print_report(args.split, source, args.frame_stride, args.use_predicted_input, result)

    if args.out:
        payload = dict(result)
        payload.update({
            "split": args.split,
            "weights": args.weights,
            "weights_source": source,
            "frame_stride": args.frame_stride,
            "use_predicted_input": args.use_predicted_input,
            "n_sequences": len(sequence_dirs),
        })
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[evaluate_keypoint] wrote {args.out}")


if __name__ == "__main__":
    main()
