"""
eval_keynet.py -- score a set of KeyNet weights against a HOT3D split.

Moved here from py/training/keypoint/evaluate_keypoint.py. It now sits beside
eval_detnet.py and takes its preprocessing from py/evaluation/
preprocess_baseline.py, so neither evaluator reaches into a training module
for the definition of the input convention.

WHY A SEPARATE SCRIPT FROM TRAINING
-----------------------------------
Training reports a heatmap MSE loss. That is fine for watching convergence
and useless in a results table: it is the mean squared difference between
two blurred 22x22 grids, in no meaningful unit, and it cannot be compared
against the zero-shot baseline without running a training job that need not
exist. This script asks the question the thesis actually asks -- how far, in
pixels, is each predicted joint from where it really is -- and answers it
identically for any set of weights.

DETERMINISM
-----------
The previous version of this script inherited its crops from the training
dataset, which draws a fresh rotation over the full circle and a fresh
radius multiplier for every sample, centred on randomly noised keypoints.
Nothing seeded them. The baseline and the fine-tuned model were therefore
scored on different crops of the same frames, and neither run reproduced
itself. `--deterministic-crop` (the default) removes all three draws: the
crop is centred on the ground truth at the centre of the rotation and radius
distributions, so the input to both models is pixel-identical.
`--stochastic-crop` restores the training-time behaviour as an ablation, and
seeds it so at least it is reproducible.

DECODING HEATMAPS TO COORDINATES
--------------------------------
KeyNet outputs no coordinates. Per joint it outputs a 22x22 grid whose
brightest region marks the location within the 128x128 crop. One cell spans
128/22 = 5.82 px, so the brightest cell alone would quantise every
measurement to ~5.8 px -- potentially coarser than the effect being
measured, which would hide a real improvement inside rounding error.

The peak cell is therefore located first and then refined by a weighted
centroid over a 5x5 window around it, using clamped heatmap values as
weights. This stays local, unlike a global soft-argmax, which stray
activation elsewhere in the map pulls off target.

The +0.5 offsets match maker_of_augmentations.make_heatmap_output(): the
heatmap grid is built as `arange(size) + 0.5 - centre`, so cell i peaks when
the target sits at i + 0.5 in heatmap units. Depth inverts
`depth_value = ((z / 1.5 / 2) + 0.5) * 22` the same way in 1D.

MEASURING THE MEASUREMENT
-------------------------
Before scoring any model, the ground-truth heatmaps are pushed through the
identical decoder and compared against the exact coordinates in
`gt_joint_locs`. A perfect model would score exactly that, so it is the
noise floor of the whole experiment. Any difference between two models that
is not comfortably larger than this floor is not a result, and it is printed
beside every evaluation for that reason.

FAIRNESS
--------
HOT3DKeypointDataset can supply `input_predicted_keypoints`, a deliberately
noised copy of the ground truth standing in for what a tracker would know
from the previous frame. Feeding it during evaluation leaks ground truth
into the input, so it is withheld by default -- the same thing
validatoor.validation_loop does with use_prediction=False. Both models are
handicapped identically. --use-predicted-input restores it as an ablation.

Usage:
    python py/evaluation/eval_keynet.py --weights monado --split test_mixed \
        --out results/keynet_baseline_mixed.json

    python py/evaluation/eval_keynet.py \
        --weights py/training/keypoint/checkpoints/checkpoint_best.pth \
        --split test_mixed --out results/keynet_finetuned_mixed.json
"""
import argparse
import json
import os
import random
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MERCURY_TRAIN_ROOT = os.path.join(_THIS_DIR, "..", "..")
for _p in (_MERCURY_TRAIN_ROOT, os.path.join(_THIS_DIR, "..", "training", "keypoint")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
from torch.utils.data import DataLoader

import preprocess_baseline as pp
import local_config
import py.training.common.hot3d_split as hot3d_split
from HOT3DKeypointDataset import HOT3DKeypointDataset, worker_init as hot3d_worker_init
import KeyNet
from load_weights import load_keynet_weights

HEATMAP_SIDE = pp.HEATMAP_SIDE
CROP_SIDE_PX = pp.KEYPOINT_CROP_SIZE
PX_PER_CELL = CROP_SIDE_PX / HEATMAP_SIDE          # 5.818...
DEPTH_HALF_RANGE = pp.DEPTH_HALF_RANGE

REFINE_RADIUS = 2                                  # 2 gives a 5x5 window
PCK_THRESHOLDS_PX = (5.0, 10.0)

JOINT_NAMES = [
    "wrist", "thumb_mcp", "thumb_pxm", "thumb_dst", "thumb_tip",
    "index_pxm", "index_int", "index_dst", "index_tip",
    "middle_pxm", "middle_int", "middle_dst", "middle_tip",
    "ring_pxm", "ring_int", "ring_dst", "ring_tip",
    "pinky_pxm", "pinky_int", "pinky_dst", "pinky_tip",
]


# ---------------------------------------------------------------------------
# Heatmap decoding
# ---------------------------------------------------------------------------

def _refine_1d(values, peak_idx, radius):
    """Weighted-centroid refinement of a 1D peak.
    values (..., N) non-negative weights; peak_idx (...) integer index.
    Returns (...) float cell coordinate."""
    n = values.shape[-1]
    offsets = torch.arange(-radius, radius + 1, device=values.device)
    idx = peak_idx.unsqueeze(-1) + offsets
    valid = (idx >= 0) & (idx < n)
    w = torch.gather(values, -1, idx.clamp(0, n - 1)) * valid
    total = w.sum(-1)
    centroid = (w * idx.clamp(0, n - 1)).sum(-1) / total.clamp(min=1e-12)
    # A window with no positive mass falls back to the raw peak.
    return torch.where(total > 0, centroid, peak_idx.to(centroid.dtype))


def decode_xy(hmaps):
    """(B, 21, 22, 22) -> (B, 21, 2) in 128x128 crop pixels."""
    b, j, h, w = hmaps.shape
    clamped = hmaps.clamp(min=0)
    peak = clamped.reshape(b, j, h * w).argmax(dim=-1)
    peak_y = torch.div(peak, w, rounding_mode="floor")
    peak_x = peak % w
    # The heatmap is a rank-1 outer product of two 1D Gaussians by
    # construction (heatmap_1d.two_heatmaps_to_2d), so the marginals are the
    # right thing to take a centroid over.
    cx = _refine_1d(clamped.sum(dim=2), peak_x, REFINE_RADIUS)
    cy = _refine_1d(clamped.sum(dim=3), peak_y, REFINE_RADIUS)
    return torch.stack([(cx + 0.5) * PX_PER_CELL, (cy + 0.5) * PX_PER_CELL], dim=-1)


def decode_depth(hmaps):
    """(B, 21, 22) -> (B, 21) relative depth."""
    clamped = hmaps.clamp(min=0)
    d = _refine_1d(clamped, clamped.argmax(dim=-1), REFINE_RADIUS) + 0.5
    return (d / HEATMAP_SIDE - 0.5) * 2.0 * DEPTH_HALF_RANGE


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(weights, device):
    model = KeyNet.KeyNet()
    # Needed on both branches: this gives the InvertedResidual convs
    # (bias=False) real bias parameters, matching what kpest_trainer.py does
    # before it trains or checkpoints anything.
    load_keynet_weights(model)

    if weights == "monado":
        source = "monado (zero-shot, no fine-tuning)"
    else:
        # weights_only=False: checkpoints written by this project and loaded
        # by it; the torch 2.6 default rejects them for holding a plain
        # Python/numpy scalar. See kpest_trainer.py.
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
        # Written from model.module.state_dict(), so no "module." prefix.
        model.load_state_dict(checkpoint.get("state_dict", checkpoint))
        epoch = checkpoint.get("epoch")
        source = f"{weights}" + (f" (epoch {epoch})" if epoch is not None else "")

    return model.to(device).eval(), source


SPLIT_CHOICES = [
    # Current mixed Aria+Quest design.
    "val_mixed", "test_mixed",
    # Archived Aria-only-training / cross-device-evaluation design.
    "val", "test_aria", "test_quest", "device_shift_quest", "cross_device_test",
]


def sequence_dirs_for_split(split):
    """
    Resolve a --split name to sequence directories.

    "val" and "val_mixed" are not hot3d_split splits: they are the trainer's
    own carve-out from its training pool, reproduced here with the same
    function and the same seed so that each means exactly the sequences the
    corresponding trainer validated on. "val" stays pointed at the archived
    Aria-only training pool so the archived checkpoints remain scorable.
    """
    root = local_config.hot3d_dataset_root
    if split in ("val", "val_mixed"):
        train_split = "train" if split == "val" else "train_mixed"
        _, val_dirs = hot3d_split.split_train_val(
            hot3d_split.list_sequence_dirs(root, train_split))
        return val_dirs
    return hot3d_split.list_sequence_dirs(root, split)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def evaluate(model, dataloader, device, use_predicted_input, limit_batches=None):
    """
    A joint is excluded from scoring if its ground truth falls outside the
    128x128 crop (its Gaussian would sit off the grid entirely, so no
    prediction could match it), or if HOT3DKeypointDataset marked it invalid
    at index time -- behind the camera, or outside the camera model's valid
    region (see _project_hand). The second check matters as of index format
    version 3: an invalid joint's stored position is a geometry-safe
    placeholder kept only so cropping stays stable, and that placeholder can
    land inside the crop, so the crop test alone no longer keeps it out.
    Depth is gated additionally on depth_valid_per_joint, since a joint can
    be a usable 2D position while its depth is undefined.
    """
    errors, depth_errors, floor_errors = [], [], []
    joint_masks, depth_joint_masks = [], []
    n_samples = n_degenerate = 0

    total = len(dataloader) if not limit_batches else min(len(dataloader), limit_batches)
    with torch.no_grad():
        for batch_idx, doct in enumerate(dataloader):
            if limit_batches and batch_idx >= limit_batches:
                break
            if batch_idx % 20 == 0:
                print(f"  batch {batch_idx}/{total}", flush=True)

            image = doct["input_image"].to(device)
            gt_locs = doct["gt_joint_locs"].to(device)                 # (B, 21, 3)
            gt_xy_hmap = doct["gt_xy"].to(device).float()
            xy_valid = doct["xy_valid_per_joint"].to(device).bool()
            depth_valid = doct["depth_valid_per_joint"].to(device).bool()

            pred_kp = doct["input_predicted_keypoints"].to(device).float()
            pred_valid = doct["input_predicted_keypoints_valid"].to(device).float()
            if not use_predicted_input:
                pred_kp = torch.zeros_like(pred_kp)
                pred_valid = torch.zeros_like(pred_valid)

            model_xy, model_depth, _extras, _curls = model(
                image, torch.flatten(pred_kp, start_dim=1), pred_valid)

            gt_px, gt_z = gt_locs[..., :2], gt_locs[..., 2]

            # _empty_sample() puts all 21 joints at one point. An affine crop
            # maps identical points to identical points, so the signature
            # survives cropping and identifies those samples reliably.
            spread = (gt_px.max(dim=1).values - gt_px.min(dim=1).values).max(dim=1).values
            sample_ok = spread > 1.0
            n_degenerate += int((~sample_ok).sum())
            n_samples += int(sample_ok.sum())
            if not bool(sample_ok.any()):
                continue

            in_crop = ((gt_px >= 0) & (gt_px < CROP_SIDE_PX)).all(dim=-1)
            mask = in_crop & sample_ok.unsqueeze(-1) & xy_valid
            errors.append(torch.linalg.norm(decode_xy(model_xy) - gt_px, dim=-1).cpu())
            floor_errors.append(torch.linalg.norm(decode_xy(gt_xy_hmap) - gt_px, dim=-1).cpu())
            depth_errors.append((decode_depth(model_depth) - gt_z).abs().cpu())
            joint_masks.append(mask.cpu())
            depth_joint_masks.append((mask & depth_valid).cpu())

    if not errors:
        raise RuntimeError("No usable samples in this split.")

    err = torch.cat(errors).numpy()
    floor = torch.cat(floor_errors).numpy()
    derr = torch.cat(depth_errors).numpy()
    mask = torch.cat(joint_masks).numpy()
    dmask = torch.cat(depth_joint_masks).numpy()

    m, dm = mask.reshape(-1), dmask.reshape(-1)
    e, f, d = err.reshape(-1)[m], floor.reshape(-1)[m], derr.reshape(-1)[dm]

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


def print_report(split, source, frame_stride, use_predicted_input, deterministic, r):
    print("\n" + "=" * 66)
    print(f"  split            {split}")
    print(f"  weights          {source}")
    print(f"  frame_stride     {frame_stride}")
    print(f"  crop             {'deterministic' if deterministic else 'STOCHASTIC (ablation)'}")
    print(f"  predicted input  {'ENABLED (ablation)' if use_predicted_input else 'withheld (default)'}")
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
    print("    ^ what a PERFECT model would score. Any difference between")
    print("      two models smaller than this is measurement noise.")
    print("-" * 66)
    print("  per-joint mean error (px):")
    for name, v in zip(JOINT_NAMES, r["per_joint_mean_error_px"]):
        print(f"    {name:<12} {v:7.3f}" if v is not None else f"    {name:<12}      --")
    print("=" * 66 + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--weights", required=True,
                        help="'monado' for the zero-shot upstream weights, or a .pth path")
    parser.add_argument("--split", required=True,
                        choices=SPLIT_CHOICES)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--stochastic-crop", action="store_true",
                        help="ABLATION ONLY: restore the training-time random crop "
                             "rotation, radius and keypoint noise. Seeded, so at "
                             "least reproducible, but the two models are then no "
                             "longer scored on identical inputs.")
    parser.add_argument("--use-predicted-input", action="store_true",
                        help="ABLATION ONLY: feed the noised ground-truth "
                             "previous-frame keypoints. Leaks ground truth into "
                             "the input; not the headline number.")
    parser.add_argument("--limit", type=int, default=None,
                        help="score only the first N batches, for sanity checks")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers; safe above 0 thanks to "
                             "HOT3DKeypointDataset.worker_init")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="write metrics as JSON here")
    args = parser.parse_args()

    deterministic = not args.stochastic_crop
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    pp.self_check(verbose=False)

    sequence_dirs = sequence_dirs_for_split(args.split)
    if not sequence_dirs:
        raise SystemExit(f"No sequences for split {args.split!r} under "
                         f"{local_config.hot3d_dataset_root}")
    print(f"[eval_keynet] {args.split}: {len(sequence_dirs)} sequences")

    try:
        dataset = HOT3DKeypointDataset(
            sequence_dirs=sequence_dirs,
            hot3d_repo_root=local_config.hot3d_repo_root,
            object_library_path=local_config.hot3d_object_library_path,
            frame_stride=args.frame_stride,
            index_cache_dir=getattr(local_config, "hot3d_index_cache_dir", None),
            eval_mode=deterministic,
        )
    except NotImplementedError as e:
        raise SystemExit(
            f"\n[eval_keynet] HOT3DKeypointDataset could not load this "
            f"split.\n  {e}\n"
            f"If the split contains Quest recordings, check them first with "
            f"py/evaluation/check_quest_keypoints.py.")

    # Guards the photometric half of the convention: augmentation must be off.
    assert dataset.augmaker.aug_config.validation_dataset, \
        ("HOT3DKeypointDataset is using a training augmentation config. "
         "Mixup, cutout, noise and blur would be applied to evaluation "
         "inputs, so the number would describe augmented data, not the split.")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers,
                            worker_init_fn=hot3d_worker_init, timeout=0,
                            persistent_workers=False, drop_last=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, source = build_model(args.weights, device)

    result = evaluate(model, dataloader, device, args.use_predicted_input, args.limit)
    print_report(args.split, source, args.frame_stride, args.use_predicted_input,
                 deterministic, result)

    if args.out:
        payload = dict(result)
        payload.update({
            "model": "KeyNet", "split": args.split, "weights": args.weights,
            "weights_source": source, "frame_stride": args.frame_stride,
            "use_predicted_input": args.use_predicted_input,
            "deterministic_crop": deterministic, "seed": args.seed,
            "n_sequences": len(sequence_dirs),
        })
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[eval_keynet] wrote {args.out}")


if __name__ == "__main__":
    main()
