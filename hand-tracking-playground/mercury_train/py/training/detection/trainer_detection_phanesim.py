"""
trainer_detection_phanesim.py -- DetNet fine-tuning PHASE 2: continues
training from the phase-1 HOT3D fine-tuned checkpoint
(checkpoints_train_mixed/checkpoint_best.pth) using Phanesim's synthetic
data instead of HOT3D. Kept as a separate script rather than folded into
trainer_detection.py so phase 1 stays runnable/reproducible unchanged, and
so this can never accidentally overwrite the phase-1 checkpoint that the
already-submittable thesis draft's numbers are based on.

Per-thesis decision (2026-09-08): Phanesim only, no HOT3D mixed in, both
`dataset` and `dataset2` batches pooled. This carries a real catastrophic-
forgetting risk -- continuing to fine-tune on a second, fully synthetic
dataset with none of the just-learned real HOT3D signal mixed back in could
degrade what phase 1 achieved, including the cross-device consistency
Chapter 6 found for DetNet. That is exactly why this should be run as a
time-boxed TRIAL first -- re-run eval_detnet.py against test_mixed on this
phase's checkpoint_best.pth and compare against the phase-1 numbers before
committing to a full run.

Reuses train_batch/validate_epoch/set_train_mode/save_checkpoint from
trainer_detection.py unchanged -- they're not HOT3D-specific, and
PhanesimDetectionDataset's __getitem__ produces the exact same batch dict
shape (image/exists/center_x/center_y/size), so there is no reason to fork
that logic and risk the two phases silently diverging in how loss is
computed.
"""

import os
import shutil
import multiprocessing

import header
import torch
from torch.utils.data import DataLoader

import DetNet
from py.training.detection.PhanesimDetectionDataset import (
    PhanesimDetectionDataset, discover_clip_dirs)
from py.training.detection.trainer_detection import (
    train_batch, validate_epoch, set_train_mode, save_checkpoint,
    MAX_EPOCHS, EARLY_STOPPING_PATIENCE)
from py.training.detection.load_weights import load_detnet_weights
import py.training.detection.local_config as local_config
import wandb

modelinputW = header.model_input_width
modelinputH = header.model_input_height

# Every 20th clip (~5%) held out for validation, interleaved across the full
# combined clip list rather than taken from the tail -- dataset_roots is
# [dataset, dataset2], and a plain tail-slice would put nearly all of
# validation inside whichever root happens to be listed last instead of
# spreading it across both.
VAL_CLIP_STRIDE = 20

# Phase-1 checkpoint this phase continues from. Relative to this script's
# own directory, matching trainer_detection.py's own checkpoint_dir
# convention -- not an absolute path, so this still works if the repo is
# ever cloned somewhere else.
PHASE1_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "checkpoints_train_mixed", "checkpoint_best.pth")

CHECKPOINT_DIRNAME = "checkpoints_phanesim_phase2"


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    wandb.init(project="hand_detection_training", job_type="phanesim_phase2")

    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))
    batch_size = 64

    roots = getattr(local_config, "phanesim_dataset_roots", None)
    if not roots:
        raise RuntimeError(
            "[trainer_detection_phanesim] local_config.phanesim_dataset_roots "
            "is not set -- add it to local_config.py before running this.")

    all_clips = discover_clip_dirs(roots)
    if not all_clips:
        raise RuntimeError(
            f"[trainer_detection_phanesim] No usable clips found under {roots}.")

    val_clips = all_clips[::VAL_CLIP_STRIDE]
    val_clip_set = set(val_clips)
    train_clips = [c for c in all_clips if c not in val_clip_set]

    # loadfast is a smoke test: prove phase 2's pipeline runs end to end in
    # minutes -- loading the real phase-1 checkpoint, running a couple of
    # real phanesim clips through training and validation, and writing a
    # real checkpoint -- not produce a model worth keeping. Mirrors
    # trainer_detection.py's own AD4_LOADFAST path; writes to
    # checkpoints_loadfast/ so it can never collide with or be mistaken for
    # a real checkpoints_phanesim_phase2/ run.
    loadfast = bool(int(os.environ.get("AD4_LOADFAST", "0")))
    if loadfast:
        train_clips = train_clips[:2]
        val_clips = val_clips[:1] or train_clips[:1]

    print(f"[trainer_detection_phanesim] {len(train_clips)} train / "
          f"{len(val_clips)} val clips (stride={VAL_CLIP_STRIDE}) "
          f"from {len(all_clips)} total" + (" [LOADFAST]" if loadfast else ""))

    train_dataset = PhanesimDetectionDataset(clip_dirs=train_clips, augment=True)
    val_dataset = PhanesimDetectionDataset(clip_dirs=val_clips, augment=False)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, persistent_workers=num_workers > 0,
        drop_last=False)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, persistent_workers=num_workers > 0,
        drop_last=False)

    if not os.path.exists(PHASE1_CHECKPOINT):
        raise RuntimeError(
            f"[trainer_detection_phanesim] Phase-1 checkpoint not found at "
            f"{PHASE1_CHECKPOINT}. This script continues fine-tuning from the "
            f"existing HOT3D-fine-tuned model -- run trainer_detection.py "
            f"first (or check the path) before running this.")

    model = DetNet.DetNet()
    # load_detnet_weights() must run BEFORE the phase-1 state_dict is loaded
    # below, even though its actual weight VALUES get overwritten immediately
    # after. Why: load_weights.py's _load_conv_bn() dynamically ATTACHES a
    # .bias Parameter to backbone conv layers that InvertedResidual (py/
    # training/common/irb.py) builds with bias=False -- Monado's ONNX
    # baseline export has a bias per conv that this architecture otherwise
    # lacks. A bare DetNet.DetNet() therefore has fewer parameters than the
    # phase-1 checkpoint (saved AFTER phase 1's trainer_detection.py did
    # exactly this), so load_state_dict(strict=True) fails with "Unexpected
    # key(s)" on every dynamically-added bias. Caught by the AD4_LOADFAST
    # smoke test on 2026-09-09 before this ever reached the real job.
    load_detnet_weights(model)
    model = torch.nn.DataParallel(model).to(device)

    phase1 = torch.load(PHASE1_CHECKPOINT, map_location=device, weights_only=False)
    model.module.load_state_dict(phase1['state_dict'])
    print(f"[trainer_detection_phanesim] Loaded phase-1 weights from "
          f"{PHASE1_CHECKPOINT} (phase-1 epoch {phase1.get('epoch')}, "
          f"phase-1 best val loss {phase1.get('best_validation_loss')})")

    # Same backbone-frozen setup as phase 1 -- only the head continues
    # training, keeping the two phases methodologically consistent.
    for param in model.module.backbone.parameters():
        param.requires_grad = False
    model.module.backbone.eval()

    trainable_params = (p for p in model.module.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(trainable_params)

    loss_fn = torch.nn.MSELoss(reduction="mean").to(device)

    # Deliberately NOT resuming epoch/optimizer state from the phase-1
    # checkpoint -- this is a new training phase on a different data source,
    # not a continuation of the same run, so it starts its own epoch count
    # and a fresh optimizer (phase 1's Adam moment estimates were tuned for
    # HOT3D's loss landscape, not phanesim's).
    start_epoch = 0
    best_validation_loss = float('inf')

    checkpoint_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "checkpoints_loadfast" if loadfast else CHECKPOINT_DIRNAME)
    checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.pth')

    # Still allow resuming THIS phase's own training if it gets preempted
    # partway through -- same pattern as trainer_detection.py.
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
        if 'best_validation_loss' in checkpoint:
            best_validation_loss = checkpoint['best_validation_loss']
        start_epoch = checkpoint['epoch']
        model.module.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"[trainer_detection_phanesim] Resuming phase-2 training from "
              f"its own checkpoint at epoch {start_epoch}")

    epochs_without_improvement = 0

    # Hard cap at 2 epochs under loadfast regardless of early stopping --
    # with only 2-3 tiny clips, validation loss could plausibly keep
    # "improving" by noise alone for longer than patience allows, and the
    # whole point of loadfast is a bounded few-minute run.
    effective_max_epochs = 2 if loadfast else MAX_EPOCHS

    for epoch in range(start_epoch, effective_max_epochs):
        print(f"Epoch {epoch}\n---------------------------------------")
        wandb.log({"epoch": epoch})

        set_train_mode(model)
        length = len(train_dataloader)
        for idx, batch in enumerate(train_dataloader):
            print(f"Training {idx}/{length}")
            train_batch(device, batch, loss_fn, optimizer, model)

        val_loss = validate_epoch(device, val_dataloader, loss_fn, model)
        print(f"Epoch {epoch} — val loss: {val_loss:.4f} (best: {best_validation_loss:.4f}; "
              f"{epochs_without_improvement}/{EARLY_STOPPING_PATIENCE} epochs without improvement)")
        wandb.log({"val_loss": val_loss, "best_val_loss": best_validation_loss,
                   "epochs_without_improvement": epochs_without_improvement})

        is_best = val_loss < best_validation_loss
        if is_best:
            best_validation_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        save_checkpoint({
            "epoch": epoch,
            "state_dict": model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_validation_loss": best_validation_loss,
        }, checkpoint_dir)

        if epoch % 10 == 0:
            shutil.copy(
                os.path.join(checkpoint_dir, "checkpoint.pth"),
                os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pth"))

        if is_best:
            shutil.copy(
                os.path.join(checkpoint_dir, "checkpoint.pth"),
                os.path.join(checkpoint_dir, "checkpoint_best.pth"))

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping: validation loss has not improved for "
                  f"{EARLY_STOPPING_PATIENCE} consecutive epochs.")
            break
    else:
        print(f"Reached the epoch ceiling of {effective_max_epochs} without "
              f"early stopping triggering." + (" [LOADFAST]" if loadfast else ""))


if __name__ == "__main__":
    main()
