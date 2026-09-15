"""
DetNet fine-tuning phase 2: continues from the phase-1 HOT3D checkpoint on
Phanesim's synthetic data. Separate from trainer_detection.py so phase 1 stays
reproducible and its checkpoint cannot be overwritten. Phanesim only, no HOT3D
mixed in, which carries a real catastrophic-forgetting risk: run it time-boxed and
compare eval_detnet.py on test_mixed before committing to a full run.
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

# Every 20th clip held out, interleaved: a tail slice would land in one root only.
VAL_CLIP_STRIDE = 20

# Phase-1 checkpoint to continue from, relative to this script's directory.
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

    # loadfast is a smoke test, not a model worth keeping; writes to its own dir.
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
    # load_detnet_weights() first: it attaches the .bias params a strict load needs.
    load_detnet_weights(model)
    model = torch.nn.DataParallel(model).to(device)

    phase1 = torch.load(PHASE1_CHECKPOINT, map_location=device, weights_only=False)
    model.module.load_state_dict(phase1['state_dict'])
    print(f"[trainer_detection_phanesim] Loaded phase-1 weights from "
          f"{PHASE1_CHECKPOINT} (phase-1 epoch {phase1.get('epoch')}, "
          f"phase-1 best val loss {phase1.get('best_validation_loss')})")

    # Same backbone-frozen setup as phase 1; only the head continues training.
    for param in model.module.backbone.parameters():
        param.requires_grad = False
    model.module.backbone.eval()

    trainable_params = (p for p in model.module.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(trainable_params)

    loss_fn = torch.nn.MSELoss(reduction="mean").to(device)

    # Deliberately not resuming epoch/optimizer state: a new phase, not a continuation.
    start_epoch = 0
    best_validation_loss = float('inf')

    # Distinct from trainer_detection.py's "checkpoints_loadfast"; sharing it caused a stale resume.
    checkpoint_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "checkpoints_loadfast_phanesim" if loadfast else CHECKPOINT_DIRNAME)
    checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.pth')

    # Still allow resuming this phase if it gets preempted partway through.
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

    # Hard cap at 2 epochs under loadfast regardless of early stopping,
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
