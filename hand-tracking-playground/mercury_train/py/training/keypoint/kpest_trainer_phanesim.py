"""
kpest_trainer_phanesim.py -- KeyNet fine-tuning PHASE 2: continues training
from the phase-1 HOT3D fine-tuned checkpoint (checkpoints_train_mixed/
checkpoint_best.pth) using Phanesim's synthetic data instead of HOT3D.
Mirrors trainer_detection_phanesim.py's own rationale (see that file):
kept separate from kpest_trainer.py so phase 1 stays reproducible
unchanged and can never be overwritten by this phase's checkpoints, and
should be run as a time-boxed TRIAL first -- re-run eval_keynet.py against
test_mixed on this phase's checkpoint_best.pth and compare against the
phase-1 numbers before committing to a full run. Same catastrophic-
forgetting risk applies (Phanesim only, no HOT3D mixed back in, both
`dataset` and `dataset2` pooled -- 2026-09-08 decision).

Depth is invalid for every Phanesim sample (see PhanesimKeypointDataset.py)
-- settings.py already sets depth_loss_mul/elbow_loss_mul/curls_loss_mul so
that only has_depth (never separately toggled here) gates those terms, and
existence_loss_mul is already 0, matching phase 1's own scope decision. So
this phase's loss is effectively xy-only, same as phase 1's HOT3D loss
already effectively was, for the same reasons.

Reuses train_loop/save_checkpoint/set_train_mode/MAX_EPOCHS/
EARLY_STOPPING_PATIENCE from kpest_trainer.py, and validatoor.validation_loop
directly, unchanged -- none of them are HOT3D-specific, and
PhanesimKeypointDataset's __getitem__ produces the exact same batch dict
shape kpest_trainer.py's train_loop already expects, so there is no reason
to fork that logic and risk the two phases silently diverging in how loss
is computed.
"""

import multiprocessing
import os
import shutil
import sys

_mercury_train_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../')
if _mercury_train_root not in sys.path:
    sys.path.insert(0, _mercury_train_root)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

import local_config
import KeyNet
import kpest_header as header
import validatoor
from load_weights import load_keynet_weights
from kpest_trainer import (
    train_loop, save_checkpoint, set_train_mode,
    MAX_EPOCHS, EARLY_STOPPING_PATIENCE)
from PhanesimKeypointDataset import PhanesimKeypointDataset, discover_clip_dirs

mse = nn.MSELoss(reduction='mean')

# Every 20th clip (~5%) held out for validation, interleaved across the full
# combined clip list -- same reasoning and stride as
# trainer_detection_phanesim.py's VAL_CLIP_STRIDE (dataset_roots is
# [dataset, dataset2]; a tail-slice would put nearly all validation inside
# whichever root is listed last).
VAL_CLIP_STRIDE = 20

# Phase-1 checkpoint this phase continues from. Relative to this script's
# own directory, matching kpest_trainer.py's own checkpoint_dir convention
# (checkpoints_{TRAIN_SPLIT}, TRAIN_SPLIT="train_mixed").
PHASE1_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "checkpoints_train_mixed", "checkpoint_best.pth")

CHECKPOINT_DIRNAME = "checkpoints_phanesim_phase2"


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_devices = 1
    batch_size_per_device = 64  # Same as kpest_trainer.py's own -- identical
                                 # architecture/crop size, same OOM ceiling.
    if device.type == "cuda":
        num_devices = torch.cuda.device_count()
        print(f"Let's use {num_devices} GPUs!")
    batch_size = batch_size_per_device * num_devices

    if header.env_settings.wandb_enabled:
        wandb.init(project="keypoint_estimator_training", job_type="phanesim_phase2")
    else:
        wandb.init(project="keypoint_estimator_training", mode="disabled")

    # PhanesimKeypointDataset holds no live providers (plain CSV + PNG
    # reads, unlike HOT3DKeypointDataset's VRS providers), so unlike
    # kpest_trainer.py's num_workers cap of 4 -- which exists specifically
    # because of VRS file-handle sharing across forked workers -- there is
    # no equivalent concern here. Uncapped, matching
    # trainer_detection_phanesim.py's own num_workers for the same reason.
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))

    roots = getattr(local_config, "phanesim_dataset_roots", None)
    if not roots:
        raise RuntimeError(
            "[kpest_trainer_phanesim] local_config.phanesim_dataset_roots is "
            "not set -- add it to local_config.py before running this.")

    all_clips = discover_clip_dirs(roots)
    if not all_clips:
        raise RuntimeError(
            f"[kpest_trainer_phanesim] No usable clips found under {roots}.")

    val_clips = all_clips[::VAL_CLIP_STRIDE]
    val_clip_set = set(val_clips)
    train_clips = [c for c in all_clips if c not in val_clip_set]

    # loadfast is a smoke test -- mirrors kpest_trainer.py's own loadfast
    # slicing (train_dirs[:2] / val_dirs[:1]) and
    # trainer_detection_phanesim.py's AD4_LOADFAST path.
    loadfast = header.env_settings.loadfast
    if loadfast:
        train_clips = train_clips[:2]
        val_clips = val_clips[:1] or train_clips[:1]

    print(f"[kpest_trainer_phanesim] {len(train_clips)} train / "
          f"{len(val_clips)} val clips (stride={VAL_CLIP_STRIDE}) from "
          f"{len(all_clips)} total" + (" [LOADFAST]" if loadfast else ""))

    # eval_mode=False for BOTH loaders -- matches kpest_trainer.py's own
    # make_hot3d_loader, which never passes eval_mode=True either. eval_mode
    # is only for eval_keynet.py's standalone deterministic evaluation, not
    # for validation during training.
    train_dataset = PhanesimKeypointDataset(clip_dirs=train_clips, eval_mode=False)
    val_dataset = PhanesimKeypointDataset(clip_dirs=val_clips, eval_mode=False)

    dataloader_train = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, persistent_workers=num_workers > 0,
        drop_last=True)
    dataloader_val = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, persistent_workers=num_workers > 0,
        drop_last=False)

    if not os.path.exists(PHASE1_CHECKPOINT):
        raise RuntimeError(
            f"[kpest_trainer_phanesim] Phase-1 checkpoint not found at "
            f"{PHASE1_CHECKPOINT}. This script continues fine-tuning from "
            f"the existing HOT3D-fine-tuned model -- run kpest_trainer.py "
            f"first (or check the path) before running this.")

    model = KeyNet.KeyNet()
    # load_keynet_weights() must run BEFORE the phase-1 state_dict is loaded
    # below, even though its weight VALUES get overwritten immediately
    # after -- same reason as trainer_detection_phanesim.py's
    # load_detnet_weights() call: load_weights.py's load_conv_bn()
    # dynamically ATTACHES .bias Parameters to image_network/fused_network/
    # network_2d_px_coord conv layers that InvertedResidual builds with
    # bias=False. A bare KeyNet.KeyNet() therefore has fewer parameters
    # than the phase-1 checkpoint (saved after phase 1's kpest_trainer.py
    # did exactly this), so load_state_dict(strict=True) would fail with
    # "Unexpected key(s)" on every dynamically-added bias -- this exact bug
    # was caught and fixed for DetNet's phase 2 on 2026-09-09 before it
    # reached a real job; applying that fix here up front rather than
    # rediscovering it.
    load_keynet_weights(model)

    # Same image_network-frozen setup as phase 1 -- only the rest of the
    # network continues training, keeping the two phases methodologically
    # consistent. Frozen BEFORE DataParallel wrapping, matching
    # kpest_trainer.py's own ordering.
    for param in model.image_network.parameters():
        param.requires_grad = False

    model = torch.nn.DataParallel(model).to(device)
    model.module.image_network.eval()

    phase1 = torch.load(PHASE1_CHECKPOINT, map_location=device, weights_only=False)
    model.module.load_state_dict(phase1['state_dict'])
    print(f"[kpest_trainer_phanesim] Loaded phase-1 weights from "
          f"{PHASE1_CHECKPOINT} (phase-1 epoch {phase1.get('epoch')}, "
          f"phase-1 best val loss {phase1.get('best_validation_loss')})")

    trainable_params = (p for p in model.module.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(trainable_params)

    # Deliberately NOT resuming epoch/optimizer state from the phase-1
    # checkpoint -- this is a new training phase on a different data
    # source, not a continuation of the same run. Same as
    # trainer_detection_phanesim.py.
    start_epoch = 0
    best_validation_loss = float('inf')

    # A distinct name from kpest_trainer.py's own "checkpoints_loadfast" --
    # NOT cosmetic. This exact bug (silent stale-checkpoint resume, zero
    # real training steps, still logging a success-looking message) already
    # happened once for DetNet's phase 2 and is called out by name in
    # kpest_trainer.py's own checkpoint_dir comment as a known prior
    # incident for THIS file specifically. Applying the fix up front here.
    checkpoint_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "checkpoints_loadfast_phanesim" if loadfast else CHECKPOINT_DIRNAME)
    checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.pth')

    # Still allow resuming THIS phase's own training if it gets preempted
    # partway through -- same pattern as kpest_trainer.py.
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
        if 'best_validation_loss' in checkpoint:
            best_validation_loss = checkpoint['best_validation_loss']
        start_epoch = checkpoint['epoch']
        model.module.load_state_dict(checkpoint['state_dict'])
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except BaseException:
            print("Couldn't load optimizer state dict! This shouldn't happen "
                  "except for right after model weight transfers!")
        print(f"[kpest_trainer_phanesim] Resuming phase-2 training from its "
              f"own checkpoint at epoch {start_epoch}")

    epochs_without_improvement = 0

    # Hard cap at 2 epochs under loadfast regardless of early stopping --
    # same reasoning as trainer_detection_phanesim.py's effective_max_epochs:
    # with only 2-3 tiny clips, validation loss could plausibly keep
    # "improving" by noise alone for longer than EARLY_STOPPING_PATIENCE=8
    # allows, and the whole point of loadfast is a bounded few-minute run.
    effective_max_epochs = 2 if loadfast else MAX_EPOCHS

    for epoch in range(start_epoch, effective_max_epochs):
        print(f"Epoch {epoch}\n---------------------------------------")
        wandb.log({"epoch": epoch})
        set_train_mode(model)
        mean_training_loss, mean_training_loss_xy = train_loop(
            device, dataloader_train, model, optimizer)

        model.eval()
        val_result = validatoor.validation_loop(
            device, dataloader_val, model, mse, "val", False, epoch)
        mean_validation_loss = val_result.mean_loss_no_pred
        set_train_mode(model)

        is_best = mean_validation_loss < best_validation_loss
        if is_best:
            best_validation_loss = mean_validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(f"Done with epoch {epoch} -- train loss: {mean_training_loss:.4f} "
              f"(xy {mean_training_loss_xy:.4f}), "
              f"val loss: {mean_validation_loss:.4f} "
              f"(best: {best_validation_loss:.4f}; "
              f"{epochs_without_improvement}/{EARLY_STOPPING_PATIENCE} epochs "
              f"without improvement)")
        wandb.log({
            "train_loss": mean_training_loss,
            "train_loss_xy": mean_training_loss_xy,
            "val_loss": mean_validation_loss,
            "best_val_loss": best_validation_loss,
            "epochs_without_improvement": epochs_without_improvement,
        })

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_validation_loss': float(best_validation_loss),
        }, checkpoint_dir)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, saving extra checkpoint!")
            shutil.copy(
                os.path.join(checkpoint_dir, "checkpoint.pth"),
                os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pth"))

        if is_best:
            print("Best model so far! Saving as checkpoint_best.pth")
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

    best_checkpoint = os.path.join(checkpoint_dir, "checkpoint_best.pth")
    print(f"\nTraining complete. Best validation loss: {best_validation_loss:.4f}")
    print(f"Best checkpoint: {best_checkpoint}")
    print("Score it against the held-out split with, e.g.:")
    print(f"  python py/evaluation/eval_keynet.py "
          f"--weights {best_checkpoint} --split test_mixed")
    wandb.log({"final_best_val_loss": best_validation_loss})


if __name__ == "__main__":
    main()
