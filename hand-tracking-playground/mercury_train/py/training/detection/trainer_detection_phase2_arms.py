"""
trainer_detection_phase2_arms.py -- DetNet fine-tuning PHASE 2, corrected
variant. Continues from the phase-1 HOT3D checkpoint
(checkpoints_train_mixed/checkpoint_best.pth) like trainer_detection_phanesim.py
does, but changes the two things that made the plain phase-2 run regress on
HOT3D (mean IoU 0.2117 -> 0.1228 within a single epoch, settling at 0.1149).

Kept as a THIRD script rather than folded into trainer_detection_phanesim.py:
phase 1 and the plain ("naive") phase 2 are both reported results in the thesis,
so neither script nor either checkpoint directory may change. This file is
additive only.

------------------------------------------------------------------------------
WHAT DIFFERS FROM trainer_detection_phanesim.py
------------------------------------------------------------------------------

1. LEARNING RATE. trainer_detection_phanesim.py calls
   torch.optim.Adam(trainable_params) with no lr, i.e. PyTorch's default 1e-3.
   Adam normalises by gradient magnitude, so its early steps move every
   trainable parameter by roughly the learning rate regardless of how small the
   gradient actually is. One Phanesim epoch is ~784 optimiser steps (50,185
   samples / batch 64), so the naive run took ~784 full-size steps away from the
   phase-1 optimum before it measured anything. Phase 1 used the same 1e-3 while
   moving TOWARD the evaluation domain, where a large step helps; in phase 2 the
   identical step size moves away from it. Default 1e-4 here (AD4_LR).

2. VALIDATION SET. The naive run validates on Phanesim clips only, so early
   stopping and checkpoint_best.pth optimise a criterion with no knowledge of
   HOT3D -- even a good phase 2 would be checkpointed badly by it. Every arm
   here validates on HOT3D's val split, so HOT3D drives early stopping and
   checkpoint selection. Phanesim val loss is still measured and logged
   alongside (when the arm uses Phanesim at all) so the stability/plasticity
   trade is visible per epoch, but it gates nothing.

Also: MAX_EPOCHS 30 / patience 5 rather than 120 / 8 (time-boxed against the
submission deadline; with HOT3D validation driving it the run should stop early
by design), and checkpoint_0.pth is saved unconditionally so the epoch-0 point
exists for the same forgetting-trajectory table the naive arm has.

------------------------------------------------------------------------------
THE THREE ARMS (AD4_ARM)
------------------------------------------------------------------------------

Each arm changes only the TRAINING set. Learning rate, validation set, stopping
rule, frozen backbone and batch size are identical across all three, so the arms
differ in exactly one variable and Chapter 6 can attribute the difference.

  AD4_ARM=mixed     (default)  HOT3D train_mixed TRAIN sequences + Phanesim.
                               The actual experiment: does adding synthetic data
                               to the real training set help on real data?
  AD4_ARM=phanesim             Phanesim alone. Isolates the learning rate: this
                               is the naive phase-2 setup with nothing changed
                               except lr and the validation set, so comparing it
                               against the naive numbers says how much of the
                               collapse was step size alone.
  AD4_ARM=hot3d                HOT3D alone. The CONTROL, and the reason it
                               exists: the mixed arm's training set is ~85%
                               HOT3D (see below), so more HOT3D training at a
                               new learning rate could improve the HOT3D score
                               all by itself. Without this arm, an improvement
                               in the mixed arm cannot be attributed to Phanesim
                               rather than to the extra HOT3D epochs.

------------------------------------------------------------------------------
MIXING: WHY THERE IS NO WEIGHTING MECHANISM HERE
------------------------------------------------------------------------------

The mixed arm is a plain ConcatDataset with shuffle=True, so each epoch is one
pass over both datasets and the HOT3D/Phanesim ratio is simply their relative
size. Measured on the cluster 2026-09-09 at frame_stride=5:

    HOT3D detection train   295,466 samples (212 sequences)
    HOT3D detection val      33,680 samples (23 sequences)
    Phanesim detection       50,185 samples (both dataset roots pooled)

which is a 85.5% / 14.5% split. That is already a strong rehearsal signal -- the
majority of every epoch's gradients come from the real data the model is scored
on -- with synthetic data entering at roughly the proportion one would use for
an augmentation source anyway.

An earlier draft of this file re-weighted the two to 50/50 using a repeat
wrapper. That was wrong on its own terms: it would have cut HOT3D's share of
each epoch by about six times relative to the natural ratio, weakening exactly
the anchoring the rehearsal exists to provide. No weighting is applied, no
ratio constant is tuned, and the ratio reported above is the one the run uses.

Everything else -- train_batch, validate_epoch, set_train_mode, save_checkpoint,
the frozen backbone, batch size, the load-weights-before-state_dict ordering --
is reused unchanged from trainer_detection.py / trainer_detection_phanesim.py.
"""

import os
import shutil
import multiprocessing

import header
import torch
from torch.utils.data import ConcatDataset, DataLoader

import DetNet
from py.training.detection.PhanesimDetectionDataset import (
    PhanesimDetectionDataset, discover_clip_dirs)
from py.training.detection.HOT3DVRSDetectionDataset import (
    HOT3DVRSDetectionDataset, worker_init as hot3d_worker_init)
from py.training.common.hot3d_split import list_sequence_dirs, split_train_val
from py.training.detection.trainer_detection import (
    train_batch, validate_epoch, set_train_mode, save_checkpoint,
    HOT3D_FRAME_STRIDE, TRAIN_SPLIT)
from py.training.detection.load_weights import load_detnet_weights
import py.training.detection.local_config as local_config
import wandb

modelinputW = header.model_input_width
modelinputH = header.model_input_height

# Same clip-level Phanesim val stride as trainer_detection_phanesim.py, so the
# Phanesim val curve logged here is measured on the same held-out clips the
# naive arm used and the two are directly comparable.
VAL_CLIP_STRIDE = 20

ARMS = ("mixed", "phanesim", "hot3d")

LEARNING_RATE = float(os.environ.get("AD4_LR", "1e-4"))
MAX_EPOCHS_PHASE2 = int(os.environ.get("AD4_MAX_EPOCHS", "30"))
EARLY_STOPPING_PATIENCE_PHASE2 = int(os.environ.get("AD4_PATIENCE", "5"))

PHASE1_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "checkpoints_train_mixed", "checkpoint_best.pth")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    arm = os.environ.get("AD4_ARM", "mixed").strip().lower()
    if arm not in ARMS:
        raise RuntimeError(
            f"[phase2] AD4_ARM={arm!r} is not one of {ARMS}. Refusing to guess "
            f"-- the arm decides which training set is used and names the "
            f"checkpoint directory, so a typo here would silently produce a "
            f"run that is not the experiment it claims to be.")
    uses_hot3d = arm in ("mixed", "hot3d")
    uses_phanesim = arm in ("mixed", "phanesim")

    loadfast = bool(int(os.environ.get("AD4_LOADFAST", "0")))

    wandb.init(project="hand_detection_training", job_type=f"phase2_{arm}")

    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))
    batch_size = 64

    # ---------------- Phanesim ----------------
    phanesim_train = None
    phanesim_val = None
    if uses_phanesim:
        roots = getattr(local_config, "phanesim_dataset_roots", None)
        if not roots:
            raise RuntimeError(
                f"[phase2] arm={arm} needs Phanesim, but "
                f"local_config.phanesim_dataset_roots is not set.")
        all_clips = discover_clip_dirs(roots)
        if not all_clips:
            raise RuntimeError(f"[phase2] No usable Phanesim clips under {roots}.")
        phanesim_val_clips = all_clips[::VAL_CLIP_STRIDE]
        _val_set = set(phanesim_val_clips)
        phanesim_train_clips = [c for c in all_clips if c not in _val_set]
        if loadfast:
            phanesim_train_clips = phanesim_train_clips[:2]
            phanesim_val_clips = phanesim_val_clips[:1] or phanesim_train_clips[:1]
        print(f"[phase2] Phanesim: {len(phanesim_train_clips)} train / "
              f"{len(phanesim_val_clips)} val clips")

    # ---------------- HOT3D ----------------
    # HOT3D's val split drives early stopping and checkpoint selection in EVERY
    # arm, including the Phanesim-only one, so this is loaded unconditionally.
    #
    # list_sequence_dirs(..., "train_mixed") lists only the train participants,
    # so hot3d_split.TEST_MIXED_PARTICIPANTS -- the sequences eval_detnet.py
    # scores against -- can never reach this training set or this val set.
    train_pool_dirs = list_sequence_dirs(local_config.hot3d_dataset_root, TRAIN_SPLIT)
    if not train_pool_dirs:
        raise RuntimeError(
            f"[phase2] No HOT3D '{TRAIN_SPLIT}' sequences found at "
            f"{local_config.hot3d_dataset_root}. Every arm validates on HOT3D, "
            f"so this is fatal regardless of AD4_ARM.")
    hot3d_train_dirs, hot3d_val_dirs = split_train_val(train_pool_dirs)
    if not hot3d_val_dirs:
        raise RuntimeError(
            "[phase2] HOT3D train pool is too small to carve out a validation "
            "split, and HOT3D validation is what drives early stopping and "
            "checkpoint selection here. Refusing to fall back to Phanesim-only "
            "validation silently -- that is the naive run's bug.")
    if loadfast:
        hot3d_train_dirs = hot3d_train_dirs[:2]
        hot3d_val_dirs = hot3d_val_dirs[:1]

    print(f"[phase2] arm={arm} lr={LEARNING_RATE} "
          f"max_epochs={MAX_EPOCHS_PHASE2} patience={EARLY_STOPPING_PATIENCE_PHASE2}"
          + (" [LOADFAST]" if loadfast else ""))
    print(f"[phase2] HOT3D: {len(hot3d_train_dirs)} train / "
          f"{len(hot3d_val_dirs)} val sequences, frame_stride={HOT3D_FRAME_STRIDE}")

    index_cache_dir = getattr(local_config, "hot3d_index_cache_dir", None)

    def make_hot3d_dataset(sequence_dirs, augment):
        return HOT3DVRSDetectionDataset(
            sequence_dirs=sequence_dirs,
            hot3d_repo_root=local_config.hot3d_repo_root,
            frame_stride=HOT3D_FRAME_STRIDE,
            index_cache_dir=index_cache_dir,
            augment=augment)

    hot3d_val = make_hot3d_dataset(hot3d_val_dirs, augment=False)

    if uses_phanesim:
        phanesim_train = PhanesimDetectionDataset(
            clip_dirs=phanesim_train_clips, augment=True)
        phanesim_val = PhanesimDetectionDataset(
            clip_dirs=phanesim_val_clips, augment=False)

    if arm == "mixed":
        hot3d_train = make_hot3d_dataset(hot3d_train_dirs, augment=True)
        train_dataset = ConcatDataset([hot3d_train, phanesim_train])
        n_h, n_p = len(hot3d_train), len(phanesim_train)
        print(f"[phase2] mixed training set: HOT3D {n_h} + Phanesim {n_p} = "
              f"{n_h + n_p} samples/epoch "
              f"({100.0 * n_h / (n_h + n_p):.1f}% HOT3D, unweighted)")
    elif arm == "hot3d":
        train_dataset = make_hot3d_dataset(hot3d_train_dirs, augment=True)
        print(f"[phase2] HOT3D-only control training set: "
              f"{len(train_dataset)} samples/epoch")
    else:
        train_dataset = phanesim_train
        print(f"[phase2] Phanesim-only training set: "
              f"{len(train_dataset)} samples/epoch")

    # worker_init_fn is required whenever HOT3D samples are in a loader:
    # DataLoader workers are forked, and without it they would inherit the
    # parent's already-open VRS handles. It only clears each worker's provider
    # cache, so it is harmless for Phanesim samples and every loader here
    # carries it unconditionally rather than making its presence depend on the
    # arm.
    def make_loader(dataset, shuffle):
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, worker_init_fn=hot3d_worker_init,
            persistent_workers=num_workers > 0, drop_last=False)

    train_dataloader = make_loader(train_dataset, shuffle=True)
    hot3d_val_dataloader = make_loader(hot3d_val, shuffle=False)
    phanesim_val_dataloader = (make_loader(phanesim_val, shuffle=False)
                               if uses_phanesim else None)

    # ---------------- Model ----------------
    if not os.path.exists(PHASE1_CHECKPOINT):
        raise RuntimeError(
            f"[phase2] Phase-1 checkpoint not found at {PHASE1_CHECKPOINT}. "
            f"Every arm continues from it -- run trainer_detection.py first.")

    model = DetNet.DetNet()
    # Must run BEFORE the phase-1 state_dict load: load_weights.py's
    # _load_conv_bn() dynamically ATTACHES a .bias Parameter to backbone conv
    # layers that InvertedResidual builds with bias=False, so a bare
    # DetNet.DetNet() has fewer parameters than the phase-1 checkpoint and
    # load_state_dict(strict=True) would fail on every added bias. Same
    # ordering as trainer_detection_phanesim.py.
    load_detnet_weights(model)
    model = torch.nn.DataParallel(model).to(device)

    phase1 = torch.load(PHASE1_CHECKPOINT, map_location=device, weights_only=False)
    model.module.load_state_dict(phase1['state_dict'])
    print(f"[phase2] Loaded phase-1 weights (epoch {phase1.get('epoch')}, "
          f"best val loss {phase1.get('best_validation_loss')})")

    # Same backbone-frozen setup as phases 1 and 2.
    for param in model.module.backbone.parameters():
        param.requires_grad = False
    model.module.backbone.eval()

    trainable_params = (p for p in model.module.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(trainable_params, lr=LEARNING_RATE)

    loss_fn = torch.nn.MSELoss(reduction="mean").to(device)

    # Deliberately NOT resuming epoch/optimizer state from the phase-1
    # checkpoint -- this is a new training phase, not a continuation of the same
    # run, and phase 1's Adam moment estimates were accumulated at a different
    # learning rate. Same reasoning as trainer_detection_phanesim.py.
    start_epoch = 0
    best_validation_loss = float('inf')

    # Named by arm so three concurrent arms can never collide, and so a resume
    # can never pick up a checkpoint produced by a different training set.
    dirname = (f"checkpoints_loadfast_phase2_{arm}" if loadfast
               else f"checkpoints_phase2_{arm}")
    checkpoint_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), dirname)
    checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.pth')

    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
        if 'best_validation_loss' in checkpoint:
            best_validation_loss = checkpoint['best_validation_loss']
        start_epoch = checkpoint['epoch']
        model.module.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"[phase2] Resuming arm={arm} from its own checkpoint at "
              f"epoch {start_epoch}")

    epochs_without_improvement = 0
    effective_max_epochs = 2 if loadfast else MAX_EPOCHS_PHASE2

    for epoch in range(start_epoch, effective_max_epochs):
        print(f"Epoch {epoch}\n---------------------------------------")
        wandb.log({"epoch": epoch})

        set_train_mode(model)
        length = len(train_dataloader)
        for idx, batch in enumerate(train_dataloader):
            print(f"Training {idx}/{length}")
            train_batch(device, batch, loss_fn, optimizer, model)

        # HOT3D is the selection criterion in every arm. Phanesim val is
        # measured only so the trade-off is visible per epoch; it gates nothing.
        val_loss = validate_epoch(device, hot3d_val_dataloader, loss_fn, model)
        logged = {"val_loss_hot3d": val_loss,
                  "best_val_loss_hot3d": best_validation_loss,
                  "epochs_without_improvement": epochs_without_improvement}
        phanesim_msg = ""
        if phanesim_val_dataloader is not None:
            phanesim_val_loss = validate_epoch(
                device, phanesim_val_dataloader, loss_fn, model)
            logged["val_loss_phanesim"] = phanesim_val_loss
            phanesim_msg = f" | Phanesim val loss: {phanesim_val_loss:.4f}"

        print(f"Epoch {epoch} — HOT3D val loss: {val_loss:.4f} "
              f"(best: {best_validation_loss:.4f}; {epochs_without_improvement}/"
              f"{EARLY_STOPPING_PATIENCE_PHASE2} without improvement)"
              + phanesim_msg)
        wandb.log(logged)

        is_best = val_loss < best_validation_loss
        if is_best:
            best_validation_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        state = {
            "epoch": epoch,
            "arm": arm,
            "learning_rate": LEARNING_RATE,
            "state_dict": model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_validation_loss": best_validation_loss,
            "val_loss_hot3d": val_loss,
        }
        if phanesim_val_dataloader is not None:
            state["val_loss_phanesim"] = logged["val_loss_phanesim"]
        save_checkpoint(state, checkpoint_dir)

        # Unconditional at epoch 0 (not only every 10th) so the epoch-0 point
        # exists for the forgetting-trajectory table even if the run is cut
        # short. epoch % 10 keeps the naive arm's sweep granularity.
        if epoch == 0 or epoch % 10 == 0:
            shutil.copy(os.path.join(checkpoint_dir, "checkpoint.pth"),
                        os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pth"))

        if is_best:
            shutil.copy(os.path.join(checkpoint_dir, "checkpoint.pth"),
                        os.path.join(checkpoint_dir, "checkpoint_best.pth"))

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE_PHASE2:
            print(f"Early stopping: HOT3D validation loss has not improved for "
                  f"{EARLY_STOPPING_PATIENCE_PHASE2} consecutive epochs.")
            break


if __name__ == '__main__':
    main()
