"""
kpest_trainer_finetune_arms.py -- KeyNet fine-tuning with a selectable starting
point and a selectable training set. The KeyNet counterpart of
py/training/detection/trainer_detection_finetune_arms.py; the two are meant to
read identically and that file's docstring carries the shared reasoning.

Additive only. kpest_trainer.py (phase 1, HOT3D) and kpest_trainer_phanesim.py
(the naive phase 2) are both reported results and are not touched by this file;
it never writes to their checkpoint directories.

AD4_INIT   default (default value) -- Monado Mercury's shipped ONNX weights,
                                      exactly as kpest_trainer.py starts.
           phase1                  -- continue from checkpoints_train_mixed/
                                      checkpoint_best.pth.
AD4_ARM    mixed | phanesim | hot3d -- the training set.
AD4_SELECT hot3d | phanesim         -- which validation loss drives early
                                      stopping and checkpoint_best. Defaults to
                                      phanesim when AD4_INIT=default, hot3d
                                      otherwise.

Both validation losses are measured and logged every epoch regardless of which
one selects, and both checkpoint_best.pth (by AD4_SELECT) and
checkpoint_best_hot3d.pth are written, so either selection can be reported
without a second run.

------------------------------------------------------------------------------
OPTIMISER -- and why AD4_INIT decides it
------------------------------------------------------------------------------
With AD4_INIT=default this run is a PEER of phase 1: same starting weights, same
frozen image_network, different training set. It must therefore use phase 1's
optimiser settings unchanged, or the comparison measures the optimiser as well
as the data. kpest_trainer.py calls a bare torch.optim.AdamW(trainable_params),
so this case does too -- including PyTorch's default weight_decay=0.01, which is
kept deliberately BECAUSE phase 1 had it, not because it is a good idea in
isolation. MAX_EPOCHS / EARLY_STOPPING_PATIENCE come from kpest_trainer.py
(120 / 8) for the same reason.

With AD4_INIT=phase1 the run continues from an already-converged model, and the
bare AdamW() defaults are two separate mechanisms that pull it away from what it
had learned:
  - lr=1e-3: AdamW normalises by gradient magnitude, so early steps move every
    trainable parameter by roughly the learning rate regardless of the
    gradient's actual size. Phase 1 used the same value while moving TOWARD the
    evaluation domain, where a large step helps; continuing from convergence the
    identical step size moves away from it.
  - weight_decay=0.01: decays the phase-1 weights toward zero on every step,
    independently of anything the training data's gradient says -- a forgetting
    channel with no connection to the domain shift at all.
This case therefore defaults to lr=1e-4, weight_decay=0.0, and a time-boxed
30 / 5. AD4_LR, AD4_WEIGHT_DECAY, AD4_MAX_EPOCHS and AD4_PATIENCE override
whichever case is active.

------------------------------------------------------------------------------
NOTES
------------------------------------------------------------------------------
Preprocessing is unchanged and is not a function of AD4_INIT: the input
convention belongs to the weights, not to the data, and is defined once in
py/evaluation/preprocess_baseline.py. Both dataset loaders already reproduce it
through the same calls (_pp.keynet_crop_matrix + augmaker.do_one_augmentation,
including the sRGB EOTF).

HOT3D frame stride stays at kpest_trainer.HOT3D_FRAME_STRIDE (5) for training
and validation alike. Raising it for the training stream was considered as a way
to halve the dominant cost and rejected: HOT3DKeypointDataset's index cache key
includes frame_stride (see its _cache_path), the cluster's cache holds only
stride-5 entries, and any other stride triggers a full 294-sequence .vrs index
rebuild costing more than the epoch time it saves.

Measured epoch cost on this cluster (2026-09-09, from checkpoint timestamps):
KeyNet HOT3D epochs ~50-60 min, Phanesim epochs ~3 min. Arms that touch HOT3D
are therefore roughly an hour per epoch; the phanesim arm is minutes.

Checkpoints go to checkpoints_monado_<arm>/ (AD4_INIT=default) or
checkpoints_phase2_<arm>/ (AD4_INIT=phase1).
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
from torch.utils.data import ConcatDataset, DataLoader
import wandb

import local_config
import py.training.common.hot3d_split as hot3d_split
import KeyNet
import kpest_header as header
import validatoor
from load_weights import load_keynet_weights
from kpest_trainer import (
    train_loop, save_checkpoint, set_train_mode,
    HOT3D_FRAME_STRIDE, TRAIN_SPLIT, MAX_EPOCHS, EARLY_STOPPING_PATIENCE)
from HOT3DKeypointDataset import HOT3DKeypointDataset, worker_init as hot3d_worker_init
from PhanesimKeypointDataset import PhanesimKeypointDataset, discover_clip_dirs

mse = nn.MSELoss(reduction='mean')

# Same clip-level Phanesim val stride as kpest_trainer_phanesim.py, so the
# Phanesim val curve here is measured on the same held-out clips the naive arm
# used and the two are directly comparable.
VAL_CLIP_STRIDE = 20

ARMS = ("mixed", "phanesim", "hot3d")
INITS = ("default", "phase1")
SELECTS = ("hot3d", "phanesim")

PHASE1_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "checkpoints_train_mixed", "checkpoint_best.pth")


def _env_choice(name, allowed, fallback):
    value = os.environ.get(name, fallback).strip().lower()
    if value not in allowed:
        raise RuntimeError(
            f"[finetune] {name}={value!r} is not one of {allowed}. Refusing to "
            f"guess -- these settings decide which experiment this run is and "
            f"name its checkpoint directory, so a typo would silently produce a "
            f"run that is not the one it claims to be.")
    return value


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_devices = 1
    batch_size_per_device = 64  # Same as kpest_trainer.py's own.
    if device.type == "cuda":
        num_devices = torch.cuda.device_count()
        print(f"Let's use {num_devices} GPUs!")
    batch_size = batch_size_per_device * num_devices

    arm = _env_choice("AD4_ARM", ARMS, "phanesim")
    init = _env_choice("AD4_INIT", INITS, "default")
    select = _env_choice("AD4_SELECT", SELECTS,
                         "phanesim" if init == "default" else "hot3d")

    uses_phanesim = arm in ("mixed", "phanesim")
    if select == "phanesim" and not uses_phanesim:
        raise RuntimeError(
            f"[finetune] AD4_SELECT=phanesim with AD4_ARM={arm} has no Phanesim "
            f"data to select on.")

    if init == "default":
        max_epochs = int(os.environ.get("AD4_MAX_EPOCHS", str(MAX_EPOCHS)))
        patience = int(os.environ.get("AD4_PATIENCE", str(EARLY_STOPPING_PATIENCE)))
    else:
        max_epochs = int(os.environ.get("AD4_MAX_EPOCHS", "30"))
        patience = int(os.environ.get("AD4_PATIENCE", "5"))

    loadfast = header.env_settings.loadfast

    if header.env_settings.wandb_enabled:
        wandb.init(project="keypoint_estimator_training", job_type=f"{init}_{arm}")
    else:
        wandb.init(project="keypoint_estimator_training", mode="disabled")

    # Capped at 4 because HOT3D samples are in play in every configuration (its
    # val loss is measured even when it is not trained on): HOT3DKeypointDataset
    # holds live VRS providers and DataLoader workers are forked. This is
    # kpest_trainer.py's own cap and its reason applies here.
    # kpest_trainer_phanesim.py could safely leave it uncapped only because
    # every loader it built was Phanesim-only.
    num_workers = min(4, int(os.environ.get("SLURM_CPUS_PER_TASK",
                                            multiprocessing.cpu_count())))

    # ---------------- Phanesim ----------------
    if uses_phanesim:
        roots = getattr(local_config, "phanesim_dataset_roots", None)
        if not roots:
            raise RuntimeError(
                f"[finetune] arm={arm} needs Phanesim, but "
                f"local_config.phanesim_dataset_roots is not set.")
        all_clips = discover_clip_dirs(roots)
        if not all_clips:
            raise RuntimeError(f"[finetune] No usable Phanesim clips under {roots}.")
        phanesim_val_clips = all_clips[::VAL_CLIP_STRIDE]
        _val_set = set(phanesim_val_clips)
        phanesim_train_clips = [c for c in all_clips if c not in _val_set]
        if loadfast:
            phanesim_train_clips = phanesim_train_clips[:2]
            phanesim_val_clips = phanesim_val_clips[:1] or phanesim_train_clips[:1]
        print(f"[finetune] Phanesim: {len(phanesim_train_clips)} train / "
              f"{len(phanesim_val_clips)} val clips")

    # ---------------- HOT3D ----------------
    # Loaded in every configuration: even when HOT3D is not trained on and not
    # selected on, its val loss is measured each epoch so the run records the
    # real-domain trajectory the thesis reports.
    #
    # list_sequence_dirs(..., "train_mixed") lists only the train participants,
    # so hot3d_split.TEST_MIXED_PARTICIPANTS -- the sequences eval_keynet.py
    # scores against -- can never reach this training set or this val set.
    train_pool = hot3d_split.list_sequence_dirs(
        local_config.hot3d_dataset_root, TRAIN_SPLIT)
    if not train_pool:
        raise RuntimeError(
            f"[finetune] No HOT3D '{TRAIN_SPLIT}' sequences found under "
            f"{local_config.hot3d_dataset_root}. HOT3D validation is measured in "
            f"every configuration, so this is fatal regardless of AD4_ARM.")
    hot3d_train_dirs, hot3d_val_dirs = hot3d_split.split_train_val(train_pool)
    if not hot3d_val_dirs:
        raise RuntimeError(
            "[finetune] HOT3D train pool too small to carve out a validation split.")
    if loadfast:
        hot3d_train_dirs = hot3d_train_dirs[:2]
        hot3d_val_dirs = hot3d_val_dirs[:1]

    print(f"[finetune] init={init} arm={arm} select={select} "
          f"max_epochs={max_epochs} patience={patience}"
          + (" [LOADFAST]" if loadfast else ""))
    print(f"[finetune] HOT3D: {len(hot3d_train_dirs)} train / "
          f"{len(hot3d_val_dirs)} val sequences, frame_stride={HOT3D_FRAME_STRIDE}")

    def make_hot3d_dataset(sequence_dirs):
        return HOT3DKeypointDataset(
            sequence_dirs=sequence_dirs,
            hot3d_repo_root=local_config.hot3d_repo_root,
            object_library_path=local_config.hot3d_object_library_path,
            frame_stride=HOT3D_FRAME_STRIDE,
            index_cache_dir=getattr(local_config, "hot3d_index_cache_dir", None),
        )

    hot3d_val = make_hot3d_dataset(hot3d_val_dirs)

    # eval_mode=False throughout, matching kpest_trainer.py's own
    # make_hot3d_loader and kpest_trainer_phanesim.py -- eval_mode is only for
    # eval_keynet.py's standalone deterministic evaluation.
    phanesim_train = phanesim_val = None
    if uses_phanesim:
        phanesim_train = PhanesimKeypointDataset(
            clip_dirs=phanesim_train_clips, eval_mode=False)
        phanesim_val = PhanesimKeypointDataset(
            clip_dirs=phanesim_val_clips, eval_mode=False)

    if arm == "mixed":
        hot3d_train = make_hot3d_dataset(hot3d_train_dirs)
        train_dataset = ConcatDataset([hot3d_train, phanesim_train])
        n_h, n_p = len(hot3d_train), len(phanesim_train)
        print(f"[finetune] mixed training set: HOT3D {n_h} + Phanesim {n_p} = "
              f"{n_h + n_p} samples/epoch "
              f"({100.0 * n_h / (n_h + n_p):.1f}% HOT3D, unweighted)")
    elif arm == "hot3d":
        train_dataset = make_hot3d_dataset(hot3d_train_dirs)
        print(f"[finetune] HOT3D-only training set: {len(train_dataset)} samples/epoch")
    else:
        train_dataset = phanesim_train
        print(f"[finetune] Phanesim-only training set: "
              f"{len(train_dataset)} samples/epoch")

    # worker_init_fn is required whenever HOT3D samples are in a loader (forked
    # workers would otherwise share the parent's open VRS handles). It only
    # clears provider caches, so it is harmless for Phanesim samples and every
    # loader carries it unconditionally.
    #
    # drop_last=False everywhere, NOT kpest_trainer.py's drop_last=True for its
    # train loader: under loadfast the whole train set can be smaller than one
    # batch, which would leave train_loop with zero iterations and
    # `total_loss / loss_divisor` a division by zero. Same trap
    # kpest_trainer_phanesim.py already avoids the same way (job 1700699).
    # Harmless for a real run -- at most one short batch per epoch.
    def make_loader(dataset, shuffle):
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, worker_init_fn=hot3d_worker_init,
            timeout=0, persistent_workers=num_workers > 0, drop_last=False)

    dataloader_train = make_loader(train_dataset, shuffle=True)
    dataloader_val_hot3d = make_loader(hot3d_val, shuffle=False)
    dataloader_val_phanesim = (make_loader(phanesim_val, shuffle=False)
                               if uses_phanesim else None)

    # ---------------- Model ----------------
    model = KeyNet.KeyNet()
    # load_keynet_weights() is what loads Monado's shipped ONNX weights, and it
    # runs in BOTH cases. With init=phase1 its weight VALUES are overwritten
    # below, but the call is still required first: load_weights.py's
    # load_conv_bn() dynamically attaches .bias Parameters to image_network/
    # fused_network/network_2d_px_coord conv layers that InvertedResidual builds
    # with bias=False, so a bare KeyNet.KeyNet() has fewer parameters than any
    # checkpoint saved after this call and a strict load_state_dict would fail
    # on every one of them.
    load_keynet_weights(model)

    # Frozen BEFORE DataParallel wrapping, matching kpest_trainer.py's ordering.
    # Frozen in every configuration, matching phase 1 and the naive phase 2.
    for param in model.image_network.parameters():
        param.requires_grad = False

    model = torch.nn.DataParallel(model).to(device)
    model.module.image_network.eval()

    if init == "phase1":
        if not os.path.exists(PHASE1_CHECKPOINT):
            raise RuntimeError(
                f"[finetune] AD4_INIT=phase1 but no checkpoint at "
                f"{PHASE1_CHECKPOINT}. Run kpest_trainer.py first.")
        phase1 = torch.load(PHASE1_CHECKPOINT, map_location=device, weights_only=False)
        model.module.load_state_dict(phase1['state_dict'])
        print(f"[finetune] Loaded phase-1 weights (epoch {phase1.get('epoch')}, "
              f"best val loss {phase1.get('best_validation_loss')})")
    else:
        print("[finetune] Starting from Monado Mercury's shipped ONNX weights "
              "(no checkpoint loaded)")

    trainable_params = (p for p in model.module.parameters() if p.requires_grad)
    if "AD4_LR" in os.environ or "AD4_WEIGHT_DECAY" in os.environ:
        lr = float(os.environ.get("AD4_LR", "1e-3"))
        wd = float(os.environ.get("AD4_WEIGHT_DECAY", "0.01"))
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=wd)
        print(f"[finetune] AdamW(lr={lr}, weight_decay={wd}) -- explicit override")
    elif init == "default":
        # Bare AdamW(), byte-for-byte what kpest_trainer.py does, weight decay
        # included. Matching phase 1 exactly is the point of this configuration.
        optimizer = torch.optim.AdamW(trainable_params)
        print("[finetune] AdamW() with PyTorch defaults (lr=1e-3, "
              "weight_decay=0.01) -- matches kpest_trainer.py, so this run "
              "differs from phase 1 only in its training set")
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.0)
        print("[finetune] AdamW(lr=1e-4, weight_decay=0.0) -- both defaults "
              "reduced; each was an independent forgetting channel in the naive "
              "phase 2")

    start_epoch = 0
    best_selection_loss = float('inf')
    best_hot3d_loss = float('inf')

    prefix = "checkpoints_monado" if init == "default" else "checkpoints_phase2"
    dirname = f"{'checkpoints_loadfast_' + init if loadfast else prefix}_{arm}"
    checkpoint_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), dirname)
    checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.pth')

    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
        best_selection_loss = checkpoint.get('best_validation_loss', best_selection_loss)
        best_hot3d_loss = checkpoint.get('best_hot3d_loss', best_hot3d_loss)
        start_epoch = checkpoint['epoch']
        model.module.load_state_dict(checkpoint['state_dict'])
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except BaseException:
            print("Couldn't load optimizer state dict! This shouldn't happen "
                  "except for right after model weight transfers!")
        print(f"[finetune] Resuming init={init} arm={arm} from its own "
              f"checkpoint at epoch {start_epoch}")

    epochs_without_improvement = 0
    effective_max_epochs = 2 if loadfast else max_epochs

    for epoch in range(start_epoch, effective_max_epochs):
        print(f"Epoch {epoch}\n---------------------------------------")
        wandb.log({"epoch": epoch})
        set_train_mode(model)
        mean_training_loss, mean_training_loss_xy = train_loop(
            device, dataloader_train, model, optimizer)

        model.eval()
        # Both losses every epoch regardless of which selects. Distinct
        # output_folder names so validatoor's own wandb keys and dumped images
        # don't collide.
        hot3d_result = validatoor.validation_loop(
            device, dataloader_val_hot3d, model, mse, "val_hot3d", False, epoch)
        hot3d_loss = hot3d_result.mean_loss_no_pred

        phanesim_loss = None
        if dataloader_val_phanesim is not None:
            phanesim_result = validatoor.validation_loop(
                device, dataloader_val_phanesim, model, mse, "val_phanesim",
                False, epoch)
            phanesim_loss = phanesim_result.mean_loss_no_pred
        set_train_mode(model)

        selection_loss = phanesim_loss if select == "phanesim" else hot3d_loss

        print(f"Done with epoch {epoch} -- train loss {mean_training_loss:.4f} "
              f"(xy {mean_training_loss_xy:.4f}) | HOT3D val {hot3d_loss:.4f}"
              + (f" | Phanesim val {phanesim_loss:.4f}" if phanesim_loss is not None else "")
              + f" | selecting on {select} (best {best_selection_loss:.4f}; "
                f"{epochs_without_improvement}/{patience} without improvement)")
        logged = {
            "train_loss": mean_training_loss,
            "train_loss_xy": mean_training_loss_xy,
            "val_loss_hot3d": hot3d_loss,
            "val_loss_selection": selection_loss,
            "best_val_loss_selection": best_selection_loss,
            "epochs_without_improvement": epochs_without_improvement,
        }
        if phanesim_loss is not None:
            logged["val_loss_phanesim"] = phanesim_loss
        wandb.log(logged)

        is_best = selection_loss < best_selection_loss
        if is_best:
            best_selection_loss = selection_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        is_best_hot3d = hot3d_loss < best_hot3d_loss
        if is_best_hot3d:
            best_hot3d_loss = hot3d_loss

        state = {
            'epoch': epoch,
            'init': init,
            'arm': arm,
            'select': select,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_validation_loss': best_selection_loss,
            'best_hot3d_loss': best_hot3d_loss,
            'val_loss_hot3d': hot3d_loss,
        }
        if phanesim_loss is not None:
            state['val_loss_phanesim'] = phanesim_loss
        save_checkpoint(state, checkpoint_dir)

        # Unconditional at epoch 0 so the epoch-0 point exists for the
        # trajectory table even if the run is cut short.
        if epoch == 0 or epoch % 10 == 0:
            shutil.copy(os.path.join(checkpoint_dir, "checkpoint.pth"),
                        os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pth"))
        if is_best:
            shutil.copy(os.path.join(checkpoint_dir, "checkpoint.pth"),
                        os.path.join(checkpoint_dir, "checkpoint_best.pth"))
        # Written separately so the HOT3D-optimal checkpoint is available even
        # when selection is deliberately blind to HOT3D. Identical to
        # checkpoint_best.pth whenever select == "hot3d".
        if is_best_hot3d:
            shutil.copy(os.path.join(checkpoint_dir, "checkpoint.pth"),
                        os.path.join(checkpoint_dir, "checkpoint_best_hot3d.pth"))

        if epochs_without_improvement >= patience:
            print(f"Early stopping: {select} validation loss has not improved "
                  f"for {patience} consecutive epochs.")
            break


if __name__ == '__main__':
    main()
