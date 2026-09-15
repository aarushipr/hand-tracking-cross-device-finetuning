"""
KeyNet fine-tuning with a selectable starting point and training set, the
counterpart of trainer_detection_finetune_arms.py, which carries the shared
reasoning. Additive only. AD4_INIT also decides the optimiser: from Monado's
weights it keeps phase 1's bare AdamW including weight_decay=0.01, while from
phase1 both that decay and lr=1e-3 pull a converged model away from what it
learned, so it defaults to lr=1e-4, weight_decay=0.0 and a time-boxed 30/5.
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

# Same Phanesim val clips as kpest_trainer_phanesim.py, so the curves compare.
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

    # Capped at 4: HOT3D is in every arm, and forked workers can't share its VRS providers.
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
    # Loaded always: its val loss is measured each epoch for the real-domain trajectory.
    # "train_mixed" lists only train participants, so test sequences can never leak in.
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

    # eval_mode=False throughout; eval_mode is only for eval_keynet.py's standalone run.
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

    # worker_init_fn on every loader: forked workers would otherwise share open VRS handles.
    # drop_last=False: under loadfast the train set can be under one batch (job 1700699).
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
    # load_keynet_weights() first even with init=phase1: it attaches the .bias params.
    load_keynet_weights(model)

    # Frozen before DataParallel wrapping, in every configuration, as in phase 1.
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
        # Bare AdamW(), byte-for-byte what kpest_trainer.py does; that is the point here.
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
        # Both losses every epoch; distinct output_folder names so wandb keys don't collide.
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

        # Unconditional at epoch 0 so the trajectory table has that point if the run is cut.
        if epoch == 0 or epoch % 10 == 0:
            shutil.copy(os.path.join(checkpoint_dir, "checkpoint.pth"),
                        os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pth"))
        if is_best:
            shutil.copy(os.path.join(checkpoint_dir, "checkpoint.pth"),
                        os.path.join(checkpoint_dir, "checkpoint_best.pth"))
        # Separate file so the HOT3D-optimal checkpoint exists even when selection ignores HOT3D.
        if is_best_hot3d:
            shutil.copy(os.path.join(checkpoint_dir, "checkpoint.pth"),
                        os.path.join(checkpoint_dir, "checkpoint_best_hot3d.pth"))

        if epochs_without_improvement >= patience:
            print(f"Early stopping: {select} validation loss has not improved "
                  f"for {patience} consecutive epochs.")
            break


if __name__ == '__main__':
    main()
