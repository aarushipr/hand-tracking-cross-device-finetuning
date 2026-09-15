"""
DetNet fine-tuning with a selectable starting point and training set, so every
comparison comes from the same loss, frozen backbone and stopping procedure.
Additive only: phase 1 and the naive phase 2 are never written to. AD4_INIT picks
the weights and with them the schedule, since Adam at 1e-3 from a converged model
is what collapsed the naive phase 2. AD4_ARM picks mixed / phanesim / hot3d,
unweighted, so the mixed arm stays about 85% HOT3D. AD4_SELECT picks which
validation loss drives stopping; both are logged and both checkpoints written.
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
    HOT3D_FRAME_STRIDE, TRAIN_SPLIT, MAX_EPOCHS, EARLY_STOPPING_PATIENCE)
from py.training.detection.load_weights import load_detnet_weights
import py.training.detection.local_config as local_config
import wandb

modelinputW = header.model_input_width
modelinputH = header.model_input_height

# Same Phanesim val clips as trainer_detection_phanesim.py, so the curves compare.
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

    arm = _env_choice("AD4_ARM", ARMS, "phanesim")
    init = _env_choice("AD4_INIT", INITS, "default")
    select = _env_choice("AD4_SELECT", SELECTS,
                         "phanesim" if init == "default" else "hot3d")

    uses_hot3d_train = arm in ("mixed", "hot3d")
    uses_phanesim = arm in ("mixed", "phanesim")

    if select == "phanesim" and not uses_phanesim:
        raise RuntimeError(
            f"[finetune] AD4_SELECT=phanesim with AD4_ARM={arm} has no Phanesim "
            f"data to select on.")

    # Schedule: phase 1's own when starting from Monado's weights, time-boxed otherwise.
    if init == "default":
        max_epochs = int(os.environ.get("AD4_MAX_EPOCHS", str(MAX_EPOCHS)))
        patience = int(os.environ.get("AD4_PATIENCE", str(EARLY_STOPPING_PATIENCE)))
    else:
        max_epochs = int(os.environ.get("AD4_MAX_EPOCHS", "30"))
        patience = int(os.environ.get("AD4_PATIENCE", "5"))

    loadfast = bool(int(os.environ.get("AD4_LOADFAST", "0")))

    wandb.init(project="hand_detection_training", job_type=f"{init}_{arm}")

    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))
    batch_size = 64

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
    train_pool_dirs = list_sequence_dirs(local_config.hot3d_dataset_root, TRAIN_SPLIT)
    if not train_pool_dirs:
        raise RuntimeError(
            f"[finetune] No HOT3D '{TRAIN_SPLIT}' sequences found at "
            f"{local_config.hot3d_dataset_root}. HOT3D validation is measured in "
            f"every configuration, so this is fatal regardless of AD4_ARM.")
    hot3d_train_dirs, hot3d_val_dirs = split_train_val(train_pool_dirs)
    if not hot3d_val_dirs:
        raise RuntimeError(
            "[finetune] HOT3D train pool is too small to carve out a validation "
            "split.")
    if loadfast:
        hot3d_train_dirs = hot3d_train_dirs[:2]
        hot3d_val_dirs = hot3d_val_dirs[:1]

    print(f"[finetune] init={init} arm={arm} select={select} "
          f"max_epochs={max_epochs} patience={patience}"
          + (" [LOADFAST]" if loadfast else ""))
    print(f"[finetune] HOT3D: {len(hot3d_train_dirs)} train / "
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

    phanesim_train = phanesim_val = None
    if uses_phanesim:
        phanesim_train = PhanesimDetectionDataset(
            clip_dirs=phanesim_train_clips, augment=True)
        phanesim_val = PhanesimDetectionDataset(
            clip_dirs=phanesim_val_clips, augment=False)

    if arm == "mixed":
        hot3d_train = make_hot3d_dataset(hot3d_train_dirs, augment=True)
        train_dataset = ConcatDataset([hot3d_train, phanesim_train])
        n_h, n_p = len(hot3d_train), len(phanesim_train)
        print(f"[finetune] mixed training set: HOT3D {n_h} + Phanesim {n_p} = "
              f"{n_h + n_p} samples/epoch "
              f"({100.0 * n_h / (n_h + n_p):.1f}% HOT3D, unweighted)")
    elif arm == "hot3d":
        train_dataset = make_hot3d_dataset(hot3d_train_dirs, augment=True)
        print(f"[finetune] HOT3D-only training set: {len(train_dataset)} samples/epoch")
    else:
        train_dataset = phanesim_train
        print(f"[finetune] Phanesim-only training set: "
              f"{len(train_dataset)} samples/epoch")

    # worker_init_fn on every loader: forked workers would otherwise share open VRS handles.
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
    model = DetNet.DetNet()
    # load_detnet_weights() first even with init=phase1: it attaches the .bias params.
    load_detnet_weights(model)
    model = torch.nn.DataParallel(model).to(device)

    if init == "phase1":
        if not os.path.exists(PHASE1_CHECKPOINT):
            raise RuntimeError(
                f"[finetune] AD4_INIT=phase1 but no checkpoint at "
                f"{PHASE1_CHECKPOINT}. Run trainer_detection.py first.")
        phase1 = torch.load(PHASE1_CHECKPOINT, map_location=device, weights_only=False)
        model.module.load_state_dict(phase1['state_dict'])
        print(f"[finetune] Loaded phase-1 weights (epoch {phase1.get('epoch')}, "
              f"best val loss {phase1.get('best_validation_loss')})")
    else:
        print("[finetune] Starting from Monado Mercury's shipped ONNX weights "
              "(no checkpoint loaded)")

    # Backbone frozen in every configuration, as in phase 1. Only the head trains.
    for param in model.module.backbone.parameters():
        param.requires_grad = False
    model.module.backbone.eval()

    trainable_params = (p for p in model.module.parameters() if p.requires_grad)
    if "AD4_LR" in os.environ:
        lr = float(os.environ["AD4_LR"])
        optimizer = torch.optim.Adam(trainable_params, lr=lr)
        print(f"[finetune] Adam(lr={lr}) -- explicit AD4_LR override")
    elif init == "default":
        # Bare Adam(), byte-for-byte what trainer_detection.py does; that is the point here.
        optimizer = torch.optim.Adam(trainable_params)
        print("[finetune] Adam() with PyTorch defaults (lr=1e-3) -- matches "
              "trainer_detection.py, so this run differs from phase 1 only in "
              "its training set")
    else:
        optimizer = torch.optim.Adam(trainable_params, lr=1e-4)
        print("[finetune] Adam(lr=1e-4) -- reduced from the 1e-3 default that "
              "produced the naive phase 2's first-epoch collapse")

    loss_fn = torch.nn.MSELoss(reduction="mean").to(device)

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
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"[finetune] Resuming init={init} arm={arm} from its own "
              f"checkpoint at epoch {start_epoch}")

    epochs_without_improvement = 0
    effective_max_epochs = 2 if loadfast else max_epochs

    for epoch in range(start_epoch, effective_max_epochs):
        print(f"Epoch {epoch}\n---------------------------------------")
        wandb.log({"epoch": epoch})

        set_train_mode(model)
        length = len(train_dataloader)
        for idx, batch in enumerate(train_dataloader):
            print(f"Training {idx}/{length}")
            train_batch(device, batch, loss_fn, optimizer, model)

        # Both losses every epoch, so the real-domain trajectory is recorded either way.
        hot3d_loss = validate_epoch(device, hot3d_val_dataloader, loss_fn, model)
        logged = {"val_loss_hot3d": hot3d_loss,
                  "epochs_without_improvement": epochs_without_improvement}
        phanesim_loss = None
        if phanesim_val_dataloader is not None:
            phanesim_loss = validate_epoch(
                device, phanesim_val_dataloader, loss_fn, model)
            logged["val_loss_phanesim"] = phanesim_loss

        selection_loss = phanesim_loss if select == "phanesim" else hot3d_loss
        logged["val_loss_selection"] = selection_loss
        logged["best_val_loss_selection"] = best_selection_loss

        print(f"Epoch {epoch} — HOT3D val {hot3d_loss:.4f}"
              + (f" | Phanesim val {phanesim_loss:.4f}" if phanesim_loss is not None else "")
              + f" | selecting on {select} (best {best_selection_loss:.4f}; "
                f"{epochs_without_improvement}/{patience} without improvement)")
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
            "epoch": epoch,
            "init": init,
            "arm": arm,
            "select": select,
            "state_dict": model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_validation_loss": best_selection_loss,
            "best_hot3d_loss": best_hot3d_loss,
            "val_loss_hot3d": hot3d_loss,
        }
        if phanesim_loss is not None:
            state["val_loss_phanesim"] = phanesim_loss
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
