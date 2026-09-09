"""
trainer_detection_finetune_arms.py -- DetNet fine-tuning with a selectable
starting point and a selectable training set. One script, so that every
comparison in the thesis is produced by the same loss, the same frozen
backbone, the same batch size and the same stopping procedure, and the runs
differ only in the two variables named below.

Additive only. trainer_detection.py (phase 1, HOT3D) and
trainer_detection_phanesim.py (the naive phase 2) are both reported results and
are not touched by this file; it never writes to their checkpoint directories.

------------------------------------------------------------------------------
AD4_INIT -- what the network starts from
------------------------------------------------------------------------------
  default (default value)  Monado Mercury's shipped ONNX weights, exactly as
                           trainer_detection.py starts. Nothing else is loaded.
  phase1                   Continue from checkpoints_train_mixed/
                           checkpoint_best.pth, i.e. the HOT3D-fine-tuned model.

AD4_INIT also selects the hyperparameters, because the two cases want different
ones for principled reasons rather than by preference:

  With AD4_INIT=default the run is a PEER of phase 1 -- the same starting
  weights, the same frozen backbone, a different training set -- so it must use
  phase 1's own schedule and optimiser settings unchanged, or the comparison
  measures the schedule as well as the data. That means a bare
  torch.optim.Adam(), i.e. PyTorch's default lr=1e-3, and MAX_EPOCHS /
  EARLY_STOPPING_PATIENCE imported from trainer_detection.py (120 / 8).

  With AD4_INIT=phase1 the run continues from an already-converged model, where
  a bare Adam() at 1e-3 is what produced the naive phase 2's collapse: Adam
  normalises by gradient magnitude, so its early steps move every trainable
  parameter by roughly the learning rate regardless of how small the gradient
  is, and one Phanesim epoch is ~784 steps (50,185 samples / batch 64). This
  case therefore defaults to lr=1e-4 and a time-boxed 30 / 5.

AD4_LR, AD4_MAX_EPOCHS and AD4_PATIENCE override whichever case is active.

------------------------------------------------------------------------------
AD4_ARM -- what the network trains on
------------------------------------------------------------------------------
  mixed     HOT3D train_mixed TRAIN sequences + Phanesim (plain ConcatDataset).
  phanesim  Phanesim alone.
  hot3d     HOT3D train_mixed TRAIN sequences alone. With AD4_INIT=default this
            reproduces phase 1; with AD4_INIT=phase1 it is the control that
            separates "more HOT3D training helped" from "Phanesim helped".

Measured sizes on this cluster (2026-09-09, frame_stride=5):
    HOT3D detection train   295,466 samples (212 sequences)
    HOT3D detection val      33,680 samples (23 sequences)
    Phanesim detection       50,185 samples (both roots pooled, all 640x480)
so the mixed arm is ~85.5% HOT3D by sample count. No re-weighting is applied:
an earlier draft rebalanced the two to 50/50 with a repeat wrapper, which would
have cut HOT3D's share of each epoch about six times below the natural ratio and
weakened exactly the anchoring that mixing exists to provide. The realised ratio
is printed at startup rather than assumed.

------------------------------------------------------------------------------
AD4_SELECT -- which validation loss drives early stopping and checkpoint_best
------------------------------------------------------------------------------
  Defaults to phanesim when AD4_INIT=default, and to hot3d otherwise.

  The naive phase 2 selected on Phanesim val while being scored on HOT3D, so its
  checkpoint_best.pth optimised a criterion with no knowledge of the evaluation
  domain. That is a bug when continuing from a HOT3D-tuned model, which is why
  AD4_INIT=phase1 selects on HOT3D.

  For AD4_INIT=default + AD4_ARM=phanesim it is not a bug but a choice: selecting
  on Phanesim keeps real data out of every training decision, so the resulting
  model is honestly describable as trained purely on synthetic data. Both losses
  are measured and logged every epoch either way, and BOTH checkpoint_best.pth
  (by AD4_SELECT) and checkpoint_best_hot3d.pth are written, so either selection
  can be reported without a second run.

Preprocessing is unchanged and is not a function of AD4_INIT: the input
convention belongs to the weights, not to the data, and is defined once in
py/evaluation/preprocess_baseline.py. Both dataset loaders already reproduce it
through the same calls (_pp.rotate_upright + augmentation.augment_image).

Checkpoints go to checkpoints_monado_<arm>/ (AD4_INIT=default) or
checkpoints_phase2_<arm>/ (AD4_INIT=phase1).
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

# Same clip-level Phanesim val stride as trainer_detection_phanesim.py, so the
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

    # Schedule: phase 1's own when starting from Monado's weights (this run is
    # then a peer of phase 1 and must not differ in schedule), time-boxed
    # otherwise. See the module docstring.
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
    # Loaded in every configuration: even when HOT3D is not trained on and not
    # selected on, its val loss is measured each epoch so the run produces the
    # real-domain trajectory the thesis reports.
    #
    # list_sequence_dirs(..., "train_mixed") lists only the train participants,
    # so hot3d_split.TEST_MIXED_PARTICIPANTS -- the sequences eval_detnet.py
    # scores against -- can never reach this training set or this val set.
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

    # worker_init_fn is required whenever HOT3D samples are in a loader:
    # DataLoader workers are forked, and without it they would inherit the
    # parent's already-open VRS handles. It only clears each worker's provider
    # cache, so it is harmless for Phanesim samples and every loader carries it
    # unconditionally rather than making its presence depend on the arm.
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
    # load_detnet_weights() is what loads Monado's shipped ONNX weights, and it
    # runs in BOTH cases. With init=phase1 its weight VALUES are overwritten
    # immediately below, but the call is still required first: load_weights.py's
    # _load_conv_bn() dynamically ATTACHES a .bias Parameter to backbone conv
    # layers that InvertedResidual builds with bias=False, so a bare
    # DetNet.DetNet() has fewer parameters than any checkpoint saved after this
    # call and load_state_dict(strict=True) would fail on every added bias.
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

    # Backbone frozen in every configuration, matching phase 1 and the naive
    # phase 2. Only the head trains.
    for param in model.module.backbone.parameters():
        param.requires_grad = False
    model.module.backbone.eval()

    trainable_params = (p for p in model.module.parameters() if p.requires_grad)
    if "AD4_LR" in os.environ:
        lr = float(os.environ["AD4_LR"])
        optimizer = torch.optim.Adam(trainable_params, lr=lr)
        print(f"[finetune] Adam(lr={lr}) -- explicit AD4_LR override")
    elif init == "default":
        # Bare Adam(), byte-for-byte what trainer_detection.py does. Matching
        # phase 1 exactly is the point of this configuration.
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

        # Both losses are measured every epoch regardless of which one selects,
        # so the run records the real-domain trajectory even when it is not
        # optimising against it.
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

        # Unconditional at epoch 0 (not only every 10th) so the epoch-0 point
        # exists for the trajectory table even if the run is cut short.
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
