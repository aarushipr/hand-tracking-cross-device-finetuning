"""
kpest_trainer_phase2_arms.py -- KeyNet fine-tuning PHASE 2, corrected
variant. The KeyNet counterpart of
py/training/detection/trainer_detection_phase2_arms.py; that file's
docstring carries the full reasoning and the two are meant to read identically.

Continues from the phase-1 HOT3D checkpoint like kpest_trainer_phanesim.py does,
but changes three things that made the plain phase-2 run regress on HOT3D.

Kept as a THIRD script, additive only: phase 1 and the naive phase 2 are both
reported results, so neither of their scripts nor checkpoint directories change.

------------------------------------------------------------------------------
WHAT DIFFERS FROM kpest_trainer_phanesim.py
------------------------------------------------------------------------------

1. LEARNING RATE. kpest_trainer_phanesim.py calls
   torch.optim.AdamW(trainable_params) with no arguments, i.e. PyTorch's default
   lr=1e-3. AdamW normalises by gradient magnitude, so its early steps move every
   trainable parameter by roughly the learning rate regardless of the gradient's
   actual size. Phase 1 used the same value while moving TOWARD the evaluation
   domain, where a large step helps; in phase 2 the identical step size moves
   away from it. Default 1e-4 here (AD4_LR).

2. WEIGHT DECAY. The same bare AdamW() call also takes PyTorch's default
   weight_decay=0.01, which decays the phase-1 weights toward zero on every
   step independently of anything the training data's gradient says. That is a
   second forgetting channel with no connection to the domain shift at all, and
   it is switched off here (AD4_WEIGHT_DECAY) rather than left as an unexamined
   default. This has no counterpart in the DetNet variant, which uses Adam
   rather than AdamW and therefore never had a decay term.

3. VALIDATION SET. The naive run validates on Phanesim clips only, so early
   stopping and checkpoint_best.pth optimise a criterion with no knowledge of
   HOT3D. Every arm here validates on HOT3D's val split. Phanesim val loss is
   still measured and logged alongside (when the arm uses Phanesim) so the
   stability/plasticity trade is visible per epoch, but it gates nothing.

Also: MAX_EPOCHS 20 / patience 4 rather than 120 / 8. Lower than the DetNet
variant's 30 / 5 because KeyNet's HOT3D epochs are the expensive ones -- measured
~50-60 min/epoch in phase 1 from its checkpoint timestamps, against Phanesim's
~3 min/epoch -- so an arm that touches HOT3D costs roughly an hour per epoch and
has to be time-boxed harder against the submission deadline. The naive KeyNet
phase 2 early-stopped at 23 epochs, so 20 is not a severe cut.

checkpoint_0.pth is saved unconditionally so the epoch-0 point exists for the
forgetting-trajectory table even if a run is cut short.

------------------------------------------------------------------------------
THE THREE ARMS (AD4_ARM)
------------------------------------------------------------------------------

Each arm changes only the TRAINING set. Learning rate, weight decay, validation
set, stopping rule, frozen image_network and batch size are identical across all
three, so the arms differ in exactly one variable.

  AD4_ARM=mixed     (default)  HOT3D train_mixed TRAIN sequences + Phanesim.
  AD4_ARM=phanesim             Phanesim alone -- isolates the learning-rate and
                               weight-decay changes against the naive numbers.
  AD4_ARM=hot3d                HOT3D alone -- the CONTROL. On the detection side
                               the mixed training set measured ~85% HOT3D, so
                               more HOT3D training at a new learning rate could
                               improve the HOT3D score on its own; without this
                               arm an improvement in the mixed arm cannot be
                               attributed to Phanesim rather than to the extra
                               HOT3D epochs. The keypoint dataset sizes are
                               printed at startup rather than assumed to match
                               detection's, but the confound is the same either
                               way.

------------------------------------------------------------------------------
MIXING
------------------------------------------------------------------------------

The mixed arm is a plain ConcatDataset with shuffle=True: one pass over both
datasets per epoch, with the HOT3D/Phanesim ratio simply their relative size.
No weighting is applied and no ratio constant is tuned; the realised ratio is
printed at startup so the thesis can report what the run actually trained on.

An earlier draft re-weighted the two to 50/50 with a repeat wrapper. On the
detection side that would have cut HOT3D's share of each epoch by about six
times relative to the natural ratio, weakening exactly the anchoring the
rehearsal exists to provide -- so the weighting was removed rather than
reproduced here.

HOT3D frame stride stays at kpest_trainer.HOT3D_FRAME_STRIDE (5) for both
training and validation. Raising it for the training stream was considered as a
way to halve the dominant cost, and rejected: HOT3DKeypointDataset's index cache
key includes frame_stride (see its _cache_path), the cluster's cache holds only
stride-5 entries, and a different stride would trigger a full 294-sequence .vrs
index rebuild whose cost exceeds the epoch time it would save.
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
    HOT3D_FRAME_STRIDE, TRAIN_SPLIT)
from HOT3DKeypointDataset import HOT3DKeypointDataset, worker_init as hot3d_worker_init
from PhanesimKeypointDataset import PhanesimKeypointDataset, discover_clip_dirs

mse = nn.MSELoss(reduction='mean')

# Same clip-level Phanesim val stride as kpest_trainer_phanesim.py, so the
# Phanesim val curve here is measured on the same held-out clips the naive arm
# used and the two are directly comparable.
VAL_CLIP_STRIDE = 20

ARMS = ("mixed", "phanesim", "hot3d")

LEARNING_RATE = float(os.environ.get("AD4_LR", "1e-4"))
WEIGHT_DECAY = float(os.environ.get("AD4_WEIGHT_DECAY", "0.0"))
MAX_EPOCHS_PHASE2 = int(os.environ.get("AD4_MAX_EPOCHS", "20"))
EARLY_STOPPING_PATIENCE_PHASE2 = int(os.environ.get("AD4_PATIENCE", "4"))

PHASE1_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "checkpoints_train_mixed", "checkpoint_best.pth")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_devices = 1
    batch_size_per_device = 64  # Same as kpest_trainer.py's own.
    if device.type == "cuda":
        num_devices = torch.cuda.device_count()
        print(f"Let's use {num_devices} GPUs!")
    batch_size = batch_size_per_device * num_devices

    arm = os.environ.get("AD4_ARM", "mixed").strip().lower()
    if arm not in ARMS:
        raise RuntimeError(
            f"[phase2] AD4_ARM={arm!r} is not one of {ARMS}. Refusing to guess "
            f"-- the arm decides which training set is used and names the "
            f"checkpoint directory, so a typo here would silently produce a run "
            f"that is not the experiment it claims to be.")
    uses_phanesim = arm in ("mixed", "phanesim")

    loadfast = header.env_settings.loadfast

    if header.env_settings.wandb_enabled:
        wandb.init(project="keypoint_estimator_training", job_type=f"phase2_{arm}")
    else:
        wandb.init(project="keypoint_estimator_training", mode="disabled")

    # Capped at 4 because HOT3D samples are in play in every arm (its val split
    # drives selection even in the Phanesim-only arm): HOT3DKeypointDataset holds
    # live VRS providers and DataLoader workers are forked. This is
    # kpest_trainer.py's own cap and its reason applies here. kpest_trainer_
    # phanesim.py could safely leave it uncapped only because every loader it
    # built was Phanesim-only.
    num_workers = min(4, int(os.environ.get("SLURM_CPUS_PER_TASK",
                                            multiprocessing.cpu_count())))

    # ---------------- Phanesim ----------------
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
    # Loaded unconditionally: HOT3D's val split drives early stopping and
    # checkpoint selection in every arm. list_sequence_dirs(..., "train_mixed")
    # lists only the train participants, so hot3d_split.TEST_MIXED_PARTICIPANTS
    # -- the sequences eval_keynet.py scores against -- can never reach this
    # training set or this val set.
    train_pool = hot3d_split.list_sequence_dirs(
        local_config.hot3d_dataset_root, TRAIN_SPLIT)
    if not train_pool:
        raise RuntimeError(
            f"[phase2] No HOT3D '{TRAIN_SPLIT}' sequences found under "
            f"{local_config.hot3d_dataset_root}. Every arm validates on HOT3D, "
            f"so this is fatal regardless of AD4_ARM.")
    hot3d_train_dirs, hot3d_val_dirs = hot3d_split.split_train_val(train_pool)
    if not hot3d_val_dirs:
        raise RuntimeError(
            "[phase2] HOT3D train pool too small to carve out a validation "
            "split, and HOT3D validation is what drives early stopping and "
            "checkpoint selection here. Refusing to fall back to Phanesim-only "
            "validation silently -- that is the naive run's bug.")
    if loadfast:
        hot3d_train_dirs = hot3d_train_dirs[:2]
        hot3d_val_dirs = hot3d_val_dirs[:1]

    print(f"[phase2] arm={arm} lr={LEARNING_RATE} weight_decay={WEIGHT_DECAY} "
          f"max_epochs={MAX_EPOCHS_PHASE2} patience={EARLY_STOPPING_PATIENCE_PHASE2}"
          + (" [LOADFAST]" if loadfast else ""))
    print(f"[phase2] HOT3D: {len(hot3d_train_dirs)} train / "
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
    # eval_keynet.py's standalone deterministic evaluation, not for validation
    # during training.
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
        print(f"[phase2] mixed training set: HOT3D {n_h} + Phanesim {n_p} = "
              f"{n_h + n_p} samples/epoch "
              f"({100.0 * n_h / (n_h + n_p):.1f}% HOT3D, unweighted)")
    elif arm == "hot3d":
        train_dataset = make_hot3d_dataset(hot3d_train_dirs)
        print(f"[phase2] HOT3D-only control training set: "
              f"{len(train_dataset)} samples/epoch")
    else:
        train_dataset = phanesim_train
        print(f"[phase2] Phanesim-only training set: "
              f"{len(train_dataset)} samples/epoch")

    # worker_init_fn is required whenever HOT3D samples are in a loader (forked
    # workers would otherwise share the parent's open VRS handles). It only
    # clears provider caches, so it is harmless for Phanesim samples and every
    # loader carries it unconditionally rather than depending on the arm.
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
    if not os.path.exists(PHASE1_CHECKPOINT):
        raise RuntimeError(
            f"[phase2] Phase-1 checkpoint not found at {PHASE1_CHECKPOINT}. "
            f"Every arm continues from it -- run kpest_trainer.py first.")

    model = KeyNet.KeyNet()
    # Must run BEFORE the phase-1 state_dict load: load_weights.py's
    # load_conv_bn() dynamically attaches .bias Parameters to image_network/
    # fused_network/network_2d_px_coord conv layers that InvertedResidual builds
    # with bias=False, so a bare KeyNet.KeyNet() has fewer parameters than the
    # phase-1 checkpoint and a strict load_state_dict would fail on every one of
    # them. Same ordering as kpest_trainer_phanesim.py.
    load_keynet_weights(model)

    # Frozen BEFORE DataParallel wrapping, matching kpest_trainer.py's ordering.
    for param in model.image_network.parameters():
        param.requires_grad = False

    model = torch.nn.DataParallel(model).to(device)
    model.module.image_network.eval()

    phase1 = torch.load(PHASE1_CHECKPOINT, map_location=device, weights_only=False)
    model.module.load_state_dict(phase1['state_dict'])
    print(f"[phase2] Loaded phase-1 weights (epoch {phase1.get('epoch')}, "
          f"best val loss {phase1.get('best_validation_loss')})")

    trainable_params = (p for p in model.module.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(
        trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Deliberately NOT resuming epoch/optimizer state from the phase-1
    # checkpoint -- this is a new training phase, not a continuation of the same
    # run, and phase 1's AdamW moment estimates were accumulated at a different
    # learning rate and a different weight decay.
    start_epoch = 0
    best_validation_loss = float('inf')

    # Named by arm so concurrent arms can never collide, and so a resume can
    # never pick up a checkpoint produced by a different training set.
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
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except BaseException:
            print("Couldn't load optimizer state dict! This shouldn't happen "
                  "except for right after model weight transfers!")
        print(f"[phase2] Resuming arm={arm} from its own checkpoint at "
              f"epoch {start_epoch}")

    epochs_without_improvement = 0
    effective_max_epochs = 2 if loadfast else MAX_EPOCHS_PHASE2

    for epoch in range(start_epoch, effective_max_epochs):
        print(f"Epoch {epoch}\n---------------------------------------")
        wandb.log({"epoch": epoch})
        set_train_mode(model)
        mean_training_loss, mean_training_loss_xy = train_loop(
            device, dataloader_train, model, optimizer)

        model.eval()
        # HOT3D is the selection criterion in every arm; Phanesim is measured
        # only so the trade-off is visible. Distinct output_folder names so
        # validatoor's own wandb keys and dumped images don't collide.
        hot3d_result = validatoor.validation_loop(
            device, dataloader_val_hot3d, model, mse, "val_hot3d", False, epoch)
        mean_validation_loss = hot3d_result.mean_loss_no_pred

        logged = {
            "train_loss": mean_training_loss,
            "train_loss_xy": mean_training_loss_xy,
            "val_loss_hot3d": mean_validation_loss,
            "best_val_loss_hot3d": best_validation_loss,
            "epochs_without_improvement": epochs_without_improvement,
        }
        phanesim_msg = ""
        if dataloader_val_phanesim is not None:
            phanesim_result = validatoor.validation_loop(
                device, dataloader_val_phanesim, model, mse, "val_phanesim",
                False, epoch)
            logged["val_loss_phanesim"] = phanesim_result.mean_loss_no_pred
            phanesim_msg = (f" | Phanesim val loss: "
                            f"{phanesim_result.mean_loss_no_pred:.4f}")
        set_train_mode(model)

        is_best = mean_validation_loss < best_validation_loss
        if is_best:
            best_validation_loss = mean_validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(f"Done with epoch {epoch} -- train loss: {mean_training_loss:.4f} "
              f"(xy {mean_training_loss_xy:.4f}), HOT3D val loss: "
              f"{mean_validation_loss:.4f} (best: {best_validation_loss:.4f}; "
              f"{epochs_without_improvement}/{EARLY_STOPPING_PATIENCE_PHASE2} "
              f"without improvement)" + phanesim_msg)
        wandb.log(logged)

        state = {
            'epoch': epoch,
            'arm': arm,
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_validation_loss': best_validation_loss,
            'val_loss_hot3d': mean_validation_loss,
        }
        if dataloader_val_phanesim is not None:
            state['val_loss_phanesim'] = logged["val_loss_phanesim"]
        save_checkpoint(state, checkpoint_dir)

        # Unconditional at epoch 0 so the epoch-0 point exists for the
        # forgetting-trajectory table even if the run is cut short.
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
