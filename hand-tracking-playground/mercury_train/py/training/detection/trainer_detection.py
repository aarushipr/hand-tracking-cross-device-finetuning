import os
import sys
import shutil
import multiprocessing

import header

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import DetNet
import py.training.common.a_geometry as geo

from py.training.detection.HOT3DVRSDetectionDataset import (
    HOT3DVRSDetectionDataset, worker_init as hot3d_worker_init)
from py.training.common.hot3d_split import list_sequence_dirs, split_train_val
from py.training.detection.load_weights import load_detnet_weights
import py.training.detection.local_config as local_config
from py.training.common.a_geometry import *
import wandb


modelinputW = header.model_input_width
modelinputH = header.model_input_height

# --- Fine-tuning schedule (Chapter 4, Table 4.2) ---------------------------
# Same schedule as kpest_trainer.py, so the two networks stop the same way.

# Hard ceiling on epochs; early stopping normally fires first.
MAX_EPOCHS = 120

# Patience is generous: val loss over a handful of sequences is noisy.
EARLY_STOPPING_PATIENCE = 8

# Split to train on; also names the checkpoint dir so resume can't cross splits.
TRAIN_SPLIT = "train_mixed"

# Keep every Nth frame; at 30 Hz they are near-duplicates. Matches KeyNet's stride.
HOT3D_FRAME_STRIDE = 5


def save_checkpoint(states, output_dir, filename='checkpoint.pth'):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(states, os.path.join(output_dir, filename))


def train_batch(device, batch, loss_fn, optimizer, model):
    inp = batch['image'].to(device)

    exists_gt   = batch['exists'].to(device)
    center_x_gt = batch['center_x'].to(device)
    center_y_gt = batch['center_y'].to(device)
    size_gt     = batch['size'].to(device)

    pred = model(inp)
    exists_pred   = pred[0]
    center_x_pred = pred[1]
    center_y_pred = pred[2]
    size_pred     = pred[3]

    # Masked by exists_gt so hand-free samples don't pull boxes toward zero.
    loss_exists   = loss_fn(exists_gt, exists_pred)
    loss_center_x = loss_fn(center_x_gt * exists_gt, center_x_pred * exists_gt)
    loss_center_y = loss_fn(center_y_gt * exists_gt, center_y_pred * exists_gt)
    loss_size     = loss_fn(size_gt * exists_gt, size_pred * exists_gt)

    loss = loss_exists + loss_center_x + loss_center_y + loss_size

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    wandb.log({"loss": float(loss)})


def validate_epoch(device, val_dataloader, loss_fn, model):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in val_dataloader:
            inp         = batch['image'].to(device)
            exists_gt   = batch['exists'].to(device)
            center_x_gt = batch['center_x'].to(device)
            center_y_gt = batch['center_y'].to(device)
            size_gt     = batch['size'].to(device)

            pred = model(inp)
            exists_pred   = pred[0]
            center_x_pred = pred[1]
            center_y_pred = pred[2]
            size_pred     = pred[3]

            loss_exists   = loss_fn(exists_gt, exists_pred)
            loss_center_x = loss_fn(center_x_gt * exists_gt, center_x_pred * exists_gt)
            loss_center_y = loss_fn(center_y_gt * exists_gt, center_y_pred * exists_gt)
            loss_size     = loss_fn(size_gt * exists_gt, size_pred * exists_gt)

            total_loss += (loss_exists + loss_center_x + loss_center_y + loss_size).item()

    set_train_mode(model)
    return total_loss / len(val_dataloader)

def set_train_mode(model):
    # Backbone BatchNorm stays in eval mode; its running stats were reset to identity.
    model.train()
    model.module.backbone.eval()

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set WANDB_MODE=disabled in the SLURM script to run without logging.
    # No entity=: the hardcoded "col" fails for any other wandb login.
    wandb.init(project="hand_detection_training")

    # cpu_count() sees the whole node on SLURM; SLURM_CPUS_PER_TASK is the allocation.
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))

    batch_size = 64
    
    train_pool_dirs = list_sequence_dirs(local_config.hot3d_dataset_root, TRAIN_SPLIT)
    
    if not train_pool_dirs:
        raise RuntimeError(
            "[trainer_detection] No HOT3D train sequences found at "
            "local_config.hot3d_dataset_root — nothing to train on.")
    train_seq_dirs, val_seq_dirs = split_train_val(train_pool_dirs)

    # loadfast is a smoke test, not a model worth keeping.
    # Read straight from env: this package's header.py has no env_settings object.
    loadfast = bool(int(os.environ.get("AD4_LOADFAST", "0")))
    if loadfast:
        train_seq_dirs = train_seq_dirs[:2]
        val_seq_dirs = val_seq_dirs[:1] or train_seq_dirs[:1]

    print(f"[trainer_detection] {len(train_seq_dirs)} train / "
          f"{len(val_seq_dirs)} val HOT3D sequences, "
          f"frame_stride={HOT3D_FRAME_STRIDE}")

    index_cache_dir = getattr(local_config, "hot3d_index_cache_dir", None)

    def make_hot3d_loader(sequence_dirs, shuffle, augment):
        dataset = HOT3DVRSDetectionDataset(
            sequence_dirs=sequence_dirs,
            hot3d_repo_root=local_config.hot3d_repo_root,
            frame_stride=HOT3D_FRAME_STRIDE,
            index_cache_dir=index_cache_dir,
            augment=augment)
        # num_workers>0 is safe only via worker_init(); forked workers would share VRS handles.
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, worker_init_fn=hot3d_worker_init,
            persistent_workers=num_workers > 0, drop_last=False)

    train_dataloader = make_hot3d_loader(train_seq_dirs, shuffle=True,
                                         augment=True)

    val_dataloader = None
    if val_seq_dirs:
        # augment=False: an augmented val split re-measures a different distribution each epoch.
        val_dataloader = make_hot3d_loader(val_seq_dirs, shuffle=False,
                                           augment=False)
    else:
        print("[trainer_detection] Too few HOT3D train-pool sequences to carve out "
            "a validation split — training will proceed with no validation-loss "
            "tracking until more sequences are available.")
    # No test set here; all evaluation is eval_detnet.py --split test_mixed.

    model = DetNet.DetNet()
    load_detnet_weights(model)
    
    for param in model.backbone.parameters():
        param.requires_grad = False
        
    model = torch.nn.DataParallel(model).to(device)
    model.module.backbone.eval()
    
    trainable_params = (p for p in model.module.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(trainable_params)
    
    loss_fn = nn.MSELoss(reduction="mean").to(device)

    start_epoch = 0
    best_validation_loss = float('inf')

    # Absolute path so checkpoints land in one place whatever dir SLURM starts in.
    # Scoped by split so resume can't pick up a checkpoint trained on different data.
    checkpoint_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "checkpoints_loadfast" if loadfast else f"checkpoints_{TRAIN_SPLIT}")
    checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.pth')

    if os.path.exists(checkpoint_file):
        # weights_only=False: PyTorch 2.6's default refuses our non-tensor best_validation_loss.
        checkpoint = torch.load(checkpoint_file, map_location=device,
                                weights_only=False)
        if 'best_validation_loss' in checkpoint:
            best_validation_loss = checkpoint['best_validation_loss']
        start_epoch = checkpoint['epoch']
        model.module.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    epochs_without_improvement = 0

    for epoch in range(start_epoch, MAX_EPOCHS):
        print(f"Epoch {epoch}\n---------------------------------------")
        wandb.log({"epoch": epoch})

        set_train_mode(model)
        length = len(train_dataloader)
        for idx, batch in enumerate(train_dataloader):
            print(f"Training {idx}/{length}")
            train_batch(device, batch, loss_fn, optimizer, model)

        is_best = False
        if val_dataloader is not None:
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
        else:
            print(f"Epoch {epoch} — no validation set available, skipping val loss / best-model tracking.")

        save_checkpoint({
            "epoch": epoch,
            "state_dict": model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_validation_loss": best_validation_loss,
        }, checkpoint_dir)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, saving extra checkpoint!")
            shutil.copy(
                os.path.join(checkpoint_dir, "checkpoint.pth"),
                os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pth"))

        if is_best:
            print(f"Best model so far! Saving as checkpoint_best.pth")
            shutil.copy(
                os.path.join(checkpoint_dir, "checkpoint.pth"),
                os.path.join(checkpoint_dir, "checkpoint_best.pth"))

        if val_dataloader is not None and epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping: validation loss has not improved for "
                  f"{EARLY_STOPPING_PATIENCE} consecutive epochs.")
            break
    else:
        print(f"Reached the MAX_EPOCHS ceiling of {MAX_EPOCHS} without "
              f"early stopping triggering.")


if __name__ == "__main__":
    main()
