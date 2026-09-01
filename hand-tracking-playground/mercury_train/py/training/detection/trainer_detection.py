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

from py.training.detection.HOT3DVRSDetectionDataset import HOT3DVRSDetectionDataset
from py.training.common.hot3d_split import list_sequence_dirs, split_train_val
from py.training.detection.load_weights import load_detnet_weights
import py.training.detection.local_config as local_config
from py.training.common.a_geometry import *
import wandb


modelinputW = header.model_input_width
modelinputH = header.model_input_height

# --- Fine-tuning schedule (Chapter 4, Table 4.2) ---------------------------
# Matches KeyNet's schedule (py/training/keypoint/kpest_trainer.py) so that
# both networks are subject to the same stopping procedure, and any
# difference in how much each benefits from fine-tuning reflects the
# networks and data rather than a difference in training length.

# Hard ceiling on epochs. Training normally stops earlier, through the
# early-stopping patience below; this only bounds the SLURM job.
MAX_EPOCHS = 120

# Stop after this many consecutive epochs with no improvement in validation
# loss. Deliberately generous: the validation split is a handful of HOT3D
# sequences, so epoch-to-epoch validation loss is noisy, and a tight patience
# would stop on that noise rather than on genuine convergence.
EARLY_STOPPING_PATIENCE = 8


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

    # Center and size losses are masked by exists_gt so that samples without
    # a hand don't pull bounding-box predictions toward zero.
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
    # model.train() flips every submodule to train mode, including the frozen
    # backbone. But the backbone's BatchNorm layers must stay in eval mode --
    # their running stats were reset by load_detnet_weights to represent a
    # neutral/identity transform, and would drift away from that if allowed
    # to update during training.
    model.train()
    model.module.backbone.eval()

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # wandb reads WANDB_MODE from the environment — set WANDB_MODE=disabled in
    # your SLURM script to run without logging, or omit it to log normally.
    # No entity= specified: was hardcoded to "col" (the original author's
    # Collabora team), which fails with a permission error for any other
    # wandb login. Omitting it uses whatever account is actually logged in.
    wandb.init(project="hand_detection_training")

    # On SLURM, cpu_count() returns all CPUs on the node, not just the ones
    # allocated to this job. SLURM_CPUS_PER_TASK is the correct value to use.
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))

    batch_size = 64
    
    train_pool_dirs = list_sequence_dirs(local_config.hot3d_dataset_root, "train")
    
    if not train_pool_dirs:
        raise RuntimeError(
            "[trainer_detection] No HOT3D train sequences found at "
            "local_config.hot3d_dataset_root — nothing to train on.")
    train_seq_dirs, val_seq_dirs = split_train_val(train_pool_dirs)

    train_dataset = HOT3DVRSDetectionDataset(
        sequence_dirs=train_seq_dirs,
        hot3d_repo_root=local_config.hot3d_repo_root)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


    val_dataloader = None
    if val_seq_dirs:
        val_dataset = HOT3DVRSDetectionDataset(
            sequence_dirs=val_seq_dirs,
            hot3d_repo_root=local_config.hot3d_repo_root)
        val_dataloader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        print("[trainer_detection] Too few HOT3D train-pool sequences to carve out "
            "a validation split — training will proceed with no validation-loss "
            "tracking until more sequences are available.")
    # Test (run once after training): no held-out test dataset yet.
    # Replace with HOT3D once obtained — that represents true cross-device
    # generalisation from the HMD capture setup to a different XR device.
    # test_dataloader = DataLoader(HOT3DDataset(...), ...)

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

    # Use an absolute path so checkpoints are always written to the same place
    # regardless of what directory SLURM starts the job from.
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.pth')

    if os.path.exists(checkpoint_file):
        # weights_only=False: PyTorch 2.6 flipped this default to True, which
        # refuses any checkpoint containing a non-tensor object -- including
        # the numpy scalar that best_validation_loss used to be. These are
        # checkpoints this script wrote itself, not untrusted files, so the
        # restriction buys nothing here and breaks resume-after-preemption.
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
