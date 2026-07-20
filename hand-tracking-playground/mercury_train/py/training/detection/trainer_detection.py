import os
import sys
import shutil
import multiprocessing

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../'))
    from common import visualize_directreg

import header

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import DetNet
import py.training.common.a_geometry as geo

from CombinedDataset import CombinedDataset
from py.training.detection.HMDHandRectsDataset import HMDHandRectsDataset
import py.training.detection.local_config as local_config
from py.training.common.a_geometry import *
import wandb


modelinputW = header.model_input_width
modelinputH = header.model_input_height


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

    model.train()
    return total_loss / len(val_dataloader)


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

    # Training: subject00 + subject01, EgoHands, EpicKitchens.
    # subject02 is excluded from CombinedDataset — held out for validation.
    train_dataloader = DataLoader(
        CombinedDataset(),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers)

    # Validation (run every epoch): subject02 sequences — same device as
    # training data but a different subject not seen during training.
    val_dataset = torch.utils.data.ConcatDataset([
        HMDHandRectsDataset(
            f"{local_config.hmdhandrects_location}/sequences/train_subject02_sequence00"),
        HMDHandRectsDataset(
            f"{local_config.hmdhandrects_location}/sequences/train_subject02_sequence01"),
    ])
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers)

    # Test (run once after training): no held-out test dataset yet.
    # Replace with HOT3D once obtained — that represents true cross-device
    # generalisation from the HMD capture setup to a different XR device.
    # test_dataloader = DataLoader(HOT3DDataset(...), ...)

    model = DetNet.DetNet()
    model = torch.nn.DataParallel(model).to(device)
    optimizer = torch.optim.Adam(model.module.parameters())
    loss_fn = nn.MSELoss(reduction="mean").to(device)

    start_epoch = 0
    best_validation_loss = float('inf')

    # Use an absolute path so checkpoints are always written to the same place
    # regardless of what directory SLURM starts the job from.
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.pth')

    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location=device)
        if 'best_validation_loss' in checkpoint:
            best_validation_loss = checkpoint['best_validation_loss']
        start_epoch = checkpoint['epoch']
        model.module.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    for epoch in range(start_epoch, 200):
        print(f"Epoch {epoch}\n---------------------------------------")
        wandb.log({"epoch": epoch})

        model.train()
        length = len(train_dataloader)
        for idx, batch in enumerate(train_dataloader):
            print(f"Training {idx}/{length}")
            train_batch(device, batch, loss_fn, optimizer, model)

        val_loss = validate_epoch(device, val_dataloader, loss_fn, model)
        print(f"Epoch {epoch} — val loss: {val_loss:.4f} (best: {best_validation_loss:.4f})")
        wandb.log({"val_loss": val_loss, "best_val_loss": best_validation_loss})

        is_best = val_loss < best_validation_loss
        if is_best:
            best_validation_loss = val_loss

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


if __name__ == "__main__":
    main()
