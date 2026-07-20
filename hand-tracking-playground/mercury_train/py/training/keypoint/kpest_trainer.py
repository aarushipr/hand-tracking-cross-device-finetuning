import os
import sys
import shutil

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../'))

import local_config

import CombinedDataset
from RandoData import RandoDataset
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import KeyNet
import settings
import kpest_header as header
import wandb
import visualizer
import validatoor
import multiprocessing


# https://gitanswer.com/pytorch-too-many-open-files-error-cplusplus-356516297
torch.multiprocessing.set_sharing_strategy('file_system')

mse = nn.MSELoss(reduction='mean')
gnll = nn.GaussianNLLLoss(reduction='none')


def save_checkpoint(states, output_dir, filename='checkpoint.pth'):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(states, os.path.join(output_dir, filename))


def train_loop(device, dataloader, model, optimizer):
    total_loss = 0
    loss_divisor = 0
    l = len(dataloader)

    for batch, doct in enumerate(dataloader):
        print(f"Training {batch}/{l}")
        input_image = doct['input_image'].to(device)
        input_predicted_keypoints = doct['input_predicted_keypoints'] \
            .to(device)
        input_predicted_keypoints_valid = doct['input_predicted_keypoints_valid'] \
            .to(device)

        # Elements of this vector are set to 1 if the image contains a hand, 0 if it doesn't.
        gt_is_hand = doct['is_hand'].to(device)

        gt_depth = doct['gt_depth'].to(device)
        has_depth = doct['has_depth'].to(device)

        has_depth = has_depth * gt_is_hand

        # batch_size x 1 x 1
        has_depth_expanded = has_depth[:, None, None]

        # No, this isn't a bug. Currently (dec 31 2022) we only have
        # "RandoData" which is always (21,2) and no elbow and "ArtificialData"
        # which has everything.
        has_elbow_curls = has_depth[:, None]

        gt_xy = doct['gt_xy'].to(device)
        has_xy = doct['has_xy'].to(device)
        has_xy = has_xy * gt_is_hand
        # batch_size x 1 x 1 x 1
        has_xy_expanded = has_xy[:, None, None, None]

        gt_elbow = doct["elbow"].to(device)

        gt_curls = doct["curls"].to(device)

        if not settings.using_pose_predicted_input:
            input_predicted_keypoints_valid = torch.zeros(input_predicted_keypoints_valid.shape)
            input_predicted_keypoints = torch.zeros(input_predicted_keypoints.shape)

        model_pred_xy, model_pred_depth, model_extras, model_pred_curls_gnll = model(
            input_image, torch.flatten(input_predicted_keypoints, start_dim=1), input_predicted_keypoints_valid)

        # Unpack extras: index 0 is hand existence (passed through sigmoid to get a probability),
        # indices 1-3 are the elbow direction vector.
        model_pred_is_hand = torch.special.expit(model_extras[:, 0])
        model_pred_elbow = model_extras[:, 1:4]

        # Unpack curls: first 5 are the curl angles, last 5 are the predicted variances.
        model_pred_curls = model_pred_curls_gnll[:, 0:5]
        model_pred_curl_variances = model_pred_curls_gnll[:, 5:10]

        # Variance must be positive and never too close to zero — GNLL is numerically
        # unstable at very low variances, and a network can't reliably estimate uncertainty
        # to that precision anyway.
        model_pred_curl_variances = model_pred_curl_variances.abs() + settings.curl_min_variance

        # Each loss is masked by its availability flag so that samples without
        # labels for that output contribute zero gradient.
        loss_xy = mse(model_pred_xy * has_xy_expanded, gt_xy * has_xy_expanded)
        loss_depth = mse(model_pred_depth * has_depth_expanded, gt_depth * has_depth_expanded) * settings.depth_loss_mul
        loss_existence = mse(model_pred_is_hand, gt_is_hand) * settings.existence_loss_mul
        loss_elbow = mse(model_pred_elbow * has_elbow_curls, gt_elbow * has_elbow_curls) * settings.elbow_loss_mul
        loss_curls = (gnll(model_pred_curls, gt_curls, model_pred_curl_variances) * has_elbow_curls).mean() * settings.curls_loss_mul

        loss = loss_xy + loss_depth + loss_existence + loss_elbow + loss_curls

        total_loss += float(loss)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        wandb.log({
            "loss_xy": float(loss_xy),
            "loss_depth": float(loss_depth),
            "loss_existence": float(loss_existence),
            "loss_elbow": float(loss_elbow),
            "loss_curls": float(loss_curls),
        })
        freq = 300
        if (header.env_settings.loadfast):
            freq = 1
        if (batch % freq == 0):

            gt = visualizer.model_output(
                gt_xy[0].detach().cpu().numpy(),
                gt_depth[0].detach().cpu().numpy(),
                gt_elbow[0].detach().cpu().numpy(),
                gt_curls[0].detach().cpu().numpy(),
                gt_is_hand[0].detach().cpu().numpy(),
            )

            model_pred = visualizer.model_output(
                model_pred_xy[0].detach().cpu().numpy(),
                model_pred_depth[0].detach().cpu().numpy(),
                model_pred_elbow[0].detach().cpu().numpy(),
                model_pred_curls[0].detach().cpu().numpy(),
                model_pred_is_hand[0].detach().cpu().numpy(),
            )

            predinp = None
            if input_predicted_keypoints_valid[0].detach().cpu().numpy():
                predinp = input_predicted_keypoints[0].detach().cpu().numpy()

            visualizer.display_and_log_output(
                "train",
                input_image[0][0].detach().cpu().numpy(),
                model_pred,
                gt,
                predinp
            )

        loss_divisor += 1

    avg_loss = total_loss / loss_divisor

    print(
        f"Avg loss this epoch: {avg_loss}")
    return avg_loss


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_devices = 1
    # 256 OOM'd on a 7.92GB GPU (confirmed 2026-07-20). Lowered to a safer
    # default that should fit on most single GPUs; raise it back up if you
    # confirm a bigger card (e.g. the 24GB train.sbatch requests) handles it.
    batch_size_per_device = 64  # Warning: high values can OOM RAM, be careful
    if device.type == "cuda":
        num_devices = torch.cuda.device_count()
        print(f"Let's use {num_devices} GPUs!")

    wandb_name = "keypoint_estimator_training"
    # No entity= specified: this was hardcoded to "col" (the original
    # author's Collabora team), which the current wandb login has no write
    # access to and fails with a permission error. Omitting entity lets
    # wandb use whatever account is actually logged in via `wandb login`.
    if header.env_settings.wandb_enabled:
        wandb.init(project=wandb_name)
    else:
        wandb.init(project=wandb_name, mode="disabled")

    # On SLURM, cpu_count() returns all CPUs on the node, not just the ones
    # allocated to this job. SLURM_CPUS_PER_TASK is the correct value to use.
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))

    batch_size = batch_size_per_device * num_devices

    # Training: synthetic + panoptic + nikitha. freihand and tom are excluded
    # from CombinedDataset — they are held out for evaluation only.
    dataloader_train = DataLoader(
        CombinedDataset.AllOfTheDatasetsCombined(),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        timeout=100,
        persistent_workers=True,
        drop_last=True)

    # Validation (run every epoch): FreiHand — a different capture setup from
    # training data, tests whether the model generalises to a new real dataset.
    #
    # Skipped entirely in loadfast mode: RandoDataset's __init__ eagerly
    # pd.read_csv()s frei_gs.csv/tom.csv, which crashes immediately (before
    # any training batch even runs) if those real datasets aren't present.
    # loadfast is meant to be a synthetic-data-only smoke test — CombinedDataset
    # already skips real datasets for training on the same principle, this
    # just extends it to validation/test.
    dataloader_val = None
    dataloader_test = None
    if not header.env_settings.loadfast:
        dataloader_val = DataLoader(
            RandoDataset(local_config.real_datasets_basepath, "frei_gs.csv"),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers)

        # Test (run once after training): Tom OpenHands — held out entirely.
        # Replace with HOT3D / UmeTrack once those datasets are obtained, as they
        # represent true cross-device generalisation to different XR hardware.
        #
        # tom.csv doesn't exist yet as of 2026-07-20 (the source dataset is
        # harder to source than FreiHand/Panoptic) — skip gracefully rather
        # than crash on startup, same reasoning as CombinedDataset's
        # b_if_present. Final test evaluation below is skipped too if this
        # is None.
        tom_csv_path = os.path.join(local_config.real_datasets_basepath, "tom.csv")
        if os.path.exists(tom_csv_path):
            dataloader_test = DataLoader(
                RandoDataset(local_config.real_datasets_basepath, "tom.csv"),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers)
        else:
            print(f"[kpest_trainer] Skipping test set — tom.csv not found at {tom_csv_path}")

    model = KeyNet.KeyNet()
    model = torch.nn.DataParallel(model).to(device)
    optimizer = torch.optim.AdamW(model.module.parameters())

    # Use an absolute path so checkpoints are always written to the same place
    # regardless of what directory SLURM starts the job from.
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.pth')

    start_epoch = 0
    best_validation_loss = float('inf')

    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location=torch.device(device))
        if 'best_validation_loss' in checkpoint:
            best_validation_loss = checkpoint['best_validation_loss']
        start_epoch = checkpoint['epoch']
        model.module.load_state_dict(checkpoint['state_dict'])
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except BaseException:
            print("Couldn't load optimizer state dict! This shouldn't happen except for right after model weight transfers!")

    for epoch in range(start_epoch, 2000000000000):
        print(f"Epoch {epoch}\n---------------------------------------")
        wandb.log({"epoch": epoch})
        model.train()
        train_loop(device, dataloader_train, model, optimizer)

        # Skip validation and checkpointing when in fast/debug mode — model is
        # only trained on a tiny slice of data and isn't worth keeping, and
        # dataloader_val is None (real datasets weren't loaded, see above).
        if header.env_settings.loadfast:
            continue

        model.eval()
        val_result = validatoor.validation_loop(
            device, dataloader_val, model, mse, "val", False, epoch)
        mean_validation_loss = val_result.mean_loss_no_pred
        model.train()

        is_best = mean_validation_loss < best_validation_loss
        if is_best:
            best_validation_loss = mean_validation_loss

        print(f'Done with epoch {epoch} — val loss: {mean_validation_loss:.4f} (best: {best_validation_loss:.4f})')
        wandb.log({"val_loss": mean_validation_loss, "best_val_loss": best_validation_loss})

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_validation_loss': best_validation_loss,
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

    # Run the test set once after training is complete.
    # The test set is never seen during training or used for checkpoint selection,
    # so this gives an unbiased measure of final model performance.
    # (In practice the epoch loop above runs until manually stopped, so this
    # only executes if that loop is ever given a real exit condition.)
    if dataloader_test is not None:
        print("Training complete. Running final test evaluation...")
        model.eval()
        test_result = validatoor.validation_loop(
            device, dataloader_test, model, mse, "test", False, epoch)
    else:
        print("Skipping final test evaluation — tom.csv was not available.")
    test_loss = test_result.mean_loss_no_pred
    print(f"Final test loss: {test_loss:.4f}")
    wandb.log({"test_loss": test_loss})


if __name__ == "__main__":
    main()
