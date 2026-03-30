import os
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../'))

import local_config

import CombinedDataset
from ArtificialData import ArtificialDataset
from RandoData import RandoDataset
import traceback
from dataclasses import dataclass

import logging
from typing import Any


import random
import math
import csv
import pandas as pd
import numpy as np

import subprocess
import json
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import KeyNet

import maker_of_augmentations
import a_aug_config
import py.training.common.a_geometry as geo
import settings
import kpest_header as header
import wandb
import visualizer
import validatoor
import multiprocessing

# This namedWindow call is magic, and required to make cv2.imshow() not crash on Arch. Feel free to comment it out but please don't remove
# cv2.namedWindow('a', 0)


# This is useful if your loss suddenly goes to inf/nan, it's just `feenableexcept` but for PyTorch/CUDA
# Quite slow, keep it off if you don't need it.
# torch.autograd.set_detect_anomaly(True)


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
        #
        gt_is_hand = doct['is_hand'].to(device)
        is_hand_expanded_for_scalars = gt_is_hand[:, None]

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
        print(gt_curls.shape)

        if (not settings.using_pose_predicted_input):
            input_predicted_keypoints_valid = torch.zeros(
                input_predicted_keypoints_valid.shape)
            input_predicted_keypoints = torch.zeros(
                (input_predicted_keypoints.shape))
            print(input_predicted_keypoints.shape)
            # input_predicted_keypoints_real = torch.zeros((1, 63))

        model_pred_xy, model_pred_depth, model_extras, model_pred_curls_gnll = model(
            input_image, torch.flatten(
                input_predicted_keypoints, start_dim=1), input_predicted_keypoints_valid)

        model_pred_is_hand = torch.special.expit(model_extras[:, 0])

        model_pred_elbow = model_extras[:, 1:4]

        model_pred_curls = model_pred_curls_gnll[:, 0:5]
        model_pred_curl_variances = model_pred_curls_gnll[:, 5:10]

        # We keep the minimum variance well above zero, because of how crazy GNLL is with low variances
        # and because it's not... really, possible? for ML to estimate variance
        # super accurately
        model_pred_curl_variances = model_pred_curl_variances.abs() + \
            settings.curl_min_variance

        loss_xy = mse(model_pred_xy * has_xy_expanded,
                      gt_xy * has_xy_expanded)

        loss_depth = mse(model_pred_depth * has_depth_expanded,
                         gt_depth * has_depth_expanded) * 0.03  # ??

        loss_existence = mse(model_pred_is_hand, gt_is_hand) * \
            settings.existence_loss_mul

        loss_elbow = mse(model_pred_elbow * has_elbow_curls,
                         gt_elbow * has_elbow_curls) * settings.elbow_loss_mul

        loss_curls = (
            gnll(
                model_pred_curls,
                gt_curls,
                model_pred_curl_variances) * has_elbow_curls).mean() * settings.curls_loss_mul

        loss = loss_xy + loss_depth + loss_existence + loss_elbow + loss_curls

        total_loss += float(loss)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        wandb.log({f"loss_xy": float(loss_xy),
                  "loss_depth": float(loss_depth),
                   "loss_existence": float(loss_existence),
                   "loss_elbow": float(loss_elbow),
                   "loss_curls": float(loss_curls)})
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


@dataclass
class val_dataset_and_name:
    dataset: torch.utils.data.Dataset
    name: str
    use_prediction_too: bool = False
    dataloader: torch.utils.data.DataLoader = None


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_devices = 1
    batch_size_per_device = 64
    batch_size_per_device = 256
    if (device == "cuda:0"):
        num_devices = torch.cuda.device_count()
        print(f"Let's use {num_devices} GPUs!")
        # Warning, this can OOM your RAM if too high. At least right now. Be
        # careful :)
    wandb_name = "free_scans_2d_input_jan17"
    if header.env_settings.wandb_enabled:
        wandb.init(project=wandb_name, entity="col")
    else:
        wandb.init(project=wandb_name, entity="col", mode="disabled")
    # hd_train = ArtificialData.ArtificialDataset(
    #     maker_of_augmentations.AugmentationMaker(a_aug_config.the_aug_config))
    # d = torch.utils.data.ConcatDataset([a, b])

    hd_train = CombinedDataset.AllOfTheDatasetsCombined()

    # pls work aaaaaaaaa
    dataloader_train = DataLoader(
        hd_train,
        batch_size=batch_size_per_device *
        num_devices,
        shuffle=True,
        num_workers=multiprocessing.cpu_count(),
        timeout=100,
        persistent_workers=True,
        drop_last=True)
    #   worker_init_fn=ArtificialData.init_fn)

    val_datasets = []

    if True:
        # val_datasets.append(val_dataset_and_name(ArtificialDataset(
        #     "/media/moses/raid/inshallah8_validation/", a_aug_config.aug_config_validatoor_not_hand, validation_dataset=True), "not_hand", True))
        # val_datasets.append(val_dataset_and_name(ArtificialDataset(
        #     "/media/moses/raid/inshallah8_validation/", a_aug_config.aug_config_validatoor, validation_dataset=True), "artificial", True))

        val_datasets.append(
            val_dataset_and_name(
                RandoDataset(
                    local_config.real_datasets_basepath,
                    "panoptic_manual.csv"),
                "panoptic_manual"))
        # Disabling panoptic_panoptic because it's big and inaccurate
        # val_datasets.append(val_dataset_and_name(RandoDataset(
        # "panoptic_panoptic"))
        val_datasets.append(val_dataset_and_name(RandoDataset(
            local_config.real_datasets_basepath, "frei_gs.csv"), "freihand"))
        val_datasets.append(val_dataset_and_name(RandoDataset(
            local_config.real_datasets_basepath, "tom.csv"), "tom_openhands"))

    for vn in val_datasets:
        vn.dataloader = DataLoader(
            vn.dataset,
            batch_size=batch_size_per_device * num_devices,
            shuffle=False,
            num_workers=24
        )

    model = KeyNet.KeyNet()

    model = torch.nn.DataParallel(model).to(device)

    optimizer = torch.optim.AdamW(model.module.parameters())

    checkpoint_file = os.path.join(
        f"checkpoints", 'checkpoint.pth'
    )

    start_epoch = 0
    last_validation_loss = 10000000000

    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(
            checkpoint_file, map_location=torch.device(device))

        # Can delete this later btw
        if 'validation_loss' in checkpoint.keys():
            last_validation_loss = checkpoint['validation_loss']
        start_epoch = checkpoint['epoch']
        model.module.load_state_dict(checkpoint['state_dict'])
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except BaseException:
            print("Couldn't load optimizer state dict! This shouldn't happen except for right after model weight transfers!")

    for epoch in range(start_epoch, 2000000000000):
        print(f"Epoch {epoch}\n---------------------------------------")
        wandb.log({"epoch": epoch})
        avg_loss_train = train_loop(device, dataloader_train,
                                    model, optimizer)

        model.eval()
        for vn in val_datasets:
            a = validatoor.validation_loop(
                device,
                vn.dataloader,
                model,
                mse,
                vn.name,
                vn.use_prediction_too,
                epoch)
            validation_loss = a.mean_loss_no_pred + a.mean_loss_pred
        model.train()

        # We don't want to save models trained on 0.1% of our dataset.
        if (header.env_settings.loadfast):
            continue
        # losses: validatoor.validation_losses = validatoor.validation_losses(0, 0)
        # model.eval()
        # for vn in val_datasets:
        #     a = validatoor.validation_loop(
        #         device, vn.dataloader, model, loss_fn, vn.name, vn.use_prediction_too, epoch)
        #     validation_loss = a.mean_loss_no_pred + a.mean_loss_pred
        # model.train()
        final_output_dir = f"checkpoints"
        # best_model = validation_loss < last_validation_loss
        # if avg_loss_train < validation_loss:
        #     best_model = True

        print(f'Done with epoch {epoch}')

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, final_output_dir)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, saving extra!")
            os.system(
                f"cp {final_output_dir}/checkpoint.pth {final_output_dir}/checkpoint_{epoch}.pth")
        # if best_model:
        #     print(
        #         f"Best! Saving as such! (last best was {last_validation_loss} new best is {validation_loss}")
        #     validation_loss = avg_loss_train
        #     os.system(
        # f"cp {final_output_dir}/checkpoint.pth
        # {final_output_dir}/checkpoint_best.pth")
        else:
            print(f"Not best performance!")


if __name__ == "__main__":
    main()
