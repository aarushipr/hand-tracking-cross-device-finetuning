import typing
from ArtificialData import ArtificialDataset
import KeyNet
import wandb
import select
import sys
from torch.utils.data import DataLoader
import torch.multiprocessing
import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import py.training.common.a_geometry as geo
import cv2
import visualizer
from dataclasses import dataclass
import settings

# This namedWindow call is magic, and required to make cv2.imshow() not crash on Arch. Feel free to comment it out but please don't remove
# cv2.namedWindow('a', 0)


# https://gitanswer.com/pytorch-too-many-open-files-error-cplusplus-356516297
torch.multiprocessing.set_sharing_strategy('file_system')


@dataclass
class validation_losses:
    mean_loss_no_pred: float
    mean_loss_pred: float
    # mean_loss_existence: float


def validation_loop_just_one(
        device,
        dataloader,
        model,
        loss_fn,
        output_folder,
        epoch_num,
        use_prediction) -> float:

    l = len(dataloader)
    # total_loss = 0
    # loss_divisor = 0

    loss_array = np.empty((l))

    loss_array_depth = np.empty((l))

    loss_array_existence = np.empty((l))

    if use_prediction:
        pstring = "use_prediction"
    else:
        pstring = "no_use_prediction"
    for batch, doct in enumerate(dataloader):
        print(f"Validating {output_folder} {pstring} {batch}/{l}")
        input_image = doct['input_image'].to(device)
        input_predicted_keypoints = doct['input_predicted_keypoints'] \
            .to(device)
        input_predicted_keypoints_valid = doct['input_predicted_keypoints_valid'] \
            .to(device)

        gt_depth = doct['gt_depth'].to(device)
        has_depth = doct['has_depth'].to(device)

        gt_is_hand = doct['is_hand'].to(device)

        if not use_prediction:
            input_predicted_keypoints = torch.zeros(
                input_predicted_keypoints.shape,
                dtype=torch.float32).to(device)
            input_predicted_keypoints_valid = torch.zeros(
                input_predicted_keypoints_valid.shape, dtype=torch.float32).to(device)

        gt_xy = doct['gt_xy'].to(device)

        model_pred_xy, model_pred_depth, model_extras, model_pred_curls_gnll = model(
            input_image, torch.flatten(
                input_predicted_keypoints, start_dim=1), input_predicted_keypoints_valid)

        model_is_hand = model_extras[:, 0]

        loss_is_hand = loss_fn(gt_is_hand, model_is_hand) * \
            settings.existence_loss_mul

        loss_hmap = loss_fn(model_pred_xy, gt_xy)

        # XXX: Rethink this! A lot of datasets don't have depth.
        loss_depth = loss_fn(model_pred_depth, gt_depth) * 0.03

        loss_existence = loss_fn(model_is_hand, gt_is_hand)

        # loss = loss_hmap

        loss_array[batch] = loss_hmap
        loss_array_depth[batch] = loss_depth
        loss_array_existence[batch] = loss_is_hand

        # total_loss += float(loss)

        name = f"validation_{pstring}"

        # imgo = visualizer.make_visualization_images(
        #     input_image[0][0].detach().cpu().numpy(),
        #     gt_xy[0].detach().cpu().numpy(),
        #     model_pred_xy[0].detach().cpu().numpy(),
        #     gt_depth[0].detach().cpu().numpy(),
        #     model_pred_depth[0].detach().cpu().numpy(),
        #     pose_prediction=input_predicted_keypoints[0].detach(
        #     ).cpu().numpy().reshape((21, 3)),
        #     pose_prediction_valid=int(
        #         input_predicted_keypoints_valid[0].detach().cpu().numpy()),
        #     pred_is_hand = model_is_hand[0].detach().cpu().numpy()
        # )
        # out_folder: os.path = os.path.join(
        #     f"validation_result_{output_folder}_{pstring}", f"img{batch}")

        # os.makedirs(out_folder, exist_ok=True)

        # out_path: os.path = os.path.join(out_folder, f"epoch{epoch_num}.jpg")

        # geo.rgb_imwrite(out_path, imgo)

    return np.mean(loss_array), np.mean(
        loss_array_depth), np.mean(loss_array_existence)


def validation_loop(
        device,
        dataloader,
        model,
        loss_fn,
        output_folder,
        artificial_dataset,
        epoch_num) -> validation_losses:

    if (artificial_dataset):
        mean_loss_pred, mean_loss_pred_depth, mean_loss_pred_existence = validation_loop_just_one(
            device, dataloader, model, loss_fn, output_folder, epoch_num, True)
        mean_loss_no_pred, mean_loss_no_pred_depth, mean_loss_no_pred_existence = validation_loop_just_one(
            device, dataloader, model, loss_fn, output_folder, epoch_num, False)
        wandb.log({f"{output_folder}_validation_loss_xy_use_prediction": mean_loss_pred,
                   f"{output_folder}_validation_loss_xy": mean_loss_no_pred,
                   f"{output_folder}_validation_loss_depth_use_prediction": mean_loss_pred_depth,
                   f"{output_folder}_validation_loss_depth": mean_loss_no_pred_depth,
                   f"{output_folder}_validation_loss_existence_use_prediction": mean_loss_pred_existence,
                   f"{output_folder}_validation_loss_existence": mean_loss_no_pred_existence})

        return validation_losses(
            mean_loss_no_pred=mean_loss_no_pred,
            mean_loss_pred=mean_loss_pred)
    else:
        mean_loss_no_pred, mean_loss_no_pred_depth, mean_loss_no_pred_existence = validation_loop_just_one(
            device, dataloader, model, loss_fn, output_folder, epoch_num, False)
        wandb.log(
            {f"{output_folder}_validation_loss_xy": mean_loss_no_pred,  # , f"{output_folder}_validation_loss_depth": mean_loss_no_pred_depth })
             f"{output_folder}_validation_loss_existence": mean_loss_no_pred_existence})
        return validation_losses(
            mean_loss_no_pred=mean_loss_no_pred,
            mean_loss_pred=0)


def main():

    # Subsequent GPU/CPU stuff is cursed but I will not make it perfect, sorry.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # CPU defaults
    num_devices = 1
    batch_size_per_device = 32

    wandb.init(project="artificial_data", entity="col",
               name="0", mode="disabled")

    hd_val = ArtificialDataset("/3/inshallah7_validation/")

    dataloader_val = DataLoader(
        hd_val,
        batch_size=batch_size_per_device *
        num_devices,
        shuffle=True,
        num_workers=1)

    model = KeyNet.KeyNet()

    model = torch.nn.DataParallel(model).to(device)

    checkpoint_file = os.path.join(
        f"checkpoints_normal", 'checkpoint.pth'
    )

    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(
            checkpoint_file, map_location=torch.device(device))
        model.module.load_state_dict(checkpoint['state_dict'])

    loss_fn = nn.MSELoss(reduction='mean')

    validation_loop(device, dataloader_val, model, loss_fn, gui=True)


if __name__ == "__main__":
    main()
