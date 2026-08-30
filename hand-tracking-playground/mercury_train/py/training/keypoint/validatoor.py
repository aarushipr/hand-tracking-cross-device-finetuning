import os
import torch
import torch.multiprocessing
import numpy as np
import wandb
from dataclasses import dataclass
import settings

# https://gitanswer.com/pytorch-too-many-open-files-error-cplusplus-356516297
torch.multiprocessing.set_sharing_strategy('file_system')


@dataclass
class validation_losses:
    mean_loss_no_pred: float
    mean_loss_pred: float


def validation_loop_just_one(
        device,
        dataloader,
        model,
        loss_fn,
        output_folder,
        epoch_num,
        use_prediction) -> float:

    l = len(dataloader)
    loss_array = np.empty((l))
    loss_array_depth = np.empty((l))
    loss_array_existence = np.empty((l))

    pstring = "use_prediction" if use_prediction else "no_use_prediction"

    with torch.no_grad():
        for batch, doct in enumerate(dataloader):
            print(f"Validating {output_folder} {pstring} {batch}/{l}")

            input_image = doct['input_image'].to(device)
            input_predicted_keypoints = doct['input_predicted_keypoints'].to(device)
            input_predicted_keypoints_valid = doct['input_predicted_keypoints_valid'].to(device)

            gt_xy = doct['gt_xy'].to(device)
            gt_depth = doct['gt_depth'].to(device)
            has_depth = doct['has_depth'].to(device)
            gt_is_hand = doct['is_hand'].to(device)

            # Apply the same masking as the training loop so validation loss
            # is comparable. has_depth is 0 for datasets without depth labels —
            # without this mask, those samples would incorrectly penalise the
            # model on their zero-depth ground truth values.
            #
            # Per-joint validity (see HOT3DKeypointDataset._project_hand): a
            # hand can have some joints usable and others not, so xy and
            # depth get masked per joint, matching kpest_trainer.py's
            # train_batch exactly so train and validation loss stay
            # comparable.
            depth_valid_per_joint = doct['depth_valid_per_joint'].to(device)
            has_depth_expanded = (depth_valid_per_joint * gt_is_hand[:, None])[:, :, None]

            xy_valid_per_joint = doct['xy_valid_per_joint'].to(device)
            has_xy_expanded = (xy_valid_per_joint * gt_is_hand[:, None])[:, :, None, None]

            if not use_prediction:
                input_predicted_keypoints = torch.zeros(
                    input_predicted_keypoints.shape, dtype=torch.float32).to(device)
                input_predicted_keypoints_valid = torch.zeros(
                    input_predicted_keypoints_valid.shape, dtype=torch.float32).to(device)

            model_pred_xy, model_pred_depth, model_extras, _ = model(
                input_image,
                torch.flatten(input_predicted_keypoints, start_dim=1),
                input_predicted_keypoints_valid)

            model_pred_is_hand = model_extras[:, 0]

            loss_hmap = loss_fn(model_pred_xy * has_xy_expanded, gt_xy * has_xy_expanded)
            loss_depth = loss_fn(model_pred_depth * has_depth_expanded, gt_depth * has_depth_expanded) * settings.depth_loss_mul
            loss_existence = loss_fn(model_pred_is_hand, gt_is_hand) * settings.existence_loss_mul

            loss_array[batch] = float(loss_hmap)
            loss_array_depth[batch] = float(loss_depth)
            loss_array_existence[batch] = float(loss_existence)

    return np.mean(loss_array), np.mean(loss_array_depth), np.mean(loss_array_existence)


def validation_loop(
        device,
        dataloader,
        model,
        loss_fn,
        output_folder,
        artificial_dataset,
        epoch_num) -> validation_losses:

    if artificial_dataset:
        mean_loss_pred, mean_loss_pred_depth, mean_loss_pred_existence = validation_loop_just_one(
            device, dataloader, model, loss_fn, output_folder, epoch_num, True)
        mean_loss_no_pred, mean_loss_no_pred_depth, mean_loss_no_pred_existence = validation_loop_just_one(
            device, dataloader, model, loss_fn, output_folder, epoch_num, False)
        wandb.log({
            f"{output_folder}_validation_loss_xy": mean_loss_no_pred,
            f"{output_folder}_validation_loss_xy_use_prediction": mean_loss_pred,
            f"{output_folder}_validation_loss_depth": mean_loss_no_pred_depth,
            f"{output_folder}_validation_loss_depth_use_prediction": mean_loss_pred_depth,
            f"{output_folder}_validation_loss_existence": mean_loss_no_pred_existence,
            f"{output_folder}_validation_loss_existence_use_prediction": mean_loss_pred_existence,
        })
        return validation_losses(
            mean_loss_no_pred=mean_loss_no_pred,
            mean_loss_pred=mean_loss_pred)
    else:
        mean_loss_no_pred, mean_loss_no_pred_depth, mean_loss_no_pred_existence = validation_loop_just_one(
            device, dataloader, model, loss_fn, output_folder, epoch_num, False)
        wandb.log({
            f"{output_folder}_validation_loss_xy": mean_loss_no_pred,
            f"{output_folder}_validation_loss_depth": mean_loss_no_pred_depth,
            f"{output_folder}_validation_loss_existence": mean_loss_no_pred_existence,
        })
        return validation_losses(
            mean_loss_no_pred=mean_loss_no_pred,
            mean_loss_pred=0)
