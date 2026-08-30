import os
import sys
import shutil

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../'))

import local_config

import py.training.common.hot3d_split as hot3d_split
from HOT3DKeypointDataset import HOT3DKeypointDataset, worker_init as hot3d_worker_init
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
from load_weights import load_keynet_weights


# https://gitanswer.com/pytorch-too-many-open-files-error-cplusplus-356516297
torch.multiprocessing.set_sharing_strategy('file_system')

mse = nn.MSELoss(reduction='mean')
gnll = nn.GaussianNLLLoss(reduction='none')

# --- Fine-tuning schedule (Chapter 4, Table 4.1) ---------------------------
# Hard ceiling on epochs. Training normally stops earlier, through the
# early-stopping patience below; this only bounds the SLURM job.
MAX_EPOCHS = 40

# Stop after this many consecutive epochs with no improvement in validation
# loss. Deliberately generous: the validation split is a handful of HOT3D
# sequences, so epoch-to-epoch validation loss is noisy, and a tight patience
# would stop on that noise rather than on genuine convergence.
EARLY_STOPPING_PATIENCE = 8

# Keep every Nth frame of each HOT3D recording. The cameras run at 30 Hz, so
# consecutive frames are near-duplicates -- see HOT3DKeypointDataset's
# docstring for the full reasoning.
#
# 5, not 10: KeyNet fine-tunes 830,016 trainable parameters (the frozen
# image_network is only 16.3% of the network), so the ratio of trainable
# parameters to training samples is the weakest point in the procedure.
# Stride 5 roughly doubles the training set for the same parameter count,
# at ~1.5 h/epoch against the 72 h job limit -- affordable, and it directly
# addresses the overfitting risk. Raise back to 10 if epoch time turns out
# materially worse than that in practice.
HOT3D_FRAME_STRIDE = 5


def save_checkpoint(states, output_dir, filename='checkpoint.pth'):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(states, os.path.join(output_dir, filename))


def train_loop(device, dataloader, model, optimizer):
    total_loss = 0
    # Accumulated separately from total_loss so the epoch mean of the xy term
    # alone can be logged. validatoor's validation loss is the xy heatmap term
    # ONLY (see validation_loop_just_one: loss_array holds loss_hmap), whereas
    # total_loss here is the full objective, xy + depth_loss_mul * depth. The
    # two are therefore not comparable, and plotting them against each other
    # as "training vs validation loss" would compare an objective against one
    # of its own components. train_loss_xy is the like-for-like counterpart.
    total_loss_xy = 0
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

        # No, this isn't a bug. Currently (dec 31 2022) we only have
        # "RandoData" which is always (21,2) and no elbow and "ArtificialData"
        # which has everything.
        has_elbow_curls = has_depth[:, None]

        # Per-joint validity (see HOT3DKeypointDataset._project_hand): a
        # hand can have some joints usable and others not, so xy and depth
        # get masked per joint here instead of once for the whole sample.
        # batch_size x 21 x 1
        depth_valid_per_joint = doct['depth_valid_per_joint'].to(device)
        has_depth_expanded = (depth_valid_per_joint * gt_is_hand[:, None])[:, :, None]

        gt_xy = doct['gt_xy'].to(device)
        # batch_size x 21 x 1 x 1
        xy_valid_per_joint = doct['xy_valid_per_joint'].to(device)
        has_xy_expanded = (xy_valid_per_joint * gt_is_hand[:, None])[:, :, None, None]

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
        total_loss_xy += float(loss_xy)

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
    avg_loss_xy = total_loss_xy / loss_divisor

    print(
        f"Avg loss this epoch: {avg_loss} (xy term alone: {avg_loss_xy})")
    return avg_loss, avg_loss_xy

def set_train_mode(model):
    model.train()
    model.module.image_network.eval()
    
    
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

    # ------------------------------------------------------------------
    # Data: HOT3D only, Aria only.
    #
    # Training and validation both come out of hot3d_split's "train" split,
    # which is Aria-only by design: Quest recordings are never seen during
    # training, so that evaluating on Quest measures generalisation to an
    # unseen *device* rather than merely to unseen subjects. See
    # py/training/common/hot3d_split.py for the full split rationale.
    #
    # There is deliberately no test set here. All evaluation lives in
    # evaluate_keypoint.py, so that the zero-shot Monado baseline and this
    # fine-tuned model are scored by exactly the same code path, and the
    # metric can be changed without retraining anything.
    # ------------------------------------------------------------------
    train_pool = hot3d_split.list_sequence_dirs(local_config.hot3d_dataset_path, "train")
    if not train_pool:
        raise RuntimeError(
            f"[kpest_trainer] No HOT3D Aria training sequences found under "
            f"{local_config.hot3d_dataset_path} -- nothing to train on.")

    train_dirs, val_dirs = hot3d_split.split_train_val(train_pool)

    # loadfast is a smoke test: prove the pipeline runs end to end in
    # minutes, not produce a model worth keeping. Two training sequences and
    # one validation sequence exercise every code path below.
    if header.env_settings.loadfast:
        train_dirs = train_dirs[:2]
        val_dirs = val_dirs[:1] or train_dirs[:1]

    print(f"[kpest_trainer] {len(train_dirs)} train / {len(val_dirs)} val HOT3D "
          f"sequences, frame_stride={HOT3D_FRAME_STRIDE}")

    # num_workers>0 is safe here ONLY because of HOT3DKeypointDataset's
    # worker_init(). DataLoader workers are forked processes, so they would
    # otherwise inherit the parent's already-open Hot3dDataProvider objects,
    # and two processes reading the same C++ VRS file handle is exactly what
    # produced garbled timestamps and JPEG decode failures before, then
    # crashed with "DataLoader worker exited unexpectedly". worker_init
    # clears each worker's provider cache so every worker opens its own
    # handles and none is ever shared.
    #
    # This is the "proper fix" the previous num_workers=0 comment described
    # as existing-but-unbuilt. It became possible only once the sample index
    # stopped holding live provider objects (INDEX_FORMAT_VERSION 2).
    #
    # Loading is bound by .vrs image reads over network storage -- measured
    # at ~0.15 s per sample on the cluster, at under 10% CPU -- so this is
    # close to a linear speedup in the number of workers. persistent_workers
    # keeps them (and their open providers) alive between epochs, so the
    # per-epoch reopen cost is paid once.
    num_workers = min(4, int(os.environ.get("SLURM_CPUS_PER_TASK",
                                            multiprocessing.cpu_count())))

    def make_hot3d_loader(sequence_dirs, shuffle, drop_last):
        return DataLoader(
            HOT3DKeypointDataset(
                sequence_dirs=sequence_dirs,
                hot3d_repo_root=local_config.hot3d_repo_root,
                object_library_path=local_config.hot3d_object_library_path,
                frame_stride=HOT3D_FRAME_STRIDE,
                index_cache_dir=getattr(local_config, "hot3d_index_cache_dir", None),
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            worker_init_fn=hot3d_worker_init,
            timeout=0,
            persistent_workers=num_workers > 0,
            drop_last=drop_last)

    dataloader_train = make_hot3d_loader(train_dirs, shuffle=True, drop_last=True)
    dataloader_val = make_hot3d_loader(val_dirs, shuffle=False, drop_last=False)

    model = KeyNet.KeyNet()
    load_keynet_weights(model)
    
    for param in model.image_network.parameters():
        param.requires_grad = False
    
    model = torch.nn.DataParallel(model).to(device)
    model.module.image_network.eval()
    
    
    trainable_params = (p for p in model.module.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(trainable_params)

    start_epoch = 0
    best_validation_loss = float('inf')

    # Use an absolute path so checkpoints are always written to the same place
    # regardless of what directory SLURM starts the job from.
    # Smoke-test runs get their own checkpoint directory. Otherwise a
    # loadfast run would write checkpoint.pth into the real one, and the
    # resume block just below would silently pick up a model trained on two
    # sequences at the start of the next real run.
    checkpoint_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "checkpoints_loadfast" if header.env_settings.loadfast else "checkpoints")
    checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.pth')

    if os.path.exists(checkpoint_file):
        # weights_only=False: PyTorch 2.6 flipped this default to True, which
        # refuses any checkpoint containing a non-tensor object -- including
        # the numpy scalar that best_validation_loss used to be. These are
        # checkpoints this script wrote itself, not untrusted files, so the
        # restriction buys nothing here and breaks resume-after-preemption.
        checkpoint = torch.load(checkpoint_file, map_location=torch.device(device),
                                weights_only=False)
        if 'best_validation_loss' in checkpoint:
            best_validation_loss = checkpoint['best_validation_loss']
        start_epoch = checkpoint['epoch']
        model.module.load_state_dict(checkpoint['state_dict'])
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except BaseException:
            print("Couldn't load optimizer state dict! This shouldn't happen except for right after model weight transfers!")

    epochs_without_improvement = 0

    for epoch in range(start_epoch, MAX_EPOCHS):
        print(f"Epoch {epoch}\n---------------------------------------")
        wandb.log({"epoch": epoch})
        set_train_mode(model)
        # Capture the epoch-mean training loss and log it alongside the
        # validation loss. Without this only per-batch training loss reaches
        # wandb, which is too noisy to plot against a per-epoch validation
        # curve -- and train-versus-validation on shared axes is exactly the
        # figure that shows whether the trainable head overfits.
        mean_training_loss, mean_training_loss_xy = train_loop(
            device, dataloader_train, model, optimizer)

        model.eval()
        val_result = validatoor.validation_loop(
            device, dataloader_val, model, mse, "val", False, epoch)
        mean_validation_loss = val_result.mean_loss_no_pred
        set_train_mode(model)

        is_best = mean_validation_loss < best_validation_loss
        if is_best:
            best_validation_loss = mean_validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(f"Done with epoch {epoch} -- train loss: {mean_training_loss:.4f} "
              f"(xy {mean_training_loss_xy:.4f}), "
              f"val loss: {mean_validation_loss:.4f} "
              f"(best: {best_validation_loss:.4f}; "
              f"{epochs_without_improvement}/{EARLY_STOPPING_PATIENCE} epochs "
              f"without improvement)")
        wandb.log({
            # train_loss is the full objective; train_loss_xy is the term that
            # is directly comparable to val_loss. Plot train_loss_xy against
            # val_loss for the convergence/overfitting figure.
            "train_loss": mean_training_loss,
            "train_loss_xy": mean_training_loss_xy,
            "val_loss": mean_validation_loss,
            "best_val_loss": best_validation_loss,
            "epochs_without_improvement": epochs_without_improvement,
        })

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            # float(), not the numpy scalar np.mean() returns: keeping the
            # checkpoint free of numpy objects means it also loads under
            # torch.load's stricter weights_only=True default.
            'best_validation_loss': float(best_validation_loss),
        }, checkpoint_dir)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, saving extra checkpoint!")
            shutil.copy(
                os.path.join(checkpoint_dir, "checkpoint.pth"),
                os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pth"))

        if is_best:
            print("Best model so far! Saving as checkpoint_best.pth")
            shutil.copy(
                os.path.join(checkpoint_dir, "checkpoint.pth"),
                os.path.join(checkpoint_dir, "checkpoint_best.pth"))

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping: validation loss has not improved for "
                  f"{EARLY_STOPPING_PATIENCE} consecutive epochs.")
            break
    else:
        print(f"Reached the MAX_EPOCHS ceiling of {MAX_EPOCHS} without "
              f"early stopping triggering.")

    # No test evaluation here by design -- see the data section above.
    best_checkpoint = os.path.join(checkpoint_dir, "checkpoint_best.pth")
    print(f"\nTraining complete. Best validation loss: {best_validation_loss:.4f}")
    print(f"Best checkpoint: {best_checkpoint}")
    print("Score it against the held-out splits with, e.g.:")
    print(f"  python py/training/keypoint/evaluate_keypoint.py "
          f"--weights {best_checkpoint} --split test_aria")
    wandb.log({"final_best_val_loss": best_validation_loss})


if __name__ == "__main__":
    main()
