import ad4_stereographic_projection
import traceback


import logging
import os
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

from torch.utils.data import DataLoader

import KeyNet

import maker_of_augmentations
import a_aug_config
import py.training.common.a_geometry as geo
import settings
import kpest_header as header
import local_config


superroot = local_config.artificial_dataset_path

instance_num = 0


def init_fn(idx):
    global instance_num
    instance_num = idx


def _25_to_21(x: np.ndarray, y: np.ndarray, depth: np.ndarray) -> np.ndarray:
    start = np.zeros((25, 3))

    start[:, 0] = x
    start[:, 1] = y
    start[:, 2] = depth

    ret = np.zeros((21, 3))

    # Wrist
    ret[0] = start[0]

    # Thumb MCP
    ret[1] = start[1]
    # Thumb PXM
    ret[2] = start[2]
    # Thumb DST
    ret[3] = start[3]
    # Thumb tip
    ret[4] = start[4]

    acc_idx = 5
    # Index to pinky
    for finger in range(0, 4):
        # (not mcp.) Pxm to tip
        for joint in range(1, 5):
            pos = 5 + (finger * 5) + joint
            ret[acc_idx] = start[pos]
            acc_idx += 1
    return ret


def _25_to_21(start: np.ndarray) -> np.ndarray:
    # start = np.zeros((25, 3))

    # start[:, 0] = x
    # start[:, 1] = y
    # start[:, 2] = depth

    ret = np.zeros((21, 3))

    # Wrist
    ret[0] = start[0]

    # Thumb MCP
    ret[1] = start[1]
    # Thumb PXM
    ret[2] = start[2]
    # Thumb DST
    ret[3] = start[3]
    # Thumb tip
    ret[4] = start[4]

    acc_idx = 5
    # Index to pinky
    for finger in range(0, 4):
        # (not mcp.) Pxm to tip
        for joint in range(1, 5):
            pos = 5 + (finger * 5) + joint
            ret[acc_idx] = start[pos]
            acc_idx += 1
    return ret


def pad_int(num):
    return str(num).zfill(4)


class ArtificialDataset(torch.utils.data.Dataset):

    augmaker: maker_of_augmentations.AugmentationMaker
    # stub: Any
    num_sequences: int
    camera_poses_seq_array: np.ndarray
    hand_poses_seq_array: np.ndarray

    def __init__(self):
        self.augmaker = maker_of_augmentations.AugmentationMaker(
            a_aug_config.the_aug_config)
        self.num_sequences = len(os.listdir(superroot))

        if (header.env_settings.loadfast):
            # Was hardcoded to 25, which crashes on a smaller local set; cap at what's on disk.
            self.num_sequences = min(25, self.num_sequences)

        self.camera_poses_seq_array = np.zeros(
            (self.num_sequences, 200, 7 + 4))
        self.hand_poses_seq_array = np.zeros((self.num_sequences, 200, 26 * 7))

        for i in range(self.num_sequences):
            print(f"loading {i}", end="\r")
            seqname = f"seq{i}"
            self.camera_poses_seq_array[i] = pd.read_csv(
                os.path.join(superroot, seqname, "camera_info.csv"))
            self.hand_poses_seq_array[i] = pd.read_csv(
                os.path.join(superroot, seqname, "hand_poses.csv"))

        self.len = self.num_sequences * 200
        self.addr = 0

    def __len__(self):
        return self.len

    def __getitem__(self, idx, retries_left=5):

        seq_idx = idx // 200  # // is integer division! Rounds down!
        frame_idx = idx % 200

        seqname = f"seq{seq_idx}"

        out_joints_gt = np.zeros((25, 3), dtype=np.float32)
        out_joints_pose_predicted = np.zeros((25, 3), dtype=np.float32)
        out_image = np.zeros((128, 128), dtype=np.uint8)
        out_mask = np.zeros((128, 128), dtype=np.uint8)
        out_elbow = np.zeros((3), dtype=np.float32)
        out_curls = np.zeros((5), dtype=np.float32)

        numstr = pad_int(frame_idx)
        # Both now derive from the same `superroot`; a stale absolute path used to split them.
        # Naming matches mlib.py output (file_name{NNNN}.png); the old .jpg matched nothing.
        img_color_path = os.path.join(superroot, seqname, "imgs_color", f"file_name{numstr}.png")
        img_alpha_path = os.path.join(superroot, seqname, "imgs_alpha", f"file_name{numstr}.png")

        alpha: bool = os.path.exists(img_alpha_path)
        if not alpha:
            img_alpha_path = ""
        # print(f"getting sample, instance {instance_num}")
        try:
            ad4_stereographic_projection.prepare_sample(
                img_color_path,
                img_alpha_path,
                self.hand_poses_seq_array[seq_idx],
                self.camera_poses_seq_array[seq_idx],
                frame_idx,
                out_joints_gt,
                out_joints_pose_predicted,
                out_image,
                out_mask,
                out_elbow,
                out_curls)
        except Exception as e:
            # One missing or corrupt frame shouldn't kill a multi-hour unattended job.
            # Log it and substitute another random sample; retries are bounded.
            print(f"[ArtificialData] WARNING: failed to load {img_color_path} "
                  f"({type(e).__name__}: {e}). Substituting a random sample.")
            if retries_left <= 0:
                raise RuntimeError(
                    f"[ArtificialData] Too many consecutive failed samples — "
                    f"last attempted {img_color_path}. This looks like a "
                    f"systemic data problem, not one bad frame."
                ) from e
            return self.__getitem__(random.randrange(self.len), retries_left - 1)
        # print(f"DONE, instance {instance_num}")

        # in_img = np.frombuffer(response.image_data, dtype='uint8').copy()

        # in_img = np.reshape(in_img, (128, 128))

        keypoints_px = _25_to_21(out_joints_gt)
        if settings.using_pose_predicted_input:
            keypoints_px_pose_predicted = _25_to_21(out_joints_pose_predicted)
            # HACK: Remove depth because it was making badness
            keypoints_px_pose_predicted = keypoints_px_pose_predicted[:, :2]
        else:
            keypoints_px_pose_predicted = None

        d = self.augmaker.do_one_augmentation(
            out_image,
            keypoints_px,
            keypoints_px_pose_predicted,
            mask=out_mask if alpha else None)
        d["elbow"] = torch.from_numpy(out_elbow).float()
        d["curls"] = torch.from_numpy(out_curls).float()

        return d


if __name__ == "__main__":
    ds = ArtificialDataset()

    # Can also do "forkserver" and "spawn"
    # Both result in ✨strange gRPC errors✨ and don't seem to be faster (but
    # unsure)
    dataloader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
    # multiprocessing_context="fork")
    # dataloader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
    print(ds.__len__(), dataloader.__len__())
    for d in dataloader:
        # print(d)
        print("lol")
        guy = d["input_image"][0][0].detach().cpu().numpy()
        # kps = d["keypoints"][0].detach().cpu().numpy()
        # kps_pose_predicted = d["keypoints_predicted"][0].detach().cpu().numpy()

        guy = cv2.cvtColor(guy, cv2.COLOR_GRAY2BGR)

        # for kp in kps_pose_predicted:
        #     cv2.circle(guy, (int(kp[0]), int(kp[1])), 2, (0, 255, 0))

        # for kp in kps:
        #     cv2.circle(guy, (int(kp[0]), int(kp[1])), 2, (255, 0, 255))
    #   guy.
        cv2.imshow("h", guy)
        cv2.waitKey(0)
