import math
import typing
import pandas as pd
import random
import numpy as np
import torch
import cv2
import os

from a_aug_config import the_aug_config, aug_config_validatoor

from maker_of_augmentations import AugmentationMaker

from local_config import artificial_dataset_path as datasets_basepath

import py.training.common.a_geometry as geo


np.set_printoptions(precision=3, suppress=True)


# Get a bounding square of some keypoints
def bsqr(pts):  # -> typing.Tuple(np.ndarray, float)
    min_ = pts[0].copy()
    max_ = min_.copy()

    for i in range(len(pts)):
        kp = pts[i]
        min_[0] = min(kp[0], min_[0])
        min_[1] = min(kp[1], min_[1])
        max_[0] = max(kp[0], max_[0])
        max_[1] = max(kp[1], max_[1])

    center = (min_ + max_) / 2

    width = abs(min_[0] - max_[0])
    height = abs(min_[1] - max_[1])

    square_side = max(width, height)

    return center, square_side


def palm_length_2d(pts):

    wrist = pts[0]
    middle_proximal = pts[9]
    index_proximal = pts[5]
    ring_proximal = pts[17]

    fwd = np.linalg.norm(wrist - middle_proximal)  # Just 2D vector length
    side = np.linalg.norm(index_proximal - ring_proximal)

    length = max(fwd, side)
    return length


def uniformcr(c, r):
    return random.uniform(c - r, c + r)


def rotate_hand(kps, mat):
    kps_new = np.zeros((21, 2))
    for i in range(21):
        kps_new[i] = geo.transformVecBy2x3(kps[i][:2], mat)
    return kps_new


def crop(img: np.ndarray, keypoints: np.ndarray, flip: bool):
    # Pick an amount we want to rotate
    rotate: float = np.random.uniform(0, math.pi * 2)

    keypoints_new = np.zeros((21, 2))

    # Get a rotation matrix that corresponds to that value
    m = geo.rotation_matrix(rotate)

    # Get the inverse of that matrix too
    # (Matrix inverse would work too, I just happen to know that this is also an inverse, in this special case)
    back = geo.rotation_matrix(-rotate)

    # Rotate all the keypoints in the original image space
    keypoints_new = rotate_hand(keypoints, m)

    # [0.81, 1.18]
    radius_change_from_average = uniformcr(1.65, 0.2) / 1.65

    # Get bounding box *in rotated space*
    center, radius = bsqr(keypoints_new)

    radius2 = palm_length_2d(keypoints_new)

    radius *= 1.65
    radius2 *= 2.2

    radius = max(radius, radius2)

    radius *= radius_change_from_average

    halfr = radius / 2
    down = np.array((0, halfr))
    right = np.array((halfr, 0))

    # Probably enough to put the hand outside.
    if False:
        amt_move = radius * 0.3
        center += np.random.uniform(-amt_move, amt_move, (2))

    if flip:
        right *= -1

    tl = center - down - right
    tr = center - down + right
    bl = center + down - right
    br = center + down + right

    tl = geo.transformVecBy2x3(tl, back)
    tr = geo.transformVecBy2x3(tr, back)
    bl = geo.transformVecBy2x3(bl, back)
    br = geo.transformVecBy2x3(br, back)

    if False:
        color = (255, 0, 255)
        cv2.line(img, geo.cvtup(tl), geo.cvtup(tr), color)
        cv2.line(img, geo.cvtup(tr), geo.cvtup(br), color)
        cv2.line(img, geo.cvtup(br), geo.cvtup(bl), color)
        cv2.line(img, geo.cvtup(bl), geo.cvtup(tl), color)

    os = 128

    tl_o = (0, 0)
    tr_o = (os, 0)
    bl_o = (0, os)

    src_tri = np.float32((tl, tr, bl))
    dst_tri = np.float32((tl_o, tr_o, bl_o))

    trans = cv2.getAffineTransform(src_tri, dst_tri)

    return trans


def add_2d_noise_to_keypoints(kps_in: np.ndarray):

    homothety_scale: float = random.uniform(0.8, 1.2)
    # homothety_scale: float  = random.uniform(0.7, 1.4)
    rotate_amount: float = np.random.normal(0, 0.4)
    # Homothety
    center = kps_in[9][:2].copy()

    # Move to origin
    kps = kps_in - center
    # Multiply by some amount (around hand center)
    kps *= homothety_scale

    # Rotate and shear by some amount (around hand center)
    m = geo.rotation_matrix(rotate_amount)
    m[:2, :2] += np.random.normal(0, 0.15)

    kps = rotate_hand(kps, m)

    kps += center

    _, sz = bsqr(kps)

    stddev_overall = sz * 0.3
    stddev_per_joint = sz * 0.07

    kps += np.random.normal(0, stddev_overall, 2)

    kps += np.random.normal(0, stddev_per_joint, (21, 2))

    return kps


class RandoDataset(torch.utils.data.Dataset):
    base_path: str  # = f"{datasets_basepath}/munge_april26"
    source: str  # = "nikitha.csv"
    num_times_to_repeat: int

    def __init__(self, base_path, source):
        self.base_path = base_path
        self.source = source
        self.augmaker = AugmentationMaker(aug_config_validatoor)
        self.csvframe = pd.read_csv(os.path.join(
            self.base_path, self.source), delimiter=" ", quotechar="|")
        self.num_times_to_repeat = 1
        self.actual_size = len(self.csvframe)

    def __len__(self):
        return self.actual_size * self.num_times_to_repeat

    def __getitem__(self, idx):
        idx = idx % self.actual_size
        b = self.csvframe.iloc[idx]

        acc_idx = 0

        filename = b[acc_idx]
        acc_idx += 1

        kps = np.zeros((22, 3))
        gt_xy_valid = np.zeros((22))
        gt_depth_valid = np.zeros((22))

        for i in range(22):
            for j in range(3):
                kps[i][j] = b[acc_idx]
                acc_idx += 1

        for i in range(22):
            gt_xy_valid[i] = b[acc_idx]
            acc_idx += 2
            gt_depth_valid[i] = b[acc_idx]
            acc_idx += 1

        is_right = bool(b[acc_idx])
        acc_idx += 1

        mask_filename = None
        if (len(b) == acc_idx + 1):
            # has mask
            mask_filename = b[acc_idx]

        mask = None

        if (mask_filename):
            path = os.path.join(self.base_path, mask_filename)
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            # mask = mask.astype(np.float32) * 1.0/255.0

        img = cv2.imread(os.path.join(self.base_path, filename),
                         cv2.IMREAD_GRAYSCALE)
        # img = img.astype(np.float32) * 1.0/255.0

        # note (moses, august 24) I did [:21] here because right now I'm just doing 21 hand keypoints, but this was written at a time when I really wanted a forearm keypoint.
        # All the acc_idx stuff makes it way easier to just hack it like this.
        # for i in range(21):
        #     cv2.circle(img, (int(kps[i, 0]), int(kps[i, 1])), 2, (255, 255, 0))

        noisy_keypoints = add_2d_noise_to_keypoints(kps[:21, :2])

        trans = crop(img, noisy_keypoints, is_right)

        keypoints_moved = rotate_hand(kps, trans)
        keypoints_noisy_moved = rotate_hand(noisy_keypoints, trans)

        img = cv2.warpAffine(img, trans, (128, 128))
        if mask is not None:
            mask = cv2.warpAffine(mask, trans, (128, 128))

        # You can't stop meeeee
        if __name__ == "__main__":
            img_old = cv2.imread(os.path.join(self.base_path, filename))

            bleg = add_2d_noise_to_keypoints(kps[:21, :2])
            geo.draw_hand_rainbow_pts(img_old, bleg)
            geo.draw_21_hand_lines(img_old, bleg, (255, 0, 0))

            geo.draw_hand_rainbow_pts(img_old, kps[:21, :2])

            cv2.imshow("h", img)
            cv2.imshow("y", img_old)
            cv2.waitKey(0)

        if (random.uniform(0, 1) < 0.3):
            # 30% chance of no predicted input
            keypoints_noisy_moved = None

        ret = self.augmaker.do_one_augmentation(
            img,
            keypoints_moved,
            predicted_px=keypoints_noisy_moved,
            mask=mask,
            img_alpha_premultiplied=False,
            is_right=is_right)
        ret["elbow"] = torch.zeros(3).float()
        ret["curls"] = torch.zeros(5).float()

        return ret


if __name__ == '__main__':

    a = RandoDataset(f"{datasets_basepath}", "nikitha.csv")
    b = RandoDataset(f"{datasets_basepath}", "tom.csv")

    d = torch.utils.data.ConcatDataset([a, b])

    gts_valid = 0
    gts_invalid = 0

    preds_valid = 0
    preds_invalid = 0

    valid = 0
    invalid = 0

    e = list(range(len(d)))
    random.shuffle(e)

    for i in e:
        r = d[i]
        print("l44 ->", i)
