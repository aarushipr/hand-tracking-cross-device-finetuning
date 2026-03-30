import math
import cv2
import numpy as np
import scipy
from opencv_transforms import transforms
import random
import a_geometry as geo
from a_aug_config import aug_config
import torch
import colour

import smallestenclosingcircle
import heatmap_1d

import settings
from dataclasses import dataclass

# For later :)
# import imgaug

# sap = imgaug.augmenters.SaltAndPepper()


@dataclass
class center_radius:
    cx: float
    cy: float
    radius_around_predicted: float
    radius_around_predicted_and_gt: float


amtWeCare = [1.0]
amtWeCare += [0.7, 0.7, 0.7, 1.0]
for i in range(4):
    amtWeCare += [0.7, 0.5, 0.5, 1.0]

amtWeCare += [1.0]  # Forearm


'''
img: np.ndarray normalized to [0,1]
'''


def maybe_add_image_edge_simulation(img: np.ndarray) -> np.ndarray:

    # cover part of the image in black, either a hard axial line or a circular gradient

    sz = int((img.shape[0] + img.shape[1])/2)

    edge_mode_opts = {"axial": 0.05,
                      "radial": 0.05}
    # None is a valid key to use in a dict.
    edge_mode_opts[None] = 1.0 - sum(edge_mode_opts.values())
    edge_mode = np.random.choice(
        list(edge_mode_opts.keys()),
        p=list(edge_mode_opts.values()))
    if edge_mode is not None:
        if edge_mode == "radial":
            # radial vignette mask
            border_angle = np.random.uniform(-math.pi, math.pi)
            coverage = np.random.uniform(-1.0, 0.5)
            radius = np.random.uniform(1.0, 5.0) ** 2 * sz
            softness = np.random.uniform(0.0, 0.05) * radius
            center_distance = radius + (coverage * sz)
            center_x = center_distance * np.sin(border_angle) + sz/2.0
            center_y = center_distance * np.cos(border_angle) + sz/2.0

            edge_mask = geo.make_radial_gradient(
                sz, center_x, center_y, radius,
                softness=softness)
        elif edge_mode == "axial":
            # image-border-like mask
            sample_location = np.random.rand(2) * 2 - 1
            sample_location *= 0.75  # up to 75% covered per axis
            # zero one axis to simulate non-corner edges
            zero_axis_opts = {0: 0.4, 1: 0.4}
            zero_axis_opts[None] = 1.0 - sum(zero_axis_opts.values())
            zero_axis = np.random.choice(
                list(zero_axis_opts.keys()),
                p=list(zero_axis_opts.values()))
            if zero_axis is not None:
                sample_location[zero_axis] = 0

            edge_mask = np.zeros((sz, sz), np.float32)
            sample_location_int = (sample_location * sz).astype(np.int32)
            x, y = sample_location_int
            cv2.rectangle(edge_mask, (x, y), (x+sz, y+sz),
                          color=1.0, thickness=-1)
        else:
            raise ValueError(f"unexpected edge mode {edge_mode}")

        img = cv2.multiply(img, edge_mask)
    return img


def maybe_add_cutout(img):
    # 10% chance of doing it
    if (random.random() > 0.1):
        return img
    h = img.shape[0]
    w = img.shape[1]

    mask = np.ones((h, w), np.float32)

    n_holes = int(random.uniform(1, 7))

    for n in range(n_holes):
        center_y = np.random.randint(h)
        center_x = np.random.randint(w)

        length_x = int(1 + (random.uniform(0, 1)**3)*100)
        length_y = int(1 + (random.uniform(0, 1)**3)*100)

        y1 = np.clip(center_y - length_y // 2, 0, h)
        y2 = np.clip(center_y + length_y // 2, 0, h)
        x1 = np.clip(center_x - length_x // 2, 0, w)
        x2 = np.clip(center_x + length_x // 2, 0, w)

        shape = mask[y1: y2, x1: x2].shape

        luminance_center = random.uniform(0, 1)

        luminance_radius = random.uniform(0, 0.5)

        e = np.clip(np.random.uniform(luminance_center - luminance_radius,
                    luminance_center + luminance_radius, shape), 0, 1)

        img[y1: y2, x1: x2] = e  # np.clip(random.uniform(0,1) +

    # mask = torch.from_numpy(mask)
    # mask = mask.expand_as(img)
    # img = img * mask

    return img


def maybe_add_blur(img):
    if (random.random() > 0.1):
        return img
    b = scipy.ndimage.filters.gaussian_filter(img, sigma=random.uniform(0, 2))
    return b


def maybe_add_noise(img):
    if (random.random() > 0.1):
        return img
    img += np.random.normal(0, np.random.uniform(0.0, 0.02), img.shape)
    return img


class AugmentationMaker:
    def __init__(self, inp_aug_config: aug_config):
        self.aug_config = inp_aug_config
        self.output_size: int = self.aug_config.output_size

        self.outputIdx: int = 0

        # These are probably used for background?
        self.rrot = transforms.RandomRotation([0, 360])
        self.hflip = transforms.RandomHorizontalFlip()
        self.rcrop1 = transforms.RandomResizedCrop(
            size=self.output_size*2, scale=(math.sqrt(0.01), math.sqrt(.5)))
        self.ccrop = transforms.CenterCrop(size=(self.output_size, self.output_size)
                                           )

        with open(f"/3/epics/gk3/indoor/BothImages.txt") as f:
            self.backgrounds_list = [(f"/3/epics/gk3/indoor/indoorCVPR_09/Images/"+ele).replace("\n", "")
                                     for ele in f.readlines()]

    def make_heatmap_output(self,
                            input_as_tensor: np.ndarray,
                            pose_predicted_input_keypoints: torch.Tensor,
                            pred_valid: float,
                            gt_px: np.ndarray,
                            has_depth: bool,
                            make_not_hand_output: bool = False):

        if not self.aug_config.validation_dataset:

            if not settings.using_pose_predicted_input:
                # Generate random input data
                # if random.random() < 0.25:
                pred_valid = 0
                # else:
                #     pred_valid = 1
                pose_predicted_input_keypoints = np.random.normal(
                    0, 0.4, size=(21, 3))

        # Should be 128x128
        assert input_as_tensor.shape[1] == input_as_tensor.shape[2]

        img_xy_size = 128  # input_as_tensor.shape[1]

        heatmap_side_px = 22

        divisor_xy = heatmap_side_px/img_xy_size

        num_output_heatmap_joints = 21

        target_xy_heatmaps = np.zeros(
            (num_output_heatmap_joints, heatmap_side_px, heatmap_side_px))

        target_depth_heatmaps = np.zeros(
            (num_output_heatmap_joints, heatmap_side_px), dtype=np.float32)

        for i in range(num_output_heatmap_joints):
            stddev_px = 1
            hmap_x = heatmap_1d.heatmap_1d(
                22, gt_px[i][0]*divisor_xy, stddev_px)
            hmap_y = heatmap_1d.heatmap_1d(
                22, gt_px[i][1]*divisor_xy, stddev_px)

            target_xy_heatmaps[i] = heatmap_1d.two_heatmaps_to_2d(
                hmap_x, hmap_y)

            if (has_depth):
                depth_value = ((gt_px[i][2] / 1.5 / 2) + 0.5)*22
                if ((depth_value < 0) or (depth_value >= 22)):
                    print(f"depth value bad! {gt_px[i][2]} -> {depth_value}")
                    # print(depth_value)

                target_depth_heatmaps[i] = heatmap_1d.heatmap_1d(
                    22, depth_value, stddev_px)
        is_hand = not make_not_hand_output
        has_2d = is_hand
        has_depth = has_depth and is_hand
        # else: the hands are just zeros

        # print(pose_predicted_input_keypoints.shape)

        # if not has_depth:

        #!@todo Why isn't this a dataclass?
        # Because PyTorch's default collate function dies if you try to use dataclasses.
        sample = {
            'input_image': torch.from_numpy(input_as_tensor).float(),
            'input_predicted_keypoints': torch.from_numpy(pose_predicted_input_keypoints).float(),
            'input_predicted_keypoints_valid': np.float32(pred_valid),
            'gt_xy': torch.from_numpy(target_xy_heatmaps).float(),
            'has_xy': np.float32(has_2d),
            # note: in /3/epics/grayscale-keypoint-2 is where this came from, and there's more useful stuff there.
            'gt_depth': torch.from_numpy(target_depth_heatmaps).float(),
            'has_depth': np.float32(has_depth),
            "gt_joint_locs": torch.from_numpy(gt_px).float(),
            'is_hand': np.float32(is_hand),
        }

        return sample

    # sigh, I did remove the code for smallestenclosingcircle again, sorry.

    def center_and_radius_prediction(self, gt_keypoints, prediction):
        center = prediction[9].copy()

        radius_around_predicted = 0
        radius_around_gt_and_predicted = 0

        for keypoint in prediction:
            radius_around_predicted = max(
                radius_around_predicted, np.linalg.norm(keypoint-center))

        radius_around_gt_and_predicted = radius_around_predicted
        for keypoint in gt_keypoints:
            radius_around_gt_and_predicted = max(
                radius_around_gt_and_predicted, np.linalg.norm(keypoint-center))

        return center[0], center[1], radius_around_predicted, radius_around_gt_and_predicted

    def center_and_radius_no_prediction(self, keypoints):
        # middle-proximal joint is the 9th joint.
        # uh oh
        center = keypoints[9].copy()

        r = 0

        for keypoint in keypoints:
            r = max(r, np.linalg.norm(keypoint-center))

        return center[0], center[1], r, r

    def warp_matrix_validation(self, keypoints, is_right, pose_predicted_keypoints=None):

        if pose_predicted_keypoints is None:
            our_radius_move = self.aug_config.rm_no_prediction
            cx, cy, r_calculate_max, r_calculate_min = self.center_and_radius_no_prediction(
                keypoints)
        else:
            our_radius_move = self.aug_config.rm_prediction
            cx, cy, r_calculate_max, r_calculate_min = self.center_and_radius_prediction(
                keypoints, pose_predicted_keypoints)

        center = np.float32([cx, cy])

        # Use our distribution code to pick a radius relative either ground truth or predicted keypoints
        mul = 1.5
        final_r = r_calculate_max * mul

        # If we had a prediction and ground truth, the final_r might crop out some of the ground truth keypoints. If so, make it bigger.
        final_r = max(r_calculate_min, final_r)

        x_axis = np.float32([1, 0])*final_r
        if is_right:
            x_axis *= -1
        y_axis = np.float32([0, 1])*final_r

        tl = center - x_axis - y_axis
        tr = center + x_axis - y_axis
        bl = center - x_axis + y_axis

        tl_o = (0, 0)
        tr_o = (self.aug_config.output_size, 0)
        bl_o = (0, self.aug_config.output_size)

        src_tri = np.float32((tl, tr, bl))
        dst_tri = np.float32((tl_o, tr_o, bl_o))

        trans = cv2.getAffineTransform(src_tri, dst_tri)

        return trans

    def get_rotate_scale_background(self):
        background_j = None
        while background_j is None:
            b_name = random.choice(self.backgrounds_list)
            background_j = cv2.imread(b_name)
        background_j = self.rcrop1(background_j)  # Crop and resize to sz*2
        background_j = self.rcrop1(background_j)  # Do it again
        background_j = self.rrot(background_j)  # Randomly rotate
        # Crop at the center to sz - ensures no black edges - also, randomly flip
        background_j = self.hflip(self.ccrop(background_j))

        background_j = cv2.cvtColor(background_j, cv2.COLOR_BGR2GRAY)

        background_float = geo.mat_uint8tofloat32(background_j)

        # Convert the background to linear as well. I don't _quite_ know if they come as sRGB, but it's a good guess and doesn't matter too much.
        background_float = colour.models.eotf_sRGB(
            background_float).astype(np.float32)
        return background_float

    def maybe_mixup(self, img):
        if (random.random() > 0.1):
            return img
        extra_background = self.get_rotate_scale_background()

        mix_amount = random.uniform(0, 0.5)

        new = (img * 1-mix_amount) + (extra_background * mix_amount)
        return new

    '''
        in_mat should be a sRGB image normalized to 0,255.

        gt_px should be (21,2) px coords
        predicted_px should be the same

        mask should be a *LINEAR* image normalized to 0,255. https://www.khronos.org/opengl/wiki/Image_Format#sRGB_colorspace
        If it's not linear, you have made a mistake.

        img_alpha_premultiplied should be True if it's something from ie. Blender where the background is already black.
        If it's that data you collected in front of a greenscreen (or something like that), do False.
        Refer to https://limnu.com/premultiplied-alpha-primer-artists

        In this function, we convert the hand from sRGB 0,255 -> sRGB 0,1 -> linear 0,1, composite if necessary, then send it on to the NN.

        Uhh, okay, so I couldn't find a way to convret from sRGB to linear that felt right. I guess I'm just not going to do any conversion
        and hope it's okay. Ughhhh

        @todo, do we want the NN to ever see sRGB images? I feel like "no, because we do mean,std normalization anyway and the data coming from hardware image sensors should really not be sRGB"
    '''

    def do_one_augmentation(self,
                            in_mat,
                            gt_px,
                            predicted_px=None,
                            mask=None,
                            img_alpha_premultiplied=True,
                            is_right=False):
        # Move fast and break things

        not_hand: bool = (random.random() <
                          self.aug_config.not_hand_likelihood)
        if (not_hand):
            print("not hand!")
        prediction_found = predicted_px is not None
        has_depth = gt_px.shape == (21, 3)
        not_has_depth = gt_px.shape == (21, 2)

        assert (has_depth != not_has_depth)

        transformedHand = in_mat

        transformedHand = geo.mat_uint8tofloat32(transformedHand)
        transformedHand = colour.models.eotf_sRGB(
            transformedHand).astype(np.float32)

        # XXX: This won't work if you use greenscreen data as a validation dataset. So don't.
        if mask is not None:

            background_float = self.get_rotate_scale_background()

            transformedMask = geo.mat_uint8tofloat32(mask)

            if img_alpha_premultiplied:
                transformedHand = transformedHand + \
                    (background_float * (transformedMask))
            else:
                transformedHand = (transformedHand * (1-transformedMask)
                                   ) + (background_float * (transformedMask))


        #!@todo Very hacky and inefficient: just replace the transformedHand with a background image.
        if not_hand:
            transformedHand = self.get_rotate_scale_background()

        if not self.aug_config.validation_dataset:
            transformedHand = self.maybe_mixup(transformedHand)
            transformedHand = maybe_add_image_edge_simulation(transformedHand)
            transformedHand = maybe_add_cutout(transformedHand)
            transformedHand = maybe_add_noise(transformedHand)
            transformedHand = maybe_add_blur(transformedHand)
        transformedHand = geo.normalizeGrayscaleImage(
            transformedHand)

        # has_depth: bool = True

        if not prediction_found:
            predicted_px = np.zeros((21, 3))
        if not has_depth:
            aanew = np.zeros((21, 3))
            aanew[:, :2] = gt_px
            gt_px = aanew
        # if not has_depth:

        # for i in range(21):
        #     if prediction_found:
        #         predicted_px[i] = geo.transformVecBy2x3(predicted_px[i], trans)
        #         for j in range(2):
        #             predicted_px[i, j] = geo.map_ranges(
        #                 predicted_px[i, j], 0, 128, -1, 1)
        #     gt_px[i] = geo.transformVecBy2x3(gt_px[i], trans)

        transformedHand = transformedHand[np.newaxis, ...]

        # pose_predicted_input_keypoints = torch.zeros(
        #     21, 2, dtype=torch.float32)
        # if prediction_found:
        #     pose_predicted_input_keypoints = torch.as_tensor(
        #         predicted_px[:21], dtype=torch.float32)

        pred_valid = 1.0 if prediction_found else 0.0

        return self.make_heatmap_output(
            transformedHand, predicted_px, pred_valid, gt_px, has_depth, not_hand)

    def close(self):
        self.csvfile.close()
