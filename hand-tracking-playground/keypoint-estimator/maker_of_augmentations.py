import math
import cv2
import numpy as np
from opencv_transforms import transforms
import random
import a_geometry as geo
from a_aug_config import aug_config
import torch
import colour

import smallestenclosingcircle
import heatmap_1d

from settings import datasets_basepath


# for if you want to center on middle-prox:
'''
center = in_keypoints[9].copy()  # middle-proximal
        radius_surrounding_joints = 0
        for kp in in_keypoints:
            radius_surrounding_joints = max(radius_surrounding_joints, np.linalg.norm(center-kp))
            if do_debug:
                cv2.circle(debug_scribble_mat, geo.interize(kp), 3, (255, 0, 0))
        if do_debug:
            cv2.circle(debug_scribble_mat, geo.interize(center), int(radius_surrounding_joints), debug_scribble_color)
'''


amtWeCare = [1.0]
amtWeCare += [0.7, 0.7, 0.7, 1.0]
for i in range(4):
  amtWeCare += [0.7, 0.5, 0.5, 1.0]

amtWeCare += [1.0]  # Forearm

print(len(amtWeCare))


def scribble(img_input, gt, valids_gt, prediction=None) -> np.ndarray:
  colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1)]
  new_img_input = cv2.cvtColor(img_input, cv2.COLOR_GRAY2BGR)

  if prediction is None:
    arr = [(gt, valids_gt)]
    colors = [(0, 1, 0)]
  else:
    arr = ((prediction, np.concatenate(
        (np.ones((21, 3)), np.zeros((1, 3))))), (gt, valids_gt))

  for idx, je in enumerate(arr):
    joint_coord = je[0]
    valid = je[1]

    for coord, valid_this in zip(joint_coord, valid):
      if not valid_this.any():
        continue
      cv2.circle(new_img_input, (int(coord[0]), int(coord[1])), 4, colors[idx])

    for finger in range(5):
      last_idx = 0
      for joint in range(4):
        coord = joint_coord[last_idx]
        prev = (int(coord[0]), int(coord[1]))

        curr_idx = 1 + finger*4 + joint

        coord = joint_coord[curr_idx]
        next = (int(coord[0]), int(coord[1]))

        if (valid[curr_idx].any() and valid[last_idx].any()):
          cv2.line(new_img_input, prev, next, colors[idx], 1)
        last_idx = curr_idx

    if valid[0].any() and valid[21].any():
      coord = joint_coord[0]
      prev = (int(coord[0]), int(coord[1]))
      coord = joint_coord[21]
      next = (int(coord[0]), int(coord[1]))
      cv2.line(new_img_input, prev, next, colors[idx], 1)
  return new_img_input


def maybe_fix_elbow(kps, gt_xy_valid):
  # kps

  if not (gt_xy_valid[0] and gt_xy_valid[21]):
    return kps
  from_ = kps[0]
  to_ = kps[21]
  dir_ = to_ - from_

  dir_ *= 30/np.linalg.norm(dir_[:2])

  dir_[0] *= (1/30)
  dir_[1] *= (1/30)

  kps[21] = dir_

  return kps


def make_heatmap_and_directreg_forearm_output(input_as_tensor: np.ndarray,
                                              pose_predicted_input_keypoints: torch.Tensor,
                                              pred_valid: float,
                                              gt_px: np.ndarray,
                                              gt_xy_valid: np.ndarray,
                                              gt_depth_valid: np.ndarray):

  # Should be 128x128
  assert input_as_tensor.shape[1] == input_as_tensor.shape[2]

  img_xy_size = 128  # input_as_tensor.shape[1]

  heatmap_side_px = 22

  divisor_xy = heatmap_side_px/img_xy_size

  num_output_heatmap_joints = 21

  target_xy_heatmaps = np.zeros(
      (num_output_heatmap_joints, heatmap_side_px, heatmap_side_px))
  target_depth_heatmaps = np.zeros(
      (num_output_heatmap_joints, heatmap_side_px))

  xy_heatmap_attentions = np.zeros(
      (num_output_heatmap_joints, heatmap_side_px, heatmap_side_px))
  depth_heatmap_attentions = np.zeros(
      (num_output_heatmap_joints, heatmap_side_px))

  for i in range(num_output_heatmap_joints):
    if gt_xy_valid[i]:
      stddev_px = 1
      hmap_x = heatmap_1d.heatmap_1d(22, gt_px[i][0]*divisor_xy, stddev_px)
      hmap_y = heatmap_1d.heatmap_1d(22, gt_px[i][1]*divisor_xy, stddev_px)

      target_xy_heatmaps[i] = heatmap_1d.two_heatmaps_to_2d(hmap_x, hmap_y)
      xy_heatmap_attentions[i] = amtWeCare[i]
    if gt_depth_valid[i]:
      target_depth_heatmaps[i] = heatmap_1d.heatmap_1d(
          22, gt_px[i][2]*divisor_xy, stddev_px)
      depth_heatmap_attentions[i] = amtWeCare[i]

  use_forearm = 0.0
  forearmLocation = np.zeros((3))
  if gt_xy_valid[21]:  # and gt_depth_valid[21]: # Change back later
    use_forearm = 1.0
    forearmLocation[0] = gt_px[21][0]
    forearmLocation[1] = gt_px[21][1]
    forearmLocation[2] = gt_px[21][2]

  sample = {
      'input_image': torch.from_numpy(input_as_tensor).float(),
      'input_predicted_keypoints': pose_predicted_input_keypoints.float(),
      'input_predicted_keypoints_valid': float(pred_valid),
      'gt_xy': torch.from_numpy(target_xy_heatmaps).float(),
      'gt_depth': torch.from_numpy(target_depth_heatmaps).float(),
      'gt_xy_valid': torch.from_numpy(gt_xy_valid),
      'gt_depth_valid': torch.from_numpy(gt_depth_valid),
      'attention_xy': xy_heatmap_attentions,
      'attention_depth': depth_heatmap_attentions,
      "gt_joint_locs": torch.from_numpy(gt_px).float(),
      'use_forearm': use_forearm,
      'forearm_direction': forearmLocation,
  }

  return sample

  # return value: input image, input predicted keypoints, is the predicted keypoints valid?
  # ground truth heatmaps (22, 22x22, I guess? that'll be confusing)
  # which of the ground truth heatmaps are valid (22,)


class AugmentationMaker:
  def __init__(self, inp_aug_config: aug_config):
    self.aug_config = inp_aug_config
    self.output_size: int = self.aug_config.output_size

    self.columns = ["hand_filename",
                    "WRIST_X", "WRIST_Y",
                    "THMB_MCP_X", "THMB_MCP_Y",
                    "THMB_PXM_X", "THMB_PXM_Y",
                    "THMB_DST_X", "THMB_DST_Y",
                    "THMB_TIP_X", "THMB_TIP_Y",

                    "INDX_PXM_X", "INDX_PXM_Y",
                    "INDX_INT_X", "INDX_INT_Y",
                    "INDX_DST_X", "INDX_DST_Y",
                    "INDX_TIP_X", "INDX_TIP_Y",

                    "MIDL_PXM_X", "MIDL_PXM_Y",
                    "MIDL_INT_X", "MIDL_INT_Y",
                    "MIDL_DST_X", "MIDL_DST_Y",
                    "MIDL_TIP_X", "MIDL_TIP_Y",

                    "RING_PXM_X", "RING_PXM_Y",
                    "RING_INT_X", "RING_INT_Y",
                    "RING_DST_X", "RING_DST_Y",
                    "RING_TIP_X", "RING_TIP_Y",

                    "LITL_PXM_X", "LITL_PXM_Y",
                    "LITL_INT_X", "LITL_INT_Y",
                    "LITL_DST_X", "LITL_DST_Y",
                    "LITL_TIP_X", "LITL_TIP_Y"]

    self.outputIdx: int = 0

    self.normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])  # This is what ImageNet uses?

    # Use this in three places - one, in the hand, two, in the background, three when the background and hand are combined
    self.rpsh_smaller = transforms.ColorJitter(.2, .2, .05, .05)
    self.rpsh_bigger = transforms.ColorJitter(.5, .5, .10, .10)

    self.rpsh_no_mask = transforms.ColorJitter(.6, .6, .125, .125)

    self.rrot = transforms.RandomRotation([0, 360])
    self.hflip = transforms.RandomHorizontalFlip()
    self.rcrop1 = transforms.RandomResizedCrop(
        size=self.output_size*2, scale=(math.sqrt(0.01), math.sqrt(.5)))
    self.ccrop = transforms.CenterCrop(size=(self.output_size, self.output_size)
                                       )

    with open(f"{datasets_basepath}/indoor/BothImages.txt") as f:
      self.backgrounds_list = [(f"{datasets_basepath}/indoor/indoorCVPR_09/Images/"+ele).replace("\n", "")
                               for ele in f.readlines()]

  def warp_matrix_no_prediction(self, keypoints, gt_xy_valid, is_right):

    kps_for_circle = []
    for i in range(21):
      # Intentionally not including forearm.
      if gt_xy_valid[i]:
        kps_for_circle.append([keypoints[i][0], keypoints[i][1]])

    cx, cy, radius_surrounding_joints = smallestenclosingcircle.make_circle(
        kps_for_circle)

    center = np.float32([cx, cy])

    final_r = radius_surrounding_joints * \
        geo.arbitrary_density(self.aug_config.radius_function_no_prediction)

    min_movement = 0
    max_movement = final_r - (radius_surrounding_joints*1.01)

    move_amount = random.uniform(min_movement, max_movement)
    move_angle = random.uniform(0, math.pi*2)

    move_direction = np.float32((math.sin(move_angle), math.cos(move_angle))
                                )*move_amount

    center += move_direction

    rot = random.uniform(-math.pi, math.pi)

    x_axis = np.float32([math.cos(rot), math.sin(rot)])*final_r
    if is_right:
      x_axis *= -1
    y_axis = np.float32([-math.sin(rot), math.cos(rot)])*final_r

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

  def warp_matrix_prediction(self, predicted_px, is_right):

    rot = random.uniform(-math.pi, math.pi)

    x_axis = np.float32([math.cos(rot), math.sin(rot)])
    y_axis = np.float32([-math.sin(rot), math.cos(rot)])

    # Just rotation
    tl = (0, 0)
    tr = (1, 0)
    bl = (0, 1)

    tl_o = (0, 0)
    tr_o = x_axis
    bl_o = y_axis

    src_tri = np.float32((tl, tr, bl))
    dst_tri = np.float32((tl_o, tr_o, bl_o))

    trans_rot = cv2.getAffineTransform(src_tri, dst_tri)

    in_rotated = []
    # vectorize!
    for i in range(21):
      in_rotated.append(geo.transformVecBy2x3(predicted_px[i], trans_rot))
    in_rotated = np.array(in_rotated)

    x, y, r = geo.minicircle(in_rotated)

    # one at the end doesn't matter.
    radius_function = [(1.2, 1.0), (1.4, 1.0), (1.6, 0.8),
                       (1.8, 0.6), (2.0, 0.5), (2.2, 0.0)]
    radius_function = [(1.2, 1.0), (1.4, 0.9), (1.6, 0.8),
                       (1.8, 0.5), (2.0, 0.2), (2.4, 0.0)]

    r_mul = geo.arbitrary_density(radius_function)
    r_mul = 1.5

    final_r = r * r_mul  # NO.
    sz = self.aug_config.output_size
    center = np.array((x, y))

    center += np.random.normal(0, 0.03*r, (2))

    # Translation and scale, on top of the rotation we've already done.
    rot = 0
    x_axis = np.float32([math.cos(rot), math.sin(rot)])*final_r
    if is_right:
      x_axis *= -1

    y_axis = np.float32([-math.sin(rot), math.cos(rot)])*final_r

    tl = center - x_axis - y_axis
    tr = center + x_axis - y_axis
    bl = center - x_axis + y_axis

    tl_o = (0, 0)
    tr_o = (sz, 0)
    bl_o = (0, sz)

    src_tri = np.float32((tl, tr, bl))
    dst_tri = np.float32((tl_o, tr_o, bl_o))

    trans_movescale = cv2.getAffineTransform(src_tri, dst_tri)

    trans_rot_3x3 = np.eye(3)
    trans_movescale_3x3 = np.eye(3)

    trans_rot_3x3[0:2, 0:3] = trans_rot
    trans_movescale_3x3[0:2, 0:3] = trans_movescale

    trans = np.dot(trans_movescale_3x3, trans_rot_3x3)
    trans = trans[0:2]
    return trans

  def do_one_augmentation(
          self, in_mat, gt_px, gt_xy_valid, gt_depth_valid, predicted_px=None, mask=None, is_right=False,
          extra_subdir=None, do_debug=False, debug_scribble_mat=None, debug_scribble_color=None, debug_winname=None):
    # Move fast and break things

    prediction_found = predicted_px is not None

    if prediction_found:
      trans = self.warp_matrix_prediction(predicted_px, is_right)
    else:
      trans = self.warp_matrix_no_prediction(gt_px, gt_xy_valid, is_right)
    sz = self.aug_config.output_size

    transformedHand = cv2.warpAffine(in_mat, trans, (sz, sz))

    if mask is not None:
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

      transformedMask = cv2.warpAffine(mask, trans, (sz, sz))
      transformedHand = (transformedHand * (1-transformedMask)
                         ) + (background_float * transformedMask)

    # cover part of the image in black, either a hard axial line or a circular gradient
    transformedHand = geo.mat_uint8tofloat32(transformedHand)
    transformedHand = colour.models.eotf_inverse_sRGB(
        transformedHand).astype(np.float32)
    edge_mode_opts = {"axial": 0.05,
                      "radial": 0.05}
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

      transformedHand = cv2.multiply(transformedHand, edge_mask)
      transformedHand = colour.models.eotf_sRGB(
          transformedHand).astype(np.float32)

    transformedHand = geo.normalizeGrayscaleImage(
        transformedHand)

    if not prediction_found:
      predicted_px = None

    for i in range(22):
      if prediction_found:
        predicted_px[i] = geo.transformVecBy2x3(predicted_px[i], trans)
      gt_px[i] = geo.transformVecBy2x3(gt_px[i], trans)

    gt_px = maybe_fix_elbow(gt_px, gt_xy_valid)

    transformedHand = transformedHand[np.newaxis, ...]

    pose_predicted_input_keypoints = torch.zeros(21, 3, dtype=torch.float32)
    if prediction_found:
      pose_predicted_input_keypoints = torch.as_tensor(
          predicted_px[:21], dtype=torch.float32)

    pred_valid = 1.0 if prediction_found else 0.0

    # print(inp_image.storage_type(), pred_kp.storage_type(), pred_valid.storage_type(), gt_px.storage_type(), valids_gt.storage_type())

    return make_heatmap_and_directreg_forearm_output(
        transformedHand, pose_predicted_input_keypoints, pred_valid, gt_px, gt_xy_valid, gt_depth_valid)

  def close(self):
    self.csvfile.close()
