import pandas as pd
import ujson as json
import numpy as np
import torch
import cv2
import os.path as osp
import os

from a_aug_config import the_aug_config

from maker_of_augmentations import AugmentationMaker
from settings import datasets_basepath




def cam2pixel_depth(cam_coord__, f, c):
  cam_coord = cam_coord__.copy()

  hand_length_mm = np.linalg.norm(cam_coord[0]-cam_coord[9])

  wrist_distance_mm = np.linalg.norm(cam_coord[0])

  x_px = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
  y_px = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]

  dist_compared_to_wrist = cam_coord[:, 2]

  for i in range(21):
    dist = np.linalg.norm(cam_coord[i])
    dist_compared_to_wrist[i] = (dist-wrist_distance_mm) / hand_length_mm

  img_coord = np.concatenate(
      (x_px[:, None], y_px[:, None], dist_compared_to_wrist[:, None]), 1)
  return img_coord


def world2cam(world_coord, R, T):
  cam_coord = np.dot(R, world_coord - T)
  return cam_coord


np.set_printoptions(precision=3, suppress=True)

root_path = f"{datasets_basepath}/InterHand2.6M_30fps_batch1/"

img_path = root_path+'images'
annot_path = root_path+"annotations"


def wobble_predictions_3d(predictions_3d):
  # distance from wrist to middle-pxm
  hand_size = np.linalg.norm(predictions_3d[0]-predictions_3d[9])

  move_all_fac = 0.3
  move_individual_fac = 0.02
  # (21, 3)
  move_all = np.ones((21, 3)) * np.random.normal(0, hand_size*move_all_fac, 3)

  move_individual = np.random.normal(0, hand_size*move_individual_fac, (21, 3))

  ret = predictions_3d + move_all + move_individual

  return ret


def to_single_hand_order(_joint_coord, hand_idx):
  new_joint_coord = np.empty((21, 3))
  if hand_idx == 1:
    joint_coord = _joint_coord[0:21]
  else:
    joint_coord = _joint_coord[21:]
  # Wrist.
  new_joint_coord[0] = joint_coord[20]
  for finger in range(5):
    offs = finger*4
    new_joint_coord[1+offs][:3] = joint_coord[3+offs]
    new_joint_coord[2+offs][:3] = joint_coord[2+offs]
    new_joint_coord[3+offs][:3] = joint_coord[1+offs]
    new_joint_coord[4+offs][:3] = joint_coord[0+offs]
  return new_joint_coord


def valids_to_single_hand_order(the_input, hand_idx):
  output = np.zeros((21), dtype=np.float32)

  if hand_idx == 1:
    the_input = the_input[0:21]
  else:
    the_input = the_input[21:]
  # Wrist.
  output[0] = float(the_input[20])
  for finger in range(5):
    offs = finger*4
    output[1+offs] = float(the_input[3+offs])
    output[2+offs] = float(the_input[2+offs])
    output[3+offs] = float(the_input[1+offs])
    output[4+offs] = float(the_input[0+offs])
  return output


class InterSequentialDataset(torch.utils.data.Dataset):
  def __init__(self, file):
    self.augmaker = AugmentationMaker(the_aug_config)
    self.joints = {"train": [None]*26, "test": [None]*26, "val": [None]*26}
    self.cams = {}
    for mode in "train", "test", "val":
      with open(osp.join(annot_path, mode, 'InterHand2.6M_' + mode + '_camera.json')) as f:
        self.cams[mode] = json.load(f)
    self.cam_annotation_hand_file = pd.read_csv(
        file, delimiter=" ", quotechar="|", header=None)

    self.frame_indices_file = pd.read_csv(
        file+"_frame_indices", delimiter=" ", quotechar="|", header=None)

  def fingertips_to_sphere_centers(self, joints):
    lerp_amount = 0.3
    for finger in range(5):
      offs_tip = 1 + finger*4 + 3
      offs_dst = 1 + finger*4 + 2

      tip = joints[offs_tip]
      dst = joints[offs_dst]

      joints[offs_tip] = (tip * (1.0-lerp_amount)) + (dst * lerp_amount)

  def predict_last_two(self, joint_n_2, joint_n_1):
    dir = joint_n_1 - joint_n_2

    out = joint_n_1 + dir

    return out

  def to_cam_coord(self, cameras, capture_id, cam, joints, hand_idx):
    campos, camrot = np.array(
        cameras[str(capture_id)]['campos'][str(cam)],
        dtype=np.float32), np.array(
        cameras[str(capture_id)]['camrot'][str(cam)],
        dtype=np.float32)
    joint_world = np.array(joints['world_coord'], dtype=np.float32)
    valids = np.array(joints['joint_valid'])

    valids = valids_to_single_hand_order(valids, hand_idx)

    joint_world = to_single_hand_order(joint_world, hand_idx)
    self.fingertips_to_sphere_centers(joint_world)
    joint_cam = world2cam(
        joint_world.transpose(1, 0),
        camrot,
        campos.reshape(3, 1),
    ).transpose(1, 0)
    return joint_cam, valids

  def to_px_coord(self, cameras, capture_id, cam, joint_cam, hand_idx):
    focal, princpt = np.array(
        cameras[str(capture_id)]['focal'][str(cam)],
        dtype=np.float32), np.array(
        cameras[str(capture_id)]['princpt'][str(cam)],
        dtype=np.float32)
    joint_img = cam2pixel_depth(joint_cam, focal, princpt)

    return joint_img

  def pd_frame_idx_to_frame_idx(self, pd_frame_idx):
    return int(self.cam_annotation_hand_file.iloc[pd_frame_idx, 5])

  def simple_try_find_prediction(self, mode, capture_idx, start_looking_at_idx, cam_id, hand_idx, really_sequential_frame_idx):
    tries = []
    tries.append(self.to_cam_coord(self.cams[mode], capture_idx, cam_id,
                                   self.joints[mode][capture_idx][str(self.frame_indices_file.iloc[really_sequential_frame_idx-1, 0])], hand_idx))
    tries.append(self.to_cam_coord(self.cams[mode], capture_idx, cam_id,
                                   self.joints[mode][capture_idx][str(self.frame_indices_file.iloc[really_sequential_frame_idx-2, 0])], hand_idx))
    if not (np.all(tries[-1][1]) and np.all(tries[-2][1])):
      return False, None

    # direction from the old to the new.
    # new - old.
    old = tries[-1][0]
    new = tries[-2][0]
    dir = new - old
    out = new + (dir)

    return True, out

  def scribble(self, img_input, prediction, gt) -> np.ndarray:
    colors = (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1)
    new_img_input = cv2.cvtColor(img_input, cv2.COLOR_GRAY2BGR)
    for idx, je in enumerate(((prediction, np.ones(21)), (gt, np.ones(21)))):
      joint_coord = je[0]
      valid = je[1]

      for coord in joint_coord:
        cv2.circle(
            new_img_input,
            (int(coord[0]), int(coord[1])),
            4, colors[idx])

      for finger in range(5):
        last_idx = 0
        for joint in range(4):
          coord = joint_coord[last_idx]
          prev = (int(coord[0]), int(coord[1]))

          curr_idx = 1 + finger*4 + joint

          coord = joint_coord[curr_idx]
          next = (int(coord[0]), int(coord[1]))

          if (valid[curr_idx] and valid[last_idx]):
            cv2.line(new_img_input, prev, next, colors[idx], 1)
          last_idx = curr_idx
    return new_img_input

  def __len__(self):
    return len(self.cam_annotation_hand_file)

  def __getitem__(self, idx):

    b = self.cam_annotation_hand_file.iloc[idx]

    hand_idx = int(b[0])
    mode = str(b[1])
    capture_idx = int(b[2])  # 0 to 26 for train
    name = str(b[3])
    cam_id = int(b[4])
    frame_idx = int(b[5])
    # .index() is slow as heck
    really_sequential_frame_idx = int(b[6])

    # Lazy load
    if (self.joints[mode][capture_idx] == None):

      print("2")
      joint_name = osp.join(
          annot_path, mode, 'InterHand2.6M_' + mode + f'_joint_3d{capture_idx}.json')
      with open(joint_name) as f:
        joints = json.load(f)

      self.joints[mode][capture_idx] = joints

    img_file = osp.join(
        img_path, mode, f"Capture{capture_idx}", name, f"cam{cam_id}", f"image{frame_idx}.jpg")

    joints_curr, valids_gt = self.to_cam_coord(self.cams[mode], capture_idx, cam_id,
                                               self.joints[mode][capture_idx][str(frame_idx)], hand_idx)

    prediction_found, predicted = self.simple_try_find_prediction(
        mode, capture_idx, frame_idx, cam_id, hand_idx, really_sequential_frame_idx)

    if not prediction_found:
      predicted = joints_curr
    else:
      predicted = wobble_predictions_3d(predicted)

    predicted_px = self.to_px_coord(
        self.cams[mode], capture_idx, cam_id, predicted, hand_idx)
    gt_px = self.to_px_coord(
        self.cams[mode], capture_idx, cam_id, joints_curr, hand_idx)

    predicted_px = np.concatenate((predicted_px, np.array([[0, 0, 0]])))
    gt_px = np.concatenate((gt_px, np.array([[0, 0, 0]])))
    valids_gt = np.concatenate((valids_gt, np.array([0])))

    # Shouldn't need IGNORE_ORIENTATION
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE |
                     cv2.IMREAD_IGNORE_ORIENTATION)
    img = img.astype(np.float32) * 1.0/255.0

    give_as_predicted = None

    if prediction_found:
      give_as_predicted = predicted_px

    is_right = (hand_idx == 1)

    return self.augmaker.do_one_augmentation(img, gt_px, valids_gt, valids_gt, give_as_predicted, None, is_right)


class AwfulCombinedInterHandDataset(torch.utils.data.Dataset):
  num_times_to_repeat: int
  actual_size: int

  def __init__(self):
    names = [os.path.join("/TRAINDATA_FAST/inter_parse_files_sequential", f) for f in os.listdir(
        "/TRAINDATA_FAST/inter_parse_files_sequential") if not f.endswith("frame_indices")][:2]

    dses = []

    for name in names:
      print(name)
      dses.append(InterSequentialDataset(name))

    self.ds = torch.utils.data.ConcatDataset(dses)
    self.actual_size = len(self.ds)
    self.num_times_to_repeat = 1

  def __len__(self):
    return self.actual_size * self.num_times_to_repeat

  def __getitem__(self, idx):
    return self.ds[idx % self.actual_size]


def get_combined_interhand_dataset():
  dses = []
  d = torch.utils.data.ConcatDataset(dses)
  return d


if __name__ == "__main__":

  d = InterSequentialDataset(
      f"{datasets_basepath}/hd/train_1_0390_dh_touchROM")

  gts_valid = 0
  gts_invalid = 0

  preds_valid = 0
  preds_invalid = 0

  valid = 0
  invalid = 0

  for i in range(0, 500000):
    r = d[i]
    print("l44 ->", i)
