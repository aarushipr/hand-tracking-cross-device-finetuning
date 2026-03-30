import json
import math
import cv2
import numpy as np
import os
from opencv_transforms import transforms
import random
import csv
import sys
from munge_writer import MungeWriter

from settings import datasets_basepath


load_base = f"{datasets_basepath}/munge_april26/panoptic/panoptic"

base_path = f"{datasets_basepath}/munge_april26"
dataset_name = "panoptic_panoptic"


folderPath = "panoptic/panoptic"
jsonPath = os.path.join(base_path, folderPath, "hands_v143_14817.json")


def get_num_unique_hands():

  with open(jsonPath, 'r') as fid:
    dat_all = json.load(fid)
    dat_all = dat_all["root"]

  count = 0

  for dat in dat_all:
    pts = np.array(dat["joint_self"])
    invalid = pts[:, 2] != 1

    if (invalid.any()):
      continue
    count += 1
  print(count)
  return count


def do_the_thing(aug_config, requested_num_images):

  bob = MungeWriter(base_path, dataset_name)

  with open(jsonPath, 'r') as fid:
    dat_all = json.load(fid)
    dat_all = dat_all["root"]

  count = 0
  for dat in dat_all:
    print(count)
    count += 1
    pts = np.array(dat["joint_self"])
    invalid = pts[:, 2] != 1

    if (invalid.any()):
      continue

    hand_pts = []
    valids = []

    for pt in dat["joint_self"]:
      hand_pts.append(np.float32([pt[0], pt[1], 0]))
      valids.append(np.float32([1, 1, 0]))
    hand_pts.append(np.float32([0, 0, 0]))
    valids.append(np.float32([0, 0, 0]))
    fn = os.path.join("panoptic", "panoptic", dat["img_paths"])
    bob.write_thing(fn, hand_pts, valids, is_right=True)

    print("panoptic")


if __name__ == "__main__":
  do_the_thing(1, 1)
