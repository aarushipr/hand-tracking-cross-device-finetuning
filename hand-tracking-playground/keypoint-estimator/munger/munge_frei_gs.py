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


base_path = f"{datasets_basepath}/munge_april26"
sub_path = "frei_gs"
base_load_path = os.path.join(base_path, "FreiHAND_pub_v2")

""" General util functions. """


def _assert_exist(p):
  msg = 'File does not exists: %s' % p
  assert os.path.exists(p), msg


def json_load(p):
  _assert_exist(p)
  with open(p, 'r') as fi:
    d = json.load(fi)
  return d


def projectPoints(xyz, K):
  """ Project 3D coordinates into image space. """
  xyz = np.array(xyz)
  K = np.array(K)
  uv = np.matmul(K, xyz.T).T
  return uv[:, :2] / uv[:, -1:]


def read_msk(idx):
  mask_path = os.path.join(base_load_path, 'training', 'mask',
                           '%08d.jpg' % idx)
  _assert_exist(mask_path)
  return cv2.imread(mask_path)


def get_num_unique_hands():
  return 32560


def do_the_thing():

  K_list = json_load(os.path.join(base_load_path, "training_K.json"))
  xyz_list = json_load(os.path.join(base_load_path, "training_xyz.json"))

  bob = MungeWriter(base_path, sub_path)

  for idx in range(32560):
    K = np.array(K_list[idx])
    xyz = np.array(xyz_list[idx])

    uv = projectPoints(xyz, K)

    hand_pts = []
    valids = []

    for j in range(21):
      x = uv[j, 0]
      y = uv[j, 1]
      hand_pts.append(np.array([x, y, 0]))
      valids.append(np.array([1, 1, 0]))
      # Could do depth here, no time.
    hand_pts.append(np.float32([0, 0, 0]))
    valids.append(np.float32([0, 0, 0]))

    print(idx)
    fn = os.path.join("FreiHAND_pub_v2", "training", 'rgb', '%08d.jpg' % idx)

    bob.write_thing(fn, hand_pts, valids, is_right=True)

    print("frei")


if __name__ == "__main__":
  do_the_thing()
