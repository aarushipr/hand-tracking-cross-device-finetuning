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


load_base = f"{datasets_basepath}/munge_april26/panoptic/synthetic"

base_path = f"{datasets_basepath}/munge_april26"
dataset_name = "panoptic_synth"


def get_num_unique_hands():
  b = 0
  for dir in "manual_train", "manual_test":

    jsonfiles = sorted([f for f in os.listdir(
        os.path.join(load_base, dir)) if f.endswith('.json')])
    for jsonfile in jsonfiles:
      with open(os.path.join(load_base, dir, jsonfile), "r") as fid:
        dat = json.load(fid)
      pts = np.array(dat["hand_pts"])
      invalid = pts[:, 2] != 1
      print(f"{b}/{len(jsonfiles)}")
      if (invalid.any()):
        continue
      b += 1
  return b


def do_the_thing(aug_config, requested_num_images):

  bob = MungeWriter(base_path, dataset_name)
  for dir in ["synth1", "synth2", "synth3", "synth4"]:
    jsonfiles = sorted([f for f in os.listdir(
        os.path.join(load_base, dir)) if f.endswith('.json')])

    b = 0
    for jsonfile in jsonfiles:
      with open(os.path.join(load_base, dir, jsonfile), "r") as fid:
        dat = json.load(fid)
      pts = np.array(dat["hand_pts"])
      invalid = pts[:, 2] != 1
      print(f"{b}/{len(jsonfiles)}")
      b += 1
      print(invalid)
      if (invalid.any()):
        print("invalid!")
        continue
      else:
        print("valid!")
      is_left = dat["is_left"]

      hand_pts = []
      valids = []
      for pt in dat["hand_pts"]:
        hand_pts.append(np.float32([pt[0], pt[1], 0]))
        valids.append(np.float32([1, 1, 0]))
      hand_pts.append(np.float32([0, 0, 0]))
      valids.append(np.float32([0, 0, 0]))

      fn = os.path.join("panoptic", "synthetic", dir, jsonfile[0:-5]+".jpg")

      bob.write_thing(fn, hand_pts, valids, not is_left)

      print("juman")


if __name__ == "__main__":
  do_the_thing(1, 1)
