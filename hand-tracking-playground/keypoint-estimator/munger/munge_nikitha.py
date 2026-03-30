import json
import math
import cv2
import numpy as np
import os
from opencv_transforms import transforms
import glob
import random
import smallestenclosingcircle
import csv

from munge_writer import MungeWriter

from settings import datasets_basepath


rr = f"{datasets_basepath}/nikitha_october"

base_path = f"{datasets_basepath}/munge_april26"
sub_path = "nikitha"


def make_keypoints_good(in_keypoints):
  keypointList = []
  validList = []
  for i in range(0, 21):
    x = in_keypoints[i*3]
    y = in_keypoints[i*3 + 1]
    v = in_keypoints[(i*3) + 2]
    keypointList.append(np.array((x, y, 0)))
    validList.append(np.array((1, 1, 0)))
  keypointList.append(np.array((0, 0, 0)))  # Ignored, forearm keypoint
  validList.append(np.array((0, 0, 0)))

  return keypointList, validList


def get_num_unique_hands():
  count = 0
  for thou in range(29, 38):
    idx = 0  # starts at 57
    with open(f"{rr}/nikitha-1635539816/{thou}/annotations-{thou}.json") as f:
      j = json.load(f)
    while idx < len(j["things"]):
      thing = j["things"][idx]
      for view_key, view in zip(("view_left", "view_right"), ("left", "right")):
        if "keypoints_left_hand" in thing[view_key]:
          count += 1
        if "keypoints_right_hand" in thing[view_key]:
          count += 1
      idx += 1
  return count


def do_the_thing():

  bob = MungeWriter(base_path, sub_path)

  for thoue in range(29, 38):
    thou = thoue
    idx = 0
    f = open(f"{rr}/nikitha-1635539816/{thou}/annotations-{thou}.json")
    j = json.load(f)

    while idx < len(j["things"]):
      thing = j["things"][idx]
      for view_key, view in zip(("view_left", "view_right"), ("left", "right")):
        if ("keypoints_left_hand" not in thing[view_key]) and ("keypoints_right_hand" not in thing[view_key]):
          continue
        filename = f"{sub_path}/{thou}/{thing['source_id']}-{view}.jpg"
        filename_mask = f"{sub_path}/{thou}/{thing['source_id']}-{view}_mask.jpg"
        num_times = 1
        if "keypoints_left_hand" in thing[view_key]:
          kps, valids = make_keypoints_good(
              thing[view_key]["keypoints_left_hand"])
          bob.write_thing(
              filename,
              kps,
              valids,
              is_right=False,
              has_mask=True,
              mask_filename=filename_mask
          )

        # NOT elif
        if "keypoints_right_hand" in thing[view_key]:
          kps, valids = make_keypoints_good(
              thing[view_key]["keypoints_right_hand"])

          bob.write_thing(
              filename,
              kps,
              valids,
              is_right=True,
              has_mask=True,
              mask_filename=filename_mask
          )

      idx += 1
      print("nikitha")
  bob.close()


if __name__ == "__main__":
  do_the_thing()
