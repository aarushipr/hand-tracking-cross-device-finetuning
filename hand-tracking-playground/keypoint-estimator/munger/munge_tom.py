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

import pandas as pd

from settings import datasets_basepath


from munge_writer import MungeWriter


'''
Bad candidates:
FB*
p1,p2

bad-ish: 29, not seperated between L and R
28: same


good-ish:
synth2 - seems to be synthetic, only left hand. good! but, the forearm isn't long enough - bad!

synth5 - ditto, also only left?

16 is different! only right hands, I think! No, 16's keypoints don't make any sense. Investigate later
17 is just right hands!
23 is just right hands!
24, like the last two, is a continuous video, with just right hands. This one is slighly worse because the background doesn't change - same monitor, no movement. (Was this recorded on his phone?)
26: right hand
27: right hand

meh:
13 - continuous-ish, might be able to identify ranges

14 ditto. 24-141 is left hand
332-413 are right hand

620-847 are right hand
15 ditto
25: has left hand in the middle, ugh


'''


def get_num_unique_hands():
  count = 0
  for num in ["17", "23", "24", "26", "27"]:
    dir = sorted(os.listdir(os.path.join(
        f"{datasets_basepath}/openhands", "Pose", num, "points")))
    count += len(dir)
  return count


subdir = "openhands"

data_root = f"{datasets_basepath}/openhands/Pose/"

base_path = f"{datasets_basepath}/munge_april26"
sub_path = "tom"


def do_the_thing():
  num_unique_hands = get_num_unique_hands()

  bob = MungeWriter(base_path, sub_path)

  for num in ["17", "23", "24", "26", "27"]:
    dir = sorted(os.listdir(os.path.join(data_root, num, "points")))

    for d in dir:
      e = pd.read_csv(f"{data_root}/{num}/points/{d}",
                      delimiter=" ", header=None)
      img_path = f"{data_root}/{num}/images/{d}.jpg"
      img = cv2.imread(img_path)
      jts = []
      valids = []

      for i in range(21):
        x = e.iloc[i, 0]
        y = e.iloc[i, 1]
        jts.append(np.array([x, y, 0]))
        valids.append(np.array((1, 1, 0)))
      # Forearm keypoint, which we don't have.
      jts.append(np.array((0, 0, 0)))
      valids.append(np.array((0, 0, 0)))

      bob.write_thing(
          f"tom/{num}/images/{d}.jpg",
          jts,
          valids,
          is_right=True)
      l = []
      for jt in jts:
        l.append(list(jt))

      print("tom")
  bob.close()


if __name__ == "__main__":
  print(get_num_unique_hands())
  do_the_thing()
