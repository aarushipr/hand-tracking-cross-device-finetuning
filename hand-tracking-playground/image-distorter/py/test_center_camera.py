from dataclasses import dataclass
import sys  # nopep8
import os  # nopep8
sys.path.insert(0, os.path.dirname(__file__))  # nopep8

import traceback


import readcsv_fingerpose
import readcsv_wristpose
import bpy
import mathutils
import mlib
# import guys_we_like
# import mblab
import random
import math
import csv
import pandas as pd
import numpy as np

import subprocess
import json

import header
from header import State
from header import env_settings


make_guy = True


XRT_HAND_JOINT_PALM = 0
XRT_HAND_JOINT_WRIST = 1
XRT_HAND_JOINT_THUMB_METACARPAL = 2
XRT_HAND_JOINT_THUMB_PROXIMAL = 3
XRT_HAND_JOINT_THUMB_DISTAL = 4
XRT_HAND_JOINT_THUMB_TIP = 5
XRT_HAND_JOINT_INDEX_METACARPAL = 6
XRT_HAND_JOINT_INDEX_PROXIMAL = 7
XRT_HAND_JOINT_INDEX_INTERMEDIATE = 8
XRT_HAND_JOINT_INDEX_DISTAL = 9
XRT_HAND_JOINT_INDEX_TIP = 10
XRT_HAND_JOINT_MIDDLE_METACARPAL = 11
XRT_HAND_JOINT_MIDDLE_PROXIMAL = 12
XRT_HAND_JOINT_MIDDLE_INTERMEDIATE = 13
XRT_HAND_JOINT_MIDDLE_DISTAL = 14
XRT_HAND_JOINT_MIDDLE_TIP = 15
XRT_HAND_JOINT_RING_METACARPAL = 16
XRT_HAND_JOINT_RING_PROXIMAL = 17
XRT_HAND_JOINT_RING_INTERMEDIATE = 18
XRT_HAND_JOINT_RING_DISTAL = 19
XRT_HAND_JOINT_RING_TIP = 20
XRT_HAND_JOINT_LITTLE_METACARPAL = 21
XRT_HAND_JOINT_LITTLE_PROXIMAL = 22
XRT_HAND_JOINT_LITTLE_INTERMEDIATE = 23
XRT_HAND_JOINT_LITTLE_DISTAL = 24
XRT_HAND_JOINT_LITTLE_TIP = 25


# WXYZ, not XYZW.
sqrt2_2 = math.sqrt(2)/2
if False:
    camera_forward = mathutils.Quaternion((sqrt2_2, sqrt2_2, 0, 0))
    camera_left = mathutils.Quaternion((0.5, 0.5, 0.5, 0.5))
    camera_right = mathutils.Quaternion((0.5, 0.5, -0.5, -0.5))
    camera_top = mathutils.Quaternion((0, 1, 0, 0))
    camera_bottom = mathutils.Quaternion((1, 0, 0, 0))
else:
    camera_forward = mathutils.Quaternion((1, 0, 0, 0))
    camera_left = mathutils.Quaternion((sqrt2_2, 0, sqrt2_2, 0))
    camera_right = mathutils.Quaternion((sqrt2_2, 0, -sqrt2_2, 0))
    camera_top = mathutils.Quaternion((sqrt2_2, sqrt2_2, 0., 0.))
    camera_bottom = mathutils.Quaternion((sqrt2_2, -sqrt2_2, 0., 0.))


def main():
    st = State()




    mlib.fake_get_right_camera_pose(st)
    mlib.fake_get_center_camera_pose(st)
    # print(st.right_in_left_pos, st.right_in_left_rot)

    # init_random(st)

    mlib.stereoscopy()
    mlib.make_cameras(st)



if __name__ == "__main__":
        main()
