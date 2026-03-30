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


def load_file_get_properties(st, root_dir, name):

    # Don't load UI - we want the artist to be able to leave whatever they want there and get the same UI as a programmer
    # Do use scripts (even though this is a security risk if we were loading unknown files) - we need them for Drivers for corrective blendshapes.

    bpy.ops.wm.open_mainfile(filepath=str(os.path.join(
        root_dir, name+".blend")), load_ui=False, use_scripts=True)

    with open(os.path.join(root_dir, name+".json")) as f:
        j = json.load(f)
        st.arm_scale = j["arm_scale"]


def main():
    st = State()

    names = ["3dscanstore_black_male",
             "3dscanstore_black_female",
             "3dscanstore_white_male",
             "3dscanstore_white_female",
              "3dscanstore_asian",
             #  "mblab_light",
             "uhh"]

    for name in names:

        print(f"Loading {name}!")

        load_file_get_properties(st, "/3/epics/artificial_data_3/hands/",
                                 name)

        e = bpy.data.objects["thumb_tip_target"]
        print(e)
        e = bpy.data.objects["index_tip_target"]
        print(e)
        e = bpy.data.objects["middle_tip_target"]
        print(e)
        e = bpy.data.objects["ring_tip_target"]
        print(e)
        e = bpy.data.objects["little_tip_target"]
        print(e)


if __name__ == "__main__":
    main()
