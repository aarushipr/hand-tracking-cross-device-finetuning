import cv2
import numpy as np


from typing import Union, List
from pathlib import Path
import re
import random
import epic_kitchens.hoa
import os
import json


def math_map_ranges(value: float, from_low: float, from_high: float, to_low: float, to_high: float):
    return (value - from_low) * (to_high - to_low) / (from_high - from_low) + to_low


root = "kitchen_formatted"

files = sorted(os.listdir(root))

names = [thing[:-5] for thing in files if thing.endswith(".json")]


for name in names[3:]:
    with open(os.path.join(root, name+".json")) as j_f:
        j = json.load(j_f)

    print("WHAT")
    lob = j["num_frames"]
    for i in range(lob-500, lob):  # len(video_detections)):

        out_filename = 'frame_{:010d}.jpg'.format(i)

        frame = cv2.imread(os.path.join(root, name, out_filename))
        bob = frame

        for hand in j["frames"][i]["hands"]:
            print(hand)
            top = hand["top"]
            bottom = hand["bottom"]
            left = hand["left"]
            right = hand["right"]
            # left = math_map_ranges(hand.bbox.left, 0, 1, -((426-320)/2)/320, 1+((426-320)/2)/320,       )
            # right = math_map_ranges(hand.bbox.right, 0, 1, -((426-320)/2)/320, 1+((426-320)/2)/320,       )

            top_in_px = top * bob.shape[0]
            bottom_in_px = bottom * bob.shape[0]

            left_in_px = left * bob.shape[1]
            right_in_px = right * bob.shape[1]

            # cv2.line(bob, (int(left_in_px), int(top_in_px)), (int(right_in_px), int(top_in_px)), 250, thickness=1)
            # cv2.line(bob, (int(right_in_px), int(top_in_px)), (int(right_in_px), int(bottom_in_px)),  250, thickness=1)
            # cv2.line(bob, (int(right_in_px), int(bottom_in_px)), (int(left_in_px), int(bottom_in_px)), 250, thickness=1)
            # cv2.line(bob, (int(left_in_px), int(bottom_in_px)), (int(left_in_px), int(top_in_px)), 250, thickness=1)
        cv2.imshow("hi", frame)
        cv2.waitKey(0)
