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


root = "/home/moses/EPIC-KITCHENS/P01"

out_root = "kitchen_formatted"

files = sorted(os.listdir(os.path.join(root, "masks")))

names = [thing[:-4] for thing in files]
print(names)

start = 500

for name in names[1:]:
    os.system(f"mkdir -p {os.path.join(out_root, name)}")
    videofile = os.path.join(root, "videos", name+".MP4")
    print(f"opening video at {videofile}")
    eart = cv2.VideoCapture(videofile)
    # video_id = 'P01_01'
    # participant_id = video_id[:3]
    print(os.path.join(root, "masks", name+".pkl"))
    video_detections = epic_kitchens.hoa.load_detections(
        os.path.join(root, "hand-objects", name+".pkl"))

    dork = {"num_frames": len(video_detections), "frames": []}

    i = 0
    while i < len(video_detections):

        this_frame_data = video_detections[i]

        ret, frame = eart.read()
        print(i, ret)
        if ret == False:
            print("BAD!")
            continue
        # cv2.imshow("hi", frame)
        # cv2.waitKey(1)
        # print(frame.shape)
        out_filename = 'frame_{:010d}.jpg'.format(i)
        desired_height = 240
        desired_width = int(frame.shape[1] * (desired_height/frame.shape[0]))
        # print("width",  desired_width)
        bob = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bob = cv2.resize(bob, (desired_width, desired_height))
        diff = desired_width-320
        b = int(diff/2)

        bob = bob[:, b: -b]

        this_frame = {"idx": i, "filename": out_filename, "hands": []}

        for hand in this_frame_data.hands:
            top = hand.bbox.top
            bottom = hand.bbox.bottom
            left = math_map_ranges(
                hand.bbox.left, 0, 1, -((426-320)/2)/320, 1+((426-320)/2)/320,)
            right = math_map_ranges(
                hand.bbox.right, 0, 1, -((426-320)/2)/320, 1+((426-320)/2)/320,)

            top_in_px = top * bob.shape[0]
            bottom_in_px = bottom * bob.shape[0]

            left_in_px = left * bob.shape[1]
            right_in_px = right * bob.shape[1]

            cv2.line(bob, (int(left_in_px), int(top_in_px)),
                     (int(right_in_px), int(top_in_px)), 250, thickness=1)
            cv2.line(bob, (int(right_in_px), int(top_in_px)),
                     (int(right_in_px), int(bottom_in_px)),  250, thickness=1)
            cv2.line(bob, (int(right_in_px), int(bottom_in_px)),
                     (int(left_in_px), int(bottom_in_px)), 250, thickness=1)
            cv2.line(bob, (int(left_in_px), int(bottom_in_px)),
                     (int(left_in_px), int(top_in_px)), 250, thickness=1)
            side_str = "left"
            if hand.side == epic_kitchens.hoa.HandSide.RIGHT:
                side_str = "right"
            h = {"side": side_str, "top": top,
                 "bottom": bottom, "left": left, "right": right}
            this_frame["hands"].append(h)
        dork["frames"].append(this_frame)

        # print(bob.shape)
        cv2.imwrite(os.path.join(out_root, name, out_filename), bob)

        # if i > start:
        cv2.imshow("hi", bob)
        key = cv2.waitKey(1)
        if key == ord("q"):
            exit(0)
        i += 1
        # print(i)
        # print(dork)
    with open(os.path.join(out_root, f"{name}.json"), "w+") as f:
        json.dump(dork, f)
