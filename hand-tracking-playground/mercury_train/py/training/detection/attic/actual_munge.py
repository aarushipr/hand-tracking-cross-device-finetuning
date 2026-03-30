import cv2
import numpy as np


from typing import Union, List
from pathlib import Path
import re
import random
import epic_kitchens.hoa
import os
import json
import csv

def math_map_ranges(value: float, from_low: float, from_high: float, to_low: float, to_high: float):
	return (value - from_low) * (to_high - to_low) / (from_high - from_low) + to_low

def process_hands(hands):
  # We get a list of hands, each of which has a score.
  # Since this is epic-kitchens, there are only two hands. If we find more than one hand on each side, then we pick the one with the best score.
  # Also we don't use hands with a low score.
  best_left_score: float = -1
  hand_left = None

  best_right_score: float = -1
  hand_right = None

  for hand in hands:
    # or (hand.score < 0.0001)
    if (hand.score > 0.2) :  # Low. Better to over-detect than under-detect, says Moses on jan 21+feb8
      if hand.side == epic_kitchens.hoa.HandSide.RIGHT:
        if (hand.score > best_right_score):
          hand_right = hand
          best_right_score = hand.score
      if hand.side == epic_kitchens.hoa.HandSide.LEFT:
        if (hand.score > best_left_score):
          hand_left = hand
          best_left_score = hand.score
  out = []
  if hand_left is not None:
    out.append(hand_left)
  if hand_right is not None:
    out.append(hand_right)
  return out


roots = []

# guys = ["P01", "P02", "P03", "P04", "P05", "P06", "P07"]
# guys = ["P03", "P04", "P05", "P06", "P07"]
# guys = ["P04", "P05", "P06", "P07", "P08", "P09", "P10"]

# guys =  ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09", "P10", "P11"]

guys = sorted(os.listdir("/home/moses/EPIC-KITCHENS/"))

num_images = 0
num_one_hand = 0
num_two_hand = 0

for guy in guys:
  root = f"/home/moses/EPIC-KITCHENS/{guy}"

  out_root = "../kitchen_labels_no_zero_hands"

  files = sorted(os.listdir(os.path.join(root, "hand-objects"))) #= P01_01.pkl, P01_02, etc.

  names = [thing[:-4] for thing in files]
  print(names)


  for name in names:
    print(f"mkdir -p {os.path.join(out_root, guy, name)}")
    os.system(f"mkdir -p {os.path.join(out_root, guy, name)}")
    # file = open(os.path.join(out_root, guy, name))
    # csv.writer()
    # os.system(f"mkdir -p {os.path.join(out_root, name)}")

    print(os.path.join(root, "hand-objects", name+".pkl"))
    video_detections = epic_kitchens.hoa.load_detections(os.path.join(root, "hand-objects", name+".pkl"))
    # print(video_detections)

    i = 0
    print(video_detections[233].hands)


    for i in range(len(video_detections)):
      filename = 'frame_{:010d}.jpg'.format(i+1)
   
      this_frame_data = video_detections[i]

      csvfile = open(os.path.join(out_root, guy, name, "frame_{:010d}.csv").format(i+1), "w+")

      lines = []

      hands = process_hands(this_frame_data.hands)
      # print(hands)

      there_is_hand = False

      num_images += 1

      if len(hands) == 1:
        num_one_hand += 1
      if len(hands) == 2:
        num_two_hand += 1

      



      for hand in hands:
        there_is_hand = True
        # top = hand.bbox.top*720
        # bottom = hand.bbox.bottom*720
        # left = math_map_ranges(hand.bbox.left, 0, 1, -((426-320)/2)/320, 1+((426-320)/2)/320,       )
        # right = math_map_ranges(hand.bbox.right, 0, 1, -((426-320)/2)/320, 1+((426-320)/2)/320,       )
        top = hand.bbox.top
        bottom = hand.bbox.bottom
        left = hand.bbox.left
        right = hand.bbox.right

        cx = (left+right)/2
        cy = (top+bottom)/2

        w = abs(left-right)
        h = abs(top-bottom)

        if hand.side == epic_kitchens.hoa.HandSide.RIGHT:
          cls = 1
        else:
          cls = 0

        stri = f"{cx} {cy} {w} {h} {cls} {hand.score}\n"
        lines.append(stri)
      if (there_is_hand):
        csvfile.writelines(lines);
      csvfile.close();
        
print(num_images, num_one_hand, num_two_hand)