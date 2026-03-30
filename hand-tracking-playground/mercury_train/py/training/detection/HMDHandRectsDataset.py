import torch
import numpy as np
import cv2
import json
import os
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../'))
    from common import visualize_directreg


import py.training.common.a_geometry as geo
import augmentation
from dataclasses import dataclass

from a_structs import *

class HMDHandRectsDataset(torch.utils.data.Dataset):
    ann = "human_annotated_last.json"
    json: dict

    valid_indices: list


    def __init__(self, root="/3/epics/hand_bbox_T32969/bbox-captures/moses-feb6-livingroom/"):
        self.root = root
    
        self.valid_indices = []
        with open(os.path.join(self.root, self.ann)) as f:
            self.json = json.load(f)


        for idx, j in enumerate(self.json["frames"]):
            valid_sample = j["position_confirmed"]
            if (valid_sample):
                self.valid_indices.append(idx)

        self.json = dict(self.json)



    def __len__(self):
        return len(self.valid_indices)*2

    def __getitem__(self, idx):
        left_camera: bool = (idx % 2 == 0)
        side_str = "left" if left_camera else "right"
        
        # PyTorch asks us for inidividual images, but our annotation json is indexed by frames, so we have two images per frame.

        # So, we need to turn the index PyTorch asked for into an index within the annotation json.
        # First: if this is the right camera, the pytorch index is uneven. Remove one from there
        if not left_camera:
            idx -= 1
        # Then divide by two, so that we're accessing just the capture frame number
        idx /= 2
        idx = int(idx)

        # access by that index, get the camera we want.
        annotation = self.json["frames"][self.valid_indices[idx]][side_str]

        im = cv2.imread(os.path.join(
            self.root, annotation["filename"]), cv2.IMREAD_GRAYSCALE)

        bbox_list = [None, None]

        for h in annotation["hands"]:
            b = bbox(h[0], h[1], h[2], h[3])

            # This will be 0, or 1.
            hand_idx = h[4]
            if not (hand_idx == 0 or hand_idx == 1):
                # This got annotated wrong?
                print(f"Index {self.valid_indices[idx]} in {self.root} has a wrong hand class: {hand_idx}")
                continue
            bbox_list[hand_idx] = b

            # bbox_liste.bboxes.append(b)

        e = ImageWithBoundingBoxes(image=im, bboxes = bbox_list)
        # print("len here is,", len(j["hands"]))

        e = augmentation.augment_image(e)
        e = augmentation.imgwithboundingboxes320_to_heatmaps_2hand(e)

        return e

if __name__ == "__main__":
    d = HMDHandRectsDataset("/3/epics/resurrect_detection/HMDHandRects/sequences/train_subject02_sequence00/")

    for samp in d:
        # samp = d[1]

        visualize_directreg(samp["image"], samp["exists"], samp["center_x"], samp["center_y"], samp["size"], "asd")

        # cv2.imshow("h", samp["image"][0])
        cv2.waitKey(0)
        # print(samp)
