import torch
import numpy as np
import cv2
import json
import os
import sys

if __name__ == "__main__":
    from common import visualize_directreg
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../'))

import py.training.common.a_geometry as geo
import augmentation
from dataclasses import dataclass

from a_structs import *


class DarknetDataset(torch.utils.data.Dataset):

    def __init__(self, root):
        self.root = root
        # self.num_frames = len(os.listdir(os.path.join(self.root, "images", "train")))
        self.files = [label[:-4]
                      for label in sorted(os.listdir(os.path.join(self.root, "labels", "train")))]
        self.num_frames = len(self.files)

    def __len__(self):
        return self.num_frames

    def __getitem__(self, index_within_sequence):
        fn = self.files[index_within_sequence]
        img_file = os.path.join(self.root, "images", "train", fn+".jpg")
        im = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        im_w = geo.npImgWidth(im)
        im_h = geo.npImgHeight(im)

        labels_file = open(os.path.join(os.path.join(
            self.root, "labels", "train"), fn+".txt")).readlines()

        bbox_list = [None, None]
        for label in labels_file:
            a = label.split()
            b = bbox(float(a[1])*im_w, float(a[2])*im_h,
                     float(a[3])*im_w, float(a[4])*im_h)
            cls = int(a[0])
            if cls == -1:
                # Unknown handedness - not using it
                continue
            if cls == 0:
                # ego left hand
                bbox_list[cls] = b
            if cls == 1:
                # ego right hand
                bbox_list[cls] = b
            if cls == 2:
                # other left hand - don't use
                continue
            if cls == 3:
                # other right hand - don't use
                continue
        e = ImageWithBoundingBoxes(image=im, bboxes=bbox_list)
        e = augmentation.augment_image(e)
        e = augmentation.imgwithboundingboxes320_to_heatmaps_2hand(e)

        return e


if __name__ == "__main__":
    d = DarknetDataset("/excluded_epics_from_sync/initial_frontend_T32624/hand-convert/ego-hands/")

    # samp = d[100]
    for samp in d:
        print(samp)
        visualize_directreg(samp["image"], samp["exists"], samp["center_x"], samp["center_y"], samp["size"], "asd")
        cv2.waitKey(0)
