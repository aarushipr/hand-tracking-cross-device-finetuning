import json
import math
import cv2
import numpy as np
import os
from opencv_transforms import transforms
import random
import csv
import torch


class MungeWriter:
  def __init__(self, base_path, dataset_name):

    self.columns = ["hand_filename",
                    "WRIST_X", "WRIST_Y",
                    "THMB_MCP_X", "THMB_MCP_Y",
                    "THMB_PXM_X", "THMB_PXM_Y",
                    "THMB_DST_X", "THMB_DST_Y",
                    "THMB_TIP_X", "THMB_TIP_Y",

                    "INDX_PXM_X", "INDX_PXM_Y",
                    "INDX_INT_X", "INDX_INT_Y",
                    "INDX_DST_X", "INDX_DST_Y",
                    "INDX_TIP_X", "INDX_TIP_Y",

                    "MIDL_PXM_X", "MIDL_PXM_Y",
                    "MIDL_INT_X", "MIDL_INT_Y",
                    "MIDL_DST_X", "MIDL_DST_Y",
                    "MIDL_TIP_X", "MIDL_TIP_Y",

                    "RING_PXM_X", "RING_PXM_Y",
                    "RING_INT_X", "RING_INT_Y",
                    "RING_DST_X", "RING_DST_Y",
                    "RING_TIP_X", "RING_TIP_Y",

                    "LITL_PXM_X", "LITL_PXM_Y",
                    "LITL_INT_X", "LITL_INT_Y",
                    "LITL_DST_X", "LITL_DST_Y",
                    "LITL_TIP_X", "LITL_TIP_Y"]

    self.outputIdx: int = 0
    self.csvfile = open(os.path.join(
        base_path, f"{dataset_name}.csv"), "w+", newline='')
    self.augwriter = csv.writer(
        self.csvfile, delimiter=" ", quotechar='|', quoting=csv.QUOTE_NONNUMERIC)

  def write_thing(self, sub_filename, keypoints, keypoints_valid, is_right=False, has_mask=False, mask_filename=None):
    kp_list = []
    valid_list = []
    row = [sub_filename]

    for i in range(22):
      row += [float(keypoints[i][0]), float(keypoints[i][1]),
              float(keypoints[i][2])]
    for i in range(22):
      row += [float(keypoints_valid[i][0]),
              float(keypoints_valid[i][1]), float(keypoints_valid[i][2])]

    row.append(bool(is_right))

    if has_mask:
      row.append(mask_filename)

    self.augwriter.writerow(row)
    self.outputIdx += 1

  def close(self):
    self.csvfile.close()
