import pandas as pd
import random
import numpy as np
import torch
import cv2
import os

from a_aug_config import the_aug_config

from maker_of_augmentations import AugmentationMaker

from settings import datasets_basepath




np.set_printoptions(precision=3, suppress=True)


class RandoDataset(torch.utils.data.Dataset):
  base_path = f"{datasets_basepath}/munge_april26"
  source = "nikitha.csv"
  num_times_to_repeat: int

  def __init__(self, base_path, source):
    self.base_path = base_path
    self.source = source
    self.augmaker = AugmentationMaker(the_aug_config)
    self.csvframe = pd.read_csv(os.path.join(
        self.base_path, self.source), delimiter=" ", quotechar="|")
    self.num_times_to_repeat = 1
    self.actual_size = len(self.csvframe)

  def __len__(self):
    return self.actual_size * self.num_times_to_repeat

  def __getitem__(self, idx):
    idx = idx % self.actual_size
    b = self.csvframe.iloc[idx]

    acc_idx = 0

    filename = b[acc_idx]
    acc_idx += 1

    kps = np.zeros((22, 3))
    gt_xy_valid = np.zeros((22))
    gt_depth_valid = np.zeros((22))

    for i in range(22):
      for j in range(3):
        kps[i][j] = b[acc_idx]
        acc_idx += 1

    for i in range(22):
      gt_xy_valid[i] = b[acc_idx]
      acc_idx += 2
      gt_depth_valid[i] = b[acc_idx]
      acc_idx += 1

    is_right = bool(b[acc_idx])
    acc_idx += 1

    mask_filename = None
    if (len(b) == acc_idx + 1):
      # has mask
      mask_filename = b[acc_idx]

    mask = None

    if (mask_filename):
      path = os.path.join(self.base_path, mask_filename)
      mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
      mask = mask.astype(np.float32) * 1.0/255.0

    img = cv2.imread(os.path.join(self.base_path, filename),
                     cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) * 1.0/255.0

    return self.augmaker.do_one_augmentation(img, kps, gt_xy_valid, gt_depth_valid, predicted_px=None, mask=mask, is_right=is_right)


if __name__ == '__main__':

  a = RandoDataset(f"{datasets_basepath}/munge_april26", "nikitha.csv")
  b = RandoDataset(f"{datasets_basepath}/munge_april26", "tom.csv")

  d = torch.utils.data.ConcatDataset([a, b])

  gts_valid = 0
  gts_invalid = 0

  preds_valid = 0
  preds_invalid = 0

  valid = 0
  invalid = 0

  e = list(range(len(d)))
  random.shuffle(e)

  for i in e:
    r = d[i]
    print("l44 ->", i)
