import torch
import numpy as np
import cv2
import json
import os
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../'))

import py.training.common.a_geometry as geo
import augmentation
from dataclasses import dataclass

from a_structs import *
import local_config

'''
/excluded_epics_from_sync/kitchen_labels/
  -> P**
    ->P**_**
      ->*****.csv

/media/moses/TRAINDATA1/EPIC-KITCHENS/
  -> P**/rgb_frames/
    -> frame_0000009908.jpg

'''


class OneEpicKitchensDataset(torch.utils.data.Dataset):
    '''
    Made for expedience's sake - there are a lot of sequences!
    '''
    filename_list: list
    annotation_folder: str
    img_folder: str

    def __init__(self, annotation_folder, img_folder):
        self.filename_list = []
        self.annotation_folder = annotation_folder
        self.img_folder = img_folder
        # self.list_of_seq_roots_ann.append(os.path.join(subject, subject_sequence))
        # self.list_of_seq_roots_img.append(os.path.join(subject, "rgb_frames", subject_sequence))
        for fn in sorted(os.listdir(img_folder)):
            self.filename_list.append(fn[:-4])

    def __getitem__(self, idx):

        filename_without_extension = self.filename_list[idx]

        img_file = os.path.join(
            self.img_folder,
            filename_without_extension + ".jpg")

        im = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)

        im_w = geo.npImgWidth(im)
        im_h = geo.npImgHeight(im)

        bbox_list = [None, None]
        with open(os.path.join(self.annotation_folder, filename_without_extension+".csv")) as bsv:
            for a in bsv.readlines():
                # print(a)
                a = a.split(" ")
                b = bbox(float(a[0])*im_w, float(a[1])*im_h,
                         float(a[2])*im_w, float(a[3])*im_h)
                if (a[4] == "1"):
                    # right hand
                    bbox_list[1] = b
                else:
                    # left hand
                    bbox_list[0] = b

        e = ImageWithBoundingBoxes(image=im, bboxes=bbox_list)
        e = augmentation.augment_image(e)
        e = augmentation.imgwithboundingboxes320_to_heatmaps_2hand(e)

        return e
    def __len__(self):
      return len(self.filename_list)


def EpicKitchensDataset(fast: bool = False):
    ann_root = local_config.kitchens_annotations
    img_root = local_config.kitchens_images
    datasets = []

    subjects = sorted(os.listdir(ann_root))

    for subject_idx, subject in enumerate(subjects):
        if local_config.kitchens_only_1st_sequence:
          if subject_idx > 0:
            break
        print(f"Loading subject idx {subject_idx}/{len(subjects)}")
        for sequence_idx, subject_sequence in enumerate(sorted(os.listdir(os.path.join(ann_root, subject)))):
            if fast:
              if sequence_idx > 0:
                break
            annotation_folder = os.path.join(
                ann_root, subject, subject_sequence)
            img_folder = os.path.join(
                img_root, subject, "rgb_frames", subject_sequence)
            datasets.append(OneEpicKitchensDataset(
                annotation_folder, img_folder))
    return torch.utils.data.ConcatDataset(datasets)



if __name__ == "__main__":
    d = EpicKitchensDataset()

    # samp = d[100]
    for samp in d:
      print(samp)
      cv2.imshow("h", samp["image"][0])
      cv2.waitKey(1)
    # print(samp.bboxes)





# class EpicKitchensSequenceProvider(torch.utils.data.Dataset):
#     ann_root = "/excluded_epics_from_sync/kitchen_labels/"
#     img_root = "/media/moses/TRAINDATA1/EPIC-KITCHENS"

#     def __init__(self):
#         self.list_of_seq_roots_img = []
#         self.list_of_seq_roots_ann = []
#         self.filename_lists = []

#         datasets = []
#         # for guy in sorted(os.listdir(self.ann_root))[:1]:
#         for subject in sorted(os.listdir(self.ann_root)):
#             for subject_sequence in sorted(os.listdir(os.path.join(self.ann_root, subject))):
#                 fn_list = []
#                 self.list_of_seq_roots_ann.append(
#                     os.path.join(subject, subject_sequence))
#                 self.list_of_seq_roots_img.append(os.path.join(
#                     subject, "rgb_frames", subject_sequence))
#                 for fn in sorted(os.listdir(os.path.join(self.img_root, self.list_of_seq_roots_img[-1]))):
#                     fn_list.append(fn[:-4])
#                 self.filename_lists.append(fn_list)

#     def getSequences(self):
#         out = []
#         for l in self.filename_lists:
#             out.append(len(l))
#         return out

#     def getSample(self, sequence, index_within_sequence):
#         img_file = os.path.join(
#             self.img_root, self.list_of_seq_roots_img[sequence],
#             self.filename_lists[sequence][index_within_sequence] + ".jpg")
#         e = ImageWithBoundingBoxes(
#             image=cv2.imread(img_file, cv2.IMREAD_GRAYSCALE))

#         with open(os.path.join(self.ann_root, self.list_of_seq_roots_ann[sequence], self.filename_lists[sequence][index_within_sequence]+".csv")) as bsv:
#             for a in bsv.readlines():
#                 # print(a)
#                 a = a.split(" ")
#                 b = bbox(float(a[0])*npImgWidth(e.image), float(a[1])*npImgHeight(e.image),
#                          float(a[2])*npImgWidth(e.image), float(a[3])*npImgHeight(e.image))
#                 if (a[4] == "1"):
#                     b.handedness = Handedness.RIGHT
#                 else:
#                     b.handedness = Handedness.LEFT
#                 e.bboxes.append(b)

#         return e