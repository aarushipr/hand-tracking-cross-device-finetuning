import torch.nn as nn
from torch.utils.data import DataLoader

import cv2
import numpy as np


import py.training.common.a_geometry as geo

from py.training.common.a_geometry import *

def draw_rectangle_in_image_px_coord(image, top, bottom, left, right, color):
    # width = image.shape[1]
    # height = image.shape[0]
    # left *= width
    # right *= width
    # top *= height
    # bottom *= height
    # I wish I had any idea why the normal cv2.rectangle() crashes my machine.
    # like it legit crashes my GPU and I have to restart my computer.
    cv2.line(image, (int(left), int(top)),
             (int(right), int(top)), color, thickness=1)
    cv2.line(image, (int(right), int(top)),
             (int(right), int(bottom)), color, thickness=1)
    cv2.line(image, (int(right), int(bottom)),
             (int(left), int(bottom)), color, thickness=1)
    cv2.line(image, (int(left), int(bottom)),
             (int(left), int(top)), color, thickness=1)


def draw_square_in_image_px_coord(image, center, square_side, color):
    draw_rectangle_in_image_px_coord(
        image, center[1] - (square_side / 2),
        center[1] + (square_side / 2),
        center[0] - (square_side / 2),
        center[0] + (square_side / 2),
        color)

def visualize_directreg(img, exists, center_x, center_y, size, name):
    vis_pred = cv2.cvtColor(img[0], cv2.COLOR_GRAY2BGR)

    colors = [(0, 1, 1), (0, 0, 1)]

    for i in range(2):
        if exists[i] < 0.5:
            continue
        cx = geo.map_ranges(center_x[i], -1, 1, 0, geo.npImgWidth(vis_pred))
        cy = geo.map_ranges(center_y[i], -1, 1, 0, geo.npImgHeight(vis_pred))
        sz = geo.map_ranges(size[i], 0, 1, 0, geo.npImgWidth(vis_pred))

        draw_square_in_image_px_coord(vis_pred, (cx, cy), sz, colors[i])

    cv2.imshow(f"{name}img", vis_pred)
