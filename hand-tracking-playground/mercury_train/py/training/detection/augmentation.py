import header

import cv2
import numpy as np


import random

import py.training.common.a_geometry as geo

from py.training.common.a_geometry import *
from a_structs import *


def overlap(x1, w1, x2, w2):
    l1 = x1 - w1 / 2
    l2 = x2 - w2 / 2
    left = max(l1, l2)

    r1 = x1 + w1 / 2
    r2 = x2 + w2 / 2
    right = min(r1, r2)
    return right - left


def boxIntersection(a: bbox, b: bbox):
    w = overlap(a.cx, a.w, b.cx, b.w)
    h = overlap(a.cy, a.h, b.cy, b.h)

    if (w < 0 or h < 0):
        return 0
    return w * h


def boxUnion(a: bbox, b: bbox):
    # returns 0?
    return a.w * a.h + b.w * b.h - boxIntersection(a, b)


def box_iou(a: bbox, b: bbox):
    return boxIntersection(a, b) / boxUnion(a, b)

def box_in_image(a: bbox, img_box: bbox) -> float:
    area = a.w * a.h
    ret = boxIntersection(a, img_box) / area
    # print(ret)
    return ret


def augment_image(thing: ImageWithBoundingBoxes):
    img = thing.image
    hands = thing.bboxes
    flip = random.random() < 0.5

    origW = npImgWidth(img)
    origH = npImgHeight(img)

    # ok I don't like this - src_tri is what's randomized and dst_tri stays constant.
    # this is probably because that _is_ what is done in the keypoint estimator and it makes a lot more sense there.
    # but: since we don't know the input image size, we have to have a reasonable starting src_tri
    # that'll be binned in the output image.
    if origW/origH > 4/3:

        y_axis_len = npImgHeight(img)/2
        x_axis_len = npImgHeight(img)*(header.model_input_width/header.model_input_height)/2
    else:
        x_axis_len = npImgWidth(img)/2
        y_axis_len = npImgWidth(img)*(header.model_input_height/header.model_input_width)/2

    rot = random.uniform(-math.pi*.1, math.pi*.1)

    x_axis = np.float32([math.cos(rot), math.sin(rot)])*x_axis_len
    y_axis = np.float32([-math.sin(rot), math.cos(rot)])*y_axis_len

    if flip:
        x_axis = -x_axis

    center = np.array((img.shape[1]/2, img.shape[0]/2))

    # Upper limit should be the amount we have to zoom out to get a perfect bin, plus some amount.

    # BAD HACK: Let's just pick that by saying it's *to get a perfect bin in a square* - that happens to be what I'm doing right now on Nov 1
    max_amt = origW/origH
    max_amt = max(max_amt, origH/origW)

    amt = random.uniform(.9, max_amt+0.1)

    x_axis *= amt
    y_axis *= amt

    # This should be at least the amount required to move the binned image towards the top, bottom, left or right edge but now it's hard coded. Ugh.
    center[0] += random.uniform(-0.04*origW, 0.04*origW)  # no!
    center[1] += random.uniform(-0.1*origH, 0.1*origH)

    tl = center - x_axis - y_axis
    tr = center + x_axis - y_axis
    bl = center - x_axis + y_axis

    tl_o = (0, 0)
    tr_o = (header.model_input_width, 0)
    bl_o = (0, header.model_input_height)

    src_tri = np.float32((tl, tr, bl))
    dst_tri = np.float32((tl_o, tr_o, bl_o))

    trans = cv2.getAffineTransform(src_tri, dst_tri)

    bbox_mul = np.linalg.norm([trans[0, 0], trans[1, 0]])

    thing.image = cv2.warpAffine(
        img, trans, (header.model_input_width, header.model_input_height), borderValue=random.randint(0, 255))

    for idx, hand in enumerate(hands):
        if (hand is None):
            continue
        hand.cx, hand.cy = transformVecBy2x3(np.array((hand.cx, hand.cy)), trans)

        hand.w = max(hand.w, hand.h)
        hand.w *= bbox_mul
        hand.h = hand.w

    if flip:

        lefthand = hands[0]
        righthand = hands[1]
        hands[0] = righthand
        hands[1] = lefthand
        # for hand in hands:
        #     hand.cx *= -1
    for idx, hand in enumerate(hands):
        if hand is None:
            continue
        img_w = npImgWidth(thing.image)
        img_h = npImgHeight(thing.image)
        # img_w = 320
        # img_h = 240
        box_val = box_in_image(hand, bbox(img_w/2, img_h/2, img_w, img_h))
        if box_val < 0.2:
            if False:
                print(f"Ok hand {idx} at {hand.cx}, {hand.cy} is probably outside of the image.", box_val);
                cv2.imshow("outside?", thing.image)
                cv2.waitKey(0)
            hands[idx] = None

    return thing


def imgwithboundingboxes320_to_heatmaps_2hand(thing: ImageWithBoundingBoxes):

    img = normalizeGrayscaleImage(thing.image)
    w = geo.npImgWidth(img)
    h = geo.npImgHeight(img)
    # (240,320) -> (1, 240, 320)
    img = img[np.newaxis, ...]

    output = {"image": img,
              "exists": np.zeros((2), dtype=np.float32),
              "center_x": np.zeros((2), dtype=np.float32),
              "center_y": np.zeros((2), dtype=np.float32),
              "size": np.zeros((2), dtype=np.float32)}

    for idx, bbox in enumerate(thing.bboxes):

        if (bbox is None):
            # redundant here but i paranoid
            output["exists"][idx] = 0
            continue
        output["exists"][idx] = 1

        bbox = thing.bboxes[idx]

        output['center_x'][idx] = geo.map_ranges(bbox.cx, 0, w, -1, 1)
        output['center_y'][idx] = geo.map_ranges(bbox.cy, 0, h, -1, 1)
        # Normalized to the width of the image
        output['size'][idx] = geo.map_ranges(bbox.w, 0, w, 0, 1)
        assert math.isfinite(output["exists"][idx])
        assert math.isfinite(output["center_x"][idx])
        assert math.isfinite(output["center_y"][idx])
        assert math.isfinite(output["size"][idx])
    return output
