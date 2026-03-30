import numpy as np
import cv2
import matplotlib
import random
import math


def mat_float32touint8(mat):
    return (np.clip(mat, 0, 1) * 255).astype(np.uint8)


def mat_uint8tofloat32(mat):
    return mat.astype(np.float32) * 1.0 / 255.0


# based very loosely on https://stackoverflow.com/a/49848093
def make_radial_gradient(sz, pos_x, pos_y, radius, softness=1.0):
    # Generate (x,y) coordinate arrays
    y, x = np.mgrid[0:sz, 0:sz]
    # Calculate the weight for each pixel
    distance_squared = ((x - pos_x)**2 + (y - pos_y)**2)

    return np.clip((radius**2 - distance_squared) /
                   (softness * radius) + 0.5, 0, 1).astype(np.float32)


# Very useful for OpenCV drawing primitives
def cvtup(b):
    return (int(b[0]), int(b[1]))


# There really should be a library function for this...
def rotation_matrix(theta: float):
    c, s = math.cos(theta), math.sin(theta)
    R = np.array(((c, -s, 0),
                  (s, c, 0)))
    return R


def transformVecBy2x3(input, M):
    thing = input.copy()
    thing[0] = (input[0] * M[0, 0]) + (input[1] * M[0, 1]) + M[0, 2]
    thing[1] = (input[0] * M[1, 0]) + (input[1] * M[1, 1]) + M[1, 2]

    return thing


def map_ranges(value, from_low, from_high, to_low, to_high):
    return (value - from_low) * (to_high - to_low) / \
        (from_high - from_low) + to_low


# Heh heh is this the same as map_ranges? Or rather close?
def in_line(x, p1, p2):
    return ((p2[1] - p1[1]) / (p2[0] - p1[0])) * (x - p1[0]) + p1[1]


def piecewise(x, pts_list):
    for i in range(len(pts_list)):
        if (x > pts_list[i][0]) and (x < pts_list[i + 1][0]):
            return in_line(x, pts_list[i], pts_list[i + 1])
    raise RuntimeError("Input not in list of points!")


def arbitrary_density(val_and_density_list):
    curr_pt = 0
    old_curr_pt = 0

    sum = 0

    for i in range(len(val_and_density_list) - 1):
        sum += val_and_density_list[i][1]
    x = random.uniform(0, sum)

    for i in range(len(val_and_density_list) - 1):
        old_curr_pt = curr_pt
        curr_pt += val_and_density_list[i][1]
        if (x < curr_pt):
            the_val = in_line(x,
                              (old_curr_pt,
                               val_and_density_list[i][0]),
                              (curr_pt,
                                  val_and_density_list[i + 1][0]))
            return the_val

    raise RuntimeError("Input not in list of points!")


def draw_hand_rainbow_pts(img, pts):
    j = 0
    for pt in pts:
        rgb = matplotlib.colors.hsv_to_rgb([j / float(len(pts)), 1.0, 1.0])
        # for ele in rgb:
        #     ele *= 255.0
        if img.dtype == np.uint8:
            rgb *= 255.0
        cv2.circle(img, (int(pt[0]), int(pt[1])),
                   2, (rgb[2], rgb[1], rgb[0]), cv2.FILLED)
        j += 1


def draw_21_hand_lines(input_img, joints, color, width=1):
    for finger in range(5):
        for joint in range(4):
            idx = 1 + (finger * 4) + joint

            previdx = idx - 1
            if joint == 0:
                previdx = 0

            prev_gt = joints[previdx]
            curr_gt = joints[idx]

            cv2.line(input_img,
                     (int(prev_gt[0]), int(prev_gt[1])),
                     (int(curr_gt[0]), int(curr_gt[1])),
                     color, width)


def calc_num_times_to_make_image(requested_num_images, num_unique_hands):
    base_num = math.floor(requested_num_images / num_unique_hands)
    extra_probability = (requested_num_images %
                         num_unique_hands) / num_unique_hands
    am = base_num
    if (random.random() < extra_probability):
        am += 1
    return am


def npImgWidth(img):
    return img.shape[1]


def npImgHeight(img):
    return img.shape[0]


def normalizeGrayscaleImage(
        img,
        report=None,
        target_mean=0.5,
        target_std=0.25):
    if img.dtype == np.uint8:
        img = mat_uint8tofloat32(img)
    std = np.std(img)
    if std < 0.0001:
        if report is not None:
            print(f"Very low contrast: {report}")
        img = np.random.random(img.shape)
        std = np.std(img)
    img *= target_std / std
    img += target_mean - np.mean(img)

    np.clip(img, 0, 1)

    return img


def draw_rectangle_in_image_px_coord(image, top, bottom, left, right, color):
    # I wish I had any idea why the normal cv2.rectangle() crashes my machine.
    # like it legit crashes my GPU and I have to restart my computer.
    cv2.line(image,
             (int(left), int(top)),
             (int(right), int(top)),
             color, thickness=1)
    cv2.line(image,
             (int(right), int(top)),
             (int(right), int(bottom)),
             color, thickness=1)
    cv2.line(image,
             (int(right), int(bottom)),
             (int(left), int(bottom)),
             color, thickness=1)
    cv2.line(image,
             (int(left), int(bottom)),
             (int(left), int(top)),
             color, thickness=1)


def draw_square_in_image_px_coord(image, center, square_side, color):
    draw_rectangle_in_image_px_coord(
        image, center[1] - (square_side / 2),
        center[1] + (square_side / 2),
        center[0] - (square_side / 2),
        center[0] + (square_side / 2),
        color)


def draw_rectangle_in_hmap(hmap, top, bottom, left, right, val):
    width = hmap.shape[1]
    height = hmap.shape[0]

    top = int(top * height)
    bottom = int(bottom * height)

    left = int(left * width)
    right = int(right * width)

    hmap[top:bottom, left:right] = val


# 2D Miniball. May be unused but please don't remove, we end up coming back to this a lot.
def minicircle(pts_):
    pts = pts_[:, :2]
    min_x = pts[0][0]
    min_y = pts[0][1]

    max_x = pts[0][0]
    max_y = pts[0][1]

    # Find axis-aligned bounding box.
    # This is our "close enough" to the center of the smallest enclosing circle
    for pt in pts:
        min_x = min(min_x, pt[0])
        max_x = max(max_x, pt[0])

        min_y = min(min_y, pt[1])
        max_y = max(max_y, pt[1])

    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2

    c = np.array([cx, cy])

    # Find radius
    r = 0
    for pt in pts:
        r = max(r, np.linalg.norm(c - pt))

    return cx, cy, r

# OpenCV does things in BGR for some unknown, insane reason. As normal
# humans, we do things in RGB, so we have these helpers.


def rgb_imshow(name, img):
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow(name, new_img)


def rgb_imwrite(name, img):
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(name, new_img)

# This stuff is wrong! At least for 25-joint hands or 21-joint hands! No trust!
# XRT_HAND_JOINT_PALM = 0
# XRT_HAND_JOINT_WRIST = 1
# XRT_HAND_JOINT_THUMB_METACARPAL = 2
# XRT_HAND_JOINT_THUMB_PROXIMAL = 3
# XRT_HAND_JOINT_THUMB_DISTAL = 4
# XRT_HAND_JOINT_THUMB_TIP = 5
# XRT_HAND_JOINT_INDEX_METACARPAL = 6
# XRT_HAND_JOINT_INDEX_PROXIMAL = 7
# XRT_HAND_JOINT_INDEX_INTERMEDIATE = 8
# XRT_HAND_JOINT_INDEX_DISTAL = 9
# XRT_HAND_JOINT_INDEX_TIP = 10
# XRT_HAND_JOINT_MIDDLE_METACARPAL = 11
# XRT_HAND_JOINT_MIDDLE_PROXIMAL = 12
# XRT_HAND_JOINT_MIDDLE_INTERMEDIATE = 13
# XRT_HAND_JOINT_MIDDLE_DISTAL = 14
# XRT_HAND_JOINT_MIDDLE_TIP = 15
# XRT_HAND_JOINT_RING_METACARPAL = 16
# XRT_HAND_JOINT_RING_PROXIMAL = 17
# XRT_HAND_JOINT_RING_INTERMEDIATE = 18
# XRT_HAND_JOINT_RING_DISTAL = 19
# XRT_HAND_JOINT_RING_TIP = 20
# XRT_HAND_JOINT_LITTLE_METACARPAL = 21
# XRT_HAND_JOINT_LITTLE_PROXIMAL = 22
# XRT_HAND_JOINT_LITTLE_INTERMEDIATE = 23
# XRT_HAND_JOINT_LITTLE_DISTAL = 24
# XRT_HAND_JOINT_LITTLE_TIP = 25
# XRT_HAND_JOINT_MAX_ENUM = 0x7FFFFFFF


# joints_5x5_to_26 = (
#     (
#         XRT_HAND_JOINT_WRIST,
#         XRT_HAND_JOINT_THUMB_METACARPAL,
#         XRT_HAND_JOINT_THUMB_PROXIMAL,
#         XRT_HAND_JOINT_THUMB_DISTAL,
#         XRT_HAND_JOINT_THUMB_TIP,
#     ),
#     (
#         XRT_HAND_JOINT_INDEX_METACARPAL,
#         XRT_HAND_JOINT_INDEX_PROXIMAL,
#         XRT_HAND_JOINT_INDEX_INTERMEDIATE,
#         XRT_HAND_JOINT_INDEX_DISTAL,
#         XRT_HAND_JOINT_INDEX_TIP,
#     ),
#     (
#         XRT_HAND_JOINT_MIDDLE_METACARPAL,
#         XRT_HAND_JOINT_MIDDLE_PROXIMAL,
#         XRT_HAND_JOINT_MIDDLE_INTERMEDIATE,
#         XRT_HAND_JOINT_MIDDLE_DISTAL,
#         XRT_HAND_JOINT_MIDDLE_TIP,
#     ),
#     (
#         XRT_HAND_JOINT_RING_METACARPAL,
#         XRT_HAND_JOINT_RING_PROXIMAL,
#         XRT_HAND_JOINT_RING_INTERMEDIATE,
#         XRT_HAND_JOINT_RING_DISTAL,
#         XRT_HAND_JOINT_RING_TIP,
#     ),
#     (
#         XRT_HAND_JOINT_LITTLE_METACARPAL,
#         XRT_HAND_JOINT_LITTLE_PROXIMAL,
#         XRT_HAND_JOINT_LITTLE_INTERMEDIATE,
#         XRT_HAND_JOINT_LITTLE_DISTAL,
#         XRT_HAND_JOINT_LITTLE_TIP,
#     )
# )
