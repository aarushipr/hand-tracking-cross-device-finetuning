import math
import wandb
import numpy as np
import a_geometry as geo
import cv2
import kpest_header as header
from dataclasses import dataclass

import heatmap_1d


@dataclass
class model_output:
    xy_heatmaps: np.ndarray
    depth_heatmaps: np.ndarray
    elbow_dir: np.ndarray
    curls: np.ndarray
    is_hand: float


def find_center_of_distribution(data):
    idx = np.unravel_index(np.argmax(data), data.shape)
    coarse_x = idx[1]
    coarse_y = idx[0]

    w = data.shape[1]
    h = data.shape[0]

    max_kern_width = 3

    # can coarse_x and coarse_y be negative?
    # if not, can we remove the abs?
    kern_width_x = max(0, min(coarse_x, min(
        max_kern_width, abs(coarse_x - w)-1)))
    kern_width_y = max(0, min(coarse_y, min(
        max_kern_width, abs(coarse_y - h)-1)))

    min_x = coarse_x - kern_width_x
    max_x = coarse_x + kern_width_x

    min_y = coarse_y - kern_width_y
    max_y = coarse_y + kern_width_y

    sum_of_values = 0
    sum_of_values_times_locations_x = 0
    sum_of_values_times_locations_y = 0

    for y in range(min_y, max_y+1):
        for x in range(min_x, max_x+1):
            val = data[y][x]
            sum_of_values += val
            sum_of_values_times_locations_y += val * (y + 0.5)
            sum_of_values_times_locations_x += val * (x + 0.5)

    if (sum_of_values == 0):
        print("Ugh, what?")
        return coarse_x, coarse_y

    out_refined_x = sum_of_values_times_locations_x / sum_of_values
    out_refined_y = sum_of_values_times_locations_y / sum_of_values

    return out_refined_x, out_refined_y


def make_little_heatmap(row_idx, col_idx, big_image, heatmap):
    col = 2+(col_idx)*25
    row = 2+(row_idx)*25

    big_image[row:row+22, col:col+22] = heatmap


def make_little_depth_heatmap(row_idx, col_idx, big_image, heatmap):

    col = 2+(col_idx)*25
    row = 2+(row_idx)*25

    one_heatmap = np.ones((22), dtype=np.float32)

    big_image[row:row+22, col:col +
              22] = heatmap_1d.two_heatmaps_to_2d(one_heatmap, heatmap)


def make_visualization_column(input_img_rgb_, mo: model_output, color):

    heatmap_xy = mo.xy_heatmaps
    heatmap_depth = mo.depth_heatmaps
    elbow = mo.elbow_dir
    curls = mo.curls
    # , heatmap_depth, elbow, curls, color

    canvas = np.ones((512, 128, 3), dtype=np.float32)

    input_img_rgb = input_img_rgb_.copy()

    num = 21
    jts = np.zeros((num, 2))

    for i in range(21):
        x, y = find_center_of_distribution(heatmap_xy[i])
        x *= (128/22)
        y *= (128/22)

        jts[i][0] = x
        jts[i][1] = y

    geo.draw_21_hand_lines(input_img_rgb, jts, color)

    geo.draw_hand_rainbow_pts(input_img_rgb, jts)

    # Elbow visualization stuff!

    oth = np.zeros((2))
    oth += elbow[:2]
    oth *= 20
    oth += jts[0]

    cv2.line(input_img_rgb, (int(oth[0]), int(
        oth[1])), (int(jts[0][0]), int(jts[0][1])), color)

    #

    canvas[0:128, 0:128] = input_img_rgb

    lil_xy = np.ones((128, 128), np.float32)
    lil_depth = np.ones((128, 128), np.float32)
    # lil_xy = canvas[128:256]
    # lil_depth = canvas[256:384]
    acc_idx: int = 0

    make_little_heatmap(0, 0, lil_xy, heatmap_xy[0])
    make_little_depth_heatmap(0, 0, lil_depth, heatmap_depth[acc_idx])

    acc_idx += 1

    for finger in range(5):
        for joint in range(4):
            make_little_heatmap(finger, joint+1, lil_xy, heatmap_xy[acc_idx])
            make_little_depth_heatmap(
                finger, joint+1, lil_depth, heatmap_depth[acc_idx])
            acc_idx += 1

    canvas[128:256] = cv2.cvtColor(lil_xy, cv2.COLOR_GRAY2BGR)
    canvas[256:384] = cv2.cvtColor(lil_depth, cv2.COLOR_GRAY2BGR)

    cv2.putText(canvas, "{:.2f}".format(mo.is_hand), (2, 128+60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0))

    for i in range(5):

        r = 15

        center = (r+(20*i), 384+60)

        cv2.circle(canvas, center, 1, (0,0,0), 1)
        # curls[i] = 0
        curls[i] *= 0.3 # (-270, 10) -> (-70, 2)
        x = int(math.cos(curls[i])*r)
        y = int(-math.sin(curls[i])*r) # negated becayse +y is down

        pt2 = (center[0]+x, center[1]+y)

        cv2.circle(canvas, pt2, 1, (0,0,0), 1)

        cv2.line(canvas, center, pt2, (0,0 ,0), 1)

    return canvas


def display_and_log_output(name: str, input_image: np.ndarray, model: model_output, ground_truth: model_output, predicted_keypoints = None ):
    canvas = np.ones((512, 384, 3), dtype=np.float32)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)

    input_image_with_maybe_predicted = input_image.copy()

    if predicted_keypoints is not None:
        predicted_keypoints_img = np.zeros((21, 2))
        for idx, kp in enumerate(predicted_keypoints):
            predicted_keypoints_img[idx] = geo.map_ranges(kp, -1, 1, 0, 128)


            cv2.circle(input_image_with_maybe_predicted, (int(predicted_keypoints_img[idx][0]), int(predicted_keypoints_img[idx][1])), 2, (255, 0, 255))

        geo.draw_21_hand_lines(input_image_with_maybe_predicted, predicted_keypoints_img, (255, 0, 255))



    canvas[:, 0:128] = make_visualization_column(input_image_with_maybe_predicted, ground_truth, (0, 255, 0))
    canvas[:, 128:256] = make_visualization_column(input_image, model, (255, 0, 0))
    canvas[0:128, 256:384] = input_image

    if header.env_settings.gui_enabled:
        geo.rgb_imshow(f"{name}", canvas)

        key = cv2.waitKey(1)
        # Exiting cleanly is important; sometimes you can accidentally ctrl+c when saving the state dict
        if (key == ord('q')):
            exit(0)

    images = [wandb.Image(geo.mat_float32touint8(canvas))]
    wandb.log({name: images})


