import numpy as np
import cv2
import matplotlib
import random
import math
import os

# IMG_PATH_COLOR=/3/inshallah4/seq3/imgs/frame0050_camera0_color.jpg IMG_PATH_ALPHA=/3/inshallah4/seq3/imgs/frame0050_camera0_alpha.jpg python3 py/add_index_vignette.py


def mat_float32touint8(mat):
    return (mat*255).astype(np.uint8)


def mat_uint8tofloat32(mat):
    return mat.astype(np.float32)*1.0/255.0


# based very loosely on https://stackoverflow.com/a/49848093
def make_radial_gradient(size_x, size_y, pos_x, pos_y, radius, softness=1.0):
    # Generate (x,y) coordinate arrays
    y, x = np.mgrid[0:size_x, 0:size_y]
    # Calculate the weight for each pixel
    distance_squared = ((x-pos_x)**2 + (y-pos_y)**2)

    return np.clip((radius**2 - distance_squared) / (softness*radius) + 0.5, 0, 1).astype(np.float32)


def create_circle(shape):
    radius = (shape[0]+shape[1])/4
    softness = 0.25 * radius
    a = make_radial_gradient(
        shape[0], shape[1], shape[1]/2, shape[0]/2, radius, softness)
    print(shape)
    # a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    # a = a[:,:,None]
    # a.exten
    # print(a.shape)
    # assert(false)
    # print(np.mean(a))
    # print(a.dtype)
    return a



def main():
    img_path_color = os.getenv("IMG_PATH_COLOR")
    img_path_alpha = os.getenv("IMG_PATH_ALPHA")
    use_alpha = os.getenv("USE_ALPHA")
    if ((img_path_alpha == None) or (img_path_color == None) or (use_alpha == None)):
        print("Set your env vars nerd!")
        exit(1)

    img_color = mat_uint8tofloat32(cv2.imread(img_path_color))

    img_circle_gray = create_circle(img_color.shape)
    img_circle_rgb = cv2.cvtColor(img_circle_gray, cv2.COLOR_GRAY2BGR)

    img_color = mat_uint8tofloat32(cv2.imread(img_path_color)) * img_circle_rgb
    img_color = mat_float32touint8(img_color)
    cv2.imwrite(img_path_color, img_color)

    if use_alpha:
        img_alpha = mat_uint8tofloat32(cv2.imread(
        img_path_alpha, cv2.IMREAD_GRAYSCALE)) + (img_circle_gray-1.0)

        img_alpha = np.clip(img_alpha, 0, 1)

        img_alpha = mat_float32touint8(img_alpha)
        cv2.imwrite(img_path_alpha, img_alpha)


if __name__ == "__main__":
    main()
