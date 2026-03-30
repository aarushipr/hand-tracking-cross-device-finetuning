import numpy as np
import cv2
import matplotlib
import random
import math
import os

# IMG_PATH_COLOR=/3/inshallah4/seq3/imgs/frame0050_camera0_color.jpg IMG_PATH_ALPHA=/3/inshallah4/seq3/imgs/frame0050_camera0_alpha.jpg python3 py/circletest.py


def mat_float32touint8(mat):
  return (mat*255).astype(np.uint8)


def mat_uint8tofloat32(mat):
  return mat.astype(np.float32)*1.0/255.0


# based very loosely on https://stackoverflow.com/a/49848093
def make_radial_gradient(sz, pos_x, pos_y, radius, softness=1.0):
    # Generate (x,y) coordinate arrays
    y, x = np.mgrid[0:sz, 0:sz]
    # Calculate the weight for each pixel
    distance_squared = ((x-pos_x)**2 + (y-pos_y)**2)

    return np.clip((radius**2 - distance_squared) / (softness*radius) + 0.5, 0, 1).astype(np.float32)


def create_circle():
    image_size = 960
    radius = 480
    center_x = 480
    center_y = 480
    softness = 0.25 * radius
    a = make_radial_gradient(960, 480, 480, 480, softness)
    # print(np.mean(a))
    # print(a.dtype)
    return a


def main():
    img_path_color = os.getenv("IMG_PATH_COLOR")
    img_path_alpha = os.getenv("IMG_PATH_ALPHA")
    if ((img_path_alpha == None) or (img_path_color == None)):
        print("Set your env vars nerd!")
        exit(1)
    img_circle = create_circle()


    # It comes out 960x960, to make it line up with RGB image make it 960x960x3
    wrong = np.empty((960, 960, 3))
    for i in range(3):
      wrong[:,:,i] = img_circle

    img_color = mat_uint8tofloat32(cv2.imread(img_path_color)) * wrong
    img_alpha = mat_uint8tofloat32(cv2.imread(img_path_alpha, cv2.IMREAD_GRAYSCALE)) + (1.0-img_circle)

    img_alpha = np.clip(img_alpha, 0, 1)

    img_color = mat_float32touint8(img_color)
    img_alpha = mat_float32touint8(img_alpha)

    cv2.imwrite(img_path_color, img_color)
    cv2.imwrite(img_path_alpha, img_alpha)
    


if __name__ == "__main__":
  main()
