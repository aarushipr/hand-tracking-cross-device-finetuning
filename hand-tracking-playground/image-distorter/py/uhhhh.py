import numpy as np
import cv2
import random
import os
import colour


def mat_float32touint8(mat):
    return (mat*255).astype(np.uint8)


def mat_uint8tofloat32(mat):
    return mat.astype(np.float32)*1.0/255.0


datasets_basepath = "/3/epics/gk3/"


def main():
    img_path_color = os.getenv("IMG_PATH_COLOR")
    img_path_alpha = os.getenv("IMG_PATH_ALPHA")
    img_path_out   = os.getenv("IMG_PATH_OUT")

    transformedHand = mat_uint8tofloat32(
        cv2.imread(img_path_color))  # load image!
    transformedHand = colour.models.eotf_sRGB(
            transformedHand).astype(np.float32)

    transformedMask = mat_uint8tofloat32(
        cv2.imread(img_path_alpha))  # load mask!

    if False:
        # Use this if you'd like a specific image
        # background_path = "/3/epics/gk3//indoor/indoorCVPR_09/Images/subway/metropolitana_106_01_flickr.jpg"
        background_path = "/3/epics/gk3//indoor/indoorCVPR_09/Images/airport_inside/airport_inside_0164.jpg"
        background_j = cv2.imread(background_path)
        background_j = cv2.resize(background_j, (960, 960))
    else:
        # Picks a random image :)
        with open(f"{datasets_basepath}/indoor/BothImages.txt") as f:
            backgrounds_list = [(f"{datasets_basepath}/indoor/indoorCVPR_09/Images/"+ele).replace("\n", "")
                                for ele in f.readlines()]

        
        background_j = None
        while background_j is None:
            b_name = random.choice(backgrounds_list)
            background_j = cv2.imread(b_name)
            print(b_name)
        background_j = cv2.resize(background_j, (transformedHand.shape[1], transformedHand.shape[0]))

    background_float = mat_uint8tofloat32(background_j)


    output_size = 960

    # background_float = np.zeros((960,960,3))
    # background_float *= 0.3





    # transformedMask = cv2.warpAffine(mask, trans, (sz, sz))

    # note!!! this is premultiplied alpha so we don't need `* (1-transformedMask)`
    transformedHand = transformedHand + (background_float * (transformedMask))

    transformedHand = mat_float32touint8(transformedHand)

    print(f"writing out to {img_path_out}")

    cv2.imwrite(img_path_out, transformedHand)


if __name__ == "__main__":
        main()
