import ad4_stereographic_projection
import numpy as np
import pandas as pd
import os
import cv2
import a_geometry
# example.looky("hahaha", "file")


superroot = "/3/inshallah10"

# seqname = "seq9"
seqname = "seq0"

camera_poses = pd.read_csv(os.path.join(superroot, seqname, "camera_info.csv"))
hand_poses = pd.read_csv(os.path.join(superroot, seqname, "hand_poses.csv"))

def pad_int(num):
    return str(num).zfill(4)

def thing(idx):
    # camera_pose = np.array(camera_poses.iloc[idx], dtype=np.float32)

    # hand_pose = np.array(camera_poses.iloc[idx], dtype=np.float32)
    # hand_pose_1 = np.array(camera_poses.iloc[idx-1], dtype=np.float32)
    # hand_pose_2 = np.array(camera_poses.iloc[idx-2], dtype=np.float32)

    # hand_pose_sequence = np.array((hand_pose, hand_pose_1, hand_pose_2))
    # a2_5 = np.zeros((21, 3), dtype=np.float32)
    out_joints_gt = np.zeros((25, 3), dtype=np.float32)
    out_joints_pose_predicted = np.zeros((25, 3), dtype=np.float32)
    out_image = np.zeros((128, 128), dtype=np.uint8)
    out_mask = np.zeros((128, 128), dtype=np.uint8)
    out_elbow = np.zeros((3), dtype=np.float32)
    out_curls = np.zeros((5), dtype=np.float32)

    print(hand_poses.shape)
    numstr = pad_int(idx)
    img_color_path = f"/3/inshallah10/{seqname}/imgs_color/Image{numstr}.jpg"
    img_alpha_path = f"/3/inshallah10/{seqname}/imgs_alpha/Image{numstr}.jpg"

    alpha: bool = os.path.exists(img_alpha_path)
    if not alpha:
      img_alpha_path = ""

    # img_alpha_path = ""  # "/3/inshallah10/seq0/imgs_alpha/Image0000.jpg"
    print(out_joints_gt)
    ad4_stereographic_projection.prepare_sample(img_color_path,
                                                img_alpha_path,
                                                hand_poses,
                                                camera_poses,
                                                idx,
                                                out_joints_gt,
                                                out_joints_pose_predicted,
                                                out_image,
                                                out_mask,
                                                out_elbow,
                                                out_curls)
    

    print(out_curls)
    out_image = cv2.cvtColor(out_image, cv2.COLOR_GRAY2BGR)
    a_geometry.draw_hand_rainbow_pts(out_image, out_joints_gt)
    
    cv2.imshow("a", out_image)
    if (alpha):
      cv2.imshow("b", out_mask)
    cv2.waitKey(0)

    print(out_joints_gt)

print(camera_poses.shape)
print(hand_poses.shape)
# for i in range(200):
#     thing(i)
