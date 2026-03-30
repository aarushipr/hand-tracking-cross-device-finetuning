# The below 4 lines NEED to go first.
import sys  # nopep8
import os  # nopep8
sys.path.insert(0, os.path.dirname(__file__))  # nopep8
sys.path.append('/home/moses/.local/lib/python3.10/site-packages')  # nopep8

from dataclasses import dataclass  # nopep8
import enum  # nopep8
import pandas as pd  # nopep8
import mathutils  # nopep8
import header

XRT_HAND_JOINT_PALM = 0
XRT_HAND_JOINT_WRIST = 1
XRT_HAND_JOINT_THUMB_METACARPAL = 2
XRT_HAND_JOINT_THUMB_PROXIMAL = 3
XRT_HAND_JOINT_THUMB_DISTAL = 4
XRT_HAND_JOINT_THUMB_TIP = 5
XRT_HAND_JOINT_INDEX_METACARPAL = 6
XRT_HAND_JOINT_INDEX_PROXIMAL = 7
XRT_HAND_JOINT_INDEX_INTERMEDIATE = 8
XRT_HAND_JOINT_INDEX_DISTAL = 9
XRT_HAND_JOINT_INDEX_TIP = 10
XRT_HAND_JOINT_MIDDLE_METACARPAL = 11
XRT_HAND_JOINT_MIDDLE_PROXIMAL = 12
XRT_HAND_JOINT_MIDDLE_INTERMEDIATE = 13
XRT_HAND_JOINT_MIDDLE_DISTAL = 14
XRT_HAND_JOINT_MIDDLE_TIP = 15
XRT_HAND_JOINT_RING_METACARPAL = 16
XRT_HAND_JOINT_RING_PROXIMAL = 17
XRT_HAND_JOINT_RING_INTERMEDIATE = 18
XRT_HAND_JOINT_RING_DISTAL = 19
XRT_HAND_JOINT_RING_TIP = 20
XRT_HAND_JOINT_LITTLE_METACARPAL = 21
XRT_HAND_JOINT_LITTLE_PROXIMAL = 22
XRT_HAND_JOINT_LITTLE_INTERMEDIATE = 23
XRT_HAND_JOINT_LITTLE_DISTAL = 24
XRT_HAND_JOINT_LITTLE_TIP = 25
XRT_HAND_JOINT_MAX_ENUM = 0x7FFFFFFF


def get_file(st):
    fn = header.env_settings.fingerpose_csv_path
    c = pd.read_csv(fn)
    st.file = c
    st.finger_filename = fn
    return c


@dataclass
class readcsv_fingerpose_settings:
    file: pd.core.frame.DataFrame
    scale: float


def get_tip(file, frame_idx: int, finger_idx: int):
    joint_idx = 0
    if finger_idx == 0:
        joint_idx = XRT_HAND_JOINT_THUMB_TIP
    elif finger_idx == 1:
        joint_idx = XRT_HAND_JOINT_INDEX_TIP
    elif finger_idx == 2:
        joint_idx = XRT_HAND_JOINT_MIDDLE_TIP
    elif finger_idx == 3:
        joint_idx = XRT_HAND_JOINT_RING_TIP
    elif finger_idx == 4:
        joint_idx = XRT_HAND_JOINT_LITTLE_TIP
    else:
        assert "what"
    arr = file.iloc[frame_idx]
    root = 1 + (joint_idx * 7)
    # X is unchanged, Y is -Z, Z is Y
    p = mathutils.Vector((arr[root], arr[root+1], arr[root+2]))
    # p = mathutils.Vector((arr[root], -arr[root+2], arr[root+1]))

    q = mathutils.Quaternion()
    q.w = arr[root+3]
    q.x = arr[root+4]
    q.y = arr[root+5]
    q.z = arr[root+6]

    return (p, q)


def get_joint(st, s: readcsv_fingerpose_settings, frame_idx, joint_idx):
    try:
        arr = s.file.iloc[frame_idx]
    except IndexError:
        print("reeee!!!!")
        print(st.finger_filename)
    root = 1 + (joint_idx * 7)
    # X is unchanged, Y is -Z, Z is Y
    p = mathutils.Vector(
        (s.scale*arr[root], s.scale*arr[root+1], s.scale*arr[root+2]))
    # p = mathutils.Vector((arr[root], -arr[root+2], arr[root+1]))

    q = mathutils.Quaternion()
    q.w = arr[root+3]
    q.x = arr[root+4]
    q.y = arr[root+5]
    q.z = arr[root+6]

    return (p, q)


def main():
    c = get_file()
    # print(c)
    # print(list(c.iloc[2]))
    print(get_tip(c, 5, 1))
    print(get_tip(c, 6, 1))
    print(get_tip(c, 7, 1))


if __name__ == "__main__":
    main()
