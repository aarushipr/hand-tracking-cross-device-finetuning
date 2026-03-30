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


def get_pos(file, frame_idx: int, elbow: bool = False):

    arr = file.iloc[frame_idx]
    root = 1
    if elbow:
        # Size of a vec3+quaternion
        root += 7

    p = mathutils.Vector((arr[root], arr[root+1], arr[root+2]))

    q = mathutils.Quaternion()
    q.w = arr[root+3]
    q.x = arr[root+4]
    q.y = arr[root+5]
    q.z = arr[root+6]

    return (p, q)
