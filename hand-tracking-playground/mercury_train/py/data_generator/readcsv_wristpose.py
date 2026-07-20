# The below 4 lines NEED to go first.
import sys  # nopep8
import os  # nopep8
sys.path.insert(0, os.path.dirname(__file__))  # nopep8
import site  # nopep8
# See header.py for why this is computed rather than hardcoded.
sys.path.append(site.getusersitepackages())  # nopep8

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

    # pandas >=2.0 dropped positional fallback for integer keys on a
    # string-indexed Series (KeyError instead) -- must use .iloc.
    p = mathutils.Vector((arr.iloc[root], arr.iloc[root + 1], arr.iloc[root + 2]))

    q = mathutils.Quaternion()
    q.w = arr.iloc[root + 3]
    q.x = arr.iloc[root + 4]
    q.y = arr.iloc[root + 5]
    q.z = arr.iloc[root + 6]

    return (p, q)
