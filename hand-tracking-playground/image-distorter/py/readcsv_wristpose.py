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


def get_file():
    fn = header.env_settings.wristpose_csv_path
    print(fn)
    c = pd.read_csv(fn)
    return c


def get_pos(file, frame_idx: int):

    arr = file.iloc[frame_idx]
    root = 1
    # X is unchanged, Y is -Z, Z is Y
    p = mathutils.Vector((arr[root], arr[root+1], arr[root+2]))
    p *= 0.68
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
    print(get_pos(c, 5))
    print(get_pos(c, 6))
    print(get_pos(c, 7))


if __name__ == "__main__":
    main()
