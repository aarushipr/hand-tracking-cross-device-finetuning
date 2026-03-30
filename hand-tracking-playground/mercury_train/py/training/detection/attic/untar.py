import cv2
import numpy as np


from typing import Union, List
from pathlib import Path
import re
import random
import epic_kitchens.hoa
import os
import json


def math_map_ranges(value: float, from_low: float, from_high: float, to_low: float, to_high: float):
    return (value - from_low) * (to_high - to_low) / (from_high - from_low) + to_low


guys = ["P03", "P04", "P05", "P06", "P07"]

guys = sorted(os.listdir("/home/moses/EPIC-KITCHENS/"))
guys = ["P22"]

for guy in guys:
    root = f"/home/moses/EPIC-KITCHENS/{guy}/rgb_frames"

    files = sorted(os.listdir(root))

    files = [file for file in files if file.endswith("tar")]

    for file in files:
        name = file[:-4]
        print(name)
        os.chdir(root)
        os.system(f"mkdir -p {os.path.join(root, name)}")
        os.chdir(os.path.join(root, name))
        print(os.getcwd())
        os.system(f"tar -xf ../{name+'.tar'}")
