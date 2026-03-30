# The below 4 lines NEED to go first.
import sys  # nopep8
import os
from typing import Any  # nopep8
sys.path.insert(0, os.path.dirname(__file__))  # nopep8
sys.path.append('/home/moses/.local/lib/python3.10/site-packages')  # nopep8

from dataclasses import dataclass  # nopep8
import bpy  # nopep8
import mathutils  # nopep8


make_guy = True


class State:
    blender_scene = None
    collection = None
    finger_filename = None
    file_wristpose = None
    empties = []
    objects_to_delete = []
    bone_constraints_to_delete = []

    # Sigh, it was really easy to make this be the center of a stereo camera.
    # Shakes fist at sky
    camera_center_empty = None
    # Center of output joints space
    left_camera_empty = None

    camera = None

    left_camera_pos: mathutils.Vector = mathutils.Vector((0, 0, 0))

    right_in_left_pos: mathutils.Vector = mathutils.Vector()
    right_in_left_rot: mathutils.Quaternion = mathutils.Quaternion()

    left_in_center_pos: mathutils.Vector = mathutils.Vector()
    left_in_center_rot: mathutils.Quaternion = mathutils.Quaternion()

    orig_elbow_pos: mathutils.Vector

    arm_scale: float

    json_response: Any  # json.
    num_frames: int

    # We keep a reference to it around so that we can delete it between runs and not leak it
    # Maybe we can just clear orphan data?
    hdri_background: Any

    bones_object: Any
    proportions_json: str

    model_idx: int
    slot_idx: int
    server_port: int
    model_file: str

    hand_size: float

    def __init__(self):
        self.collection = bpy.data.collections['Collection']
        self.frame = 0
