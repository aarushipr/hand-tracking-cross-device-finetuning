# The below 4 lines NEED to go first.
import sys  # nopep8
import os  # nopep8
sys.path.insert(0, os.path.dirname(__file__))  # nopep8
sys.path.append('/home/moses/.local/lib/python3.10/site-packages')  # nopep8

from dataclasses import dataclass  # nopep8
import enum  # nopep8
import pandas as pd  # nopep8
import bpy #nopep8
import mathutils  # nopep8

make_guy = True
num_frames = 700


class State:
    blender_scene = None
    collection = None
    file = None
    finger_filename = None
    file_wristpose = None
    empties = []

    # Sigh, it was really easy to make this be the center of a stereo camera. Shakes fist at sky
    camera_center_empty = None
    # Center of output joints space
    left_camera_empty = None


    camera = None

    left_camera_pos: mathutils.Vector = mathutils.Vector((0, 0, 0))

    right_in_left_pos: mathutils.Vector = mathutils.Vector()
    right_in_left_rot: mathutils.Quaternion = mathutils.Quaternion()

    left_in_center_pos: mathutils.Vector = mathutils.Vector()
    left_in_center_rot: mathutils.Quaternion = mathutils.Quaternion()

    arm_scale: float
    hand_scale: float

    # output_images_color_base_path = "/3/inshallah3/render_color"
    # output_images_alpha_base_path = "/3/inshallah3/render_alpha"

    # file_output_color = None
    # file_output_alpha = None



    def __init__(self):
        self.collection = bpy.data.collections['Collection']
        self.frame = 0


@dataclass
class EnvSettings():

  camera_right_in_left: str = "{\"pos\": [0.121850, 0.001008, 0.009092], \"rot\": [-0.010584, -0.080179, -0.006399, 0.996704]}"
  camera_left_in_center: str = "{\"pos\": [-0.030114, 0.000210, 0.000023], \"rot\": [0.001447, -0.000732, 0.000750, 0.999998]}"

  wristpose_csv_path: str =  "/3/whatever/three.csv"
  wristpose_framerate: float = 120
  wristpose_start_idx: int = 0

  fingerpose_csv_path: str = "/3/whatever/left_smoothed.csv"
  fingerpose_framerate: float = 54
  fingerpose_start_idx: int = 0

  out_framerate: float = 30

  num_frames: float = 20 #!< Number of frames to generate, at 30fps.

  use_exr_background: bool = True
  render_alpha: bool = False

  output_csv_path_openxr: str = "/3/inshallah3/out.csv"
  output_csv_path_opencv: str = "/3/inshallah3/out.csv"
  output_images_color_base_path: str = "/3/inshallah3/render_color_does"
  output_images_alpha_base_path: str = "/3/inshallah3/render_alpha_does"

  hand_model_index: int = 0

  dont_exit_immediately: bool = False
  dont_render: bool = False

env_settings = EnvSettings()

def try_get_env_setting(name, setting):
  thing = os.getenv(name)
  if thing == None:
    print("Environment variable {} not set. Using default value {}".format(name, setting))
    return setting
  else:
    print("Environment variable {} set to {}".format(name, thing))
    return thing

env_settings.camera_right_in_left = try_get_env_setting("CAMERA_RIGHT_IN_LEFT", env_settings.camera_right_in_left)
env_settings.camera_left_in_center = try_get_env_setting("CAMERA_LEFT_IN_CENTER", env_settings.camera_left_in_center)

env_settings.wristpose_csv_path = try_get_env_setting("WRISTPOSE_CSV", env_settings.wristpose_csv_path)
env_settings.wristpose_start_idx = int(try_get_env_setting("WRISTPOSE_START_IDX", env_settings.wristpose_start_idx))

env_settings.fingerpose_csv_path = try_get_env_setting("FINGERPOSE_CSV", env_settings.fingerpose_csv_path)
env_settings.fingerpose_start_idx = int(try_get_env_setting("FINGERPOSE_START_IDX", env_settings.fingerpose_start_idx))

env_settings.use_exr_background = bool(int(try_get_env_setting("USE_EXR_BACKGROUND", env_settings.use_exr_background)))
env_settings.render_alpha = bool(int(try_get_env_setting("RENDER_ALPHA", env_settings.render_alpha)))

env_settings.num_frames = int(try_get_env_setting("NUM_FRAMES", env_settings.num_frames))

env_settings.output_csv_path_openxr = try_get_env_setting("OUTPUT_CSV_OPENXR", env_settings.output_csv_path_openxr)
env_settings.output_csv_path_opencv = try_get_env_setting("OUTPUT_CSV_OPENCV", env_settings.output_csv_path_opencv)
env_settings.output_images_color_base_path = try_get_env_setting("OUTPUT_COLOR_BASE", env_settings.output_images_color_base_path)
env_settings.output_images_alpha_base_path = try_get_env_setting("OUTPUT_ALPHA_BASE", env_settings.output_images_alpha_base_path)

env_settings.hand_model_index = int(try_get_env_setting("HAND_MODEL_INDEX", env_settings.hand_model_index))


# Please only set DONT_EXIT_IMMEDIATELY to "0" or "1".
env_settings.dont_exit_immediately = bool(int(try_get_env_setting("DONT_EXIT_IMMEDIATELY", env_settings.dont_exit_immediately)))
env_settings.dont_render = bool(int(try_get_env_setting("DONT_RENDER", env_settings.dont_render)))



# We should do:
# But, later, or maybe not at all.
# state = State()