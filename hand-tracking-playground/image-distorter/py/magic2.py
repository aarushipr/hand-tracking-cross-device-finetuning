from dataclasses import dataclass
import sys  # nopep8
import os  # nopep8
sys.path.insert(0, os.path.dirname(__file__))  # nopep8

import traceback


import readcsv_fingerpose
import readcsv_wristpose
import bpy
import mathutils
import mlib
# import guys_we_like
# import mblab
import random
import math
import csv
import pandas as pd
import numpy as np

import subprocess
import json

import header
from header import State
from header import env_settings


make_guy = True


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


# WXYZ, not XYZW.
sqrt2_2 = math.sqrt(2)/2
if False:
    camera_forward = mathutils.Quaternion((sqrt2_2, sqrt2_2, 0, 0))
    camera_left = mathutils.Quaternion((0.5, 0.5, 0.5, 0.5))
    camera_right = mathutils.Quaternion((0.5, 0.5, -0.5, -0.5))
    camera_top = mathutils.Quaternion((0, 1, 0, 0))
    camera_bottom = mathutils.Quaternion((1, 0, 0, 0))
else:
    camera_forward = mathutils.Quaternion((1, 0, 0, 0))
    camera_left = mathutils.Quaternion((sqrt2_2, 0, sqrt2_2, 0))
    camera_right = mathutils.Quaternion((sqrt2_2, 0, -sqrt2_2, 0))
    camera_top = mathutils.Quaternion((sqrt2_2, sqrt2_2, 0., 0.))
    camera_bottom = mathutils.Quaternion((sqrt2_2, -sqrt2_2, 0., 0.))


def init_random(st: State):

    bpy.context.scene.frame_end = 1000
    bpy.context.scene.render.fps = 30
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512



def dumbest_possible_add_lights():
    # XXX: No!
    world = bpy.context.scene.world
    world.node_tree.nodes["Background"].inputs[1].default_value = (
        random.uniform(0.0, 1)**2)*0.5

    world.node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)


    if False:
        num_lights = int(random.uniform(1, 6))
    else:
        num_lights = int(1 + 5*(random.random()**2))
    # center = mathutils.Vector((0, 0.1, 0.3))
    center = np.array((0, 0.1, 0.3))
    print(num_lights)
    for i in range(num_lights):
        obj = mlib.create_light(str(i))
        max_light_dist = 2

        move_amount = np.random.uniform(-max_light_dist, max_light_dist, 3)
        square_move_len = np.linalg.norm(move_amount)**2
        pos = center +square_move_len
        obj.location = center + mathutils.Vector((random.uniform(-max_light_dist, max_light_dist),
                                                  random.uniform(-max_light_dist,
                                                                 max_light_dist),
                                                  random.uniform(-max_light_dist, max_light_dist)))
        obj.data.energy = 10.0 + random.random()*70.5
        obj.data.energy /= num_lights
        obj.data.energy *= square_move_len
        print(f"ENERGY {obj.data.energy} MOVE_LEN {square_move_len}")

    # for obj in bpy.data.collections["lights"].objects:
    #     max_light_dist = 1
    #     obj.location = center + mathutils.Vector((random.uniform(-max_light_dist, max_light_dist),
    #     random.uniform(-max_light_dist, max_light_dist),
    #     random.uniform(-max_light_dist, max_light_dist)))
    #     obj.data.energy = 0.0 + random.random()*13.5


def dumb_render_settings():
    # bpy.context.scene.eevee.taa_render_samples = 16
    bpy.context.scene.eevee.taa_render_samples = 1

    bpy.context.scene.cycles.adaptive_threshold = 0.1

    # Bloom adds seams between cubemap directions. It's bad.
    bpy.context.scene.eevee.use_bloom = False

    # 1% chance of no motion blur
    if (random.random() > 0.01):
        bpy.context.scene.eevee.use_motion_blur = True

        # bias towards lower blur using square
        bpy.context.scene.eevee.motion_blur_shutter = random.uniform(
            0.0, 1.0)**2.3

        # bpy.context.scene.eevee.motion_blur_shutter = 1.0

        print("ADDING MOTION BLUR:", bpy.context.scene.eevee.motion_blur_shutter)

        bpy.context.scene.eevee.motion_blur_steps = 5
    else:
        print("NOT ADDING MOTION BLUR")

# Called by main() to remove anything the artist made as an aid when rigging/creating the hand asset


def remove_artist_lights_and_cameras():
    for b_object in bpy.data.objects:
        do_delete = False
        do_delete = do_delete or b_object.type == "LIGHT"
        do_delete = do_delete or b_object.type == "CAMERA"
        if do_delete:
            bpy.data.objects.remove(b_object)


def make_tmp_render_folders(st):
    # sigh we will just have an empty directory for no-alpha rendering
    # (Unnecessary except for paranoia) If they're symlinks, delete them non-recursively
    output_images_color_base_path = header.env_settings.output_images_color_base_path
    output_images_alpha_base_path = header.env_settings.output_images_alpha_base_path
    os.system(
        f"rm {output_images_color_base_path} {output_images_alpha_base_path}")
    # (Regular case) Delete them recursively, since they're probably folders
    os.system(
        f"rm -r {output_images_color_base_path} {output_images_alpha_base_path}")

    # Create the folders again
    os.system(
        f"mkdir -p {output_images_color_base_path} {output_images_alpha_base_path}")


def load_file_get_properties(st, root_dir, name):

    # Don't load UI - we want the artist to be able to leave whatever they want there and get the same UI as a programmer
    # Do use scripts (even though this is a security risk if we were loading unknown files) - we need them for Drivers for corrective blendshapes.

    bpy.ops.wm.open_mainfile(filepath=str(os.path.join(
        root_dir, name+".blend")), load_ui=False, use_scripts=True)

    with open(os.path.join(root_dir, name+".json")) as f:
        j = json.load(f)
        st.arm_scale = j["arm_scale"]

        try:
            st.hand_scale = j["hand_scale"]
        except KeyError:
            st.hand_scale = 1.0


def hand_joints_bone_root_position(bones, bone, left_camera_matrix_world_inverse):
    return list((left_camera_matrix_world_inverse @ bones.matrix_world @ bone.matrix).translation)


def hand_joints(st, bones):
    locations = []

    # camera_location = st.left_camera_empty.location
    left_camera_matrix_world_inverse = st.left_camera_empty.matrix_world.inverted()
    print(left_camera_matrix_world_inverse)

    locations.append(hand_joints_bone_root_position(
        bones, bones.pose.bones["Wrist"], left_camera_matrix_world_inverse))

    for finger in 'Thumb', 'Index', 'Middle', 'Ring', 'Little':
        if finger == 'Thumb':
            ls = 'Metacarpal', 'Proximal', 'Distal'
        else:
            ls = 'Proximal', 'Intermediate', 'Distal'
        for joint in ls:
            name = finger+joint
            locations.append(hand_joints_bone_root_position(
                bones, bones.pose.bones[name], left_camera_matrix_world_inverse))

        # Get tip bone
        name = finger+'Distal'
        bone = bones.pose.bones[name]
        loc = left_camera_matrix_world_inverse @ bones.matrix_world @ bone.matrix @ mathutils.Vector(
            (0, bone.length, 0))
        locations.append(loc)

    locations_openxr = []
    locations_opencv = []

    for i in range(21):
        l = locations[i]

        # The camera empties are "in" OpenXR space.
        openxr = [l[0], l[1], l[2]]
        # Regular conversion from OpenXR to OpenCV
        opencv = [l[0], -l[1], -l[2]]

        locations_openxr.append(openxr)
        locations_opencv.append(opencv)

    # print(len(locations))
    return (locations_openxr, locations_opencv)


def main():
    st = State()

    names = ["3dscanstore_black_male",
             "3dscanstore_black_female",
             "3dscanstore_white_male",
             "3dscanstore_white_female",
             "3dscanstore_asian",
            #  "mblab_light",
             "uhh"]

    load_file_get_properties(st, "/3/epics/artificial_data_3/hands/",
                             names[header.env_settings.hand_model_index])

    # load_file_get_properties(st, "/3/epics/artificial_data_3/hands/",
    #                          "uhh")
    
    # load_file_get_properties(st, "/3/epics/artificial_data_3/hands/",
    #                          names[0])
    # load_file_get_properties(st, "/3/epics/artificial_data_3/hands/", random.choice(names))
    # load_file_get_properties(st, "/3/epics/artificial_data_3/hands/", names[2])

    # bpy.ops.wm.open_mainfile(filepath="/3/epics/artificial_data_3/hands/3dscanstore_asian.blend", load_ui=False, use_scripts = True)
    # bpy.ops.wm.open_mainfile(filepath="/3/epics/artificial_data_3/hands/mblab_light.blend", load_ui=False, use_scripts = True)

    # Autodetection is annoying. Just make it right in the blend file.
    st.blender_scene = bpy.data.scenes["Scene"]


    remove_artist_lights_and_cameras()
    dumb_render_settings()

    mlib.add_ambient_occlusion(st)

    if (header.env_settings.use_exr_background):
        mlib.make_exr_background(st)
    else:
        dumbest_possible_add_lights()

    
    mlib.make_render_output(st, header.env_settings.render_alpha)
    make_tmp_render_folders(st)

    st.file_wristpose = readcsv_wristpose.get_file()

    wrist_empty = bpy.data.objects["xr_wrist_target"]
    wrist_empty.rotation_mode = "QUATERNION"

    tip_correct_rot = mathutils.Quaternion((-0.707107, 0.707107, 0, 0))

    # might be equivalent to above
    wrist_correct_prerot = mathutils.Quaternion((0.707107, 0.707107, 0, 0))

    for i in range(header.env_settings.num_frames):
        framerate_divisor = \
            header.env_settings.wristpose_framerate / header.env_settings.out_framerate
        p, q = readcsv_wristpose.get_pos(
            st.file_wristpose, env_settings.wristpose_start_idx + int(i*framerate_divisor))
        wrist_empty.location.x = p.x
        wrist_empty.location.y = -p.z
        wrist_empty.location.z = p.y

        #!@todo make this configurable by-model!
        wrist_empty.location *= st.arm_scale

        q.rotate(wrist_correct_prerot)
        # wrist_empty.location = p
        # q.
        wrist_empty.rotation_quaternion = q

        # wrist_empty.location.z = start_z + 0.02*math.sin(st.frame*0.1)
        # wrist_empty.location.x = start_x + 0.03*math.cos(st.frame*0.1)
        wrist_empty.keyframe_insert(data_path="location", frame=st.frame)
        wrist_empty.keyframe_insert(
            data_path="rotation_quaternion", frame=st.frame)
        st.frame += 1

    for i in range(26):
        if i == XRT_HAND_JOINT_THUMB_TIP:
            e = bpy.data.objects["thumb_tip_target"]
            e.rotation_mode = 'QUATERNION'
        elif i == XRT_HAND_JOINT_INDEX_TIP:
            e = bpy.data.objects["index_tip_target"]
            e.rotation_mode = 'QUATERNION'
        elif i == XRT_HAND_JOINT_MIDDLE_TIP:
            e = bpy.data.objects["middle_tip_target"]
            e.rotation_mode = 'QUATERNION'
        elif i == XRT_HAND_JOINT_RING_TIP:
            e = bpy.data.objects["ring_tip_target"]
            e.rotation_mode = 'QUATERNION'
        elif i == XRT_HAND_JOINT_LITTLE_TIP:
            e = bpy.data.objects["little_tip_target"]
            e.rotation_mode = 'QUATERNION'
        else:
            e = mlib.create_empty()

        e.empty_display_size = .01
        # e.empty_display_type = 'ARROWS'
        e.parent = wrist_empty
        st.empties.append(e)

    readcsv_fingerpose.get_file(st)

    for i in range(header.env_settings.num_frames):
        for j in range(26):
            s = readcsv_fingerpose.readcsv_fingerpose_settings(st.file, 1.1*st.hand_scale)
            # s = readcsv_fingerpose.readcsv_fingerpose_settings(st.file, 1.02)
            framerate_divisor = \
                header.env_settings.fingerpose_framerate / header.env_settings.out_framerate
            p, q = readcsv_fingerpose.get_joint(
                st, s, env_settings.fingerpose_start_idx + int(i*(framerate_divisor)), j)

            # q = q.rotate()
            newguy = tip_correct_rot.copy()
            newguy.rotate(q)
            q = newguy
            # q = tip_correct_rot.rotate(q)
            e = st.empties[j]
            e.rotation_quaternion = q

            e.location = p
            e.keyframe_insert(data_path="location", frame=i)
            e.keyframe_insert(data_path="rotation_quaternion", frame=i)





    # print(st.right_in_left_pos, st.right_in_left_rot)

    init_random(st)

    mlib.stereoscopy()
    mlib.make_cameras(st)

    # MBLab sets this for some reason
    for scene in bpy.data.scenes:
        print("scene is", scene)
        scene.render.engine = 'BLENDER_EEVEE'
        # scene.render.engine = 'CYCLES'
    # mlib.make_background_voronoi()

    # for i in range(20):

    # If you even PROPOSE changing the name for the armature object I will do unspeakable things
    bones_object = bpy.data.objects["doot doot"]

    # Artist might have set it to rest position, let's change it back in case.
    bones_object.data.pose_position = 'POSE'

    render = True

    if (not render):
        return
    num_frames_to_render = header.env_settings.num_frames

    #
    arr_openxr = np.zeros((num_frames_to_render, 21*3))
    arr_opencv = np.zeros((num_frames_to_render, 21*3))

    for i in range(num_frames_to_render):
        bpy.context.scene.frame_current = i

        bpy.context.view_layer.update()

        hand_joints_array_openxr, hand_joints_array_opencv = hand_joints(
            st, bones_object)
        arr_slice_openxr = np.asarray(hand_joints_array_openxr)
        arr_slice_openxr = arr_slice_openxr.flatten()
        arr_openxr[i] = arr_slice_openxr

        arr_slice_opencv = np.asarray(hand_joints_array_opencv)
        arr_slice_opencv = arr_slice_opencv.flatten()
        arr_opencv[i] = arr_slice_opencv

    df_openxr = pd.DataFrame(arr_openxr)
    df_opencv = pd.DataFrame(arr_opencv)
    # print(df)
    df_openxr.to_csv(header.env_settings.output_csv_path_openxr,
                     encoding='utf-8', index=False)
    df_opencv.to_csv(header.env_settings.output_csv_path_opencv,
                     encoding='utf-8', index=False)
    #

    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = num_frames_to_render-1

    if not header.env_settings.dont_render:
        bpy.ops.render.render(animation=True, write_still=True)

    print("hey kids")

    # if (header.env_settings.exit_immediately):
    #     print("Exiting because EXIT_IMMEDIATELY is set")
    #     exit(0)
    # else:

    return


if __name__ == "__main__":
    try:
        main()
    finally:
        print(traceback.format_exc())

        # Make it so that if you instinctively hit ctrl+S it doesn't overwrite anything important

    # or
        # print(sys.exc_info())
        if header.env_settings.dont_exit_immediately:
            print("Not exiting because DONT_EXIT_IMMEDIATELY is set")

            bpy.ops.wm.save_as_mainfile(filepath="/tmp/tmp.blend")
            bpy.ops.wm.open_mainfile(filepath="/tmp/tmp.blend")
        else:
            print("Exiting because DONT_EXIT_IMMEDIATELY is not set")
            exit(0)
