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


# XXX: No
# num_frames = 700
make_guy = True


# Type can be "IK",
# Used by something to

# def create_empty(name = "empty"):
#   o = bpy.data.objects.new( name, None )

#   # due to the new mechanism of "collection"
#   bpy.context.scene.collection.objects.link( o )

#   # empty_draw was replaced by empty_display
#   o.empty_display_size = .2
#   o.empty_display_type = 'PLAIN_AXES'
#   return o


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

    st.camera_root_empty = mlib.create_empty("camera_root_empty")

    # if False:
    #     # This is as close to where the index camera really would be
    #     # Half of Index camera baseline
    #     st.camera_root_empty.location.x = -0.067489996552467346
    #     # Came up with this on a whim
    #     st.camera_root_empty.location.y = 0.069146
    #     # Ditto
    #     st.camera_root_empty.location.z = -0.025
    # else:
    #     # This is moved back and up a little bit to get out of the way of the IK arm
    #     # Half of Index camera baseline
    #     st.camera_root_empty.location.x = -0.067489996552467346
    #     # Came up with this on a whim
    #     st.camera_root_empty.location.y = 0.061095
    #     # Ditto
    #     st.camera_root_empty.location.z = -0.011319

    # # 90-degree rotation so that -z points backwards by default.
    # # Eventually, do something smarter (or add aNOTHER thing to the hierarchy)
    # # so that the canting is balanced. For now this is fine though
    # st.camera_root_empty.rotation_quaternion = (0.707, 0.707, 0, 0)

    # st.camera_root_rel_empty = mlib.create_empty("camera_root_rel_empty")
    # st.camera_root_rel_empty.parent = st.camera_root_empty

    # camera = mlib.create_camera("camera_forward")
    # bpy.context.scene.camera = camera
    # st.camera = camera
    # st.camera.parent = st.camera_root_rel_empty

    # # point it up
    # # camera.rotation_quaternion = (0.707, 0.707, 0, 0)

    # camera.data.lens_unit = 'FOV'
    # camera.data.angle = math.pi/2
    # # 1mm
    # camera.data.clip_start = 0.001
    # # 5 meters. Overkill but fine
    # camera.data.clip_end = 5


def setup_arm_ik(doot, hand_target):
    ik = mlib.new_constraint(doot, "hand_L", "IK")

    ik.target = hand_target
    ik.chain_count = 3
    ik.use_tail = False

    copyrot = mlib.new_constraint(doot, "hand_L", "COPY_ROTATION")
    copyrot.target = hand_target

    # setup stiffness and constraints for shoulder
    clavicle = doot.pose.bones["clavicle_L"]
    clavicle.ik_stiffness_y = 0.6

    clavicle.ik_stiffness_x = 0.3
    clavicle.ik_stiffness_z = 0.3

    clavicle.use_ik_limit_x = True
    clavicle.use_ik_limit_y = True
    clavicle.use_ik_limit_z = True

    # Twist-ish
    clavicle.ik_min_y = math.radians(-10)
    clavicle.ik_max_y = math.radians(10)

    clavicle.ik_min_x = math.radians(-50)
    clavicle.ik_max_x = math.radians(50)

    clavicle.ik_min_z = math.radians(-50)
    clavicle.ik_max_z = math.radians(50)


def dumbest_possible_add_lights():
    # XXX: No!
    world = bpy.data.worlds["World.001"]
    world.node_tree.nodes["Background"].inputs[1].default_value = random.uniform(
        0.0, 0.5)

    num_lights = int(random.uniform(1, 6))
    center = mathutils.Vector((0, 0.1, 0.3))
    print(num_lights)
    for i in range(num_lights):
        obj = mlib.create_light(str(i))
        max_light_dist = 1
        obj.location = center + mathutils.Vector((random.uniform(-max_light_dist, max_light_dist),
                                                  random.uniform(-max_light_dist,
                                                                 max_light_dist),
                                                  random.uniform(-max_light_dist, max_light_dist)))
        obj.data.energy = 7.0 + random.random()*26.5

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
        bpy.context.scene.eevee.motion_blur_shutter = random.uniform(0.0, 1.0)**3
        bpy.context.scene.eevee.motion_blur_shutter = 1000;
        print("ADDING MOTION BLUR:", bpy.context.scene.eevee.motion_blur_shutter)

        bpy.context.scene.eevee.motion_blur_steps = 5
    else:
        print("NOT ADDING MOTION BLUR")
        bpy.context.scene.eevee.use_motion_blur = False



def hand_joints_bone_root_position(bones, bone, camera_pos):
    return list((bones.matrix_world @ bone.matrix).translation - camera_pos)


def hand_joints(st, bones):
    locations = []

    camera_location = st.camera_root_empty.location
    locations.append(hand_joints_bone_root_position(
        bones, bones.pose.bones["hand_L"], camera_location))

    for finger in 'thumb', 'index', 'middle', 'ring', 'pinky':
        for joint in '01', '02', '03':
            name = finger+joint+"_L"
            locations.append(hand_joints_bone_root_position(
                bones, bones.pose.bones[name], camera_location))

        name = finger+'03_L'
        bone = bones.pose.bones[name]
        loc = bones.matrix_world @ bone.matrix @ mathutils.Vector(
            (0, bone.length, 0))
        loc -= camera_location
        locations.append(loc)

    locations_openxr = []
    locations_opencv = []

    for i in range(21):
        l = locations[i]
        openxr = [l[0], l[2], -l[1]]
        opencv = [l[0], -l[2], l[1]]
        locations_openxr.append(openxr)
        locations_opencv.append(opencv)

    # print(len(locations))
    return (locations_openxr, locations_opencv)


def make_tmp_render_folders(st):
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


def main():
    st = State()

    mlib.load_temp_blenddata(st)

    mlib.make_render_output_with_alpha(st)

    make_tmp_render_folders(st)

    mlib.get_right_camera_pose(st)
    # print(st.right_in_left_pos, st.right_in_left_rot)

    init_random(st)
    dumbest_possible_add_lights()
    dumb_render_settings()

    mlib.stereoscopy()
    mlib.make_cameras(st)

    body_object = bpy.data.objects["m_ca01"]
    body_object.modifiers["mbastlab_subdvision"].render_levels = 1

    wrist_empty = mlib.create_empty('wrist_empty')
    wrist_empty.rotation_mode = 'QUATERNION'
    # wrist_empty.rotation_quaternion = (0, -0.903, 0, 0.430)
    # wrist_empty.rotation_quaternion = (0, 0.430, 0, 0.903)
    wrist_empty.rotation_quaternion = (-.304, .304, 0.639, 0.639)

    wrist_empty.location = (0.122, -0.29, 1.13+0.2)

    # st.camera_root_rel_empty.location = st.right_in_left_pos
    # st.camera_root_rel_empty.rotation_quaternion = st.right_in_left_rot
    # return

    # This is dumb; the default rig's hand bone is 180 degrees around the forward axis wrong. But probably good to have a layer of indirection anyhow
    hand_target = mlib.create_empty('hand_target')
    if False:
        hand_target.rotation_quaternion.w = 0
        hand_target.rotation_quaternion.x = 0
        hand_target.rotation_quaternion.y = 0.785892
        hand_target.rotation_quaternion.z = -0.618364
    elif False:
        hand_target.rotation_quaternion = (-0.019, 0.009, 0.763, -0.647)
    else:
        hand_target.rotation_quaternion = (-0.013, 0.016, 0.763, -0.647)
    hand_target.parent = wrist_empty

    start_z = wrist_empty.location.z
    start_x = wrist_empty.location.x

    tip_correct_rot = mathutils.Quaternion((-0.707107, 0.707107, 0, 0))

    # might be equivalent to above
    wrist_correct_prerot = mathutils.Quaternion((0.707107, 0.707107, 0, 0))

    st.file_wristpose = readcsv_wristpose.get_file()

    for i in range(header.env_settings.num_frames):
        framerate_divisor = \
            header.env_settings.wristpose_framerate / header.env_settings.out_framerate
        p, q = readcsv_wristpose.get_pos(
            st.file_wristpose, env_settings.wristpose_start_idx + int(i*framerate_divisor))
        wrist_empty.location.x = p.x
        wrist_empty.location.y = -p.z
        wrist_empty.location.z = p.y

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

    st.file = readcsv_fingerpose.get_file()

    for i in range(26):
        e = mlib.create_empty()
        # e.rotation_mode = 'QUATERNION'
        e.empty_display_size = .01
        # e.empty_display_type = 'ARROWS'
        e.parent = wrist_empty
        st.empties.append(e)

    for i in range(header.env_settings.num_frames):
        for j in range(26):
            # s = readcsv_fingerpose.readcsv_fingerpose_settings(st.file, 1.1)
            s = readcsv_fingerpose.readcsv_fingerpose_settings(st.file, 1.02)
            framerate_divisor = \
                header.env_settings.fingerpose_framerate / header.env_settings.out_framerate
            p, q = readcsv_fingerpose.get_joint(
                s, env_settings.fingerpose_start_idx + int(i*(framerate_divisor)), j)

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

    if make_guy:

        root_empty = mlib.create_empty()
        root_empty.location.z = -1.7158
        root_empty.location.y = -0.12389

        # rotate 180 degrees on Blender's Z/vertical axis - MB-Lab characters are pointing backwards for some reason.
        root_empty.rotation_quaternion = (0, 0, 0, -1)

        bones_object = None
        for obj in bpy.data.objects:
            if "doot doot" in obj.name:
                bones_object = obj
        if bones_object == None:
            raise
        bones_object.parent = root_empty

        setup_arm_ik(bones_object, hand_target)

        pairings = [
            (readcsv_fingerpose.XRT_HAND_JOINT_THUMB_TIP, "thumb03_L"),
            (readcsv_fingerpose.XRT_HAND_JOINT_INDEX_TIP, "index03_L"),
            (readcsv_fingerpose.XRT_HAND_JOINT_MIDDLE_TIP, "middle03_L"),
            (readcsv_fingerpose.XRT_HAND_JOINT_RING_TIP, "ring03_L"),
            (readcsv_fingerpose.XRT_HAND_JOINT_LITTLE_TIP, "pinky03_L"),
        ]

        for pair in pairings:
            ik = mlib.new_constraint(bones_object, pair[1], "IK")
            ik.target = st.empties[pair[0]]
            ik.chain_count = 3
            ik.use_tail = True
            ik.use_rotation = True
            ik.orient_weight = 0.05

        joints_1dof = ["thumb02_L", "thumb03_L",
                       "index02_L", "index03_L",
                       "middle02_L", "middle03_L",
                       "ring02_L", "ring03_L",
                       "pinky02_L", "pinky03_L", ]

        for joint in joints_1dof:
            bones_object.pose.bones[joint].lock_ik_y = True
            bones_object.pose.bones[joint].lock_ik_z = True

            bones_object.pose.bones[joint].use_ik_limit_x = True
            bones_object.pose.bones[joint].ik_min_x = math.radians(-100)
            bones_object.pose.bones[joint].ik_max_x = math.radians(6)
            # bones_object.pose.bones[joint].ik_max_x = math.radians(0)

        proximals = ["index01_L",
                     "middle01_L",
                     "ring01_L",
                     "pinky01_L"]

        for joint in proximals:
            bone = bones_object.pose.bones[joint]
            bone.use_ik_limit_x = True
            bone.ik_min_x = math.radians(-100)
            bone.ik_max_x = math.radians(35)

            bone.use_ik_limit_y = True
            bone.ik_min_y = math.radians(-10)
            bone.ik_max_y = math.radians(10)

            bone.use_ik_limit_z = True
            bone.ik_min_z = math.radians(-30)
            bone.ik_max_z = math.radians(30)

        joint = "thumb01_L"
        bone = bones_object.pose.bones[joint]
        bone.use_ik_limit_x = True
        bone.ik_min_x = math.radians(-45)
        bone.ik_max_x = math.radians(45)

        bone.use_ik_limit_y = True
        bone.ik_min_y = math.radians(-10)
        bone.ik_max_y = math.radians(10)

        bone.use_ik_limit_z = True
        bone.ik_min_z = math.radians(-40)
        bone.ik_max_z = math.radians(40)

        # mlib.add_1dof_constraint(bones_object, joint)

        # for j_idx in readcsv_fingerpose.

    # MBLab sets this for some reason
    for scene in bpy.data.scenes:
        print("scene is", scene)
        scene.render.engine = 'BLENDER_EEVEE'
        # scene.render.engine = 'CYCLES'
    # mlib.make_background_voronoi()

    # for i in range(20):

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
    raise
    bpy.ops.render.render(animation=True, write_still=True)

    print("hey kids")

    if (header.env_settings.dont_exit_immediately):
        print("Exiting because EXIT_IMMEDIATELY is set")
        exit(0)
    return


if __name__ == "__main__":
    try:
        main()
    finally:
        print(traceback.format_exc())
    # or
        # print(sys.exc_info())
        # if header.env_settings.exit_immediately:
        #     exit(0)
