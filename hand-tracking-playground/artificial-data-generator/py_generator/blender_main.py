from dataclasses import dataclass
import sys  # nopep8
import os  # nopep8
sys.path.insert(0, os.path.dirname(__file__))  # nopep8

import traceback


import logging
import os

import grpc
import artificialdata_pb2
import artificialdata_pb2_grpc

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
from io import StringIO


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


def init_random(st: State):

    st.blender_scene.render.fps = 30
    st.blender_scene.render.resolution_x = 768
    st.blender_scene.render.resolution_y = 768


def dumbest_possible_add_lights(st: State):
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
        st.objects_to_delete.append(obj)
        max_light_dist = 2

        move_amount = np.random.uniform(-max_light_dist, max_light_dist, 3)
        square_move_len = np.linalg.norm(move_amount)**2
        pos = center + square_move_len
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

# Called to remove anything the artist made as an aid when rigging/creating the hand asset


def remove_artist_lights_and_cameras():
    for b_object in bpy.data.objects:
        do_delete = False
        do_delete = do_delete or b_object.type == "LIGHT"
        do_delete = do_delete or b_object.type == "CAMERA"
        if do_delete:
            bpy.data.objects.remove(b_object)


def load_file_get_properties(st, root_dir, name):

    # Don't load UI - we want the artist to be able to leave whatever they want there and get the same UI as a programmer
    # Do use scripts (even though this is a security risk if we were loading unknown files) - we need them for Drivers for corrective blendshapes.

    # bpy.ops.wm.open_mainfile(filepath=str(os.path.join(
    #     root_dir, name+".blend")), load_ui=False, use_scripts=True)
    old_collections = set([col.name for col in bpy.data.collections])

    filepath = root_dir + "/" + name + ".blend"
    with bpy.data.libraries.load(filepath) as (data_from, data_to):

        # This also worked but was jank because we didn't know which ones were orphan data or not

        # data_to.objects = data_from.objects

        # This will make another scene called "Scene.001"
        data_to.scenes = data_from.scenes

    # For anything that was in the scene collection from the file, append it to our scene.
    for obj in bpy.data.scenes["Scene.001"].collection.objects:
        st.blender_scene.collection.objects.link(obj)

    new_collections = set(
        [col.name for col in bpy.data.collections]) - old_collections
    print(new_collections)

    # For anything that was in any other collection from the file, append it to our scene.
    for col in new_collections:
        for obj in bpy.data.collections[col].objects:
            st.blender_scene.collection.objects.link(obj)

    # (Don't do anything with orphan data from the file.)


# It's one of these two. I think it's rot2 but unsure.
rot1: mathutils.Matrix = mathutils.Matrix((
    (1, 0, 0, 0),
    (0, 0, -1, 0),
    (0, 1, 0, 0),
    (0, 0, 0, 1),
))

rot2: mathutils.Matrix = mathutils.Matrix((
    (1, 0, 0, 0),
    (0, 0, 1, 0),
    (0, -1, 0, 0),
    (0, 0, 0, 1),
))

# empirically rot1 seems to be the right one, based on what seems to work in get_finger_curls.cpp
rot = rot1

def hand_joints_bone_root(bones, bone):
    # bones.matrix_world is the matrix that "moves" points into the armatrue object's coordinate space.
    # bone.matrix is the matrix that moves points from the armatrue object's coordinate space to that bone's coordinate space
    # Note that we used to do left_camera.matrix_world.inverse() @ ... in order to get things in the space of the left camera
    # as opposed to in the space of the world.
    # For now though we will do things in world space because we're not simulating a stereo camera anymore
    # @todo for when we do validation datasets again
    m = bones.matrix_world @ bone.matrix @ rot
    return (m.to_translation(), m.to_quaternion())


def hand_joints_bone_tip(bones, bone):
    # bones.matrix_world is the matrix that "moves" points into the armatrue object's coordinate space.
    # bone.matrix is the matrix that moves points from the armatrue object's coordinate space to that bone's coordinate space
    # Note that we used to do left_camera.matrix_world.inverse() @ ... in order to get things in the space of the left camera
    # as opposed to in the space of the world.
    # For now though we will do things in world space because we're not simulating a stereo camera anymore
    # @todo for when we do validation datasets again
    matrix_length = mathutils.Matrix()
    matrix_length.translation.y = bone.length
    m = bones.matrix_world @ bone.matrix @ matrix_length @ rot
    return (m.to_translation(), m.to_quaternion())


def hand_joints(st, bones):
    poses = []  # has a bunch of (position, orientation)s in it

    poses.append(hand_joints_bone_root(
        bones, bones.pose.bones["Wrist"]))

    for finger in 'Thumb', 'Index', 'Middle', 'Ring', 'Little':
        if finger == 'Thumb':
            ls = 'Metacarpal', 'Proximal', 'Distal'
        else:
            ls = 'Metacarpal', 'Proximal', 'Intermediate', 'Distal'
        for joint in ls:
            name = finger+joint
            poses.append(hand_joints_bone_root(
                bones, bones.pose.bones[name]))

        # Get tip bone
        name = finger+'Distal'

        # bone = bones.pose.bones[name]
        poses.append(hand_joints_bone_tip(bones, bones.pose.bones[name]))
    poses.append(hand_joints_bone_tip(bones, bones.pose.bones["Forearm"]))
    return poses


'''
# All of these shall have -z forward.
"metacarpal_roots": Array<Vec3>
"metacarpal_plus_z": Array<NormalizedVec3>
"metacarpal_plus_x": Array<NormalizedVec3>
"metacarpal_lengths": Array<float>
#"nonx_lengths": ["thumb": Array<float>, "index": Array<float>, ...]
"nonx_lengths": Array<Array<float>> 
'''


def make_blender_to_openxr(l: mathutils.Vector):
    return [l[0], l[2], -l[1]]


quat_blender_to_openxr = mathutils.Quaternion(
    (0.707, 0.707, 0, 0))  # (PI/2, 0, 0)
quat_blender_to_openxr = mathutils.Quaternion(
    (0.707, -0.707, 0, 0))  # (-PI/2, 0, 0)


def get_proportions(bones_object) -> dict:

    bobo = {"metacarpal_roots": [],
            "metacarpal_plus_x": [],
            "metacarpal_plus_z": [],
            "metacarpal_length": [],
            "metacarpal_min_max_x": [],
            "metacarpal_min_max_y": [],
            "metacarpal_min_max_z": [],
            "nonx_length": []}
    hand_size = bones_object.data.bones["Wrist"].length
    bobo["hand_size"] = float(hand_size)

    positions = []
    positions_tail = []
    lengths = []

    for finger in 'Thumb', 'Index', 'Middle', 'Ring', 'Little':

        # note: data.head/data.tail seems to give parent-local
        # yarr this is true https://docs.blender.org/api/current/bpy.types.Bone.html#bpy.types.Bone.head
        # yarrrrr
        databone = bones_object.data.bones[finger+"Metacarpal"]
        posebone = bones_object.pose.bones[finger+"Metacarpal"]
        place = databone.head.copy()
        place2 = databone.tail

        # All the metacarpal bones are parented to the "wrist" _tip_, not the root. Facepalm.
        place.y += hand_size

        # Needs to be a factor of the overall hand size
        place /= hand_size

        bobo["metacarpal_roots"].append(make_blender_to_openxr(place))

        bobo["metacarpal_plus_x"].append(
            make_blender_to_openxr(databone.x_axis))
        bobo["metacarpal_plus_z"].append(
            make_blender_to_openxr(-databone.y_axis))

        bobo["metacarpal_length"].append(databone.length/hand_size)

        # confirmed
        bobo["metacarpal_min_max_x"].append(
            [posebone.ik_min_x, posebone.ik_max_x])
        bobo["metacarpal_min_max_y"].append(
            [posebone.ik_min_z, posebone.ik_max_z])
        bobo["metacarpal_min_max_z"].append(
            [posebone.ik_min_y, posebone.ik_max_y])

        positions.append(place)
        lengths.append((place-place2).length/hand_size)

        nonx_lengths = []

        if finger == 'Thumb':
            ls = 'Proximal', 'Distal'
        else:
            ls = 'Proximal', 'Intermediate', 'Distal'
        for joint in ls:
            place = bones_object.data.bones[finger+joint].head
            place2 = bones_object.data.bones[finger+joint].tail
            dir = place2-place
            dir.x = 0

            nonx_lengths.append(dir.length/hand_size)

        bobo["nonx_length"].append(nonx_lengths)
    return bobo


def get_arm_length_mul(bones_object) -> float:
    upper = bones_object.data.bones["UpperArm"].length
    lower = bones_object.data.bones["Forearm"].length
    arm_len = upper + lower
    moses_arm_len = 0.5969

    mul = arm_len/moses_arm_len

    print("Multiplying mocap data by", mul)

    return mul

    # Moses arn is ~23.5 inches = 0.5969 meters


def main():
    st = State()

    st.server_address = os.getenv("SERVER_ADDRESS")
    st.slot_idx = int(str(os.getenv("SLOT_IDX")))
    st.model_idx = int(str(os.getenv("MODEL_IDX")))

    print(
        f"I've been created with slot idx {st.slot_idx} and model idx {st.model_idx}")

    names = ["3dscanstore_black_male_iik",  # _itasc
             "3dscanstore_black_female",
             "3dscanstore_white_male",
             "3dscanstore_white_female",
             "3dscanstore_asian",
             "mblab_light",
             "uhh"]
    st.blender_scene = bpy.data.scenes["Scene"]  # hack; dry
    load_file_get_properties(st, "/3/epics/artificial_data_4/hands/",
                             names[st.model_idx])
    before_start(st)

    # eventually like 50, but for now 3.
    for i in range(15):
        if one_run(st):
            return
        else:
            clean_up_one_run(st)
    with grpc.insecure_channel(st.server_address) as channel:
        stub = artificialdata_pb2_grpc.ArtificialDataNexusStub(channel)
        t = artificialdata_pb2.sayonara(slot_idx=st.slot_idx)
        stub.goodbye(t)
    print("Goodbye!")


def before_start(st: State):
    # Autodetection is annoying. Just make it right in the blend file.
    st.blender_scene = bpy.data.scenes["Scene"]
    mlib.NO_stereoscopy(st)
    bones_object = bpy.data.objects["doot doot"]
    st.bones_object = bones_object

    st.arm_scale = get_arm_length_mul(bones_object)
    j = get_proportions(st.bones_object)
    st.hand_size = j["hand_size"]
    st.proportions_json = json.dumps(j)

# Note, I am lazy and


def one_run(st: State):

    with grpc.insecure_channel(st.server_address) as channel:
        stub = artificialdata_pb2_grpc.ArtificialDataNexusStub(channel)
        t = artificialdata_pb2.sequenceRequest(
            slot_idx=st.slot_idx, proportions_json=st.proportions_json)
        response: artificialdata_pb2.sequenceReply = stub.askForSequence(t)
    st.pb_response = response

    pa = os.path.dirname(response.output_color_images_folder)
    pa = os.path.join(pa, "model_idx")

    with open(pa, "w") as f:
        f.write(f"{st.model_idx}\n")

    init_random(st)

    st.blender_scene.frame_start = 0
    st.blender_scene.frame_end = response.num_frames-1

    remove_artist_lights_and_cameras()
    dumb_render_settings()

    mlib.add_ambient_occlusion(st)

    if (response.use_exr_background):
        mlib.make_exr_background(st)
    else:
        dumbest_possible_add_lights(st)

    mlib.make_render_output(st, response.render_alpha)

    #

    tip_correct_rot = mathutils.Quaternion((-0.707107, 0.707107, 0, 0))
    rotate_finger_joints = mathutils.Quaternion((0.707107, 0.707107, 0, 0))

    wrist_empty = bpy.data.objects["xr_wrist_target"]
    wrist_empty.rotation_mode = "QUATERNION"  # yes this is necessary - nov 22

    elbow_empty = bpy.data.objects["simple_elbow_target"]
    elbow_empty.rotation_mode = "QUATERNION"  # yes this is necessary - nov 22
    st.orig_elbow_pos = elbow_empty.location.copy()

    # bpy.data.objects["xr_wrist_target"]
    palm_empty = mlib.create_empty("palmmmmm")
    st.objects_to_delete.append(palm_empty)
    palm_empty.parent = wrist_empty
    palm_empty.location.z = -0.04  # Doesnt matter?
    st.empties.append(palm_empty)

    # bpy.data.objects["xr_wrist_target"]
    fake_wrist_empty = mlib.create_empty("fakewrist")
    st.objects_to_delete.append(fake_wrist_empty)
    fake_wrist_empty.parent = wrist_empty
    st.empties.append(fake_wrist_empty)

    # bpy.data.objects["xr_wrist_target"]
    wrist_tail_target = mlib.create_empty("tail_target")
    st.objects_to_delete.append(wrist_tail_target)
    wrist_tail_target.parent = wrist_empty
    wrist_tail_target.location.z = -st.hand_size
    wrist_tail_target.rotation_quaternion = rotate_finger_joints

    wrist_parented_to_bone = mlib.create_empty("empty_parented_to_wrist")
    st.objects_to_delete.append(wrist_parented_to_bone)
    wrist_parented_to_bone.rotation_quaternion = rotate_finger_joints
    wrist_parented_to_bone.location.y -= st.hand_size
    wrist_parented_to_bone.parent = st.bones_object
    wrist_parented_to_bone.parent_bone = "Wrist"
    wrist_parented_to_bone.parent_type = "BONE"

    for finger in 'Thumb', 'Index', 'Middle', 'Ring', 'Little':
        if finger == 'Thumb':
            ls = 'Metacarpal', 'Proximal', 'Distal', "Tip"
        else:
            ls = "Metacarpal", 'Proximal', 'Intermediate', 'Distal', "Tip"
        empties = []
        for joint in ls:
            e = mlib.create_empty(finger+joint)
            st.objects_to_delete.append(e)
            e.empty_display_size = .01
            # e.empty_display_type = 'ARROWS'
            e.parent = wrist_parented_to_bone
            # e.parent_bone = "Wrist"
            # e.parent_type = "BONE"
            # e.parent_type = F"ARMATURE"
            empties.append(e)
            st.empties.append(e)
        for idx, joint in enumerate(ls[:-1]):
            e = empties[idx+1]

            # we want metacarpal _and_ proximal to have chain count of 1.
            # proximal joint IK fucks up accuracy in metacarpal IK

            # but not for the thumb.
            if finger == "Thumb":
                idx += 1
            con = mlib.new_constraint(st.bones_object, finger+joint, "IK")
            print(con)
            st.bone_constraints_to_delete.append(con)
            con.chain_count = max(1, idx)
            con.target = e

    # might be equivalent to above
    wrist_correct_prerot = mathutils.Quaternion((0.707107, 0.707107, 0, 0))

    f = pd.read_csv(StringIO(response.wristpose_csv))

    for i in range(response.num_frames):

        p, q = readcsv_wristpose.get_pos(f, i)
        wrist_empty.location.x = p.x
        wrist_empty.location.y = -p.z
        wrist_empty.location.z = p.y

        wrist_empty.location *= st.arm_scale

        q.rotate(wrist_correct_prerot)
        # wrist_empty.location = p
        # q.
        wrist_empty.rotation_quaternion = q

        # wrist_empty.location.z = start_z + 0.02*math.sin(st.frame*0.1)
        # wrist_empty.location.x = start_x + 0.03*math.cos(st.frame*0.1)
        wrist_empty.keyframe_insert(data_path="location", frame=i)
        wrist_empty.keyframe_insert(
            data_path="rotation_quaternion", frame=i)

        ########

        p, q = readcsv_wristpose.get_pos(f, i, True)
        elbow_empty.location.x = p.x
        elbow_empty.location.y = -p.z
        elbow_empty.location.z = p.y

        #!@todo make this configurable by-model!
        elbow_empty.location *= st.arm_scale

        q.rotate(wrist_correct_prerot)
        elbow_empty.rotation_quaternion = q

        elbow_empty.keyframe_insert(data_path="location", frame=i)
        elbow_empty.keyframe_insert(
            data_path="rotation_quaternion", frame=i)

    fcsv = readcsv_fingerpose.fingerpose_csv(
        st, response.fingerpose_csv)

    for i in range(response.num_frames):
        for j in range(26):
            # Sigh, "just" making the target joints bigger makes it look a lot better (but it shouldn't)
            # s = readcsv_fingerpose.readcsv_fingerpose_settings(st.file, st.)
            # s = readcsv_fingerpose.readcsv_fingerpose_settings(st.file, 1.02)

            if j == 0:
                continue

            p, q = fcsv.get_joint(
                st, i, j)

            # q = q.rotate()
            newguy = tip_correct_rot.copy()
            newguy.rotate(q)
            q = newguy
            # q = tip_correct_rot.rotate(q)
            e = st.empties[j]
            # eh?
            e.rotation_quaternion = q

            e.location = p
            # e.location[1] -= 0.095
            e.keyframe_insert(data_path="location", frame=i)
            e.keyframe_insert(data_path="rotation_quaternion", frame=i)

    # Ok, pose is made.

    # yeah, 26. we do have a forearm joint at the end, but we're skipping palm as it is for nerds.
    hand_pose_flat = np.zeros((st.pb_response.num_frames, 26*7))
    # For now, they're all valid. I _think_ we have none to exclude, for once!
    valid_samples = np.ones((st.pb_response.num_frames), dtype=np.int32)
    # vec3, quat, fx, fy, cx, cy
    camera_info = np.zeros((st.pb_response.num_frames, 3+4+4))

    arr_blender = []

    for i in range(st.pb_response.num_frames):
        bpy.context.scene.frame_current = i

        bpy.context.view_layer.update()

        hand_joint_pose_array = hand_joints(
            st, st.bones_object)
        arr_blender.append([a[0] for a in hand_joint_pose_array])
        for j in range(26):  # 26th is the forearm tip
            root = j * 7
            # position
            hand_pose_flat[i, root+0] = hand_joint_pose_array[j][0].x
            hand_pose_flat[i, root+1] = hand_joint_pose_array[j][0].y
            hand_pose_flat[i, root+2] = hand_joint_pose_array[j][0].z
            # orientation
            hand_pose_flat[i, root+3] = hand_joint_pose_array[j][1].w
            hand_pose_flat[i, root+4] = hand_joint_pose_array[j][1].x
            hand_pose_flat[i, root+5] = hand_joint_pose_array[j][1].y
            hand_pose_flat[i, root+6] = hand_joint_pose_array[j][1].z
        # arr_slice = np.asarray(hand_joint_pose_array)
        # arr_slice = arr_slice.flatten()
        # hand_pose_flat[i] = arr_slice

    # We write out the camera info later.

    df_hp = pd.DataFrame(hand_pose_flat)
    df_vs = pd.DataFrame(valid_samples, columns=["valid"])

    df_hp.to_csv(response.hand_poses_csv, encoding='utf-8',
                 index=False)  # , columns=columns);
    df_vs.to_csv(response.valid_samples_csv, encoding='utf-8',
                 index=False)  # , columns=columns);

    miniball_sphere = mlib.create_empty("mini_sphere")
    st.objects_to_delete.append(miniball_sphere)
    miniball_sphere.empty_display_size = 1.0
    miniball_sphere.empty_display_type = 'SPHERE'

    for i in range(st.pb_response.num_frames):
        marker = st.blender_scene.timeline_markers.new(str(i))
        marker.frame = i
        camera = mlib.create_camera(f"camera{i}")
        st.objects_to_delete.append(camera)
        # camera.location.x = i/1000
        marker.camera = camera

        vec_list = arr_blender[i]
        center, radius = mlib.miniball(vec_list[:25])

        miniball_sphere.location = center
        miniball_sphere.scale = [radius]*3
        miniball_sphere.keyframe_insert(data_path="scale", frame=i)
        miniball_sphere.keyframe_insert(data_path="location", frame=i)

        camera.rotation_quaternion = mlib.simple_rotation(
            mathutils.Vector((0, 0, -1)), center.normalized())
        camera.keyframe_insert(data_path="rotation_quaternion", frame=i)

        # A little padding
        target_radius = radius * 2.01

        angular_radius = math.sin(target_radius/center.length)

        camera.data.angle = angular_radius*1.9

        camera.data.display_size = center.length * \
            math.tan(camera.data.angle/2) * 2
        camera.data.lens_unit = 'FOV'

        rr = st.blender_scene.render.resolution_x

        # If you look at the above, it looks like we might be able to just use angular_radius.
        # Don't! We add padding to that. Safest thing is to use half of the actual angle used.
        tan_ = 1/math.tan(camera.data.angle/2)

        camera_info[i][0:3] = camera.location
        camera_info[i][3] = camera.rotation_quaternion.x
        camera_info[i][4] = camera.rotation_quaternion.y
        camera_info[i][5] = camera.rotation_quaternion.z
        camera_info[i][6] = camera.rotation_quaternion.w
        camera_info[i][7] = (rr/2)*tan_  # fx
        camera_info[i][8] = (rr/2)*tan_  # fy
        camera_info[i][9] = rr/2  # cx
        camera_info[i][10] = rr/2  # cy
        df_ci = pd.DataFrame(camera_info, columns=[
                             "px", "py", "pz", "qx", "qy", "qz", "qw", "fx", "fy", "cx", "cy"])
        df_ci.to_csv(response.camera_info_csv, encoding='utf-8',
                     index=False)  # , columns=columns);

    if not response.dont_render:
        bpy.ops.render.render(animation=True, write_still=True)
        return False
    else:
        return True


def clean_up_one_run(st: State):
    for obj in st.objects_to_delete:
        bpy.data.objects.remove(obj)
    # todo this is VERY questionable: we could just leave these around and only create them once
    for con in st.bone_constraints_to_delete:
        for bone in st.bones_object.pose.bones:
            try:
                bone.constraints.remove(con)
            except RuntimeError:
                pass
            except ReferenceError:
                # ReferenceError: StructRNA of type KinematicConstraint has been removed
                pass
            # bones_object.constraints.remove(con)
    st.objects_to_delete.clear()
    st.bone_constraints_to_delete.clear()
    st.empties.clear()

    # for image in bpy.data.images:
    #     if image.users == 0:
    #         print(f"Found orphan image {image.name}!")
    #         bpy.data.images.remove(image)
    mlib.remove_orphans_of_datatype(bpy.data.images)
    mlib.remove_orphans_of_datatype(bpy.data.objects)


if __name__ == "__main__":

    dont_exit_immediately = bool(int(str(os.getenv("DONT_EXIT_IMMEDIATELY"))))
    try:
        main()
    except:
        print("failed!", traceback.format_exc())
    if dont_exit_immediately:
        print("Not exiting because DONT_EXIT_IMMEDIATELY is set")

        bpy.ops.wm.save_as_mainfile(filepath="/tmp/tmp.blend")
        bpy.ops.wm.open_mainfile(filepath="/tmp/tmp.blend")
    else:
        print("Exiting because DONT_EXIT_IMMEDIATELY is not set")
        exit(0)
