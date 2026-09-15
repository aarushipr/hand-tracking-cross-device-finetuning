import sys  # nopep8
import os  # nopep8
sys.path.insert(0, os.path.dirname(__file__))  # nopep8
import site  # nopep8
# See header.py for why this is computed rather than hardcoded.
sys.path.append(site.getusersitepackages())  # nopep8

import bpy  # nopep8
import math  # nopep8

import mathutils
import subprocess
import json
import header
from dataclasses import dataclass
import random
import numpy as np


# WXYZ, not XYZW.
sqrt2_2 = math.sqrt(2) / 2
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


@dataclass
class camera_dir_pairing:
    name: str
    direction: mathutils.Quaternion


camera_dir_pairings = [
    camera_dir_pairing("forward", camera_forward),
    camera_dir_pairing("left", camera_left),
    camera_dir_pairing("right", camera_right),
    camera_dir_pairing("top", camera_top),
    camera_dir_pairing("bottom", camera_bottom),
]


def create_empty(name="empty"):
    o = bpy.data.objects.new(name, None)

    bpy.context.scene.collection.objects.link(o)

    o.empty_display_size = .02
    o.empty_display_type = 'ARROWS'
    o.rotation_mode = 'QUATERNION'
    o.show_in_front = True
    return o


def create_camera(name="empty"):
    c = bpy.data.cameras.new(name)
    o = bpy.data.objects.new(name, c)

    o.data.clip_start = 0.0001  # 0.1mm
    o.data.clip_end = 3  # 3 meters. Note this is for *hands*

    bpy.context.scene.collection.objects.link(o)

    o.rotation_mode = 'QUATERNION'
    return o


def create_light(name="empty"):
    # ('POINT', 'SUN', 'SPOT', 'AREA')
    c = bpy.data.lights.new(name, "POINT")
    o = bpy.data.objects.new(name, c)

    bpy.context.scene.collection.objects.link(o)

    return o


def new_constraint(obj, bone_name, type):
    return obj.pose.bones[bone_name].constraints.new(type)


def add_1dof_constraint(obj, bone_name):
    c = new_constraint(obj, bone_name, 'LIMIT_ROTATION')

    c.owner_space = 'LOCAL'

    c.use_limit_y = True
    c.use_limit_z = True

    c.use_limit_x = True
    c.max_x = math.radians(0)
    c.min_x = math.radians(-90)


def load_image(path):
    img = bpy.data.images.load(path)
    return img


def remove_orphans_of_datatype(dt):
    # Don't remove() while iterating bpy.data.*: it skips entries and leaked HDRIs until OOM.
    orphans = [obj for obj in dt if obj.users == 0]
    for obj in orphans:
        print(
            f"Found orphan object in {dt} with name {obj.name}! Purging!")
        dt.remove(obj)

# Cursed


def make_background_voronoi():
    # XXX: Fragile
    world = bpy.context.scene.world
    world = bpy.data.worlds["World.001"]
    node_tree = world.node_tree
    voronoi_node = node_tree.nodes.new('ShaderNodeTexVoronoi')
    voronoi_node_color_output = voronoi_node.outputs['Color']
    background_color_input = world.node_tree.nodes["Background"].inputs["Color"]

    node_tree.links.new(voronoi_node_color_output, background_color_input)


def make_exr_background(st):
    # XXX: Fragile
    world = st.blender_scene.world

    node_tree = world.node_tree

    for node in node_tree.nodes:
        node_tree.nodes.remove(node)

    texcoordnode = node_tree.nodes.new('ShaderNodeTexCoord')

    mappingnode = node_tree.nodes.new('ShaderNodeMapping')

    exrnode = node_tree.nodes.new('ShaderNodeTexEnvironment')

    backgroundnode = node_tree.nodes.new('ShaderNodeBackground')

    worldnode = node_tree.nodes.new('ShaderNodeOutputWorld')

    node_tree.links.new(
        texcoordnode.outputs["Generated"],
        mappingnode.inputs["Vector"])
    node_tree.links.new(
        mappingnode.outputs["Vector"],
        exrnode.inputs["Vector"])
    node_tree.links.new(
        exrnode.outputs["Color"],
        backgroundnode.inputs["Color"])
    node_tree.links.new(
        backgroundnode.outputs["Background"],
        worldnode.inputs["Surface"])

    # np.normal(0.3) would probably be good.
    mappingnode.inputs["Location"].default_value[0] = np.random.normal(0, 0.2)
    mappingnode.inputs["Location"].default_value[1] = np.random.normal(0, 0.2)
    mappingnode.inputs["Location"].default_value[2] = np.random.normal(0, 0.2)

    mappingnode.inputs["Rotation"].default_value[0] = np.random.normal(0, 0.4)
    mappingnode.inputs["Rotation"].default_value[1] = np.random.normal(0, 0.4)
    mappingnode.inputs["Rotation"].default_value[2] = random.uniform(
        0, math.pi * 2)

    mappingnode.inputs["Scale"].default_value[0] = np.random.normal(1.0, 0.1)
    mappingnode.inputs["Scale"].default_value[1] = np.random.normal(1.0, 0.1)
    mappingnode.inputs["Scale"].default_value[2] = np.random.normal(1.0, 0.1)

    # Configurable, falling back to <thesis root>/hdris/, five levels up from here.
    default_hdris_dir = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..", "..", "..", "hdris"))
    slug = os.environ.get("GEN_HDRIS_DIR", default_hdris_dir)

    if not os.path.isdir(slug):
        raise FileNotFoundError(
            f"HDRI directory not found: {slug!r}. Set the GEN_HDRIS_DIR "
            f"environment variable to point at a folder of .exr/.hdr files.")

    choice = random.choice(os.listdir(slug))

    exrnode.image = load_image(os.path.join(slug, choice))


def add_ambient_occlusion(st):
    # EEVEE Next replaced GTAO: use_fast_gi/fast_gi_distance/fast_gi_quality <- use_gtao/*.
    # gtao_factor, use_gtao_bent_normals and use_gtao_bounce have no equivalent; dropped.
    st.blender_scene.eevee.use_fast_gi = True
    st.blender_scene.eevee.fast_gi_distance = 0.23
    st.blender_scene.eevee.fast_gi_quality = 0.25


def make_render_output(st: header.State, make_alpha_output):
    # No compositor graph: render_and_save_frame() renders directly and processes pixels here.
    # CompositorNodeOutputFile is forced to multilayer EXR that reads back as 0 channels.
    # CompositorNodeViewer works in the UI but is black under headless -b; plain PNG avoids both.
    # The compositor is disabled explicitly so the artist's own .blend nodes can't interfere.
    st.blender_scene.use_nodes = False

    if make_alpha_output:
        st.blender_scene.render.film_transparent = True
    else:
        st.blender_scene.render.film_transparent = False


def render_and_save_frame(st: header.State, frame_idx: int, save_alpha: bool):
    """Renders the scene's *current* frame (caller sets frame_current
    beforehand) to a temporary plain RGBA PNG, reads that back, and saves
    the processed 8-bit monochrome PNG(s) derived from it; see
    make_render_output()'s comment for the two approaches that silently
    broke before this one and why. Deletes the temporary raw PNG and its
    in-memory datablock once read.

    Called once per frame instead of a single
    bpy.ops.render.render(animation=True) covering the whole sequence,
    safe because every bone/empty/camera transform is keyframed per-frame
    before this runs, so rendering frame_current one at a time is
    equivalent to rendering the animation in one call.

    Applies the same per-frame Gaussian gain noise (sigma drawn uniformly
    from [2, 12] in 8-bit/uint8 units) matching the noise floor of real XR
    mono sensors that convert_folder_exr_to_png() used to add.

    imgs_alpha stores 1 - render_alpha, matching the old compositor
    Math-node MULTIPLY(-1) + ADD(1) pair this replaces, computed here in
    numpy instead since the full buffer is already in Python, no need to
    round-trip it through compositor nodes at all.
    """
    scene = st.blender_scene
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '16'

    raw_path = os.path.join(
        st.json_response["output_color_images_folder"],
        f"_raw_{frame_idx:04d}.png")
    scene.render.filepath = raw_path
    bpy.ops.render.render(write_still=True)

    # check_existing=False: force a fresh read, not a stale datablock from a reused temp name.
    raw_img = bpy.data.images.load(raw_path, check_existing=False)
    w, h = raw_img.size
    pixels = np.array(raw_img.pixels[:], dtype=np.float32).reshape(h, w, 4)
    bpy.data.images.remove(raw_img)
    os.remove(raw_path)

    def save_mono_png(channel, out_path):
        sigma = float(np.random.uniform(2.0, 12.0)) / 255.0
        noisy = channel + \
            np.random.normal(0.0, sigma, channel.shape).astype(np.float32)
        noisy = np.clip(noisy, 0.0, 1.0)

        out = np.ones((h, w, 4), dtype=np.float32)
        out[:, :, 0] = noisy
        out[:, :, 1] = noisy
        out[:, :, 2] = noisy

        save_img = bpy.data.images.new("_frame_save_tmp", width=w, height=h)
        save_img.pixels[:] = out.flatten()
        save_img.filepath_raw = out_path
        save_img.file_format = 'PNG'
        save_img.save()
        bpy.data.images.remove(save_img)

    luma = (0.2126 * pixels[:, :, 0]
            + 0.7152 * pixels[:, :, 1]
            + 0.0722 * pixels[:, :, 2])
    color_path = os.path.join(
        st.json_response["output_color_images_folder"],
        f"frame_{frame_idx:04d}.png")
    save_mono_png(luma, color_path)

    if save_alpha:
        inverted_alpha = 1.0 - pixels[:, :, 3]
        alpha_path = os.path.join(
            st.json_response["output_alpha_images_folder"],
            f"frame_{frame_idx:04d}.png")
        save_mono_png(inverted_alpha, alpha_path)


def load_temp_blenddata(st):
    filepath = "/3/epics/artificial_data_2/mblab/temp_data.blend"
    content = []
    with bpy.data.libraries.load(filepath) as (data_from, data_to):
        data_to.scenes = ["hands"]
    bpy.context.window.scene = bpy.data.scenes['hands']
    st.blender_scene = bpy.context.window.scene


def stereoscopy():
    re = bpy.context.scene.render

    re.use_multiview = True
    re.views_format = 'MULTIVIEW'

    # Blender won't let us have 0 views, so create this one then delete the
    # default ones.
    re.views.new("tmp")
    re.views.remove(bpy.context.scene.render.views["left"])
    re.views.remove(bpy.context.scene.render.views["right"])

    for camera in 0, 1:
        for direction in "forward", "left", "right", "top", "bottom":
            the = re.views.new(f"camera_{camera}_{direction}")
            the.camera_suffix = f"_{camera}_{direction}"

    # At end
    re.views.remove(bpy.context.scene.render.views["tmp"])


def NO_stereoscopy(st):
    re = st.blender_scene.render

    re.use_multiview = False
    re.views_format = 'STEREO_3D'


def get_right_camera_pose(st):  # :State
    string = header.env_settings.camera_right_in_left

    j = json.loads(string)
    st.right_in_left_pos.x = j['pos'][0]
    st.right_in_left_pos.y = j['pos'][1]
    st.right_in_left_pos.z = j['pos'][2]

    st.right_in_left_rot.x = j['rot'][0]
    st.right_in_left_rot.y = j['rot'][1]
    st.right_in_left_rot.z = j['rot'][2]
    st.right_in_left_rot.w = j['rot'][3]


def get_center_camera_pose(st):  # :State
    string = header.env_settings.camera_left_in_center

    j = json.loads(string)
    st.left_in_center_pos.x = j['pos'][0]
    st.left_in_center_pos.y = j['pos'][1]
    st.left_in_center_pos.z = j['pos'][2]

    st.left_in_center_rot.x = j['rot'][0]
    st.left_in_center_rot.y = j['rot'][1]
    st.left_in_center_rot.z = j['rot'][2]
    st.left_in_center_rot.w = j['rot'][3]

# def fake_get_right_camera_pose(st):  # :State
#     j = {"pos": [0.121850, 0.001008, 0.009092], "rot": [-0.010584, -0.080179, -0.006399, 0.996704]}

#     # j = json.loads(string)
#     st.right_in_left_pos.x = j['pos'][0]
#     st.right_in_left_pos.y = j['pos'][1]
#     st.right_in_left_pos.z = j['pos'][2]

#     st.right_in_left_rot.x = j['rot'][0]
#     st.right_in_left_rot.y = j['rot'][1]
#     st.right_in_left_rot.z = j['rot'][2]
#     st.right_in_left_rot.w = j['rot'][3]

# def fake_get_center_camera_pose(st):  # :State
#     j = {"pos": [-0.060859, -0.001293, 0.005232], "rot": [0.005296, 0.040122, 0.003202, 0.999176]}
#     j = {"pos": [-0.061097, 0.000000, -0.000000], "rot": [0.042914, -0.005425, 0.999010, 0.010358]} # good but strangely needed Z axis rotation
#     j = {"pos": [-0.061097, 0.000000, 0.000000], "rot": [-0.005425, -0.042914, -0.010358, 0.999010]}
#     j = {"pos": [-0.061097, 0.000000, -0.000000], "rot": [0.005425, 0.042914, -0.010358, 0.999010]}
#     j = {"pos": [-0.061097, 0.000000, -0.000000], "rot": [0.005722, 0.037252, -0.003912, 0.999282]}
#     # j = json.loads(string)
#     st.half_pos.x = j['pos'][0]
#     st.half_pos.y = j['pos'][1]
#     st.half_pos.z = j['pos'][2]

#     st.half_rot.x = j['rot'][0]
#     st.half_rot.y = j['rot'][1]
#     st.half_rot.z = j['rot'][2]
#     st.half_rot.w = j['rot'][3]


# Call after get_right_camera_pose
def make_cameras(st):
    get_right_camera_pose(st)
    get_center_camera_pose(st)

    st.camera_center_empty = create_empty("camera_center_empty")

    # This is moved back and up a little bit to get out of the way of the IK
    # arm

    st.camera_center_empty.location.x = 0
    # Came up with this on a whim
    st.camera_center_empty.location.y = 0.06
    # Ditto
    st.camera_center_empty.location.z = -0.01

    st.camera_center_empty.location += mathutils.Vector(
        tuple(np.random.uniform(-0.03, 0.03, 3)))

    # 90-degree rotation so that -z points backwards by default.
    # Eventually, do something smarter (or add aNOTHER thing to the hierarchy)
    # so that the canting is balanced. For now this is fine though
    st.camera_center_empty.rotation_quaternion = (0.707, 0.707, 0, 0)

    # Making a quaternion out of a random axis-angle rotation
    extra_random_rot = mathutils.Quaternion(
        tuple(np.random.uniform(-0.04, 0.04, 3)))
    st.camera_center_empty.rotation_quaternion.rotate(extra_random_rot)

    # mathutils

    # slerped_pose_ori = mathutils.Quaternion(1,0,0,0).slerp()

    camera_empties = []

    for view in range(2):

        camera_empty = create_empty("camera_empty")
        camera_empties.append(camera_empty)
        if view == 0:
            st.left_camera_empty = camera_empty
            camera_empty.parent = st.camera_center_empty
            camera_empty.location = st.left_in_center_pos
            camera_empty.rotation_quaternion = st.left_in_center_rot
        else:
            camera_empty.parent = camera_empties[0]
            camera_empty.location = st.right_in_left_pos
            camera_empty.rotation_quaternion = st.right_in_left_rot

        for e in camera_dir_pairings:
            camera = create_camera(f"camera_{view}_{e.name}")
            camera.data.display_size = 0.1
            camera.parent = camera_empty

            camera.data.lens_unit = 'FOV'
            camera.data.angle = math.pi / 2
            # 1mm
            camera.data.clip_start = 0.001
            # 5 meters. Overkill but fine
            camera.data.clip_end = 5

            camera.rotation_quaternion = e.direction

        # Eh sure
        bpy.context.scene.camera = bpy.data.objects["camera_0_forward"]


def miniball(pts):
    min_x = pts[0].x
    min_y = pts[0].y
    min_z = pts[0].z

    max_x = pts[0].x
    max_y = pts[0].y
    max_z = pts[0].z

    for pt in pts:
        min_x = min(min_x, pt.x)
        min_y = min(min_y, pt.y)
        min_z = min(min_z, pt.z)

        max_x = max(max_x, pt.x)
        max_y = max(max_y, pt.y)
        max_z = max(max_z, pt.z)

    c = mathutils.Vector()
    c.x = (min_x + max_x) / 2
    c.y = (min_y + max_y) / 2
    c.z = (min_z + max_z) / 2

    r = 0
    for pt in pts:
        v = pt - c
        r = max(v.length, r)
    return c, r


def simple_rotation(vector_from, vector_to):
    axis = vector_from.cross(vector_to).normalized()
    angle = math.acos(vector_from.dot(vector_to))

    return mathutils.Quaternion(axis, angle)
