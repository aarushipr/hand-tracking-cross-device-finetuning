import sys  # nopep8
import os  # nopep8
sys.path.insert(0, os.path.dirname(__file__))  # nopep8
sys.path.append('/home/moses/.local/lib/python3.10/site-packages')  # nopep8

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
    return o


def create_camera(name="empty"):
    c = bpy.data.cameras.new(name)
    o = bpy.data.objects.new(name, c)

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

    node_tree.links.new(texcoordnode.outputs["Generated"], mappingnode.inputs["Vector"])
    node_tree.links.new(mappingnode.outputs["Vector"], exrnode.inputs["Vector"])
    node_tree.links.new(exrnode.outputs["Color"], backgroundnode.inputs["Color"])
    node_tree.links.new(backgroundnode.outputs["Background"], worldnode.inputs["Surface"])

    # np.normal(0.3) would probably be good.
    mappingnode.inputs["Location"].default_value[0] = np.random.normal(0, 0.2)
    mappingnode.inputs["Location"].default_value[1] = np.random.normal(0, 0.2)
    mappingnode.inputs["Location"].default_value[2] = np.random.normal(0, 0.2)

    mappingnode.inputs["Rotation"].default_value[0] = np.random.normal(0, 0.4)
    mappingnode.inputs["Rotation"].default_value[1] = np.random.normal(0, 0.4)
    mappingnode.inputs["Rotation"].default_value[2] = random.uniform(0, math.pi*2)

    mappingnode.inputs["Scale"].default_value[0] = np.random.normal(1.0, 0.1)
    mappingnode.inputs["Scale"].default_value[1] = np.random.normal(1.0, 0.1)
    mappingnode.inputs["Scale"].default_value[2] = np.random.normal(1.0, 0.1)

    slug = "/3/epics/artificial_data_3/hdris/"
    choice = random.choice(os.listdir(slug))

    exrnode.image = load_image(os.path.join(slug, choice))

def add_ambient_occlusion(st):
    st.blender_scene.eevee.use_gtao = True
    st.blender_scene.eevee.gtao_distance = 0.23
    st.blender_scene.eevee.gtao_factor = 1.0
    st.blender_scene.eevee.gtao_quality = 0.25
    st.blender_scene.eevee.use_gtao_bent_normals = True
    st.blender_scene.eevee.use_gtao_bounce = True


def make_render_output(st, make_alpha_output):

    st.blender_scene.render.image_settings.file_format = 'JPEG'
    
    if make_alpha_output:
        st.blender_scene.render.film_transparent = True
    else:
        st.blender_scene.render.film_transparent = False


    st.blender_scene.use_nodes = True
    node_tree = st.blender_scene.node_tree

    # Learned this lesson on august 21:
    # If the artist (ie. me, Moses) is dumb, there can be dumb stuff in the
    # compositor node tree that we don't want
    # and we need to get rid of all of it, no matter what it is.
    for node in node_tree.nodes:
        node_tree.nodes.remove(node)

    # node_tree.nodes.remove(node_tree.nodes["Composite"])

    # renderlayer
    renderlayer = node_tree.nodes.new("CompositorNodeRLayers")
    
    # ['Render Layers']

    file_output_rgb = node_tree.nodes.new('CompositorNodeOutputFile')

    node_tree.links.new(
        renderlayer.outputs['Image'], file_output_rgb.inputs['Image'])

    file_output_alpha = node_tree.nodes.new('CompositorNodeOutputFile')

    file_output_rgb.base_path = header.env_settings.output_images_color_base_path

    if not make_alpha_output:
        return



    if False:
        node_tree.links.new(
            renderlayer.outputs['Alpha'], file_output_alpha.inputs['Image'])

    else:
        math_0_multiply = node_tree.nodes.new('CompositorNodeMath')
        math_0_multiply.operation = 'MULTIPLY'
        math_0_multiply.inputs[0].default_value = -1

        math_1_add = node_tree.nodes.new('CompositorNodeMath')
        math_1_add.operation = 'ADD'
        math_1_add.inputs[0].default_value = 1

        node_tree.links.new(
            renderlayer.outputs['Alpha'], math_0_multiply.inputs[1])
        node_tree.links.new(math_0_multiply.outputs[0], math_1_add.inputs[1])
        node_tree.links.new(
            math_1_add.outputs[0], file_output_alpha.inputs['Image'])



    file_output_alpha.base_path = header.env_settings.output_images_alpha_base_path


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

    # Blender won't let us have 0 views, so create this one then delete the default ones.
    re.views.new("tmp")
    re.views.remove(bpy.context.scene.render.views["left"])
    re.views.remove(bpy.context.scene.render.views["right"])

    for camera in 0, 1:
        for direction in "forward", "left", "right", "top", "bottom":
            the = re.views.new(f"camera_{camera}_{direction}")
            the.camera_suffix = f"_{camera}_{direction}"

    # At end
    re.views.remove(bpy.context.scene.render.views["tmp"])



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

def get_center_camera_pose(st): # :State
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

    # This is moved back and up a little bit to get out of the way of the IK arm

    st.camera_center_empty.location.x = 0
    # Came up with this on a whim
    st.camera_center_empty.location.y = 0.06
    # Ditto
    st.camera_center_empty.location.z = -0.01

    st.camera_center_empty.location += mathutils.Vector(tuple(np.random.uniform(-0.03, 0.03, 3)))

    # 90-degree rotation so that -z points backwards by default.
    # Eventually, do something smarter (or add aNOTHER thing to the hierarchy)
    # so that the canting is balanced. For now this is fine though
    st.camera_center_empty.rotation_quaternion = (0.707, 0.707, 0, 0)

    extra_random_rot = mathutils.Quaternion(tuple(np.random.uniform(-0.04, 0.04, 3))) # Making a quaternion out of a random axis-angle rotation
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
            camera.data.angle = math.pi/2
            # 1mm
            camera.data.clip_start = 0.001
            # 5 meters. Overkill but fine
            camera.data.clip_end = 5

            camera.rotation_quaternion = e.direction

        # Eh sure
        bpy.context.scene.camera = bpy.data.objects["camera_0_forward"]
