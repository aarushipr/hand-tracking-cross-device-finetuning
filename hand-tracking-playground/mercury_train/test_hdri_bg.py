# Standalone diagnostic -- NOT part of the pipeline, just isolates the
# "does an HDRI world background actually show up under WSL software
# rendering" question. Run with:
#   LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe blender -b --python test_hdri_bg.py
import bpy
import os
import random

hdris_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "hdris")
hdris_dir = os.path.normpath(hdris_dir)
listing = os.listdir(hdris_dir)
print(f"HDRIs found: {listing}")

world = bpy.context.scene.world
world.use_nodes = True
node_tree = world.node_tree
for node in list(node_tree.nodes):
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

chosen = random.choice(listing)
img = bpy.data.images.load(os.path.join(hdris_dir, chosen))
exrnode.image = img
print(f"Loaded {chosen}, size={img.size[:]}, channels={img.channels}")

scene = bpy.context.scene
scene.render.film_transparent = False
scene.render.resolution_x = 256
scene.render.resolution_y = 256
scene.render.engine = 'BLENDER_EEVEE'  # the "EEVEE Next" rewrite kept this old enum string

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hdri_bg_test.png")
scene.render.filepath = out_path
scene.render.image_settings.file_format = 'PNG'

cam = scene.camera
if cam is None:
    bpy.ops.object.camera_add(location=(0, -5, 1.6))
    cam = bpy.context.object
    scene.camera = cam
import mathutils
cam.location = (0, -5, 1.6)
cam.rotation_euler = mathutils.Quaternion((1, 0, 0), 1.4).to_euler()

bpy.ops.render.render(write_still=True)
print(f"Rendered to {out_path}")

# Sample a background-region pixel directly and print it, so the answer
# doesn't depend on how a PNG viewer displays it.
result_img = bpy.data.images.load(out_path)
w, h = result_img.size
px = list(result_img.pixels[:])
# top-left corner, well away from the cube -- should be pure background
idx = (0 * w + 5) * 4
print(f"Top-left-ish background pixel RGBA: {px[idx:idx+4]}")
idx_mid = ((h // 2) * w + 5) * 4
print(f"Mid-left background pixel RGBA: {px[idx_mid:idx_mid+4]}")
