import os
import sys
import bpy
import json
import math
import bmesh
import random
import argparse
import numpy as np
from PIL import Image
from glob import glob
from mathutils import Vector


def setup_scene(resolution=1024):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    configure_world()
    configure_render_settings(resolution=resolution)
    enable_gpu()
    setup_compositor_passes()
    setup_camera_and_target()


def configure_world():
    bpy.context.scene.world = bpy.data.worlds.new("World")
    bpy.context.scene.world.use_nodes = True
    background = bpy.context.scene.world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.7, 0.7, 0.7, 1.0)


def configure_render_settings(segmented=False, resolution=1024):
    scene = bpy.context.scene
    if segmented:
        scene.render.engine = "BLENDER_WORKBENCH"
        scene.display.shading.light = "FLAT"
        scene.display.shading.color_type = "VERTEX"
        scene.display.render_aa = "OFF"
    else:
        scene.render.engine = "CYCLES"
        scene.cycles.samples = 64
        scene.cycles.device = "GPU"
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.frame_start = 0
    scene.frame_current = 0


def enable_gpu():
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = "CUDA"
    preferences.get_devices()
    for device in preferences.devices:
        device.use = True


def setup_compositor_passes():
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    scene.view_layers["ViewLayer"].use_pass_z = True
    scene.view_layers["ViewLayer"].use_pass_normal = True
    scene.view_layers["ViewLayer"].use_pass_object_index = True


def setup_camera_and_target():
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = 40
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam

    target = bpy.data.objects.new("Empty", None)
    target.location = Vector((0.0, 0.0, 0.0))
    bpy.context.collection.objects.link(target)

    dtrack = cam.constraints.new(type="DAMPED_TRACK")
    dtrack.target = target
    dtrack.track_axis = "TRACK_NEGATIVE_Z"

    ltrack = cam.constraints.new(type="LOCKED_TRACK")
    ltrack.target = target
    ltrack.track_axis = "TRACK_NEGATIVE_Z"
    ltrack.lock_axis = "LOCK_Y"


def set_camera_position(camera_position):
    bpy.context.scene.camera.location = Vector(camera_position)
    bpy.context.view_layer.update()


def load_obj(obj_path):
    bpy.ops.import_scene.obj(filepath=obj_path)
    for obj in bpy.context.selected_objects:
        if obj.type == "MESH":
            obj.rotation_euler = (0, 0, 0)
            obj.pass_index = 1


def load_ply(file_path, segmented=False):
    bpy.ops.import_mesh.ply(filepath=file_path)
    obj = bpy.context.selected_objects[0]

    assign_vertex_color_material(obj)

    if segmented:
        bpy.ops.object.shade_flat()

    obj.pass_index = 1


def assign_vertex_color_material(obj):
    mat = bpy.data.materials.new(name="VertexColorMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new(type="ShaderNodeOutputMaterial")
    vcol = nodes.new(type="ShaderNodeVertexColor")
    diffuse = nodes.new(type="ShaderNodeBsdfDiffuse")
    transp = nodes.new(type="ShaderNodeBsdfTransparent")
    mix = nodes.new(type="ShaderNodeMixShader")

    output.location = (600, 0)
    vcol.location = (0, 0)
    vcol.layer_name = obj.data.color_attributes[0].name
    diffuse.location = (200, 200)
    transp.location = (200, -200)
    mix.location = (400, 0)

    links.new(vcol.outputs["Color"], diffuse.inputs["Color"])
    links.new(vcol.outputs["Alpha"], mix.inputs["Fac"])
    links.new(transp.outputs["BSDF"], mix.inputs[1])
    links.new(diffuse.outputs["BSDF"], mix.inputs[2])
    links.new(mix.outputs["Shader"], output.inputs["Surface"])

    mat.blend_method = "BLEND"
    mat.shadow_method = "HASHED"

    obj.data.materials.clear()
    obj.data.materials.append(mat)


def unload():
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            bpy.data.objects.remove(obj, do_unlink=True)

    for mesh in bpy.data.meshes:
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)

    for material in bpy.data.materials:
        if material.users == 0:
            bpy.data.materials.remove(material)


def render_frame(output_dir, split, frame_idx, render_depth=True, render_normal=True, segmented=False):
    tree = bpy.context.scene.node_tree
    tree.nodes.clear()
    rl = tree.nodes.new("CompositorNodeRLayers")

    bpy.context.scene.frame_current = frame_idx

    id_mask = tree.nodes.new("CompositorNodeIDMask")
    id_mask.index = 1
    tree.links.new(rl.outputs["IndexOB"], id_mask.inputs["ID value"])

    if render_normal:
        add_normal_output(tree, rl, id_mask, output_dir, split)

    if render_depth:
        add_depth_output(tree, rl, id_mask, output_dir, split)

    add_rgb_output(tree, rl, id_mask, output_dir, split, segmented)
    bpy.ops.render.render(write_still=True)


def add_normal_output(tree, rl, id_mask, output_dir, split):
    normal_out = tree.nodes.new("CompositorNodeOutputFile")
    normal_out.base_path = output_dir
    normal_out.file_slots[0].path = f"normals/{split}_####"
    normal_out.format.file_format = "PNG"

    masked_normal = tree.nodes.new("CompositorNodeSetAlpha")
    tree.links.new(rl.outputs["Normal"], masked_normal.inputs[0])
    tree.links.new(id_mask.outputs["Alpha"], masked_normal.inputs[1])
    tree.links.new(masked_normal.outputs[0], normal_out.inputs[0])


# Render inverted normalized depth masked by object ID
def add_depth_output(tree, rl, id_mask, output_dir, split):
    normalize = tree.nodes.new("CompositorNodeNormalize")
    invert = tree.nodes.new("CompositorNodeInvert")
    invert.invert_rgb = True
    comb_rgb = tree.nodes.new("CompositorNodeCombRGBA")
    masked_depth = tree.nodes.new("CompositorNodeSetAlpha")
    depth_out = tree.nodes.new("CompositorNodeOutputFile")

    depth_out.base_path = output_dir
    depth_out.file_slots[0].path = f"depths/{split}_####"
    depth_out.format.file_format = "PNG"
    depth_out.format.color_mode = "RGBA"

    tree.links.new(rl.outputs["Depth"], normalize.inputs[0])
    tree.links.new(normalize.outputs[0], invert.inputs[1])
    for i in range(3):
        tree.links.new(invert.outputs[0], comb_rgb.inputs[i])
    tree.links.new(comb_rgb.outputs[0], masked_depth.inputs[0])
    tree.links.new(id_mask.outputs["Alpha"], masked_depth.inputs[1])
    tree.links.new(masked_depth.outputs[0], depth_out.inputs[0])


# Render RGB image masked by object ID
def add_rgb_output(tree, rl, id_mask, output_dir, split, segmented):
    set_alpha_rgb = tree.nodes.new("CompositorNodeSetAlpha")
    tree.links.new(rl.outputs["Image"], set_alpha_rgb.inputs[0])
    tree.links.new(id_mask.outputs["Alpha"], set_alpha_rgb.inputs[1])

    rgb_out = tree.nodes.new("CompositorNodeOutputFile")
    rgb_out.base_path = output_dir
    rgb_out.file_slots[0].path = f"segmentation_masks/{split}_####" if segmented else f"images/{split}_####"
    rgb_out.format.file_format = "PNG"
    rgb_out.format.color_mode = "RGBA"
    tree.links.new(set_alpha_rgb.outputs[0], rgb_out.inputs[0])


def get_circular_trajectory(num_views, radius):
    locations = []
    for i in range(num_views):
        angle = 2 * math.pi * i / num_views
        x = radius * math.sin(angle)
        y = 0.0
        z = radius * math.cos(angle)
        locations.append(Vector((x, y, z)))
    return locations


# Generate a spherical trajectory for the camera
def get_spherical_trajectory(num_views, radius):
    locations = []
    for _ in range(num_views):
        theta = math.acos(1 - 2 * random.random())
        phi = 2 * math.pi * random.random()
        x = radius * math.sin(theta) * math.cos(phi)
        y = radius * math.sin(theta) * math.sin(phi)
        z = radius * math.cos(theta)
        locations.append(Vector((x, y, z)))
    return locations


def quantize_mask(output_dir, split, frame_idx):
    segmentation_mask_path = os.path.join(output_dir, "segmentation_masks", f"{split}_{frame_idx:04d}.png")
    sementation_mask = np.array(Image.open(segmentation_mask_path))
    palette = np.array([0, 128, 255])
    quantized_rgb = palette[np.argmin(np.abs(sementation_mask[..., :3, None] - palette), axis=-1)]
    sementation_mask[..., :3] = quantized_rgb
    Image.fromarray(sementation_mask).save(segmentation_mask_path)


def main():
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :]

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", type=str, help="Subject folder name")
    parser.add_argument("--resolution", type=int, default=1024, help="Render resolution")
    parser.add_argument("--num_train_views", type=int, default=20, help="Number of training views")
    parser.add_argument("--num_test_views", type=int, default=13, help="Number of test views")
    parser.add_argument("--num_vis_views", type=int, default=360, help="Number of visualization views")
    parser.add_argument("--normals", action="store_true", help="Render normal maps if set")
    parser.add_argument("--depths", action="store_true", help="Render depth maps if set")
    parser.add_argument("--render_vis", action="store_true", help="Render visualization frames if set")
    args = parser.parse_args(argv)

    # Create folders
    subject_dir = os.path.join("data", "4D-DRESS", args.s)
    os.makedirs(os.path.join(subject_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(subject_dir, "depths"), exist_ok=True) if args.depths else None
    os.makedirs(os.path.join(subject_dir, "normals"), exist_ok=True) if args.normals else None
    os.makedirs(os.path.join(subject_dir, "segmentation_masks"), exist_ok=True)
    if args.render_vis:
        vis_gt_dir = os.path.join(subject_dir, "vis", "gt")
        os.makedirs(os.path.join(vis_gt_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(vis_gt_dir, "segmentation_masks"), exist_ok=True)

    # Scene
    setup_scene(resolution=args.resolution)

    # Trajectories
    trajectory_train = get_circular_trajectory(num_views=args.num_train_views, radius=1.5)
    trajectory_test = get_circular_trajectory(num_views=args.num_test_views, radius=1.5)
    trajectory_vis = get_circular_trajectory(num_views=args.num_vis_views, radius=1.5)

    # transforms_train.json
    transforms_train = {
        "camera_angle_x": bpy.data.objects["Camera"].data.angle_x,
        "height": args.resolution,
        "width": args.resolution,
        "frames": [],
    }
    for camera_location in trajectory_train:
        set_camera_position(camera_location)
        transforms_train["frames"].append(
            {
                "transform_matrix": [list(row) for row in bpy.data.objects["Camera"].matrix_world],
            }
        )

    # transforms_test.json
    transforms_test = {
        "camera_angle_x": bpy.data.objects["Camera"].data.angle_x,
        "height": args.resolution,
        "width": args.resolution,
        "frames": [],
    }
    for camera_location in trajectory_test:
        set_camera_position(camera_location)
        transforms_test["frames"].append(
            {
                "transform_matrix": [list(row) for row in bpy.data.objects["Camera"].matrix_world],
            }
        )

    # transforms_vis.json
    transforms_vis = {
        "camera_angle_x": bpy.data.objects["Camera"].data.angle_x,
        "height": args.resolution,
        "width": args.resolution,
        "frames": [],
    }
    for camera_location in trajectory_vis:
        set_camera_position(camera_location)
        transforms_vis["frames"].append(
            {
                "transform_matrix": [list(row) for row in bpy.data.objects["Camera"].matrix_world],
            }
        )

    # Render scan
    scan_path = os.path.join("./data", "4D-DRESS", args.s, "meshes", "scan.obj")
    configure_render_settings(segmented=False, resolution=args.resolution)
    load_obj(scan_path)

    for i, camera_location in enumerate(trajectory_train):
        set_camera_position(camera_location)
        render_frame(
            subject_dir, "train", i, render_depth=args.depths, render_normal=args.normals, segmented=False
        )
        transforms_train["frames"][i][f"image_path"] = f"./images/train_{i:04d}"
        if args.depths:
            transforms_train["frames"][i][f"depth_path"] = f"./depths/train_{i:04d}"
        if args.normals:
            transforms_train["frames"][i][f"normal_path"] = f"./normals/train_{i:04d}"

    for i, camera_location in enumerate(trajectory_test):
        set_camera_position(camera_location)
        render_frame(
            subject_dir, "test", i, render_depth=args.depths, render_normal=args.normals, segmented=False
        )
        transforms_test["frames"][i][f"image_path"] = f"./images/test_{i:04d}"
        if args.depths:
            transforms_test["frames"][i][f"depth_path"] = f"./depths/test_{i:04d}"
        if args.normals:
            transforms_test["frames"][i][f"normal_path"] = f"./normals/test_{i:04d}"

    for i, camera_location in enumerate(trajectory_vis):
        set_camera_position(camera_location)
        if args.render_vis:
            render_frame(vis_gt_dir, "vis", i, render_depth=False, render_normal=False, segmented=False)
            transforms_vis["frames"][i][f"image_path"] = f"./vis/gt/images/vis_{i:04d}"

    unload()

    # Render segmented scan
    segmented_scan_path = os.path.join("./data", "4D-DRESS", args.s, "meshes", "segmented_scan.ply")
    configure_render_settings(segmented=True, resolution=args.resolution)
    load_ply(segmented_scan_path, segmented=True)

    for i, camera_location in enumerate(trajectory_train):
        set_camera_position(camera_location)
        render_frame(subject_dir, "train", i, render_depth=False, render_normal=False, segmented=True)
        quantize_mask(subject_dir, "train", i)
        transforms_train["frames"][i][f"segmentation_mask_path"] = f"./segmentation_masks/train_{i:04d}"

    for i, camera_location in enumerate(trajectory_test):
        set_camera_position(camera_location)
        render_frame(subject_dir, "test", i, render_depth=False, render_normal=False, segmented=True)
        quantize_mask(subject_dir, "test", i)
        transforms_test["frames"][i][f"segmentation_mask_path"] = f"./segmentation_masks/test_{i:04d}"

    for i, camera_location in enumerate(trajectory_vis):
        set_camera_position(camera_location)
        if args.render_vis:
            render_frame(vis_gt_dir, "vis", i, render_depth=False, render_normal=False, segmented=True)
            quantize_mask(vis_gt_dir, "vis", i)
            transforms_vis["frames"][i][
                f"segmentation_mask_path"
            ] = f"./vis/gt/segmentation_masks/vis_{i:04d}"

    unload()

    # Save transforms
    with open(os.path.join(subject_dir, f"transforms_train.json"), "w") as f:
        json.dump(transforms_train, f, indent=4)
    with open(os.path.join(subject_dir, f"transforms_test.json"), "w") as f:
        json.dump(transforms_test, f, indent=4)
    with open(os.path.join(subject_dir, f"transforms_vis.json"), "w") as f:
        json.dump(transforms_vis, f, indent=4)


if __name__ == "__main__":
    main()
