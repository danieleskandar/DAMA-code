import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
import re
import torch
import subprocess
import numpy as np
from tqdm import tqdm
from plyfile import PlyData
from scipy.spatial.transform import Rotation as R
from utils.loss_utils import *
from gaussian_renderer import render
from arguments import get_animate_args
from torchvision.utils import save_image
from scene.smplx_model import SMPLXModel
from gaussian_models.gaussian_model import GaussianModel
from gaussian_models.smplx_gaussian_model import SMPLXGaussianModel
from gaussian_models.texture_gaussian_model import TextureGaussianModel
from scene.scene_loader import read_camera_from_tranforms, read_cameras_from_transforms, read_synthetic_info


def get_filenames(folder_path):
    pattern = re.compile(r"^\d+(?:_\d+)?\.ply$")
    return sorted(f for f in os.listdir(folder_path) if pattern.match(f))


def create_gif_from_images(folder, args, fps=30):
    input_pattern = os.path.join(folder, "frame_%06d.png")
    palette_path = os.path.join(folder, "palette.png")

    subject_name = os.path.basename(args.s)
    if not args.circular:
        gif_path = os.path.join(folder, f"{subject_name}_{args.n}_{args.cam_index}.gif")
    else:
        gif_path = os.path.join(folder, f"{subject_name}_{args.n}_circular.gif")

    subprocess.run(
        ["ffmpeg", "-y", "-framerate", str(fps), "-i", input_pattern, "-vf", "palettegen", palette_path]
    )

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            input_pattern,
            "-i",
            palette_path,
            "-filter_complex",
            "paletteuse",
            gif_path,
        ]
    )

    os.remove(palette_path)


def create_video_from_images(folder, args, fps=30):
    subject_name = os.path.basename(args.s)
    render_layers_suffix = "_layers" if args.render_layers else ""
    if not args.circular:
        video_path = os.path.join(
            folder, f"{subject_name}_{args.n}_{args.cam_index}{render_layers_suffix}.mp4"
        )
    else:
        video_path = os.path.join(folder, f"{subject_name}_{args.n}_circular{render_layers_suffix}.mp4")
    input_pattern = os.path.join(folder, "frame_%06d.png")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            input_pattern,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            video_path,
        ]
    )


def RT_to_transform(rotvec, trans):
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec(rotvec).as_matrix()
    T[:3, 3] = trans
    return T


def transform_to_RT(T):
    rotvec = R.from_matrix(T[:3, :3]).as_rotvec()
    trans = T[:3, 3]
    return rotvec, trans


def animate(args):
    background = torch.tensor([1, 1, 1]).float().cuda()

    # Create directories
    animation_dir = os.path.join(args.s, "animation", args.e)
    folder_dir = os.path.join(animation_dir, args.f)
    sequence_dir = os.path.join(folder_dir, args.n)
    camera_dir = os.path.join(sequence_dir, str(args.cam_index) if not args.circular else "circular")
    os.makedirs(camera_dir, exist_ok=True)

    # Load scene info
    scene_info = read_synthetic_info(args, read_cameras=False)

    # Initialize gaussians model
    gaussians = GaussianModel()

    # Load SMPLX gaussians
    smplx_gaussians = SMPLXGaussianModel()
    smplx_gaussians.load_ply(
        scene_info.smplx_gaussians, scene_info.segmentation_gaussians, invisible_only=True
    )

    # Load texture gaussians
    folder_path = os.path.join(args.s, "gaussians", args.e, args.f)
    filenames = get_filenames(folder_path)
    texture_gaussians = []
    for filename in filenames:
        texture_gaussians_path = os.path.join(folder_path, filename)
        texture_gaussians_ply = PlyData.read(texture_gaussians_path).elements[0]
        texture_gaussians_model = TextureGaussianModel()
        texture_gaussians_model.load_from_texture_ply(
            scene_info.smplx_mesh, scene_info.smplx_gaussians, texture_gaussians_ply, None, args
        )
        texture_gaussians.append(texture_gaussians_model)

    # Load SMPLX model
    smplx_model = SMPLXModel(
        scene_info.basic_info,
        scene_info.smplx_params,
        scene_info.normalization,
        args.use_subdivided_mesh,
    )
    # smplx_model.use_full_hand_pose()

    # Read transform_vis.json
    if args.circular:
        print("Reading transform_vis.json")
        cameras = read_cameras_from_transforms(args.s, "transforms_vis.json")
    else:
        camera = read_camera_from_tranforms(args.s, "transforms_vis.json", args.cam_index)

    # Load motion sequence
    motion_sequence = np.load(args.m)

    # Reference transforms
    rotvec_ref = scene_info.smplx_params["global_orient"]
    trans_ref = scene_info.smplx_params["transl"]
    T_ref = RT_to_transform(rotvec_ref, trans_ref)

    # Rest transforms
    rotvec_rest = motion_sequence["root_orient"][0]
    trans_rest = motion_sequence["trans"][0]
    T_rest = RT_to_transform(rotvec_rest, trans_rest)
    T_rest_inv = np.linalg.inv(T_rest)

    num_frames = motion_sequence["trans"].shape[0]
    for frame in tqdm(range(num_frames), desc="Animating"):
        rotvec_frame = motion_sequence["root_orient"][frame]
        trans_frame = motion_sequence["trans"][frame]
        T_frame = RT_to_transform(rotvec_frame, trans_frame)

        T = T_ref @ T_rest_inv @ T_frame
        rotvec, trans = transform_to_RT(T)

        smplx_model.set_params(
            {"body_pose": motion_sequence["pose_body"][frame], "global_orient": rotvec, "transl": trans}
        )

        pose_properties = smplx_model.get_properties()

        smplx_gaussians.update(pose_properties)
        for g in texture_gaussians:
            g.update(pose_properties)

        if args.circular:
            camera = cameras[int((frame / num_frames) * len(cameras)) % len(cameras)]

        gaussians.merge(texture_gaussians)
        full_render = render(camera, gaussians, args, background)["render"]

        if args.render_layers:
            layer_renders = []
            for g in texture_gaussians[2:]:
                layer_renders.append(render(camera, g, args, background)["render"])
            gaussians.merge([smplx_gaussians, texture_gaussians[0], texture_gaussians[1]])
            layer_renders.append(render(camera, gaussians, args, background)["render"])
            combined = torch.cat([full_render, *layer_renders], dim=2)
            save_image(combined, os.path.join(camera_dir, f"frame_{frame:06d}.png"))
        else:
            save_image(full_render, os.path.join(camera_dir, f"frame_{frame:06d}.png"))

    create_video_from_images(camera_dir, args, fps=60)


def main():
    args = get_animate_args()
    animate(args)


if __name__ == "__main__":
    main()
