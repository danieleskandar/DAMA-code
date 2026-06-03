import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
import cv2
import torch
import subprocess
import numpy as np
from tqdm import tqdm
from random import randint
from utils.loss_utils import *
from utils.image_utils import psnr
from gaussian_renderer import render
from arguments import get_train_texture_args
from scene.texture_scene import TextureScene
from utils.vis_utils import tensor_to_cv2, overlay_text
from scene.scene_loader import read_cameras_from_transforms


def save(
    frame,
    label,
    gt_image,
    labels_with_smplx_image,
    label_with_smplx_image,
    label_image,
    smplx_image,
    train_vis_dir,
    full=False,
):
    images = [
        overlay_text(tensor_to_cv2(gt_image), "Ground Truth"),
        overlay_text(tensor_to_cv2(labels_with_smplx_image), "Labels + SMPL-X"),
        overlay_text(tensor_to_cv2(label_with_smplx_image), f"Label {label} + SMPL-X" if not full else ""),
        overlay_text(tensor_to_cv2(label_image), f"Label {label}" if not full else ""),
        overlay_text(tensor_to_cv2(smplx_image), "SMPL-X"),
    ]

    cv2.imwrite(f"{train_vis_dir}/frame_{frame:06d}.png", np.concatenate(images, axis=1))


def create_training_video(train_vis_dir, args, fps=10):
    video_path = os.path.join(train_vis_dir, f"{os.path.basename(args.s)}_train_texture.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            f"{train_vis_dir}/frame_%06d.png",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            video_path,
        ]
    )


def train_labels(args, scene, train_vis_dir, vis_cams, frame):
    scene.training_setup(args)

    if args.vis and not args.circular:
        vis_cam = scene.get_train_cameras()[0]

    gaussians = scene.gaussians
    smplx_gaussians = scene.smplx_gaussians
    texture_gaussians = scene.texture_gaussians

    black_background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    white_background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    viewpoint_stack = None

    print("Training labels")
    for label in scene.labels:
        hair = label == 1
        skin_or_hair = label == 0 or label == 1

        ema_total_loss_for_log = 0.0
        ema_color_loss_for_log = 0.0
        ema_mask_loss_for_log = 0.0
        ema_normal_loss_for_log = 0.0
        ema_max_scale_loss_for_log = 0.0
        ema_anisotropic_loss_for_log = 0.0
        ema_canonical_distance_loss_for_log = 0.0
        ema_canonical_rotation_loss_for_log = 0.0

        progress_bar = tqdm(range(0, args.num_iterations), desc=str(label))
        for iteration in range(1, args.num_iterations + 1):
            background = torch.rand(3).float().cuda()

            if not viewpoint_stack:
                viewpoint_stack = scene.get_train_cameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

            # GT mask and image
            gt_mask = (
                (viewpoint_cam.segmentation.permute(1, 2, 0) == scene.label_colors[label])
                .all(dim=-1)
                .unsqueeze(0)
                .float()
            )
            gt_image = gt_mask * viewpoint_cam.image + (1.0 - gt_mask) * background.view(3, 1, 1).expand_as(
                viewpoint_cam.image
            )

            # Render mask
            gaussians.merge(
                [texture_gaussians[label]] + [scene.texture_gaussians[i] for i in scene.labels if i != label]
            )
            num_white_gaussians = texture_gaussians[label].num_gaussians
            num_black_gaussians = sum(
                [texture_gaussians[i].num_gaussians for i in scene.labels if i != label]
            )
            override_color = torch.cat(
                [torch.ones(num_white_gaussians, 3).cuda(), torch.zeros(num_black_gaussians, 3).cuda()], dim=0
            )
            render_pkg = render(
                viewpoint_cam, gaussians, args, black_background, override_color=override_color
            )
            render_mask = render_pkg["render"]

            # Render image
            smplx_gaussians.set_random_color()
            gaussians.merge([texture_gaussians[label], smplx_gaussians])
            render_pkg = render(viewpoint_cam, gaussians, args, background)
            render_image = gt_mask * render_pkg["render"] + (1.0 - gt_mask) * background.view(
                3, 1, 1
            ).expand_as(render_pkg["render"])

            # Losses
            L_color = args.lambda_color * l1_loss(gt_image, render_image)
            L_mask = args.lambda_mask * l1_loss(gt_mask, render_mask)
            L_normal = args.lambda_normal * (
                (1 - (render_pkg["rend_normal"] * render_pkg["surf_normal"]).sum(dim=0))[None].mean()
            )
            L_anisotropic = (
                args.lambda_anisotropic
                * anisotropic_loss(texture_gaussians[label].get_scaling, r=args.anisotropic_r)
                if skin_or_hair
                else torch.tensor(0)
            )
            L_max_scale = (
                args.lambda_max_scale
                * max_scale_loss(texture_gaussians[label].get_scaling, max_scale=args.max_scale)
                if not skin_or_hair
                else torch.tensor(0)
            )

            L_canonical_distance = (
                args.lambda_canonical_distance
                * canonical_distance_loss(
                    canonical_triangle_vertices=texture_gaussians[label].canonical_triangle_vertices,
                    canonical_xyz=texture_gaussians[label].get_canonical_xyz,
                )
                if not hair
                else torch.tensor(0)
            )

            L_canonical_rotation = (
                args.lambda_canonical_rotation
                * canonical_rotation_loss(
                    canonical_base_rotation=texture_gaussians[label].canonical_base_rotation,
                    canonical_rotation=texture_gaussians[label].get_canonical_rotation,
                )
                if not hair
                else torch.tensor(0)
            )

            L_total = (
                L_color
                + L_mask
                + L_normal
                + L_anisotropic
                + L_max_scale
                + L_canonical_distance
                + L_canonical_rotation
            )

            L_total.backward()

            with torch.no_grad():
                ema_total_loss_for_log = 0.4 * L_total.item() + 0.6 * ema_total_loss_for_log
                ema_color_loss_for_log = 0.4 * L_color.item() + 0.6 * ema_color_loss_for_log
                ema_mask_loss_for_log = 0.4 * L_mask.item() + 0.6 * ema_mask_loss_for_log
                ema_normal_loss_for_log = 0.4 * L_normal.item() + 0.6 * ema_normal_loss_for_log
                ema_anisotropic_loss_for_log = 0.4 * L_anisotropic.item() + 0.6 * ema_anisotropic_loss_for_log
                ema_max_scale_loss_for_log = 0.4 * L_max_scale.item() + 0.6 * ema_max_scale_loss_for_log
                ema_canonical_distance_loss_for_log = (
                    0.4 * L_canonical_distance.item() + 0.6 * ema_canonical_distance_loss_for_log
                )
                ema_canonical_rotation_loss_for_log = (
                    0.4 * L_canonical_rotation.item() + 0.6 * ema_canonical_rotation_loss_for_log
                )

                if iteration % 10 == 0:
                    loss_dict = {
                        "T": f"{ema_total_loss_for_log:.5f}",
                        "C": f"{ema_color_loss_for_log:.5f}",
                        "M": f"{ema_mask_loss_for_log:.5f}",
                        "N": f"{ema_normal_loss_for_log:.5f}",
                        "A": f"{ema_anisotropic_loss_for_log:.5f}" if skin_or_hair else f"N/A",
                        "S": f"{ema_max_scale_loss_for_log:.5f}" if not skin_or_hair else f"N/A",
                        "Cd": f"{ema_canonical_distance_loss_for_log:.5f}" if not hair else f"N/A",
                        "Cr": f"{ema_canonical_rotation_loss_for_log:.5f}" if not hair else f"N/A",
                    }
                    progress_bar.set_postfix(loss_dict)
                    progress_bar.update(10)

                if iteration == args.num_iterations:
                    progress_bar.close()

                if args.vis and iteration % args.vis_freq == 0:
                    if args.circular:
                        vis_cam = vis_cams[
                            int((iteration / args.num_iterations) * len(vis_cams)) % len(vis_cams)
                        ]
                    gt_image = vis_cam.mask * vis_cam.image + (1 - vis_cam.mask) * white_background.view(
                        3, 1, 1
                    ).expand_as(vis_cam.image)
                    gaussians.merge([texture_gaussians[label] for label in scene.labels] + [smplx_gaussians])
                    labels_with_smplx_image = render(vis_cam, gaussians, args, white_background)["render"]
                    gaussians.merge([texture_gaussians[label], smplx_gaussians])
                    label_with_smplx_image = render(vis_cam, gaussians, args, white_background)["render"]
                    label_image = render(vis_cam, texture_gaussians[label], args, white_background)["render"]
                    smplx_image = render(vis_cam, smplx_gaussians, args, white_background)["render"]
                    save(
                        frame[0],
                        label,
                        gt_image,
                        labels_with_smplx_image,
                        label_with_smplx_image,
                        label_image,
                        smplx_image,
                        train_vis_dir,
                    )
                    frame[0] += 1

                if iteration < args.num_iterations:
                    texture_gaussians[label].optimizer.step()
                    texture_gaussians[label].optimizer.zero_grad(set_to_none=True)


def train_full(args, scene, train_vis_dir, vis_cams, frame):
    scene.training_setup(args)

    if args.vis and not args.circular:
        vis_cam = scene.get_train_cameras()[0]

    gaussians = scene.gaussians
    smplx_gaussians = scene.smplx_gaussians
    texture_gaussians = scene.texture_gaussians

    # Disable color and scale optimization
    for label in scene.labels:
        texture_gaussians[label]._features_dc.requires_grad = False
        texture_gaussians[label]._scaling.requires_grad = False

    viewpoint_stack = None

    ema_total_loss_for_log = 0.0
    ema_color_loss_for_log = 0.0

    print("Training full")
    progress_bar = tqdm(range(0, args.num_iterations), desc="F")
    for iteration in range(1, args.num_iterations + 1):
        background = torch.rand(3).float().cuda()
        white_background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

        if not viewpoint_stack:
            viewpoint_stack = scene.get_train_cameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        smplx_gaussians.set_random_color()
        gaussians.merge([texture_gaussians[label] for label in scene.labels] + [smplx_gaussians])
        render_pkg = render(viewpoint_cam, gaussians, args, background)

        # GT image
        gt_image = viewpoint_cam.image * viewpoint_cam.mask + (1 - viewpoint_cam.mask) * background.view(
            3, 1, 1
        ).expand_as(viewpoint_cam.image)

        L_color = args.lambda_color * l1_loss(gt_image, render_pkg["render"])
        L_total = L_color

        L_total.backward(retain_graph=True)

        with torch.no_grad():
            ema_total_loss_for_log = 0.4 * L_total.item() + 0.6 * ema_total_loss_for_log
            ema_color_loss_for_log = 0.4 * L_color.item() + 0.6 * ema_color_loss_for_log

            if iteration % 10 == 0:
                loss_dict = {
                    "T": f"{ema_total_loss_for_log:.5f}",
                    "C": f"{ema_color_loss_for_log:.5f}",
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)

            if iteration == args.num_iterations:
                progress_bar.close()

            if args.vis and iteration % args.vis_freq == 0:
                if args.circular:
                    vis_cam = vis_cams[int((iteration / args.num_iterations) * len(vis_cams)) % len(vis_cams)]
                gt_image = vis_cam.mask * vis_cam.image + (1 - vis_cam.mask) * white_background.view(
                    3, 1, 1
                ).expand_as(vis_cam.image)
                gaussians.merge([texture_gaussians[label] for label in scene.labels] + [smplx_gaussians])
                labels_with_smplx_image = render(vis_cam, gaussians, args, white_background)["render"]
                label_with_smplx_image = white_background.view(3, 1, 1).expand_as(vis_cam.image)
                label_image = white_background.view(3, 1, 1).expand_as(vis_cam.image)
                smplx_image = render(vis_cam, smplx_gaussians, args, white_background)["render"]
                save(
                    frame[0],
                    label,
                    gt_image,
                    labels_with_smplx_image,
                    label_with_smplx_image,
                    label_image,
                    smplx_image,
                    train_vis_dir,
                    full=True,
                )
                frame[0] += 1

            if iteration < args.num_iterations:
                for label in scene.labels:
                    if label != 0 and label != 1:
                        texture_gaussians[label].optimizer.step()
                        texture_gaussians[label].optimizer.zero_grad(set_to_none=True)


def main():
    args = get_train_texture_args()
    scene = TextureScene(args, shuffle=not args.vis)

    if args.vis:
        train_vis_dir = os.path.join(args.s, "training", "texture")
        os.makedirs(train_vis_dir, exist_ok=True)
        vis_cams = read_cameras_from_transforms(args.s, "transforms_vis.json") if args.circular else None
        frame = [0]
    else:
        train_vis_dir = vis_cams = frame = None

    train_labels(args, scene, train_vis_dir, vis_cams, frame)
    train_full(args, scene, train_vis_dir, vis_cams, frame)

    scene.save("posed")
    scene.unpose_smplx()
    scene.save("canonical")

    if args.vis:
        train_vis_dir = os.path.join(args.s, "training", "texture")
        create_training_video(train_vis_dir, args, fps=10)


if __name__ == "__main__":
    main()
