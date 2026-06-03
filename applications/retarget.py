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
from gaussian_renderer import render
from arguments import get_retarget_args
from scene.retarget_scene import RetargetScene
from utils.vis_utils import tensor_to_cv2, overlay_text


def save(
    frame,
    label,
    all_layers_image,
    reposed_layer_image,
    label_with_previous_layers_image,
    label_image,
    previous_layers_image,
    train_vis_dir,
):
    images = [
        overlay_text(tensor_to_cv2(all_layers_image), "All Layers"),
        overlay_text(tensor_to_cv2(reposed_layer_image), f"Reposed Layer {label} (Supervision)"),
        overlay_text(tensor_to_cv2(label_with_previous_layers_image), f"Layer {label} + Previous Layers"),
        overlay_text(tensor_to_cv2(label_image), f"Layer {label}"),
        overlay_text(tensor_to_cv2(previous_layers_image), "Previous Layers"),
    ]

    cv2.imwrite(f"{train_vis_dir}/frame_{frame:06d}.png", np.concatenate(images, axis=1))


def create_training_video(train_vis_dir, args, fps=10):
    circular_suffix = "_circular" if args.circular else ""
    video_path = os.path.join(
        train_vis_dir, f"{os.path.basename(args.s)}_train_retargeting{circular_suffix}.mp4"
    )
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


def resolve_layer(args, scene, label, label_index, max_offset_per_smplx_face, train_vis_dir, vis_cams, frame):
    print(f"Resolving {scene.label_names[label]}")

    scene.training_setup(args)

    if args.vis and not args.circular:
        vis_cam = scene.get_train_cameras()[0]

    gaussians = scene.gaussians
    smplx_gaussians = scene.smplx_gaussians
    full_smplx_gaussians = scene.full_smplx_gaussians
    texture_gaussians = scene.texture_gaussians

    black_background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    white_background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    # Garment masks and images
    with torch.no_grad():
        for viewpoint_cam in scene.get_train_cameras():
            # Render garment mask
            gaussians.merge([texture_gaussians[label], full_smplx_gaussians])
            num_white_gaussians = texture_gaussians[label].num_gaussians
            num_black_gaussians = full_smplx_gaussians.num_gaussians
            override_color = torch.cat(
                [torch.ones(num_white_gaussians, 3).cuda(), torch.zeros(num_black_gaussians, 3).cuda()], dim=0
            )
            garment_mask = render(
                viewpoint_cam, gaussians, args, black_background, override_color=override_color
            )["render"]
            garment_mask = (garment_mask > 0.1).float()
            viewpoint_cam.garment_mask = garment_mask

            # Render garment image
            gaussians.merge([texture_gaussians[label]])
            garment_image = render(viewpoint_cam, gaussians, args, black_background)["render"]
            viewpoint_cam.garment_image = garment_image

    if args.resolve_offsets:
        texture_gaussians[label].update_previous_layers_offset(max_offset_per_smplx_face)

    viewpoint_stack = None

    ema_total_loss_for_log = 0.0
    ema_color_loss_for_log = 0.0
    ema_mask_loss_for_log = 0.0

    progress_bar = tqdm(range(0, args.num_iterations))
    for iteration in range(1, args.num_iterations + 1):
        if not viewpoint_stack:
            viewpoint_stack = scene.get_train_cameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # GT mask and image
        gt_mask = viewpoint_cam.garment_mask
        gt_image = viewpoint_cam.garment_image

        # Render mask
        label_index = args.layer_order.index(label)
        gaussians.merge(
            [texture_gaussians[label], texture_gaussians[0], texture_gaussians[1], smplx_gaussians]
            + [texture_gaussians[l] for l in args.layer_order[:label_index]]
        )
        num_white_gaussians = texture_gaussians[label].num_gaussians
        num_black_gaussians = (
            texture_gaussians[0].num_gaussians
            + texture_gaussians[1].num_gaussians
            + smplx_gaussians.num_gaussians
            + sum([texture_gaussians[l].num_gaussians for l in args.layer_order[:label_index]])
        )
        override_color = torch.cat(
            [torch.ones(num_white_gaussians, 3).cuda(), torch.zeros(num_black_gaussians, 3).cuda()], dim=0
        )
        render_pkg = render(viewpoint_cam, gaussians, args, black_background, override_color=override_color)
        render_mask = render_pkg["render"]

        # Render image
        render_pkg = render(viewpoint_cam, gaussians, args, black_background)
        render_image = gt_mask * render_pkg["render"]

        # Losses
        L_color = args.lambda_color * l1_loss(gt_image, render_image)
        L_mask = args.lambda_mask * l1_loss(gt_mask, render_mask)

        L_total = L_color + L_mask

        L_total.backward()

        with torch.no_grad():
            ema_total_loss_for_log = 0.4 * L_total.item() + 0.6 * ema_total_loss_for_log
            ema_color_loss_for_log = 0.4 * L_color.item() + 0.6 * ema_color_loss_for_log
            ema_mask_loss_for_log = 0.4 * L_mask.item() + 0.6 * ema_mask_loss_for_log

            if iteration % 10 == 0:
                loss_dict = {
                    "T": f"{ema_total_loss_for_log:.5f}",
                    "C": f"{ema_color_loss_for_log:.5f}",
                    "M": f"{ema_mask_loss_for_log:.5f}",
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)

            if iteration == args.num_iterations:
                progress_bar.close()

            if args.vis and iteration % args.vis_freq == 0:
                if args.circular:
                    vis_cam = vis_cams[int((iteration / args.num_iterations) * len(vis_cams)) % len(vis_cams)]
                gaussians.merge(
                    [texture_gaussians[l] for l in scene.texture_gaussians.keys()] + [smplx_gaussians]
                )
                all_layers_image = render(vis_cam, gaussians, args, white_background)["render"]
                reposed_layer_image = vis_cam.garment_mask * vis_cam.garment_image + (
                    1 - vis_cam.garment_mask
                ) * white_background.view(3, 1, 1).expand_as(vis_cam.image)
                gaussians.merge(
                    [texture_gaussians[0], smplx_gaussians]
                    + [texture_gaussians[l] for l in args.layer_order[: label_index + 1]]
                )
                label_with_previous_layers_image = render(vis_cam, gaussians, args, white_background)[
                    "render"
                ]
                label_image = render(vis_cam, texture_gaussians[label], args, white_background)["render"]
                gaussians.merge(
                    [texture_gaussians[0], smplx_gaussians]
                    + [texture_gaussians[l] for l in args.layer_order[:label_index]]
                )
                previous_layers_image = render(vis_cam, gaussians, args, white_background)["render"]
                save(
                    frame[0],
                    label,
                    all_layers_image,
                    reposed_layer_image,
                    label_with_previous_layers_image,
                    label_image,
                    previous_layers_image,
                    train_vis_dir,
                )
                frame[0] += 1

            if iteration < args.num_iterations:
                texture_gaussians[label].optimizer.step()
                texture_gaussians[label].optimizer.zero_grad(set_to_none=True)


def resolve_layers(args, scene, train_vis_dir, vis_cams, frame):
    args.layer_order = [label for label in args.layer_order if label in scene.texture_gaussians.keys()]

    if len(args.layer_order) < 2:
        return

    # Initialize face set and max offset tensor
    previous_layers_faces = set()
    max_offset_per_smplx_face = torch.zeros(scene.num_smplx_faces, 1).cuda()

    # Update with skin layer
    previous_layers_faces.update(set(scene.texture_gaussians[0].face_indices.cpu().numpy()))
    face_indices, max_offset_per_layer_face = scene.texture_gaussians[0].get_max_offset_per_face()
    max_offset_per_smplx_face[face_indices] = torch.maximum(
        max_offset_per_smplx_face[face_indices], max_offset_per_layer_face
    )

    for label_index, label in enumerate(args.layer_order):
        label_gaussians = scene.texture_gaussians[label]

        label_faces = set(label_gaussians.face_indices.cpu().numpy())

        intersects_previous_layers = bool(label_faces & previous_layers_faces)
        if intersects_previous_layers:
            if args.gaussian_opt:
                resolve_layer(
                    args, scene, label, label_index, max_offset_per_smplx_face, train_vis_dir, vis_cams, frame
                )
            elif args.resolve_offsets:
                label_gaussians.update_previous_layers_offset(max_offset_per_smplx_face)

        previous_layers_faces.update(label_faces)
        face_indices, max_offset_per_layer_face = label_gaussians.get_max_offset_per_face()
        max_offset_per_smplx_face[face_indices] = torch.maximum(
            max_offset_per_smplx_face[face_indices], max_offset_per_layer_face
        )


def main():
    args = get_retarget_args()
    scene = RetargetScene(args, shuffle=not args.vis)

    if args.vis and args.gaussian_opt:
        train_vis_dir = os.path.join(args.s, "training", "retargeting")
        os.makedirs(train_vis_dir, exist_ok=True)
        vis_cams = scene.get_train_cameras() if args.circular else None
        frame = [0]
    else:
        train_vis_dir = vis_cams = frame = None

    resolve_layers(args, scene, train_vis_dir, vis_cams, frame)

    scene.save()

    if args.vis and args.gaussian_opt:
        create_training_video(train_vis_dir, args, fps=10)


if __name__ == "__main__":
    main()
