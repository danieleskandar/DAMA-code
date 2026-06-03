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
from gaussian_renderer import render
from arguments import get_train_segmentation_args
from scene.segmentation_scene import SegmentationScene
from utils.vis_utils import tensor_to_cv2, overlay_text
from utils.loss_utils import l1_loss, label_smoothness_loss
from scene.scene_loader import read_cameras_from_transforms


def save(frame, gt_image, segmentation_with_smplx_image, segmentation_image, smplx_image, train_vis_dir):
    images = [
        overlay_text(tensor_to_cv2(gt_image), "Ground Truth"),
        overlay_text(tensor_to_cv2(segmentation_with_smplx_image), "Segmentation Gaussians + SMPL-X"),
        overlay_text(tensor_to_cv2(segmentation_image), "Segmentation Gaussians"),
        overlay_text(tensor_to_cv2(smplx_image), "SMPL-X"),
    ]

    cv2.imwrite(f"{train_vis_dir}/frame_{frame:06d}.png", np.concatenate(images, axis=1))


def create_training_video(train_vis_dir, args, fps=15):
    circular_suffix = "_circular" if args.circular else ""
    video_path = os.path.join(
        train_vis_dir, f"{os.path.basename(args.s)}_train_segmentation{circular_suffix}.mp4"
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


def train(args):
    scene = SegmentationScene(args, shuffle=not args.vis)
    scene.training_setup(args)

    gaussians = scene.gaussians
    smplx_gaussians = scene.smplx_gaussians
    segmentation_gaussians = scene.segmentation_gaussians

    black_background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    white_background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    if args.vis:
        train_vis_dir = os.path.join(args.s, "training", "segmentation")
        os.makedirs(train_vis_dir, exist_ok=True)
        if args.circular:
            vis_cams = read_cameras_from_transforms(args.s, "transforms_vis.json")
        else:
            vis_cam = scene.get_train_cameras()[0]
        frame = 0

    viewpoint_stack = None

    ema_total_loss_for_log = 0.0
    ema_color_loss_for_log = 0.0
    ema_scaling_loss_for_log = 0.0
    ema_normal_loss_for_log = 0.0
    ema_label_smoothness_loss_for_log = 0.0

    print("Training segmentation gaussians")
    progress_bar = tqdm(range(0, args.num_iterations))
    for iteration in range(1, args.num_iterations + 1):
        segmentation_gaussians.update_learning_rate(iteration)

        if not viewpoint_stack:
            viewpoint_stack = scene.get_train_cameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render smplx and segmentation gaussians
        smplx_gaussians.set_random_color()
        gaussians.merge([segmentation_gaussians, smplx_gaussians])
        render_pkg = render(viewpoint_cam, gaussians, args, black_background)

        lambda_label_smoothness = (
            args.lambda_label_smoothness if iteration > args.label_smoothness_start else 0
        )

        L_color = args.lambda_color * l1_loss(viewpoint_cam.segmentation, render_pkg["render"])
        L_scaling = args.lambda_scaling * l1_loss(
            segmentation_gaussians._scaling, segmentation_gaussians.init_scaling
        )
        L_normal = args.lambda_normal * (
            (1 - (render_pkg["rend_normal"] * render_pkg["surf_normal"]).sum(dim=0))[None].mean()
        )
        L_label_smoothness = lambda_label_smoothness * label_smoothness_loss(
            segmentation_gaussians._logits, segmentation_gaussians.neighbors
        )

        L_total = L_color + L_scaling + L_normal + L_label_smoothness

        L_total.backward()

        with torch.no_grad():
            ema_total_loss_for_log = 0.4 * L_total.item() + 0.6 * ema_total_loss_for_log
            ema_color_loss_for_log = 0.4 * L_color.item() + 0.6 * ema_color_loss_for_log
            ema_scaling_loss_for_log = 0.4 * L_scaling.item() + 0.6 * ema_scaling_loss_for_log
            ema_normal_loss_for_log = 0.4 * L_normal.item() + 0.6 * ema_normal_loss_for_log
            ema_label_smoothness_loss_for_log = (
                0.4 * L_label_smoothness.item() + 0.6 * ema_label_smoothness_loss_for_log
            )

            if iteration % 10 == 0:
                loss_dict = {
                    "T": f"{ema_total_loss_for_log:.{5}f}",
                    "C": f"{ema_color_loss_for_log:.{5}f}",
                    "S": f"{ema_scaling_loss_for_log:.{5}f}",
                    "N": f"{ema_normal_loss_for_log:.{5}f}",
                    "L": f"{ema_label_smoothness_loss_for_log:.{5}f}",
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)

            if iteration == args.num_iterations:
                progress_bar.close()

            if args.vis and iteration % args.vis_freq == 0:
                if args.circular:
                    vis_cam = vis_cams[int((iteration / args.num_iterations) * len(vis_cams)) % len(vis_cams)]
                gt_image = vis_cam.mask * vis_cam.segmentation + (1 - vis_cam.mask) * white_background.view(
                    3, 1, 1
                ).expand_as(vis_cam.image)
                segmentation_with_smplx_image = render(vis_cam, gaussians, args, white_background)["render"]
                segmentation_image = render(vis_cam, segmentation_gaussians, args, white_background)["render"]
                smplx_image = render(vis_cam, smplx_gaussians, args, white_background)["render"]
                save(
                    frame,
                    gt_image,
                    segmentation_with_smplx_image,
                    segmentation_image,
                    smplx_image,
                    train_vis_dir,
                )
                frame += 1

            if iteration < args.num_iterations:
                segmentation_gaussians.optimizer.step()
                segmentation_gaussians.optimizer.zero_grad(set_to_none=True)

    segmentation_gaussians.refine_labels(area_threshold=args.area_threshold)
    scene.save_posed()
    scene.unpose_smplx()
    scene.save_canonical()

    if args.vis:
        create_training_video(train_vis_dir, args, fps=15)


def main():
    args = get_train_segmentation_args()
    train(args)


if __name__ == "__main__":
    main()
