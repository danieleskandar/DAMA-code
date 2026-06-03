import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
import cv2
import torch
import subprocess
import numpy as np
from tqdm import tqdm
from plyfile import PlyData
from gaussian_renderer import render
from arguments import get_evaluation_args
import torchvision.transforms.functional as TF
from gaussian_models.gaussian_model import GaussianModel
from utils.vis_utils import tensor_to_cv2, overlay_text
from scene.scene_loader import read_cameras_from_transforms


def create_gif_from_images(folder, args, fps=30):
    subject_name = os.path.basename(args.s)
    folder_name = os.path.basename(folder.rstrip("/"))
    input_pattern = os.path.join(folder, "vis_%04d.png")
    palette_path = os.path.join(folder, "palette.png")
    gif_path = os.path.join(folder, f"{subject_name}_{folder_name}.gif")

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
    folder_name = os.path.basename(folder.rstrip("/"))
    video_path = os.path.join(folder, f"{subject_name}_{folder_name}.mp4")
    input_pattern = os.path.join(folder, "vis_%04d.png")
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


def visualize(args):
    background = torch.tensor([1, 1, 1]).float().cuda()

    # Initialize segmenation gaussians
    segmented_scan_gaussians = GaussianModel()
    segmented_smplx_gaussians = GaussianModel()

    # Initialize texture gaussians
    full_with_smplx_gaussians = GaussianModel()
    full_gaussians = GaussianModel()
    clothing_gaussians = GaussianModel()
    body_gaussians = GaussianModel()

    # Initialize garment gaussians
    unique_labels = np.unique(
        PlyData.read(os.path.join(args.s, "gaussians", args.e, "posed", "full_segmented.ply")).elements[0][
            "refined_labels"
        ]
    )
    garment_labels = [str(int(label)) for label in unique_labels if int(label) not in [0, 1]]
    garment_gaussians = [GaussianModel() for _ in range(len(garment_labels))]
    garments_with_smplx_gaussians = [GaussianModel() for _ in range(len(garment_labels))]

    # Initialize canonical gaussians
    canonical_full_with_smplx_gaussians = GaussianModel()
    canonical_garment_gaussians = [GaussianModel() for _ in range(len(garment_labels))]
    canonical_body_gaussians = GaussianModel()

    # Load segmentation gaussians
    segmented_scan_gaussians.load_ply(
        os.path.join(args.s, "gaussians", args.e, "posed", "full_segmented.ply")
    )
    if os.path.exists(os.path.join(args.s, "gaussians", args.e, "posed", "smplx_subdivided.ply")):
        segmented_smplx_gaussians.load_ply(
            os.path.join(args.s, "gaussians", args.e, "posed", "smplx_subdivided.ply")
        )
    else:
        segmented_smplx_gaussians.load_ply(os.path.join(args.s, "gaussians", args.e, "posed", "smplx.ply"))
    segmented_smplx_gaussians._features_dc = segmented_scan_gaussians._features_dc

    # Load texture gaussians
    full_with_smplx_gaussians.load_ply(os.path.join(args.s, "gaussians", args.e, "posed", "full+smplx.ply"))
    full_gaussians.load_ply(os.path.join(args.s, "gaussians", args.e, "posed", "full.ply"))
    clothing_gaussians.load_ply(os.path.join(args.s, "gaussians", args.e, "posed", "clothing.ply"))
    body_gaussians.load_ply(os.path.join(args.s, "gaussians", args.e, "posed", "body.ply"))

    # Load garment gaussians
    for i in range(len(garment_labels)):
        garment_gaussians[i].load_ply(
            os.path.join(args.s, "gaussians", args.e, "posed", f"{garment_labels[i]}.ply")
        )
        garments_with_smplx_gaussians[i].load_ply(
            os.path.join(args.s, "gaussians", args.e, "posed", f"{garment_labels[i]}+body.ply")
        )

    # Load canonical gaussians
    canonical_full_with_smplx_gaussians.load_ply(
        os.path.join(args.s, "gaussians", args.e, "canonical", "full+smplx.ply")
    )
    for i in range(len(garment_labels)):
        canonical_garment_gaussians[i].load_ply(
            os.path.join(args.s, "gaussians", args.e, "canonical", f"{garment_labels[i]}.ply")
        )
    canonical_body_gaussians.load_ply(os.path.join(args.s, "gaussians", args.e, "canonical", "body.ply"))

    # Create directories
    segmentation_dir = os.path.join(args.s, "vis", args.e, "segmentation")
    texture_dir = os.path.join(args.s, "vis", args.e, "texture")
    garments_dir = os.path.join(args.s, "vis", args.e, "garments")
    canonical_dir = os.path.join(args.s, "vis", args.e, "canonical")

    os.makedirs(segmentation_dir, exist_ok=True)
    os.makedirs(texture_dir, exist_ok=True)
    os.makedirs(garments_dir, exist_ok=True)
    os.makedirs(canonical_dir, exist_ok=True)

    # Read transform_vis.json
    print("Reading transform_vis.json")
    vis_cameras = read_cameras_from_transforms(args.s, "transforms_vis.json")

    print("Rendering")

    with torch.no_grad():
        # Render segmentation visualization
        for camera in tqdm(vis_cameras, desc="Segmentation"):
            if camera.segmentation is not None:
                gt_image = tensor_to_cv2(
                    camera.segmentation * camera.mask
                    + (1 - camera.mask) * background.view(3, 1, 1).expand_as(camera.segmentation)
                )
                gt_image = overlay_text(gt_image, "Ground Truth")

            segmented_scan_image = tensor_to_cv2(
                render(camera, segmented_scan_gaussians, args, background)["render"]
            )
            segmented_scan_image = overlay_text(segmented_scan_image, "Segmentation Gaussians")

            segmented_smplx_image = tensor_to_cv2(
                render(camera, segmented_smplx_gaussians, args, background)["render"]
            )
            segmented_smplx_image = overlay_text(segmented_smplx_image, "Segmented SMPL-X")

            if camera.segmentation is not None:
                vis_image = np.concatenate([gt_image, segmented_scan_image, segmented_smplx_image], axis=1)
            else:
                vis_image = np.concatenate([segmented_scan_image, segmented_smplx_image], axis=1)
            cv2.imwrite(f"{segmentation_dir}/{camera.image_name}.png", vis_image)

        # Render texture visualization
        for camera in tqdm(vis_cameras, desc="Texture"):
            if camera.image is not None:
                gt_image = tensor_to_cv2(
                    camera.image * camera.mask
                    + (1 - camera.mask) * background.view(3, 1, 1).expand_as(camera.image)
                )
                gt_image = overlay_text(gt_image, "Ground Truth")

            full_with_smplx_image = tensor_to_cv2(
                render(camera, full_with_smplx_gaussians, args, background)["render"]
            )
            full_with_smplx_image = overlay_text(full_with_smplx_image, "Full + SMPL-X")

            full_image = tensor_to_cv2(render(camera, full_gaussians, args, background)["render"])
            full_image = overlay_text(full_image, "Full")

            clothing_image = tensor_to_cv2(render(camera, clothing_gaussians, args, background)["render"])
            clothing_image = overlay_text(clothing_image, "Clothing")

            body_image = tensor_to_cv2(render(camera, body_gaussians, args, background)["render"])
            body_image = overlay_text(body_image, "Body")

            if camera.image is not None:
                vis_image = np.concatenate(
                    [gt_image, full_with_smplx_image, full_image, clothing_image, body_image], axis=1
                )
            else:
                vis_image = np.concatenate(
                    [full_with_smplx_image, full_image, clothing_image, body_image], axis=1
                )
            cv2.imwrite(f"{texture_dir}/{camera.image_name}.png", vis_image)

        # Render garments visualization
        for camera in tqdm(vis_cameras, desc="Garments"):
            garment_images = []
            for i in range(len(garment_labels)):
                garment = garment_gaussians[i]
                garment_with_smplx = garments_with_smplx_gaussians[i]

                garment_image = tensor_to_cv2(render(camera, garment, args, background)["render"])
                garment_image = overlay_text(garment_image, f"Label {garment_labels[i]}")

                garment_with_smplx_image = tensor_to_cv2(
                    render(camera, garment_with_smplx, args, background)["render"]
                )
                garment_with_smplx_image = overlay_text(
                    garment_with_smplx_image, f"Label {garment_labels[i]} + Body"
                )

                garment_images.append(np.concatenate([garment_with_smplx_image, garment_image], axis=0))

            cv2.imwrite(f"{garments_dir}/{camera.image_name}.png", np.concatenate(garment_images, axis=1))

        # Render canonical visualization
        for camera in tqdm(vis_cameras, desc="Canonical"):
            canonical_full_with_smplx_image = tensor_to_cv2(
                render(camera, canonical_full_with_smplx_gaussians, args, background)["render"]
            )
            canonical_full_with_smplx_image = overlay_text(
                canonical_full_with_smplx_image, "Canonical Full + SMPL-X"
            )

            canonical_images = [canonical_full_with_smplx_image]
            for i in range(len(garment_labels)):
                canonical_garment = canonical_garment_gaussians[i]

                canonical_garment_image = tensor_to_cv2(
                    render(camera, canonical_garment, args, background)["render"]
                )
                canonical_garment_image = overlay_text(
                    canonical_garment_image, f" Canonical Label {garment_labels[i]}"
                )

                canonical_images.append(canonical_garment_image)

            canonical_body_image = tensor_to_cv2(
                render(camera, canonical_body_gaussians, args, background)["render"]
            )
            canonical_body_image = overlay_text(canonical_body_image, "Canonical Body")

            canonical_images.append(canonical_body_image)

            cv2.imwrite(f"{canonical_dir}/{camera.image_name}.png", np.concatenate(canonical_images, axis=1))

    # Create gifs
    # create_gif_from_images(segmentation_dir, args, fps=30)
    # create_gif_from_images(texture_dir, args, fps=30)
    # create_gif_from_images(garments_dir, args, fps=30)
    # create_gif_from_images(canonical_dir, args, fps=30)

    # Create videos
    create_video_from_images(segmentation_dir, args, fps=30)
    create_video_from_images(texture_dir, args, fps=30)
    create_video_from_images(garments_dir, args, fps=30)
    create_video_from_images(canonical_dir, args, fps=30)


def main():
    args = get_evaluation_args()
    visualize(args)


if __name__ == "__main__":
    main()
