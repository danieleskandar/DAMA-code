import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
import glob
import json
import torch
import lpips
import pickle
import trimesh
import warnings
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from plyfile import PlyData
from utils.image_utils import psnr
from gaussian_renderer import render
from torchvision.utils import save_image
from arguments import get_evaluation_args
import torchvision.transforms.functional as tf
from gaussian_models.gaussian_model import GaussianModel
from scene.scene_loader import read_cameras_from_transforms

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")
warnings.filterwarnings("ignore", category=FutureWarning, module="lpips")


def read_image(image_path):
    image = Image.open(image_path)
    image = tf.to_tensor(image)
    if image.shape[0] == 4:
        mask = (image[3, :, :] > 0.5).float()
        image = mask * image[:3, :, :] + (1 - mask) * torch.ones_like(image[:3, :, :])
    return image.unsqueeze(0).cuda()


def read_images(gt_paths, rendering_paths):
    gts = []
    renders = []
    for gt_path, rendering_path in zip(gt_paths, rendering_paths):
        gts.append(read_image(gt_path))
        renders.append(read_image(rendering_path))
    return gts, renders


def to_lp(t):
    return t * 2.0 - 1.0


def psnr_and_lpips(gt_train_paths, gt_test_paths, renderings_train_paths, renderings_test_paths):
    train_gts, train_renders = read_images(gt_train_paths, renderings_train_paths)
    test_gts, test_renders = read_images(gt_test_paths, renderings_test_paths)

    train_psnrs = []
    test_psnrs = []
    train_lpipss = []
    test_lpipss = []

    lpips_loss = lpips.LPIPS(net="vgg").cuda()

    for gt, render in zip(train_gts, train_renders):
        train_psnrs.append(psnr(gt, render))
        train_lpipss.append(lpips_loss(to_lp(gt), to_lp(render)))

    for gt, render in zip(test_gts, test_renders):
        test_psnrs.append(psnr(gt, render))
        test_lpipss.append(lpips_loss(to_lp(gt), to_lp(render)))

    train_psnr = torch.tensor(train_psnrs).mean().item()
    test_psnr = torch.tensor(test_psnrs).mean().item()
    train_lpipss = torch.tensor(train_lpipss).mean().item()
    test_lpipss = torch.tensor(test_lpipss).mean().item()

    return train_psnr, test_psnr, train_lpipss, test_lpipss


def chamfer_distance(gt_vertices, reconstructed_vertices, scale, chunk_size=1000):
    if not torch.is_tensor(gt_vertices):
        gt_vertices = torch.from_numpy(gt_vertices)
    if not torch.is_tensor(reconstructed_vertices):
        reconstructed_vertices = torch.from_numpy(reconstructed_vertices)

    gt_vertices = gt_vertices.float().cuda()
    reconstructed_vertices = reconstructed_vertices.float().cuda()

    min_dist_gt = []
    for start in range(0, gt_vertices.shape[0], chunk_size):
        end = start + chunk_size
        dist = torch.cdist(gt_vertices[start:end], reconstructed_vertices)
        min_dist_gt.append(dist.min(dim=1).values)
    min_dist_gt = torch.cat(min_dist_gt)

    min_dist_recon = []
    for start in range(0, reconstructed_vertices.shape[0], chunk_size):
        end = start + chunk_size
        dist = torch.cdist(reconstructed_vertices[start:end], gt_vertices)
        min_dist_recon.append(dist.min(dim=1).values)
    min_dist_recon = torch.cat(min_dist_recon)

    chamfer_dist = (min_dist_gt.mean() + min_dist_recon.mean()).item()
    chamfer_dist = chamfer_dist / scale
    return chamfer_dist


def penetration_metrics(reconstructed_vertices, smplx_mesh, scale, chunk_size=1000):
    face_centers = torch.from_numpy(smplx_mesh.triangles_center.copy()).float().cuda()
    face_normals = torch.from_numpy(smplx_mesh.face_normals.copy()).float().cuda()

    if not torch.is_tensor(reconstructed_vertices):
        reconstructed_vertices = torch.from_numpy(reconstructed_vertices)

    reconstructed_vertices = reconstructed_vertices.float().cuda()

    inside_flags = []
    penetration_depths = []

    for start in range(0, reconstructed_vertices.shape[0], chunk_size):
        end = start + chunk_size
        pts = reconstructed_vertices[start:end]  # (B,3)

        dists = torch.cdist(pts, face_centers)
        min_idx = torch.argmin(dists, dim=1)

        vecs = pts - face_centers[min_idx]
        normals = face_normals[min_idx]

        signed_dist = (vecs * normals).sum(1)

        inside = signed_dist < 0
        depth = torch.clamp(-signed_dist, min=0.0)

        inside_flags.append(inside)
        penetration_depths.append(depth)

    inside_flags = torch.cat(inside_flags)
    penetration_depths = torch.cat(penetration_depths)

    penetration_rate = inside_flags.float().mean().item()

    penetrated = penetration_depths[penetration_depths > 0]
    avg_depth = penetrated.mean().item() if penetrated.numel() > 0 else 0.0
    avg_depth = avg_depth / scale

    return penetration_rate, avg_depth


def render_train_and_test_views(args):
    renderings_dir = os.path.join(args.s, "renderings", args.e)
    os.makedirs(renderings_dir, exist_ok=True)

    background = torch.tensor(args.background).float().cuda()

    # Load Gaussians
    full_with_smplx_gaussians = GaussianModel()
    full_with_smplx_gaussians.load_ply(os.path.join(args.s, "gaussians", args.e, "posed", "full+smplx.ply"))

    # Load train and test cameras
    print("Reading transforms_train.json and transforms_test.json")
    train_cameras = read_cameras_from_transforms(args.s, "transforms_train.json")
    test_cameras = read_cameras_from_transforms(args.s, "transforms_test.json")

    # Render train and test views
    print("Rendering train and test views")
    for viewpoint_cam in tqdm(train_cameras, desc="train"):
        render_image = render(viewpoint_cam, full_with_smplx_gaussians, args, background)["render"]
        save_image(render_image, os.path.join(renderings_dir, viewpoint_cam.image_name + ".png"))

    for viewpoint_cam in tqdm(test_cameras, desc="test"):
        render_image = render(viewpoint_cam, full_with_smplx_gaussians, args, background)["render"]
        save_image(render_image, os.path.join(renderings_dir, viewpoint_cam.image_name + ".png"))


def compute_metrics(args):
    print("Computing metrics")

    # Compute psnr and lpips
    gt_dir = os.path.join(args.s, "images")
    renderings_dir = os.path.join(args.s, "renderings", args.e)

    gt_train_paths = sorted(glob.glob(f"{gt_dir}/train_*.png"))
    gt_test_paths = sorted(glob.glob(f"{gt_dir}/test_*.png"))[1:]

    renderings_train_paths = sorted(glob.glob(f"{renderings_dir}/train_*.png"))
    renderings_test_paths = sorted(glob.glob(f"{renderings_dir}/test_*.png"))[1:]

    train_psnr, test_psnr, train_lpips, test_lpips = psnr_and_lpips(
        gt_train_paths, gt_test_paths, renderings_train_paths, renderings_test_paths
    )

    # Compute 3D metrics: chamfer distance and intersection metric
    smplx_path = os.path.join(args.s, "meshes", "smplx.ply")
    scan_path = os.path.join(args.s, "meshes", "scan.ply")
    reconstruction_path = os.path.join(args.s, "gaussians", args.e, "posed", "full.ply")
    normalization_path = os.path.join(args.s, "meshes", "normalization.pkl")

    smplx_mesh = trimesh.load(smplx_path, maintain_order=True, process=False)
    scan_mesh = trimesh.load(scan_path, maintain_order=True, process=False)
    reconstruction_pcd = PlyData.read(reconstruction_path).elements[0]
    scale = pickle.load(open(normalization_path, "rb"))["scale"]

    gt_vertices = scan_mesh.vertices
    reconstructed_vertices = np.stack(
        [reconstruction_pcd["x"], reconstruction_pcd["y"], reconstruction_pcd["z"]], axis=1
    )
    mask = np.isfinite(reconstructed_vertices).all(axis=1)
    reconstructed_vertices = reconstructed_vertices[mask]

    cd = chamfer_distance(gt_vertices, reconstructed_vertices, scale)
    penetration_rate, avg_penetration_depth = penetration_metrics(reconstructed_vertices, smplx_mesh, scale)

    # Print metrics
    metrics = {
        "train_psnr": train_psnr,
        "test_psnr": test_psnr,
        "train_lpips": train_lpips,
        "test_lpips": test_lpips,
        "chamfer_distance (mm)": cd * 1000,
        "penetration_rate (%)": penetration_rate * 100,
        "avg_penetration_depth (mm)": avg_penetration_depth * 1000,
    }

    subject_name = os.path.basename(args.s)
    df = pd.DataFrame.from_dict(metrics, orient="index", columns=[subject_name]).rename_axis("metric")
    table = df.to_markdown(floatfmt=".4f")
    print(f"\n{table}\n")


def main():
    args = get_evaluation_args()
    render_train_and_test_views(args)
    compute_metrics(args)


if __name__ == "__main__":
    main()
