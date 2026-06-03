import sys
from pathlib import Path

from matplotlib import colors

sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
import torch
import trimesh
import numpy as np
from plyfile import PlyData
from utils.sh_utils import SH2RGB
from arguments import get_meshify_args
from scene.scene_loader import read_synthetic_info
from gaussian_models.smplx_gaussian_model import SMPLXGaussianModel
from gaussian_models.texture_gaussian_model import TextureGaussianModel


def get_garment_params(args, garment):
    return {
        "3": (args.g3_num_iterations, args.g3_laplacian_lambda, args.g3_alpha),  # upper
        "4": (args.g4_num_iterations, args.g4_laplacian_lambda, args.g4_alpha),  # lower
        "5": (args.g5_num_iterations, args.g5_laplacian_lambda, args.g5_alpha),  # outer
    }[garment]


def gs_to_mesh(smplx_mesh, texture_gaussians):
    # Gaussian centers
    gaussian_centers = texture_gaussians.get_xyz.detach()

    # SMPL-X mesh tensors
    vertices = torch.tensor(smplx_mesh.vertices).float().cuda()
    faces = torch.tensor(smplx_mesh.faces).long().cuda()
    num_vertices = vertices.shape[0]

    # Per-vertex accumulators
    sum_pos = torch.zeros((num_vertices, 3)).cuda()
    count = torch.zeros((num_vertices, 1)).cuda()

    # Faces corresponding to each Gaussian
    gaussian_faces = faces[texture_gaussians.face_indices]

    # Accumulate Gaussian positions to incident vertices
    for k in range(3):
        v_idx = gaussian_faces[:, k]
        sum_pos.index_add_(0, v_idx, gaussian_centers)
        count.index_add_(0, v_idx, torch.ones_like(gaussian_centers[:, :1]))

    # Update vertices influenced by Gaussians
    valid = count.squeeze(1) > 0
    vertices[valid] = sum_pos[valid] / count[valid]

    # Convert Gaussian SH features to RGB
    gaussian_features_dc = texture_gaussians._features_dc.detach().squeeze(1)
    gaussian_colors = SH2RGB(gaussian_features_dc).clamp(0.0, 1.0)

    # Per-face color accumulators
    num_faces = faces.shape[0]
    sum_face_color = torch.zeros((num_faces, 3)).cuda()
    face_count = torch.zeros((num_faces, 1)).cuda()

    # Accumulate Gaussian colors per face
    face_ids = texture_gaussians.face_indices.long().cuda()
    sum_face_color.index_add_(0, face_ids, gaussian_colors)
    face_count.index_add_(0, face_ids, torch.ones_like(gaussian_colors[:, :1]))

    # Average colors for valid faces
    valid_faces = face_count.squeeze(1) > 0
    face_colors = torch.zeros_like(sum_face_color)
    face_colors[valid_faces] = sum_face_color[valid_faces] / face_count[valid_faces]

    # Extract garment faces and vertices
    garment_face_ids = texture_gaussians.face_indices.unique()
    garment_faces = faces[garment_face_ids]
    garment_vertex_ids = torch.unique(garment_faces.view(-1))

    # Remap vertex indices
    old_to_new = -torch.ones(num_vertices).long().cuda()
    old_to_new[garment_vertex_ids] = torch.arange(len(garment_vertex_ids)).long().cuda()
    remapped_faces = old_to_new[garment_faces]

    # Gather garment vertices and colors
    garment_vertices = vertices[garment_vertex_ids]
    garment_face_colors = face_colors[garment_face_ids]

    # Build RGBA face colors
    face_colors_rgba = torch.cat(
        [garment_face_colors, torch.ones((garment_face_colors.shape[0], 1)).cuda()],
        dim=1,
    ).clamp(0.0, 1.0)
    face_colors_rgba = (face_colors_rgba.cpu().numpy() * 255).astype(np.uint8)

    # Construct garment-only mesh
    mesh = trimesh.Trimesh(
        vertices=garment_vertices.cpu().numpy(),
        faces=remapped_faces.cpu().numpy(),
        process=False,
    )
    mesh.visual.face_colors = face_colors_rgba

    return mesh


def main():
    args = get_meshify_args()

    # Read scene info
    scene_info = read_synthetic_info(args, read_cameras=False)
    smplx_mesh = scene_info.smplx_mesh

    # Load smplx gaussians
    smplx_gaussians = SMPLXGaussianModel()
    smplx_gaussians.load_ply(
        scene_info.smplx_gaussians, scene_info.segmentation_gaussians, invisible_only=False
    )

    print("Converting gaussians to meshes")
    for garment in ["3", "4", "5"]:
        texture_gaussians_path = os.path.join(args.s, "gaussians", args.e, args.f, f"{garment}.ply")

        if not os.path.exists(texture_gaussians_path):
            continue

        # Load texture gaussians
        garment_gaussians_ply = PlyData.read(texture_gaussians_path).elements[0]
        garment_gaussians = TextureGaussianModel()
        garment_gaussians.load_from_texture_ply(
            scene_info.smplx_mesh,
            scene_info.smplx_gaussians,
            garment_gaussians_ply,
            None,
            args,
        )

        # Convert gaussians to mesh
        garment_mesh = gs_to_mesh(smplx_mesh, garment_gaussians)

        # Smooth and inflate mesh
        num_iterations, laplacian_lambda, alpha = get_garment_params(args, garment)
        garment_mesh = trimesh.smoothing.filter_laplacian(
            garment_mesh, lamb=laplacian_lambda, iterations=num_iterations
        )
        garment_mesh.vertices += alpha * garment_mesh.vertex_normals

        # Override colors with normal map if specified
        if args.normal_map:
            face_colors = ((garment_mesh.face_normals * 0.5 + 0.5) * 255).astype(np.uint8)
            alpha = np.full((face_colors.shape[0], 1), 255, dtype=np.uint8)
            garment_mesh.visual.face_colors = np.hstack([face_colors, alpha])

        # Save mesh
        garment_mesh_path = os.path.join(args.s, "gaussians", args.e, args.f, f"{garment}_mesh.ply")
        garment_mesh.export(garment_mesh_path)
        print(f"     Saved gaussians/{args.e}/{args.f}/{garment}_mesh.ply")


if __name__ == "__main__":
    main()
