import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
import trimesh
import argparse
import numpy as np
from utils.sh_utils import RGB2SH
from plyfile import PlyData, PlyElement
from utils.general_utils import rotmat2qvec


def inverse_sigmoid(x):
    return np.log(x / (1 - x))


def save_gaussians(subject_folder_dir, experiment_folder_name, vertices, faces, colors, suffix=""):
    # Means
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    triangles = np.stack([v0, v1, v2], axis=1)
    means = triangles.mean(axis=1)

    # Rotations
    normals = np.cross(v1 - v0, v2 - v0)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    tangents = (v1 - v0) / (np.linalg.norm(v1 - v0, axis=1, keepdims=True) + 1e-8)
    bitangents = np.cross(normals, tangents)
    bitangents /= np.linalg.norm(bitangents, axis=1, keepdims=True) + 1e-8
    R = np.stack([tangents, bitangents, normals], axis=2)
    quats = rotmat2qvec(R)

    # Scales
    A = np.stack([v0, v1, v2], axis=1)
    local = np.einsum("nij,nkj->nki", R.transpose(0, 2, 1), A - means[:, None, :])
    xy = local[:, :, :2]
    scales = (xy.max(axis=1) - xy.min(axis=1)) / 2
    scales = np.log(scales)
    scale_2 = np.full((scales.shape[0], 1), -np.inf)
    scales = np.concatenate([scales, scale_2], axis=1)

    # Colors
    colors = colors[faces].mean(axis=1)[:, :3] / 255.0
    colors = RGB2SH(colors)

    # Opacities
    opacities = inverse_sigmoid(0.999)

    # Build vertex data for ply
    vertex_data = np.empty(
        len(means),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
            ("f_dc_0", "f4"),
            ("f_dc_1", "f4"),
            ("f_dc_2", "f4"),
            ("opacity", "f4"),
            ("scale_0", "f4"),
            ("scale_1", "f4"),
            ("scale_2", "f4"),
            ("rot_0", "f4"),
            ("rot_1", "f4"),
            ("rot_2", "f4"),
            ("rot_3", "f4"),
        ],
    )

    vertex_data["x"], vertex_data["y"], vertex_data["z"] = means.T
    vertex_data["nx"], vertex_data["ny"], vertex_data["nz"] = normals.T
    vertex_data["f_dc_0"], vertex_data["f_dc_1"], vertex_data["f_dc_2"] = colors.T
    vertex_data["opacity"] = opacities
    vertex_data["scale_0"], vertex_data["scale_1"], vertex_data["scale_2"] = scales.T
    vertex_data["rot_0"], vertex_data["rot_1"], vertex_data["rot_2"], vertex_data["rot_3"] = quats.T

    # Write ply
    gaussians_posed_dir = os.path.join(subject_folder_dir, "gaussians", experiment_folder_name, "posed")
    os.makedirs(gaussians_posed_dir, exist_ok=True)

    smplx_gaussians_path = os.path.join(gaussians_posed_dir, f"smplx{suffix}.ply")

    PlyData([PlyElement.describe(vertex_data, "vertex")], text=True).write(smplx_gaussians_path)

    print(f"     Saved gaussians/{experiment_folder_name}/posed/smplx{suffix}.ply")


def create_smplx_gaussians(subject_folder_dir, subject_folder_name, experiment_folder_name):
    print(f"Creating smplx gaussians for {subject_folder_name}")

    # Read smplx mesh
    smplx_mesh_path = os.path.join(subject_folder_dir, "meshes", "smplx.ply")
    smplx_mesh = trimesh.load(smplx_mesh_path, maintain_order=True, process=False)

    # Extract vertices, faces, and colors
    vertices = smplx_mesh.vertices
    faces = smplx_mesh.faces
    colors = smplx_mesh.visual.vertex_colors

    # Save original gaussians
    save_gaussians(subject_folder_dir, experiment_folder_name, vertices, faces, colors)

    # Subdivide mesh
    vertices_subdivided, faces_subdivided = trimesh.remesh.subdivide(vertices, faces)

    colors_subdivided = np.repeat(colors[0:1], len(vertices_subdivided), axis=0)

    smplx_subdivided_mesh_path = os.path.join(subject_folder_dir, "meshes", "smplx_subdivided.ply")

    trimesh.Trimesh(
        vertices=vertices_subdivided,
        faces=faces_subdivided,
        vertex_colors=colors_subdivided,
        maintain_order=True,
        process=False,
    ).export(smplx_subdivided_mesh_path)

    print("     Saved meshes/smplx_subdivided.ply")

    # Save subdivided gaussians
    save_gaussians(
        subject_folder_dir,
        experiment_folder_name,
        vertices_subdivided,
        faces_subdivided,
        colors_subdivided,
        suffix="_subdivided",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", type=str, help="Path to data directory containing subject folders")
    parser.add_argument("-s", type=str, help="Subject folder name")
    parser.add_argument("-e", type=str, default="main", help="Experiment folder")

    args = parser.parse_args()

    if args.s:
        subject_folder_dir = os.path.join(args.d, args.s)
        create_smplx_gaussians(subject_folder_dir, args.s, args.e)
    else:
        for subject_folder_name in sorted(os.listdir(args.d)):
            if os.path.isdir(os.path.join(args.d, subject_folder_name)):
                subject_folder_dir = os.path.join(args.d, subject_folder_name)
                create_smplx_gaussians(subject_folder_dir, subject_folder_name, args.e)


if __name__ == "__main__":
    main()
