import torch
import trimesh
import numpy as np
from smplx import SMPLX
from trimesh.grouping import unique_rows
from trimesh.geometry import faces_to_edges
from utils.general_utils import rotmat2qvec


def compute_vertex_map(original_vertices, original_faces, subdivided_vertices, use_subdivided_mesh):
    V_original, V_subdivided = len(original_vertices), len(subdivided_vertices)

    if not use_subdivided_mesh:
        return np.arange(V_original)[:, None].repeat(2, axis=1)

    vertex_map = np.zeros((V_subdivided, 2), dtype=np.int32)

    # Original vertices
    vertex_map[:V_original] = np.arange(V_original)[:, None].repeat(2, axis=1)

    # Midpoint vertices
    edges = np.sort(faces_to_edges(original_faces), axis=1)
    unique_edges = edges[unique_rows(edges)[0]]
    vertex_map[V_original:] = unique_edges

    return vertex_map


class SMPLXModel:
    smplx_model: SMPLX
    basic_info: dict
    smplx_params: dict
    normalization: dict
    use_subdivided_mesh: bool

    betas: torch.Tensor = None
    global_orient: torch.Tensor = None
    body_pose: torch.Tensor = None
    left_hand_pose: torch.Tensor = None
    right_hand_pose: torch.Tensor = None
    jaw_pose: torch.Tensor = None
    expression: torch.Tensor = None
    leye_pose: torch.Tensor = None
    reye_pose: torch.Tensor = None
    transl: torch.Tensor = None

    def __init__(self, basic_info, smplx_params, normalization, use_subdivided_mesh, model_path="./smplx"):
        self.basic_info = basic_info
        self.smplx_params = smplx_params
        self.normalization = normalization
        self.use_subdivided_mesh = use_subdivided_mesh

        self.smplx_model = SMPLX(
            model_path=model_path,
            gender=self.basic_info["gender"],
            use_pca=smplx_params.get("use_pca", True),
            num_pca_comps=len(self.smplx_params["left_hand_pose"]),
            num_betas=len(self.smplx_params["betas"]),
            num_expression_coeffs=len(self.smplx_params["expression"]),
            flat_hand_mean=smplx_params.get("flat_hand_mean", False),
            batch_size=1,
        ).cuda()

        self.num_pca_comps = len(self.smplx_params["left_hand_pose"])

        self.set_params(smplx_params)
        self.set_faces_and_vertex_map()

    def set_params(self, params: dict):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, torch.tensor(value).float().unsqueeze(0).cuda())

    def use_pca_hand_pose(self):
        self.smplx_model.use_pca = True
        self.smplx_model.num_pca_comps = self.num_pca_comps

    def use_full_hand_pose(self):
        self.smplx_model.use_pca = False
        self.smplx_model.num_pca_comps = 0

    def get_smplx_output(self):
        return self.smplx_model(
            betas=self.betas,
            global_orient=self.global_orient,
            body_pose=self.body_pose,
            left_hand_pose=self.left_hand_pose,
            right_hand_pose=self.right_hand_pose,
            jaw_pose=self.jaw_pose,
            expression=self.expression,
            transl=self.transl,
        )

    def set_faces_and_vertex_map(self):
        smplx_output = self.get_smplx_output()

        original_vertices = smplx_output.vertices[0].detach().cpu().numpy()
        original_faces = self.smplx_model.faces_tensor.cpu().numpy()
        vertices, faces = np.copy(original_vertices), np.copy(original_faces)

        if self.use_subdivided_mesh:
            vertices, faces = trimesh.remesh.subdivide(original_vertices, original_faces)

        self.faces = faces
        self.vertex_map = compute_vertex_map(
            original_vertices, original_faces, vertices, self.use_subdivided_mesh
        )

    def unpose(self):
        self.set_params({"body_pose": np.zeros(self.body_pose[0].shape)})
        return self.get_properties()

    def pose(self):
        self.set_params(self.smplx_params)
        return self.get_properties()

    def get_properties(self):
        smplx_output = self.get_smplx_output()

        # Subdivision
        vertices = smplx_output.vertices[0].detach().cpu().numpy()
        vertices = vertices[self.vertex_map].mean(axis=1)

        # Apply base rotation
        vertices = (self.basic_info["rotation"] @ vertices.T).T

        # Normalize
        vertices = (vertices - self.normalization["center"]) * self.normalization["scale"]

        # Means
        v0, v1, v2 = vertices[self.faces[:, 0]], vertices[self.faces[:, 1]], vertices[self.faces[:, 2]]
        triangles = np.stack([v0, v1, v2], axis=1)
        means = triangles.mean(axis=1)

        # Rotations
        normals = np.cross(v1 - v0, v2 - v0)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
        tangents = (v1 - v0) / (np.linalg.norm(v1 - v0, axis=1, keepdims=True) + 1e-8)
        bitangents = np.cross(normals, tangents)
        bitangents /= np.linalg.norm(bitangents, axis=1, keepdims=True) + 1e-8
        R = np.stack([tangents, bitangents, normals], axis=2)
        rotations = rotmat2qvec(R)

        # Scales
        A = np.stack([v0, v1, v2], axis=1)
        local = np.einsum("nij,nkj->nki", R.transpose(0, 2, 1), A - means[:, None, :])
        xy = local[:, :, :2]
        scales = (xy.max(axis=1) - xy.min(axis=1)) / 2
        scales = np.log(scales)

        # Compute vertex normals from mesh
        vertex_normals = trimesh.Trimesh(
            vertices=vertices, faces=self.faces, maintain_order=True, process=False
        ).vertex_normals

        return {
            "vertices": vertices,
            "faces": self.faces,
            "vertex_normals": vertex_normals,
            "means": means,
            "rotations": rotations,
            "scales": scales,
        }
