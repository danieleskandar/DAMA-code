import os
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from roma import quat_product, quat_xyzw_to_wxyz, quat_wxyz_to_xyzw
from utils.general_utils import build_scaling_rotation, inverse_sigmoid, get_expon_lr_func


class TextureGaussianModel:
    def __init__(self):
        self.active_sh_degree = 0
        self.max_sh_degree = 0

        self.free_xyz = False
        self.unsigned_offset = False

        self._d_xyz = None

        self._bary = None
        self._normal_offset = None
        self._scaling = None
        self._rotation = None
        self._opacity = None
        self._features_dc = None

        self.face_indices = None

        self.posed_triangle_vertices = None
        self.posed_vertex_normals = None
        self.posed_base_rotation = None

        self.canonical_triangle_vertices = None
        self.canonical_vertex_normals = None
        self.canonical_base_rotation = None

        self.optimizer = None
        self.spatial_lr_scale = 0

        # Used in layering
        self.is_new_layer = None
        self.layer_name = None
        self.previous_layers_offset = None

        self.setup_functions()

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(
                torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation
            ).permute(0, 2, 1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:, :3, :3] = RS
            trans[:, 3, :3] = center
            trans[:, 3, 3] = 1
            return trans

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = nn.functional.normalize

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        relative_rotation = self.rotation_activation(self._rotation)
        base_rotation = self.rotation_activation(self.posed_base_rotation)
        return quat_xyzw_to_wxyz(
            quat_product(quat_wxyz_to_xyzw(base_rotation), quat_wxyz_to_xyzw(relative_rotation))
        )

    @property
    def get_canonical_rotation(self):
        relative_rotation = self.rotation_activation(self._rotation)
        base_rotation = self.rotation_activation(self.canonical_base_rotation)
        return quat_xyzw_to_wxyz(
            quat_product(quat_wxyz_to_xyzw(base_rotation), quat_wxyz_to_xyzw(relative_rotation))
        )

    @property
    def get_xyz(self):
        if self.free_xyz:
            return self.posed_triangle_vertices.mean(dim=1) + self._d_xyz
        else:
            bary = F.softmax(self._bary, dim=-1).unsqueeze(-1)
            base = torch.sum(bary * self.posed_triangle_vertices, dim=1)
            normal = F.normalize(torch.sum(bary * self.posed_vertex_normals, dim=1), dim=-1)
            offset = self.get_offset
            return base + offset * normal

    @property
    def get_canonical_xyz(self):
        if self.free_xyz:
            return self.canonical_triangle_vertices.mean(dim=1) + self._d_xyz
        else:
            bary = F.softmax(self._bary, dim=-1).unsqueeze(-1)
            base = torch.sum(bary * self.canonical_triangle_vertices, dim=1)
            normal = F.normalize(torch.sum(bary * self.canonical_vertex_normals, dim=1), dim=-1)
            offset = self.get_offset
            return base + offset * normal

    @property
    def get_features(self):
        return self._features_dc

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_offset(self):
        if self.unsigned_offset:
            return self._normal_offset + self.previous_layers_offset
        else:
            return F.softplus(self._normal_offset) + self.previous_layers_offset

    def get_max_offset_per_face(self):
        offset = self.get_offset
        unique_face_indices, inverse = torch.unique(self.face_indices, sorted=True, return_inverse=True)
        unique_face_indices, inverse = unique_face_indices.cuda(), inverse.cuda()

        max_offset = torch.full((unique_face_indices.shape[0], 1), float("-inf")).cuda()
        max_offset.scatter_reduce_(
            0, inverse.unsqueeze(1).expand_as(offset), offset, "amax", include_self=True
        )

        return unique_face_indices, max_offset

    def update_previous_layers_offset(self, max_offset_per_smplx_face):
        with torch.no_grad():
            offset = self.get_offset
            previous_layers_max_offset = max_offset_per_smplx_face[self.face_indices].detach().clone()
            mask = previous_layers_max_offset > offset
            self.previous_layers_offset[mask] = previous_layers_max_offset[mask] - offset[mask]

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def training_setup(self, args):
        if args.free_xyz:
            l = [
                {
                    "params": [self._d_xyz],
                    "lr": args.position_lr_init * self.spatial_lr_scale,
                    "name": "d_xyz",
                },
                {"params": [self._features_dc], "lr": args.feature_lr, "name": "features"},
                {"params": [self._scaling], "lr": args.scaling_lr, "name": "scaling"},
                {"params": [self._rotation], "lr": args.rotation_lr, "name": "rotation"},
            ]
        else:
            l = [
                {"params": [self._bary], "lr": args.position_lr_init * self.spatial_lr_scale, "name": "bary"},
                {
                    "params": [self._normal_offset],
                    "lr": args.position_lr_init * self.spatial_lr_scale,
                    "name": "offset",
                },
                {"params": [self._features_dc], "lr": args.feature_lr, "name": "features"},
                {"params": [self._scaling], "lr": args.scaling_lr, "name": "scaling"},
                {"params": [self._rotation], "lr": args.rotation_lr, "name": "rotation"},
            ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=args.position_lr_init * self.spatial_lr_scale,
            lr_final=args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=args.position_lr_delay_mult,
            max_steps=args.num_iterations,
        )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] in ["bary", "offset"]:
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z"]
        if self.free_xyz:
            l.extend(["dx", "dy", "dz"])
        for i in range(self._features_dc.shape[2]):
            l.append(f"f_dc_{i}")
        l.append("opacity")
        for i in range(self._scaling.shape[1] + 1):
            l.append(f"scale_{i}")
        for i in range(self._rotation.shape[1]):
            l.append(f"rot_{i}")
        for i in range(self._rotation.shape[1]):
            l.append(f"rel_rot_{i}")
        for i in range(self._bary.shape[1]):
            l.append(f"bary_{i}")
        l.append("normal_offset")
        l.append("face_indices")
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        if self.free_xyz:
            d_xyz = self._d_xyz.detach().cpu().numpy()

        xyz = self.get_xyz.detach().cpu().numpy()
        bary = self._bary.detach().cpu().numpy()

        normal_offset = torch.log(torch.expm1(self.get_offset)).detach().cpu().numpy()

        f_dc = self.get_features[:, 0, :3].detach().cpu().numpy()

        scale = self._scaling.detach().cpu().numpy()
        scale_2 = np.full((scale.shape[0], 1), -np.inf, dtype=np.float32)
        scale = np.concatenate((scale, scale_2), axis=1)

        rotation = self.get_rotation.detach().cpu().numpy()
        relative_rotation = self._rotation.detach().cpu().numpy()

        opacity = self._opacity.detach().cpu().numpy()
        face_indices = self.face_indices.detach().cpu().numpy()[:, None]

        dtype_full = [(attr, "f4") for attr in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)

        if self.free_xyz:
            attributes = np.concatenate(
                (
                    xyz,
                    d_xyz,
                    f_dc,
                    opacity,
                    scale,
                    rotation,
                    relative_rotation,
                    bary,
                    normal_offset,
                    face_indices,
                ),
                axis=1,
            )
        else:
            attributes = np.concatenate(
                (xyz, f_dc, opacity, scale, rotation, relative_rotation, bary, normal_offset, face_indices),
                axis=1,
            )

        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def load_from_segmentation_ply(
        self,
        smplx_mesh,
        smplx_gaussians,
        segmentation_gaussians,
        labels,
        label_avg_color,
        spatial_lr_scale,
        canonical_properties,
        args,
        gaussians_per_face=1,
    ):
        mask = np.isin(segmentation_gaussians["refined_labels"], labels)
        face_indices = np.nonzero(mask)[0]
        self.face_indices = torch.tensor(face_indices).repeat_interleave(gaussians_per_face, dim=0).long()

        vertices = torch.tensor(smplx_mesh.vertices).float()
        faces = torch.tensor(smplx_mesh.faces).long()[self.face_indices]
        vertex_normals = torch.tensor(smplx_mesh.vertex_normals).float()

        self.posed_triangle_vertices = vertices[faces].detach().cuda()
        self.posed_vertex_normals = vertex_normals[faces].detach().cuda()

        self.num_gaussians = len(faces)

        self.free_xyz = args.free_xyz
        if self.free_xyz:
            self._d_xyz = nn.Parameter(
                torch.tensor(
                    np.stack(
                        [
                            np.asarray(segmentation_gaussians["dx"])[self.face_indices],
                            np.asarray(segmentation_gaussians["dy"])[self.face_indices],
                            np.asarray(segmentation_gaussians["dz"])[self.face_indices],
                        ],
                        axis=1,
                    )
                )
                .float()
                .cuda(),
                requires_grad=True,
            )

        self._bary = nn.Parameter(
            torch.tensor(
                np.stack(
                    [
                        np.asarray(segmentation_gaussians["bary_0"])[self.face_indices],
                        np.asarray(segmentation_gaussians["bary_1"])[self.face_indices],
                        np.asarray(segmentation_gaussians["bary_2"])[self.face_indices],
                    ],
                    axis=1,
                )
            )
            .float()
            .cuda(),
            requires_grad=True,
        )

        self.unsigned_offset = args.unsigned_offset
        self._normal_offset = nn.Parameter(
            torch.tensor(np.asarray(segmentation_gaussians["normal_offset"])[self.face_indices])
            .reshape(-1, 1)
            .float()
            .cuda(),
            requires_grad=True,
        )

        self._scaling = nn.Parameter(
            torch.tensor(
                np.stack(
                    [
                        np.asarray(segmentation_gaussians["scale_0"])[self.face_indices],
                        np.asarray(segmentation_gaussians["scale_1"])[self.face_indices],
                    ],
                    axis=1,
                )
            )
            .float()
            .cuda(),
            requires_grad=True,
        )

        self.posed_base_rotation = nn.Parameter(
            torch.tensor(
                np.stack(
                    [
                        np.asarray(smplx_gaussians["rot_0"])[self.face_indices],
                        np.asarray(smplx_gaussians["rot_1"])[self.face_indices],
                        np.asarray(smplx_gaussians["rot_2"])[self.face_indices],
                        np.asarray(smplx_gaussians["rot_3"])[self.face_indices],
                    ],
                    axis=1,
                )
            )
            .float()
            .cuda(),
            requires_grad=False,
        )

        self._rotation = nn.Parameter(
            torch.tensor(
                np.stack(
                    [
                        np.asarray(segmentation_gaussians["rel_rot_0"])[self.face_indices],
                        np.asarray(segmentation_gaussians["rel_rot_1"])[self.face_indices],
                        np.asarray(segmentation_gaussians["rel_rot_2"])[self.face_indices],
                        np.asarray(segmentation_gaussians["rel_rot_3"])[self.face_indices],
                    ],
                    axis=1,
                )
            )
            .float()
            .cuda(),
            requires_grad=True,
        )

        self._opacity = nn.Parameter(
            torch.tensor(np.asarray(segmentation_gaussians["opacity"])[self.face_indices])
            .unsqueeze(1)
            .float()
            .cuda(),
            requires_grad=False,
        )

        self._features_dc = nn.Parameter(
            label_avg_color.view(1, 3).repeat(self.num_gaussians, 1).unsqueeze(1).float().cuda(),
            requires_grad=True,
        )

        canonical_vertices = torch.tensor(canonical_properties["vertices"]).float()
        canonical_vertex_normals = torch.tensor(canonical_properties["vertex_normals"]).float()

        self.canonical_triangle_vertices = canonical_vertices[faces].detach().cuda()
        self.canonical_vertex_normals = canonical_vertex_normals[faces].detach().cuda()

        self.canonical_base_rotation = nn.Parameter(
            torch.tensor(
                np.stack(
                    [
                        np.asarray(canonical_properties["rotations"][:, 0])[self.face_indices],
                        np.asarray(canonical_properties["rotations"][:, 1])[self.face_indices],
                        np.asarray(canonical_properties["rotations"][:, 2])[self.face_indices],
                        np.asarray(canonical_properties["rotations"][:, 3])[self.face_indices],
                    ],
                    axis=1,
                )
            )
            .float()
            .cuda(),
            requires_grad=False,
        )

        self.previous_layers_offset = torch.zeros_like(self._normal_offset).float().cuda()

        self.spatial_lr_scale = spatial_lr_scale

    def load_from_texture_ply(
        self,
        smplx_mesh,
        smplx_gaussians,
        texture_gaussians,
        spatial_lr_scale,
        args,
        is_new_layer=None,
        layer_name=None,
    ):
        self.face_indices = torch.tensor(texture_gaussians["face_indices"]).long()

        vertices = torch.tensor(smplx_mesh.vertices).float()
        faces = torch.tensor(smplx_mesh.faces).long()[self.face_indices]
        vertex_normals = torch.tensor(smplx_mesh.vertex_normals).float()

        self.posed_triangle_vertices = vertices[faces].detach().cuda()
        self.posed_vertex_normals = vertex_normals[faces].detach().cuda()

        self.num_gaussians = len(faces)

        self.free_xyz = args.free_xyz
        if self.free_xyz:
            self._d_xyz = nn.Parameter(
                torch.tensor(
                    np.stack(
                        [
                            np.asarray(texture_gaussians["dx"]),
                            np.asarray(texture_gaussians["dy"]),
                            np.asarray(texture_gaussians["dz"]),
                        ],
                        axis=1,
                    )
                )
                .float()
                .cuda(),
                requires_grad=True,
            )

        self._bary = nn.Parameter(
            torch.tensor(
                np.stack(
                    [
                        np.asarray(texture_gaussians["bary_0"]),
                        np.asarray(texture_gaussians["bary_1"]),
                        np.asarray(texture_gaussians["bary_2"]),
                    ],
                    axis=1,
                )
            )
            .float()
            .cuda(),
            requires_grad=True,
        )

        self.unsigned_offset = args.unsigned_offset
        self._normal_offset = nn.Parameter(
            torch.tensor(np.asarray(texture_gaussians["normal_offset"])).reshape(-1, 1).float().cuda(),
            requires_grad=True,
        )

        self._scaling = nn.Parameter(
            torch.tensor(
                np.stack(
                    [np.asarray(texture_gaussians["scale_0"]), np.asarray(texture_gaussians["scale_1"])],
                    axis=1,
                )
            )
            .float()
            .cuda(),
            requires_grad=False,
        )

        self.posed_base_rotation = nn.Parameter(
            torch.tensor(
                np.stack(
                    [
                        np.asarray(smplx_gaussians["rot_0"][self.face_indices]),
                        np.asarray(smplx_gaussians["rot_1"][self.face_indices]),
                        np.asarray(smplx_gaussians["rot_2"][self.face_indices]),
                        np.asarray(smplx_gaussians["rot_3"][self.face_indices]),
                    ],
                    axis=1,
                )
            )
            .float()
            .cuda(),
            requires_grad=False,
        )

        self._rotation = nn.Parameter(
            torch.tensor(
                np.stack(
                    [
                        np.asarray(texture_gaussians["rel_rot_0"]),
                        np.asarray(texture_gaussians["rel_rot_1"]),
                        np.asarray(texture_gaussians["rel_rot_2"]),
                        np.asarray(texture_gaussians["rel_rot_3"]),
                    ],
                    axis=1,
                )
            )
            .float()
            .cuda(),
            requires_grad=False,
        )

        self._opacity = nn.Parameter(
            torch.tensor(np.asarray(texture_gaussians["opacity"])).unsqueeze(1).float().cuda(),
            requires_grad=False,
        )

        self._features_dc = nn.Parameter(
            torch.tensor(
                np.stack(
                    [
                        np.asarray(texture_gaussians["f_dc_0"]),
                        np.asarray(texture_gaussians["f_dc_1"]),
                        np.asarray(texture_gaussians["f_dc_2"]),
                    ],
                    axis=1,
                )
            )
            .unsqueeze(1)
            .float()
            .cuda(),
            requires_grad=True,
        )

        self.spatial_lr_scale = spatial_lr_scale

        self.is_new_layer = is_new_layer
        self.layer_name = layer_name
        self.previous_layers_offset = torch.zeros_like(self._normal_offset).float().cuda()

    def update(self, pose_properties):
        vertices = torch.tensor(pose_properties["vertices"]).float()
        faces = torch.tensor(pose_properties["faces"]).long()[self.face_indices]
        vertex_normals = torch.tensor(pose_properties["vertex_normals"]).float()

        self.posed_triangle_vertices = vertices[faces].detach().cuda()
        self.posed_vertex_normals = vertex_normals[faces].detach().cuda()

        self.posed_base_rotation = nn.Parameter(
            torch.tensor(
                np.stack(
                    [
                        np.asarray(pose_properties["rotations"][:, 0])[self.face_indices],
                        np.asarray(pose_properties["rotations"][:, 1])[self.face_indices],
                        np.asarray(pose_properties["rotations"][:, 2])[self.face_indices],
                        np.asarray(pose_properties["rotations"][:, 3])[self.face_indices],
                    ],
                    axis=1,
                )
            )
            .float()
            .cuda(),
            requires_grad=False,
        )
