import os
import torch
import numpy as np
from torch import nn
from utils.sh_utils import RGB2SH
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.general_utils import inverse_sigmoid, build_scaling_rotation


class SMPLXGaussianModel:

    def __init__(self):
        self.active_sh_degree = 0
        self.max_sh_degree = 0

        self.num_gaussians = None

        self._xyz = None
        self._features_dc = None
        self._scaling = None
        self._rotation = None
        self._opacity = None

        self.mask = None

        self.avg_skin_color = None

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
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return self._features_dc

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def construct_list_of_attributes(self):
        l = ["x", "y", "z"]
        for i in range(self._features_dc.shape[2]):
            l.append(f"f_dc_{i}")
        l.append("opacity")
        for i in range(self._scaling.shape[1] + 1):
            l.append(f"scale_{i}")
        for i in range(self._rotation.shape[1]):
            l.append(f"rot_{i}")
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self.get_xyz.detach().cpu().numpy()
        f_dc = self.get_features[:, 0, :3].detach().cpu().numpy()

        scale = self._scaling.detach().cpu().numpy()
        scale_2 = np.full((scale.shape[0], 1), -np.inf, dtype=np.float32)
        scale = np.concatenate((scale, scale_2), axis=1)
        rotation = self._rotation.detach().cpu().numpy()
        opacity = self._opacity.detach().cpu().numpy()

        dtype_full = [(attr, "f4") for attr in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)

        attributes = np.concatenate((xyz, f_dc, opacity, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def load_ply(self, smplx_gaussians, segmentation_gaussians, invisible_only=False):
        if invisible_only:
            self.mask = ~np.isin(segmentation_gaussians["refined_labels"], [0, 1])
        else:
            self.mask = np.ones_like(smplx_gaussians["x"], dtype=bool)

        self.num_gaussians = self.mask.sum()

        self._xyz = nn.Parameter(
            torch.tensor(
                np.stack(
                    [
                        np.asarray(smplx_gaussians["x"])[self.mask],
                        np.asarray(smplx_gaussians["y"])[self.mask],
                        np.asarray(smplx_gaussians["z"])[self.mask],
                    ],
                    axis=1,
                )
            )
            .float()
            .cuda(),
            requires_grad=False,
        )

        self._scaling = nn.Parameter(
            torch.tensor(
                np.stack(
                    [
                        np.asarray(smplx_gaussians["scale_0"])[self.mask],
                        np.asarray(smplx_gaussians["scale_1"])[self.mask],
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
                        np.asarray(smplx_gaussians["rot_0"])[self.mask],
                        np.asarray(smplx_gaussians["rot_1"])[self.mask],
                        np.asarray(smplx_gaussians["rot_2"])[self.mask],
                        np.asarray(smplx_gaussians["rot_3"])[self.mask],
                    ],
                    axis=1,
                )
            )
            .float()
            .cuda(),
            requires_grad=False,
        )

        self._opacity = nn.Parameter(
            torch.tensor(np.asarray(smplx_gaussians["opacity"])[self.mask]).unsqueeze(1).float().cuda(),
            requires_grad=False,
        )

        self._features_dc = nn.Parameter(
            torch.tensor(
                np.stack(
                    [
                        np.asarray(smplx_gaussians["f_dc_0"])[self.mask],
                        np.asarray(smplx_gaussians["f_dc_1"])[self.mask],
                        np.asarray(smplx_gaussians["f_dc_2"])[self.mask],
                    ],
                    axis=1,
                )
            )
            .unsqueeze(1)
            .float()
            .cuda(),
            requires_grad=False,
        )

        self.avg_skin_color = self._features_dc[0, 0, :3]

    def set_color(self, color):
        self._features_dc = nn.Parameter(
            RGB2SH(color).view(1, 1, 3).expand_as(self._features_dc), requires_grad=False
        )

    def set_random_color(self):
        color = torch.rand(3).float().cuda()
        self._features_dc = nn.Parameter(
            RGB2SH(color).view(1, 1, 3).expand_as(self._features_dc), requires_grad=False
        )

    def set_avg_skin_color(self):
        self._features_dc = nn.Parameter(
            self.avg_skin_color.view(1, 1, 3).expand_as(self._features_dc), requires_grad=False
        )

    def update(self, pose_properties):
        self._xyz = nn.Parameter(
            torch.tensor(pose_properties["means"][self.mask]).float().cuda(), requires_grad=False
        )
        self._rotation = nn.Parameter(
            torch.tensor(pose_properties["rotations"][self.mask]).float().cuda(), requires_grad=False
        )
        self._scaling = nn.Parameter(
            torch.tensor(pose_properties["scales"][self.mask]).float().cuda(), requires_grad=False
        )
