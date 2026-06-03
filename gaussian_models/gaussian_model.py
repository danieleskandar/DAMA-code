import os
import torch
import numpy as np
from torch import nn
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.general_utils import inverse_sigmoid, build_scaling_rotation
from gaussian_models.texture_gaussian_model import TextureGaussianModel
from gaussian_models.segmentation_gaussian_model import SegmentationGaussianModel


class GaussianModel:

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

    def __init__(self):
        self.active_sh_degree = 0
        self.max_sh_degree = 0

        self._xyz = None
        self._features_dc = None
        self._scaling = None
        self._rotation = None
        self._opacity = None

        self.setup_functions()

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

    def load_ply(self, path):
        gaussians = PlyData.read(path).elements[0]

        self._xyz = (
            torch.tensor(
                np.stack(
                    [
                        np.asarray(gaussians["x"]),
                        np.asarray(gaussians["y"]),
                        np.asarray(gaussians["z"]),
                    ],
                    axis=1,
                )
            )
            .float()
            .cuda()
        )

        self._features_dc = (
            torch.tensor(
                np.stack(
                    [
                        np.asarray(gaussians["f_dc_0"]),
                        np.asarray(gaussians["f_dc_1"]),
                        np.asarray(gaussians["f_dc_2"]),
                    ],
                    axis=1,
                )
            )
            .unsqueeze(1)
            .float()
            .cuda()
        )

        self._scaling = (
            torch.tensor(
                np.stack([np.asarray(gaussians["scale_0"]), np.asarray(gaussians["scale_1"])], axis=1)
            )
            .float()
            .cuda()
        )

        self._rotation = (
            torch.tensor(
                np.stack(
                    [
                        np.asarray(gaussians["rot_0"]),
                        np.asarray(gaussians["rot_1"]),
                        np.asarray(gaussians["rot_2"]),
                        np.asarray(gaussians["rot_3"]),
                    ],
                    axis=1,
                )
            )
            .float()
            .cuda()
        )

        self._opacity = torch.tensor(np.asarray(gaussians["opacity"])).unsqueeze(1).float().cuda()

    def construct_list_of_attributes(self):
        l = ["x", "y", "z"]
        for i in range(self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1] + 1):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        f_dc = self.get_features.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        scale_2 = np.full((scale.shape[0], 1), -np.inf, dtype=np.float32)
        scale = np.concatenate((scale, scale_2), axis=1)
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, f_dc, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def merge(self, gaussians_list):
        self._xyz = torch.cat([gaussians.get_xyz for gaussians in gaussians_list], dim=0)
        self._features_dc = torch.cat([gaussians.get_features for gaussians in gaussians_list], dim=0)
        self._scaling = torch.cat([gaussians._scaling for gaussians in gaussians_list], dim=0)
        self._opacity = torch.cat([gaussians._opacity for gaussians in gaussians_list], dim=0)

        self._rotation = torch.cat(
            [
                (
                    gaussians.get_rotation
                    if isinstance(gaussians, (SegmentationGaussianModel, TextureGaussianModel))
                    else gaussians._rotation
                )
                for gaussians in gaussians_list
            ],
            dim=0,
        )
