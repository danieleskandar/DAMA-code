#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.sh_utils import eval_sh
from utils.point_utils import depth_to_normal


def render(viewpoint_camera, pc, args, bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None):
    # Screen-space tensor to track 2D positions (with gradients)
    screenspace_points = torch.zeros_like(pc.get_xyz, requires_grad=True, device="cuda")
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Field of view tangents for rasterization setup
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # Configure rasterizer settings
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=args.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # Use precomputed covariance if enabled, else use scaling/rotation
    scales, rotations, cov3D_precomp = None, None, None
    if args.compute_cov3D_python:
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor(
            [[W / 2, 0, 0, (W - 1) / 2], [0, H / 2, 0, (H - 1) / 2], [0, 0, far - near, near], [0, 0, 0, 1]],
            dtype=torch.float32,
            device="cuda",
        ).T
        world2pix = viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (splat2world[:, [0, 1, 3]] @ world2pix[:, [0, 1, 3]]).permute(0, 2, 1).reshape(-1, 9)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # Compute SH colors or use override
    shs, colors_precomp = None, None
    if override_color is None:
        if args.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Main rendering call
    rendered_image, radii, allmap = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    # Extract rendered outputs and derived maps
    render_alpha = allmap[1:2]
    render_normal = allmap[2:5].permute(1, 2, 0) @ viewpoint_camera.world_view_transform[:3, :3].T
    render_normal = render_normal.permute(2, 0, 1)

    render_depth_median = torch.nan_to_num(allmap[5:6], 0.0, 0.0)
    render_depth_expected = torch.nan_to_num(allmap[0:1] / render_alpha, 0.0, 0.0)
    render_dist = allmap[6:7]

    # Interpolate surface depth based on depth_ratio
    surf_depth = render_depth_expected * (1 - args.depth_ratio) + args.depth_ratio * render_depth_median

    # Approximate surface normals from depth
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth).permute(2, 0, 1)
    surf_normal *= render_alpha.detach()

    # Return all relevant rendering outputs
    return {
        "render": rendered_image,
        "viewspace_points": means2D,
        "visibility_filter": radii > 0,
        "radii": radii,
        "rend_alpha": render_alpha,
        "rend_normal": render_normal,
        "rend_dist": render_dist,
        "surf_depth": surf_depth,
        "surf_normal": surf_normal,
    }
