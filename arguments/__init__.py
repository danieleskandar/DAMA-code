import argparse

experiment_folder = "main"


def get_train_segmentation_args():
    parser = argparse.ArgumentParser()
    parser.set_defaults(use_subdivided_mesh=True)

    # Paths
    parser.add_argument("-s", type=str, required=True, help="Source path")
    parser.add_argument("-c", type=str, required=True, help="Path to label_colors.npy")

    # Ablations
    parser.add_argument("-e", type=str, default=experiment_folder, help="Experiment folder")
    parser.add_argument("--unsigned_offset", action="store_true", default=False, help="Use unsigned offset")
    parser.add_argument("--free_xyz", action="store_true", default=False, help="Free xyz")

    # Use subdivided mesh
    parser.add_argument(
        "--no_use_subdivided_mesh",
        action="store_false",
        dest="use_subdivided_mesh",
        help="Disable subdivided mesh",
    )

    # Renderer
    parser.add_argument("--convert_SHs_python", action="store_true", default=False)
    parser.add_argument("--compute_cov3D_python", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--depth_ratio", type=float, default=0.0)

    # Iterations
    parser.add_argument("--num_iterations", type=int, default=10000, help="Number of iterations")
    parser.add_argument("--label_smoothness_start", type=int, default=7000, help="Label smoothness start")

    # Learning rates
    parser.add_argument("--position_lr_init", type=float, default=0.0016)
    parser.add_argument("--position_lr_final", type=float, default=0.0000016)
    parser.add_argument("--position_lr_delay_mult", type=float, default=0.01)
    parser.add_argument("--feature_lr", type=float, default=0.0025)
    parser.add_argument("--opacity_lr", type=float, default=0.05)
    parser.add_argument("--scaling_lr", type=float, default=0.005)
    parser.add_argument("--rotation_lr", type=float, default=0.001)

    # Loss weights
    parser.add_argument("--lambda_color", type=float, default=1.0)
    parser.add_argument("--lambda_scaling", type=float, default=10.0)
    parser.add_argument("--lambda_label_smoothness", type=float, default=0.1)
    parser.add_argument("--lambda_normal", type=float, default=0.1)

    # Label refining parameters
    parser.add_argument("--area_threshold", type=float, default=0.001)

    # Visualization
    parser.add_argument("--vis", action="store_true", help="Enable training visualization")
    parser.add_argument("--vis_freq", type=int, default=50, help="Visualization frequency")
    parser.add_argument("--circular", action="store_true", default=False)

    return parser.parse_args()


def get_train_texture_args():
    parser = argparse.ArgumentParser()
    parser.set_defaults(use_subdivided_mesh=True)

    # Paths
    parser.add_argument("-s", type=str, required=True, help="Source path")
    parser.add_argument("-c", type=str, required=True, help="Path to label_colors.npy")

    # Ablations
    parser.add_argument("-e", type=str, default=experiment_folder, help="Experiment folder")
    parser.add_argument("--unsigned_offset", action="store_true", default=False, help="Use unsigned offset")
    parser.add_argument("--free_xyz", action="store_true", default=False, help="Free xyz")

    # Use subdivided mesh
    parser.add_argument(
        "--no_use_subdivided_mesh",
        action="store_false",
        dest="use_subdivided_mesh",
        help="Disable subdivided mesh",
    )

    # Renderer
    parser.add_argument("--convert_SHs_python", action="store_true", default=False)
    parser.add_argument("--compute_cov3D_python", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--depth_ratio", type=float, default=0.0)

    # Iterations
    parser.add_argument("--num_iterations", type=int, default=2000, help="Number of iterations")

    # Learning rates
    parser.add_argument("--position_lr_init", type=float, default=0.0016)
    parser.add_argument("--position_lr_final", type=float, default=0.0000016)
    parser.add_argument("--position_lr_delay_mult", type=float, default=0.01)
    parser.add_argument("--feature_lr", type=float, default=0.0025)
    parser.add_argument("--opacity_lr", type=float, default=0.05)
    parser.add_argument("--scaling_lr", type=float, default=0.05)
    parser.add_argument("--rotation_lr", type=float, default=0.001)

    # Loss weights
    parser.add_argument("--lambda_color", type=float, default=1.0)
    parser.add_argument("--lambda_mask", type=float, default=10.0)
    parser.add_argument("--lambda_anisotropic", type=float, default=10.0)
    parser.add_argument("--lambda_max_scale", type=float, default=10000.0)
    parser.add_argument("--lambda_normal", type=float, default=0.1)
    parser.add_argument("--lambda_canonical_distance", type=float, default=1.0)
    parser.add_argument("--lambda_canonical_rotation", type=float, default=100.0)

    # Gaussian optimization parameters
    parser.add_argument("--gaussians_per_face", type=int, default=8)
    parser.add_argument("--max_scale", type=float, default=0.002)
    parser.add_argument("--anisotropic_r", type=float, default=1.0)

    # Visualization
    parser.add_argument("--vis", action="store_true", help="Enable training visualization")
    parser.add_argument("--vis_freq", type=int, default=40, help="Visualization frequency")
    parser.add_argument("--circular", action="store_true", default=False)

    return parser.parse_args()


def get_retarget_args():
    parser = argparse.ArgumentParser()
    parser.set_defaults(use_subdivided_mesh=True)

    # Paths
    parser.add_argument("-s", type=str, required=True, help="Target subject")
    parser.add_argument("-g1", type=str, default=None, help="Hair source subject")
    parser.add_argument("-g2", type=str, default=None, help="Shoes source subject")
    parser.add_argument("-g3", type=str, default=None, help="Inner source subject")
    parser.add_argument("-g4", type=str, default=None, help="Lower source subject")
    parser.add_argument("-g5", type=str, default=None, help="Outer source subject")

    # Ablations
    parser.add_argument("-e", type=str, default=experiment_folder, help="Experiment folder")
    parser.add_argument("--unsigned_offset", action="store_true", default=False, help="Use unsigned offset")
    parser.add_argument("--free_xyz", action="store_true", default=False, help="Free xyz")

    # Use subdivided mesh
    parser.add_argument(
        "--no_use_subdivided_mesh",
        action="store_false",
        dest="use_subdivided_mesh",
        help="Disable subdivided mesh",
    )

    # Renderer
    parser.add_argument("--convert_SHs_python", action="store_true", default=False)
    parser.add_argument("--compute_cov3D_python", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--depth_ratio", type=float, default=0.0)

    # Iterations
    parser.add_argument("--num_iterations", type=int, default=2000, help="Number of iterations")

    # Learning rates
    parser.add_argument("--position_lr_init", type=float, default=0.0016)
    parser.add_argument("--position_lr_final", type=float, default=0.0000016)
    parser.add_argument("--position_lr_delay_mult", type=float, default=0.01)
    parser.add_argument("--feature_lr", type=float, default=0.0025)
    parser.add_argument("--opacity_lr", type=float, default=0.05)
    parser.add_argument("--scaling_lr", type=float, default=0.05)
    parser.add_argument("--rotation_lr", type=float, default=0.001)

    # Loss weights
    parser.add_argument("--lambda_color", type=float, default=1.0)
    parser.add_argument("--lambda_mask", type=float, default=10.0)

    # Layering parameters
    parser.add_argument(
        "--layer_order", type=int, nargs="+", default=[2, 4, 3, 5, 1], help="Layer label order, bottom to top"
    )
    parser.add_argument(
        "--no_gaussian_opt", action="store_false", dest="gaussian_opt", help="Disable Gaussian optimization"
    )
    parser.add_argument(
        "--no_resolve_offsets", action="store_false", dest="resolve_offsets", help="Disable offset resolution"
    )

    # Visualization
    parser.add_argument("--vis", action="store_true", help="Enable training visualization")
    parser.add_argument("--vis_freq", type=int, default=40, help="Visualization frequency")
    parser.add_argument("--circular", action="store_true", default=False)

    return parser.parse_args()


def get_layer_args():
    parser = argparse.ArgumentParser()
    parser.set_defaults(use_subdivided_mesh=True)

    # Paths
    parser.add_argument("-s", type=str, required=True, help="Source path")
    parser.add_argument("-g2", nargs="*", type=str, default=[], help="Shoes source subjects")
    parser.add_argument("-g3", nargs="*", type=str, default=[], help="Inner source subjects")
    parser.add_argument("-g4", nargs="*", type=str, default=[], help="Lower source subjects")
    parser.add_argument("-g5", nargs="*", type=str, default=[], help="Outer source subjects")

    # Ablations
    parser.add_argument("-e", type=str, default=experiment_folder, help="Experiment folder")
    parser.add_argument("--unsigned_offset", action="store_true", default=False, help="Use unsigned offset")
    parser.add_argument("--free_xyz", action="store_true", default=False, help="Free xyz")

    # Use subdivided mesh
    parser.add_argument(
        "--no_use_subdivided_mesh",
        action="store_false",
        dest="use_subdivided_mesh",
        help="Disable subdivided mesh",
    )

    # Renderer
    parser.add_argument("--convert_SHs_python", action="store_true", default=False)
    parser.add_argument("--compute_cov3D_python", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--depth_ratio", type=float, default=0.0)

    # Iterations
    parser.add_argument("--num_iterations", type=int, default=2000, help="Number of iterations")

    # Learning rates
    parser.add_argument("--position_lr_init", type=float, default=0.0016)
    parser.add_argument("--position_lr_final", type=float, default=0.0000016)
    parser.add_argument("--position_lr_delay_mult", type=float, default=0.01)
    parser.add_argument("--feature_lr", type=float, default=0.0025)
    parser.add_argument("--opacity_lr", type=float, default=0.05)
    parser.add_argument("--scaling_lr", type=float, default=0.05)
    parser.add_argument("--rotation_lr", type=float, default=0.001)

    # Loss weights
    parser.add_argument("--lambda_color", type=float, default=1.0)
    parser.add_argument("--lambda_mask", type=float, default=10.0)

    # Layering parameters
    parser.add_argument(
        "--layer_order", type=int, nargs="+", default=[2, 4, 3, 5], help="Layer label order, bottom to top"
    )
    parser.add_argument(
        "--no_gaussian_opt", action="store_false", dest="gaussian_opt", help="Disable Gaussian optimization"
    )
    parser.add_argument(
        "--no_resolve_offsets", action="store_false", dest="resolve_offsets", help="Disable offset resolution"
    )

    # Visualization
    parser.add_argument("--vis", action="store_true", help="Enable training visualization")
    parser.add_argument("--vis_freq", type=int, default=40, help="Visualization frequency")
    parser.add_argument("--circular", action="store_true", default=False)

    return parser.parse_args()


def get_animate_args():
    parser = argparse.ArgumentParser()
    parser.set_defaults(use_subdivided_mesh=True)

    # Paths
    parser.add_argument("-s", type=str, required=True, help="Source path")
    parser.add_argument("-m", type=str, required=True, help="Motion path")

    # Ablations
    parser.add_argument("-e", type=str, default=experiment_folder, help="Experiment folder")
    parser.add_argument("--unsigned_offset", action="store_true", default=False, help="Use unsigned offset")
    parser.add_argument("--free_xyz", action="store_true", default=False, help="Free xyz")

    # Names
    parser.add_argument(
        "-f",
        type=str,
        required=True,
        choices=["posed", "retargeted", "layered"],
        default="posed",
        help="Folder name",
    )
    parser.add_argument("-n", type=str, required=True, help="Motion name")

    # Use subdivided mesh
    parser.add_argument(
        "--no_use_subdivided_mesh",
        action="store_false",
        dest="use_subdivided_mesh",
        help="Disable subdivided mesh",
    )

    # Renderer
    parser.add_argument("--convert_SHs_python", action="store_true", default=False)
    parser.add_argument("--compute_cov3D_python", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--depth_ratio", type=float, default=0.0)
    parser.add_argument("--cam_index", type=int, default=0)
    parser.add_argument("--circular", action="store_true", default=False)
    parser.add_argument(
        "--render_layers", action="store_true", default=False, help="Render clothing layers separately"
    )

    return parser.parse_args()


def get_meshify_args():
    parser = argparse.ArgumentParser()
    parser.set_defaults(use_subdivided_mesh=True)

    # Paths
    parser.add_argument("-s", type=str, required=True, help="Source path")

    # Folder name
    parser.add_argument(
        "-f",
        type=str,
        choices=["posed", "retargeted", "layered"],
        default="posed",
        help="Folder name",
    )

    # Ablations
    parser.add_argument("-e", type=str, default=experiment_folder, help="Experiment folder")
    parser.add_argument("--unsigned_offset", action="store_true", default=False, help="Use unsigned offset")
    parser.add_argument("--free_xyz", action="store_true", default=False, help="Free xyz")

    # Use subdivided mesh
    parser.add_argument(
        "--no_use_subdivided_mesh",
        action="store_false",
        dest="use_subdivided_mesh",
        help="Disable subdivided mesh",
    )

    # Spatial lr scale
    parser.add_argument("--spatial_lr_scale", type=float, default=1.0)

    # Parameters
    # Upper
    parser.add_argument("--g3_num_iterations", type=int, default=3, help="Number of smoothing iterations")
    parser.add_argument("--g3_laplacian_lambda", type=float, default=0.3, help="Laplacian smoothing lambda")
    parser.add_argument(
        "--g3_alpha", type=float, default=0.0023, help="Inflation amount along vertex normals"
    )
    # Lower
    parser.add_argument("--g4_num_iterations", type=int, default=3, help="Number of smoothing iterations")
    parser.add_argument("--g4_laplacian_lambda", type=float, default=0.2, help="Laplacian smoothing lambda")
    parser.add_argument(
        "--g4_alpha", type=float, default=0.0022, help="Inflation amount along vertex normals"
    )
    # Outer
    parser.add_argument("--g5_num_iterations", type=int, default=3, help="Number of smoothing iterations")
    parser.add_argument("--g5_laplacian_lambda", type=float, default=0.3, help="Laplacian smoothing lambda")
    parser.add_argument(
        "--g5_alpha", type=float, default=0.0021, help="Inflation amount along vertex normals"
    )

    # Normal map colors
    parser.add_argument(
        "--normal_map", action="store_true", default=False, help="Override colors with normal map"
    )

    return parser.parse_args()


def get_evaluation_args():
    parser = argparse.ArgumentParser()
    parser.set_defaults(use_subdivided_mesh=True)

    # Paths
    parser.add_argument("-s", type=str, required=True, help="Source path")

    # Ablations
    parser.add_argument("-e", type=str, default=experiment_folder, help="Experiment folder")

    # Renderer
    parser.add_argument("--convert_SHs_python", action="store_true", default=False)
    parser.add_argument("--compute_cov3D_python", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--depth_ratio", type=float, default=0.0)
    parser.add_argument(
        "--background", nargs=3, type=float, default=[1.0, 1.0, 1.0], help="Background RGB values in [0,1]"
    )

    return parser.parse_args()
