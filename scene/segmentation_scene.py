import os
import random
from scene.smplx_model import SMPLXModel
from scene.scene_loader import read_synthetic_info
from gaussian_models.gaussian_model import GaussianModel
from gaussian_models.smplx_gaussian_model import SMPLXGaussianModel
from gaussian_models.segmentation_gaussian_model import SegmentationGaussianModel


class SegmentationScene:

    gaussians: GaussianModel = GaussianModel()
    smplx_gaussians: SMPLXGaussianModel = SMPLXGaussianModel()
    segmentation_gaussians: SegmentationGaussianModel = SegmentationGaussianModel()

    def __init__(self, args, shuffle=True):
        self.source_path = args.s
        self.experiment_folder = args.e

        # Load scene
        scene_info = read_synthetic_info(args)

        # Cameras
        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.train_cameras = scene_info.train_cameras
        self.test_cameras = scene_info.test_cameras

        # SMPLX
        self.smplx_model = SMPLXModel(
            scene_info.basic_info,
            scene_info.smplx_params,
            scene_info.normalization,
            args.use_subdivided_mesh,
        )

        # Canonical properties
        self.smplx_model.unpose()
        canonical_properties = self.smplx_model.get_properties()

        # SMPLX gaussians
        self.smplx_gaussians.load_ply(scene_info.smplx_gaussians, scene_info.segmentation_gaussians)

        # Segmentation gaussians
        self.segmentation_gaussians.load_ply(
            scene_info.smplx_gaussians,
            scene_info.smplx_mesh,
            scene_info.label_colors,
            scene_info.nerf_normalization["radius"],
            canonical_properties,
            args,
        )

        if args.unsigned_offset or args.free_xyz:
            args.position_lr_init *= 0.1

    def training_setup(self, args):
        self.segmentation_gaussians.training_setup(args)

    def get_train_cameras(self):
        return self.train_cameras

    def get_test_cameras(self):
        return self.test_cameras

    def unpose_smplx(self):
        self.smplx_model.unpose()
        canonical_properties = self.smplx_model.get_properties()
        self.smplx_gaussians.update(canonical_properties)
        self.segmentation_gaussians.update(canonical_properties)

    def save_posed(self):
        print("Saving posed")

        os.makedirs(
            os.path.join(self.source_path, "gaussians", self.experiment_folder, "posed"), exist_ok=True
        )

        self.segmentation_gaussians.save_ply(
            os.path.join(
                self.source_path, "gaussians", self.experiment_folder, "posed", f"full_segmented.ply"
            )
        )
        print(f"     Saved gaussians/{self.experiment_folder}/posed/full_segmented.ply")

        self.smplx_gaussians.set_avg_skin_color()
        self.segmentation_gaussians.use_refined_colors = True
        self.gaussians.merge([self.segmentation_gaussians, self.smplx_gaussians])
        self.gaussians.save_ply(
            os.path.join(
                self.source_path, "gaussians", self.experiment_folder, "posed", f"full_segmented+smplx.ply"
            )
        )
        print(f"     Saved gaussians/{self.experiment_folder}/posed/full_segmented+smplx.ply")

        self.segmentation_gaussians.save_segmented_smplx_mesh(
            os.path.join(self.source_path, "meshes", f"segmented_smplx.ply")
        )
        print(f"     Saved meshes/segmented_smplx.ply")

        self.segmentation_gaussians.save_segmented_smplx_mesh_refined(
            os.path.join(self.source_path, "meshes", f"segmented_smplx_refined.ply")
        )
        print(f"     Saved meshes/segmented_smplx_refined.ply")

    def save_canonical(self):
        print("Saving canonical")

        os.makedirs(
            os.path.join(self.source_path, "gaussians", self.experiment_folder, "canonical"), exist_ok=True
        )

        self.segmentation_gaussians.save_ply(
            os.path.join(
                self.source_path, "gaussians", self.experiment_folder, "canonical", f"full_segmented.ply"
            )
        )
        print(f"     Saved gaussians/{self.experiment_folder}/canonical/full_segmented.ply")

        self.smplx_gaussians.set_avg_skin_color()
        self.segmentation_gaussians.use_refined_colors = True
        self.gaussians.merge([self.segmentation_gaussians, self.smplx_gaussians])
        self.gaussians.save_ply(
            os.path.join(
                self.source_path,
                "gaussians",
                self.experiment_folder,
                "canonical",
                f"full_segmented+smplx.ply",
            )
        )
        print(f"     Saved gaussians/{self.experiment_folder}/canonical/full_segmented+smplx.ply")
