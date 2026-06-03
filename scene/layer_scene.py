import os
import shutil
import random
import numpy as np
from plyfile import PlyData
from scene.scene_loader import read_synthetic_info
from gaussian_models.gaussian_model import GaussianModel
from gaussian_models.smplx_gaussian_model import SMPLXGaussianModel
from gaussian_models.texture_gaussian_model import TextureGaussianModel


class LayerScene:
    gaussians: GaussianModel = GaussianModel()
    smplx_gaussians: GaussianModel = SMPLXGaussianModel()
    full_smplx_gaussians: GaussianModel = SMPLXGaussianModel()
    texture_gaussians: dict = {}
    garment_gaussians: list = []

    label_names = {2: "shoes", 3: "inner", 4: "lower", 5: "outer"}

    def __init__(self, args, shuffle=True):
        self.source_path = args.s
        self.experiment_folder = args.e

        # Load scene
        scene_info = read_synthetic_info(args)

        # Save number of smplx faces
        self.num_smplx_faces = scene_info.smplx_mesh.faces.shape[0]

        # Cameras
        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.train_cameras = scene_info.train_cameras
        self.test_cameras = scene_info.test_cameras

        # SMPLX Gaussians
        self.smplx_gaussians.load_ply(
            scene_info.smplx_gaussians, scene_info.segmentation_gaussians, invisible_only=True
        )
        self.full_smplx_gaussians.load_ply(scene_info.smplx_gaussians, scene_info.segmentation_gaussians)

        print("Loading garments")

        # Skin and hair
        for label in [0, 1]:
            texture_gaussians_path = os.path.join(
                args.s, "gaussians", self.experiment_folder, "posed", f"{label}.ply"
            )
            texture_gaussians = PlyData.read(texture_gaussians_path).elements[0]
            texture_gaussian_model = TextureGaussianModel()
            texture_gaussian_model.load_from_texture_ply(
                scene_info.smplx_mesh,
                scene_info.smplx_gaussians,
                texture_gaussians,
                scene_info.nerf_normalization["radius"],
                args,
                is_new_layer=False,
                layer_name=f"{label}",
            )
            self.texture_gaussians[label] = texture_gaussian_model

        source_args = {2: args.g2, 3: args.g3, 4: args.g4, 5: args.g5}

        # Garments
        for label in [2, 3, 4, 5]:
            self.texture_gaussians[label] = []

            # Existing garments
            layer = 0
            while True:
                filename = f"{label}_{layer}.ply"
                texture_gaussians_path = os.path.join(
                    args.s, "gaussians", self.experiment_folder, "layered", filename
                )
                if not os.path.exists(texture_gaussians_path):
                    break  # No more layers
                texture_gaussians = PlyData.read(texture_gaussians_path).elements[0]
                texture_gaussian_model = TextureGaussianModel()
                texture_gaussian_model.load_from_texture_ply(
                    scene_info.smplx_mesh,
                    scene_info.smplx_gaussians,
                    texture_gaussians,
                    scene_info.nerf_normalization["radius"],
                    args,
                    is_new_layer=False,
                    layer_name=f"{label}_{layer}",
                )
                self.texture_gaussians[label].append(texture_gaussian_model)
                layer += 1

            # New garments
            sources = source_args[label]
            if sources:
                for subject_path in sources:
                    subject = os.path.basename(subject_path)
                    texture_gaussians_path = os.path.join(
                        subject_path, "gaussians", self.experiment_folder, "posed", f"{label}.ply"
                    )
                    if os.path.exists(texture_gaussians_path):
                        print(f"     Loading {self.label_names[label]} from {subject}")
                        texture_gaussians = PlyData.read(texture_gaussians_path).elements[0]
                        texture_gaussian_model = TextureGaussianModel()
                        texture_gaussian_model.load_from_texture_ply(
                            scene_info.smplx_mesh,
                            scene_info.smplx_gaussians,
                            texture_gaussians,
                            scene_info.nerf_normalization["radius"],
                            args,
                            is_new_layer=True,
                            layer_name=f"{label}_{layer}",
                        )
                        self.texture_gaussians[label].append(texture_gaussian_model)
                        layer += 1

        if args.unsigned_offset or args.free_xyz:
            args.position_lr_init *= 0.1

    def training_setup(self, args):
        for texture_gaussians in self.texture_gaussians.values():
            if isinstance(texture_gaussians, list):
                for g in texture_gaussians:
                    g.training_setup(args)
            else:
                texture_gaussians.training_setup(args)

    def get_train_cameras(self):
        return self.train_cameras

    def get_test_cameras(self):
        return self.test_cameras

    def create_garment_gaussians(self, args):
        self.garment_gaussians = []
        for label in args.layer_order:
            for texture_gaussians in self.texture_gaussians[label]:
                self.garment_gaussians.append(texture_gaussians)
        return self.garment_gaussians

    def save_body(self, save_dir):
        self.smplx_gaussians.set_avg_skin_color()
        self.gaussians.merge([self.texture_gaussians[0], self.texture_gaussians[1], self.smplx_gaussians])
        self.gaussians.save_ply(os.path.join(save_dir, f"body.ply"))
        print(f"     Saved gaussians/{self.experiment_folder}/layered/body.ply")

    def save_clothing(self, save_dir):
        self.gaussians.merge(self.garment_gaussians)
        self.gaussians.save_ply(os.path.join(save_dir, f"clothing.ply"))
        print(f"     Saved gaussians/{self.experiment_folder}/layered/clothing.ply")

    def save_full(self, save_dir):
        self.gaussians.merge([self.texture_gaussians[0], self.texture_gaussians[1]] + self.garment_gaussians)
        self.gaussians.save_ply(os.path.join(save_dir, f"full.ply"))
        print(f"     Saved gaussians/{self.experiment_folder}/layered/full.ply")

    def save_full_with_smplx(self, save_dir):
        self.gaussians.merge(
            [self.texture_gaussians[0], self.texture_gaussians[1], self.smplx_gaussians]
            + self.garment_gaussians
        )
        self.gaussians.save_ply(os.path.join(save_dir, f"full+smplx.ply"))
        print(f"     Saved gaussians/{self.experiment_folder}/layered/full+smplx.ply")

    def save_garment(self, garment, save_dir):
        garment.save_ply(os.path.join(save_dir, f"{garment.layer_name}.ply"))
        print(f"     Saved gaussians/{self.experiment_folder}/layered/{garment.layer_name}.ply")

        self.gaussians.merge(
            [self.texture_gaussians[0], self.texture_gaussians[1], self.smplx_gaussians, garment]
        )
        self.gaussians.save_ply(os.path.join(save_dir, f"{garment.layer_name}+body.ply"))
        print(f"     Saved gaussians/{self.experiment_folder}/layered/{garment.layer_name}+body.ply")

    def save(self):
        print(f"Saving layered gaussians")
        save_dir = os.path.join(self.source_path, "gaussians", self.experiment_folder, "layered")
        self.save_body(save_dir)
        self.save_clothing(save_dir)
        self.save_full(save_dir)
        self.save_full_with_smplx(save_dir)
        for garment in self.garment_gaussians:
            self.save_garment(garment, save_dir)
