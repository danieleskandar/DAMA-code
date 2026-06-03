import os
import shutil
import random
import numpy as np
from plyfile import PlyData
from scene.scene_loader import read_synthetic_info
from gaussian_models.gaussian_model import GaussianModel
from gaussian_models.smplx_gaussian_model import SMPLXGaussianModel
from gaussian_models.texture_gaussian_model import TextureGaussianModel


class RetargetScene:

    gaussians: GaussianModel = GaussianModel()
    smplx_gaussians: GaussianModel = SMPLXGaussianModel()
    full_smplx_gaussians: GaussianModel = SMPLXGaussianModel()
    texture_gaussians: dict = {}

    label_names = {1: "hair", 2: "shoes", 3: "inner", 4: "lower", 5: "outer"}

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

        print(f"     Loading body  from {os.path.basename(args.s)}")
        texture_gaussians_path = os.path.join(args.s, "gaussians", self.experiment_folder, "posed", f"0.ply")
        texture_gaussians = PlyData.read(texture_gaussians_path).elements[0]
        self.texture_gaussians[0] = TextureGaussianModel()
        self.texture_gaussians[0].load_from_texture_ply(
            scene_info.smplx_mesh,
            scene_info.smplx_gaussians,
            texture_gaussians,
            scene_info.nerf_normalization["radius"],
            args,
        )

        source_args = {1: args.g1, 2: args.g2, 3: args.g3, 4: args.g4, 5: args.g5}

        for label in [1, 2, 3, 4, 5]:
            source = source_args[label] if source_args[label] else args.s
            subject = os.path.basename(source)
            texture_gaussians_path = os.path.join(
                source, "gaussians", self.experiment_folder, "posed", f"{label}.ply"
            )

            if os.path.exists(texture_gaussians_path):
                print(f"     Loading {self.label_names[label]} from {subject}")
                texture_gaussians = PlyData.read(texture_gaussians_path).elements[0]
                self.texture_gaussians[label] = TextureGaussianModel()
                self.texture_gaussians[label].load_from_texture_ply(
                    scene_info.smplx_mesh,
                    scene_info.smplx_gaussians,
                    texture_gaussians,
                    scene_info.nerf_normalization["radius"],
                    args,
                )

        if args.unsigned_offset or args.free_xyz:
            args.position_lr_init *= 0.1

    def training_setup(self, args):
        for texture_gaussians in self.texture_gaussians.values():
            texture_gaussians.training_setup(args)

    def get_train_cameras(self):
        return self.train_cameras

    def get_test_cameras(self):
        return self.test_cameras

    def save_body(self, save_dir):
        self.smplx_gaussians.set_avg_skin_color()
        self.gaussians.merge([self.texture_gaussians[0], self.texture_gaussians[1], self.smplx_gaussians])
        self.gaussians.save_ply(os.path.join(save_dir, f"body.ply"))
        print(f"     Saved gaussians/{self.experiment_folder}/retargeted/body.ply")

    def save_clothing(self, save_dir):
        self.gaussians.merge(
            [self.texture_gaussians[label] for label in self.texture_gaussians.keys() if label not in [0, 1]]
        )
        self.gaussians.save_ply(os.path.join(save_dir, f"clothing.ply"))
        print(f"     Saved gaussians/{self.experiment_folder}/retargeted/clothing.ply")

    def save_full(self, save_dir):
        self.gaussians.merge([self.texture_gaussians[label] for label in self.texture_gaussians.keys()])
        self.gaussians.save_ply(os.path.join(save_dir, f"full.ply"))
        print(f"     Saved gaussians/{self.experiment_folder}/retargeted/full.ply")

    def save_full_with_smplx(self, save_dir):
        self.smplx_gaussians.set_avg_skin_color()
        self.gaussians.merge(
            [self.texture_gaussians[label] for label in self.texture_gaussians.keys()]
            + [self.smplx_gaussians]
        )
        self.gaussians.save_ply(os.path.join(save_dir, f"full+smplx.ply"))
        print(f"     Saved gaussians/{self.experiment_folder}/retargeted/full+smplx.ply")

    def save_label(self, label, save_dir):
        self.texture_gaussians[label].save_ply(os.path.join(save_dir, f"{str(label)}.ply"))
        print(f"     Saved gaussians/{self.experiment_folder}/retargeted/{str(label)}.ply")

        if label != 0 and label != 1:
            self.smplx_gaussians.set_avg_skin_color()
            self.gaussians.merge(
                [
                    self.texture_gaussians[label],
                    self.texture_gaussians[0],
                    self.texture_gaussians[1],
                    self.smplx_gaussians,
                ]
            )
            self.gaussians.save_ply(os.path.join(save_dir, f"{str(label)}+body.ply"))
            print(f"     Saved gaussians/{self.experiment_folder}/retargeted/{str(label)}+body.ply")

    def save(self):
        print(f"Saving retargeted gaussians")
        save_dir = os.path.join(self.source_path, "gaussians", self.experiment_folder, "retargeted")
        shutil.rmtree(save_dir, ignore_errors=True)
        os.makedirs(save_dir)
        self.save_body(save_dir)
        self.save_clothing(save_dir)
        self.save_full(save_dir)
        self.save_full_with_smplx(save_dir)
        for label in self.texture_gaussians.keys():
            self.save_label(label, save_dir)
