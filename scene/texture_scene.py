import os
import torch
import random
import numpy as np
from scene.smplx_model import SMPLXModel
from scene.scene_loader import read_synthetic_info
from gaussian_models.gaussian_model import GaussianModel
from gaussian_models.smplx_gaussian_model import SMPLXGaussianModel
from gaussian_models.texture_gaussian_model import TextureGaussianModel


class TextureScene:

    gaussians: GaussianModel = GaussianModel()
    smplx_gaussians: GaussianModel = SMPLXGaussianModel()
    texture_gaussians: dict = {}

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

        # Labels and colors
        self.label_colors = torch.tensor(scene_info.label_colors / 255.0).float().cuda()
        self.labels = np.unique(scene_info.segmentation_gaussians["refined_labels"]).astype(int).tolist()

        # SMPLX
        self.smplx_model = SMPLXModel(
            scene_info.basic_info,
            scene_info.smplx_params,
            scene_info.normalization,
            args.use_subdivided_mesh,
        )

        # Canonical properties
        canonical_properties = self.smplx_model.unpose()

        # SMPLX Gaussians
        self.smplx_gaussians.load_ply(
            scene_info.smplx_gaussians, scene_info.segmentation_gaussians, invisible_only=True
        )

        # Texture Gaussians
        for label in self.labels:
            skin_or_hair = label == 0 or label == 1

            colors = 0.0
            num_pixels = 0
            for viewpoint_cam in self.train_cameras:
                label_mask = (
                    (viewpoint_cam.segmentation.permute(1, 2, 0) == self.label_colors[label])
                    .all(dim=-1)
                    .unsqueeze(0)
                    .float()
                )
                colors += (label_mask * viewpoint_cam.image).sum(dim=(1, 2))
                num_pixels += label_mask.sum()
            label_avg_color = colors / num_pixels

            self.texture_gaussians[label] = TextureGaussianModel()
            self.texture_gaussians[label].load_from_segmentation_ply(
                scene_info.smplx_mesh,
                scene_info.smplx_gaussians,
                scene_info.segmentation_gaussians,
                label,
                label_avg_color,
                scene_info.nerf_normalization["radius"],
                canonical_properties,
                args,
                gaussians_per_face=args.gaussians_per_face if not skin_or_hair else 1,
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

    def save_body(self, save_dir, folder_name):
        self.smplx_gaussians.set_avg_skin_color()
        self.gaussians.merge([self.texture_gaussians[0], self.texture_gaussians[1], self.smplx_gaussians])
        self.gaussians.save_ply(os.path.join(save_dir, f"body.ply"))
        print(f"     Saved gaussians/{self.experiment_folder}/{folder_name}/body.ply")

    def save_clothing(self, save_dir, folder_name):
        self.gaussians.merge([self.texture_gaussians[self.labels[i]] for i in range(2, len(self.labels))])
        self.gaussians.save_ply(os.path.join(save_dir, f"clothing.ply"))
        print(f"     Saved gaussians/{self.experiment_folder}/{folder_name}/clothing.ply")

    def save_full(self, save_dir, folder_name):
        self.gaussians.merge([self.texture_gaussians[label] for label in self.labels])
        self.gaussians.save_ply(os.path.join(save_dir, f"full.ply"))
        print(f"     Saved gaussians/{self.experiment_folder}/{folder_name}/full.ply")

    def save_full_with_smplx(self, save_dir, folder_name):
        self.smplx_gaussians.set_avg_skin_color()
        self.gaussians.merge(
            [self.texture_gaussians[label] for label in self.labels] + [self.smplx_gaussians]
        )
        self.gaussians.save_ply(os.path.join(save_dir, f"full+smplx.ply"))
        print(f"     Saved gaussians/{self.experiment_folder}/{folder_name}/full+smplx.ply")

    def save_label(self, label, save_dir, folder_name):
        self.texture_gaussians[label].save_ply(os.path.join(save_dir, f"{str(label)}.ply"))
        print(f"     Saved gaussians/{self.experiment_folder}/{folder_name}/{str(label)}.ply")

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
            print(f"     Saved gaussians/{self.experiment_folder}/{folder_name}/{str(label)}+body.ply")

    def unpose_smplx(self):
        canonical_properties = self.smplx_model.unpose()
        self.smplx_gaussians.update(canonical_properties)
        for texture_gaussians in self.texture_gaussians.values():
            texture_gaussians.update(canonical_properties)

    def pose_smplx(self):
        pose_properties = self.smplx_model.pose()
        self.smplx_gaussians.update(pose_properties)
        for texture_gaussians in self.texture_gaussians.values():
            texture_gaussians.update(pose_properties)

    def save(self, folder_name):
        print(f"Saving {folder_name} gaussians")
        save_dir = os.path.join(self.source_path, "gaussians", self.experiment_folder, folder_name)
        os.makedirs(save_dir, exist_ok=True)
        self.save_body(save_dir, folder_name)
        self.save_clothing(save_dir, folder_name)
        self.save_full(save_dir, folder_name)
        self.save_full_with_smplx(save_dir, folder_name)
        for label in self.labels:
            self.save_label(label, save_dir, folder_name)
