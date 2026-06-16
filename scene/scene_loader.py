import os
import json
import pickle
import trimesh
import numpy as np
from tqdm import tqdm
from plyfile import PlyData
from typing import NamedTuple
from scene.camera import Camera
from torchvision.io import read_image
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal


class SceneInfo(NamedTuple):
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    smplx_mesh: trimesh.Trimesh
    smplx_gaussians: PlyData
    segmentation_gaussians: PlyData
    label_colors: np.array
    normalization: dict
    basic_info: dict
    smplx_params: dict


def get_nerf_pp_norm(cameras):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cameras:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def read_camera_from_tranforms(source_path, transforms_file, cam_index, extension=".png"):
    with open(os.path.join(source_path, transforms_file)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]
        height = contents["height"] if "height" in contents else None
        width = contents["width"] if "width" in contents else None

        frames = contents["frames"]
        idx = cam_index
        frame = frames[idx]

        c2w = np.array(frame["transform_matrix"])  # NeRF 'transform_matrix' is a camera-to-world transform
        c2w[
            :3, 1:3
        ] *= -1  # Change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)

        # Get world-to-camera transform and set R and T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        image_name = f"vis_{idx:04d}"
        image, mask, segmentation = None, None, None
        if "image_path" in frame and "segmentation_mask_path" in frame:
            # Paths
            image_path = os.path.join(source_path, frame["image_path"] + extension)
            segmentation_path = os.path.join(source_path, frame["segmentation_mask_path"] + extension)

            # Image name
            image_name = os.path.splitext(os.path.basename(image_path))[0]

            # Read image and segmentation
            image = read_image(image_path) / 255.0
            segmentation = read_image(segmentation_path) / 255.0

            # Mask
            mask = (image[3, :, :] > 0.5).float()

            # Process image and segmentation
            image = image[:3, :, :] * mask
            segmentation = segmentation[:3, :, :] * mask

            # Move image, segmentation to GPU
            image = image.cuda()
            mask = mask.cuda()
            segmentation = segmentation.cuda()

            # Height and width from image
            height = image.shape[1]
            width = image.shape[2]

        fovy = focal2fov(fov2focal(fovx, width), height)
        FovY = fovy
        FovX = fovx

        return Camera(
            colmap_id=idx,
            R=R,
            T=T,
            FoVx=FovX,
            FoVy=FovY,
            image_name=image_name,
            uid=idx,
            image=image,
            mask=mask,
            segmentation=segmentation,
            height=height,
            width=width,
        )


def read_cameras_from_transforms(source_path, transforms_file, extension=".png"):
    cameras = []

    split_name = os.path.basename(transforms_file).split("_")[-1].split(".")[0]

    with open(os.path.join(source_path, transforms_file)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]
        height = contents["height"] if "height" in contents else None
        width = contents["width"] if "width" in contents else None

        frames = contents["frames"]
        for idx, frame in tqdm(enumerate(frames), total=len(frames), desc=f"{split_name}"):
            c2w = np.array(
                frame["transform_matrix"]
            )  # NeRF 'transform_matrix' is a camera-to-world transform
            c2w[
                :3, 1:3
            ] *= -1  # Change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)

            # Get world-to-camera transform and set R and T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_name = f"{split_name}_{idx:04d}"
            (
                image,
                mask,
                segmentation,
            ) = (
                None,
                None,
                None,
            )
            if "image_path" in frame and "segmentation_mask_path" in frame:
                # Paths
                image_path = os.path.join(source_path, frame["image_path"] + extension)
                segmentation_path = os.path.join(source_path, frame["segmentation_mask_path"] + extension)

                # Image name
                image_name = os.path.splitext(os.path.basename(image_path))[0]

                # Read image and segmentation
                image = read_image(image_path) / 255.0
                segmentation = read_image(segmentation_path) / 255.0

                # Mask
                mask = (image[3, :, :] > 0.5).float()

                # Process image and segmentation
                image = image[:3, :, :] * mask
                segmentation = segmentation[:3, :, :] * mask

                # Move image, segmentation to GPU
                image = image.cuda()
                mask = mask.cuda()
                segmentation = segmentation.cuda()

                # Height and width from image
                height = image.shape[1]
                width = image.shape[2]

            fovy = focal2fov(fov2focal(fovx, width), height)
            FovY = fovy
            FovX = fovx

            cameras.append(
                Camera(
                    colmap_id=idx,
                    R=R,
                    T=T,
                    FoVx=FovX,
                    FoVy=FovY,
                    image_name=image_name,
                    uid=idx,
                    image=image,
                    mask=mask,
                    segmentation=segmentation,
                    height=height,
                    width=width,
                )
            )

    return cameras


def read_synthetic_info(args, read_cameras=True):
    print("Loading scene")

    # Read cameras
    if read_cameras:
        train_cameras = read_cameras_from_transforms(args.s, "transforms_train.json")
        test_cameras = read_cameras_from_transforms(args.s, "transforms_test.json")

        # Get nerf normalization
        nerf_normalization = get_nerf_pp_norm(train_cameras)
    else:
        train_cameras = []
        test_cameras = []
        nerf_normalization = {}

    # Read smplx mesh and smplx gaussians
    if args.use_subdivided_mesh:
        smplx_mesh = trimesh.load(
            os.path.join(args.s, "meshes", "smplx_subdivided.ply"), maintain_order=True, process=False
        )
        smplx_gaussians = PlyData.read(
            os.path.join(args.s, "gaussians", args.e, "posed", "smplx_subdivided.ply")
        ).elements[0]
    else:
        smplx_mesh = trimesh.load(
            os.path.join(args.s, "meshes", "smplx.ply"), maintain_order=True, process=False
        )
        smplx_gaussians = PlyData.read(
            os.path.join(args.s, "gaussians", args.e, "posed", "smplx.ply")
        ).elements[0]

    # Read segmentation gaussians if they exist
    segmentation_gaussians_path = os.path.join(args.s, "gaussians", args.e, "posed", "full_segmented.ply")
    if os.path.exists(segmentation_gaussians_path):
        segmentation_gaussians = PlyData.read(segmentation_gaussians_path).elements[0]
    else:
        segmentation_gaussians = None

    # Read label_colors.npy
    label_colors = np.load(args.c).astype(np.float32) if hasattr(args, "c") else None

    # Read normalization info
    normalization_path = os.path.join(args.s, "meshes", "normalization.pkl")
    with open(normalization_path, "rb") as f:
        normalization = pickle.load(f)

    # Read basic info
    basic_info_path = os.path.join(args.s, "raw", "basic_info.pkl")
    with open(basic_info_path, "rb") as f:
        basic_info = pickle.load(f)

    # Read smplx params
    smplx_params_path = os.path.join(args.s, "raw", "smplx.pkl")
    with open(smplx_params_path, "rb") as f:
        smplx_params = pickle.load(f)

    return SceneInfo(
        train_cameras=train_cameras,
        test_cameras=test_cameras,
        nerf_normalization=nerf_normalization,
        smplx_mesh=smplx_mesh,
        smplx_gaussians=smplx_gaussians,
        segmentation_gaussians=segmentation_gaussians,
        label_colors=label_colors,
        normalization=normalization,
        basic_info=basic_info,
        smplx_params=smplx_params,
    )
