import os
import pickle
import trimesh
import argparse
import numpy as np
from PIL import Image


def noramlize(subject_folder_dir, subject_folder_name, label_colors):
    print(f"Normalizing scan and smplx meshes of {subject_folder_name}")

    raw_dir = os.path.join(subject_folder_dir, "raw")
    meshes_dir = os.path.join(subject_folder_dir, "meshes")
    os.makedirs(meshes_dir, exist_ok=True)

    # Load scan data
    scan = pickle.load(open(os.path.join(raw_dir, "scan.pkl"), "rb"))
    vertices, faces, normals, colors, uvs = (
        np.array(scan["vertices"]),
        np.array(scan["faces"]),
        np.array(scan["normals"]),
        np.array(scan["colors"]),
        np.array(scan["uvs"]),
    )

    # Load labels
    labels = np.array(pickle.load(open(os.path.join(raw_dir, "labels.pkl"), "rb"))["scan_labels"])

    # Load smplx
    smplx = trimesh.load(os.path.join(raw_dir, "smplx.ply"), maintain_order=True, process=False)

    # Load texture
    texture = pickle.load(open(os.path.join(raw_dir, "texture.pkl"), "rb"))

    # Load basic info
    basic_info = pickle.load(open(os.path.join(raw_dir, "basic_info.pkl"), "rb"))

    # Apply rotation
    rotation = basic_info["rotation"]
    if rotation is not None:
        vertices = (rotation @ vertices.T).T
        smplx.vertices = (rotation @ smplx.vertices.T).T

    # Compute shared normalization for scan and smplx
    all_vertices = np.concatenate([vertices, smplx.vertices], axis=0)
    min_corner = all_vertices.min(axis=0)
    max_corner = all_vertices.max(axis=0)
    center = (min_corner + max_corner) / 2
    scale = 1.0 / np.max(max_corner - min_corner)
    vertices = (vertices - center) * scale
    smplx.vertices = (smplx.vertices - center) * scale

    # Save normalization info
    normalization = {"center": center.tolist(), "scale": float(scale)}
    with open(os.path.join(meshes_dir, "normalization.pkl"), "wb") as f:
        pickle.dump(normalization, f)
    print("     Saved normalization.pkl")

    # Save scan.ply
    scan = trimesh.Trimesh(
        vertices=vertices, faces=faces, vertex_normals=normals, vertex_colors=colors, process=False
    )
    scan.export(os.path.join(meshes_dir, "scan.ply"))
    print("     Saved scan.ply")

    # Save smplx.ply
    skin_color = np.mean(colors[np.isin(labels, [0])], axis=0).astype(np.uint8)
    smplx.visual.vertex_colors = np.tile(skin_color, (smplx.vertices.shape[0], 1))
    smplx.export(os.path.join(meshes_dir, "smplx.ply"))
    print("     Saved smplx.ply")

    # Save segmented mesh with per-face majority color (duplicated vertices)
    segmented_scan_vertices = []
    segmented_scan_faces = []
    segmented_scan_normals = []
    segmented_scan_colors = []

    for face in faces:
        face_labels = labels[face]
        unique, counts = np.unique(face_labels, return_counts=True)
        majority = unique[np.argmax(counts)] if len(unique) > 1 else unique[0]
        if len(unique) == 3:
            majority = np.random.choice(face_labels)
        color = label_colors[int(majority)]

        start_idx = len(segmented_scan_vertices)
        for vi in face:
            segmented_scan_vertices.append(vertices[vi])
            segmented_scan_normals.append(normals[vi])
            segmented_scan_colors.append(color)
        segmented_scan_faces.append([start_idx, start_idx + 1, start_idx + 2])

    segmented_scan = trimesh.Trimesh(
        vertices=np.array(segmented_scan_vertices, dtype=np.float32),
        faces=np.array(segmented_scan_faces, dtype=np.int32),
        vertex_normals=np.array(segmented_scan_normals, dtype=np.float32),
        vertex_colors=np.array(segmented_scan_colors, dtype=np.uint8),
        process=False,
    )

    segmented_scan.export(os.path.join(meshes_dir, "segmented_scan.ply"))
    print("     Saved segmented_scan.ply")

    # Save texture image
    texture_path = os.path.join(meshes_dir, "texture.png")
    Image.fromarray(texture).save(texture_path)
    print("     Saved texture.png")

    # Save .mtl file
    mtl_path = os.path.join(meshes_dir, "texture.mtl")
    with open(mtl_path, "w") as f:
        f.write("newmtl material_0\n")
        f.write("Ka 1.000 1.000 1.000\n")
        f.write("Kd 1.000 1.000 1.000\n")
        f.write("Ks 0.000 0.000 0.000\n")
        f.write("d 1.0\n")
        f.write("illum 1\n")
        f.write("map_Kd texture.png\n")
    print("     Saved texture.mtl")

    # Save .obj file
    obj_path = os.path.join(meshes_dir, "scan.obj")
    with open(obj_path, "w") as f:
        f.write("mtllib texture.mtl\n")
        f.write("usemtl material_0\n")

        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")

        for uv in uvs:
            f.write(f"vt {uv[0]} {1 - uv[1]}\n")  # Flip V axis

        for face in faces:
            idx = face + 1  # OBJ is 1-indexed
            f.write(f"f {idx[0]}/{idx[0]} {idx[1]}/{idx[1]} {idx[2]}/{idx[2]}\n")
    print("     Saved scan.obj")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", type=str, help="Subject folder name")

    args = parser.parse_args()
    dataset_dir = os.path.join("data", "4D-DRESS")

    label_colors = np.load(os.path.join(dataset_dir, "label_colors.npy"))

    if args.s:
        subject_folder_dir = os.path.join(dataset_dir, args.s)
        noramlize(subject_folder_dir, args.s, label_colors)
    else:
        for subject_folder_name in sorted(os.listdir(dataset_dir)):
            if os.path.isdir(os.path.join(dataset_dir, subject_folder_name)):
                subject_folder_dir = os.path.join(dataset_dir, subject_folder_name)
                noramlize(subject_folder_dir, subject_folder_name, label_colors)


if __name__ == "__main__":
    main()
