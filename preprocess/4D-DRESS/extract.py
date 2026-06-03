import re
import shutil
import tarfile
import argparse
import numpy as np
from pathlib import Path


def extract(subject_tar_path: Path):
    print(f"Extracting scan from {subject_tar_path.stem}")

    subject_folder_name = subject_tar_path.stem.replace(".tar", "")
    extract_path = subject_tar_path.parent / subject_folder_name
    raw_out_dir = Path("data") / "4D-DRESS" / subject_folder_name / "raw"
    raw_out_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(subject_tar_path, "r:gz") as tar:
        tar.extractall(extract_path)

    # Find Inner or Outer folder if it exists
    if (extract_path / "Inner").exists():
        inner_or_outer_dir = extract_path / "Inner"
    elif (extract_path / "Outer").exists():
        inner_or_outer_dir = extract_path / "Outer"
    else:
        inner_or_outer_dir = extract_path

    # Select first take and first frame
    take = sorted(inner_or_outer_dir.glob("Take*"))[0]
    mesh_dir = take / "Meshes_pkl"
    mesh_files = sorted(
        mesh_dir.glob("mesh-f*.pkl"), key=lambda f: int(re.search(r"f(\d+)", f.name).group(1))
    )
    frame_id = re.search(r"f(\d+)", mesh_files[0].name).group(1)

    print(f"     {take.name}")
    print(f"     f{frame_id}")

    # Define file paths
    smplx_dir = take / "SMPLX"
    labels_dir = take / "Semantic" / "labels"
    files = {
        mesh_dir / f"mesh-f{frame_id}.pkl": raw_out_dir / "scan.pkl",
        mesh_dir / f"atlas-f{frame_id}.pkl": raw_out_dir / "texture.pkl",
        smplx_dir / f"mesh-f{frame_id}_smplx.pkl": raw_out_dir / "smplx.pkl",
        smplx_dir / f"mesh-f{frame_id}_smplx.ply": raw_out_dir / "smplx.ply",
        labels_dir / f"label-f{frame_id}.pkl": raw_out_dir / "labels.pkl",
        take / "basic_info.pkl": raw_out_dir / "basic_info.pkl",
    }

    # Copy files
    for src, dst in files.items():
        if src.exists():
            shutil.copy(src, dst)

    # Remove extracted files
    print(f"Removing {extract_path}")
    shutil.rmtree(extract_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", type=str, help="Path to 4D-DRESS dataset directory containing .tar.gz files")
    parser.add_argument("-s", type=str, help="Subject folder name")

    args = parser.parse_args()
    dataset_dir = Path(args.d)

    # Create dataset directory
    (Path("data") / "4D-DRESS").mkdir(parents=True, exist_ok=True)

    # Save 4D-DRESS label colors
    label_colors = np.array(
        [[128, 128, 128], [255, 128, 0], [128, 0, 255], [255, 0, 0], [0, 255, 0], [0, 128, 255]]
    )
    np.save(Path("data") / "4D-DRESS" / "label_colors.npy", label_colors)
    print("Saved label_colors.npy")

    if args.s:
        # Extract scan of one subject
        subject_tar_path = dataset_dir / f"{args.s}.tar.gz"
        extract(subject_tar_path)
    else:
        # Extract scans of all subjects
        for subject_tar_path in sorted(dataset_dir.glob("*.tar.gz")):
            if re.match(r"^\d{5}_(Inner|Outer)(_\d+)?\.tar\.gz$", subject_tar_path.name):
                extract(subject_tar_path)


if __name__ == "__main__":
    main()
