# Installation Guide

Tested with Python 3.9, PyTorch 2.7.1, and CUDA 11.8.

## 0. Clone the repository

```bash
git clone https://github.com/danieleskandar/DAMA-code.git DAMA
cd DAMA
```

## 1. Create and activate environment

```bash
conda create -n dama python=3.9 -y
conda activate dama
```

## 2. Install PyTorch

Example for CUDA 11.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 3. Install dependencies

```bash
conda install ffmpeg=4.2.2 pillow=10.2.0 typing_extensions=4.9.0 -y
pip install open3d mediapy lpips scikit-image tqdm trimesh plyfile opencv-python smplx roma tabulate black
```

## 4. Install local submodules

```bash
pip install ./submodules/diff-surfel-rasterization --no-build-isolation
pip install ./submodules/simple-knn --no-build-isolation
```

## 5. Blender setup

```bash
mkdir blender
cd blender

wget https://download.blender.org/release/Blender3.6/blender-3.6.2-linux-x64.tar.xz
tar -xf blender-3.6.2-linux-x64.tar.xz

export PATH=$PWD/blender-3.6.2-linux-x64:$PATH

$PWD/blender-3.6.2-linux-x64/3.6/python/bin/python3.10 -m pip install pillow

cd ..
```

## 6. Download SMPL-X models

1. Visit <a href="https://smpl-x.is.tue.mpg.de/index.html" target="_blank">SMPL-X website</a>

2. Download `SMPL-X v1.1 (NPZ+PKL, 830 MB)`

3. Create a `smplx` directory and place the following files inside:

```text
smplx/
├── SMPLX_FEMALE.npz
├── SMPLX_MALE.npz
└── SMPLX_NEUTRAL.npz
```
