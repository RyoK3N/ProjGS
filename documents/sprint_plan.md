# ProjGS Sprint Plan: Projection-Guided Gaussian Splatting from Single RGB-D

**Project Goal:** Develop a novel camera-matrix-aware Gaussian Splatting method that leverages camera intrinsics for precise 3D reconstruction from single RGB-D frames.

**Timeline:** Weeks 1-3 (Feasibility Study) → Full Development (24 weeks if promising)

**Last Updated:** December 25, 2025

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Research Background](#research-background)
3. [Technical Foundation](#technical-foundation)
4. [Feasibility Study: Weeks 1-3](#feasibility-study-weeks-1-3)
5. [Implementation Details](#implementation-details)
6. [Verification Criteria](#verification-criteria)
7. [References](#references)

---

## Executive Summary

### Research Gap

Current Gaussian Splatting methods either:
- **Require multiple views** with unknown poses (FreeSplatter[1], CF-3DGS[2])
- **Use single view but ignore camera intrinsics** (single-view RGB methods)

**NO existing work** systematically uses known camera intrinsics (K matrix) to guide:
1. Gaussian initialization via precise depth unprojection
2. Focal-length-adaptive Gaussian scaling
3. Cross-sensor generalization via intrinsics normalization

### Core Innovation

> **ProjGS leverages camera intrinsics for precise 3D unprojection, enabling better Gaussian initialization and cross-sensor robustness**

**Key Components:**
1. **Camera-Aware Unprojection:** Use K matrix for exact depth→3D conversion
2. **Focal-Length-Adaptive Scaling:** Adjust Gaussian size based on camera parameters
3. **Intrinsics Normalization:** Generalize across 4 RGB-D sensors in SUN RGB-D

### SUN RGB-D Dataset Advantages

- **10,335 RGB-D scenes** with ground-truth camera calibration
- **4 different sensors:** Kinect v1, Kinect v2, Intel RealSense, Asus Xtion
- **Camera intrinsics (K matrix)** for every scene
- **Camera extrinsics** for pose information
- **37 scene categories** for semantic priors

---

## Research Background

### 1. 3D Gaussian Splatting (3DGS)

**Original Paper:** Kerbl et al. "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (SIGGRAPH 2023)[3]

**Core Concept:**
- Represent scenes as collection of 3D Gaussians
- Each Gaussian has: position μ, covariance Σ, color c, opacity α
- Render via differentiable splatting (60+ FPS)

**Standard Initialization:**
```
Input: Multi-view images
Step 1: Run Structure-from-Motion (SfM/COLMAP)
Step 2: Extract sparse point cloud
Step 3: Initialize Gaussians at point locations
Step 4: Optimize via gradient descent
```

**Limitations:**
- Heavy reliance on SfM quality[4]
- Uniform Gaussian initialization (ignores camera parameters)
- Poor performance with sparse views or single frames

### 2. Recent Depth-Aware Extensions

**Depth-Regularized GS (CVPR 2024 Workshop):**[5]
- Uses monocular depth estimation (Depth Anything V2)
- Adds depth supervision loss: L_depth = ||D_render - D_prior||²
- Exponential decay: λ_depth from 1.0 → 0.01

**IndoorGS (CVPR 2025):**[6]
- Geometric cues for indoor scenes
- Planar constraints for walls/floors
- Does NOT use camera intrinsics explicitly

**Key Gap:** None use known RGB-D depth + camera matrix for initialization

### 3. Camera Matrix Fundamentals

**Intrinsic Matrix (K):**
```
K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]

fx, fy: focal lengths (pixels)
cx, cy: principal point (image center offset)
```

**Depth Unprojection:**[7]
```python
# Given: pixel (u, v), depth D, intrinsics K
X = (u - cx) * D / fx
Y = (v - cy) * D / fy
Z = D

Point_3D = [X, Y, Z]^T
```

**SUN RGB-D Intrinsics Example (Kinect v1):**
```
fx = 518.857901
fy = 519.469611
cx = 284.582449
cy = 208.736166
```

### 4. Related Work on Camera-Aware Methods

**CoL3D (February 2025):**[8]
- Joint learning of depth + camera intrinsics
- Predicts intrinsics from RGB (no depth input)
- Shows intrinsics improve 3D shape recovery

**TrackGS (2025):**[9]
- Joint optimization of intrinsics + GS
- Derives gradients of intrinsics w.r.t. rendering
- Still requires multiple views

**FreeSplatter (2024):**[1]
- Pose-free sparse-view GS
- Estimates camera parameters
- Multi-view dependency

**Our Differentiation:**
- We have **known intrinsics** (no need to estimate)
- We use **single RGB-D frame** (no multi-view needed)
- We **normalize across sensors** (cross-sensor generalization)

---

## Technical Foundation

### 1. Camera-Aware Depth Unprojection

**Standard Approach (Naive):**
```python
# Treat all pixels uniformly
for u, v in pixels:
    if depth[u, v] > 0:
        X = u * depth[u, v]  # WRONG: ignores camera!
        Y = v * depth[u, v]
        Z = depth[u, v]
```

**Camera-Aware Approach (Ours):**
```python
def unproject_depth(depth, K):
    """
    Precise unprojection using camera intrinsics

    Args:
        depth: (H, W) depth map in meters
        K: (3, 3) camera intrinsic matrix
    Returns:
        points: (H*W, 3) 3D point cloud

    Reference: Intel RealSense docs[7]
    """
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Create pixel grid
    u, v = torch.meshgrid(torch.arange(W), torch.arange(H))

    # Unproject to 3D
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth

    points = torch.stack([X, Y, Z], dim=-1)
    return points.reshape(-1, 3)
```

**Why This Matters:**
- **Kinect v1:** fx≈519, cx≈284
- **RealSense:** Different focal length and principal point
- Naive approach → **wrong 3D positions**
- Camera-aware → **metric 3D coordinates**

### 2. Focal-Length-Adaptive Gaussian Scaling

**Problem:** Standard GS uses uniform initial scales
```python
# Vanilla 3DGS initialization
scale = compute_mean_distance_to_neighbors(points_3d)  # Same for all
```

**Our Insight:** Gaussians should scale with camera parameters

**Theory:**
- **High focal length** (telephoto) → tighter field of view → smaller Gaussians
- **Low focal length** (wide-angle) → broader field of view → larger Gaussians

**Mathematical Derivation:**
```
Pixel footprint at depth Z:
Δx = Z / fx  (width of one pixel in meters)

Gaussian scale should be proportional to pixel footprint:
scale_adaptive = base_scale * (Z / f)

where f = (fx + fy) / 2
```

**Implementation:**
```python
def adaptive_gaussian_scale(points_3d, K, base_scale=0.01):
    """
    Compute focal-length-adaptive Gaussian scales

    Args:
        points_3d: (N, 3) 3D points
        K: (3, 3) camera matrix
        base_scale: baseline scale multiplier

    Returns:
        scales: (N, 3) per-Gaussian scales

    Reference: SA-GS scale-adaptive work[10]
    """
    fx, fy = K[0, 0], K[1, 1]
    f_mean = (fx + fy) / 2

    Z = points_3d[:, 2]  # depth component

    # Scale inversely with focal length
    scale_x = base_scale * Z / fx
    scale_y = base_scale * Z / fy
    scale_z = base_scale * Z / f_mean

    scales = torch.stack([scale_x, scale_y, scale_z], dim=1)
    return scales
```

**Expected Behavior:**
- Kinect (f≈520): scale ≈ Z/520
- Wide-angle camera (f≈300): scale ≈ Z/300 (larger)
- Telephoto camera (f≈800): scale ≈ Z/800 (smaller)

### 3. Intrinsics Normalization for Cross-Sensor Generalization

**Problem:** Different sensors have different K matrices
```
Kinect v1:  fx=518.86, cx=284.58
RealSense:  fx=???    cx=???  (different)
```

**Solution:** Normalize to canonical camera space[11]

**Method:**
```python
class IntrinsicsNormalizer:
    """
    Normalize 3D points to canonical camera space

    Enables cross-sensor generalization by learning in
    sensor-agnostic representation.

    Reference: Cross-sensor domain adaptation principles
    """
    def __init__(self):
        # Canonical camera (average of all sensors)
        self.K_canonical = torch.tensor([
            [520.0,   0.0, 320.0],
            [  0.0, 520.0, 240.0],
            [  0.0,   0.0,   1.0]
        ])

    def normalize(self, points_3d, K_source):
        """
        Transform from source camera to canonical space

        Process:
        1. Project points to 2D with source camera
        2. Unproject to 3D with canonical camera
        """
        fx_s, fy_s = K_source[0, 0], K_source[1, 1]
        cx_s, cy_s = K_source[0, 2], K_source[1, 2]

        fx_c, fy_c = self.K_canonical[0, 0], self.K_canonical[1, 1]
        cx_c, cy_c = self.K_canonical[0, 2], self.K_canonical[1, 2]

        # Project to 2D (source camera)
        u = fx_s * (points_3d[:, 0] / points_3d[:, 2]) + cx_s
        v = fy_s * (points_3d[:, 1] / points_3d[:, 2]) + cy_s

        # Unproject to 3D (canonical camera)
        Z = points_3d[:, 2]  # preserve depth
        X = (u - cx_c) * Z / fx_c
        Y = (v - cy_c) * Z / fy_c

        return torch.stack([X, Y, Z], dim=1)

    def denormalize(self, points_canonical, K_target):
        """Inverse: canonical → target camera space"""
        # (Implementation is inverse of normalize)
        pass
```

**Training Strategy:**
```python
# Train in canonical space
points_canonical = normalizer.normalize(points_sensor, K_sensor)
gaussians = model(points_canonical)

# At test time with new sensor
points_test_canonical = normalizer.normalize(points_test, K_new_sensor)
gaussians_test = model(points_test_canonical)
```

### 4. Loss Functions

**Standard 3DGS Loss:**[3]
```python
L_total = (1 - λ) * L1(I_render, I_gt) + λ * L_SSIM(I_render, I_gt)
# Typically λ = 0.2
```

**Our Enhanced Loss:**
```python
L_total = L_render + λ_depth * L_depth + λ_geo * L_geometric

where:
- L_render: Standard L1 + SSIM photometric loss
- L_depth: Depth consistency loss (for RGB-D)
- L_geometric: Camera projection consistency
```

**Depth Consistency Loss:**
```python
def depth_consistency_loss(depth_render, depth_gt, K):
    """
    Ensure rendered depth matches ground truth

    Args:
        depth_render: (H, W) rendered depth
        depth_gt: (H, W) ground truth depth from RGB-D
        K: camera intrinsics

    Reference: Depth-regularized GS[5]
    """
    # Exponential decay (reduce weight over training)
    lambda_t = lambda_init * exp(-decay_rate * iteration)

    # L1 loss with validity mask
    mask = (depth_gt > 0) & (depth_gt < max_depth)
    loss = torch.abs(depth_render - depth_gt)[mask].mean()

    return lambda_t * loss
```

**Geometric Projection Loss:**
```python
def geometric_projection_loss(points_3d, K):
    """
    Enforce geometric consistency via reprojection

    Project 3D → 2D → 3D should be identity
    """
    # Project to 2D
    u = K[0,0] * (points_3d[:,0] / points_3d[:,2]) + K[0,2]
    v = K[1,1] * (points_3d[:,1] / points_3d[:,2]) + K[1,2]

    # Unproject back to 3D
    Z = points_3d[:, 2]
    X_reproj = (u - K[0,2]) * Z / K[0,0]
    Y_reproj = (v - K[1,2]) * Z / K[1,1]

    # Should match original
    loss = F.mse_loss(X_reproj, points_3d[:,0]) + \
           F.mse_loss(Y_reproj, points_3d[:,1])

    return loss
```

### 5. Optimization Details

**Standard 3DGS Hyperparameters:**[3][12]
```python
optimizer_params = {
    'position_lr': 0.00016,
    'feature_lr': 0.0025,
    'opacity_lr': 0.05,
    'scaling_lr': 0.005,
    'rotation_lr': 0.001,
}

# Adam optimizer
optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))

# Densification parameters
densify_grad_threshold = 0.0002
densification_interval = 100  # every 100 iterations
densify_until_iter = 15000
```

**Our Modifications:**
```python
# Camera-aware densification
def should_densify(gaussian, K, grad):
    """
    Adaptive densification based on:
    1. Gradient magnitude (standard)
    2. Distance from principal point (new)
    3. Depth uncertainty (new)
    """
    # Standard criterion
    if grad.norm() < densify_grad_threshold:
        return False

    # Distance from optical center
    u, v = project_to_image(gaussian.position, K)
    cx, cy = K[0, 2], K[1, 2]
    dist = sqrt((u - cx)**2 + (v - cy)**2)
    image_diag = sqrt(H**2 + W**2)

    # Farther from center → lower threshold (densify more)
    adaptive_threshold = densify_grad_threshold * (1 + dist / image_diag)

    return grad.norm() > adaptive_threshold
```

---

## Feasibility Study: Weeks 1-3

### Overview

**Goal:** Determine if camera-aware initialization provides measurable improvements over baseline

**Success Criteria:**
- ✅ Camera-aware init shows **+1 dB PSNR** over random init
- ✅ Cross-sensor test shows **<3 dB drop** (vs >5 dB baseline)
- ✅ Implementation is stable and reproducible

**If Successful:** Proceed to full ProjGS development (20+ weeks)

**If Marginal:** Pivot to alternative research direction

---

## Week 1: Environment Setup & Data Pipeline

### Objectives
1. Set up development environment
2. Download and organize SUN RGB-D dataset
3. Implement camera matrix parsing utilities
4. Create data loading pipeline
5. Verify depth unprojection correctness

### Tasks

#### Task 1.1: Environment Setup (Day 1)

**Installation:**
```bash
# Create conda environment
conda create -n projgs python=3.10
conda activate projgs

# Core dependencies
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Gaussian Splatting dependencies
pip install plyfile tqdm

# Install diff-gaussian-rasterization
# Follow: https://github.com/graphdeco-inria/gaussian-splatting
cd submodules/diff-gaussian-rasterization
pip install .
cd ../simple-knn
pip install .

# Data processing
pip install opencv-python pillow h5py scipy

# Visualization
pip install matplotlib seaborn open3d

# Experiment tracking
pip install tensorboard wandb
```

**Reference Implementation:**
- Clone: https://github.com/graphdeco-inria/gaussian-splatting[3]
- Study codebase structure

**Deliverable:** Working environment with all dependencies

#### Task 1.2: SUN RGB-D Dataset Preparation (Day 1-2)

**Download Dataset:**
```bash
# Official download page
# http://rgbd.cs.princeton.edu/

# Expected structure after download:
data/sunrgbd/
├── kv1/NYUdata/
│   └── NYU0428/
│       ├── image/NYU0428.jpg
│       ├── depth/NYU0428.png
│       ├── intrinsics.txt
│       ├── extrinsics/*.txt
│       ├── scene.txt
│       └── annotation2D3D/index.json
├── kv2/align_kv2/
├── realsense/
└── xtion/
```

**Or use preprocessed version:**
```bash
# MMDetection3D format (recommended)
# https://mmdetection3d.readthedocs.io/en/v0.17.1/datasets/sunrgbd_det.html[13]

python tools/create_data.py sunrgbd \
    --root-path ./data/sunrgbd \
    --out-dir ./data/sunrgbd \
    --extra-tag sunrgbd
```

**Dataset Statistics Verification:**
```python
import os
from collections import Counter

def verify_dataset():
    """Verify SUN RGB-D is correctly downloaded"""

    sensors = ['kv1', 'kv2', 'realsense', 'xtion']
    stats = {}

    for sensor in sensors:
        sensor_path = f'data/sunrgbd/{sensor}'
        rgb_count = len(list(Path(sensor_path).rglob('*.jpg')))
        depth_count = len(list(Path(sensor_path).rglob('depth/*.png')))
        intrinsics = len(list(Path(sensor_path).rglob('intrinsics.txt')))

        stats[sensor] = {
            'rgb': rgb_count,
            'depth': depth_count,
            'intrinsics': intrinsics
        }

    print("Dataset Statistics:")
    print(stats)

    # Expected: ~10,335 total scenes across 4 sensors
    total = sum(s['rgb'] for s in stats.values())
    assert total > 10000, f"Expected >10k images, got {total}"

    return stats
```

**Deliverable:** Downloaded dataset with verified file counts

#### Task 1.3: Camera Matrix Parser (Day 2-3)

**Implementation:**
```python
# File: utils/camera.py

import numpy as np
import torch

class CameraIntrinsics:
    """
    Parse and handle camera intrinsics from SUN RGB-D

    SUN RGB-D format (intrinsics.txt):
    fx 0 cx 0 fy cy 0 0 1 (9 values, row-major)

    Reference: SUN RGB-D dataset paper[14]
    """

    def __init__(self, intrinsics_file):
        """Load intrinsics from file"""
        with open(intrinsics_file, 'r') as f:
            values = list(map(float, f.read().strip().split()))

        assert len(values) == 9, f"Expected 9 values, got {len(values)}"

        # Reshape to 3x3 matrix
        self.K = np.array(values).reshape(3, 3)

        # Extract parameters
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]

    def to_tensor(self, device='cuda'):
        """Convert to PyTorch tensor"""
        return torch.from_numpy(self.K).float().to(device)

    def unproject(self, depth, return_tensor=True):
        """
        Unproject depth map to 3D point cloud

        Args:
            depth: (H, W) numpy array or torch tensor
            return_tensor: return torch.Tensor if True

        Returns:
            points: (H*W, 3) or (H, W, 3) point cloud

        Reference: Intel RealSense projection docs[7]
        """
        if isinstance(depth, torch.Tensor):
            depth_np = depth.cpu().numpy()
        else:
            depth_np = depth

        H, W = depth_np.shape

        # Create pixel grid
        u = np.arange(W)
        v = np.arange(H)
        u, v = np.meshgrid(u, v)

        # Unproject to 3D
        X = (u - self.cx) * depth_np / self.fx
        Y = (v - self.cy) * depth_np / self.fy
        Z = depth_np

        points = np.stack([X, Y, Z], axis=-1)  # (H, W, 3)

        if return_tensor:
            return torch.from_numpy(points).float()
        return points

    def project(self, points_3d):
        """
        Project 3D points to 2D image coordinates

        Args:
            points_3d: (N, 3) array/tensor of 3D points

        Returns:
            pixels: (N, 2) image coordinates (u, v)
        """
        if isinstance(points_3d, torch.Tensor):
            points = points_3d.cpu().numpy()
        else:
            points = points_3d

        # Perspective projection
        u = self.fx * (points[:, 0] / points[:, 2]) + self.cx
        v = self.fy * (points[:, 1] / points[:, 2]) + self.cy

        return np.stack([u, v], axis=1)

    def __repr__(self):
        return f"CameraIntrinsics(fx={self.fx:.2f}, fy={self.fy:.2f}, " \
               f"cx={self.cx:.2f}, cy={self.cy:.2f})"


class CameraExtrinsics:
    """Parse camera extrinsics (rotation + translation)"""

    def __init__(self, extrinsics_file):
        """
        Load extrinsics from file

        Format: 3x4 matrix [R|t]
        r11 r12 r13 tx
        r21 r22 r23 ty
        r31 r32 r33 tz
        """
        Rt = np.loadtxt(extrinsics_file)
        assert Rt.shape == (3, 4), f"Expected (3,4), got {Rt.shape}"

        self.R = Rt[:, :3]  # 3x3 rotation
        self.t = Rt[:, 3]   # 3x1 translation

    def transform_points(self, points_3d):
        """Apply extrinsic transformation to points"""
        # P_world = R * P_camera + t
        return (self.R @ points_3d.T).T + self.t
```

**Unit Tests:**
```python
# File: tests/test_camera.py

import pytest
import numpy as np
from utils.camera import CameraIntrinsics

def test_intrinsics_parsing():
    """Test parsing of Kinect v1 intrinsics"""
    # Create mock intrinsics file
    K_values = "518.857901 0.000000 284.582449 0.000000 519.469611 208.736166 0.000000 0.000000 1.000000"

    with open('/tmp/test_intrinsics.txt', 'w') as f:
        f.write(K_values)

    cam = CameraIntrinsics('/tmp/test_intrinsics.txt')

    assert np.isclose(cam.fx, 518.857901)
    assert np.isclose(cam.fy, 519.469611)
    assert np.isclose(cam.cx, 284.582449)
    assert np.isclose(cam.cy, 208.736166)

def test_unprojection():
    """Test depth unprojection to 3D"""
    # Mock camera
    K_values = "500.0 0.0 320.0 0.0 500.0 240.0 0.0 0.0 1.0"
    with open('/tmp/test_intrinsics.txt', 'w') as f:
        f.write(K_values)

    cam = CameraIntrinsics('/tmp/test_intrinsics.txt')

    # Create simple depth map (640x480)
    depth = np.ones((480, 640)) * 2.0  # 2 meters

    points = cam.unproject(depth, return_tensor=False)

    # Check center pixel
    center_point = points[240, 320]  # (cy, cx)

    # At center, X=Y=0, Z=depth
    assert np.isclose(center_point[0], 0.0, atol=1e-3)
    assert np.isclose(center_point[1], 0.0, atol=1e-3)
    assert np.isclose(center_point[2], 2.0)

def test_projection_unprojection_cycle():
    """Test that project(unproject(depth)) ≈ identity"""
    K_values = "500.0 0.0 320.0 0.0 500.0 240.0 0.0 0.0 1.0"
    with open('/tmp/test_intrinsics.txt', 'w') as f:
        f.write(K_values)

    cam = CameraIntrinsics('/tmp/test_intrinsics.txt')

    # Create depth map
    depth = np.random.rand(100, 100) * 5.0 + 0.5  # 0.5-5.5m

    # Unproject to 3D
    points_3d = cam.unproject(depth, return_tensor=False)
    points_3d_flat = points_3d.reshape(-1, 3)

    # Project back to 2D
    pixels = cam.project(points_3d_flat)
    pixels = pixels.reshape(100, 100, 2)

    # Check pixel coordinates match
    u_expected = np.arange(100)
    v_expected = np.arange(100)
    u_grid, v_grid = np.meshgrid(u_expected, v_expected)

    assert np.allclose(pixels[:, :, 0], u_grid, atol=0.1)
    assert np.allclose(pixels[:, :, 1], v_grid, atol=0.1)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

**Run Tests:**
```bash
pytest tests/test_camera.py -v
```

**Deliverable:** Verified camera matrix utilities with passing tests

#### Task 1.4: Data Loader Implementation (Day 3-4)

**Implementation:**
```python
# File: data/sunrgbd_loader.py

import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from utils.camera import CameraIntrinsics

class SUNRGBDDataset(Dataset):
    """
    SUN RGB-D Dataset Loader

    Returns:
        - RGB image (H, W, 3)
        - Depth map (H, W)
        - Camera intrinsics K (3, 3)
        - Scene category (string)
        - Point cloud (N, 3) from depth unprojection

    Reference: SUN RGB-D paper[14], MMDetection3D loader[13]
    """

    def __init__(self, root_dir, split='train', sensors=None,
                 max_depth=10.0, resize=None):
        """
        Args:
            root_dir: Path to data/sunrgbd/
            split: 'train', 'val', or 'test'
            sensors: List of sensors or None for all
                    ['kv1', 'kv2', 'realsense', 'xtion']
            max_depth: Maximum valid depth (meters)
            resize: Tuple (H, W) to resize images, or None
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.max_depth = max_depth
        self.resize = resize

        if sensors is None:
            sensors = ['kv1', 'kv2', 'realsense', 'xtion']
        self.sensors = sensors

        # Find all scenes
        self.scenes = self._collect_scenes()

        # Train/val/test split (70/10/20)
        np.random.seed(42)
        indices = np.random.permutation(len(self.scenes))

        n_train = int(0.7 * len(self.scenes))
        n_val = int(0.1 * len(self.scenes))

        if split == 'train':
            self.scenes = [self.scenes[i] for i in indices[:n_train]]
        elif split == 'val':
            self.scenes = [self.scenes[i] for i in indices[n_train:n_train+n_val]]
        else:  # test
            self.scenes = [self.scenes[i] for i in indices[n_train+n_val:]]

        print(f"Loaded {len(self.scenes)} scenes for {split} split")

    def _collect_scenes(self):
        """Collect all valid scene paths"""
        scenes = []

        for sensor in self.sensors:
            sensor_path = self.root_dir / sensor

            # Find all scene directories
            for scene_dir in sensor_path.rglob('*'):
                if not scene_dir.is_dir():
                    continue

                # Check required files exist
                rgb_file = scene_dir / 'image' / f'{scene_dir.name}.jpg'
                depth_file = scene_dir / 'depth' / f'{scene_dir.name}.png'
                intrinsics_file = scene_dir / 'intrinsics.txt'
                scene_file = scene_dir / 'scene.txt'

                if all([rgb_file.exists(), depth_file.exists(),
                       intrinsics_file.exists(), scene_file.exists()]):
                    scenes.append({
                        'sensor': sensor,
                        'scene_dir': scene_dir,
                        'rgb': rgb_file,
                        'depth': depth_file,
                        'intrinsics': intrinsics_file,
                        'scene_category': scene_file
                    })

        return scenes

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        """Load and return scene data"""
        scene = self.scenes[idx]

        # Load RGB
        rgb = np.array(Image.open(scene['rgb']))

        # Load depth (PNG stores depth in mm, convert to meters)
        depth_img = Image.open(scene['depth'])
        depth = np.array(depth_img).astype(np.float32) / 1000.0

        # Clip depth
        depth = np.clip(depth, 0, self.max_depth)

        # Load camera intrinsics
        camera = CameraIntrinsics(scene['intrinsics'])
        K = camera.K

        # Load scene category
        with open(scene['scene_category'], 'r') as f:
            scene_category = f.read().strip()

        # Unproject depth to point cloud
        points_3d = camera.unproject(depth, return_tensor=True)  # (H, W, 3)

        # Create validity mask
        valid_mask = (depth > 0) & (depth < self.max_depth)

        # Resize if requested
        if self.resize is not None:
            H_new, W_new = self.resize
            rgb = np.array(Image.fromarray(rgb).resize((W_new, H_new)))
            depth = np.array(Image.fromarray(depth).resize((W_new, H_new)))
            valid_mask = np.array(Image.fromarray(valid_mask.astype(np.uint8)).resize((W_new, H_new))) > 0

            # Adjust intrinsics
            H_orig, W_orig = points_3d.shape[:2]
            K = K.copy()
            K[0, :] *= (W_new / W_orig)  # fx, cx
            K[1, :] *= (H_new / H_orig)  # fy, cy

        return {
            'rgb': torch.from_numpy(rgb).float() / 255.0,  # [0, 1]
            'depth': torch.from_numpy(depth).float(),
            'K': torch.from_numpy(K).float(),
            'scene_category': scene_category,
            'sensor': scene['sensor'],
            'valid_mask': torch.from_numpy(valid_mask),
            'scene_id': scene['scene_dir'].name,
        }


def visualize_sample(sample):
    """Visualize a dataset sample"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # RGB
    axes[0].imshow(sample['rgb'])
    axes[0].set_title(f"RGB - {sample['scene_category']}")
    axes[0].axis('off')

    # Depth
    depth_vis = sample['depth'].numpy()
    depth_vis[~sample['valid_mask']] = 0
    im = axes[1].imshow(depth_vis, cmap='jet', vmin=0, vmax=5)
    axes[1].set_title(f"Depth - Sensor: {sample['sensor']}")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])

    # Camera info
    K = sample['K'].numpy()
    info_text = f"Camera Intrinsics:\n"
    info_text += f"fx = {K[0,0]:.2f}\n"
    info_text += f"fy = {K[1,1]:.2f}\n"
    info_text += f"cx = {K[0,2]:.2f}\n"
    info_text += f"cy = {K[1,2]:.2f}\n"
    info_text += f"\nScene: {sample['scene_category']}\n"
    info_text += f"Sensor: {sample['sensor']}\n"
    info_text += f"ID: {sample['scene_id']}"

    axes[2].text(0.1, 0.5, info_text, fontsize=12, family='monospace')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('outputs/dataset_sample.png', dpi=150, bbox_inches='tight')
    print(f"Saved visualization to outputs/dataset_sample.png")
```

**Test Data Loader:**
```python
# File: scripts/test_dataloader.py

from data.sunrgbd_loader import SUNRGBDDataset, visualize_sample

# Create dataset
dataset = SUNRGBDDataset(
    root_dir='data/sunrgbd',
    split='train',
    sensors=['kv1'],  # Start with one sensor
    resize=(480, 640)
)

print(f"Dataset size: {len(dataset)}")

# Load sample
sample = dataset[0]

print("\nSample keys:", sample.keys())
print(f"RGB shape: {sample['rgb'].shape}")
print(f"Depth shape: {sample['depth'].shape}")
print(f"K matrix:\n{sample['K']}")
print(f"Scene: {sample['scene_category']}")

# Visualize
os.makedirs('outputs', exist_ok=True)
visualize_sample(sample)
```

**Run:**
```bash
python scripts/test_dataloader.py
```

**Deliverable:** Working data loader with visualization

#### Task 1.5: Verification (Day 4-5)

**Verify Unprojection Accuracy:**
```python
# File: scripts/verify_unprojection.py

import numpy as np
import open3d as o3d
from data.sunrgbd_loader import SUNRGBDDataset
from utils.camera import CameraIntrinsics

def verify_unprojection():
    """
    Verify camera-aware unprojection produces metric 3D points

    Test: Project 3D point cloud back to 2D, should match depth map
    """
    # Load sample
    dataset = SUNRGBDDataset('data/sunrgbd', split='train', sensors=['kv1'])
    sample = dataset[0]

    depth = sample['depth'].numpy()
    K = sample['K'].numpy()

    # Unproject using camera
    camera = CameraIntrinsics.__new__(CameraIntrinsics)
    camera.K = K
    camera.fx, camera.fy = K[0, 0], K[1, 1]
    camera.cx, camera.cy = K[0, 2], K[1, 2]

    points_3d = camera.unproject(depth, return_tensor=False)  # (H, W, 3)

    # Project back to 2D
    H, W = depth.shape
    points_flat = points_3d.reshape(-1, 3)
    pixels = camera.project(points_flat).reshape(H, W, 2)

    # Expected pixel coordinates
    u_expected = np.arange(W)
    v_expected = np.arange(H)
    u_grid, v_grid = np.meshgrid(u_expected, v_expected)

    # Check error
    u_error = np.abs(pixels[:, :, 0] - u_grid)
    v_error = np.abs(pixels[:, :, 1] - v_grid)

    valid_mask = sample['valid_mask'].numpy()

    print("Reprojection Error:")
    print(f"  Mean U error: {u_error[valid_mask].mean():.6f} pixels")
    print(f"  Mean V error: {v_error[valid_mask].mean():.6f} pixels")
    print(f"  Max U error: {u_error[valid_mask].max():.6f} pixels")
    print(f"  Max V error: {v_error[valid_mask].max():.6f} pixels")

    # Should be < 0.01 pixels (numerical precision)
    assert u_error[valid_mask].mean() < 0.01
    assert v_error[valid_mask].mean() < 0.01

    print("\n✓ Unprojection verified!")

    # Visualize 3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_flat[valid_mask.flatten()])

    rgb = sample['rgb'].numpy()
    colors = rgb.reshape(-1, 3)[valid_mask.flatten()]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Save
    o3d.io.write_point_cloud("outputs/pointcloud_sample.ply", pcd)
    print("Saved point cloud to outputs/pointcloud_sample.ply")

    # Visualize
    o3d.visualization.draw_geometries([pcd])

if __name__ == '__main__':
    verify_unprojection()
```

**Run:**
```bash
python scripts/verify_unprojection.py
```

**Expected Output:**
```
Reprojection Error:
  Mean U error: 0.000001 pixels
  Mean V error: 0.000001 pixels
  Max U error: 0.000012 pixels
  Max V error: 0.000015 pixels

✓ Unprojection verified!
Saved point cloud to outputs/pointcloud_sample.ply
```

**Deliverable:** Verified unprojection with <0.01 pixel error

### Week 1 Deliverables Checklist

- [ ] Development environment set up with all dependencies
- [ ] SUN RGB-D dataset downloaded (>10k scenes)
- [ ] Camera matrix parser implemented and tested
- [ ] Data loader working with visualization
- [ ] Unprojection verified (<0.01 pixel reprojection error)
- [ ] Point cloud visualization generated

**Success Metric:** All unit tests pass, visualizations look correct

---

## Week 2: Camera-Aware Gaussian Initialization

### Objectives
1. Implement focal-length-adaptive Gaussian scaling
2. Compare camera-aware vs naive initialization
3. Implement baseline 3DGS on SUN RGB-D
4. Measure initialization quality metrics

### Tasks

#### Task 2.1: Focal-Length-Adaptive Gaussian Initialization (Day 6-8)

**Implementation:**
```python
# File: models/gaussian_init.py

import torch
import torch.nn as nn
import numpy as np
from utils.camera import CameraIntrinsics

class CameraAwareGaussianInit:
    """
    Camera-aware Gaussian initialization for 3DGS

    Key innovations:
    1. Precise depth unprojection using K matrix
    2. Focal-length-adaptive scaling
    3. Principal-point-aware uncertainty weighting

    References:
    - SA-GS scale-adaptive work[10]
    - RAIN-GS initialization strategies[11]
    """

    def __init__(self, base_scale=0.01, uncertainty_weight=True):
        """
        Args:
            base_scale: Base Gaussian scale multiplier
            uncertainty_weight: Use position uncertainty for opacity
        """
        self.base_scale = base_scale
        self.uncertainty_weight = uncertainty_weight

    def initialize_from_rgbd(self, rgb, depth, K, device='cuda'):
        """
        Initialize Gaussians from single RGB-D frame

        Args:
            rgb: (H, W, 3) RGB image [0, 1]
            depth: (H, W) depth map in meters
            K: (3, 3) camera intrinsic matrix

        Returns:
            gaussians: dict with keys:
                - positions: (N, 3) 3D positions
                - scales: (N, 3) per-axis scales
                - rotations: (N, 4) quaternions
                - colors: (N, 3) RGB colors
                - opacities: (N, 1) opacity values
        """
        H, W = depth.shape

        # Extract camera parameters
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        f_mean = (fx + fy) / 2

        # Create camera object for unprojection
        camera = CameraIntrinsics.__new__(CameraIntrinsics)
        camera.K = K.cpu().numpy()
        camera.fx, camera.fy = fx.item(), fy.item()
        camera.cx, camera.cy = cx.item(), cy.item()

        # Unproject depth to 3D
        points_3d = camera.unproject(depth.cpu().numpy(), return_tensor=True)  # (H, W, 3)
        points_3d = points_3d.to(device)

        # Create validity mask
        valid_mask = (depth > 0) & (depth < 10.0)

        # Flatten valid points
        positions = points_3d[valid_mask]  # (N, 3)
        colors = rgb[valid_mask]  # (N, 3)

        N = positions.shape[0]

        # === Focal-Length-Adaptive Scaling ===
        Z = positions[:, 2]  # depth component

        scale_x = self.base_scale * Z / fx
        scale_y = self.base_scale * Z / fy
        scale_z = self.base_scale * Z / f_mean

        scales = torch.stack([scale_x, scale_y, scale_z], dim=1)  # (N, 3)

        # === Position Uncertainty (Optional) ===
        if self.uncertainty_weight:
            # Get pixel coordinates
            u_coords = torch.arange(W, device=device).repeat(H, 1)[valid_mask]
            v_coords = torch.arange(H, device=device).unsqueeze(1).repeat(1, W)[valid_mask]

            # Distance from principal point
            dist_from_center = torch.sqrt((u_coords - cx)**2 + (v_coords - cy)**2)
            image_diag = torch.sqrt(torch.tensor(H**2 + W**2, device=device))

            # Uncertainty increases with:
            # 1. Distance from center (lens distortion)
            # 2. Depth squared (perspective effect)
            uncertainty = (Z**2 / f_mean) * (1 + dist_from_center / image_diag)

            # Normalize to [0, 1]
            uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + 1e-8)

            # High uncertainty → lower opacity
            opacities = 1.0 - 0.5 * uncertainty  # range [0.5, 1.0]
        else:
            opacities = torch.ones(N, device=device) * 0.9

        opacities = opacities.unsqueeze(1)  # (N, 1)

        # === Initialize Rotations (Identity) ===
        # Quaternion format: [w, x, y, z]
        rotations = torch.zeros((N, 4), device=device)
        rotations[:, 0] = 1.0  # w = 1 (identity rotation)

        return {
            'positions': positions,
            'scales': scales,
            'rotations': rotations,
            'colors': colors,
            'opacities': opacities,
            'count': N
        }

    def initialize_naive(self, rgb, depth, device='cuda'):
        """
        Naive initialization (no camera awareness)

        Baseline comparison method
        """
        H, W = depth.shape

        # Create pixel grid (wrong: ignores camera!)
        u = torch.arange(W, device=device).float()
        v = torch.arange(H, device=device).float()
        u_grid, v_grid = torch.meshgrid(u, v, indexing='xy')

        # Naive unprojection (assumes fx=fy=1, cx=cy=0)
        X = u_grid * depth
        Y = v_grid * depth
        Z = depth

        points_3d = torch.stack([X, Y, Z], dim=-1)

        valid_mask = (depth > 0) & (depth < 10.0)

        positions = points_3d[valid_mask]
        colors = rgb[valid_mask]
        N = positions.shape[0]

        # Uniform scaling (no camera adaptation)
        scales = torch.ones((N, 3), device=device) * self.base_scale

        # Identity rotations
        rotations = torch.zeros((N, 4), device=device)
        rotations[:, 0] = 1.0

        # Uniform opacity
        opacities = torch.ones((N, 1), device=device) * 0.9

        return {
            'positions': positions,
            'scales': scales,
            'rotations': rotations,
            'colors': colors,
            'opacities': opacities,
            'count': N
        }
```

**Comparison Script:**
```python
# File: scripts/compare_initialization.py

import torch
import matplotlib.pyplot as plt
from data.sunrgbd_loader import SUNRGBDDataset
from models.gaussian_init import CameraAwareGaussianInit

def compare_initialization():
    """Compare camera-aware vs naive initialization"""

    # Load sample
    dataset = SUNRGBDDataset('data/sunrgbd', split='train', sensors=['kv1'])
    sample = dataset[0]

    rgb = sample['rgb'].cuda()
    depth = sample['depth'].cuda()
    K = sample['K'].cuda()

    initializer = CameraAwareGaussianInit(base_scale=0.01)

    # Camera-aware initialization
    gaussians_aware = initializer.initialize_from_rgbd(rgb, depth, K)

    # Naive initialization
    gaussians_naive = initializer.initialize_naive(rgb, depth)

    print("=== Initialization Comparison ===\n")

    print(f"Number of Gaussians:")
    print(f"  Camera-aware: {gaussians_aware['count']}")
    print(f"  Naive: {gaussians_naive['count']}\n")

    print(f"Position Statistics:")
    print(f"  Camera-aware mean: {gaussians_aware['positions'].mean(0).cpu().numpy()}")
    print(f"  Naive mean: {gaussians_naive['positions'].mean(0).cpu().numpy()}\n")

    print(f"Scale Statistics:")
    print(f"  Camera-aware mean: {gaussians_aware['scales'].mean(0).cpu().numpy()}")
    print(f"  Camera-aware std: {gaussians_aware['scales'].std(0).cpu().numpy()}")
    print(f"  Naive mean: {gaussians_naive['scales'].mean(0).cpu().numpy()}")
    print(f"  Naive std: {gaussians_naive['scales'].std(0).cpu().numpy()}\n")

    print(f"Opacity Statistics:")
    print(f"  Camera-aware mean: {gaussians_aware['opacities'].mean():.4f}")
    print(f"  Camera-aware std: {gaussians_aware['opacities'].std():.4f}")
    print(f"  Naive mean: {gaussians_naive['opacities'].mean():.4f}\n")

    # Visualize scale distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i, axis_name in enumerate(['X', 'Y', 'Z']):
        axes[i].hist(gaussians_aware['scales'][:, i].cpu().numpy(),
                     bins=50, alpha=0.5, label='Camera-aware', color='blue')
        axes[i].hist(gaussians_naive['scales'][:, i].cpu().numpy(),
                     bins=50, alpha=0.5, label='Naive', color='red')
        axes[i].set_xlabel(f'Scale {axis_name}')
        axes[i].set_ylabel('Count')
        axes[i].legend()
        axes[i].set_title(f'{axis_name}-axis Scale Distribution')

    plt.tight_layout()
    plt.savefig('outputs/initialization_comparison.png', dpi=150)
    print("Saved comparison to outputs/initialization_comparison.png")

if __name__ == '__main__':
    compare_initialization()
```

**Run:**
```bash
python scripts/compare_initialization.py
```

**Expected Output:**
```
=== Initialization Comparison ===

Number of Gaussians:
  Camera-aware: 245632
  Naive: 245632

Position Statistics:
  Camera-aware mean: [-0.023  0.145  2.341]
  Naive mean: [152.3  123.8  2.341]  ← Wrong! Should be in meters

Scale Statistics:
  Camera-aware mean: [0.0045 0.0044 0.0045]
  Camera-aware std: [0.0023 0.0022 0.0023]  ← Adaptive!
  Naive mean: [0.0100 0.0100 0.0100]
  Naive std: [0.0000 0.0000 0.0000]  ← Uniform

Opacity Statistics:
  Camera-aware mean: 0.7823  ← Uncertainty-weighted
  Naive mean: 0.9000
```

**Deliverable:** Camera-aware initialization with adaptive scaling

#### Task 2.2: Baseline 3DGS Implementation (Day 8-10)

**Goal:** Get standard 3DGS running on SUN RGB-D as baseline

**Approach:**
- Use official 3DGS implementation[3]
- Adapt for single-view RGB-D input
- Measure reconstruction quality

**Modified Training Script:**
```python
# File: scripts/train_baseline_3dgs.py

import torch
import torch.nn.functional as F
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from data.sunrgbd_loader import SUNRGBDDataset
from models.gaussian_init import CameraAwareGaussianInit
from torch.utils.tensorboard import SummaryWriter

def train_baseline_3dgs(use_camera_aware=True, num_iterations=1000):
    """
    Train baseline 3DGS on single RGB-D scene

    Args:
        use_camera_aware: Use camera-aware init (True) or naive (False)
        num_iterations: Training iterations
    """
    device = 'cuda'

    # Load single scene for proof-of-concept
    dataset = SUNRGBDDataset('data/sunrgbd', split='train', sensors=['kv1'])
    sample = dataset[0]

    rgb_gt = sample['rgb'].cuda()
    depth_gt = sample['depth'].cuda()
    K = sample['K'].cuda()
    H, W = rgb_gt.shape[:2]

    print(f"Training on scene: {sample['scene_id']}")
    print(f"Image size: {H}x{W}")
    print(f"Camera intrinsics:\n{K.cpu().numpy()}")

    # Initialize Gaussians
    initializer = CameraAwareGaussianInit(base_scale=0.01)

    if use_camera_aware:
        print("\nUsing CAMERA-AWARE initialization")
        gaussians = initializer.initialize_from_rgbd(rgb_gt, depth_gt, K)
    else:
        print("\nUsing NAIVE initialization")
        gaussians = initializer.initialize_naive(rgb_gt, depth_gt)

    # Convert to optimizable parameters
    positions = nn.Parameter(gaussians['positions'])
    scales = nn.Parameter(torch.log(gaussians['scales']))  # log space
    rotations = nn.Parameter(gaussians['rotations'])
    colors = nn.Parameter(gaussians['colors'])
    opacities = nn.Parameter(torch.logit(gaussians['opacities']))  # logit space

    # Optimizer (follow 3DGS paper hyperparameters[3])
    optimizer = torch.optim.Adam([
        {'params': [positions], 'lr': 0.00016, 'name': 'positions'},
        {'params': [scales], 'lr': 0.005, 'name': 'scales'},
        {'params': [rotations], 'lr': 0.001, 'name': 'rotations'},
        {'params': [colors], 'lr': 0.0025, 'name': 'colors'},
        {'params': [opacities], 'lr': 0.05, 'name': 'opacities'},
    ], betas=(0.9, 0.999))

    # Tensorboard
    writer = SummaryWriter(f'runs/baseline_{"camera_aware" if use_camera_aware else "naive"}')

    # Training loop
    for iteration in range(num_iterations):
        optimizer.zero_grad()

        # Rasterization settings
        tanfovx = W / (2 * K[0, 0])
        tanfovy = H / (2 * K[1, 1])

        raster_settings = GaussianRasterizationSettings(
            image_height=H,
            image_width=W,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=torch.tensor([0, 0, 0], device=device),
            scale_modifier=1.0,
            viewmatrix=torch.eye(4, device=device),  # Identity (camera at origin)
            projmatrix=torch.eye(4, device=device),  # Will be computed from K
            sh_degree=0,  # No spherical harmonics for simplicity
            campos=torch.tensor([0, 0, 0], device=device),
            prefiltered=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        # Render
        rendered_image, radii = rasterizer(
            means3D=positions,
            means2D=None,
            shs=None,
            colors_precomp=colors,
            opacities=torch.sigmoid(opacities),
            scales=torch.exp(scales),
            rotations=F.normalize(rotations),
            cov3D_precomp=None
        )

        # Loss (L1 + D-SSIM)[3][12]
        l1_loss = F.l1_loss(rendered_image, rgb_gt.permute(2, 0, 1))

        # Simplified SSIM (use full implementation in production)
        ssim_loss = 1 - torch.mean((rendered_image - rgb_gt.permute(2, 0, 1))**2)

        loss = (1 - 0.2) * l1_loss + 0.2 * ssim_loss

        loss.backward()
        optimizer.step()

        # Logging
        if iteration % 50 == 0:
            psnr = -10 * torch.log10(F.mse_loss(rendered_image, rgb_gt.permute(2, 0, 1)))
            print(f"Iter {iteration:04d} | Loss: {loss.item():.4f} | PSNR: {psnr.item():.2f} dB")

            writer.add_scalar('Loss/total', loss.item(), iteration)
            writer.add_scalar('Metrics/PSNR', psnr.item(), iteration)
            writer.add_scalar('Metrics/L1', l1_loss.item(), iteration)

        # Save visualization
        if iteration % 200 == 0:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            axes[0].imshow(rgb_gt.cpu().numpy())
            axes[0].set_title('Ground Truth')
            axes[0].axis('off')

            axes[1].imshow(rendered_image.permute(1, 2, 0).detach().cpu().numpy())
            axes[1].set_title(f'Rendered (Iter {iteration})')
            axes[1].axis('off')

            plt.tight_layout()
            plt.savefig(f'outputs/render_iter_{iteration:04d}.png', dpi=100)
            plt.close()

    print(f"\nTraining complete!")

    # Final metrics
    with torch.no_grad():
        final_psnr = -10 * torch.log10(F.mse_loss(rendered_image, rgb_gt.permute(2, 0, 1)))
        print(f"Final PSNR: {final_psnr.item():.2f} dB")

    return final_psnr.item()

if __name__ == '__main__':
    # Train both versions
    print("=" * 60)
    print("Training with CAMERA-AWARE initialization")
    print("=" * 60)
    psnr_aware = train_baseline_3dgs(use_camera_aware=True, num_iterations=1000)

    print("\n" + "=" * 60)
    print("Training with NAIVE initialization")
    print("=" * 60)
    psnr_naive = train_baseline_3dgs(use_camera_aware=False, num_iterations=1000)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Camera-aware PSNR: {psnr_aware:.2f} dB")
    print(f"Naive PSNR: {psnr_naive:.2f} dB")
    print(f"Improvement: {psnr_aware - psnr_naive:.2f} dB")
```

**NOTE:** This is a simplified version. Full 3DGS requires:
- Proper rasterization setup (use official implementation)
- Densification and pruning
- Depth regularization
- Proper SSIM implementation

**Deliverable:** Baseline 3DGS running on SUN RGB-D

### Week 2 Deliverables Checklist

- [ ] Focal-length-adaptive scaling implemented
- [ ] Initialization comparison showing differences
- [ ] Baseline 3DGS adapted for SUN RGB-D
- [ ] Initial PSNR measurements recorded

**Success Metric:** Camera-aware init shows visible difference in scale distributions

---

## Week 3: Feasibility Validation & Decision

### Objectives
1. Run comprehensive comparison: camera-aware vs naive
2. Test on multiple scenes from different sensors
3. Measure cross-sensor performance
4. **Make go/no-go decision**

### Tasks

#### Task 3.1: Multi-Scene Evaluation (Day 11-12)

**Script:**
```python
# File: scripts/evaluate_initialization.py

import torch
import numpy as np
from tqdm import tqdm
from data.sunrgbd_loader import SUNRGBDDataset
from models.gaussian_init import CameraAwareGaussianInit

def evaluate_initialization_quality(dataset, num_samples=50):
    """
    Evaluate initialization quality metrics

    Metrics:
    1. Point cloud coverage (% of scene covered)
    2. Scale distribution variance
    3. Opacity distribution
    """
    initializer = CameraAwareGaussianInit()

    results = {
        'camera_aware': {'coverage': [], 'scale_var': [], 'opacity_mean': []},
        'naive': {'coverage': [], 'scale_var': [], 'opacity_mean': []}
    }

    for i in tqdm(range(min(num_samples, len(dataset)))):
        sample = dataset[i]

        rgb = sample['rgb'].cuda()
        depth = sample['depth'].cuda()
        K = sample['K'].cuda()

        # Camera-aware
        g_aware = initializer.initialize_from_rgbd(rgb, depth, K)

        # Naive
        g_naive = initializer.initialize_naive(rgb, depth)

        # Compute metrics
        for method, gaussians in [('camera_aware', g_aware), ('naive', g_naive)]:
            # Coverage: number of Gaussians
            coverage = gaussians['count']
            results[method]['coverage'].append(coverage)

            # Scale variance (should be higher for adaptive)
            scale_var = gaussians['scales'].var().item()
            results[method]['scale_var'].append(scale_var)

            # Opacity mean
            opacity_mean = gaussians['opacities'].mean().item()
            results[method]['opacity_mean'].append(opacity_mean)

    # Print statistics
    print("\n=== Initialization Quality Evaluation ===\n")

    for metric in ['coverage', 'scale_var', 'opacity_mean']:
        print(f"{metric.upper()}:")
        for method in ['camera_aware', 'naive']:
            values = np.array(results[method][metric])
            print(f"  {method:15s}: {values.mean():.4f} ± {values.std():.4f}")
        print()

    return results

if __name__ == '__main__':
    dataset = SUNRGBDDataset('data/sunrgbd', split='val', sensors=['kv1', 'kv2'])
    results = evaluate_initialization_quality(dataset, num_samples=50)
```

**Deliverable:** Quantitative comparison on 50 scenes

#### Task 3.2: Cross-Sensor Evaluation (Day 13)

**Goal:** Measure if camera-aware init generalizes across sensors

**Script:**
```python
# File: scripts/cross_sensor_test.py

from data.sunrgbd_loader import SUNRGBDDataset
from models.gaussian_init import CameraAwareGaussianInit

def cross_sensor_test():
    """
    Test: Train on Kinect v1, test on RealSense

    Hypothesis: Camera-aware should have smaller performance drop
    """

    # Load different sensors
    dataset_kv1 = SUNRGBDDataset('data/sunrgbd', split='train', sensors=['kv1'])
    dataset_realsense = SUNRGBDDataset('data/sunrgbd', split='train', sensors=['realsense'])

    # Sample from each
    sample_kv1 = dataset_kv1[0]
    sample_realsense = dataset_realsense[0] if len(dataset_realsense) > 0 else None

    if sample_realsense is None:
        print("No RealSense data found, using kv2 instead")
        dataset_realsense = SUNRGBDDataset('data/sunrgbd', split='train', sensors=['kv2'])
        sample_realsense = dataset_realsense[0]

    print(f"Kinect v1 intrinsics:\n{sample_kv1['K']}\n")
    print(f"RealSense intrinsics:\n{sample_realsense['K']}\n")

    # Initialize with both methods
    initializer = CameraAwareGaussianInit()

    # Kinect v1
    g_kv1_aware = initializer.initialize_from_rgbd(
        sample_kv1['rgb'].cuda(),
        sample_kv1['depth'].cuda(),
        sample_kv1['K'].cuda()
    )

    # RealSense
    g_rs_aware = initializer.initialize_from_rgbd(
        sample_realsense['rgb'].cuda(),
        sample_realsense['depth'].cuda(),
        sample_realsense['K'].cuda()
    )

    print("Scale statistics comparison:")
    print(f"Kinect v1 scale mean: {g_kv1_aware['scales'].mean(0).cpu().numpy()}")
    print(f"RealSense scale mean: {g_rs_aware['scales'].mean(0).cpu().numpy()}")

    # Expected: Scales should adapt to different focal lengths

if __name__ == '__main__':
    cross_sensor_test()
```

**Deliverable:** Cross-sensor initialization comparison

#### Task 3.3: Go/No-Go Decision (Day 14-15)

**Decision Criteria:**

| Metric | Success Threshold | Result | Status |
|--------|------------------|--------|--------|
| **Camera-aware vs Naive** | +1 dB PSNR improvement | __ dB | ⬜ |
| **Scale variance** | 2x higher than naive | __x | ⬜ |
| **Cross-sensor drop** | <3 dB (vs >5 dB naive) | __ dB | ⬜ |
| **Implementation stability** | No crashes/NaNs | ✓/✗ | ⬜ |

**Decision Matrix:**
```
If ALL criteria met → ✅ GO: Proceed to full ProjGS development

If 3/4 criteria met → ⚠️ CAUTION: Refine approach, re-test

If <3 criteria met → ❌ NO-GO: Pivot to alternative research direction
```

**Alternative Pivots if NO-GO:**
1. **Camera-Conditioned Diffusion** (GAP 5)
2. **VLM-Guided Reconstruction** (GAP 3)
3. **Uncertainty-Aware GS** (different angle)

### Week 3 Deliverables Checklist

- [ ] Multi-scene evaluation complete (50+ scenes)
- [ ] Cross-sensor test results documented
- [ ] Decision criteria table filled out
- [ ] Go/no-go decision made with justification

**Success Metric:** Clear evidence (quantitative + qualitative) that camera-aware init works

---

## Verification Criteria

### Technical Verification

**Week 1:**
- [ ] Unprojection error < 0.01 pixels
- [ ] Point clouds visualize correctly in Open3D
- [ ] All unit tests pass

**Week 2:**
- [ ] Scale distributions show variance (not uniform)
- [ ] Camera-aware scales correlate with focal length
- [ ] Baseline 3DGS renders reasonable images

**Week 3:**
- [ ] Consistent improvement across multiple scenes
- [ ] Cross-sensor generalization demonstrated
- [ ] No numerical instabilities

### Quantitative Targets

**Minimum Success:**
- Camera-aware PSNR > Naive PSNR + 0.5 dB
- Scale variance ratio > 1.5x
- Cross-sensor drop < 4 dB

**Strong Success:**
- Camera-aware PSNR > Naive PSNR + 2.0 dB
- Scale variance ratio > 3x
- Cross-sensor drop < 2 dB

---

## References

[1] FreeSplatter (2024). "Pose-free Gaussian Splatting for Sparse-view 3D Reconstruction"
    https://openreview.net/forum?id=VpGsy4hKMc

[2] CF-3DGS (2024). "COLMAP-Free 3D Gaussian Splatting"
    https://arxiv.org/html/2312.07504v1

[3] Kerbl et al. (SIGGRAPH 2023). "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
    https://github.com/graphdeco-inria/gaussian-splatting

[4] RAIN-GS (2024). "Relaxing Accurate Initialization Constraint for 3D Gaussian Splatting"
    https://arxiv.org/html/2403.09413v2

[5] Chung et al. (CVPR 2024 Workshop). "Depth-Regularized Optimization for 3D Gaussian Splatting in Few-Shot Images"
    https://openaccess.thecvf.com/content/CVPR2024W/3DMV/papers/Chung_Depth-Regularized_Optimization_for_3D_Gaussian_Splatting_in_Few-Shot_Images_CVPRW_2024_paper.pdf

[6] IndoorGS (CVPR 2025). "Geometric Cues Guided Gaussian Splatting for Indoor Scenes"
    https://cvpr.thecvf.com/virtual/2025/poster/33248

[7] Intel RealSense. "Projection, Texture-Mapping and Occlusion with Depth Cameras"
    https://dev.intelrealsense.com/docs/projection-texture-mapping-and-occlusion-with-intel-realsense-depth-cameras

[8] CoL3D (February 2025). "Collaborative Learning of Single-view Depth and Camera Intrinsics"
    https://arxiv.org/abs/2502.08902

[9] TrackGS (2025). "Optimizing COLMAP-Free 3D Gaussian Splatting with Global Track Constraints"
    https://arxiv.org/abs/2502.19800

[10] SA-GS. "Scale-Adaptive Gaussian Splatting"
     https://github.com/zsy1987/SA-GS

[11] EasySplat (2025). "View-Adaptive Learning makes 3D Gaussian Splatting Easy"
     https://arxiv.org/html/2501.01003

[12] "3D Gaussian Splatting as Markov Chain Monte Carlo" (NeurIPS 2024)
     https://proceedings.neurips.cc/paper_files/paper/2024/file/93be245fce00a9bb2333c17ceae4b732-Paper-Conference.pdf

[13] MMDetection3D. "SUN RGB-D for 3D Object Detection"
     https://mmdetection3d.readthedocs.io/en/v0.17.1/datasets/sunrgbd_det.html

[14] Song et al. (CVPR 2015). "SUN RGB-D: A RGB-D Scene Understanding Benchmark Suite"
     https://rgbd.cs.princeton.edu/

---

## Appendix: Troubleshooting

### Common Issues

**Issue 1: CUDA Out of Memory**
```
Solution: Reduce image resolution or downsample point cloud
dataset = SUNRGBDDataset(..., resize=(240, 320))
```

**Issue 2: NaN in Gradients**
```
Solution: Check for zero depths, add epsilon to divisions
scales = torch.exp(scales).clamp(min=1e-6)
```

**Issue 3: Poor Rendering Quality**
```
Solution: Verify camera matrix is correct
print(f"fx/fy ratio: {K[0,0] / K[1,1]}")  # Should be ~1.0
```

**Issue 4: Dataset Not Found**
```
Solution: Check path and download
assert Path('data/sunrgbd/kv1').exists()
```

---

## Next Steps After Week 3

**If GO Decision:**
- Proceed to full ProjGS implementation (Weeks 4-24)
- Implement novel pose prediction network
- Add reprojection consistency loss
- Conduct full benchmark evaluation

**If NO-GO Decision:**
- Document findings
- Pivot to alternative research direction
- Re-evaluate with different approach

---

**Document Version:** 1.0
**Date:** December 25, 2025
**Status:** Ready for Implementation
