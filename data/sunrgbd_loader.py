"""
SUN RGB-D Dataset Loader
=========================

PyTorch Dataset class for the SUN RGB-D benchmark with camera-aware loading.

Features:
- Loads RGB images, depth maps, and camera intrinsics
- Supports all 4 sensors: Kinect v1, Kinect v2, RealSense, Asus Xtion
- Camera-aware depth unprojection to 3D point clouds
- Train/val/test splits with reproducible random seed
- Image resizing with intrinsic matrix adjustment
- Efficient caching and preprocessing

Dataset Structure:
    data/sunrgbd/
    ├── kv1/NYUdata/          # Kinect v1
    ├── kv2/align_kv2/        # Kinect v2
    ├── realsense/            # Intel RealSense
    └── xtion/                # Asus Xtion

References:
- SUN RGB-D Paper: Song et al. "SUN RGB-D: A RGB-D Scene Understanding Benchmark Suite" (CVPR 2015)
- Dataset: http://rgbd.cs.princeton.edu/
- MMDetection3D Loader: https://mmdetection3d.readthedocs.io/

Author: ProjGS Research Team
Date: December 2025
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
import warnings

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from utils.camera import CameraIntrinsics


class SUNRGBDDataset(Dataset):
    """
    PyTorch Dataset for SUN RGB-D with camera-aware loading.

    Returns dictionary with:
        - rgb: (H, W, 3) RGB image in [0, 1]
        - depth: (H, W) depth map in meters
        - K: (3, 3) camera intrinsic matrix
        - scene_category: string (e.g., 'bedroom', 'kitchen')
        - sensor: string ('kv1', 'kv2', 'realsense', 'xtion')
        - valid_mask: (H, W) boolean mask for valid depth
        - scene_id: string scene identifier

    Example:
        >>> dataset = SUNRGBDDataset(
        ...     root_dir='data/sunrgbd',
        ...     split='train',
        ...     sensors=['kv1', 'kv2'],
        ...     resize=(480, 640)
        ... )
        >>> sample = dataset[0]
        >>> print(sample['rgb'].shape, sample['K'])
    """

    # SUN RGB-D sensor directories
    SENSOR_DIRS = {
        'kv1': 'kv1/NYUdata',
        'kv2': 'kv2/kinect2data',
        'realsense': 'realsense/sh',
        'xtion': 'xtion/xtion_align_data'
    }

    # Canonical sensor stats (for normalization)
    SENSOR_STATS = {
        'kv1': {'fx': 518.86, 'fy': 519.47, 'cx': 284.58, 'cy': 208.74},
        'kv2': {'fx': 365.46, 'fy': 365.46, 'cx': 254.88, 'cy': 205.40},
        # Note: RealSense and Xtion have varying intrinsics per scene
    }

    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = 'train',
        sensors: Optional[List[str]] = None,
        max_depth: float = 10.0,
        resize: Optional[Tuple[int, int]] = None,
        seed: int = 42
    ):
        """
        Initialize SUN RGB-D dataset.

        Args:
            root_dir: Path to data/sunrgbd/ directory
            split: 'train', 'val', or 'test'
            sensors: List of sensors to include. If None, use all.
                    Options: ['kv1', 'kv2', 'realsense', 'xtion']
            max_depth: Maximum valid depth in meters (clip beyond this)
            resize: Optional (H, W) to resize images. Intrinsics are adjusted.
            seed: Random seed for reproducible splits

        Raises:
            ValueError: If split is invalid or no scenes found
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.max_depth = max_depth
        self.resize = resize
        self.seed = seed

        # Validate split
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Split must be 'train', 'val', or 'test', got '{split}'")

        # Set sensors
        if sensors is None:
            sensors = ['kv1', 'kv2', 'realsense', 'xtion']

        # Validate sensors
        invalid_sensors = set(sensors) - set(self.SENSOR_DIRS.keys())
        if invalid_sensors:
            raise ValueError(
                f"Invalid sensors: {invalid_sensors}. "
                f"Valid options: {list(self.SENSOR_DIRS.keys())}"
            )

        self.sensors = sensors

        # Collect all valid scenes
        print(f"Scanning SUN RGB-D dataset in {self.root_dir}...")
        self.scenes = self._collect_scenes()

        if len(self.scenes) == 0:
            raise ValueError(
                f"No valid scenes found in {self.root_dir} for sensors {self.sensors}. "
                f"Please check dataset structure."
            )

        # Create train/val/test split
        self._create_split()

        print(f"Loaded {len(self.scenes)} scenes for {split} split from sensors: {self.sensors}")

    def _collect_scenes(self) -> List[Dict]:
        """
        Scan dataset and collect all valid scene paths.

        Returns:
            List of dictionaries with scene metadata
        """
        scenes = []

        for sensor in self.sensors:
            sensor_path = self.root_dir / self.SENSOR_DIRS[sensor]

            if not sensor_path.exists():
                warnings.warn(f"Sensor path not found: {sensor_path}")
                continue

            # Find all scene directories
            scene_dirs = sorted([d for d in sensor_path.iterdir() if d.is_dir()])

            for scene_dir in scene_dirs:
                # Determine file naming pattern based on sensor
                if sensor == 'kv1':
                    # kv1/NYUdata/NYU####/
                    # Files: image/NYU####.jpg, depth/NYU####.png
                    scene_id = scene_dir.name
                    rgb_file = scene_dir / 'image' / f'{scene_id}.jpg'
                    depth_file = scene_dir / 'depth' / f'{scene_id}.png'

                elif sensor == 'kv2':
                    # kv2/kinect2data/######_timestamp_rgbf######-resize/
                    # Files: image/######.jpg, depth/######.png
                    # Extract frame number from directory name
                    scene_id = scene_dir.name
                    # Pattern: ######_timestamp_rgbf######-resize
                    # Extract the last frame number
                    if '_rgbf' in scene_id:
                        frame_num = scene_id.split('_rgbf')[-1].replace('-resize', '')
                        rgb_file = scene_dir / 'image' / f'{frame_num}.jpg'
                        depth_file = scene_dir / 'depth' / f'{frame_num}.png'
                    else:
                        # Fallback to glob search
                        rgb_candidates = list((scene_dir / 'image').glob('*.jpg'))
                        depth_candidates = list((scene_dir / 'depth').glob('*.png'))
                        rgb_file = rgb_candidates[0] if rgb_candidates else None
                        depth_file = depth_candidates[0] if depth_candidates else None

                elif sensor in ['realsense', 'xtion']:
                    # realsense/sh/timestamp/
                    # xtion/xtion_align_data/timestamp/
                    # Files typically match directory name or use glob
                    scene_id = scene_dir.name

                    # Try directory name first
                    rgb_file = scene_dir / 'image' / f'{scene_id}.jpg'
                    depth_file = scene_dir / 'depth' / f'{scene_id}.png'

                    # If not found, use glob to find any jpg/png
                    if not rgb_file.exists():
                        rgb_candidates = list((scene_dir / 'image').glob('*.jpg'))
                        rgb_file = rgb_candidates[0] if rgb_candidates else None

                    if not depth_file.exists():
                        depth_candidates = list((scene_dir / 'depth').glob('*.png'))
                        depth_file = depth_candidates[0] if depth_candidates else None
                else:
                    continue

                intrinsics_file = scene_dir / 'intrinsics.txt'
                scene_file = scene_dir / 'scene.txt'

                # Validate scene has all required files
                if all([
                    rgb_file is not None and (isinstance(rgb_file, Path) and rgb_file.exists()),
                    depth_file is not None and (isinstance(depth_file, Path) and depth_file.exists()),
                    intrinsics_file.exists(),
                    scene_file.exists()
                ]):
                    # Parse scene category
                    try:
                        with open(scene_file, 'r') as f:
                            scene_category = f.read().strip()
                    except Exception:
                        scene_category = 'unknown'

                    scenes.append({
                        'sensor': sensor,
                        'scene_dir': scene_dir,
                        'rgb': rgb_file,
                        'depth': depth_file,
                        'intrinsics': intrinsics_file,
                        'scene_category': scene_category,
                        'scene_id': scene_dir.name
                    })

        return scenes

    def _create_split(self):
        """
        Create deterministic train/val/test splits.

        Split ratios: 70% train, 10% val, 20% test
        """
        # Reproducible random split
        np.random.seed(self.seed)
        indices = np.random.permutation(len(self.scenes))

        n_train = int(0.7 * len(self.scenes))
        n_val = int(0.1 * len(self.scenes))

        if self.split == 'train':
            split_indices = indices[:n_train]
        elif self.split == 'val':
            split_indices = indices[n_train:n_train + n_val]
        else:  # test
            split_indices = indices[n_train + n_val:]

        # Filter scenes
        self.scenes = [self.scenes[i] for i in split_indices]

    def __len__(self) -> int:
        """Return number of scenes in dataset."""
        return len(self.scenes)

    def __getitem__(self, idx: int) -> Dict:
        """
        Load and return a single scene.

        Args:
            idx: Scene index

        Returns:
            Dictionary with scene data

        Raises:
            IndexError: If idx out of range
            IOError: If files cannot be loaded
        """
        if idx < 0 or idx >= len(self.scenes):
            raise IndexError(f"Index {idx} out of range [0, {len(self.scenes)})")

        scene = self.scenes[idx]

        try:
            # Load RGB image
            rgb_pil = Image.open(scene['rgb']).convert('RGB')
            rgb = np.array(rgb_pil, dtype=np.float32)

            # Load depth map
            # SUN RGB-D stores depth as 16-bit PNG in millimeters
            depth_pil = Image.open(scene['depth'])
            depth = np.array(depth_pil, dtype=np.float32)

            # Convert millimeters to meters
            depth = depth / 1000.0

            # Clip depth to max_depth
            depth = np.clip(depth, 0, self.max_depth)

            # Load camera intrinsics
            camera = CameraIntrinsics(scene['intrinsics'])
            K = camera.K.copy()

            # Original dimensions
            H_orig, W_orig = rgb.shape[:2]

            # Resize if requested
            if self.resize is not None:
                H_new, W_new = self.resize

                # Resize RGB
                rgb_pil_resized = rgb_pil.resize((W_new, H_new), Image.BILINEAR)
                rgb = np.array(rgb_pil_resized, dtype=np.float32)

                # Resize depth
                depth_pil_resized = depth_pil.resize((W_new, H_new), Image.NEAREST)
                depth = np.array(depth_pil_resized, dtype=np.float32) / 1000.0
                depth = np.clip(depth, 0, self.max_depth)

                # Scale intrinsics
                scale_x = W_new / W_orig
                scale_y = H_new / H_orig

                K = K.copy()
                K[0, :] *= scale_x  # fx, cx
                K[1, :] *= scale_y  # fy, cy

            # Normalize RGB to [0, 1]
            rgb = rgb / 255.0

            # Create validity mask
            valid_mask = (depth > 0) & (depth < self.max_depth) & np.isfinite(depth)

            # Convert to tensors
            return {
                'rgb': torch.from_numpy(rgb).float(),
                'depth': torch.from_numpy(depth).float(),
                'K': torch.from_numpy(K).float(),
                'scene_category': scene['scene_category'],
                'sensor': scene['sensor'],
                'valid_mask': torch.from_numpy(valid_mask),
                'scene_id': scene['scene_id'],
            }

        except Exception as e:
            raise IOError(
                f"Failed to load scene {scene['scene_id']} from {scene['scene_dir']}: {e}"
            )


def visualize_sample(sample: Dict, save_path: Optional[Union[str, Path]] = None):
    """
    Visualize a dataset sample with RGB, depth, and camera info.

    Args:
        sample: Dictionary returned by SUNRGBDDataset.__getitem__()
        save_path: Optional path to save visualization. If None, displays only.

    Example:
        >>> dataset = SUNRGBDDataset('data/sunrgbd', split='train')
        >>> sample = dataset[0]
        >>> visualize_sample(sample, 'outputs/sample_vis.png')
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # RGB
    rgb_np = sample['rgb'].numpy()
    axes[0].imshow(rgb_np)
    axes[0].set_title(f"RGB - {sample['scene_category']}", fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Depth
    depth_np = sample['depth'].numpy()
    valid_mask = sample['valid_mask'].numpy()

    # Mask invalid regions
    depth_vis = depth_np.copy()
    depth_vis[~valid_mask] = 0

    im = axes[1].imshow(depth_vis, cmap='turbo', vmin=0, vmax=5)
    axes[1].set_title(f"Depth Map - Sensor: {sample['sensor']}", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Depth (meters)', fontsize=12)

    # Camera info
    K = sample['K'].numpy()
    info_text = (
        f"Camera Intrinsics:\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"fx = {K[0, 0]:8.2f} px\n"
        f"fy = {K[1, 1]:8.2f} px\n"
        f"cx = {K[0, 2]:8.2f} px\n"
        f"cy = {K[1, 2]:8.2f} px\n"
        f"\n"
        f"Scene Info:\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"Category: {sample['scene_category']}\n"
        f"Sensor: {sample['sensor']}\n"
        f"Scene ID: {sample['scene_id']}\n"
        f"\n"
        f"Statistics:\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"Valid pixels: {valid_mask.sum():,} / {valid_mask.size:,}\n"
        f"Coverage: {100 * valid_mask.sum() / valid_mask.size:.1f}%\n"
        f"Depth range: [{depth_vis[valid_mask].min():.2f}, {depth_vis[valid_mask].max():.2f}] m\n"
        f"Mean depth: {depth_vis[valid_mask].mean():.2f} m"
    )

    axes[2].text(0.1, 0.5, info_text, fontsize=11, family='monospace',
                 verticalalignment='center', transform=axes[2].transAxes)
    axes[2].axis('off')

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    return fig
