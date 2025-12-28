"""
Camera-Aware Gaussian Initialization Module
============================================

Implements camera-aware initialization strategies for 3D Gaussian Splatting.

Key Innovations:
1. Precise depth unprojection using camera intrinsics
2. Focal-length-adaptive Gaussian scaling
3. Position uncertainty weighting based on distance from principal point
4. Support for both camera-aware and naive baseline methods

Mathematical Foundation:
-----------------------
Focal-Length-Adaptive Scaling:
    scale_x = base_scale * Z / fx
    scale_y = base_scale * Z / fy
    scale_z = base_scale * Z / f_mean

    where f_mean = (fx + fy) / 2

Rationale:
    - High focal length (telephoto) → smaller pixel footprint → smaller Gaussians
    - Low focal length (wide-angle) → larger pixel footprint → larger Gaussians
    - Depth-dependent scaling ensures Gaussians represent physical space accurately

Position Uncertainty:
    uncertainty = (Z² / f_mean) * (1 + dist_from_center / image_diagonal)

    Higher uncertainty at:
    - Greater depths (quadratic relationship)
    - Edges of image (lens distortion, projection distortion)

References:
- SA-GS: Scale-Adaptive Gaussian Splatting
- RAIN-GS: Relaxing Accurate Initialization Constraint for 3DGS
- EasySplat: View-Adaptive Learning makes 3DGS Easy

Author: ProjGS Research Team
Date: December 2025
"""

from typing import Dict, Union, Optional
import numpy as np
import torch
import torch.nn as nn

from utils.camera import CameraIntrinsics


class CameraAwareGaussianInit:
    """
    Camera-aware initialization for 3D Gaussian Splatting.

    This class implements novel initialization strategies that leverage known camera
    intrinsics for improved Gaussian placement and scaling.

    Attributes:
        base_scale (float): Base scaling factor for Gaussians
        uncertainty_weight (bool): Whether to use uncertainty-based opacity
        device (str): Compute device ('cuda' or 'cpu')

    Example:
        >>> initializer = CameraAwareGaussianInit(base_scale=0.01)
        >>> gaussians = initializer.initialize_from_rgbd(rgb, depth, K)
        >>> print(gaussians['positions'].shape)  # (N, 3)
    """

    def __init__(
        self,
        base_scale: float = 0.1,
        uncertainty_weight: bool = True,
        device: str = None,
        min_scale: float = 1e-6,
        use_nearest_neighbor: bool = False
    ):
        """
        Initialize the Gaussian initializer.

        Args:
            base_scale: Base scale multiplier for Gaussians (default: 0.1)
                       Research suggests 0.1-0.5 for better variance
            uncertainty_weight: Use position uncertainty for opacity weighting
            device: Compute device (defaults to 'cuda' if available, else 'cpu')
            min_scale: Minimum scale to prevent numerical issues (default: 1e-6)
            use_nearest_neighbor: Use k-NN distance for adaptive scaling
        """
        self.base_scale = base_scale
        self.uncertainty_weight = uncertainty_weight
        self.min_scale = min_scale
        self.use_nearest_neighbor = use_nearest_neighbor

        # Auto-detect device if not specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

    def initialize_from_rgbd(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        K: torch.Tensor,
        max_depth: float = 10.0
    ) -> Dict[str, torch.Tensor]:
        """
        Initialize Gaussians from single RGB-D frame using camera intrinsics.

        This method implements the camera-aware initialization strategy:
        1. Unproject depth to 3D using precise camera model
        2. Compute focal-length-adaptive scales
        3. Apply uncertainty-based opacity weighting
        4. Initialize rotations to identity

        Args:
            rgb: (H, W, 3) RGB image in [0, 1]
            depth: (H, W) depth map in meters
            K: (3, 3) camera intrinsic matrix
            max_depth: Maximum valid depth in meters

        Returns:
            Dictionary containing:
                - positions: (N, 3) 3D Gaussian positions
                - scales: (N, 3) per-axis Gaussian scales
                - rotations: (N, 4) quaternions (w, x, y, z)
                - colors: (N, 3) RGB colors from image
                - opacities: (N, 1) opacity values [0, 1]
                - count: int, number of Gaussians

        Note:
            - N is the number of valid depth pixels
            - Positions are in camera coordinate system
            - Scales are anisotropic (different per axis)
        """
        # Move to target device
        rgb = rgb.to(self.device)
        depth = depth.to(self.device)
        K = K.to(self.device)

        H, W = depth.shape

        # Extract camera parameters
        fx = K[0, 0].item()
        fy = K[1, 1].item()
        cx = K[0, 2].item()
        cy = K[1, 2].item()
        f_mean = (fx + fy) / 2.0

        # === Precise Camera-Aware Unprojection ===
        # Create camera object for unprojection
        camera = CameraIntrinsics.__new__(CameraIntrinsics)
        camera.K = K.cpu().numpy()
        camera.fx, camera.fy = fx, fy
        camera.cx, camera.cy = cx, cy

        # Unproject depth to 3D point cloud
        points_3d = camera.unproject(
            depth.cpu().numpy(),
            return_tensor=True,
            mask_invalid=True,
            max_depth=max_depth
        )  # (H, W, 3)
        points_3d = points_3d.to(self.device)

        # Create validity mask
        valid_mask = (depth > 0) & (depth < max_depth) & torch.isfinite(depth)

        # Filter valid points
        positions = points_3d[valid_mask]  # (N, 3)
        colors = rgb[valid_mask]  # (N, 3)
        N = positions.shape[0]

        if N == 0:
            raise ValueError("No valid depth points found in frame")

        # === Focal-Length-Adaptive Scaling ===
        Z = positions[:, 2]  # Depth component

        if self.use_nearest_neighbor and N > 10:
            # Use k-NN distances for adaptive scaling (standard 3DGS approach)
            # This is more robust for varying point density
            try:
                from scipy.spatial import KDTree
                # Build KDTree
                points_np = positions.cpu().numpy()
                kdtree = KDTree(points_np)
                # Query 4 nearest neighbors (including self)
                distances, _ = kdtree.query(points_np, k=min(4, N))
                # Mean distance to 3 nearest neighbors (excluding self)
                nn_dist = np.mean(distances[:, 1:], axis=1) if N > 1 else np.ones(N)
                # Convert to tensor
                nn_dist_tensor = torch.from_numpy(nn_dist).float().to(self.device)

                # Combine with focal-length adaptation
                scale_x = nn_dist_tensor * (self.base_scale * Z / fx)
                scale_y = nn_dist_tensor * (self.base_scale * Z / fy)
                scale_z = nn_dist_tensor * (self.base_scale * Z / f_mean)
            except ImportError:
                # Fallback to simple focal-length adaptation
                scale_x = self.base_scale * Z / fx
                scale_y = self.base_scale * Z / fy
                scale_z = self.base_scale * Z / f_mean
        else:
            # Standard focal-length-adaptive scaling
            # Pixel footprint at depth Z: Δx = Z / fx
            # Scale inversely with focal length
            scale_x = self.base_scale * Z / fx
            scale_y = self.base_scale * Z / fy
            scale_z = self.base_scale * Z / f_mean

        scales = torch.stack([scale_x, scale_y, scale_z], dim=1)  # (N, 3)

        # Ensure positive scales with minimum threshold
        scales = torch.clamp(scales, min=self.min_scale)

        # === Uncertainty-Based Opacity Weighting ===
        if self.uncertainty_weight:
            # Get pixel coordinates for valid points
            # This requires reconstructing which pixels are valid
            v_coords, u_coords = torch.where(valid_mask)

            # Distance from principal point
            dist_from_center = torch.sqrt(
                (u_coords.float() - cx) ** 2 + (v_coords.float() - cy) ** 2
            )
            image_diag = np.sqrt(H ** 2 + W ** 2)

            # Uncertainty model:
            # - Increases quadratically with depth (perspective projection)
            # - Increases with distance from principal point (lens distortion)
            uncertainty = (Z ** 2 / f_mean) * (1.0 + dist_from_center / image_diag)

            # Normalize to [0, 1]
            uncertainty_min = uncertainty.min()
            uncertainty_max = uncertainty.max()
            uncertainty_norm = (uncertainty - uncertainty_min) / (uncertainty_max - uncertainty_min + 1e-8)

            # Map to opacity: high uncertainty → lower opacity
            # Range: [0.5, 1.0] to avoid completely transparent Gaussians
            # Based on research: standard 3DGS uses 0.1, RGB-D SLAM uses 0.5-0.9
            opacities = 1.0 - 0.5 * uncertainty_norm

            opacities = opacities.unsqueeze(1)  # (N, 1)
        else:
            # Uniform opacity (research suggests 0.1 for conservative, 0.5-0.9 for RGB-D)
            # Using 0.5 as middle ground
            opacities = torch.ones((N, 1), device=self.device) * 0.5

        # === Initialize Rotations ===
        # Start with identity rotation (aligned with coordinate axes)
        # Quaternion format: [w, x, y, z] where w is real part
        rotations = torch.zeros((N, 4), device=self.device)
        rotations[:, 0] = 1.0  # w = 1 (identity rotation)

        return {
            'positions': positions,
            'scales': scales,
            'rotations': rotations,
            'colors': colors,
            'opacities': opacities,
            'count': N
        }

    def initialize_naive(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        max_depth: float = 10.0
    ) -> Dict[str, torch.Tensor]:
        """
        Naive initialization WITHOUT camera awareness (baseline comparison).

        This method ignores camera intrinsics and treats all pixels uniformly.
        Used as baseline to demonstrate the value of camera-aware approach.

        Args:
            rgb: (H, W, 3) RGB image in [0, 1]
            depth: (H, W) depth map in meters
            max_depth: Maximum valid depth

        Returns:
            Dictionary with same structure as initialize_from_rgbd()

        Note:
            This produces INCORRECT 3D coordinates because it assumes:
            - fx = fy = 1 pixel
            - cx = cy = 0
            Leading to wrong spatial relationships.
        """
        rgb = rgb.to(self.device)
        depth = depth.to(self.device)

        H, W = depth.shape

        # === Naive Unprojection (WRONG - ignores camera!) ===
        # Create pixel grids
        u = torch.arange(W, device=self.device, dtype=torch.float32)
        v = torch.arange(H, device=self.device, dtype=torch.float32)
        u_grid, v_grid = torch.meshgrid(u, v, indexing='xy')

        # Naive: assume fx=fy=1, cx=cy=0
        # This is WRONG but commonly done without camera awareness
        X = u_grid * depth
        Y = v_grid * depth
        Z = depth

        points_3d = torch.stack([X, Y, Z], dim=-1)

        # Create validity mask
        valid_mask = (depth > 0) & (depth < max_depth) & torch.isfinite(depth)

        # Filter valid points
        positions = points_3d[valid_mask]
        colors = rgb[valid_mask]
        N = positions.shape[0]

        if N == 0:
            raise ValueError("No valid depth points found in frame")

        # === Uniform Scaling (no camera adaptation) ===
        scales = torch.ones((N, 3), device=self.device) * self.base_scale

        # === Identity Rotations ===
        rotations = torch.zeros((N, 4), device=self.device)
        rotations[:, 0] = 1.0

        # === Uniform Opacity ===
        # Naive baseline uses higher opacity since it doesn't account for uncertainty
        opacities = torch.ones((N, 1), device=self.device) * 0.5

        return {
            'positions': positions,
            'scales': scales,
            'rotations': rotations,
            'colors': colors,
            'opacities': opacities,
            'count': N
        }

    def compute_initialization_metrics(
        self,
        gaussians: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Compute quality metrics for initialized Gaussians.

        Metrics:
        - Scale variance (higher is better for adaptive methods)
        - Opacity distribution statistics
        - Position distribution (mean, std)

        Args:
            gaussians: Dictionary from initialize_from_rgbd() or initialize_naive()

        Returns:
            Dictionary of metric name → value
        """
        metrics = {}

        # Scale statistics
        scales = gaussians['scales']
        metrics['scale_mean_x'] = scales[:, 0].mean().item()
        metrics['scale_mean_y'] = scales[:, 1].mean().item()
        metrics['scale_mean_z'] = scales[:, 2].mean().item()
        metrics['scale_std_x'] = scales[:, 0].std().item()
        metrics['scale_std_y'] = scales[:, 1].std().item()
        metrics['scale_std_z'] = scales[:, 2].std().item()
        metrics['scale_variance'] = scales.var().item()

        # Opacity statistics
        opacities = gaussians['opacities']
        metrics['opacity_mean'] = opacities.mean().item()
        metrics['opacity_std'] = opacities.std().item()

        # Position statistics
        positions = gaussians['positions']
        metrics['position_mean_x'] = positions[:, 0].mean().item()
        metrics['position_mean_y'] = positions[:, 1].mean().item()
        metrics['position_mean_z'] = positions[:, 2].mean().item()
        metrics['position_std_x'] = positions[:, 0].std().item()
        metrics['position_std_y'] = positions[:, 1].std().item()
        metrics['position_std_z'] = positions[:, 2].std().item()

        # Count
        metrics['count'] = gaussians['count']

        return metrics


def compare_initialization_methods(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    K: torch.Tensor,
    max_depth: float = 10.0
) -> Dict[str, Dict]:
    """
    Compare camera-aware vs naive initialization on same RGB-D frame.

    Args:
        rgb: (H, W, 3) RGB image
        depth: (H, W) depth map
        K: (3, 3) camera intrinsics
        max_depth: Maximum valid depth

    Returns:
        Dictionary with keys 'camera_aware' and 'naive', each containing:
            - gaussians: initialization results
            - metrics: quality metrics
    """
    initializer = CameraAwareGaussianInit()

    # Camera-aware initialization
    gaussians_aware = initializer.initialize_from_rgbd(rgb, depth, K, max_depth)
    metrics_aware = initializer.compute_initialization_metrics(gaussians_aware)

    # Naive initialization
    gaussians_naive = initializer.initialize_naive(rgb, depth, max_depth)
    metrics_naive = initializer.compute_initialization_metrics(gaussians_naive)

    return {
        'camera_aware': {
            'gaussians': gaussians_aware,
            'metrics': metrics_aware
        },
        'naive': {
            'gaussians': gaussians_naive,
            'metrics': metrics_naive
        }
    }
