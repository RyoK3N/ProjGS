"""
Camera Intrinsics and Extrinsics Module
========================================

Provides camera matrix parsing, depth unprojection, and geometric transformations
for the SUN RGB-D dataset and camera-aware Gaussian Splatting.

Key Features:
- Parse SUN RGB-D intrinsics format (3x3 matrix in row-major order)
- Camera-aware depth unprojection using precise projection equations
- 2D-3D projection/unprojection cycle with numerical verification
- Support for extrinsic transformations

Mathematical Foundation:
-----------------------
Intrinsic Matrix K:
    [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]

Depth Unprojection (pixel to 3D):
    X = (u - cx) * D / fx
    Y = (v - cy) * D / fy
    Z = D

3D to 2D Projection:
    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy

References:
- Intel RealSense Projection Docs:
  https://dev.intelrealsense.com/docs/projection-texture-mapping-and-occlusion
- SUN RGB-D Dataset Paper (Song et al., CVPR 2015)
- Multiple View Geometry in Computer Vision (Hartley & Zisserman)

Author: ProjGS Research Team
Date: December 2025
"""

from pathlib import Path
from typing import Union, Tuple, Optional
import numpy as np
import torch


class CameraIntrinsics:
    """
    Camera intrinsic parameters handler for RGB-D sensors.

    Handles parsing, validation, and geometric operations with camera intrinsic matrices.
    Designed for precision and numerical stability in depth unprojection tasks.

    Attributes:
        K (np.ndarray): 3x3 intrinsic matrix
        fx (float): Focal length in x direction (pixels)
        fy (float): Focal length in y direction (pixels)
        cx (float): Principal point x coordinate (pixels)
        cy (float): Principal point y coordinate (pixels)

    Example:
        >>> camera = CameraIntrinsics('data/sunrgbd/kv1/NYUdata/NYU0001/intrinsics.txt')
        >>> print(camera)
        CameraIntrinsics(fx=518.86, fy=519.47, cx=284.58, cy=208.74)
        >>> points_3d = camera.unproject(depth_map)
    """

    def __init__(self, intrinsics_file: Union[str, Path]):
        """
        Load and parse camera intrinsics from file.

        Args:
            intrinsics_file: Path to intrinsics file in SUN RGB-D format
                            (9 space-separated values in row-major order)

        Raises:
            FileNotFoundError: If intrinsics file doesn't exist
            ValueError: If file format is invalid or contains non-positive focal lengths

        Note:
            SUN RGB-D format: "fx 0 cx 0 fy cy 0 0 1" (9 values)
        """
        intrinsics_file = Path(intrinsics_file)

        if not intrinsics_file.exists():
            raise FileNotFoundError(f"Intrinsics file not found: {intrinsics_file}")

        with open(intrinsics_file, 'r') as f:
            content = f.read().strip()

        try:
            values = list(map(float, content.split()))
        except ValueError as e:
            raise ValueError(f"Invalid intrinsics format: {e}")

        if len(values) != 9:
            raise ValueError(
                f"Expected 9 values in intrinsics file, got {len(values)}. "
                f"Format should be: fx 0 cx 0 fy cy 0 0 1"
            )

        # Reshape to 3x3 matrix (row-major order)
        self.K = np.array(values, dtype=np.float64).reshape(3, 3)

        # Extract parameters with validation
        self.fx = float(self.K[0, 0])
        self.fy = float(self.K[1, 1])
        self.cx = float(self.K[0, 2])
        self.cy = float(self.K[1, 2])

        # Validate focal lengths
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError(
                f"Invalid focal lengths: fx={self.fx}, fy={self.fy}. "
                f"Focal lengths must be positive."
            )

        # Validate matrix structure (should be upper triangular with [2,2]=1)
        if not np.isclose(self.K[2, 2], 1.0):
            raise ValueError(f"K[2,2] should be 1.0, got {self.K[2, 2]}")

        # Check for zero elements where expected
        expected_zeros = [(1, 0), (2, 0), (2, 1)]
        for i, j in expected_zeros:
            if not np.isclose(self.K[i, j], 0.0, atol=1e-6):
                raise ValueError(
                    f"K[{i},{j}] should be 0.0, got {self.K[i, j]}. "
                    f"Invalid intrinsic matrix structure."
                )

    def to_tensor(self, device: Union[str, torch.device] = 'cuda') -> torch.Tensor:
        """
        Convert intrinsic matrix to PyTorch tensor.

        Args:
            device: Target device ('cuda', 'cpu', or torch.device)

        Returns:
            torch.Tensor: 3x3 intrinsic matrix on specified device
        """
        return torch.from_numpy(self.K).float().to(device)

    def unproject(
        self,
        depth: Union[np.ndarray, torch.Tensor],
        return_tensor: bool = True,
        mask_invalid: bool = True,
        max_depth: float = 10.0
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Unproject depth map to 3D point cloud using camera intrinsics.

        Implements the precise pinhole camera unprojection model:
            X = (u - cx) * D / fx
            Y = (v - cy) * D / fy
            Z = D

        Args:
            depth: (H, W) depth map in meters
            return_tensor: If True, return torch.Tensor; else numpy array
            mask_invalid: If True, set invalid points (depth=0 or depth>max_depth) to NaN
            max_depth: Maximum valid depth in meters

        Returns:
            Point cloud of shape (H, W, 3) with coordinates [X, Y, Z]
            Invalid points are set to NaN if mask_invalid=True

        Example:
            >>> depth = np.random.rand(480, 640) * 5.0  # Random depth 0-5m
            >>> points_3d = camera.unproject(depth)
            >>> print(points_3d.shape)  # (480, 640, 3)

        Note:
            - Resulting coordinates are in camera coordinate system
            - Z-axis points away from camera (into the scene)
            - X-axis points right, Y-axis points down (image coordinates)
        """
        # Convert to numpy if tensor
        if isinstance(depth, torch.Tensor):
            depth_np = depth.cpu().numpy()
            was_tensor = True
        else:
            depth_np = depth
            was_tensor = False

        if depth_np.ndim != 2:
            raise ValueError(f"Depth must be 2D, got shape {depth_np.shape}")

        H, W = depth_np.shape

        # Create pixel coordinate grids
        # Use np.arange for better numerical stability than meshgrid
        u = np.arange(W, dtype=np.float64)
        v = np.arange(H, dtype=np.float64)
        u_grid, v_grid = np.meshgrid(u, v)

        # Unproject to 3D using pinhole camera model
        # Using double precision for numerical stability
        X = (u_grid - self.cx) * depth_np / self.fx
        Y = (v_grid - self.cy) * depth_np / self.fy
        Z = depth_np.copy()

        # Stack to form point cloud
        points = np.stack([X, Y, Z], axis=-1)  # (H, W, 3)

        # Mask invalid depths
        if mask_invalid:
            invalid_mask = (depth_np <= 0) | (depth_np > max_depth) | ~np.isfinite(depth_np)
            points[invalid_mask] = np.nan

        # Convert to tensor if requested
        if return_tensor:
            points_tensor = torch.from_numpy(points).float()
            return points_tensor

        return points.astype(np.float32)

    def project(
        self,
        points_3d: Union[np.ndarray, torch.Tensor],
        return_integers: bool = False
    ) -> np.ndarray:
        """
        Project 3D points to 2D image coordinates.

        Args:
            points_3d: (N, 3) or (H, W, 3) array of 3D points [X, Y, Z]
            return_integers: If True, round pixel coordinates to integers

        Returns:
            (N, 2) or (H, W, 2) array of pixel coordinates [u, v]

        Note:
            Points with Z <= 0 will have invalid pixel coordinates
        """
        # Handle tensor input
        if isinstance(points_3d, torch.Tensor):
            points = points_3d.cpu().numpy()
        else:
            points = points_3d

        original_shape = points.shape

        # Reshape to (N, 3) if necessary
        if points.ndim == 3:  # (H, W, 3)
            H, W, _ = points.shape
            points = points.reshape(-1, 3)
            reshape_output = True
        else:
            reshape_output = False

        # Extract coordinates
        X, Y, Z = points[:, 0], points[:, 1], points[:, 2]

        # Perspective projection
        # Add epsilon to avoid division by zero
        Z_safe = np.where(Z > 1e-6, Z, np.nan)

        u = self.fx * (X / Z_safe) + self.cx
        v = self.fy * (Y / Z_safe) + self.cy

        # Stack pixel coordinates
        pixels = np.stack([u, v], axis=1)

        # Reshape back if needed
        if reshape_output:
            pixels = pixels.reshape(H, W, 2)

        # Convert to integers if requested
        if return_integers:
            pixels = np.round(pixels).astype(np.int32)

        return pixels

    def compute_fov(self, image_width: int, image_height: int) -> Tuple[float, float]:
        """
        Compute field of view angles from intrinsics.

        Args:
            image_width: Image width in pixels
            image_height: Image height in pixels

        Returns:
            (fov_x, fov_y): Field of view in degrees
        """
        fov_x = 2 * np.arctan(image_width / (2 * self.fx)) * 180 / np.pi
        fov_y = 2 * np.arctan(image_height / (2 * self.fy)) * 180 / np.pi
        return fov_x, fov_y

    def scale_intrinsics(self, scale_x: float, scale_y: float) -> 'CameraIntrinsics':
        """
        Scale intrinsics for resized images.

        Args:
            scale_x: Width scaling factor (new_width / old_width)
            scale_y: Height scaling factor (new_height / old_height)

        Returns:
            New CameraIntrinsics object with scaled parameters
        """
        # Create new instance
        new_cam = CameraIntrinsics.__new__(CameraIntrinsics)

        # Scale intrinsic matrix
        new_cam.K = self.K.copy()
        new_cam.K[0, :] *= scale_x  # fx, cx
        new_cam.K[1, :] *= scale_y  # fy, cy

        # Update parameters
        new_cam.fx = new_cam.K[0, 0]
        new_cam.fy = new_cam.K[1, 1]
        new_cam.cx = new_cam.K[0, 2]
        new_cam.cy = new_cam.K[1, 2]

        return new_cam

    def __repr__(self) -> str:
        """String representation showing key parameters."""
        return (f"CameraIntrinsics(fx={self.fx:.2f}, fy={self.fy:.2f}, "
                f"cx={self.cx:.2f}, cy={self.cy:.2f})")

    def __str__(self) -> str:
        """Detailed string representation."""
        return (f"Camera Intrinsics:\n"
                f"  Focal Length: fx={self.fx:.4f}, fy={self.fy:.4f}\n"
                f"  Principal Point: cx={self.cx:.4f}, cy={self.cy:.4f}\n"
                f"  Matrix:\n{self.K}")


class CameraExtrinsics:
    """
    Camera extrinsic parameters (rotation and translation).

    Handles the transformation from camera coordinates to world coordinates.
    Format: [R|t] where R is 3x3 rotation matrix and t is 3x1 translation vector.

    Attributes:
        Rt (np.ndarray): 3x4 extrinsic matrix [R|t]
        R (np.ndarray): 3x3 rotation matrix
        t (np.ndarray): 3x1 translation vector
    """

    def __init__(self, extrinsics_file: Union[str, Path]):
        """
        Load camera extrinsics from file.

        Args:
            extrinsics_file: Path to extrinsics file
                           Expected format: 3x4 matrix [R|t]

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If matrix has invalid shape or rotation is not orthogonal
        """
        extrinsics_file = Path(extrinsics_file)

        if not extrinsics_file.exists():
            raise FileNotFoundError(f"Extrinsics file not found: {extrinsics_file}")

        # Load matrix
        try:
            Rt = np.loadtxt(extrinsics_file, dtype=np.float64)
        except Exception as e:
            raise ValueError(f"Failed to load extrinsics: {e}")

        if Rt.shape != (3, 4):
            raise ValueError(
                f"Expected extrinsics shape (3, 4), got {Rt.shape}"
            )

        self.Rt = Rt
        self.R = Rt[:, :3]  # 3x3 rotation
        self.t = Rt[:, 3]   # 3x1 translation

        # Validate rotation matrix (should be orthogonal: R^T * R = I)
        should_be_identity = self.R.T @ self.R
        if not np.allclose(should_be_identity, np.eye(3), atol=1e-3):
            import warnings
            warnings.warn(
                f"Rotation matrix is not orthogonal. "
                f"R^T * R - I max error: {np.abs(should_be_identity - np.eye(3)).max():.6f}"
            )

    def transform_points(
        self,
        points_3d: Union[np.ndarray, torch.Tensor],
        inverse: bool = False
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply extrinsic transformation to 3D points.

        Args:
            points_3d: (N, 3) points in camera coordinates
            inverse: If True, transform from world to camera coordinates

        Returns:
            Transformed points (N, 3)

        Note:
            Forward: P_world = R * P_camera + t
            Inverse: P_camera = R^T * (P_world - t)
        """
        is_tensor = isinstance(points_3d, torch.Tensor)

        if is_tensor:
            points = points_3d.cpu().numpy()
            device = points_3d.device
        else:
            points = points_3d

        if inverse:
            # World to camera: R^T * (P - t)
            transformed = (self.R.T @ (points - self.t).T).T
        else:
            # Camera to world: R * P + t
            transformed = (self.R @ points.T).T + self.t

        if is_tensor:
            return torch.from_numpy(transformed).float().to(device)

        return transformed.astype(np.float32)

    def __repr__(self) -> str:
        """String representation."""
        return f"CameraExtrinsics(R shape={self.R.shape}, t={self.t})"

    def __str__(self) -> str:
        """Detailed string representation."""
        return (f"Camera Extrinsics:\n"
                f"Rotation:\n{self.R}\n"
                f"Translation: {self.t}")
