"""
Differentiable 3D Gaussian Rasterizer
======================================

PyTorch implementation of differentiable Gaussian splatting rasterization.
Optimized for both CUDA (A100) and MPS (Apple Silicon).

This implements the forward rasterization from:
"3D Gaussian Splatting for Real-Time Radiance Field Rendering" (Kerbl et al., 2023)

Key operations:
1. Project 3D Gaussians to 2D screen space
2. Compute 2D covariance matrices
3. Alpha-blend Gaussians in depth order

Author: ProjGS Research Team
Date: December 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict


def build_rotation_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Build rotation matrices from quaternions.

    Args:
        quaternions: [N, 4] tensor of quaternions (w, x, y, z)

    Returns:
        Rotation matrices [N, 3, 3]
    """
    # Normalize quaternions
    quaternions = F.normalize(quaternions, p=2, dim=1)

    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]

    # Build rotation matrix
    R = torch.zeros(quaternions.shape[0], 3, 3, device=quaternions.device, dtype=quaternions.dtype)

    R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    R[:, 0, 1] = 2 * (x*y - w*z)
    R[:, 0, 2] = 2 * (x*z + w*y)

    R[:, 1, 0] = 2 * (x*y + w*z)
    R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    R[:, 1, 2] = 2 * (y*z - w*x)

    R[:, 2, 0] = 2 * (x*z - w*y)
    R[:, 2, 1] = 2 * (y*z + w*x)
    R[:, 2, 2] = 1 - 2 * (x**2 + y**2)

    return R


def build_scaling_matrix(scales: torch.Tensor) -> torch.Tensor:
    """
    Build diagonal scaling matrices.

    Args:
        scales: [N, 3] tensor of scales

    Returns:
        Scaling matrices [N, 3, 3]
    """
    N = scales.shape[0]
    S = torch.zeros(N, 3, 3, device=scales.device, dtype=scales.dtype)
    S[:, 0, 0] = scales[:, 0]
    S[:, 1, 1] = scales[:, 1]
    S[:, 2, 2] = scales[:, 2]
    return S


def project_gaussians_to_2d(
    means_3d: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    K: torch.Tensor,
    image_size: Tuple[int, int]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Project 3D Gaussians to 2D screen space.

    Args:
        means_3d: [N, 3] 3D positions
        scales: [N, 3] Gaussian scales
        rotations: [N, 4] quaternions
        K: [3, 3] camera intrinsic matrix
        image_size: (H, W) image dimensions

    Returns:
        means_2d: [N, 2] 2D projected centers
        cov_2d: [N, 2, 2] 2D covariance matrices
        depths: [N] depth values
    """
    N = means_3d.shape[0]
    device = means_3d.device

    # Extract camera intrinsics
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Project centers to 2D
    X, Y, Z = means_3d[:, 0], means_3d[:, 1], means_3d[:, 2]

    # Perspective projection
    z_inv = 1.0 / (Z + 1e-6)
    u = fx * X * z_inv + cx
    v = fy * Y * z_inv + cy

    means_2d = torch.stack([u, v], dim=1)

    # Build 3D covariance matrix: Σ = R S Sᵀ Rᵀ
    R = build_rotation_matrix(rotations)  # [N, 3, 3]
    S = build_scaling_matrix(scales)      # [N, 3, 3]

    # Compute RS
    RS = torch.bmm(R, S)  # [N, 3, 3]

    # Compute covariance: Σ = RS(RS)ᵀ
    cov_3d = torch.bmm(RS, RS.transpose(1, 2))  # [N, 3, 3]

    # Jacobian of projection (simplified)
    # J = [fx/Z, 0, -fx*X/Z²]
    #     [0, fy/Z, -fy*Y/Z²]
    J = torch.zeros(N, 2, 3, device=device, dtype=means_3d.dtype)
    J[:, 0, 0] = fx * z_inv
    J[:, 0, 2] = -fx * X * z_inv * z_inv
    J[:, 1, 1] = fy * z_inv
    J[:, 1, 2] = -fy * Y * z_inv * z_inv

    # Project covariance to 2D: Σ' = J Σ Jᵀ
    temp = torch.bmm(J, cov_3d)  # [N, 2, 3]
    cov_2d = torch.bmm(temp, J.transpose(1, 2))  # [N, 2, 2]

    # Add small value for numerical stability
    cov_2d[:, 0, 0] += 1e-4
    cov_2d[:, 1, 1] += 1e-4

    return means_2d, cov_2d, Z


class GaussianRasterizer(nn.Module):
    """
    Differentiable Gaussian rasterizer.

    Renders 3D Gaussians into 2D images using alpha blending.

    Args:
        image_size: (H, W) output image size
        device: Device to run on
    """

    def __init__(
        self,
        image_size: Tuple[int, int],
        device: str = 'cuda'
    ):
        super().__init__()
        self.image_size = image_size
        self.device = device
        self.H, self.W = image_size

        # Create pixel grid
        y, x = torch.meshgrid(
            torch.arange(self.H, device=device, dtype=torch.float32),
            torch.arange(self.W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        self.register_buffer('pixel_coords', torch.stack([x, y], dim=-1))  # [H, W, 2]

    def forward(
        self,
        positions: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        K: torch.Tensor,
        background: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Render Gaussians to image.

        Args:
            positions: [N, 3] 3D positions
            scales: [N, 3] Gaussian scales
            rotations: [N, 4] quaternions
            colors: [N, 3] RGB colors
            opacities: [N, 1] opacity values
            K: [3, 3] camera intrinsic matrix
            background: [3] or [H, W, 3] background color

        Returns:
            Dictionary with:
                - image: [H, W, 3] rendered image
                - depth: [H, W] depth map
                - alpha: [H, W] accumulated alpha
        """
        N = positions.shape[0]
        device = positions.device

        # Set default background
        if background is None:
            background = torch.zeros(3, device=device)

        # Project Gaussians to 2D
        means_2d, cov_2d, depths = project_gaussians_to_2d(
            positions, scales, rotations, K, self.image_size
        )

        # Sort by depth (back to front for alpha blending)
        depth_order = torch.argsort(depths, descending=True)
        means_2d = means_2d[depth_order]
        cov_2d = cov_2d[depth_order]
        colors = colors[depth_order]
        opacities = opacities[depth_order]
        depths = depths[depth_order]

        # Initialize output
        image = background.view(1, 1, 3).expand(self.H, self.W, 3).clone()
        depth_map = torch.zeros(self.H, self.W, device=device)
        alpha_acc = torch.zeros(self.H, self.W, device=device)

        # Render each Gaussian
        for i in range(N):
            # Get Gaussian parameters
            mean = means_2d[i]  # [2]
            cov = cov_2d[i]     # [2, 2]
            color = colors[i]   # [3]
            opacity = opacities[i].item()
            depth = depths[i].item()

            # Check if Gaussian is visible
            if mean[0] < -50 or mean[0] > self.W + 50 or \
               mean[1] < -50 or mean[1] > self.H + 50:
                continue

            # Compute inverse covariance
            try:
                cov_inv = torch.inverse(cov)
            except:
                continue

            # Compute Mahalanobis distance for all pixels
            diff = self.pixel_coords - mean.view(1, 1, 2)  # [H, W, 2]

            # d² = (x - μ)ᵀ Σ⁻¹ (x - μ)
            temp = torch.einsum('hwi,ij->hwj', diff, cov_inv)  # [H, W, 2]
            mahalanobis = torch.einsum('hwi,hwi->hw', temp, diff)  # [H, W]

            # Gaussian weight: exp(-0.5 * d²)
            weight = torch.exp(-0.5 * mahalanobis)

            # Apply opacity threshold for efficiency
            weight = weight * opacity
            mask = weight > 0.01

            if not mask.any():
                continue

            # Alpha blending (back to front)
            alpha = weight[mask]  # [num_masked_pixels]
            T = 1.0 - alpha_acc[mask]  # Transmittance [num_masked_pixels]

            # Compute contribution for each pixel
            contribution = T * alpha  # [num_masked_pixels]

            # Compute RGB contribution: contribution * color
            # contribution: [num_masked_pixels], color: [3]
            # Result: [num_masked_pixels, 3]
            contribution_rgb = contribution.unsqueeze(-1) * color.unsqueeze(0)

            # Update image using direct boolean indexing
            # image[mask] gives [num_masked_pixels, 3]
            image[mask] = image[mask] + contribution_rgb

            # Update depth
            depth_map[mask] = depth_map[mask] + contribution * depth

            # Update accumulated alpha
            alpha_acc[mask] = alpha_acc[mask] + contribution

        # Clamp to valid range
        image = torch.clamp(image, 0.0, 1.0)

        return {
            'image': image,
            'depth': depth_map,
            'alpha': alpha_acc
        }


class FastGaussianRasterizer(nn.Module):
    """
    Fast Gaussian rasterizer using tile-based rendering.

    More efficient for large number of Gaussians.
    Uses tiling to reduce memory footprint.

    Args:
        image_size: (H, W) output image size
        tile_size: Size of tiles for rendering (default: 16)
        device: Device to run on
    """

    def __init__(
        self,
        image_size: Tuple[int, int],
        tile_size: int = 16,
        device: str = 'cuda'
    ):
        super().__init__()
        self.image_size = image_size
        self.tile_size = tile_size
        self.device = device
        self.H, self.W = image_size

        # For now, use the standard rasterizer
        # TODO: Implement tile-based rendering for production
        self.rasterizer = GaussianRasterizer(image_size, device)

    def forward(
        self,
        positions: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        K: torch.Tensor,
        background: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass (delegates to standard rasterizer for now)."""
        return self.rasterizer(
            positions, scales, rotations, colors, opacities, K, background
        )


# Test code
if __name__ == '__main__':
    print("Testing Gaussian Rasterizer...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create test Gaussians
    N = 100
    positions = torch.randn(N, 3, device=device) * 2.0
    positions[:, 2] += 5.0  # Move away from camera

    scales = torch.rand(N, 3, device=device) * 0.1 + 0.01
    rotations = torch.randn(N, 4, device=device)
    rotations = F.normalize(rotations, p=2, dim=1)

    colors = torch.rand(N, 3, device=device)
    opacities = torch.rand(N, 1, device=device)

    # Camera intrinsics
    K = torch.tensor([
        [500.0, 0.0, 320.0],
        [0.0, 500.0, 240.0],
        [0.0, 0.0, 1.0]
    ], device=device)

    # Create rasterizer
    rasterizer = GaussianRasterizer((480, 640), device=device)

    # Render
    with torch.no_grad():
        outputs = rasterizer(
            positions, scales, rotations, colors, opacities, K
        )

    print(f"✓ Rendered image: {outputs['image'].shape}")
    print(f"✓ Depth map: {outputs['depth'].shape}")
    print(f"✓ Alpha: {outputs['alpha'].shape}")
    print(f"  Image range: [{outputs['image'].min():.3f}, {outputs['image'].max():.3f}]")
    print(f"  Depth range: [{outputs['depth'].min():.3f}, {outputs['depth'].max():.3f}]")

    print("\n✓ Rasterizer working!")
