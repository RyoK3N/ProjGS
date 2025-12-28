"""
3D Gaussian Splatting Model with Adaptive Density Control
==========================================================

Full implementation of learnable 3D Gaussians with:
- Learnable parameters (position, color, opacity, scale, rotation)
- Adaptive densification (split & clone)
- Adaptive pruning
- Opacity reset

Based on "3D Gaussian Splatting for Real-Time Radiance Field Rendering"

Author: ProjGS Research Team
Date: December 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DensificationConfig:
    """Configuration for adaptive densification."""
    start_iteration: int = 500
    stop_iteration: int = 15000
    interval: int = 100
    grad_threshold: float = 2e-4
    opacity_reset_interval: int = 3000
    split_threshold: float = 0.02
    min_opacity: float = 0.005
    max_screen_size: int = 20


class GaussianModel(nn.Module):
    """
    Learnable 3D Gaussian model.

    Parameters:
        positions: [N, 3] 3D positions
        features: [N, 3] RGB colors (SH coefficients can be added later)
        opacities: [N, 1] opacity values (logit space)
        scales: [N, 3] log-space scales
        rotations: [N, 4] quaternions

    All parameters are learnable and optimized during training.
    """

    def __init__(
        self,
        positions: torch.Tensor,
        features: torch.Tensor,
        opacities: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        device: str = 'cuda'
    ):
        super().__init__()
        self.device = device

        # Initialize learnable parameters
        self._positions = nn.Parameter(positions.to(device).requires_grad_(True))
        self._features = nn.Parameter(features.to(device).requires_grad_(True))
        self._opacities = nn.Parameter(opacities.to(device).requires_grad_(True))
        self._scales = nn.Parameter(scales.to(device).requires_grad_(True))
        self._rotations = nn.Parameter(rotations.to(device).requires_grad_(True))

        # Gradient accumulators for densification (scalar norm per Gaussian)
        # Based on official 3DGS: accumulates torch.norm(grad, dim=-1)
        self.position_grad_accum = torch.zeros(positions.shape[0], device=device)
        self.position_grad_count = torch.zeros(positions.shape[0], device=device)

        # Track scene extent for normalization
        self.scene_extent = self._compute_scene_extent()

    def _compute_scene_extent(self) -> float:
        """Compute scene bounding box extent."""
        with torch.no_grad():
            mins = self._positions.min(dim=0)[0]
            maxs = self._positions.max(dim=0)[0]
            extent = (maxs - mins).norm().item()
        return max(extent, 1.0)

    @property
    def positions(self) -> torch.Tensor:
        """Get positions."""
        return self._positions

    @property
    def features(self) -> torch.Tensor:
        """Get RGB features."""
        return torch.sigmoid(self._features)  # Map to [0, 1]

    @property
    def opacities(self) -> torch.Tensor:
        """Get opacities."""
        return torch.sigmoid(self._opacities)  # Map to [0, 1]

    @property
    def scales(self) -> torch.Tensor:
        """Get scales."""
        return torch.exp(self._scales)  # Log-space to positive

    @property
    def rotations(self) -> torch.Tensor:
        """Get normalized quaternions."""
        return F.normalize(self._rotations, p=2, dim=1)

    def get_num_gaussians(self) -> int:
        """Get current number of Gaussians."""
        return self._positions.shape[0]

    def accumulate_position_gradients(self):
        """
        Accumulate position gradients for densification.
        Call this after each backward pass.
        """
        if self._positions.grad is not None:
            with torch.no_grad():
                grad_norm = self._positions.grad.norm(dim=1, keepdim=False)
                self.position_grad_accum += grad_norm
                self.position_grad_count += 1

    def get_average_position_gradient(self) -> torch.Tensor:
        """Get average accumulated position gradient."""
        with torch.no_grad():
            avg_grad = self.position_grad_accum / (self.position_grad_count + 1e-6)
        return avg_grad

    def reset_gradient_accumulators(self):
        """Reset gradient accumulators."""
        self.position_grad_accum.zero_()
        self.position_grad_count.zero_()

    def densify_and_split(
        self,
        grad_threshold: float,
        split_threshold: float,
        N: int = 2
    ):
        """
        Densify by splitting large Gaussians with high gradients.

        Args:
            grad_threshold: Gradient threshold for densification
            split_threshold: Scale threshold (relative to scene extent)
            N: Number of splits per Gaussian
        """
        with torch.no_grad():
            avg_grad = self.get_average_position_gradient()

            # Find Gaussians to split: high gradient AND large scale
            max_scale = self.scales.max(dim=1)[0]
            split_mask = (avg_grad > grad_threshold) & \
                        (max_scale > split_threshold * self.scene_extent)

            if not split_mask.any():
                return

            # Get Gaussians to split
            positions = self._positions[split_mask]
            features = self._features[split_mask]
            opacities = self._opacities[split_mask]
            scales = self._scales[split_mask]
            rotations = self._rotations[split_mask]

            # Create N new Gaussians per split
            new_positions_list = []
            new_features_list = []
            new_opacities_list = []
            new_scales_list = []
            new_rotations_list = []

            for i in range(N):
                # Sample position offset from Gaussian distribution
                scale_vals = torch.exp(scales)
                offset = torch.randn_like(positions) * scale_vals * 0.5

                new_positions_list.append(positions + offset)
                new_features_list.append(features.clone())
                new_opacities_list.append(opacities.clone())
                new_scales_list.append(scales - torch.log(torch.tensor(1.6, device=self.device)))  # Reduce scale
                new_rotations_list.append(rotations.clone())

            # Concatenate new Gaussians
            new_positions = torch.cat(new_positions_list, dim=0)
            new_features = torch.cat(new_features_list, dim=0)
            new_opacities = torch.cat(new_opacities_list, dim=0)
            new_scales = torch.cat(new_scales_list, dim=0)
            new_rotations = torch.cat(new_rotations_list, dim=0)

            # Remove old Gaussians and add new ones
            keep_mask = ~split_mask
            self._positions = nn.Parameter(torch.cat([self._positions[keep_mask], new_positions], dim=0).clone().detach().requires_grad_(True))
            self._features = nn.Parameter(torch.cat([self._features[keep_mask], new_features], dim=0).clone().detach().requires_grad_(True))
            self._opacities = nn.Parameter(torch.cat([self._opacities[keep_mask], new_opacities], dim=0).clone().detach().requires_grad_(True))
            self._scales = nn.Parameter(torch.cat([self._scales[keep_mask], new_scales], dim=0).clone().detach().requires_grad_(True))
            self._rotations = nn.Parameter(torch.cat([self._rotations[keep_mask], new_rotations], dim=0).clone().detach().requires_grad_(True))

            # Update gradient accumulators
            new_grad_accum = torch.zeros(new_positions.shape[0], device=self.device)
            new_grad_count = torch.zeros(new_positions.shape[0], device=self.device)
            self.position_grad_accum = torch.cat([self.position_grad_accum[keep_mask], new_grad_accum], dim=0)
            self.position_grad_count = torch.cat([self.position_grad_count[keep_mask], new_grad_count], dim=0)

    def densify_and_clone(
        self,
        grad_threshold: float,
        split_threshold: float
    ):
        """
        Densify by cloning small Gaussians with high gradients.

        Args:
            grad_threshold: Gradient threshold for densification
            split_threshold: Scale threshold (relative to scene extent)
        """
        with torch.no_grad():
            avg_grad = self.get_average_position_gradient()

            # Find Gaussians to clone: high gradient AND small scale
            max_scale = self.scales.max(dim=1)[0]
            clone_mask = (avg_grad > grad_threshold) & \
                        (max_scale <= split_threshold * self.scene_extent)

            if not clone_mask.any():
                return

            # Clone Gaussians
            new_positions = self._positions[clone_mask].clone()
            new_features = self._features[clone_mask].clone()
            new_opacities = self._opacities[clone_mask].clone()
            new_scales = self._scales[clone_mask].clone()
            new_rotations = self._rotations[clone_mask].clone()

            # Add to parameters
            self._positions = nn.Parameter(torch.cat([self._positions, new_positions], dim=0).clone().detach().requires_grad_(True))
            self._features = nn.Parameter(torch.cat([self._features, new_features], dim=0).clone().detach().requires_grad_(True))
            self._opacities = nn.Parameter(torch.cat([self._opacities, new_opacities], dim=0).clone().detach().requires_grad_(True))
            self._scales = nn.Parameter(torch.cat([self._scales, new_scales], dim=0).clone().detach().requires_grad_(True))
            self._rotations = nn.Parameter(torch.cat([self._rotations, new_rotations], dim=0).clone().detach().requires_grad_(True))

            # Update gradient accumulators
            new_grad_accum = torch.zeros(new_positions.shape[0], device=self.device)
            new_grad_count = torch.zeros(new_positions.shape[0], device=self.device)
            self.position_grad_accum = torch.cat([self.position_grad_accum, new_grad_accum], dim=0)
            self.position_grad_count = torch.cat([self.position_grad_count, new_grad_count], dim=0)

    def prune(
        self,
        min_opacity: float,
        max_scale: Optional[float] = None
    ):
        """
        Prune Gaussians with low opacity or large scale.

        Args:
            min_opacity: Minimum opacity threshold
            max_scale: Maximum scale threshold (relative to scene extent)
        """
        with torch.no_grad():
            # Opacity pruning
            opacity_mask = self.opacities.squeeze() > min_opacity

            # Scale pruning (optional)
            if max_scale is not None:
                max_scale_vals = self.scales.max(dim=1)[0]
                scale_mask = max_scale_vals < max_scale * self.scene_extent
                keep_mask = opacity_mask & scale_mask
            else:
                keep_mask = opacity_mask

            if keep_mask.all():
                return  # Nothing to prune

            # Keep only valid Gaussians
            self._positions = nn.Parameter(self._positions[keep_mask].clone().detach().requires_grad_(True))
            self._features = nn.Parameter(self._features[keep_mask].clone().detach().requires_grad_(True))
            self._opacities = nn.Parameter(self._opacities[keep_mask].clone().detach().requires_grad_(True))
            self._scales = nn.Parameter(self._scales[keep_mask].clone().detach().requires_grad_(True))
            self._rotations = nn.Parameter(self._rotations[keep_mask].clone().detach().requires_grad_(True))

            # Update gradient accumulators
            self.position_grad_accum = self.position_grad_accum[keep_mask]
            self.position_grad_count = self.position_grad_count[keep_mask]

    def reset_opacity(self, new_opacity: float = 0.01):
        """
        Reset opacity of all Gaussians.

        Used periodically during training to prevent mode collapse.

        Args:
            new_opacity: New opacity value
        """
        with torch.no_grad():
            # Convert to logit space
            logit_opacity = torch.logit(torch.tensor(new_opacity, device=self.device))
            self._opacities.fill_(logit_opacity)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Get state dictionary for checkpointing."""
        return {
            'positions': self._positions.data,
            'features': self._features.data,
            'opacities': self._opacities.data,
            'scales': self._scales.data,
            'rotations': self._rotations.data,
            'position_grad_accum': self.position_grad_accum,
            'position_grad_count': self.position_grad_count,
            'scene_extent': torch.tensor(self.scene_extent)
        }

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """Load from state dictionary."""
        self._positions = nn.Parameter(state_dict['positions'].to(self.device))
        self._features = nn.Parameter(state_dict['features'].to(self.device))
        self._opacities = nn.Parameter(state_dict['opacities'].to(self.device))
        self._scales = nn.Parameter(state_dict['scales'].to(self.device))
        self._rotations = nn.Parameter(state_dict['rotations'].to(self.device))
        self.position_grad_accum = state_dict['position_grad_accum'].to(self.device)
        self.position_grad_count = state_dict['position_grad_count'].to(self.device)
        self.scene_extent = state_dict['scene_extent'].item()


# Test code
if __name__ == '__main__':
    print("Testing Gaussian Model...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create test Gaussians
    N = 1000
    positions = torch.randn(N, 3, device=device)
    features = torch.randn(N, 3, device=device)
    opacities = torch.ones(N, 1, device=device) * 0.5
    scales = torch.ones(N, 3, device=device) * 0.01
    rotations = torch.randn(N, 4, device=device)
    rotations = F.normalize(rotations, p=2, dim=1)

    # Create model
    model = GaussianModel(
        positions, features, opacities, scales, rotations, device=device
    )

    print(f"✓ Initial Gaussians: {model.get_num_gaussians()}")

    # Simulate gradient accumulation
    for _ in range(10):
        model._positions.grad = torch.randn_like(model._positions) * 0.01
        model.accumulate_position_gradients()

    # Test densification
    print("\nTesting densification...")
    model.densify_and_split(grad_threshold=0.005, split_threshold=0.02)
    print(f"✓ After split: {model.get_num_gaussians()} Gaussians")

    model.densify_and_clone(grad_threshold=0.005, split_threshold=0.02)
    print(f"✓ After clone: {model.get_num_gaussians()} Gaussians")

    # Test pruning
    print("\nTesting pruning...")
    model.prune(min_opacity=0.1, max_scale=0.1)
    print(f"✓ After prune: {model.get_num_gaussians()} Gaussians")

    # Test opacity reset
    model.reset_opacity(0.5)
    print(f"✓ Opacity reset: mean={model.opacities.mean().item():.3f}")

    # Test checkpointing
    state = model.state_dict()
    print(f"✓ State dict keys: {list(state.keys())}")

    print("\n✓ Gaussian model working!")
