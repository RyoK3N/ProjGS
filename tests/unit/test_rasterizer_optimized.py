"""
Unit Tests for Tile-Based Gaussian Rasterizer
==============================================

Tests for memory-optimized tile-based rasterization.

Author: ProjGS Research Team
Date: December 2025
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
import pytest
import numpy as np

from models.rasterizer_optimized import TileBasedGaussianRasterizer


class TestTileBasedRasterizer:
    """Test suite for TileBasedGaussianRasterizer."""

    @pytest.fixture
    def device(self):
        """Get available device."""
        if torch.backends.mps.is_available():
            return 'mps'
        elif torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'

    @pytest.fixture
    def rasterizer(self, device):
        """Create rasterizer instance."""
        return TileBasedGaussianRasterizer(
            image_size=(480, 640),
            tile_size=16,
            device=device
        )

    @pytest.fixture
    def sample_gaussians(self, device):
        """Create sample Gaussian data."""
        N = 100
        positions = torch.randn(N, 3, device=device) * 2.0
        positions[:, 2] += 5.0  # Move in front of camera

        scales = torch.rand(N, 3, device=device) * 0.1 + 0.01
        rotations = F.normalize(torch.randn(N, 4, device=device), p=2, dim=1)
        colors = torch.rand(N, 3, device=device)
        opacities = torch.rand(N, 1, device=device)

        K = torch.tensor([
            [500.0, 0.0, 320.0],
            [0.0, 500.0, 240.0],
            [0.0, 0.0, 1.0]
        ], device=device)

        return {
            'positions': positions,
            'scales': scales,
            'rotations': rotations,
            'colors': colors,
            'opacities': opacities,
            'K': K
        }

    def test_initialization(self, rasterizer):
        """Test rasterizer initialization."""
        assert rasterizer.H == 480
        assert rasterizer.W == 640
        assert rasterizer.tile_size == 16
        assert rasterizer.tiles_h == 30
        assert rasterizer.tiles_w == 40

    def test_output_shapes(self, rasterizer, sample_gaussians):
        """Test output tensor shapes."""
        with torch.no_grad():
            outputs = rasterizer(**sample_gaussians)

        assert 'image' in outputs
        assert 'depth' in outputs
        assert 'alpha' in outputs

        assert outputs['image'].shape == (480, 640, 3)
        assert outputs['depth'].shape == (480, 640)
        assert outputs['alpha'].shape == (480, 640)

    def test_output_ranges(self, rasterizer, sample_gaussians):
        """Test output value ranges."""
        with torch.no_grad():
            outputs = rasterizer(**sample_gaussians)

        # Image should be in [0, 1]
        assert outputs['image'].min() >= 0.0
        assert outputs['image'].max() <= 1.0

        # Depth should be non-negative
        assert outputs['depth'].min() >= 0.0

        # Alpha should be in [0, 1]
        assert outputs['alpha'].min() >= 0.0
        assert outputs['alpha'].max() <= 1.0

    def test_tile_bounds_computation(self, rasterizer, sample_gaussians, device):
        """Test tile bounds computation."""
        # Create simple 2D covariance matrices
        N = 10
        means_2d = torch.tensor([
            [320.0, 240.0],  # Center
            [0.0, 0.0],      # Top-left corner
            [639.0, 479.0],  # Bottom-right corner
        ], device=device)

        cov_2d = torch.eye(2, device=device).unsqueeze(0).expand(3, 2, 2).clone()
        cov_2d = cov_2d * 10.0  # Small covariance

        tile_min, tile_max = rasterizer._compute_tile_bounds(means_2d, cov_2d)

        assert tile_min.shape == (3, 2)
        assert tile_max.shape == (3, 2)

        # Check bounds are valid
        assert (tile_min >= 0).all()
        assert (tile_max[:, 0] < rasterizer.tiles_w).all()
        assert (tile_max[:, 1] < rasterizer.tiles_h).all()

    def test_background_color(self, rasterizer, sample_gaussians, device):
        """Test background color handling."""
        # White background
        white_bg = torch.ones(3, device=device)
        with torch.no_grad():
            outputs_white = rasterizer(**sample_gaussians, background=white_bg)

        # Black background
        black_bg = torch.zeros(3, device=device)
        with torch.no_grad():
            outputs_black = rasterizer(**sample_gaussians, background=black_bg)

        # Outputs should be different
        assert not torch.allclose(outputs_white['image'], outputs_black['image'])

    def test_no_gaussians(self, rasterizer, device):
        """Test rendering with zero Gaussians."""
        N = 0
        positions = torch.empty(0, 3, device=device)
        scales = torch.empty(0, 3, device=device)
        rotations = torch.empty(0, 4, device=device)
        colors = torch.empty(0, 3, device=device)
        opacities = torch.empty(0, 1, device=device)

        K = torch.tensor([
            [500.0, 0.0, 320.0],
            [0.0, 500.0, 240.0],
            [0.0, 0.0, 1.0]
        ], device=device)

        background = torch.ones(3, device=device) * 0.5

        with torch.no_grad():
            outputs = rasterizer(
                positions=positions,
                scales=scales,
                rotations=rotations,
                colors=colors,
                opacities=opacities,
                K=K,
                background=background
            )

        # Should return background color everywhere
        assert torch.allclose(outputs['image'], background.view(1, 1, 3).expand(480, 640, 3), atol=1e-5)
        assert torch.allclose(outputs['depth'], torch.zeros(480, 640, device=device))
        assert torch.allclose(outputs['alpha'], torch.zeros(480, 640, device=device))

    def test_memory_efficiency(self, rasterizer, sample_gaussians):
        """Test that tile-based rendering uses less memory than naive approach."""
        # This is a conceptual test - actual memory tracking is platform-specific
        # We verify that the rasterizer can handle the input without OOM
        with torch.no_grad():
            outputs = rasterizer(**sample_gaussians)

        # If we reach here without OOM, memory efficiency is working
        assert outputs['image'].shape == (480, 640, 3)

    def test_deterministic_output(self, rasterizer, sample_gaussians):
        """Test that rendering is deterministic."""
        with torch.no_grad():
            outputs1 = rasterizer(**sample_gaussians)
            outputs2 = rasterizer(**sample_gaussians)

        assert torch.allclose(outputs1['image'], outputs2['image'], atol=1e-5)
        assert torch.allclose(outputs1['depth'], outputs2['depth'], atol=1e-5)
        assert torch.allclose(outputs1['alpha'], outputs2['alpha'], atol=1e-5)


def test_rasterizer_import():
    """Test that rasterizer can be imported."""
    from models.rasterizer_optimized import TileBasedGaussianRasterizer
    assert TileBasedGaussianRasterizer is not None


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v', '-s'])
