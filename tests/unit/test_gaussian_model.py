"""
Unit Tests for Gaussian Model
==============================

Tests for learnable 3D Gaussian model with adaptive density control.

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

from models.gaussian_model import GaussianModel, DensificationConfig


class TestGaussianModel:
    """Test suite for GaussianModel."""

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
    def sample_model(self, device):
        """Create sample Gaussian model."""
        N = 100
        positions = torch.randn(N, 3, device=device)
        features = torch.randn(N, 3, device=device)
        opacities = torch.ones(N, 1, device=device) * 0.5
        scales = torch.ones(N, 3, device=device) * 0.01
        rotations = F.normalize(torch.randn(N, 4, device=device), p=2, dim=1)

        model = GaussianModel(
            positions=positions,
            features=features,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
            device=device
        )
        return model

    def test_initialization(self, sample_model):
        """Test model initialization."""
        assert sample_model.get_num_gaussians() == 100
        assert sample_model._positions.requires_grad
        assert sample_model._features.requires_grad
        assert sample_model._opacities.requires_grad
        assert sample_model._scales.requires_grad
        assert sample_model._rotations.requires_grad

    def test_parameter_shapes(self, sample_model):
        """Test parameter tensor shapes."""
        N = 100
        assert sample_model._positions.shape == (N, 3)
        assert sample_model._features.shape == (N, 3)
        assert sample_model._opacities.shape == (N, 1)
        assert sample_model._scales.shape == (N, 3)
        assert sample_model._rotations.shape == (N, 4)

    def test_gradient_accumulator_shape(self, sample_model):
        """Test gradient accumulator has correct shape (scalar per Gaussian)."""
        N = 100
        # CRITICAL: Should be [N], not [N, 3]
        assert sample_model.position_grad_accum.shape == (N,)
        assert sample_model.position_grad_count.shape == (N,)

    def test_property_transforms(self, sample_model):
        """Test that properties apply correct transformations."""
        # Features: sigmoid
        features = sample_model.features
        assert features.min() >= 0.0
        assert features.max() <= 1.0

        # Opacities: sigmoid
        opacities = sample_model.opacities
        assert opacities.min() >= 0.0
        assert opacities.max() <= 1.0

        # Scales: exp
        scales = sample_model.scales
        assert (scales > 0).all()

        # Rotations: normalized
        rotations = sample_model.rotations
        norms = rotations.norm(p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_gradient_accumulation(self, sample_model, device):
        """Test position gradient accumulation."""
        # Simulate gradients
        sample_model._positions.grad = torch.randn_like(sample_model._positions) * 0.01

        # Accumulate
        sample_model.accumulate_position_gradients()

        # Check accumulation shape (CRITICAL: should be [N])
        assert sample_model.position_grad_accum.shape == (100,)
        assert sample_model.position_grad_count.shape == (100,)

        # Check values
        assert (sample_model.position_grad_accum > 0).all()
        assert (sample_model.position_grad_count == 1).all()

        # Accumulate again
        sample_model._positions.grad = torch.randn_like(sample_model._positions) * 0.01
        sample_model.accumulate_position_gradients()

        assert (sample_model.position_grad_count == 2).all()

    def test_average_gradient(self, sample_model):
        """Test average gradient computation."""
        # Simulate accumulated gradients
        sample_model.position_grad_accum = torch.rand(100, device=sample_model.device) * 10
        sample_model.position_grad_count = torch.ones(100, device=sample_model.device) * 5

        avg_grad = sample_model.get_average_position_gradient()

        expected = sample_model.position_grad_accum / 5
        assert torch.allclose(avg_grad, expected, atol=1e-5)

    def test_gradient_reset(self, sample_model):
        """Test gradient accumulator reset."""
        # Add some values
        sample_model.position_grad_accum = torch.rand(100, device=sample_model.device)
        sample_model.position_grad_count = torch.ones(100, device=sample_model.device)

        # Reset
        sample_model.reset_gradient_accumulators()

        assert (sample_model.position_grad_accum == 0).all()
        assert (sample_model.position_grad_count == 0).all()

    def test_densify_and_split(self, sample_model, device):
        """Test Gaussian splitting for densification."""
        # Set high gradients on some Gaussians
        sample_model.position_grad_accum = torch.rand(100, device=device) * 0.001
        sample_model.position_grad_accum[:10] = 0.01  # High gradient
        sample_model.position_grad_count = torch.ones(100, device=device)

        # Make some Gaussians large
        with torch.no_grad():
            sample_model._scales[:10] = torch.log(torch.ones(10, 3, device=device) * 0.1)

        initial_count = sample_model.get_num_gaussians()

        # Split
        sample_model.densify_and_split(
            grad_threshold=0.005,
            split_threshold=0.02,
            N=2
        )

        final_count = sample_model.get_num_gaussians()

        # Should have more Gaussians (split into 2, but removed originals)
        # 10 Gaussians split into 20, 10 removed, net gain of 10
        assert final_count == initial_count + 10

        # Check gradient accumulators updated
        assert sample_model.position_grad_accum.shape == (final_count,)
        assert sample_model.position_grad_count.shape == (final_count,)

    def test_densify_and_clone(self, sample_model, device):
        """Test Gaussian cloning for densification."""
        # Set high gradients on some Gaussians
        sample_model.position_grad_accum = torch.rand(100, device=device) * 0.001
        sample_model.position_grad_accum[:10] = 0.01  # High gradient
        sample_model.position_grad_count = torch.ones(100, device=device)

        # Make some Gaussians small
        with torch.no_grad():
            sample_model._scales[:10] = torch.log(torch.ones(10, 3, device=device) * 0.001)

        initial_count = sample_model.get_num_gaussians()

        # Clone
        sample_model.densify_and_clone(
            grad_threshold=0.005,
            split_threshold=0.02
        )

        final_count = sample_model.get_num_gaussians()

        # Should have more Gaussians (cloned 10)
        assert final_count == initial_count + 10

        # Check gradient accumulators updated
        assert sample_model.position_grad_accum.shape == (final_count,)
        assert sample_model.position_grad_count.shape == (final_count,)

    def test_prune_opacity(self, sample_model, device):
        """Test pruning by opacity."""
        # Set some Gaussians to low opacity
        with torch.no_grad():
            sample_model._opacities[:20] = torch.logit(torch.tensor(0.001, device=device))

        initial_count = sample_model.get_num_gaussians()

        # Prune
        sample_model.prune(min_opacity=0.005)

        final_count = sample_model.get_num_gaussians()

        # Should have fewer Gaussians
        assert final_count == initial_count - 20

        # All remaining should have opacity > threshold
        assert (sample_model.opacities > 0.005).all()

    def test_prune_scale(self, sample_model, device):
        """Test pruning by scale."""
        # Set some Gaussians to large scale
        with torch.no_grad():
            sample_model._scales[:20] = torch.log(torch.ones(20, 3, device=device) * 100.0)

        initial_count = sample_model.get_num_gaussians()

        # Prune
        sample_model.prune(min_opacity=0.001, max_scale=0.5)

        final_count = sample_model.get_num_gaussians()

        # Should have fewer Gaussians
        assert final_count < initial_count

    def test_opacity_reset(self, sample_model, device):
        """Test opacity reset."""
        # Reset to specific value
        sample_model.reset_opacity(new_opacity=0.5)

        # All opacities should be 0.5
        expected = torch.ones(100, 1, device=device) * 0.5
        assert torch.allclose(sample_model.opacities, expected, atol=1e-3)

    def test_state_dict(self, sample_model):
        """Test state dictionary save/load."""
        # Get state
        state = sample_model.state_dict()

        # Check keys
        expected_keys = {
            'positions', 'features', 'opacities', 'scales', 'rotations',
            'position_grad_accum', 'position_grad_count', 'scene_extent'
        }
        assert set(state.keys()) == expected_keys

        # Check shapes
        assert state['positions'].shape == (100, 3)
        assert state['position_grad_accum'].shape == (100,)  # CRITICAL: should be [N]
        assert state['position_grad_count'].shape == (100,)

    def test_load_state_dict(self, sample_model, device):
        """Test loading from state dictionary."""
        # Save state
        state = sample_model.state_dict()

        # Create new model
        new_model = GaussianModel(
            positions=torch.randn(10, 3, device=device),
            features=torch.randn(10, 3, device=device),
            opacities=torch.ones(10, 1, device=device),
            scales=torch.ones(10, 3, device=device),
            rotations=F.normalize(torch.randn(10, 4, device=device), p=2, dim=1),
            device=device
        )

        # Load state
        new_model.load_state_dict(state)

        # Check loaded
        assert new_model.get_num_gaussians() == 100
        assert torch.allclose(new_model._positions, sample_model._positions)
        assert torch.allclose(new_model.position_grad_accum, sample_model.position_grad_accum)

    def test_scene_extent_computation(self, device):
        """Test scene extent calculation."""
        # Create model with known bounds
        positions = torch.tensor([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=torch.float32, device=device)

        model = GaussianModel(
            positions=positions,
            features=torch.zeros(4, 3, device=device),
            opacities=torch.ones(4, 1, device=device),
            scales=torch.ones(4, 3, device=device),
            rotations=F.normalize(torch.randn(4, 4, device=device), p=2, dim=1),
            device=device
        )

        # Scene extent should be sqrt(3) for unit cube
        expected = np.sqrt(3)
        assert abs(model.scene_extent - expected) < 0.01


def test_densification_config():
    """Test DensificationConfig dataclass."""
    config = DensificationConfig()

    assert config.start_iteration == 500
    assert config.stop_iteration == 15000
    assert config.interval == 100
    assert config.grad_threshold == 2e-4
    assert config.opacity_reset_interval == 3000
    assert config.split_threshold == 0.02
    assert config.min_opacity == 0.005


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v', '-s'])
