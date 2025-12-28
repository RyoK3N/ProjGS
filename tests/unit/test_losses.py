"""
Unit Tests for Loss Functions
==============================

Tests for L1, SSIM, and combined Gaussian Splatting loss.

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

from models.losses import SSIMLoss, GaussianSplattingLoss, compute_psnr, compute_ssim


class TestSSIMLoss:
    """Test suite for SSIM loss."""

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
    def ssim_loss(self, device):
        """Create SSIM loss instance."""
        return SSIMLoss(window_size=11).to(device)

    def test_identical_images(self, ssim_loss, device):
        """Test SSIM loss on identical images (should be 0)."""
        img = torch.rand(1, 480, 640, 3, device=device)
        loss = ssim_loss(img, img)

        # SSIM of identical images should be 1, so loss should be 0
        assert torch.allclose(loss, torch.tensor(0.0, device=device), atol=1e-5)

    def test_completely_different_images(self, ssim_loss, device):
        """Test SSIM loss on completely different images."""
        img1 = torch.zeros(1, 480, 640, 3, device=device)
        img2 = torch.ones(1, 480, 640, 3, device=device)

        loss = ssim_loss(img1, img2)

        # Loss should be close to 1 (SSIM close to 0)
        assert loss > 0.5

    def test_output_range(self, ssim_loss, device):
        """Test SSIM loss is in valid range [0, 1]."""
        img1 = torch.rand(1, 480, 640, 3, device=device)
        img2 = torch.rand(1, 480, 640, 3, device=device)

        loss = ssim_loss(img1, img2)

        assert loss >= 0.0
        assert loss <= 1.0

    def test_gradient_flow(self, ssim_loss, device):
        """Test that gradients flow through SSIM loss."""
        img1 = torch.rand(1, 480, 640, 3, device=device, requires_grad=True)
        img2 = torch.rand(1, 480, 640, 3, device=device)

        loss = ssim_loss(img1, img2)
        loss.backward()

        assert img1.grad is not None
        assert not torch.isnan(img1.grad).any()


class TestGaussianSplattingLoss:
    """Test suite for combined Gaussian Splatting loss."""

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
    def gs_loss(self, device):
        """Create Gaussian Splatting loss instance."""
        return GaussianSplattingLoss(
            lambda_l1=0.8,
            lambda_dssim=0.2
        ).to(device)

    def test_identical_images(self, gs_loss, device):
        """Test loss on identical images (should be 0)."""
        img = torch.rand(1, 480, 640, 3, device=device)
        loss = gs_loss(img, img)

        # Both L1 and SSIM should be 0
        assert torch.allclose(loss, torch.tensor(0.0, device=device), atol=1e-5)

    def test_return_components(self, gs_loss, device):
        """Test returning loss components."""
        img1 = torch.rand(1, 480, 640, 3, device=device)
        img2 = torch.rand(1, 480, 640, 3, device=device)

        total_loss, l1_loss, ssim_loss = gs_loss(img1, img2, return_components=True)

        # Check that total loss is weighted combination
        expected_total = 0.8 * l1_loss + 0.2 * ssim_loss
        assert torch.allclose(total_loss, expected_total, atol=1e-5)

    def test_loss_weights(self, device):
        """Test different loss weights."""
        # L1 only
        loss_l1_only = GaussianSplattingLoss(lambda_l1=1.0, lambda_dssim=0.0).to(device)

        # SSIM only
        loss_ssim_only = GaussianSplattingLoss(lambda_l1=0.0, lambda_dssim=1.0).to(device)

        img1 = torch.rand(1, 480, 640, 3, device=device)
        img2 = torch.rand(1, 480, 640, 3, device=device)

        loss1 = loss_l1_only(img1, img2)
        loss2 = loss_ssim_only(img1, img2)

        # Different weights should give different losses
        assert not torch.allclose(loss1, loss2)

    def test_gradient_flow(self, gs_loss, device):
        """Test that gradients flow through combined loss."""
        img1 = torch.rand(1, 480, 640, 3, device=device, requires_grad=True)
        img2 = torch.rand(1, 480, 640, 3, device=device)

        loss = gs_loss(img1, img2)
        loss.backward()

        assert img1.grad is not None
        assert not torch.isnan(img1.grad).any()
        assert not torch.isinf(img1.grad).any()


class TestPSNR:
    """Test suite for PSNR computation."""

    @pytest.fixture
    def device(self):
        """Get available device."""
        if torch.backends.mps.is_available():
            return 'mps'
        elif torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'

    def test_identical_images(self, device):
        """Test PSNR of identical images (should be infinite)."""
        img = torch.rand(1, 480, 640, 3, device=device)
        psnr = compute_psnr(img, img)

        # Should be very high (clamped at 100)
        assert psnr >= 50.0

    def test_completely_different_images(self, device):
        """Test PSNR of completely different images."""
        img1 = torch.zeros(1, 480, 640, 3, device=device)
        img2 = torch.ones(1, 480, 640, 3, device=device)

        psnr = compute_psnr(img1, img2)

        # Should be low
        assert psnr < 10.0

    def test_known_mse(self, device):
        """Test PSNR with known MSE."""
        # Create images with known MSE
        img1 = torch.zeros(1, 100, 100, 3, device=device)
        img2 = torch.ones(1, 100, 100, 3, device=device) * 0.1  # MSE = 0.01

        psnr = compute_psnr(img1, img2)

        # PSNR = 10 * log10(1^2 / 0.01) = 20
        expected_psnr = 20.0
        assert abs(psnr - expected_psnr) < 0.5

    def test_output_type(self, device):
        """Test PSNR returns float."""
        img1 = torch.rand(1, 480, 640, 3, device=device)
        img2 = torch.rand(1, 480, 640, 3, device=device)

        psnr = compute_psnr(img1, img2)

        assert isinstance(psnr, float)


class TestSSIMMetric:
    """Test suite for SSIM metric computation."""

    @pytest.fixture
    def device(self):
        """Get available device."""
        if torch.backends.mps.is_available():
            return 'mps'
        elif torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'

    def test_identical_images(self, device):
        """Test SSIM of identical images (should be 1)."""
        img = torch.rand(1, 480, 640, 3, device=device)
        ssim = compute_ssim(img, img)

        # Should be very close to 1
        assert ssim >= 0.99

    def test_completely_different_images(self, device):
        """Test SSIM of completely different images."""
        img1 = torch.zeros(1, 480, 640, 3, device=device)
        img2 = torch.ones(1, 480, 640, 3, device=device)

        ssim = compute_ssim(img1, img2)

        # Should be low
        assert ssim < 0.5

    def test_output_range(self, device):
        """Test SSIM is in valid range [0, 1]."""
        img1 = torch.rand(1, 480, 640, 3, device=device)
        img2 = torch.rand(1, 480, 640, 3, device=device)

        ssim = compute_ssim(img1, img2)

        assert ssim >= 0.0
        assert ssim <= 1.0

    def test_output_type(self, device):
        """Test SSIM returns float."""
        img1 = torch.rand(1, 480, 640, 3, device=device)
        img2 = torch.rand(1, 480, 640, 3, device=device)

        ssim = compute_ssim(img1, img2)

        assert isinstance(ssim, float)

    def test_symmetry(self, device):
        """Test SSIM is symmetric."""
        img1 = torch.rand(1, 480, 640, 3, device=device)
        img2 = torch.rand(1, 480, 640, 3, device=device)

        ssim1 = compute_ssim(img1, img2)
        ssim2 = compute_ssim(img2, img1)

        assert abs(ssim1 - ssim2) < 1e-5


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v', '-s'])
