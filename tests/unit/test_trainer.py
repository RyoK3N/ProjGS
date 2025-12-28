"""
Unit Tests for ProjGS Trainer
==============================

Tests for training loop, optimizers, and checkpointing.

Author: ProjGS Research Team
Date: December 2025
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Disable torch.compile to avoid ONNX import issues
import os
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import torch.nn.functional as F
import pytest
import numpy as np
import tempfile
import shutil

from models.gaussian_model import GaussianModel
from training.trainer import ProjGSTrainer


class TestProjGSTrainer:
    """Test suite for ProjGSTrainer."""

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
    def temp_dir(self):
        """Create temporary directory for outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def config(self, temp_dir):
        """Create test configuration."""
        return {
            'rendering': {'resolution': [240, 320]},  # Smaller for testing
            'training': {
                'optimizer': {
                    'position_lr': 1.6e-4,
                    'feature_lr': 2.5e-3,
                    'opacity_lr': 5.0e-2,
                    'scaling_lr': 5.0e-3,
                    'rotation_lr': 1.0e-3
                },
                'lr_schedule': {
                    'position_lr_init': 1.6e-4,
                    'position_lr_final': 1.6e-6,
                    'position_lr_delay_mult': 0.01,
                    'position_lr_max_steps': 1000
                },
                'loss': {
                    'lambda_l1': 0.8,
                    'lambda_dssim': 0.2
                },
                'amp': {'enabled': False}
            },
            'densification': {
                'start_iteration': 10,
                'stop_iteration': 100,
                'interval': 10,
                'grad_threshold': 2e-4,
                'opacity_reset_interval': 50,
                'split_threshold': 0.02,
                'min_opacity': 0.005
            },
            'pruning': {
                'interval': 10,
                'min_opacity': 0.005
            },
            'logging': {
                'tensorboard_dir': f'{temp_dir}/runs',
                'save_dir': f'{temp_dir}/checkpoints',
                'log_interval': 5,
                'image_interval': 10
            },
            'checkpointing': {
                'save_interval': 20,
                'keep_last_n': 2
            },
            'memory': {
                'empty_cache_interval': 10
            }
        }

    @pytest.fixture
    def sample_model(self, device):
        """Create sample Gaussian model."""
        N = 100
        positions = torch.randn(N, 3, device=device)
        features = torch.randn(N, 3, device=device)
        opacities = torch.ones(N, 1, device=device) * 0.5
        scales = torch.ones(N, 3, device=device) * 0.01
        rotations = F.normalize(torch.randn(N, 4, device=device), p=2, dim=1)

        return GaussianModel(
            positions=positions,
            features=features,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
            device=device
        )

    @pytest.fixture
    def trainer(self, sample_model, config, device):
        """Create trainer instance."""
        return ProjGSTrainer(sample_model, config, device)

    def test_initialization(self, trainer, sample_model):
        """Test trainer initialization."""
        assert trainer.model is sample_model
        assert trainer.iteration == 0
        assert trainer.rasterizer is not None
        assert trainer.criterion is not None
        assert len(trainer.optimizers) == 5

    def test_optimizers_created(self, trainer):
        """Test that all optimizers are created."""
        expected_opts = {'position', 'features', 'opacities', 'scales', 'rotations'}
        assert set(trainer.optimizers.keys()) == expected_opts

        # Check each has correct parameter
        assert trainer.optimizers['position'].param_groups[0]['params'][0] is trainer.model._positions
        assert trainer.optimizers['features'].param_groups[0]['params'][0] is trainer.model._features

    def test_learning_rate_update(self, trainer):
        """Test learning rate scheduling."""
        initial_lr = trainer.optimizers['position'].param_groups[0]['lr']

        # Update at iteration 500
        trainer.iteration = 500
        trainer._update_learning_rate(500)

        updated_lr = trainer.optimizers['position'].param_groups[0]['lr']

        # LR should have decayed
        assert updated_lr < initial_lr

    def test_train_iteration_shapes(self, trainer, device):
        """Test training iteration output shapes."""
        # Create dummy input
        rgb_gt = torch.rand(240, 320, 3, device=device)
        K = torch.tensor([
            [250.0, 0.0, 160.0],
            [0.0, 250.0, 120.0],
            [0.0, 0.0, 1.0]
        ], device=device)

        metrics, outputs = trainer.train_iteration(rgb_gt, K)

        # Check metrics
        assert 'loss' in metrics
        assert 'l1_loss' in metrics
        assert 'ssim_loss' in metrics
        assert 'psnr' in metrics
        assert 'ssim' in metrics
        assert 'num_gaussians' in metrics

        # Check outputs
        assert 'image' in outputs
        assert 'depth' in outputs
        assert 'alpha' in outputs

    def test_gradient_accumulation(self, trainer, device):
        """Test that gradients are accumulated."""
        rgb_gt = torch.rand(240, 320, 3, device=device)
        K = torch.tensor([
            [250.0, 0.0, 160.0],
            [0.0, 250.0, 120.0],
            [0.0, 0.0, 1.0]
        ], device=device)

        # Before training
        assert (trainer.model.position_grad_accum == 0).all()

        # Train
        metrics, outputs = trainer.train_iteration(rgb_gt, K)

        # After training, gradients should be accumulated
        assert (trainer.model.position_grad_accum > 0).any()

    def test_densify_and_prune_interval(self, trainer, device):
        """Test densification happens at correct intervals."""
        rgb_gt = torch.rand(240, 320, 3, device=device)
        K = torch.tensor([
            [250.0, 0.0, 160.0],
            [0.0, 250.0, 120.0],
            [0.0, 0.0, 1.0]
        ], device=device)

        # Set iteration to densification start
        trainer.iteration = 10

        # Accumulate gradients
        for _ in range(5):
            trainer.train_iteration(rgb_gt, K)

        initial_count = trainer.model.get_num_gaussians()

        # Trigger densification
        trainer.densify_and_prune()

        # Count may change (split/clone/prune)
        # Just verify it doesn't crash and maintains valid state
        assert trainer.model.get_num_gaussians() > 0

    def test_checkpoint_save_load(self, trainer, temp_dir, device):
        """Test checkpoint saving and loading."""
        # Train for a few iterations
        rgb_gt = torch.rand(240, 320, 3, device=device)
        K = torch.tensor([
            [250.0, 0.0, 160.0],
            [0.0, 250.0, 120.0],
            [0.0, 0.0, 1.0]
        ], device=device)

        for i in range(5):
            trainer.train_iteration(rgb_gt, K)
            trainer.iteration += 1

        # Save checkpoint
        trainer.save_checkpoint()

        # Check file exists
        checkpoint_path = Path(temp_dir) / 'checkpoints' / f'checkpoint_iter_{trainer.iteration:06d}.pth'
        assert checkpoint_path.exists()

        # Create new trainer
        from models.gaussian_model import GaussianModel
        N = 100
        positions = torch.randn(N, 3, device=device)
        features = torch.randn(N, 3, device=device)
        opacities = torch.ones(N, 1, device=device) * 0.5
        scales = torch.ones(N, 3, device=device) * 0.01
        rotations = F.normalize(torch.randn(N, 4, device=device), p=2, dim=1)

        new_model = GaussianModel(positions, features, opacities, scales, rotations, device)
        new_trainer = ProjGSTrainer(new_model, trainer.config, device)

        # Load checkpoint
        new_trainer.load_checkpoint(str(checkpoint_path))

        # Check iteration restored
        assert new_trainer.iteration == trainer.iteration

    def test_log_metrics(self, trainer):
        """Test metrics logging doesn't crash."""
        metrics = {
            'loss': 0.5,
            'psnr': 25.0,
            'ssim': 0.85,
            'num_gaussians': 100
        }

        # Should not crash
        trainer.log_metrics(metrics)

    def test_log_images(self, trainer, device):
        """Test image logging doesn't crash."""
        outputs = {
            'image': torch.rand(240, 320, 3, device=device),
            'depth': torch.rand(240, 320, device=device),
            'alpha': torch.rand(240, 320, device=device)
        }
        rgb_gt = torch.rand(240, 320, 3, device=device)

        # Should not crash
        trainer.log_images(outputs, rgb_gt)

    def test_cleanup(self, trainer):
        """Test cleanup doesn't crash."""
        trainer.cleanup()

    def test_amp_disabled_on_mps(self, config, sample_model):
        """Test that AMP is disabled on MPS."""
        if torch.backends.mps.is_available():
            config['training']['amp']['enabled'] = True
            trainer = ProjGSTrainer(sample_model, config, 'mps')

            # Should be disabled
            assert trainer.use_amp is False
            assert trainer.scaler is None

    def test_multiple_iterations(self, trainer, device):
        """Test running multiple training iterations."""
        rgb_gt = torch.rand(240, 320, 3, device=device)
        K = torch.tensor([
            [250.0, 0.0, 160.0],
            [0.0, 250.0, 120.0],
            [0.0, 0.0, 1.0]
        ], device=device)

        losses = []
        for i in range(10):
            metrics, outputs = trainer.train_iteration(rgb_gt, K)
            losses.append(metrics['loss'])
            trainer.iteration += 1

        # Should complete without errors
        assert len(losses) == 10
        assert all(isinstance(loss, float) for loss in losses)


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v', '-s'])
