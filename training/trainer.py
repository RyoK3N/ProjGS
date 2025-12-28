"""
ProjGS Trainer
==============

Complete training loop for 3D Gaussian Splatting with:
- Separate optimizers for different parameter groups
- Learning rate scheduling
- Adaptive densification and pruning
- TensorBoard logging
- Checkpointing
- Mixed precision training (AMP)

Author: ProjGS Research Team
Date: December 2025
"""

import torch
import torch.nn as nn

# Disable torch.compile globally to avoid ONNX import issues
import os
os.environ['TORCH_COMPILE_DISABLE'] = '1'

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any
from tqdm import tqdm
import time

from models.gaussian_model import GaussianModel, DensificationConfig
from models.rasterizer_optimized import TileBasedGaussianRasterizer
from models.losses import GaussianSplattingLoss, compute_psnr, compute_ssim


class ProjGSTrainer:
    """
    Trainer for Projection-Guided Gaussian Splatting.

    Args:
        model: GaussianModel instance
        config: Training configuration dictionary
        device: Device to train on
    """

    def __init__(
        self,
        model: GaussianModel,
        config: Dict[str, Any],
        device: str = 'cuda'
    ):
        self.model = model
        self.config = config
        self.device = device
        self.iteration = 0

        # Create memory-optimized tile-based rasterizer
        image_size = tuple(config['rendering']['resolution'])
        self.rasterizer = TileBasedGaussianRasterizer(
            image_size=image_size,
            tile_size=16,  # 16x16 tiles for optimal memory/performance
            device=device
        )

        # Create loss function
        loss_config = config['training']['loss']
        self.criterion = GaussianSplattingLoss(
            lambda_l1=loss_config['lambda_l1'],
            lambda_dssim=loss_config['lambda_dssim']
        ).to(device)

        # Setup optimizers (separate for each parameter group)
        self._setup_optimizers()

        # Setup learning rate schedulers
        self._setup_schedulers()

        # Mixed precision training
        self.use_amp = config['training']['amp']['enabled']
        if self.use_amp:
            # Use device-specific scaler
            if device == 'cuda':
                self.scaler = GradScaler('cuda')
            elif device == 'mps':
                # MPS doesn't support GradScaler yet, disable AMP
                self.use_amp = False
                self.scaler = None
                print("⚠ MPS does not support GradScaler, disabling mixed precision")
            else:
                self.scaler = None
        else:
            self.scaler = None

        # Densification config
        dens_cfg = config['densification']
        self.dens_config = DensificationConfig(
            start_iteration=dens_cfg['start_iteration'],
            stop_iteration=dens_cfg['stop_iteration'],
            interval=dens_cfg['interval'],
            grad_threshold=dens_cfg['grad_threshold'],
            opacity_reset_interval=dens_cfg['opacity_reset_interval'],
            split_threshold=dens_cfg['split_threshold'],
            min_opacity=dens_cfg['min_opacity']
        )

        # Pruning config
        self.prune_config = config['pruning']

        # TensorBoard
        log_dir = Path(config['logging']['tensorboard_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(log_dir))

        # Checkpointing
        self.save_dir = Path(config['logging']['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Logging intervals
        self.log_interval = config['logging']['log_interval']
        self.image_interval = config['logging']['image_interval']
        self.save_interval = config['checkpointing']['save_interval']

        # Memory optimization
        self.empty_cache_interval = config['memory']['empty_cache_interval']

        print(f"\n{'='*80}")
        print("ProjGS Trainer Initialized")
        print(f"{'='*80}")
        print(f"Device: {device}")
        print(f"Initial Gaussians: {model.get_num_gaussians():,}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Tensorboard: {log_dir}")
        print(f"Checkpoints: {self.save_dir}")
        print(f"{'='*80}\n")

    def _setup_optimizers(self):
        """Setup separate optimizers for each parameter group."""
        opt_config = self.config['training']['optimizer']

        # Parameter groups with different learning rates
        # Using capturable=False to avoid torch.compile/ONNX issues
        self.optimizers = {
            'position': optim.Adam(
                [self.model._positions],
                lr=opt_config['position_lr'],
                eps=1e-15,
                capturable=False
            ),
            'features': optim.Adam(
                [self.model._features],
                lr=opt_config['feature_lr'],
                eps=1e-15,
                capturable=False
            ),
            'opacities': optim.Adam(
                [self.model._opacities],
                lr=opt_config['opacity_lr'],
                eps=1e-15,
                capturable=False
            ),
            'scales': optim.Adam(
                [self.model._scales],
                lr=opt_config['scaling_lr'],
                eps=1e-15,
                capturable=False
            ),
            'rotations': optim.Adam(
                [self.model._rotations],
                lr=opt_config['rotation_lr'],
                eps=1e-15,
                capturable=False
            )
        }

    def _setup_schedulers(self):
        """Setup exponential learning rate decay for position."""
        lr_config = self.config['training']['lr_schedule']

        # Exponential decay for position learning rate
        self.position_lr_init = lr_config['position_lr_init']
        self.position_lr_final = lr_config['position_lr_final']
        self.position_lr_delay_mult = lr_config['position_lr_delay_mult']
        self.position_lr_max_steps = lr_config['position_lr_max_steps']

    def _update_learning_rate(self, iteration: int):
        """Update learning rates based on iteration."""
        # Exponential decay for position learning rate
        if iteration < self.position_lr_max_steps:
            t = iteration / self.position_lr_max_steps
            lr = self.position_lr_init * (self.position_lr_final / self.position_lr_init) ** t

            # Apply delay multiplier
            delay_rate = self.position_lr_delay_mult + (1 - self.position_lr_delay_mult) * \
                        np.sin(0.5 * np.pi * np.clip(t, 0, 1))
            lr = lr * delay_rate

            for param_group in self.optimizers['position'].param_groups:
                param_group['lr'] = lr

    def train_iteration(
        self,
        rgb_gt: torch.Tensor,
        K: torch.Tensor,
        background: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Single training iteration.

        Args:
            rgb_gt: Ground truth RGB image [H, W, 3]
            K: Camera intrinsic matrix [3, 3]
            background: Background color [3] or None

        Returns:
            Dictionary with loss values and metrics
        """
        # Update learning rate
        self._update_learning_rate(self.iteration)

        # Zero gradients
        for opt in self.optimizers.values():
            opt.zero_grad()

        # Forward pass with mixed precision
        device_type = 'cuda' if self.device == 'cuda' else 'cpu'  # MPS uses cpu autocast
        with autocast(device_type=device_type, enabled=self.use_amp, dtype=torch.float16 if self.use_amp else torch.float32):
            # Render
            outputs = self.rasterizer(
                positions=self.model.positions,
                scales=self.model.scales,
                rotations=self.model.rotations,
                colors=self.model.features,
                opacities=self.model.opacities,
                K=K,
                background=background
            )

            # Get rendered image [H, W, 3]
            rgb_pred = outputs['image']

            # Ensure rgb_gt is on correct device and in [H, W, 3] format
            if rgb_gt.dim() == 4:
                rgb_gt = rgb_gt.squeeze(0)
            rgb_gt = rgb_gt.to(self.device)

            # Add batch dimension for loss computation
            rgb_pred_batch = rgb_pred.unsqueeze(0)  # [1, H, W, 3]
            rgb_gt_batch = rgb_gt.unsqueeze(0)      # [1, H, W, 3]

            # Compute loss
            loss, l1_loss, ssim_loss = self.criterion(
                rgb_pred_batch, rgb_gt_batch, return_components=True
            )

        # Backward pass
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Accumulate position gradients for densification
        self.model.accumulate_position_gradients()

        # Optimizer step
        if self.use_amp and self.scaler is not None:
            for opt in self.optimizers.values():
                self.scaler.step(opt)
            self.scaler.update()
        else:
            for opt in self.optimizers.values():
                opt.step()

        # Compute metrics
        with torch.no_grad():
            psnr = compute_psnr(rgb_pred_batch, rgb_gt_batch)
            ssim = compute_ssim(rgb_pred_batch, rgb_gt_batch)

        metrics = {
            'loss': loss.item(),
            'l1_loss': l1_loss.item(),
            'ssim_loss': ssim_loss.item(),
            'psnr': psnr,
            'ssim': ssim,
            'num_gaussians': self.model.get_num_gaussians()
        }

        return metrics, outputs

    def densify_and_prune(self):
        """Perform adaptive densification and pruning."""
        # Densification
        if self.iteration >= self.dens_config.start_iteration and \
           self.iteration <= self.dens_config.stop_iteration and \
           self.iteration % self.dens_config.interval == 0:

            # Split large Gaussians
            self.model.densify_and_split(
                grad_threshold=self.dens_config.grad_threshold,
                split_threshold=self.dens_config.split_threshold
            )

            # Clone small Gaussians
            self.model.densify_and_clone(
                grad_threshold=self.dens_config.grad_threshold,
                split_threshold=self.dens_config.split_threshold
            )

            # Reset gradient accumulators
            self.model.reset_gradient_accumulators()

            # Recreate optimizers with new parameters
            self._setup_optimizers()

        # Pruning
        if self.iteration % self.prune_config['interval'] == 0:
            self.model.prune(
                min_opacity=self.prune_config['min_opacity'],
                max_scale=self.prune_config.get('max_scale', None)
            )

            # Recreate optimizers
            self._setup_optimizers()

        # Opacity reset
        if self.iteration % self.dens_config.opacity_reset_interval == 0:
            self.model.reset_opacity(new_opacity=0.01)

    def log_metrics(self, metrics: Dict[str, float], phase: str = 'train'):
        """Log metrics to TensorBoard."""
        for key, value in metrics.items():
            self.writer.add_scalar(f'{phase}/{key}', value, self.iteration)

        # Log learning rates
        for name, opt in self.optimizers.items():
            lr = opt.param_groups[0]['lr']
            self.writer.add_scalar(f'lr/{name}', lr, self.iteration)

    def log_images(self, outputs: Dict[str, torch.Tensor], rgb_gt: torch.Tensor):
        """Log images to TensorBoard."""
        with torch.no_grad():
            # Rendered image
            rgb_pred = outputs['image'].cpu().numpy()
            self.writer.add_image('train/rendered', rgb_pred, self.iteration, dataformats='HWC')

            # Ground truth
            rgb_gt_np = rgb_gt.cpu().numpy()
            if rgb_gt_np.ndim == 4:
                rgb_gt_np = rgb_gt_np.squeeze(0)
            self.writer.add_image('train/ground_truth', rgb_gt_np, self.iteration, dataformats='HWC')

            # Depth map
            depth = outputs['depth'].cpu().numpy()
            depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
            self.writer.add_image('train/depth', depth_norm, self.iteration, dataformats='HW')

            # Alpha map
            alpha = outputs['alpha'].cpu().numpy()
            self.writer.add_image('train/alpha', alpha, self.iteration, dataformats='HW')

    def save_checkpoint(self, name: Optional[str] = None):
        """Save training checkpoint."""
        if name is None:
            name = f'checkpoint_iter_{self.iteration:06d}.pth'

        checkpoint = {
            'iteration': self.iteration,
            'model_state': self.model.state_dict(),
            'optimizer_states': {
                name: opt.state_dict() for name, opt in self.optimizers.items()
            },
            'config': self.config
        }

        if self.scaler is not None:
            checkpoint['scaler_state'] = self.scaler.state_dict()

        save_path = self.save_dir / name
        torch.save(checkpoint, save_path)

        # Keep only last N checkpoints
        keep_last_n = self.config['checkpointing']['keep_last_n']
        checkpoints = sorted(self.save_dir.glob('checkpoint_iter_*.pth'))
        if len(checkpoints) > keep_last_n:
            for old_ckpt in checkpoints[:-keep_last_n]:
                old_ckpt.unlink()

        print(f"  Saved checkpoint: {save_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.iteration = checkpoint['iteration']
        self.model.load_state_dict(checkpoint['model_state'])

        for name, opt in self.optimizers.items():
            opt.load_state_dict(checkpoint['optimizer_states'][name])

        if self.scaler is not None and 'scaler_state' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state'])

        print(f"✓ Loaded checkpoint from iteration {self.iteration}")

    def cleanup(self):
        """Cleanup resources."""
        self.writer.close()

        # Clear GPU cache
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        elif self.device == 'mps':
            torch.mps.empty_cache()


if __name__ == '__main__':
    print("Testing Trainer...")

    # Create dummy config
    config = {
        'rendering': {'resolution': [480, 640]},
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
                'position_lr_max_steps': 30000
            },
            'loss': {
                'lambda_l1': 0.8,
                'lambda_dssim': 0.2
            },
            'amp': {'enabled': False}
        },
        'densification': {
            'start_iteration': 500,
            'stop_iteration': 15000,
            'interval': 100,
            'grad_threshold': 2e-4,
            'opacity_reset_interval': 3000,
            'split_threshold': 0.02,
            'min_opacity': 0.005
        },
        'pruning': {
            'interval': 100,
            'min_opacity': 0.005
        },
        'logging': {
            'tensorboard_dir': 'runs/test',
            'save_dir': 'checkpoints/test',
            'log_interval': 10,
            'image_interval': 100
        },
        'checkpointing': {
            'save_interval': 500,
            'keep_last_n': 3
        },
        'memory': {
            'empty_cache_interval': 10
        }
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create dummy model
    from models.gaussian_model import GaussianModel
    import torch.nn.functional as F

    N = 100
    positions = torch.randn(N, 3, device=device)
    features = torch.randn(N, 3, device=device)
    opacities = torch.ones(N, 1, device=device) * 0.5
    scales = torch.ones(N, 3, device=device) * 0.01
    rotations = F.normalize(torch.randn(N, 4, device=device), p=2, dim=1)

    model = GaussianModel(positions, features, opacities, scales, rotations, device)

    # Create trainer
    trainer = ProjGSTrainer(model, config, device)

    print("✓ Trainer initialized!")

    trainer.cleanup()
