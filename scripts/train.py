"""
ProjGS Training Script
======================

Main training script for Projection-Guided Gaussian Splatting.

Usage:
    # Mac M4 Pro training:
    python scripts/train.py --config configs/train_mac.yaml

    # A100 training:
    python scripts/train.py --config configs/train_a100.yaml

    # Resume from checkpoint:
    python scripts/train.py --config configs/train_mac.yaml --resume checkpoints/mac/checkpoint_iter_001000.pth

Author: ProjGS Research Team
Date: December 2025
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Disable torch.compile to avoid ONNX import issues
import os
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch

# Monkey patch to disable torch.compile completely
if hasattr(torch, '_dynamo'):
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.reset()

# Disable compile decorator
def _noop_compile(model, *args, **kwargs):
    return model

if hasattr(torch, 'compile'):
    torch.compile = _noop_compile

import torch.nn.functional as F
import numpy as np
import yaml
import argparse
from tqdm import tqdm
import time
from typing import Optional

from data.sunrgbd_loader import SUNRGBDDataset
from models.gaussian_init import CameraAwareGaussianInit
from models.gaussian_model import GaussianModel
from training.trainer import ProjGSTrainer


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def initialize_gaussians_from_scene(
    sample: dict,
    initializer: CameraAwareGaussianInit,
    device: str
) -> GaussianModel:
    """
    Initialize Gaussian model from a single RGB-D scene.

    Args:
        sample: Dataset sample with rgb, depth, K
        initializer: Camera-aware initializer
        device: Device to use

    Returns:
        Initialized GaussianModel
    """
    # Initialize Gaussians
    gaussians = initializer.initialize_from_rgbd(
        rgb=sample['rgb'],
        depth=sample['depth'],
        K=sample['K']
    )

    # Extract parameters
    positions = gaussians['positions'].to(device)        # [N, 3]
    colors = gaussians['colors'].to(device)              # [N, 3]
    opacities = gaussians['opacities'].to(device)        # [N, 1]
    scales = gaussians['scales'].to(device)              # [N, 3]
    rotations = gaussians['rotations'].to(device)        # [N, 4]

    # Convert to model format
    # Features: inverse sigmoid(colors) for optimization
    features = torch.logit(torch.clamp(colors, 0.01, 0.99))

    # Opacities: inverse sigmoid
    opacities = torch.logit(torch.clamp(opacities, 0.01, 0.99))

    # Scales: log-space
    scales = torch.log(scales)

    # Rotations: already normalized quaternions
    rotations = F.normalize(rotations, p=2, dim=1)

    # Create model
    model = GaussianModel(
        positions=positions,
        features=features,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        device=device
    )

    return model


def train(config: dict, resume_path: Optional[str] = None):
    """
    Main training function.

    Args:
        config: Configuration dictionary
        resume_path: Path to checkpoint to resume from
    """
    print("=" * 80)
    print("PROJECTION-GUIDED GAUSSIAN SPLATTING - TRAINING")
    print("=" * 80)

    # Setup device
    device_name = config['device']
    if device_name == 'cuda' and not torch.cuda.is_available():
        print("⚠ CUDA not available, falling back to CPU")
        device = 'cpu'
    elif device_name == 'mps' and not torch.backends.mps.is_available():
        print("⚠ MPS not available, falling back to CPU")
        device = 'cpu'
    else:
        device = device_name

    print(f"\nDevice: {device}")

    # Set random seed
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"Random seed: {seed}")

    # Load dataset
    print("\nLoading dataset...")
    dataset_config = config['dataset']
    dataset = SUNRGBDDataset(
        root_dir=dataset_config['root_dir'],
        split=dataset_config['split'],
        sensors=dataset_config['sensors'],
        resize=tuple(dataset_config['resize'])
    )

    max_scenes = dataset_config.get('max_scenes', None)
    if max_scenes is not None:
        dataset_size = min(max_scenes, len(dataset))
    else:
        dataset_size = len(dataset)

    print(f"✓ Loaded {len(dataset)} scenes (using {dataset_size})")

    # Initialize Gaussians from first scene
    print("\nInitializing Gaussians from first scene...")
    init_config = config['initialization']
    initializer = CameraAwareGaussianInit(
        base_scale=init_config['base_scale'],
        uncertainty_weight=init_config['uncertainty_weight'],
        use_nearest_neighbor=init_config.get('use_nearest_neighbor', False),
        device=device
    )

    # Try to load first valid scene (try more scenes to find good initialization)
    model = None
    max_attempts = min(50, len(dataset))  # Try up to 50 scenes
    print(f"  Searching for valid scene (trying up to {max_attempts} scenes)...")

    for i in range(max_attempts):
        try:
            sample = dataset[i]

            # Check if depth has valid data
            if sample['depth'].numel() == 0 or not sample['valid_mask'].any():
                continue

            model = initialize_gaussians_from_scene(sample, initializer, device)

            # Check if we got enough Gaussians
            if model.get_num_gaussians() < 100:
                print(f"  ⚠ Scene {i}: Only {model.get_num_gaussians()} Gaussians (too few)")
                continue

            print(f"✓ Initialized {model.get_num_gaussians():,} Gaussians from scene {i} ({sample.get('scene_id', 'unknown')})")
            break
        except ValueError as e:
            # Expected error for scenes with no valid depth
            if i < 5:  # Only show first few errors
                print(f"  ⚠ Scene {i}: {e}")
            continue
        except Exception as e:
            if i < 5:
                print(f"  ✗ Scene {i}: Unexpected error - {e}")
            continue

    if model is None:
        print(f"\n✗ Failed to initialize Gaussians from any of {max_attempts} scenes!")
        print("  This dataset may have depth quality issues.")
        print("  Try:")
        print("  1. Using a different sensor in config (kv1, kv2, realsense, xtion)")
        print("  2. Checking depth map validity in dataset")
        return False

    # Create trainer
    print("\nCreating trainer...")
    trainer = ProjGSTrainer(model, config, device)

    # Resume from checkpoint if specified
    if resume_path is not None:
        print(f"\nResuming from checkpoint: {resume_path}")
        trainer.load_checkpoint(resume_path)

    # Training loop
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    num_iterations = config['training']['num_iterations']
    log_interval = config['logging']['log_interval']
    image_interval = config['logging']['image_interval']
    save_interval = config['checkpointing']['save_interval']
    empty_cache_interval = config['memory']['empty_cache_interval']

    pbar = tqdm(total=num_iterations, initial=trainer.iteration, desc="Training")

    start_time = time.time()
    scene_idx = 0
    iteration_times = []  # Track iteration times

    try:
        while trainer.iteration < num_iterations:
            iter_start = time.time()
            # Get next scene
            try:
                sample = dataset[scene_idx % dataset_size]
                scene_idx += 1
            except Exception as e:
                print(f"\n⚠ Failed to load scene {scene_idx}: {e}")
                scene_idx += 1
                continue

            # Train on this scene
            try:
                metrics, outputs = trainer.train_iteration(
                    rgb_gt=sample['rgb'],
                    K=sample['K'],
                    background=None
                )

            except Exception as e:
                print(f"\n⚠ Training iteration failed: {e}")
                import traceback
                if trainer.iteration < 5:  # Print full error for first few iterations
                    traceback.print_exc()
                continue

            # Densification and pruning
            trainer.densify_and_prune()

            # Track iteration time
            iter_time = time.time() - iter_start
            iteration_times.append(iter_time)
            if len(iteration_times) > 10:
                iteration_times.pop(0)

            avg_iter_time = sum(iteration_times) / len(iteration_times)

            # Print confirmation on first iteration
            if trainer.iteration == 0:
                print(f"\n✓ First iteration completed successfully! ({iter_time:.2f}s)")
                print(f"  Loss: {metrics['loss']:.4f}, PSNR: {metrics['psnr']:.2f} dB")
                print(f"  Training will continue...\n")

            # Logging
            if trainer.iteration % log_interval == 0:
                trainer.log_metrics(metrics)

                # Update progress bar with timing info
                pbar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'psnr': f"{metrics['psnr']:.2f}",
                    'gaussians': f"{metrics['num_gaussians']:,}",
                    'it/s': f"{1.0/avg_iter_time:.2f}",
                    's/it': f"{avg_iter_time:.2f}"
                })

            # Image logging
            if trainer.iteration % image_interval == 0:
                trainer.log_images(outputs, sample['rgb'])

            # Checkpointing
            if trainer.iteration % save_interval == 0 and trainer.iteration > 0:
                trainer.save_checkpoint()

            # Memory cleanup
            if device in ['cuda', 'mps'] and trainer.iteration % empty_cache_interval == 0:
                if device == 'cuda':
                    torch.cuda.empty_cache()
                elif device == 'mps':
                    torch.mps.empty_cache()

            trainer.iteration += 1
            pbar.update(1)

    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")

    finally:
        pbar.close()

        # Save final checkpoint
        print("\nSaving final checkpoint...")
        trainer.save_checkpoint(name='final_checkpoint.pth')

        # Training summary
        elapsed = time.time() - start_time
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"Total iterations: {trainer.iteration}")
        print(f"Total time: {elapsed/3600:.2f} hours")
        print(f"Time per iteration: {elapsed/trainer.iteration:.3f} seconds")
        print(f"Final Gaussians: {model.get_num_gaussians():,}")
        print("=" * 80)

        # Cleanup
        trainer.cleanup()

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train ProjGS model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file (e.g., configs/train_mac.yaml)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"✗ Config file not found: {config_path}")
        return 1

    config = load_config(str(config_path))
    print(f"✓ Loaded config: {config_path}")

    # Train
    success = train(config, args.resume)

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
