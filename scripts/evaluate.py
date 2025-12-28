"""
ProjGS Evaluation Script
========================

Evaluate trained Gaussian Splatting model on validation set.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/mac/final_checkpoint.pth --config configs/train_mac.yaml

Author: ProjGS Research Team
Date: December 2025
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import yaml
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from data.sunrgbd_loader import SUNRGBDDataset
from models.gaussian_model import GaussianModel
from models.rasterizer_optimized import TileBasedGaussianRasterizer
from models.losses import compute_psnr, compute_ssim


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model_from_checkpoint(checkpoint_path: str, device: str) -> GaussianModel:
    """
    Load GaussianModel from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load on

    Returns:
        Loaded GaussianModel
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model state
    model_state = checkpoint['model_state']

    # Create model
    model = GaussianModel(
        positions=model_state['positions'],
        features=model_state['features'],
        opacities=model_state['opacities'],
        scales=model_state['scales'],
        rotations=model_state['rotations'],
        device=device
    )

    # Load full state
    model.load_state_dict(model_state)

    return model


def evaluate_scene(
    model: GaussianModel,
    rasterizer: GaussianRasterizer,
    sample: dict,
    device: str
) -> dict:
    """
    Evaluate model on a single scene.

    Args:
        model: GaussianModel
        rasterizer: Rasterizer
        sample: Dataset sample
        device: Device

    Returns:
        Dictionary with metrics and rendered outputs
    """
    with torch.no_grad():
        # Get ground truth
        rgb_gt = sample['rgb'].to(device)
        if rgb_gt.dim() == 4:
            rgb_gt = rgb_gt.squeeze(0)

        K = sample['K'].to(device)

        # Render
        outputs = rasterizer(
            positions=model.positions,
            scales=model.scales,
            rotations=model.rotations,
            colors=model.features,
            opacities=model.opacities,
            K=K,
            background=None
        )

        rgb_pred = outputs['image']

        # Compute metrics
        rgb_pred_batch = rgb_pred.unsqueeze(0)
        rgb_gt_batch = rgb_gt.unsqueeze(0)

        psnr = compute_psnr(rgb_pred_batch, rgb_gt_batch)
        ssim = compute_ssim(rgb_pred_batch, rgb_gt_batch)

        # L1 error
        l1_error = torch.abs(rgb_pred - rgb_gt).mean().item()

    return {
        'psnr': psnr,
        'ssim': ssim,
        'l1_error': l1_error,
        'rgb_pred': rgb_pred.cpu().numpy(),
        'rgb_gt': rgb_gt.cpu().numpy(),
        'depth': outputs['depth'].cpu().numpy(),
        'alpha': outputs['alpha'].cpu().numpy()
    }


def visualize_results(results: list, output_dir: Path):
    """
    Create visualization of evaluation results.

    Args:
        results: List of evaluation results
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute statistics
    psnrs = [r['psnr'] for r in results]
    ssims = [r['ssim'] for r in results]
    l1_errors = [r['l1_error'] for r in results]

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Number of scenes: {len(results)}")
    print(f"\nPSNR:")
    print(f"  Mean: {np.mean(psnrs):.2f} dB")
    print(f"  Std:  {np.std(psnrs):.2f} dB")
    print(f"  Min:  {np.min(psnrs):.2f} dB")
    print(f"  Max:  {np.max(psnrs):.2f} dB")
    print(f"\nSSIM:")
    print(f"  Mean: {np.mean(ssims):.4f}")
    print(f"  Std:  {np.std(ssims):.4f}")
    print(f"\nL1 Error:")
    print(f"  Mean: {np.mean(l1_errors):.4f}")
    print("=" * 80)

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # PSNR distribution
    axes[0, 0].hist(psnrs, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].axvline(np.mean(psnrs), color='red', linestyle='--', label=f'Mean: {np.mean(psnrs):.2f}')
    axes[0, 0].set_xlabel('PSNR (dB)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('PSNR Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # SSIM distribution
    axes[0, 1].hist(ssims, bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].axvline(np.mean(ssims), color='red', linestyle='--', label=f'Mean: {np.mean(ssims):.4f}')
    axes[0, 1].set_xlabel('SSIM')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('SSIM Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # PSNR vs SSIM scatter
    axes[1, 0].scatter(ssims, psnrs, alpha=0.6, color='purple')
    axes[1, 0].set_xlabel('SSIM')
    axes[1, 0].set_ylabel('PSNR (dB)')
    axes[1, 0].set_title('PSNR vs SSIM')
    axes[1, 0].grid(True, alpha=0.3)

    # Example visualization (first scene)
    axes[1, 1].axis('off')
    if len(results) > 0:
        example = results[0]
        # Create side-by-side comparison
        comparison = np.concatenate([example['rgb_gt'], example['rgb_pred']], axis=1)
        axes[1, 1].imshow(comparison)
        axes[1, 1].set_title(f'Example: GT (left) vs Pred (right)\nPSNR: {example["psnr"]:.2f} dB, SSIM: {example["ssim"]:.4f}')

    plt.tight_layout()
    save_path = output_dir / 'evaluation_results.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization: {save_path}")

    # Save individual scene comparisons
    print("\nSaving individual scene comparisons...")
    scenes_dir = output_dir / 'scenes'
    scenes_dir.mkdir(exist_ok=True)

    for i, result in enumerate(tqdm(results[:10], desc="Saving scenes")):  # Save first 10
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Ground truth
        axes[0, 0].imshow(result['rgb_gt'])
        axes[0, 0].set_title('Ground Truth')
        axes[0, 0].axis('off')

        # Prediction
        axes[0, 1].imshow(result['rgb_pred'])
        axes[0, 1].set_title(f'Prediction\nPSNR: {result["psnr"]:.2f} dB, SSIM: {result["ssim"]:.4f}')
        axes[0, 1].axis('off')

        # Depth
        depth = result['depth']
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        axes[1, 0].imshow(depth_norm, cmap='turbo')
        axes[1, 0].set_title('Rendered Depth')
        axes[1, 0].axis('off')

        # Error map
        error = np.abs(result['rgb_gt'] - result['rgb_pred']).mean(axis=2)
        im = axes[1, 1].imshow(error, cmap='hot')
        axes[1, 1].set_title(f'Absolute Error (mean: {result["l1_error"]:.4f})')
        axes[1, 1].axis('off')
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

        plt.tight_layout()
        plt.savefig(scenes_dir / f'scene_{i:03d}.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"✓ Saved {min(10, len(results))} scene comparisons to {scenes_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Evaluate ProjGS model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--output-dir', type=str, default='outputs/evaluation',
                       help='Output directory for results')
    parser.add_argument('--num-scenes', type=int, default=None,
                       help='Number of validation scenes to evaluate (default: all)')

    args = parser.parse_args()

    print("=" * 80)
    print("PROJECTION-GUIDED GAUSSIAN SPLATTING - EVALUATION")
    print("=" * 80)

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"✗ Config file not found: {config_path}")
        return 1

    config = load_config(str(config_path))
    print(f"✓ Loaded config: {config_path}")

    # Setup device
    device_name = config['device']
    if device_name == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    elif device_name == 'mps' and not torch.backends.mps.is_available():
        device = 'cpu'
    else:
        device = device_name

    print(f"Device: {device}")

    # Load model
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return 1

    print(f"\nLoading model from: {checkpoint_path}")
    model = load_model_from_checkpoint(str(checkpoint_path), device)
    print(f"✓ Loaded model with {model.get_num_gaussians():,} Gaussians")

    # Create memory-optimized rasterizer
    image_size = tuple(config['rendering']['resolution'])
    rasterizer = TileBasedGaussianRasterizer(
        image_size=image_size,
        tile_size=16,
        device=device
    )

    # Load validation dataset
    print("\nLoading validation dataset...")
    dataset_config = config['dataset']
    dataset = SUNRGBDDataset(
        root_dir=dataset_config['root_dir'],
        split='val',  # Use validation split
        sensors=dataset_config['sensors'],
        resize=tuple(dataset_config['resize'])
    )

    num_scenes = args.num_scenes if args.num_scenes is not None else len(dataset)
    num_scenes = min(num_scenes, len(dataset))
    print(f"✓ Loaded {len(dataset)} validation scenes (evaluating {num_scenes})")

    # Evaluate
    print("\nEvaluating...")
    results = []

    for i in tqdm(range(num_scenes), desc="Evaluating"):
        try:
            sample = dataset[i]
            result = evaluate_scene(model, rasterizer, sample, device)
            results.append(result)
        except Exception as e:
            print(f"\n⚠ Failed to evaluate scene {i}: {e}")
            continue

    if len(results) == 0:
        print("\n✗ No scenes evaluated successfully!")
        return 1

    # Visualize results
    output_dir = Path(args.output_dir)
    visualize_results(results, output_dir)

    print("\n✓ Evaluation complete!")

    return 0


if __name__ == '__main__':
    sys.exit(main())
