"""
Initialization Comparison Script
=================================

Compare camera-aware vs naive Gaussian initialization on sample scenes.

Generates:
1. Visual comparison of scale distributions
2. Quantitative metrics table
3. Saved visualizations for analysis

Usage:
    python scripts/compare_initialization.py

Author: ProjGS Research Team
Date: December 2025
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from data.sunrgbd_loader import SUNRGBDDataset
from models.gaussian_init import CameraAwareGaussianInit, compare_initialization_methods


def visualize_comparison(
    results: dict,
    sample: dict,
    save_path: Path
):
    """
    Create comprehensive visualization comparing two initialization methods.

    Args:
        results: Dictionary with 'camera_aware' and 'naive' results
        sample: Original dataset sample
        save_path: Where to save visualization
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # === Row 1: Input Data ===
    # RGB
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(sample['rgb'].cpu().numpy())
    ax1.set_title('RGB Image', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Depth
    ax2 = fig.add_subplot(gs[0, 1])
    depth_vis = sample['depth'].cpu().numpy()
    valid_mask = sample['valid_mask'].cpu().numpy()
    depth_vis[~valid_mask] = 0
    im = ax2.imshow(depth_vis, cmap='turbo', vmin=0, vmax=5)
    ax2.set_title('Depth Map', fontsize=14, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046)

    # Camera Info
    ax3 = fig.add_subplot(gs[0, 2:4])
    K = sample['K'].cpu().numpy()
    info_text = (
        f"Dataset Sample Information\n"
        f"{'=' * 50}\n\n"
        f"Scene ID: {sample['scene_id']}\n"
        f"Category: {sample['scene_category']}\n"
        f"Sensor: {sample['sensor']}\n\n"
        f"Camera Intrinsics:\n"
        f"  fx = {K[0,0]:8.2f} px\n"
        f"  fy = {K[1,1]:8.2f} px\n"
        f"  cx = {K[0,2]:8.2f} px\n"
        f"  cy = {K[1,2]:8.2f} px\n\n"
        f"Depth Statistics:\n"
        f"  Valid pixels: {valid_mask.sum():,}\n"
        f"  Coverage: {100 * valid_mask.sum() / valid_mask.size:.1f}%\n"
        f"  Range: [{depth_vis[valid_mask].min():.2f}, {depth_vis[valid_mask].max():.2f}] m"
    )
    ax3.text(0.05, 0.5, info_text, fontsize=11, family='monospace',
             verticalalignment='center', transform=ax3.transAxes)
    ax3.axis('off')

    # === Row 2: Scale Distributions ===
    methods = ['camera_aware', 'naive']
    colors_map = {'camera_aware': 'blue', 'naive': 'red'}
    labels_map = {'camera_aware': 'Camera-Aware', 'naive': 'Naive'}

    for i, axis_name in enumerate(['X', 'Y', 'Z']):
        ax = fig.add_subplot(gs[1, i])

        for method in methods:
            scales = results[method]['gaussians']['scales'][:, i].cpu().numpy()

            ax.hist(scales, bins=50, alpha=0.6,
                   label=labels_map[method],
                   color=colors_map[method],
                   density=True)

        ax.set_xlabel(f'Scale {axis_name} (meters)', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{axis_name}-Axis Scale Distribution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    # === Metrics Comparison Table ===
    ax4 = fig.add_subplot(gs[1, 3])

    metrics_aware = results['camera_aware']['metrics']
    metrics_naive = results['naive']['metrics']

    table_data = [
        ['Metric', 'Camera-Aware', 'Naive', 'Ratio'],
        ['─' * 20, '─' * 15, '─' * 15, '─' * 10],
        ['Gaussians', f"{metrics_aware['count']:,}", f"{metrics_naive['count']:,}", '1.00x'],
        ['', '', '', ''],
        ['Scale Mean X', f"{metrics_aware['scale_mean_x']:.6f}", f"{metrics_naive['scale_mean_x']:.6f}",
         f"{metrics_aware['scale_mean_x'] / (metrics_naive['scale_mean_x'] + 1e-8):.2f}x"],
        ['Scale Std X', f"{metrics_aware['scale_std_x']:.6f}", f"{metrics_naive['scale_std_x']:.6f}",
         f"{(metrics_aware['scale_std_x'] + 1e-8) / (metrics_naive['scale_std_x'] + 1e-8):.2f}x"],
        ['', '', '', ''],
        ['Scale Variance', f"{metrics_aware['scale_variance']:.8f}", f"{metrics_naive['scale_variance']:.8f}",
         f"{(metrics_aware['scale_variance'] + 1e-8) / (metrics_naive['scale_variance'] + 1e-8):.2f}x"],
        ['', '', '', ''],
        ['Opacity Mean', f"{metrics_aware['opacity_mean']:.4f}", f"{metrics_naive['opacity_mean']:.4f}",
         f"{metrics_aware['opacity_mean'] / metrics_naive['opacity_mean']:.2f}x"],
        ['Opacity Std', f"{metrics_aware['opacity_std']:.4f}", f"{metrics_naive['opacity_std']:.4f}",
         f"{(metrics_aware['opacity_std'] + 1e-8) / (metrics_naive['opacity_std'] + 1e-8):.2f}x"],
    ]

    ax4.axis('off')
    table_text = '\n'.join([f"{row[0]:<20} {row[1]:>15} {row[2]:>15} {row[3]:>10}" for row in table_data])
    ax4.text(0.05, 0.5, table_text, fontsize=9, family='monospace',
             verticalalignment='center', transform=ax4.transAxes)

    # === Row 3: Opacity Distributions ===
    ax5 = fig.add_subplot(gs[2, 0:2])
    for method in methods:
        opacities = results[method]['gaussians']['opacities'].cpu().numpy().flatten()
        ax5.hist(opacities, bins=50, alpha=0.6,
                label=labels_map[method],
                color=colors_map[method],
                density=True)

    ax5.set_xlabel('Opacity', fontsize=11)
    ax5.set_ylabel('Density', fontsize=11)
    ax5.set_title('Opacity Distribution', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)

    # === Key Findings ===
    ax6 = fig.add_subplot(gs[2, 2:4])

    scale_var_ratio = (metrics_aware['scale_variance'] + 1e-8) / (metrics_naive['scale_variance'] + 1e-8)
    opacity_std_ratio = (metrics_aware['opacity_std'] + 1e-8) / (metrics_naive['opacity_std'] + 1e-8)

    findings = (
        "Key Findings\n"
        f"{'=' * 60}\n\n"
        f"1. Scale Adaptivity:\n"
        f"   Camera-aware has {scale_var_ratio:.2f}x MORE scale variance\n"
        f"   {'✓' if scale_var_ratio > 2.0 else '✗'} {'PASSES' if scale_var_ratio > 2.0 else 'FAILS'} "
        f"threshold (>2x)\n\n"
        f"2. Opacity Variation:\n"
        f"   Camera-aware has {opacity_std_ratio:.2f}x MORE opacity variance\n"
        f"   {'✓' if opacity_std_ratio > 1.0 else '✗'} Uncertainty weighting active\n\n"
        f"3. Gaussian Count:\n"
        f"   Both methods: {metrics_aware['count']:,} Gaussians\n\n"
        f"4. Position Validity:\n"
        f"   Camera-aware mean Z: {metrics_aware['position_mean_z']:.2f} m\n"
        f"   Naive mean Z: {metrics_naive['position_mean_z']:.2f} m\n\n"
        f"Conclusion:\n"
        f"  Camera-aware initialization shows {'SIGNIFICANT' if scale_var_ratio > 2.0 else 'MARGINAL'}\n"
        f"  improvement in scale adaptivity compared to naive approach."
    )

    ax6.text(0.05, 0.5, findings, fontsize=10, family='monospace',
             verticalalignment='center', transform=ax6.transAxes)
    ax6.axis('off')

    # Save
    plt.suptitle(f'Initialization Comparison - {sample["scene_id"]} ({sample["sensor"]})',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved visualization: {save_path}")


def main():
    """Run initialization comparison."""
    print("=" * 80)
    print("Gaussian Initialization Comparison: Camera-Aware vs Naive")
    print("=" * 80)

    # Check if dataset exists
    dataset_root = project_root / 'data/sunrgbd'
    if not dataset_root.exists():
        print(f"\n✗ Dataset not found at {dataset_root}")
        print("Please download SUN RGB-D dataset first.")
        return False

    # Load dataset
    print("\nLoading SUN RGB-D dataset...")
    dataset = SUNRGBDDataset(
        root_dir=str(dataset_root),
        split='train',
        sensors=['kv1', 'kv2', 'realsense', 'xtion'],  # All sensors
        resize=(480, 640)
    )

    if len(dataset) == 0:
        print("✗ No valid scenes found in dataset")
        return False

    print(f"✓ Loaded {len(dataset)} scenes")

    # Create output directory
    output_dir = project_root / 'outputs'
    output_dir.mkdir(exist_ok=True)

    # Test on more scenes for better statistics
    num_samples = min(10, len(dataset))
    print(f"\nComparing initialization on {num_samples} samples...")
    print("-" * 80)

    successful_comparisons = 0
    for i in range(num_samples):
        sample = dataset[i]

        print(f"\nSample {i + 1}/{num_samples}: {sample['scene_id']}")

        try:
            # Run comparison
            results = compare_initialization_methods(
                rgb=sample['rgb'],
                depth=sample['depth'],
                K=sample['K'],
                max_depth=10.0
            )
        except ValueError as e:
            print(f"  ⚠ Skipping scene (no valid depth): {e}")
            continue
        except Exception as e:
            print(f"  ✗ Error processing scene: {e}")
            continue

        # Print metrics
        print("\nMetrics:")
        for method in ['camera_aware', 'naive']:
            metrics = results[method]['metrics']
            print(f"  {method:15s}: {metrics['count']:6d} Gaussians, "
                  f"scale_var={metrics['scale_variance']:.6f}, "
                  f"opacity_mean={metrics['opacity_mean']:.4f}")

        # Calculate key ratio
        scale_var_ratio = (results['camera_aware']['metrics']['scale_variance'] + 1e-8) / \
                         (results['naive']['metrics']['scale_variance'] + 1e-8)
        print(f"  Scale variance ratio: {scale_var_ratio:.2f}x "
              f"({'PASS' if scale_var_ratio > 2.0 else 'FAIL'} >2x threshold)")

        # Visualize
        save_path = output_dir / f'init_comparison_sample_{successful_comparisons:02d}_{sample["scene_id"]}.png'
        visualize_comparison(results, sample, save_path)

        successful_comparisons += 1

    print("\n" + "=" * 80)
    print(f"✓ Initialization comparison complete! ({successful_comparisons}/{num_samples} successful)")
    print(f"✓ Results saved to {output_dir}/")
    print("=" * 80)

    return successful_comparisons > 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
