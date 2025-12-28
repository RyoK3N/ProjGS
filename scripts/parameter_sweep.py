"""
Parameter Sweep for Optimal Gaussian Initialization
====================================================

Test different base_scale values to find optimal parameters for
maximum scale variance ratio (camera-aware vs naive).

Based on research findings:
- Standard 3DGS: Uses k-NN distance
- Recommended range: 0.01 - 1.0
- Target: Scale variance ratio > 2.0x

Usage:
    python scripts/parameter_sweep.py [--num-scenes N]

Author: ProjGS Research Team
Date: December 2025
"""

import sys
from pathlib import Path
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm
import argparse

from data.sunrgbd_loader import SUNRGBDDataset
from models.gaussian_init import CameraAwareGaussianInit


def evaluate_parameter(base_scale, dataset, num_scenes=50):
    """
    Evaluate a single base_scale parameter.

    Args:
        base_scale: Base scale value to test
        dataset: SUNRGBDDataset instance
        num_scenes: Number of scenes to evaluate

    Returns:
        Dictionary with metrics
    """
    initializer_aware = CameraAwareGaussianInit(base_scale=base_scale, uncertainty_weight=True)
    initializer_naive = CameraAwareGaussianInit(base_scale=base_scale, uncertainty_weight=False)

    scale_vars_aware = []
    scale_vars_naive = []
    scale_means_aware = []
    opacity_means_aware = []

    successful = 0

    for i in range(min(num_scenes, len(dataset))):
        sample = dataset[i]

        try:
            # Camera-aware
            g_aware = initializer_aware.initialize_from_rgbd(
                sample['rgb'], sample['depth'], sample['K']
            )
            m_aware = initializer_aware.compute_initialization_metrics(g_aware)

            # Naive
            g_naive = initializer_naive.initialize_naive(
                sample['rgb'], sample['depth']
            )
            m_naive = initializer_naive.compute_initialization_metrics(g_naive)

            scale_vars_aware.append(m_aware['scale_variance'])
            scale_vars_naive.append(m_naive['scale_variance'])
            scale_means_aware.append(m_aware['scale_mean_x'])
            opacity_means_aware.append(m_aware['opacity_mean'])

            successful += 1

        except Exception as e:
            continue

    if successful == 0:
        return None

    # Compute metrics
    mean_scale_var_aware = np.mean(scale_vars_aware)
    mean_scale_var_naive = np.mean(scale_vars_naive)
    ratio = (mean_scale_var_aware + 1e-12) / (mean_scale_var_naive + 1e-12)

    return {
        'base_scale': base_scale,
        'scale_var_aware_mean': mean_scale_var_aware,
        'scale_var_aware_std': np.std(scale_vars_aware),
        'scale_var_naive_mean': mean_scale_var_naive,
        'scale_var_naive_std': np.std(scale_vars_naive),
        'scale_var_ratio': ratio,
        'scale_mean_aware': np.mean(scale_means_aware),
        'opacity_mean_aware': np.mean(opacity_means_aware),
        'num_successful': successful,
        'num_total': num_scenes
    }


def main():
    """Run parameter sweep."""
    parser = argparse.ArgumentParser(description='Parameter sweep for base_scale optimization')
    parser.add_argument('--num-scenes', type=int, default=50,
                       help='Number of scenes per parameter (default: 50)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: fewer parameters and scenes')
    args = parser.parse_args()

    print("=" * 80)
    print("PARAMETER SWEEP: base_scale Optimization")
    print("=" * 80)

    # Load dataset
    dataset_root = project_root / 'data/sunrgbd'
    print("\nLoading dataset...")
    dataset = SUNRGBDDataset(
        root_dir=str(dataset_root),
        split='val',
        sensors=['kv1', 'kv2', 'realsense', 'xtion'],
        resize=(480, 640)
    )

    print(f"✓ Loaded {len(dataset)} scenes")

    # Define parameter values to test
    # Based on research: 0.01-1.0 range
    if args.quick:
        base_scales = [0.01, 0.05, 0.1, 0.5]
        num_scenes = min(20, args.num_scenes)
    else:
        base_scales = [0.01, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0]
        num_scenes = args.num_scenes

    print(f"\nTesting {len(base_scales)} parameter values on {num_scenes} scenes each")
    print(f"Parameters: {base_scales}")

    # Run sweep
    results = []

    for base_scale in tqdm(base_scales, desc="Parameter sweep"):
        result = evaluate_parameter(base_scale, dataset, num_scenes)
        if result is not None:
            results.append(result)

            # Print interim result
            tqdm.write(f"\nbase_scale={base_scale:.3f}:")
            tqdm.write(f"  Scale var ratio: {result['scale_var_ratio']:.4f}x "
                      f"({'PASS' if result['scale_var_ratio'] > 2.0 else 'FAIL'})")
            tqdm.write(f"  Scale var (aware): {result['scale_var_aware_mean']:.8f}")
            tqdm.write(f"  Scale mean (aware): {result['scale_mean_aware']:.6f}")

    if len(results) == 0:
        print("\n✗ No successful evaluations")
        return False

    # Find optimal parameter
    best_result = max(results, key=lambda x: x['scale_var_ratio'])

    print("\n" + "=" * 80)
    print("PARAMETER SWEEP RESULTS")
    print("=" * 80)

    print(f"\nBest parameter: base_scale = {best_result['base_scale']:.3f}")
    print(f"  Scale variance ratio: {best_result['scale_var_ratio']:.4f}x")
    print(f"  Scale variance (aware): {best_result['scale_var_aware_mean']:.8f}")
    print(f"  Scale variance (naive): {best_result['scale_var_naive_mean']:.8f}")
    print(f"  Scale mean (aware): {best_result['scale_mean_aware']:.6f}")
    print(f"  Opacity mean (aware): {best_result['opacity_mean_aware']:.4f}")

    # Generate visualization
    output_dir = project_root / 'outputs'
    output_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Extract data
    scales = [r['base_scale'] for r in results]
    ratios = [r['scale_var_ratio'] for r in results]
    vars_aware = [r['scale_var_aware_mean'] for r in results]
    vars_naive = [r['scale_var_naive_mean'] for r in results]
    scale_means = [r['scale_mean_aware'] for r in results]

    # Plot 1: Scale variance ratio
    axes[0, 0].plot(scales, ratios, 'o-', linewidth=2, markersize=8)
    axes[0, 0].axhline(y=2.0, color='r', linestyle='--', label='Target (2.0x)')
    axes[0, 0].axvline(x=best_result['base_scale'], color='g', linestyle='--',
                       label=f'Best ({best_result["base_scale"]:.3f})')
    axes[0, 0].set_xlabel('base_scale')
    axes[0, 0].set_ylabel('Scale Variance Ratio (aware/naive)')
    axes[0, 0].set_title('Objective: Scale Variance Ratio')
    axes[0, 0].set_xscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Plot 2: Scale variances
    axes[0, 1].plot(scales, vars_aware, 'o-', label='Camera-Aware', linewidth=2, markersize=8)
    axes[0, 1].plot(scales, vars_naive, 's-', label='Naive', linewidth=2, markersize=8)
    axes[0, 1].axvline(x=best_result['base_scale'], color='g', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('base_scale')
    axes[0, 1].set_ylabel('Scale Variance')
    axes[0, 1].set_title('Scale Variance by Method')
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Plot 3: Scale means
    axes[1, 0].plot(scales, scale_means, 'o-', linewidth=2, markersize=8, color='purple')
    axes[1, 0].axvline(x=best_result['base_scale'], color='g', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('base_scale')
    axes[1, 0].set_ylabel('Mean Scale (X-axis)')
    axes[1, 0].set_title('Average Gaussian Scale')
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Results table
    axes[1, 1].axis('off')

    # Create table
    table_data = [
        ['base_scale', 'Ratio', 'Var (aware)', 'Status'],
        ['-' * 12, '-' * 8, '-' * 12, '-' * 8]
    ]

    for r in sorted(results, key=lambda x: x['scale_var_ratio'], reverse=True)[:10]:
        status = '✓ PASS' if r['scale_var_ratio'] > 2.0 else '✗ FAIL'
        table_data.append([
            f"{r['base_scale']:.3f}",
            f"{r['scale_var_ratio']:.3f}x",
            f"{r['scale_var_aware_mean']:.6f}",
            status
        ])

    table_text = '\n'.join([
        f"{row[0]:<12} {row[1]:<8} {row[2]:<12} {row[3]:<8}"
        for row in table_data
    ])

    title = "Top 10 Parameters by Ratio\n" + "=" * 45 + "\n"
    axes[1, 1].text(0.1, 0.9, title + table_text, fontsize=9, family='monospace',
                   verticalalignment='top', transform=axes[1, 1].transAxes)

    plt.suptitle(f'Parameter Sweep Results (n={num_scenes} scenes)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save
    save_path = output_dir / 'parameter_sweep_results.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization: {save_path}")

    # Save JSON
    json_path = output_dir / 'parameter_sweep_results.json'
    with open(json_path, 'w') as f:
        json.dump({
            'best_parameter': best_result['base_scale'],
            'best_ratio': best_result['scale_var_ratio'],
            'all_results': results,
            'num_scenes': num_scenes
        }, f, indent=2)
    print(f"✓ Saved results: {json_path}")

    print("\n" + "=" * 80)
    print("✓ Parameter sweep complete!")
    print("=" * 80)

    # Recommendation
    if best_result['scale_var_ratio'] > 2.0:
        print(f"\n✅ RECOMMENDATION: Use base_scale={best_result['base_scale']:.3f}")
        print(f"   This achieves {best_result['scale_var_ratio']:.2f}x scale variance improvement")
    else:
        print(f"\n⚠️ WARNING: Best parameter ({best_result['base_scale']:.3f}) achieves only "
              f"{best_result['scale_var_ratio']:.2f}x ratio")
        print("   Consider:")
        print("   1. Enabling k-NN adaptive scaling (use_nearest_neighbor=True)")
        print("   2. Testing larger base_scale values (1.0 - 5.0)")
        print("   3. Checking depth quality in dataset")

    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
