"""
Multi-Scene Evaluation Script
==============================

Evaluate initialization quality across multiple scenes from different sensors.

This script provides quantitative evidence for the feasibility study by:
1. Testing on 50+ scenes across all available sensors
2. Computing aggregate statistics for camera-aware vs naive methods
3. Analyzing cross-sensor consistency
4. Generating comprehensive evaluation report

Usage:
    python scripts/multi_scene_evaluation.py [--num-scenes N] [--sensors kv1,kv2,...]

Author: ProjGS Research Team
Date: December 2025
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm

from data.sunrgbd_loader import SUNRGBDDataset
from models.gaussian_init import CameraAwareGaussianInit


def evaluate_scene(
    sample: dict,
    initializer: CameraAwareGaussianInit
) -> Dict[str, Dict]:
    """
    Evaluate both initialization methods on a single scene.

    Args:
        sample: Dataset sample
        initializer: Gaussian initializer instance

    Returns:
        Dictionary with 'camera_aware' and 'naive' metrics
    """
    try:
        # Camera-aware
        gaussians_aware = initializer.initialize_from_rgbd(
            sample['rgb'],
            sample['depth'],
            sample['K']
        )
        metrics_aware = initializer.compute_initialization_metrics(gaussians_aware)

        # Naive
        gaussians_naive = initializer.initialize_naive(
            sample['rgb'],
            sample['depth']
        )
        metrics_naive = initializer.compute_initialization_metrics(gaussians_naive)

        return {
            'camera_aware': metrics_aware,
            'naive': metrics_naive,
            'scene_id': sample['scene_id'],
            'sensor': sample['sensor'],
            'category': sample['scene_category']
        }

    except Exception as e:
        print(f"  Warning: Failed to process scene {sample.get('scene_id', 'unknown')}: {e}")
        return None


def compute_aggregate_statistics(
    results: List[Dict]
) -> Dict:
    """
    Compute aggregate statistics across all scenes.

    Args:
        results: List of per-scene evaluation results

    Returns:
        Dictionary of aggregate metrics
    """
    # Collect metrics for each method
    metrics_by_method = {'camera_aware': [], 'naive': []}

    for result in results:
        if result is None:
            continue

        for method in ['camera_aware', 'naive']:
            metrics_by_method[method].append(result[method])

    # Compute statistics
    stats = {}

    for method in ['camera_aware', 'naive']:
        method_stats = {}
        metrics_list = metrics_by_method[method]

        if len(metrics_list) == 0:
            continue

        # Aggregate each metric
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list]
            method_stats[f'{key}_mean'] = np.mean(values)
            method_stats[f'{key}_std'] = np.std(values)
            method_stats[f'{key}_min'] = np.min(values)
            method_stats[f'{key}_max'] = np.max(values)
            method_stats[f'{key}_median'] = np.median(values)

        stats[method] = method_stats

    # Compute ratios
    stats['ratios'] = {}
    if 'camera_aware' in stats and 'naive' in stats:
        for key in ['scale_variance', 'opacity_std', 'count']:
            aware_key = f'{key}_mean'
            if aware_key in stats['camera_aware'] and aware_key in stats['naive']:
                aware_val = stats['camera_aware'][aware_key]
                naive_val = stats['naive'][aware_key]
                ratio = (aware_val + 1e-8) / (naive_val + 1e-8)
                stats['ratios'][key] = ratio

    return stats


def generate_evaluation_report(
    stats: Dict,
    results: List[Dict],
    output_path: Path
):
    """
    Generate comprehensive evaluation report with visualizations.

    Args:
        stats: Aggregate statistics
        results: Per-scene results
        output_path: Where to save report
    """
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)

    # === Summary Statistics Table ===
    ax_summary = fig.add_subplot(gs[0, :])
    ax_summary.axis('off')

    summary_text = (
        "MULTI-SCENE EVALUATION SUMMARY\n"
        f"{'=' * 100}\n\n"
        f"Total Scenes Evaluated: {len([r for r in results if r is not None])}\n"
        f"Sensors: {', '.join(set(r['sensor'] for r in results if r is not None))}\n\n"
        f"{'Metric':<30} {'Camera-Aware Mean':<20} {'Naive Mean':<20} {'Ratio':<15} {'Result':<10}\n"
        f"{'-' * 100}\n"
    )

    # Key metrics
    key_metrics = [
        ('Gaussian Count', 'count'),
        ('Scale Variance', 'scale_variance'),
        ('Scale Mean X', 'scale_mean_x'),
        ('Scale Std X', 'scale_std_x'),
        ('Opacity Mean', 'opacity_mean'),
        ('Opacity Std', 'opacity_std'),
    ]

    for metric_name, metric_key in key_metrics:
        aware_mean = stats['camera_aware'].get(f'{metric_key}_mean', 0)
        naive_mean = stats['naive'].get(f'{metric_key}_mean', 0)
        ratio = stats['ratios'].get(metric_key, 1.0)

        # Determine pass/fail for critical metrics
        if metric_key == 'scale_variance':
            result = '✓ PASS' if ratio > 2.0 else '✗ FAIL'
        elif metric_key == 'opacity_std':
            result = '✓ PASS' if ratio > 1.0 else '✗ FAIL'
        else:
            result = ''

        summary_text += f"{metric_name:<30} {aware_mean:<20.6f} {naive_mean:<20.6f} {ratio:<15.2f}x {result:<10}\n"

    summary_text += f"\n{'-' * 100}\n"
    summary_text += f"FEASIBILITY ASSESSMENT:\n"

    scale_var_ratio = stats['ratios'].get('scale_variance', 0)
    if scale_var_ratio > 3.0:
        summary_text += f"  Scale Variance Ratio: {scale_var_ratio:.2f}x - STRONG SUCCESS (>3x threshold)\n"
    elif scale_var_ratio > 2.0:
        summary_text += f"  Scale Variance Ratio: {scale_var_ratio:.2f}x - SUCCESS (>2x threshold)\n"
    elif scale_var_ratio > 1.5:
        summary_text += f"  Scale Variance Ratio: {scale_var_ratio:.2f}x - MARGINAL (>1.5x threshold)\n"
    else:
        summary_text += f"  Scale Variance Ratio: {scale_var_ratio:.2f}x - INSUFFICIENT (<1.5x threshold)\n"

    ax_summary.text(0.05, 0.5, summary_text, fontsize=9, family='monospace',
                   verticalalignment='center', transform=ax_summary.transAxes)

    # === Scale Variance Distribution ===
    ax1 = fig.add_subplot(gs[1, 0])
    scale_vars_aware = [r['camera_aware']['scale_variance'] for r in results if r]
    scale_vars_naive = [r['naive']['scale_variance'] for r in results if r]

    ax1.hist([scale_vars_aware, scale_vars_naive], bins=30, alpha=0.6,
            label=['Camera-Aware', 'Naive'], color=['blue', 'red'])
    ax1.set_xlabel('Scale Variance')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Scale Variance Distribution Across Scenes')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # === Opacity Statistics ===
    ax2 = fig.add_subplot(gs[1, 1])
    opacity_means_aware = [r['camera_aware']['opacity_mean'] for r in results if r]
    opacity_means_naive = [r['naive']['opacity_mean'] for r in results if r]

    ax2.hist([opacity_means_aware, opacity_means_naive], bins=30, alpha=0.6,
            label=['Camera-Aware', 'Naive'], color=['blue', 'red'])
    ax2.set_xlabel('Opacity Mean')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Opacity Mean Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # === Gaussian Count ===
    ax3 = fig.add_subplot(gs[1, 2])
    counts_aware = [r['camera_aware']['count'] for r in results if r]
    counts_naive = [r['naive']['count'] for r in results if r]

    ax3.scatter(range(len(counts_aware)), counts_aware, alpha=0.6, label='Camera-Aware', color='blue')
    ax3.scatter(range(len(counts_naive)), counts_naive, alpha=0.6, label='Naive', color='red')
    ax3.set_xlabel('Scene Index')
    ax3.set_ylabel('Gaussian Count')
    ax3.set_title('Gaussian Count per Scene')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # === Per-Sensor Breakdown ===
    ax4 = fig.add_subplot(gs[2, :])

    # Group by sensor
    sensor_stats = {}
    for result in results:
        if result is None:
            continue

        sensor = result['sensor']
        if sensor not in sensor_stats:
            sensor_stats[sensor] = {'camera_aware': [], 'naive': []}

        sensor_stats[sensor]['camera_aware'].append(result['camera_aware']['scale_variance'])
        sensor_stats[sensor]['naive'].append(result['naive']['scale_variance'])

    # Plot sensor comparison
    sensors = list(sensor_stats.keys())
    x = np.arange(len(sensors))
    width = 0.35

    aware_means = [np.mean(sensor_stats[s]['camera_aware']) for s in sensors]
    naive_means = [np.mean(sensor_stats[s]['naive']) for s in sensors]

    bars1 = ax4.bar(x - width/2, aware_means, width, label='Camera-Aware', alpha=0.7, color='blue')
    bars2 = ax4.bar(x + width/2, naive_means, width, label='Naive', alpha=0.7, color='red')

    ax4.set_xlabel('Sensor')
    ax4.set_ylabel('Mean Scale Variance')
    ax4.set_title('Scale Variance by Sensor Type')
    ax4.set_xticks(x)
    ax4.set_xticklabels(sensors)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.6f}', ha='center', va='bottom', fontsize=8)

    # === Scene-by-Scene Comparison ===
    ax5 = fig.add_subplot(gs[3, :2])

    scene_ids = [r['scene_id'][:10] if r else '' for r in results[:20]]  # First 20 scenes
    scale_vars_aware_subset = [r['camera_aware']['scale_variance'] for r in results[:20] if r]
    scale_vars_naive_subset = [r['naive']['scale_variance'] for r in results[:20] if r]

    x_pos = np.arange(len(scene_ids))
    ax5.plot(x_pos, scale_vars_aware_subset, 'o-', label='Camera-Aware', color='blue', alpha=0.7)
    ax5.plot(x_pos, scale_vars_naive_subset, 's-', label='Naive', color='red', alpha=0.7)

    ax5.set_xlabel('Scene')
    ax5.set_ylabel('Scale Variance')
    ax5.set_title('Scene-by-Scene Scale Variance (First 20 Scenes)')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(scene_ids, rotation=45, ha='right', fontsize=8)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # === Decision Matrix ===
    ax6 = fig.add_subplot(gs[3, 2])
    ax6.axis('off')

    scale_var_ratio = stats['ratios'].get('scale_variance', 0)

    decision_text = (
        "GO/NO-GO DECISION MATRIX\n"
        f"{'=' * 40}\n\n"
        f"Criterion 1: Scale Variance Ratio\n"
        f"  Required: >2.0x\n"
        f"  Actual: {scale_var_ratio:.2f}x\n"
        f"  Status: {'✓ PASS' if scale_var_ratio > 2.0 else '✗ FAIL'}\n\n"
        f"Criterion 2: Consistency\n"
        f"  Std/Mean: {stats['camera_aware']['scale_variance_std'] / stats['camera_aware']['scale_variance_mean']:.3f}\n"
        f"  Status: {'✓ PASS' if stats['camera_aware']['scale_variance_std'] / stats['camera_aware']['scale_variance_mean'] < 0.5 else '⚠ MARGINAL'}\n\n"
    )

    # Final decision
    if scale_var_ratio > 2.0:
        decision_text += f"\nFINAL DECISION: ✓ GO\n"
        decision_text += f"Camera-aware initialization shows\n"
        decision_text += f"significant improvement. Proceed to\n"
        decision_text += f"full ProjGS development.\n"
    elif scale_var_ratio > 1.5:
        decision_text += f"\nFINAL DECISION: ⚠ CAUTION\n"
        decision_text += f"Marginal improvement. Consider\n"
        decision_text += f"refinement before full development.\n"
    else:
        decision_text += f"\nFINAL DECISION: ✗ NO-GO\n"
        decision_text += f"Insufficient improvement. Pivot to\n"
        decision_text += f"alternative research direction.\n"

    ax6.text(0.05, 0.5, decision_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax6.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('Multi-Scene Evaluation Report', fontsize=16, fontweight='bold')

    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Generated evaluation report: {output_path}")


def main():
    """Run multi-scene evaluation."""
    parser = argparse.ArgumentParser(description='Multi-scene initialization evaluation')
    parser.add_argument('--num-scenes', type=int, default=100,
                       help='Number of scenes to evaluate (default: 100, use -1 for all)')
    parser.add_argument('--sensors', type=str, default='kv1,kv2,realsense,xtion',
                       help='Comma-separated list of sensors (default: all)')
    parser.add_argument('--base-scale', type=float, default=0.1,
                       help='Base scale parameter (default: 0.1)')
    args = parser.parse_args()

    print("=" * 80)
    print("Multi-Scene Gaussian Initialization Evaluation")
    print("=" * 80)

    # Parse sensors
    sensors = [s.strip() for s in args.sensors.split(',')]

    # Load dataset
    print(f"\nLoading SUN RGB-D dataset (sensors: {sensors})...")
    dataset_root = project_root / 'data/sunrgbd'

    dataset = SUNRGBDDataset(
        root_dir=str(dataset_root),
        split='val',  # Use validation split for evaluation
        sensors=sensors,
        resize=(480, 640)
    )

    if args.num_scenes == -1:
        num_scenes = len(dataset)
    else:
        num_scenes = min(args.num_scenes, len(dataset))

    print(f"✓ Loaded dataset with {len(dataset)} scenes")
    print(f"✓ Evaluating on {num_scenes} scenes")
    print(f"✓ Using base_scale={args.base_scale}")

    # Initialize evaluator with specified base_scale
    initializer = CameraAwareGaussianInit(
        base_scale=args.base_scale,
        uncertainty_weight=True,
        use_nearest_neighbor=False  # Can enable for k-NN adaptive scaling
    )

    # Evaluate scenes
    print(f"\nEvaluating scenes...")
    results = []

    for i in tqdm(range(num_scenes)):
        sample = dataset[i]
        result = evaluate_scene(sample, initializer)
        results.append(result)

    # Filter successful results
    valid_results = [r for r in results if r is not None]
    print(f"\n✓ Successfully evaluated {len(valid_results)}/{num_scenes} scenes")

    # Compute aggregate statistics
    print("\nComputing aggregate statistics...")
    stats = compute_aggregate_statistics(valid_results)

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"\nScale Variance Ratio: {stats['ratios'].get('scale_variance', 0):.2f}x")
    print(f"  Camera-Aware Mean: {stats['camera_aware'].get('scale_variance_mean', 0):.6f}")
    print(f"  Naive Mean: {stats['naive'].get('scale_variance_mean', 0):.6f}")

    # Save results
    output_dir = project_root / 'outputs'
    output_dir.mkdir(exist_ok=True)

    # Generate report
    report_path = output_dir / 'multi_scene_evaluation_report.png'
    generate_evaluation_report(stats, valid_results, report_path)

    # Save JSON results
    json_path = output_dir / 'multi_scene_evaluation_results.json'
    with open(json_path, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        stats_serializable = {}
        for method, method_stats in stats.items():
            stats_serializable[method] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in method_stats.items()
            }

        json.dump({
            'num_scenes': len(valid_results),
            'sensors': sensors,
            'stats': stats_serializable
        }, f, indent=2)

    print(f"✓ Saved JSON results: {json_path}")

    print("\n" + "=" * 80)
    print("✓ Multi-scene evaluation complete!")
    print("=" * 80)

    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
