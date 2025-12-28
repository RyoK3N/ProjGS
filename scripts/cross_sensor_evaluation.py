"""
Cross-Sensor Evaluation Script
===============================

Evaluate initialization consistency across different RGB-D sensors.

Tests whether camera-aware initialization provides better cross-sensor
generalization compared to naive approach.

Key Metrics:
- Consistency of scale distributions across sensors
- Adaptation to different focal lengths
- Cross-sensor performance drop

Usage:
    python scripts/cross_sensor_evaluation.py

Author: ProjGS Research Team
Date: December 2025
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from collections import defaultdict

from data.sunrgbd_loader import SUNRGBDDataset
from models.gaussian_init import CameraAwareGaussianInit


def evaluate_cross_sensor():
    """Evaluate initialization across different sensors."""
    print("=" * 80)
    print("Cross-Sensor Gaussian Initialization Evaluation")
    print("=" * 80)

    dataset_root = project_root / 'data/sunrgbd'

    # Load datasets for each sensor
    sensors = ['kv1', 'kv2']  # Start with these two
    datasets = {}

    print("\nLoading datasets...")
    for sensor in sensors:
        try:
            ds = SUNRGBDDataset(
                root_dir=str(dataset_root),
                split='val',
                sensors=[sensor],
                resize=(480, 640)
            )

            if len(ds) > 0:
                datasets[sensor] = ds
                print(f"  ✓ {sensor}: {len(ds)} scenes")
            else:
                print(f"  ✗ {sensor}: No scenes found")

        except Exception as e:
            print(f"  ✗ {sensor}: Failed to load ({e})")

    if len(datasets) == 0:
        print("\n✗ No valid datasets found")
        return False

    # Initialize evaluator
    initializer = CameraAwareGaussianInit(base_scale=0.01)

    # Collect results per sensor
    results_by_sensor = defaultdict(lambda: {'camera_aware': [], 'naive': []})

    print("\nEvaluating scenes per sensor...")
    for sensor, dataset in datasets.items():
        print(f"\n{sensor}:")

        # Test on first 10 scenes
        num_test = min(10, len(dataset))

        for i in range(num_test):
            sample = dataset[i]

            try:
                # Camera-aware
                g_aware = initializer.initialize_from_rgbd(
                    sample['rgb'], sample['depth'], sample['K']
                )
                m_aware = initializer.compute_initialization_metrics(g_aware)

                # Naive
                g_naive = initializer.initialize_naive(
                    sample['rgb'], sample['depth']
                )
                m_naive = initializer.compute_initialization_metrics(g_naive)

                results_by_sensor[sensor]['camera_aware'].append(m_aware)
                results_by_sensor[sensor]['naive'].append(m_naive)

                # Store camera parameters
                K = sample['K'].cpu().numpy()
                if 'camera_params' not in results_by_sensor[sensor]:
                    results_by_sensor[sensor]['camera_params'] = {
                        'fx': K[0, 0],
                        'fy': K[1, 1],
                        'cx': K[0, 2],
                        'cy': K[1, 2]
                    }

            except Exception as e:
                print(f"  Warning: Scene {i} failed: {e}")

        print(f"  ✓ Evaluated {len(results_by_sensor[sensor]['camera_aware'])} scenes")

    # Compute aggregate statistics
    print("\nComputing cross-sensor statistics...")
    sensor_stats = {}

    for sensor, results in results_by_sensor.items():
        sensor_stats[sensor] = {}

        for method in ['camera_aware', 'naive']:
            metrics = results[method]

            if len(metrics) == 0:
                continue

            # Compute means
            sensor_stats[sensor][method] = {
                'scale_variance_mean': np.mean([m['scale_variance'] for m in metrics]),
                'scale_mean_x_mean': np.mean([m['scale_mean_x'] for m in metrics]),
                'scale_mean_y_mean': np.mean([m['scale_mean_y'] for m in metrics]),
                'scale_mean_z_mean': np.mean([m['scale_mean_z'] for m in metrics]),
                'opacity_mean_mean': np.mean([m['opacity_mean'] for m in metrics]),
                'count_mean': np.mean([m['count'] for m in metrics]),
            }

        # Store camera params
        if 'camera_params' in results:
            sensor_stats[sensor]['camera_params'] = results['camera_params']

    # Generate visualization
    output_dir = project_root / 'outputs'
    output_dir.mkdir(exist_ok=True)

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # === Camera Parameters Table ===
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')

    cam_text = (
        "CAMERA PARAMETERS BY SENSOR\n"
        f"{'=' * 100}\n\n"
        f"{'Sensor':<10} {'fx (px)':<15} {'fy (px)':<15} {'cx (px)':<15} {'cy (px)':<15} "
        f"{'f_mean (px)':<15}\n"
        f"{'-' * 100}\n"
    )

    for sensor in sensor_stats.keys():
        if 'camera_params' in sensor_stats[sensor]:
            p = sensor_stats[sensor]['camera_params']
            f_mean = (p['fx'] + p['fy']) / 2
            cam_text += f"{sensor:<10} {p['fx']:<15.2f} {p['fy']:<15.2f} {p['cx']:<15.2f} {p['cy']:<15.2f} {f_mean:<15.2f}\n"

    ax1.text(0.05, 0.5, cam_text, fontsize=10, family='monospace',
             verticalalignment='center', transform=ax1.transAxes)

    # === Scale Variance Comparison ===
    ax2 = fig.add_subplot(gs[1, 0])

    sensors_list = [s for s in sensor_stats.keys()
                   if 'camera_aware' in sensor_stats[s] and 'naive' in sensor_stats[s]]
    x = np.arange(len(sensors_list))
    width = 0.35

    aware_vars = [sensor_stats[s]['camera_aware']['scale_variance_mean'] for s in sensors_list]
    naive_vars = [sensor_stats[s]['naive']['scale_variance_mean'] for s in sensors_list]

    bars1 = ax2.bar(x - width/2, aware_vars, width, label='Camera-Aware', alpha=0.7, color='blue')
    bars2 = ax2.bar(x + width/2, naive_vars, width, label='Naive', alpha=0.7, color='red')

    ax2.set_ylabel('Mean Scale Variance')
    ax2.set_title('Scale Variance Across Sensors')
    ax2.set_xticks(x)
    ax2.set_xticklabels(sensors_list)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.6f}', ha='center', va='bottom', fontsize=9)

    # === Focal Length vs Scale Adaptation ===
    ax3 = fig.add_subplot(gs[1, 1])

    focal_lengths = []
    scale_means = []

    for sensor in sensors_list:
        if 'camera_params' in sensor_stats[sensor]:
            p = sensor_stats[sensor]['camera_params']
            f_mean = (p['fx'] + p['fy']) / 2
            focal_lengths.append(f_mean)

            # Get scale mean from camera-aware
            scale_mean = sensor_stats[sensor]['camera_aware']['scale_mean_x_mean']
            scale_means.append(scale_mean)

    if len(focal_lengths) > 0:
        ax3.scatter(focal_lengths, scale_means, s=200, alpha=0.7, color='blue')

        # Add sensor labels
        for i, sensor in enumerate(sensors_list):
            ax3.annotate(sensor, (focal_lengths[i], scale_means[i]),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=10, fontweight='bold')

        # Expected relationship: scale ∝ 1/f
        # Fit curve
        if len(focal_lengths) > 1:
            z = np.polyfit(focal_lengths, scale_means, 1)
            p = np.poly1d(z)
            f_range = np.linspace(min(focal_lengths), max(focal_lengths), 100)
            ax3.plot(f_range, p(f_range), '--', alpha=0.5, color='gray',
                    label=f'Linear fit: y={z[0]:.6f}x+{z[1]:.6f}')
            ax3.legend()

    ax3.set_xlabel('Mean Focal Length (pixels)')
    ax3.set_ylabel('Mean Scale X (camera-aware)')
    ax3.set_title('Focal Length vs Scale Adaptation')
    ax3.grid(True, alpha=0.3)

    # === Consistency Analysis ===
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')

    # Compute consistency metrics
    if len(sensors_list) > 1:
        # Variance across sensors (lower is more consistent)
        aware_consistency = np.std(aware_vars)
        naive_consistency = np.std(naive_vars)

        consistency_text = (
            "CROSS-SENSOR CONSISTENCY\n"
            f"{'=' * 40}\n\n"
            f"Scale Variance Std Across Sensors:\n"
            f"  Camera-Aware: {aware_consistency:.8f}\n"
            f"  Naive: {naive_consistency:.8f}\n\n"
        )

        if aware_consistency < naive_consistency:
            consistency_text += f"✓ Camera-aware is MORE consistent\n"
            consistency_text += f"  ({(naive_consistency/aware_consistency):.2f}x better)\n"
        else:
            consistency_text += f"✗ Camera-aware is LESS consistent\n"

        # Focal length adaptation check
        if len(focal_lengths) > 1:
            # Check if scale inversely correlates with focal length
            corr = np.corrcoef(focal_lengths, scale_means)[0, 1]
            consistency_text += f"\nFocal Length Correlation:\n"
            consistency_text += f"  Pearson r: {corr:.3f}\n"

            if corr < -0.5:
                consistency_text += f"  ✓ Strong inverse correlation\n"
                consistency_text += f"  (scales adapt to focal length)\n"
            elif corr < 0:
                consistency_text += f"  ⚠ Weak inverse correlation\n"
            else:
                consistency_text += f"  ✗ No inverse correlation\n"

    else:
        consistency_text = "Need >1 sensor for consistency analysis"

    ax4.text(0.05, 0.5, consistency_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax4.transAxes)

    # === Detailed Metrics Table ===
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')

    metrics_text = (
        "DETAILED METRICS COMPARISON\n"
        f"{'=' * 120}\n\n"
        f"{'Sensor':<10} {'Method':<15} {'Scale Var':<15} {'Scale X':<15} "
        f"{'Opacity':<15} {'Count':<15}\n"
        f"{'-' * 120}\n"
    )

    for sensor in sensors_list:
        for method in ['camera_aware', 'naive']:
            if method in sensor_stats[sensor]:
                s = sensor_stats[sensor][method]
                metrics_text += (
                    f"{sensor:<10} {method:<15} {s['scale_variance_mean']:<15.8f} "
                    f"{s['scale_mean_x_mean']:<15.6f} {s['opacity_mean_mean']:<15.4f} "
                    f"{s['count_mean']:<15.0f}\n"
                )
        metrics_text += "\n"

    ax5.text(0.05, 0.5, metrics_text, fontsize=9, family='monospace',
            verticalalignment='center', transform=ax5.transAxes)

    plt.suptitle('Cross-Sensor Evaluation Report', fontsize=16, fontweight='bold')

    # Save
    report_path = output_dir / 'cross_sensor_evaluation_report.png'
    plt.savefig(report_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Generated report: {report_path}")

    print("\n" + "=" * 80)
    print("✓ Cross-sensor evaluation complete!")
    print("=" * 80)

    return True


if __name__ == '__main__':
    success = evaluate_cross_sensor()
    sys.exit(0 if success else 1)
