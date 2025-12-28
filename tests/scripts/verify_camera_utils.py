"""
Verification Script for Camera Utilities
=========================================

Quick verification of camera utilities using real SUN RGB-D data.

Tests:
1. Intrinsics loading from dataset
2. Depth unprojection accuracy
3. Projection-unprojection cycle consistency

Author: ProjGS Research Team
Date: December 2025
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from utils.camera import CameraIntrinsics


def main():
    """Run verification tests."""
    print("Camera Utilities Verification")
    print("=" * 80)

    # Test loading intrinsics from actual dataset
    intrinsics_path = project_root / 'data/sunrgbd/kv1/NYUdata/NYU0001/intrinsics.txt'

    if not intrinsics_path.exists():
        print(f"✗ Dataset not found at {intrinsics_path}")
        print("Please ensure SUN RGB-D dataset is downloaded.")
        sys.exit(1)

    print(f"\n1. Loading intrinsics from: {intrinsics_path.name}")
    print("-" * 80)

    cam = CameraIntrinsics(intrinsics_path)
    print(f"\n{cam}")

    # Test unprojection
    print("\n2. Testing Depth Unprojection")
    print("-" * 80)
    depth = np.ones((480, 640)) * 2.0
    points = cam.unproject(depth, return_tensor=False)
    print(f"Unprojected shape: {points.shape}")

    # Check center pixel
    center_y = int(cam.cy)
    center_x = int(cam.cx)
    center_point = points[center_y, center_x]
    print(f"Center pixel (expected ~[0, 0, 2]): [{center_point[0]:.6f}, {center_point[1]:.6f}, {center_point[2]:.6f}]")

    # Verify center pixel
    if np.abs(center_point[0]) < 0.01 and np.abs(center_point[1]) < 0.01 and np.isclose(center_point[2], 2.0):
        print("✓ Center pixel unprojection correct")
    else:
        print("✗ Center pixel unprojection incorrect")
        return False

    # Test projection-unprojection cycle
    print("\n3. Testing Projection-Unprojection Cycle")
    print("-" * 80)
    H, W = 100, 100
    depth_test = np.random.rand(H, W) * 5.0 + 0.5

    points_3d = cam.unproject(depth_test, return_tensor=False, mask_invalid=False)
    points_flat = points_3d.reshape(-1, 3)

    pixels = cam.project(points_flat)
    pixels_grid = pixels.reshape(H, W, 2)

    u_expected, v_expected = np.meshgrid(np.arange(W), np.arange(H))
    u_error = np.abs(pixels_grid[:, :, 0] - u_expected)
    v_error = np.abs(pixels_grid[:, :, 1] - v_expected)

    print(f"Mean reprojection error: U={u_error.mean():.8f}, V={v_error.mean():.8f} pixels")
    print(f"Max reprojection error:  U={u_error.max():.8f}, V={v_error.max():.8f} pixels")

    # Verification threshold
    if u_error.mean() < 0.001 and v_error.mean() < 0.001:
        print("✓ Reprojection error within tolerance (<0.001 pixels)")
    else:
        print("✗ Reprojection error too large")
        return False

    print("\n" + "=" * 80)
    print("✓ All camera utilities verification tests PASSED!")
    print("=" * 80)
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
