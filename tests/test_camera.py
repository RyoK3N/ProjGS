"""
Unit Tests for Camera Utilities
================================

Tests camera intrinsics parsing, depth unprojection, and geometric operations.

Test Cases:
1. Intrinsics parsing from SUN RGB-D format
2. Depth unprojection to 3D coordinates
3. 3D projection back to 2D
4. Projection-unprojection cycle consistency
5. Intrinsics scaling for resized images
6. FOV computation
7. Error handling and edge cases

Author: ProjGS Research Team
Date: December 2025
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile

from utils.camera import CameraIntrinsics, CameraExtrinsics


class TestCameraIntrinsics:
    """Test suite for CameraIntrinsics class."""

    @pytest.fixture
    def sample_intrinsics_file(self):
        """Create temporary intrinsics file with Kinect v1 parameters."""
        # Kinect v1 intrinsics
        K_values = "518.857901 0.000000 284.582449 0.000000 519.469611 208.736166 0.000000 0.000000 1.000000"

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(K_values)
            filepath = f.name

        yield filepath

        # Cleanup
        Path(filepath).unlink(missing_ok=True)

    def test_intrinsics_parsing(self, sample_intrinsics_file):
        """Test correct parsing of intrinsics file."""
        cam = CameraIntrinsics(sample_intrinsics_file)

        # Verify focal lengths
        assert np.isclose(cam.fx, 518.857901, atol=1e-4)
        assert np.isclose(cam.fy, 519.469611, atol=1e-4)

        # Verify principal point
        assert np.isclose(cam.cx, 284.582449, atol=1e-4)
        assert np.isclose(cam.cy, 208.736166, atol=1e-4)

        # Verify matrix structure
        assert cam.K.shape == (3, 3)
        assert np.isclose(cam.K[2, 2], 1.0)

    def test_invalid_intrinsics_file(self):
        """Test error handling for invalid files."""
        # Non-existent file
        with pytest.raises(FileNotFoundError):
            CameraIntrinsics('/nonexistent/path/intrinsics.txt')

    def test_invalid_intrinsics_format(self):
        """Test error handling for invalid format."""
        # Too few values
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("518.86 0 284.58")  # Only 3 values instead of 9
            filepath = f.name

        try:
            with pytest.raises(ValueError, match="Expected 9 values"):
                CameraIntrinsics(filepath)
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_unprojection_center_pixel(self, sample_intrinsics_file):
        """Test unprojection of center pixel."""
        cam = CameraIntrinsics(sample_intrinsics_file)

        # Create depth map with uniform depth
        depth = np.ones((480, 640)) * 2.0  # 2 meters depth

        # Unproject
        points = cam.unproject(depth, return_tensor=False)

        # Get center pixel (should be at principal point)
        center_y = int(cam.cy)
        center_x = int(cam.cx)
        center_point = points[center_y, center_x]

        # At principal point, X and Y should be ~0
        assert np.abs(center_point[0]) < 0.01  # X ≈ 0
        assert np.abs(center_point[1]) < 0.01  # Y ≈ 0
        assert np.isclose(center_point[2], 2.0, atol=0.01)  # Z = depth

    def test_unprojection_returns_correct_shape(self, sample_intrinsics_file):
        """Test unprojection output shape."""
        cam = CameraIntrinsics(sample_intrinsics_file)

        H, W = 100, 150
        depth = np.random.rand(H, W) * 5.0

        # Test numpy return
        points_np = cam.unproject(depth, return_tensor=False)
        assert points_np.shape == (H, W, 3)
        assert isinstance(points_np, np.ndarray)

        # Test tensor return
        points_tensor = cam.unproject(depth, return_tensor=True)
        assert points_tensor.shape == (H, W, 3)
        assert isinstance(points_tensor, torch.Tensor)

    def test_projection_unprojection_cycle(self, sample_intrinsics_file):
        """Test that project(unproject(depth)) gives back pixel coordinates."""
        cam = CameraIntrinsics(sample_intrinsics_file)

        # Create random depth map
        H, W = 100, 100
        depth = np.random.rand(H, W) * 5.0 + 0.5  # 0.5-5.5 meters

        # Unproject to 3D
        points_3d = cam.unproject(depth, return_tensor=False, mask_invalid=False)
        points_3d_flat = points_3d.reshape(-1, 3)

        # Project back to 2D
        pixels = cam.project(points_3d_flat)
        pixels_reshaped = pixels.reshape(H, W, 2)

        # Create expected pixel grids
        u_expected = np.arange(W)
        v_expected = np.arange(H)
        u_grid, v_grid = np.meshgrid(u_expected, v_expected)

        # Check that reprojected pixels match original
        u_error = np.abs(pixels_reshaped[:, :, 0] - u_grid)
        v_error = np.abs(pixels_reshaped[:, :, 1] - v_grid)

        # Should have very small error (numerical precision)
        assert u_error.mean() < 1e-3
        assert v_error.mean() < 1e-3
        assert u_error.max() < 1e-2
        assert v_error.max() < 1e-2

    def test_invalid_depth_masking(self, sample_intrinsics_file):
        """Test that invalid depths are properly masked."""
        cam = CameraIntrinsics(sample_intrinsics_file)

        # Create depth with invalid regions
        depth = np.ones((100, 100)) * 2.0
        depth[0:20, :] = 0  # Zero depth (invalid)
        depth[80:100, :] = 15.0  # Beyond max_depth (invalid)

        # Unproject with masking
        points = cam.unproject(depth, return_tensor=False, mask_invalid=True, max_depth=10.0)

        # Check that invalid regions are NaN
        assert np.all(np.isnan(points[0:20, :, :]))
        assert np.all(np.isnan(points[80:100, :, :]))

        # Valid region should not be NaN
        assert not np.any(np.isnan(points[20:80, :, :]))

    def test_scale_intrinsics(self, sample_intrinsics_file):
        """Test intrinsics scaling for resized images."""
        cam = CameraIntrinsics(sample_intrinsics_file)

        # Scale to half resolution
        cam_scaled = cam.scale_intrinsics(scale_x=0.5, scale_y=0.5)

        # Focal lengths should be halved
        assert np.isclose(cam_scaled.fx, cam.fx * 0.5)
        assert np.isclose(cam_scaled.fy, cam.fy * 0.5)

        # Principal point should be halved
        assert np.isclose(cam_scaled.cx, cam.cx * 0.5)
        assert np.isclose(cam_scaled.cy, cam.cy * 0.5)

    def test_compute_fov(self, sample_intrinsics_file):
        """Test field of view computation."""
        cam = CameraIntrinsics(sample_intrinsics_file)

        fov_x, fov_y = cam.compute_fov(640, 480)

        # Kinect v1 has approximately 58° horizontal FOV
        assert 55 < fov_x < 65
        assert 40 < fov_y < 50

    def test_to_tensor(self, sample_intrinsics_file):
        """Test conversion to PyTorch tensor."""
        cam = CameraIntrinsics(sample_intrinsics_file)

        # CPU tensor
        K_cpu = cam.to_tensor(device='cpu')
        assert isinstance(K_cpu, torch.Tensor)
        assert K_cpu.shape == (3, 3)
        assert K_cpu.device.type == 'cpu'

        # Verify values match
        assert torch.isclose(K_cpu[0, 0], torch.tensor(cam.fx))

    def test_repr_and_str(self, sample_intrinsics_file):
        """Test string representations."""
        cam = CameraIntrinsics(sample_intrinsics_file)

        # __repr__ should be short
        repr_str = repr(cam)
        assert 'CameraIntrinsics' in repr_str
        assert 'fx' in repr_str

        # __str__ should be detailed
        str_str = str(cam)
        assert 'Focal Length' in str_str
        assert 'Principal Point' in str_str


class TestCameraExtrinsics:
    """Test suite for CameraExtrinsics class."""

    @pytest.fixture
    def sample_extrinsics_file(self):
        """Create temporary extrinsics file."""
        # Identity rotation, zero translation
        Rt = np.eye(3, 4)

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            np.savetxt(f, Rt)
            filepath = f.name

        yield filepath

        Path(filepath).unlink(missing_ok=True)

    def test_extrinsics_loading(self, sample_extrinsics_file):
        """Test loading extrinsics from file."""
        ext = CameraExtrinsics(sample_extrinsics_file)

        assert ext.Rt.shape == (3, 4)
        assert ext.R.shape == (3, 3)
        assert ext.t.shape == (3,)

        # Should be identity
        assert np.allclose(ext.R, np.eye(3))
        assert np.allclose(ext.t, np.zeros(3))

    def test_transform_points_identity(self, sample_extrinsics_file):
        """Test transformation with identity extrinsics."""
        ext = CameraExtrinsics(sample_extrinsics_file)

        # Random points
        points = np.random.rand(100, 3)

        # Identity transform should not change points
        transformed = ext.transform_points(points, inverse=False)
        assert np.allclose(transformed, points)

    def test_transform_inverse(self, sample_extrinsics_file):
        """Test forward and inverse transformations."""
        # Create non-identity extrinsics
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # 90° rotation around Z
        t = np.array([1, 2, 3])
        Rt = np.column_stack([R, t])

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            np.savetxt(f, Rt)
            filepath = f.name

        try:
            ext = CameraExtrinsics(filepath)

            # Random points
            points_camera = np.random.rand(100, 3)

            # Forward then inverse should give back original
            points_world = ext.transform_points(points_camera, inverse=False)
            points_back = ext.transform_points(points_world, inverse=True)

            assert np.allclose(points_back, points_camera, atol=1e-6)

        finally:
            Path(filepath).unlink(missing_ok=True)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
