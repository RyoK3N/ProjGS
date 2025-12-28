"""
ProjGS Utilities Module
=======================

Core utilities for camera-aware Gaussian Splatting.

This module provides:
- Camera intrinsics and extrinsics handling
- Depth unprojection with camera matrices
- Geometric transformations
- Projection and reprojection utilities

Author: ProjGS Research Team
Date: December 2025
"""

from .camera import CameraIntrinsics, CameraExtrinsics

__all__ = ['CameraIntrinsics', 'CameraExtrinsics']
__version__ = '0.1.0'
