"""
ProjGS Models Module
====================

Neural network models and initialization strategies for camera-aware Gaussian Splatting.

Provides:
- Camera-aware Gaussian initialization
- Focal-length-adaptive scaling
- Position uncertainty weighting
- Cross-sensor normalization

Author: ProjGS Research Team
Date: December 2025
"""

from .gaussian_init import CameraAwareGaussianInit

__all__ = ['CameraAwareGaussianInit']
