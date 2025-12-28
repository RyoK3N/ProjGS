"""
ProjGS Data Module
==================

Data loading and preprocessing utilities for camera-aware Gaussian Splatting.

Provides:
- SUN RGB-D dataset loader with camera intrinsics
- Data augmentation for RGB-D scenes
- Cross-sensor dataset handling
- Batching and sampling strategies

Author: ProjGS Research Team
Date: December 2025
"""

from .sunrgbd_loader import SUNRGBDDataset, visualize_sample

__all__ = ['SUNRGBDDataset', 'visualize_sample']
