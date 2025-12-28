"""
Loss Functions for 3D Gaussian Splatting
=========================================

Implements loss functions from the original 3DGS paper:
1. L1 photometric loss
2. D-SSIM (Structural Similarity) loss
3. Combined loss with weighting

References:
- 3D Gaussian Splatting (Kerbl et al., 2023)
- SSIM: "Image quality assessment: from error visibility to structural similarity" (Wang et al., 2004)

Author: ProjGS Research Team
Date: December 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index Measure (SSIM) Loss.

    SSIM compares local patterns of pixel intensities that have been normalized
    for luminance and contrast. It provides a perceptual quality metric.

    Args:
        window_size: Size of the Gaussian window (default: 11)
        size_average: If True, return mean loss; otherwise sum
        channel: Number of image channels
    """

    def __init__(
        self,
        window_size: int = 11,
        size_average: bool = True,
        channel: int = 3
    ):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

        # Create Gaussian window
        self.window = self._create_window(window_size, channel)

    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        """
        Create a 2D Gaussian window for SSIM computation.

        Args:
            window_size: Size of the window
            channel: Number of channels

        Returns:
            Gaussian window tensor
        """
        # Create 1D Gaussian kernel
        sigma = 1.5
        gauss = torch.Tensor([
            torch.exp(torch.tensor(-(x - window_size//2)**2 / (2.0 * sigma**2)))
            for x in range(window_size)
        ])
        gauss = gauss / gauss.sum()

        # Create 2D Gaussian window
        _1D_window = gauss.unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)

        # Expand to multiple channels
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()

        return window

    def _ssim(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        window: torch.Tensor,
        window_size: int,
        channel: int,
        size_average: bool = True
    ) -> torch.Tensor:
        """
        Compute SSIM between two images.

        Args:
            img1: First image [B, C, H, W]
            img2: Second image [B, C, H, W]
            window: Gaussian window
            window_size: Size of the window
            channel: Number of channels
            size_average: Whether to average the result

        Returns:
            SSIM value(s)
        """
        # Constants for stability
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # Ensure window is on the same device
        window = window.to(img1.device)

        # Compute means
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # Compute variances and covariance
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Compute SSIM loss.

        Args:
            img1: Predicted image [B, C, H, W]
            img2: Target image [B, C, H, W]

        Returns:
            SSIM loss (1 - SSIM)
        """
        _, channel, _, _ = img1.size()

        if channel == self.channel and self.window.device == img1.device:
            window = self.window
        else:
            window = self._create_window(self.window_size, channel)
            self.window = window
            self.channel = channel

        ssim_value = self._ssim(
            img1, img2, window, self.window_size, channel, self.size_average
        )

        # Return 1 - SSIM as loss (lower is better)
        return 1.0 - ssim_value


class GaussianSplattingLoss(nn.Module):
    """
    Combined loss function for 3D Gaussian Splatting.

    Combines L1 photometric loss with D-SSIM perceptual loss:
        L = λ₁ * L1 + λ₂ * (1 - SSIM)

    Args:
        lambda_l1: Weight for L1 loss (default: 0.8)
        lambda_dssim: Weight for D-SSIM loss (default: 0.2)
    """

    def __init__(
        self,
        lambda_l1: float = 0.8,
        lambda_dssim: float = 0.2
    ):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_dssim = lambda_dssim

        # Initialize SSIM loss
        self.ssim_loss = SSIMLoss(window_size=11, size_average=True)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute combined loss.

        Args:
            pred: Predicted image [B, C, H, W] or [B, H, W, C]
            target: Target image [B, C, H, W] or [B, H, W, C]
            return_components: If True, return (total_loss, l1_loss, ssim_loss)

        Returns:
            Total loss, or tuple of (total_loss, l1_loss, ssim_loss)
        """
        # Ensure images are in [B, C, H, W] format
        if pred.dim() == 4 and pred.shape[-1] == 3:
            pred = pred.permute(0, 3, 1, 2)
        if target.dim() == 4 and target.shape[-1] == 3:
            target = target.permute(0, 3, 1, 2)

        # L1 loss
        l1_loss = F.l1_loss(pred, target)

        # D-SSIM loss
        ssim_loss = self.ssim_loss(pred, target)

        # Combined loss
        total_loss = self.lambda_l1 * l1_loss + self.lambda_dssim * ssim_loss

        if return_components:
            return total_loss, l1_loss, ssim_loss
        else:
            return total_loss


class DepthLoss(nn.Module):
    """
    Depth supervision loss for RGB-D input.

    Optional loss to enforce consistency between rendered depth
    and input depth map.

    Args:
        loss_type: Type of loss ('l1', 'l2', or 'berhu')
        weight: Loss weight
    """

    def __init__(
        self,
        loss_type: str = 'l1',
        weight: float = 0.1
    ):
        super().__init__()
        self.loss_type = loss_type
        self.weight = weight

    def forward(
        self,
        pred_depth: torch.Tensor,
        target_depth: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute depth loss.

        Args:
            pred_depth: Predicted depth [B, 1, H, W] or [B, H, W]
            target_depth: Target depth [B, 1, H, W] or [B, H, W]
            valid_mask: Valid depth mask [B, 1, H, W] or [B, H, W]

        Returns:
            Depth loss
        """
        # Ensure same shape
        if pred_depth.dim() == 3:
            pred_depth = pred_depth.unsqueeze(1)
        if target_depth.dim() == 3:
            target_depth = target_depth.unsqueeze(1)

        # Apply valid mask if provided
        if valid_mask is not None:
            if valid_mask.dim() == 3:
                valid_mask = valid_mask.unsqueeze(1)
            pred_depth = pred_depth[valid_mask]
            target_depth = target_depth[valid_mask]

        # Compute loss based on type
        if self.loss_type == 'l1':
            loss = F.l1_loss(pred_depth, target_depth)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(pred_depth, target_depth)
        elif self.loss_type == 'berhu':
            # Reverse Huber loss (better for depth)
            diff = torch.abs(pred_depth - target_depth)
            c = 0.2 * torch.max(diff).item()
            mask = diff <= c
            loss = torch.where(
                mask,
                diff,
                (diff ** 2 + c ** 2) / (2 * c)
            ).mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return self.weight * loss


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).

    Args:
        pred: Predicted image
        target: Target image

    Returns:
        PSNR in dB
    """
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute SSIM metric.

    Args:
        pred: Predicted image [B, C, H, W]
        target: Target image [B, C, H, W]

    Returns:
        SSIM value
    """
    ssim_loss = SSIMLoss()
    return 1.0 - ssim_loss(pred, target).item()


# Test code
if __name__ == '__main__':
    print("Testing Loss Functions...")

    # Create sample images
    pred = torch.randn(2, 3, 256, 256)
    target = torch.randn(2, 3, 256, 256)

    # Test SSIM loss
    ssim_loss = SSIMLoss()
    loss = ssim_loss(pred, target)
    print(f"SSIM Loss: {loss.item():.4f}")

    # Test combined loss
    combined_loss = GaussianSplattingLoss(lambda_l1=0.8, lambda_dssim=0.2)
    total, l1, ssim = combined_loss(pred, target, return_components=True)
    print(f"Combined Loss: {total.item():.4f}")
    print(f"  L1: {l1.item():.4f}")
    print(f"  SSIM: {ssim.item():.4f}")

    # Test metrics
    psnr = compute_psnr(pred, target)
    ssim_metric = compute_ssim(pred, target)
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim_metric:.4f}")

    print("\n✓ All loss functions working!")
