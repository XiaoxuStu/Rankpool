"""
RankPool: Rank-Weighted Pooling for Robust Feature Aggregation

A pooling layer that assigns weights proportional to absolute values,
achieving both high accuracy and strong robustness against occlusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RankPool2d(nn.Module):
    """
    Rank-Weighted Pooling (RankPool).

    For each spatial pooling window, computes output as a weighted average
    where weights are proportional to the absolute value of each element:

        w_i = |x_i|^p / Σ|x_j|^p
        output = Σ(w_i * x_i)

    This naturally assigns higher weight to more salient features while
    remaining differentiable everywhere.

    Args:
        kernel_size (int or tuple): Size of the pooling window.
        stride (int or tuple, optional): Stride of the pooling window.
            Default: same as kernel_size.
        padding (int or tuple): Zero-padding added to both sides.
            Default: 0.
        p (float): Exponent controlling the sharpness of the weight
            distribution. p=1 (default) assigns weights strictly
            proportional to |x|. Higher p increases focus on the
            maximum value (approaches MaxPool); lower p increases
            uniformity (approaches AvgPool).

            - p → 0: approaches AvgPool
            - p = 1: standard RankPool (recommended)
            - p → ∞: approaches MaxPool

    Shape:
        - Input:  (B, C, H_in, W_in)
        - Output: (B, C, H_out, W_out)

        H_out = floor((H_in + 2*padding - kernel_size) / stride) + 1
        W_out = floor((W_in + 2*padding - kernel_size) / stride) + 1

    Example:
        >>> pool = RankPool2d(kernel_size=2, stride=2)
        >>> x = torch.randn(16, 64, 32, 32)
        >>> y = pool(x)
        >>> y.shape
        torch.Size([16, 64, 16, 16])
    """

    def __init__(self, kernel_size, stride=None, padding=0, p=1.0):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size
        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)
        if isinstance(padding, int):
            self.padding = (padding, padding)
        self.p = p

    def forward(self, x):
        B, C, H, W = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        N = kh * kw

        # Extract patches: (B, C*N, L)
        patches = F.unfold(x, self.kernel_size,
                           stride=self.stride, padding=self.padding)
        L = patches.shape[2]
        # Reshape: (B, C, N, L)
        patches = patches.view(B, C, N, L)

        # Compute rank weights
        if self.p == 1.0:
            abs_patches = patches.abs()
        else:
            abs_patches = patches.abs().clamp(min=1e-8).pow(self.p)

        w = abs_patches / (abs_patches.sum(dim=2, keepdim=True) + 1e-8)

        # Weighted aggregation using original (signed) values
        output = (w * patches).sum(dim=2)

        # Compute output spatial dimensions
        H_out = (H + 2 * ph - kh) // sh + 1
        W_out = (W + 2 * pw - kw) // sw + 1

        return output.view(B, C, H_out, W_out)

    def extra_repr(self):
        return (f"kernel_size={self.kernel_size}, stride={self.stride}, "
                f"padding={self.padding}, p={self.p}")
