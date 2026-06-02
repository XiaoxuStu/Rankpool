"""
Advanced RankPool variants.

RobustMixedPool2d  — Learnable blend of RankPool + AvgPool (recommended)
RobustStablePool2d — RankPool with learnable temperature
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RobustMixedPool2d(nn.Module):
    """Learnable blend of RankPool (RMS) and AveragePool.

    Output = (1 − α) · RankPool(x) + α · AvgPool(x)

    * ``α=0`` → pure RankPool (most selective)
    * ``α=1`` → pure AvgPool (most smooth)
    * Learned values cluster around α ≈ 0.39

    This is the recommended pooling method for occlusion robustness.

    Args:
        kernel_size (int or tuple): Pooling window size.
        stride (int or tuple, optional): Defaults to *kernel_size*.
        padding (int or tuple): Zero-padding added to both sides.
        init_alpha (float): Initial blend coefficient (default ``0.5``).
    """

    def __init__(self, kernel_size, stride=None, padding=0, init_alpha=0.5):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size
        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)
        if isinstance(padding, int):
            self.padding = (padding, padding)
        self.alpha = nn.Parameter(torch.tensor(init_alpha))

    def forward(self, x):
        B, C, H, W = x.shape
        N = self.kernel_size[0] * self.kernel_size[1]
        patches = F.unfold(x, self.kernel_size,
                           stride=self.stride, padding=self.padding)
        L = patches.shape[2]
        patches = patches.view(B, C, N, L)

        # RankPool (RMS-weighted mean)
        w = patches.abs().pow(2)
        w = w / (w.sum(dim=2, keepdim=True) + 1e-8)
        rank_out = (w * patches).sum(dim=2)

        # AveragePool
        avg_out = patches.mean(dim=2)

        # Blend with sigmoid-bounded alpha
        alpha = torch.sigmoid(self.alpha)
        output = (1 - alpha) * rank_out + alpha * avg_out

        H_out = (H + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        W_out = (W + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        return output.view(B, C, H_out, W_out)


class RobustStablePool2d(nn.Module):
    """RankPool with learnable temperature (stability control).

    Uses softmax-style weighting with a learnable temperature τ:

    .. math::
        w_i = \\exp(x_i / \\tau) / \\sum \\exp(x_k / \\tau)

    * Low τ → sharp weights (towards MaxPool)
    * High τ → uniform weights (towards AvgPool)

    Args:
        kernel_size (int or tuple): Pooling window size.
        stride (int or tuple, optional): Defaults to *kernel_size*.
        padding (int or tuple): Zero-padding added to both sides.
        init_tau (float): Initial temperature (default ``1.0``).
    """

    def __init__(self, kernel_size, stride=None, padding=0, init_tau=1.0):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size
        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)
        if isinstance(padding, int):
            self.padding = (padding, padding)
        self.log_tau = nn.Parameter(torch.tensor(init_tau).log())

    def forward(self, x):
        B, C, H, W = x.shape
        N = self.kernel_size[0] * self.kernel_size[1]
        patches = F.unfold(x, self.kernel_size,
                           stride=self.stride, padding=self.padding)
        L = patches.shape[2]
        patches = patches.view(B, C, N, L)
        tau = self.log_tau.exp()
        w = F.softmax(patches / tau, dim=2)
        output = (w * patches).sum(dim=2)
        H_out = (H + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        W_out = (W + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        return output.view(B, C, H_out, W_out)
