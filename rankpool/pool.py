Core pooling layers.

RankPool2d   — Power mean weighted pooling (p=2 is RMS)
RobustPool2d — RankPool + max_weight clamp
SoftPool2d   — Exponentially weighted (softmax) pooling


import torch
import torch.nn as nn
import torch.nn.functional as F


class RankPool2d(nn.Module)
    rPower mean weighted pooling.

    Weight math`w_i = x_i^p  sum x_k^p`, output math`sum w_i cdot x_i`.

     ``p=1`` → inverse-harmonic mean (weights ∝ x)
     ``p=2`` → RMS-weighted mean (default)
     ``p→∞`` → approaches MaxPool

    Args
        kernel_size (int or tuple) Pooling window size.
        stride (int or tuple, optional) Defaults to kernel_size.
        padding (int or tuple) Zero-padding added to both sides.
        p (float) Power exponent (default ``2.0``).
    

    def __init__(self, kernel_size, stride=None, padding=0, p=2.0)
        super().__init__()
        if isinstance(kernel_size, int)
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size
        if isinstance(self.stride, int)
            self.stride = (self.stride, self.stride)
        if isinstance(padding, int)
            self.padding = (padding, padding)
        self.p = p

    def forward(self, x)
        B, C, H, W = x.shape
        N = self.kernel_size[0]  self.kernel_size[1]
        patches = F.unfold(x, self.kernel_size,
                           stride=self.stride, padding=self.padding)
        L = patches.shape[2]
        patches = patches.view(B, C, N, L)
        w = patches.abs().pow(self.p)
        w = w  (w.sum(dim=2, keepdim=True) + 1e-8)
        output = (w  patches).sum(dim=2)
        H_out = (H + 2  self.padding[0] - self.kernel_size[0])  self.stride[0] + 1
        W_out = (W + 2  self.padding[1] - self.kernel_size[1])  self.stride[1] + 1
        return output.view(B, C, H_out, W_out)


class RobustPool2d(nn.Module)
    RankPool with max-weight clamp.

    Clamps the maximum pooling weight to ``max_weight`` and renormalises,
    preventing a single element from fully dominating the output.

    Args
        kernel_size (int or tuple) Pooling window size.
        stride (int or tuple, optional) Defaults to kernel_size.
        padding (int or tuple) Zero-padding added to both sides.
        max_weight (float, optional) Weight cap (default ``1N``).
    

    def __init__(self, kernel_size, stride=None, padding=0, max_weight=None)
        super().__init__()
        if isinstance(kernel_size, int)
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size
        if isinstance(self.stride, int)
            self.stride = (self.stride, self.stride)
        if isinstance(padding, int)
            self.padding = (padding, padding)
        N = kernel_size[0]  kernel_size[1]
        self.max_weight = max_weight if max_weight is not None else 1.0  N

    def forward(self, x)
        B, C, H, W = x.shape
        N = self.kernel_size[0]  self.kernel_size[1]
        patches = F.unfold(x, self.kernel_size,
                           stride=self.stride, padding=self.padding)
        L = patches.shape[2]
        patches = patches.view(B, C, N, L)
        abs_sum = patches.abs().sum(dim=2, keepdim=True) + 1e-8
        w = patches.abs()  abs_sum
        w = torch.clamp(w, max=self.max_weight)
        w = w  (w.sum(dim=2, keepdim=True) + 1e-8)
        output = (w  patches).sum(dim=2)
        H_out = (H + 2  self.padding[0] - self.kernel_size[0])  self.stride[0] + 1
        W_out = (W + 2  self.padding[1] - self.kernel_size[1])  self.stride[1] + 1
        return output.view(B, C, H_out, W_out)


class SoftPool2d(nn.Module)
    Exponentially weighted (softmax) pooling.

    Args
        kernel_size (int or tuple) Pooling window size.
        stride (int or tuple, optional) Defaults to kernel_size.
        padding (int or tuple) Zero-padding added to both sides.
        tau (float) Temperature for softmax (default ``1.0``).
    

    def __init__(self, kernel_size, stride=None, padding=0, tau=1.0)
        super().__init__()
        if isinstance(kernel_size, int)
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size
        if isinstance(self.stride, int)
            self.stride = (self.stride, self.stride)
        if isinstance(padding, int)
            self.padding = (padding, padding)
        self.tau = tau

    def forward(self, x)
        B, C, H, W = x.shape
        N = self.kernel_size[0]  self.kernel_size[1]
        patches = F.unfold(x, self.kernel_size,
                           stride=self.stride, padding=self.padding)
        L = patches.shape[2]
        patches = patches.view(B, C, N, L)
        w = F.softmax(patches  self.tau, dim=2)
        output = (w  patches).sum(dim=2)
        H_out = (H + 2  self.padding[0] - self.kernel_size[0])  self.stride[0] + 1
        W_out = (W + 2  self.padding[1] - self.kernel_size[1])  self.stride[1] + 1
        return output.view(B, C, H_out, W_out)
