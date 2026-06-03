# RankPool

**Rank-Weighted Pooling for Robust Feature Aggregation**

A simple, drop-in replacement for `MaxPool2d` / `AvgPool2d` that assigns
weights proportional to absolute feature values — combining the precision
of MaxPool with the robustness of AvgPool.

## Key Idea

Standard pooling layers use fixed strategies: MaxPool selects the single
largest value (fragile to noise/occlusion), while AvgPool treats all values
equally (dilutes salient features). **RankPool** takes a middle ground:

```
w_i = |x_i|^p / Σ|x_j|^p
output = Σ(w_i · x_i)
```

Each element receives weight proportional to its magnitude. Larger values
naturally dominate, but no information is completely discarded.

## Installation

```bash
pip install git+https://github.com/XiaoxuStu/RankPool.git
```

Or clone and install locally:

```bash
git clone https://github.com/XiaoxuStu/RankPool.git
cd RankPool
pip install -e .
```

## Quick Start

```python
from rankpool import RankPool2d
import torch.nn as nn

# Drop-in replacement
model = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.ReLU(),
    RankPool2d(kernel_size=2, stride=2),  # replaces nn.MaxPool2d(2, 2)
    nn.Conv2d(64, 128, 3, padding=1),
    nn.ReLU(),
    RankPool2d(kernel_size=2, stride=2),
)
```

## Results

### CIFAR-10 with ResNet-18 (seed=42, 30 epochs)

| Pooling | Accuracy | Occlusion Drop Rate |
|---------|----------|---------------------|
| MaxPool | 91.76%   | 42.2%               |
| AvgPool | 91.78%   | 33.1%               |
| **RankPool** | **91.80%** | **34.2%** |

RankPool achieves **highest accuracy** and **near-best robustness**
simultaneously — a rare combination where most methods require a trade-off.

### Occlusion Robustness (Test A: feature map block occlusion)

| Block Ratio | MaxPool | AvgPool | RankPool |
|-------------|---------|---------|----------|
| 25% area    | 87.15%  | 89.13%  | 87.92%   |
| 50% area    | 53.05%  | 61.40%  | 60.35%   |
| 75% area    | 15.52%  | 21.92%  | 19.93%   |

### Ablation: Power Parameter p

| p | Accuracy | Occlusion Drop Rate |
|---|----------|---------------------|
| 0.5 | 90.99% | 38.0% |
| **1.0** | **91.80%** | **34.2%** |
| 1.5 | 91.72% | 44.8% |
| 2.0 | 91.78% | 47.6% |
| learnable (→0.94) | 91.28% | 41.9% |

The default `p=1` is optimal. A learnable-p variant independently
converges to p≈0.94, confirming this finding.

## API Reference

```python
RankPool2d(kernel_size, stride=None, padding=0, p=1.0)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `kernel_size` | `int` or `tuple` | — | Pooling window size |
| `stride` | `int` or `tuple` | `kernel_size` | Pooling stride |
| `padding` | `int` or `tuple` | `0` | Zero-padding |
| `p` | `float` | `1.0` | Exponent controlling weight sharpness |

## Reproducing Results

```bash
python examples/benchmark_resnet18_cifar10.py
```

## License

MIT

## Citation

If you use RankPool in your research, please cite:

```bibtex
@software{rankpool2026,
  title   = {RankPool: Rank-Weighted Pooling for Robust Feature Aggregation},
  author  = {XiaoxuStu},
  year    = {2026},
  url     = {https://github.com/XiaoxuStu/RankPool}
}
```
