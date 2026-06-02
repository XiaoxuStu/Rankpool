"""
RankPool: Power Mean Weighted Pooling for Occlusion Robustness

Implements RankPool (p-norm weighted pooling), RobustPool (clamped),
and RobustMixedPool (learnable RankPool + AvgPool blend).

Reference: RankPool assigns pooling weights proportional to |x|^p,
          naturally giving zero weight to occluded (zero) elements.
"""

from .pool import RankPool2d, RobustPool2d, SoftPool2d
from .robpool import RobustMixedPool2d, RobustStablePool2d

__version__ = "0.1.0"
__all__ = [
    "RankPool2d",
    "RobustPool2d",
    "RobustMixedPool2d",
    "RobustStablePool2d",
    "SoftPool2d",
]
