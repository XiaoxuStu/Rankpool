"""Quick-start example: plug RankPool into a ResNet18."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from rankpool import RankPool2d, RobustMixedPool2d


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet18(nn.Module):
    """
    ResNet18 with pooling after layer1.
    conv1 → layer1 → POOL → layer2 → layer3 → layer4 → fc
    """

    def __init__(self, pool_fn, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.pool = pool_fn()
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_ch, out_ch, num_blocks, stride):
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.pool(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return self.fc(x.flatten(1))


# ---- Usage ----
if __name__ == "__main__":
    # Option 1: RankPool (p=2, RMS-weighted)
    model_rank = ResNet18(lambda: RankPool2d(2, stride=2))

    # Option 2: RobustMixed (learnable RankPool + AvgPool blend)
    model_mixed = ResNet18(lambda: RobustMixedPool2d(2, stride=2))

    # Option 3: RankPool with different p values
    model_p1 = ResNet18(lambda: RankPool2d(2, stride=2, p=1))   # inverse-harmonic
    model_p3 = ResNet18(lambda: RankPool2d(2, stride=2, p=3))   # more aggressive

    # Quick forward pass check
    x = torch.randn(2, 3, 32, 32)
    for name, m in [("RankPool", model_rank), ("RobustMixed", model_mixed)]:
        y = m(x)
        n = sum(p.numel() for p in m.parameters())
        print(f"{name}: input {x.shape} → output {y.shape}, params {n:,}")
