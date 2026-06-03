
Basic usage of RankPool2d as a drop-in replacement for nn.MaxPool2d  nn.AvgPool2d.


import torch
import torch.nn as nn
from rankpool import RankPool2d


# --- Drop-in replacement ---
x = torch.randn(4, 64, 32, 32)

maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
rankpool = RankPool2d(kernel_size=2, stride=2)

print(Input        , x.shape)
print(MaxPool output, maxpool(x).shape)
print(AvgPool output, avgpool(x).shape)
print(RankPool output, rankpool(x).shape)


# --- Insert into any CNN ---
class SimpleCNN(nn.Module)
    def __init__(self, num_classes=10)
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            RankPool2d(2, stride=2),       # ← Replace MaxPool2d here
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            RankPool2d(2, stride=2),       # ← And here
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x)
        return self.classifier(self.features(x))


model = SimpleCNN()
out = model(torch.randn(2, 3, 32, 32))
print(nSimpleCNN output, out.shape)
