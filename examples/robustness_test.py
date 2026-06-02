"""
Robustness evaluation: block occlusion on feature maps and images.
Loads trained models and measures accuracy degradation.
"""

import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from rankpool import RankPool2d, RobustMixedPool2d
from rankpool.utils import (
    test_occlusion_feature,
    test_occlusion_image,
    test_occlusion_multi,
    weight_analysis,
)
from examples.basic_usage import ResNet18, BasicBlock  # reuse model definition


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    set_seed(42)

    # Data
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    test_data = datasets.CIFAR10("./data", train=False, download=False,
                                 transform=transform_test)
    test_loader = DataLoader(test_data, batch_size=256, num_workers=0)

    # Load your trained models here
    # model_rank = ResNet18(lambda: RankPool2d(2, stride=2)).to(device)
    # model_rank.load_state_dict(torch.load("checkpoints/rankpool.pth"))

    # model_mixed = ResNet18(lambda: RobustMixedPool2d(2, stride=2)).to(device)
    # model_mixed.load_state_dict(torch.load("checkpoints/robust_mixed.pth"))

    # ---- Robustness evaluation ----
    # models = {"RankPool": model_rank, "RobustMixed": model_mixed}
    # ratios = [0.0, 0.25, 0.50, 0.75]

    # print("\n=== Feature Map Occlusion ===")
    # for ratio in ratios:
    #     for name, m in models.items():
    #         acc = test_occlusion_feature(m, test_loader, ratio, device)
    #         print(f"  {name:20s}  ratio={ratio:.2f}  acc={acc:.2f}%")

    # print("\n=== Weight Analysis ===")
    # weight_analysis()

    print("Edit this script to load your trained checkpoints and run evaluations.")


if __name__ == "__main__":
    main()
