"""
Utility functions: occlusion robustness testing and weight analysis.
"""

import torch
import torch.nn.functional as F


def test_occlusion_feature(model, test_loader, block_ratio, device="cuda"):
    """Test robustness with block occlusion on layer1 feature maps."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            x = F.relu(model.bn1(model.conv1(data)))
            x = model.layer1(x)
            B, C, H, W = x.shape
            occ_h = int(H * block_ratio)
            occ_w = int(W * block_ratio)
            if occ_h > 0 and occ_w > 0:
                y = torch.randint(0, max(1, H - occ_h), (B,), device=device)
                xp = torch.randint(0, max(1, W - occ_w), (B,), device=device)
                mask = torch.ones_like(x)
                for i in range(B):
                    mask[i, :, y[i]:y[i]+occ_h, xp[i]:xp[i]+occ_w] = 0
                x = x * mask
            x = model.pool(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            x = model.avgpool(x)
            pred = model.fc(x.flatten(1)).argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return correct / total * 100


def test_occlusion_image(model, test_loader, block_ratio, device="cuda"):
    """Test robustness with block occlusion on raw images."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            B, C, H, W = data.shape
            occ_h = int(H * block_ratio)
            occ_w = int(W * block_ratio)
            if occ_h > 0 and occ_w > 0:
                y = torch.randint(0, max(1, H - occ_h), (B,), device=device)
                xp = torch.randint(0, max(1, W - occ_w), (B,), device=device)
                mask = torch.ones_like(data)
                for i in range(B):
                    mask[i, :, y[i]:y[i]+occ_h, xp[i]:xp[i]+occ_w] = 0
                data = data * mask
            pred = model(data).argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return correct / total * 100


def test_occlusion_multi(model, test_loader, num_blocks, block_ratio, device="cuda"):
    """Test robustness with multiple random block occlusions on images."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            B, C, H, W = data.shape
            occ_h = int(H * block_ratio)
            occ_w = int(W * block_ratio)
            mask = torch.ones_like(data)
            if occ_h > 0 and occ_w > 0:
                for _ in range(num_blocks):
                    y = torch.randint(0, max(1, H - occ_h), (B,), device=device)
                    xp = torch.randint(0, max(1, W - occ_w), (B,), device=device)
                    for i in range(B):
                        mask[i, :, y[i]:y[i]+occ_h, xp[i]:xp[i]+occ_w] = 0
            pred = (model(data * mask)).argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return correct / total * 100


def weight_analysis():
    """Analyse pooling weight distribution for canonical input vectors."""
    import torch.nn.functional as F

    cases = [
        ("Uniform:  [3,3,3,3]",   [3.0, 3.0, 3.0, 3.0]),
        ("Near:     [2,3,3,2]",   [2.0, 3.0, 3.0, 2.0]),
        ("Gradual:  [1,2,3,4]",   [1.0, 2.0, 3.0, 4.0]),
        ("Extreme:  [1,1,1,100]", [1.0, 1.0, 1.0, 100.0]),
    ]

    for label, vals in cases:
        v = torch.tensor(vals)
        w_rank = v.abs().pow(2)
        w_rank = w_rank / w_rank.sum()
        w_avg = torch.ones(4) / 4
        alpha = 0.3891
        w_mixed = (1 - alpha) * w_rank + alpha * w_avg

        out_max = v.max().item()
        out_avg = v.mean().item()
        out_rank = (w_rank * v).sum().item()
        out_mixed = (w_mixed * v).sum().item()

        print(f"\n  {label}:")
        print(f"    MaxPool     = {out_max:.4f}   (weight: [0,0,0,1])")
        print(f"    AvgPool     = {out_avg:.4f}   (weight: {w_avg.tolist()})")
        print(f"    RankPool    = {out_rank:.4f}   (weight: {[f'{x:.3f}' for x in w_rank.tolist()]})")
        print(f"    RobustMixed = {out_mixed:.4f}   (weight: {[f'{x:.3f}' for x in w_mixed.tolist()]})")
