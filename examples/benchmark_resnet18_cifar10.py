
Benchmark RankPool vs MaxPool vs AvgPool on CIFAR-10 with ResNet-18.

Reproduces the key results from the paper. Expected results (seed=42)

    Pooling     Accuracy    Occlusion Drop Rate (Test A)
    -------     --------    --------------------------
    MaxPool     91.76%      42.2%
    AvgPool     91.78%      33.1%
    RankPool    91.80%      34.1%

Usage
    python benchmark_resnet18_cifar10.py


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import copy
import random
import numpy as np

from rankpool import RankPool2d


def set_seed(seed=42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ======================================================================
# Network
# ======================================================================

class BasicBlock(nn.Module)
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1)
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet18(nn.Module)
    
    ResNet-18 with pool after layer1.
    Architecture conv1 → layer1 → pool → layer2 → layer3 → layer4 → fc
    
    def __init__(self, pool_layer, num_classes=10)
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.pool = pool_layer()
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_ch, out_ch, num_blocks, stride)
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, num_blocks)
            layers.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(layers)

    def forward(self, x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.pool(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return self.fc(x.flatten(1))


# ======================================================================
# Training
# ======================================================================

def train_epoch(model, loader, optimizer, criterion, device)
    model.train()
    total_loss, correct, total = 0, 0, 0
    for data, target in loader
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()  data.size(0)
        correct += (output.argmax(1) == target).sum().item()
        total += target.size(0)
    return total_loss  total, correct  total  100


@torch.no_grad()
def evaluate(model, loader, criterion, device)
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for data, target in loader
        data, target = data.to(device), target.to(device)
        output = model(data)
        total_loss += criterion(output, target).item()  data.size(0)
        correct += (output.argmax(1) == target).sum().item()
        total += target.size(0)
    return total_loss  total, correct  total  100


def train_and_eval(model, train_loader, test_loader, device,
                   epochs=30, patience=5, lr=0.001, wd=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    best_acc, best_epoch, wait = 0, 0, 0
    for epoch in range(epochs)
        start = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device)
        scheduler.step()
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        elapsed = time.time() - start
        if test_acc  best_acc
            best_acc, best_epoch, wait = test_acc, epoch + 1, 0
        else
            wait += 1
        print(f  Epoch {epoch+12d}  
              fTrainAcc {train_acc.2f}%  TestAcc {test_acc.2f}%  
              fTime {elapsed.1f}s{'  ' if wait == 0 and test_acc == best_acc else ''})
        if wait = patience
            break
    return best_acc, best_epoch


# ======================================================================
# Occlusion robustness test
# ======================================================================

@torch.no_grad()
def test_occlusion(model, test_loader, device, block_ratio=0.5)
    Block occlusion on feature maps after layer1.
    model.eval()
    correct, total = 0, 0
    for data, target in test_loader
        data, target = data.to(device), target.to(device)
        x = F.relu(model.bn1(model.conv1(data)))
        x = model.layer1(x)
        B, C, H, W = x.shape
        occ_h, occ_w = int(H  block_ratio), int(W  block_ratio)
        if occ_h  0 and occ_w  0
            y = torch.randint(0, max(1, H - occ_h), (B,), device=device)
            xp = torch.randint(0, max(1, W - occ_w), (B,), device=device)
            mask = torch.ones_like(x)
            for i in range(B)
                mask[i, , y[i]y[i]+occ_h, xp[i]xp[i]+occ_w] = 0
            x = x  mask
        x = model.pool(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
        pred = model.fc(x.flatten(1)).argmax(1)
        correct += (pred == target).sum().item()
        total += target.size(0)
    return correct  total  100


# ======================================================================
# Main
# ======================================================================

if __name__ == __main__
    SEED = 42
    EPOCHS = 30
    PATIENCE = 5
    LR = 0.001
    WD = 1e-4
    BATCH = 128

    device = torch.device(cuda if torch.cuda.is_available() else cpu)
    print(fDevice {device})
    if device.type == cuda
        print(fGPU {torch.cuda.get_device_name(0)})

    # Data
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    train_data = datasets.CIFAR10(.data, train=True,
                                  download=True, transform=transform_train)
    test_data = datasets.CIFAR10(.data, train=False,
                                 download=True, transform=transform_test)
    train_loader = DataLoader(train_data, batch_size=BATCH,
                              shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=BATCH  2, num_workers=2)

    # Experiments
    pools = {
        MaxPool  lambda nn.MaxPool2d(2, stride=2),
        AvgPool  lambda nn.AvgPool2d(2, stride=2),
        RankPool lambda RankPool2d(2, stride=2),
    }

    results = {}
    models = {}

    for name, pool_fn in pools.items()
        set_seed(SEED)
        print(fn{'='50})
        print(fTraining {name})
        print(f{'='50})
        model = ResNet18(pool_fn).to(device)
        best_acc, best_epoch = train_and_eval(
            model, train_loader, test_loader, device,
            epochs=EPOCHS, patience=PATIENCE, lr=LR, wd=WD)
        results[name] = {acc best_acc, epoch best_epoch}
        models[name] = copy.deepcopy(model.cpu())

    # Occlusion robustness
    print(fn{'='50})
    print(Occlusion Robustness Test (feature map, 50% block))
    print(f{'='50})
    for name in pools
        m = models[name].to(device)
        clean = results[name][acc]
        occ = test_occlusion(m, test_loader, device, block_ratio=0.5)
        drop = clean - occ
        results[name][occ] = occ
        results[name][drop] = drop

    # Summary
    print(fn{'='50})
    print(Summary)
    print(f{'='50})
    print(f  {'Pooling'12} {'Accuracy'10} {'Occlusion'12} {'Drop Rate'10})
    print(f  {'-'46})
    for name, r in results.items()
        drop_rate = r['drop']  r['acc']  100 if r['acc']  0 else 0
        print(f  {name12} {r['acc']9.2f}% {r['occ']11.2f}% {drop_rate9.1f}%)
