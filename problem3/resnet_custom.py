"""ResNet built from BasicBlocks, configurable depth per stage (trained from scratch, no pretrained weights).

ResNet-34 = [3, 4, 6, 3] BasicBlocks  -> 1 stem conv + 32 block convs + 1 fc = 34 layers
ResNet-36 = [3, 4, 7, 3] BasicBlocks  -> one extra block in stage 3 (256 channels, 14x14) = 36 layers
"""
import torch
import torch.nn as nn

CONFIGS = {
    "resnet34": (3, 4, 6, 3),
    "resnet36": (3, 4, 7, 3),      # proposed: extra block in stage 3
    "resnet36_s4": (3, 4, 6, 4),   # alternative: extra block in stage 4 (for an ablation)
}


class BasicBlock(nn.Module):
    """Two 3x3 convs + skip connection: out = ReLU(F(x) + shortcut(x))."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm2d(out_ch))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, blocks=(3, 4, 6, 3), widths=(64, 128, 256, 512), num_classes=1000):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1))
        stages, in_ch = [], 64
        for i, (n, w) in enumerate(zip(blocks, widths)):
            stride = 1 if i == 0 else 2
            stages.append(nn.Sequential(BasicBlock(in_ch, w, stride), *[BasicBlock(w, w) for _ in range(n - 1)]))
            in_ch = w
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        for m in self.modules():              # each residual block starts as an identity mapping
            if isinstance(m, BasicBlock):
                nn.init.zeros_(m.bn2.weight)

    def forward(self, x):
        return self.fc(torch.flatten(self.pool(self.stages(self.stem(x))), 1))


def count_layers(model):
    """Main-path weighted layers (convs + fc), excluding 1x1 shortcut convs, as in the ResNet paper."""
    return sum(isinstance(m, (nn.Conv2d, nn.Linear)) and "shortcut" not in n for n, m in model.named_modules())
