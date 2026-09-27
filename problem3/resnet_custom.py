"""ResNet built from BasicBlocks, configurable depth per stage (trained from scratch, no pretrained weights).

ResNet-34 = [3, 4, 6, 3] BasicBlocks  -> 1 stem conv + 32 block convs + 1 fc = 34 layers
ResNet-36 = [3, 4, 7, 3] BasicBlocks  -> one extra block in stage 3 (256 channels, 14x14) = 36 layers
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIGS = {
    "resnet34": (3, 4, 6, 3),
    "resnet36": (3, 4, 7, 3),      # proposed: extra block in stage 3
    "resnet36_s4": (3, 4, 6, 4),   # alternative: extra block in stage 4 (for an ablation)
}


class _ArcGateFn(torch.autograd.Function):
    """f(x) = x * (1/2 + arctan(x)/pi), with a hand-written backward that stores only x (saves memory)."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * (0.5 + torch.atan(x) / math.pi)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        # f'(x) = 1/2 + arctan(x)/pi + x / (pi * (1 + x^2))
        return grad_out * (0.5 + torch.atan(x) / math.pi + x / (math.pi * (1 + x * x)))


class ArcGate(nn.Module):
    """ArcGate activation: the input is gated by an arctan-shaped 'probability' instead of a hard 0/1 (ReLU)
    or a logistic sigmoid (Swish/SiLU).  Smooth, lets small negative values through, saturates at -1/pi for
    large negative inputs, and is ~x - 1/pi for large positive inputs."""

    def __init__(self, inplace=False):          # 'inplace' accepted so it can replace nn.ReLU(inplace=True)
        super().__init__()

    def forward(self, x):
        return _ArcGateFn.apply(x)


ACTIVATIONS = {"relu": lambda: nn.ReLU(inplace=True), "arcgate": ArcGate}


class BasicBlock(nn.Module):
    """Two 3x3 convs + skip connection: out = act(F(x) + shortcut(x)), act = ReLU or ArcGate."""

    def __init__(self, in_ch, out_ch, stride=1, act="relu"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = ACTIVATIONS[act]()
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm2d(out_ch))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class KANLinear(nn.Module):
    """Kolmogorov-Arnold layer (Liu et al., 2024): every input->output connection has its own learnable 1-D
    function phi_ij(x) = w_base * SiLU(x) + sum_k c_ijk * B_k(x), with cubic B-splines B_k on a fixed grid.
    output_j = sum_i phi_ij(x_i).  Counts as ONE layer (it replaces the single fc layer)."""

    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, grid_range=(-3.0, 3.0)):
        super().__init__()
        self.in_features, self.out_features, self.order = in_features, out_features, spline_order
        h = (grid_range[1] - grid_range[0]) / grid_size
        knots = torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32) * h + grid_range[0]
        self.register_buffer("grid", knots.expand(in_features, -1).contiguous())      # (in, G + 2k + 1)
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.randn(out_features, in_features, grid_size + spline_order) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.base_weight, a=5 ** 0.5)

    def b_splines(self, x):
        """Cox-de Boor recursion. x: (B, in) -> (B, in, grid_size + order) basis values."""
        g, x = self.grid, x.unsqueeze(-1)
        bases = ((x >= g[:, :-1]) & (x < g[:, 1:])).to(x.dtype)
        for k in range(1, self.order + 1):
            bases = ((x - g[:, :-(k + 1)]) / (g[:, k:-1] - g[:, :-(k + 1)]) * bases[..., :-1]
                     + (g[:, k + 1:] - x) / (g[:, k + 1:] - g[:, 1:-k]) * bases[..., 1:])
        return bases

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):   # splines in full precision
            x = x.float()
            base = F.linear(F.silu(x), self.base_weight)
            spline = F.linear(self.b_splines(x).flatten(1), self.spline_weight.flatten(1))
            return base + spline + self.bias


class ResNet(nn.Module):
    def __init__(self, blocks=(3, 4, 6, 3), widths=(64, 128, 256, 512), num_classes=1000, act="relu", head="linear"):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64),
                                  ACTIVATIONS[act](), nn.MaxPool2d(3, 2, 1))
        stages, in_ch = [], 64
        for i, (n, w) in enumerate(zip(blocks, widths)):
            stride = 1 if i == 0 else 2
            stages.append(nn.Sequential(BasicBlock(in_ch, w, stride, act),
                                        *[BasicBlock(w, w, 1, act) for _ in range(n - 1)]))
            in_ch = w
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        if head == "kan":        # LayerNorm puts the features into the spline grid's range [-3, 3]
            self.fc = nn.Sequential(nn.LayerNorm(in_ch), KANLinear(in_ch, num_classes))
        else:
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
    return sum(isinstance(m, (nn.Conv2d, nn.Linear, KANLinear)) and "shortcut" not in n for n, m in model.named_modules())
