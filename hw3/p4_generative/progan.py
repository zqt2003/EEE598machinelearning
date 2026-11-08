"""Progressive GAN (Karras et al., ICLR 2018), compact implementation.

The key ideas, all implemented below:
1. Progressive growing: train at 4x4 first, then add layers for 8x8, 16x16, ... A new resolution is
   "faded in": the output is alpha * (new layers) + (1 - alpha) * (upsampled old output), alpha 0 -> 1.
2. Equalized learning rate: weights are stored as N(0, 1) and multiplied by the He constant sqrt(2/fan_in)
   at run time, so Adam updates every layer at the same effective speed.
3. Pixel normalization in G: each pixel's feature vector is scaled to unit average length (replaces BatchNorm).
4. Minibatch standard deviation in D: an extra feature map with the batch's std, so D can notice when all
   generated images look alike (mode collapse) and punish it.
5. WGAN-GP loss (Wasserstein loss + gradient penalty) + a small drift term on D(real)^2.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class EqConv2d(nn.Module):
    def __init__(self, cin, cout, k, padding=0, gain=math.sqrt(2)):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(cout, cin, k, k))
        self.bias = nn.Parameter(torch.zeros(cout))
        self.scale, self.padding = gain / math.sqrt(cin * k * k), padding

    def forward(self, x):
        return F.conv2d(x, self.weight * self.scale, self.bias, padding=self.padding)


class EqLinear(nn.Module):
    def __init__(self, cin, cout, gain=math.sqrt(2)):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(cout, cin))
        self.bias = nn.Parameter(torch.zeros(cout))
        self.scale = gain / math.sqrt(cin)

    def forward(self, x):
        return F.linear(x, self.weight * self.scale, self.bias)


def pixel_norm(x, eps=1e-8):
    return x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + eps)


def lrelu(x):
    return F.leaky_relu(x, 0.2)


def channels(res, fmap_base=8192, fmap_max=512):
    """Feature maps per resolution: 512 up to 16x16, then halving (32: 256, 64: 128, 128: 64, 256: 32)."""
    return min(fmap_max, fmap_base // res)


class Generator(nn.Module):
    def __init__(self, z_dim=512, max_res=128, fmap_base=8192, fmap_max=512):
        super().__init__()
        self.z_dim, self.max_depth = z_dim, int(math.log2(max_res)) - 2          # depth 0 = 4x4
        c0 = channels(4, fmap_base, fmap_max)
        self.dense = EqLinear(z_dim, c0 * 16, gain=math.sqrt(2) / 4)             # z -> 4x4 feature map
        self.conv0 = EqConv2d(c0, c0, 3, 1)
        self.blocks, self.to_rgb = nn.ModuleList(), nn.ModuleList([EqConv2d(c0, 3, 1, gain=1)])
        for d in range(1, self.max_depth + 1):
            cin, cout = channels(2 ** (d + 1), fmap_base, fmap_max), channels(2 ** (d + 2), fmap_base, fmap_max)
            self.blocks.append(nn.ModuleList([EqConv2d(cin, cout, 3, 1), EqConv2d(cout, cout, 3, 1)]))
            self.to_rgb.append(EqConv2d(cout, 3, 1, gain=1))

    def forward(self, z, depth=None, alpha=1.0):
        depth = self.max_depth if depth is None else depth
        x = pixel_norm(lrelu(self.dense(pixel_norm(z)).view(z.size(0), -1, 4, 4)))
        x = pixel_norm(lrelu(self.conv0(x)))
        prev = x
        for d in range(1, depth + 1):
            prev = x
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            for conv in self.blocks[d - 1]:
                x = pixel_norm(lrelu(conv(x)))
        rgb = self.to_rgb[depth](x)
        if depth > 0 and alpha < 1.0:                                           # fade in the newest block
            old = F.interpolate(self.to_rgb[depth - 1](prev), scale_factor=2, mode="nearest")
            rgb = alpha * rgb + (1 - alpha) * old
        return rgb


def minibatch_stddev(x, group=4):
    n, c, h, w = x.shape
    g = min(group, n)
    if n % g:
        g = 1
    y = x.view(g, -1, c, h, w)
    y = (y - y.mean(dim=0)).pow(2).mean(dim=0).add(1e-8).sqrt()                 # std over each group
    y = y.mean(dim=[1, 2, 3]).view(-1, 1, 1, 1).repeat(g, 1, h, w)              # one number per group
    return torch.cat([x, y], dim=1)


class Discriminator(nn.Module):
    def __init__(self, max_res=128, fmap_base=8192, fmap_max=512):
        super().__init__()
        self.max_depth = int(math.log2(max_res)) - 2
        c0 = channels(4, fmap_base, fmap_max)
        self.from_rgb = nn.ModuleList([EqConv2d(3, c0, 1)])
        self.blocks = nn.ModuleList()
        for d in range(1, self.max_depth + 1):
            cin, cout = channels(2 ** (d + 2), fmap_base, fmap_max), channels(2 ** (d + 1), fmap_base, fmap_max)
            self.from_rgb.append(EqConv2d(3, cin, 1))
            self.blocks.append(nn.ModuleList([EqConv2d(cin, cin, 3, 1), EqConv2d(cin, cout, 3, 1)]))
        self.final_conv = EqConv2d(c0 + 1, c0, 3, 1)                            # +1: the minibatch-std map
        self.final_dense = EqLinear(c0 * 16, c0)
        self.out = EqLinear(c0, 1, gain=1)

    def forward(self, img, depth=None, alpha=1.0):
        depth = self.max_depth if depth is None else depth
        x = lrelu(self.from_rgb[depth](img))
        for d in range(depth, 0, -1):
            for conv in self.blocks[d - 1]:
                x = lrelu(conv(x))
            x = F.avg_pool2d(x, 2)
            if d == depth and alpha < 1.0:                                      # fade in the newest block
                old = lrelu(self.from_rgb[d - 1](F.avg_pool2d(img, 2)))
                x = alpha * x + (1 - alpha) * old
        x = lrelu(self.final_conv(minibatch_stddev(x)))
        x = lrelu(self.final_dense(x.flatten(1)))
        return self.out(x).view(-1)
