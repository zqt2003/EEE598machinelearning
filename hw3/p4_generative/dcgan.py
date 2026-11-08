"""DCGAN generator and discriminator (Radford et al., 2016), as in the PyTorch DCGAN tutorial, for any
image size 32, 64, 128 or 256 (the tutorial's 64x64 network is the special case size=64).

Generator:     z (nz x 1 x 1) -> ConvTranspose 4x4 -> BN -> ReLU -> ... (x2 resolution per block) -> Tanh
Discriminator: image -> Conv 4x4 stride 2 -> LeakyReLU(0.2) -> ... -> one logit
"""
import math

import torch.nn as nn
from torch.nn.utils import spectral_norm


def _channels(n_blocks, base, cap=16):
    return [base * min(2 ** (n_blocks - 1 - i), cap) for i in range(n_blocks)]


class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, size=64):
        super().__init__()
        n = int(math.log2(size)) - 2                     # 4x4 -> size: n doublings ... minus the last layer
        ch = _channels(n, ngf)                           # size=64: [512, 256, 128, 64] for ngf=64
        layers = [nn.ConvTranspose2d(nz, ch[0], 4, 1, 0, bias=False), nn.BatchNorm2d(ch[0]), nn.ReLU(True)]
        for a, b in zip(ch[:-1], ch[1:]):
            layers += [nn.ConvTranspose2d(a, b, 4, 2, 1, bias=False), nn.BatchNorm2d(b), nn.ReLU(True)]
        layers += [nn.ConvTranspose2d(ch[-1], 3, 4, 2, 1, bias=False), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        self.nz = nz

    def forward(self, z):
        return self.net(z.view(z.size(0), -1, 1, 1))


class Discriminator(nn.Module):
    def __init__(self, ndf=64, size=64, spectral=False):
        super().__init__()
        n = int(math.log2(size)) - 2
        ch = list(reversed(_channels(n, ndf)))           # size=64: [64, 128, 256, 512]
        sn = spectral_norm if spectral else (lambda m: m)
        layers = [sn(nn.Conv2d(3, ch[0], 4, 2, 1, bias=False)), nn.LeakyReLU(0.2, True)]
        for a, b in zip(ch[:-1], ch[1:]):
            layers += [sn(nn.Conv2d(a, b, 4, 2, 1, bias=False))]
            if not spectral:                             # spectral norm already controls D; no BatchNorm then
                layers += [nn.BatchNorm2d(b)]
            layers += [nn.LeakyReLU(0.2, True)]
        layers += [sn(nn.Conv2d(ch[-1], 1, 4, 1, 0, bias=False))]   # 4x4 -> 1x1 logit (no Sigmoid: we use
        self.net = nn.Sequential(*layers)                            # BCEWithLogitsLoss, which is more stable)

    def forward(self, x):
        return self.net(x).view(-1)


def weights_init(m):
    """DCGAN paper: all weights from N(0, 0.02); BatchNorm scale from N(1, 0.02), bias 0."""
    name = m.__class__.__name__
    if "Conv" in name and hasattr(m, "weight"):
        nn.init.normal_(m.weight.data if not hasattr(m, "weight_orig") else m.weight_orig.data, 0.0, 0.02)
    elif "BatchNorm" in name:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
