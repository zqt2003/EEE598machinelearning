"""Shared code for Problem 4: data loading, feature extractors, losses and metrics."""
import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.models import VGG16_Weights, vgg16

TAP_IDX = {"relu1_2": 3, "relu2_2": 8, "relu3_3": 15, "relu4_3": 22}
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# --------------------------------------------------------------------------- data
def load_image(path):
    return torch.from_numpy(np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)


def read_pairs(root, split):
    """Returns {clean_id: [(k, r0, turb_path), ...]} for one split."""
    root = Path(root)
    table = defaultdict(list)
    with open(root / "pairs.csv") as f:
        for r in csv.DictReader(f):
            if r["split"] == split:
                table[r["id"]].append((int(r["k"]), float(r["r0"]), root / "turb" / f"{r['id']}_{r['k']}.png"))
    return {cid: sorted(v) for cid, v in sorted(table.items())}


class PairDataset(torch.utils.data.Dataset):
    """Yields (clean, turbulent_a, turbulent_b): two random turbulence realizations of one image."""

    def __init__(self, root, split="train"):
        self.root = Path(root)
        self.table = read_pairs(root, split)
        self.ids = list(self.table)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        cid = self.ids[i]
        clean = load_image(self.root / "clean" / f"{cid}.png")
        turbs = self.table[cid]
        ta = load_image(random.choice(turbs)[2])
        tb = load_image(random.choice(turbs)[2])
        if random.random() < 0.5:                      # same flip for all three views
            clean, ta, tb = clean.flip(-1), ta.flip(-1), tb.flip(-1)
        return clean, ta, tb


def content_edit(x, donor, patch, gen):
    """Paste a patch x patch square from `donor` into `x` at a random location (same place in both)."""
    out = x.clone()
    _, _, h, w = x.shape
    for b in range(x.shape[0]):
        y0 = int(torch.randint(0, h - patch + 1, (1,), generator=gen))
        x0 = int(torch.randint(0, w - patch + 1, (1,), generator=gen))
        out[b, :, y0:y0 + patch, x0:x0 + patch] = donor[b, :, y0:y0 + patch, x0:x0 + patch]
    return out


# --------------------------------------------------------------- building blocks
class BlurPool(nn.Module):
    """Anti-aliased downsampling (Zhang, ICML 2019): low-pass with a [1,2,1]^2 filter, then stride 2."""

    def __init__(self, stride=2):
        super().__init__()
        k = torch.tensor([1.0, 2.0, 1.0])
        self.register_buffer("k", (k[:, None] * k[None, :] / 16.0)[None, None])
        self.stride = stride

    def forward(self, x):
        c = x.shape[1]
        x = F.pad(x, (1, 1, 1, 1), mode="reflect")
        return F.conv2d(x, self.k.expand(c, 1, 3, 3), stride=self.stride, groups=c)


class MaxBlurPool(nn.Module):
    """Drop-in replacement for MaxPool2d(2, 2): dense max, then anti-aliased subsampling."""

    def __init__(self):
        super().__init__()
        self.max = nn.MaxPool2d(kernel_size=2, stride=1)
        self.blur = BlurPool(2)

    def forward(self, x):
        return self.blur(self.max(x))


class VGGTrunk(nn.Module):
    """Frozen ImageNet VGG-16 that returns the activations at the requested ReLU taps."""

    def __init__(self, taps=("relu1_2", "relu2_2", "relu3_3", "relu4_3"), antialias=False, pretrained=True):
        super().__init__()
        self.taps = {TAP_IDX[t]: t for t in taps}
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        layers = list(vgg16(weights=weights).features[: max(self.taps) + 1])
        if antialias:
            layers = [MaxBlurPool() if isinstance(m, nn.MaxPool2d) else m for m in layers]
        self.layers = nn.Sequential(*layers)
        self.register_buffer("mean", _MEAN.clone())
        self.register_buffer("std", _STD.clone())
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def train(self, mode=True):          # the trunk always stays frozen / in eval mode
        return super().train(False)

    def forward(self, x):
        x = (x - self.mean) / self.std
        out = {}
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.taps:
                out[self.taps[i]] = x
        return out


class TurbHead(nn.Module):
    """Trainable head: fuses relu2_2 (anti-alias downsampled) and relu3_3 into a D-dim unit-norm map at 1/4 res."""

    def __init__(self, dim=128):
        super().__init__()
        self.down = BlurPool(2)
        self.net = nn.Sequential(
            nn.Conv2d(128 + 256, 256, 3, padding=1, padding_mode="reflect"), nn.ReLU(inplace=True),
            nn.Conv2d(256, dim, 3, padding=1, padding_mode="reflect"),
        )

    def forward(self, taps):
        z = self.net(torch.cat([self.down(taps["relu2_2"]), taps["relu3_3"]], dim=1))
        return F.normalize(z, dim=1)


class TurbFeature(nn.Module):
    """The proposed turbulence-robust feature extractor: (anti-aliased) frozen VGG trunk + trained head."""

    def __init__(self, antialias=True, dim=128, pretrained=True):
        super().__init__()
        self.trunk = VGGTrunk(("relu2_2", "relu3_3"), antialias=antialias, pretrained=pretrained)
        self.head = TurbHead(dim)

    def forward(self, x):
        with torch.no_grad():
            taps = self.trunk(x)
        return self.head(taps)


# ------------------------------------------------------------------------- losses
def dense_infonce(za, zb, n_loc=64, grid_stride=4, tau=0.1):
    """Dense contrastive loss. The feature at location p of view a must match location p of view b,
    and NOT any other sampled location of the same image or of the other images in the batch.
    The positive pair gives turbulence invariance; the negatives stop the features from collapsing."""
    B, D, h, w = za.shape
    ys = torch.arange(grid_stride // 2, h, grid_stride, device=za.device)
    xs = torch.arange(grid_stride // 2, w, grid_stride, device=za.device)
    grid = (ys[:, None] * w + xs[None, :]).flatten()                       # candidate locations
    idx = torch.stack([grid[torch.randperm(len(grid), device=za.device)[:n_loc]] for _ in range(B)])
    qa = torch.gather(za.flatten(2), 2, idx[:, None, :].expand(-1, D, -1)).permute(0, 2, 1).reshape(-1, D)
    qb = torch.gather(zb.flatten(2), 2, idx[:, None, :].expand(-1, D, -1)).permute(0, 2, 1).reshape(-1, D)
    logits = qa @ qb.t() / tau
    target = torch.arange(len(qa), device=za.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))


def invariance_only(za, zb):
    """Ablation: only pull the two views together (no negatives) -- expected to collapse."""
    return ((za - zb) ** 2).sum(1).mean()


# -------------------------------------------------------------- distances/metrics
def feature_distance(fa, fb, normalize=True):
    """Per-image distance averaged over a list of feature maps (LPIPS-style channel normalization)."""
    total = 0.0
    for a, b in zip(fa, fb):
        if normalize:
            a, b = F.normalize(a, dim=1), F.normalize(b, dim=1)
            total = total + ((a - b) ** 2).sum(1).mean(dim=(1, 2))
        else:
            total = total + ((a - b) ** 2).mean(dim=(1, 2, 3))
    return total / len(fa)


def distance_map(fa, fb, normalize=True):
    """Spatial map of squared feature change (for qualitative figures). Uses the first feature map."""
    a, b = fa[0], fb[0]
    if normalize:
        a, b = F.normalize(a, dim=1), F.normalize(b, dim=1)
        return ((a - b) ** 2).sum(1)
    return ((a - b) ** 2).mean(1)


def auc_smaller(pos, neg):
    """P(pos < neg): how often the turbulent copy is judged closer than the negative (1 = perfect)."""
    pos, neg = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float((pos < neg).mean() + 0.5 * (pos == neg).mean())
