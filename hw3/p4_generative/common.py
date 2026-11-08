"""Helpers shared by the GAN and diffusion scripts."""
import copy
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torchvision.transforms as T  # noqa: E402
import torchvision.utils as vutils  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


class ImageDir(Dataset):
    """Every image under `root` (any sub-folders), resized and center-cropped to `size`, scaled to [-1, 1].
    flip=True adds random horizontal flips (cheap augmentation that is safe for faces and animals)."""

    def __init__(self, root, size=64, flip=False):
        self.files = sorted(f for f in glob.glob(os.path.join(root, "**", "*"), recursive=True)
                            if f.lower().endswith(IMG_EXT))
        if not self.files:
            raise FileNotFoundError(f"no images found under {root}")
        tf = [T.Resize(size), T.CenterCrop(size)]
        if flip:
            tf.append(T.RandomHorizontalFlip())
        self.tf = T.Compose(tf + [T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with Image.open(self.files[i]) as im:
            return self.tf(im.convert("RGB"))


def save_grid(images, path, nrow=8, title=None):
    """images in [-1, 1] -> PNG grid (optionally with a matplotlib title)."""
    grid = vutils.make_grid(images.detach().float().cpu(), nrow=nrow, padding=2, normalize=True, value_range=(-1, 1))
    if title is None:
        vutils.save_image(grid, path)
        return
    plt.figure(figsize=(nrow * 1.1 + 1, grid.shape[1] / grid.shape[2] * nrow * 1.1 + 1))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.title(title); plt.xlabel("sample"); plt.ylabel("sample")
    plt.xticks([]); plt.yticks([])
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


class EMA:
    """Exponential moving average of a model's weights. Sampling from the EMA copy usually gives smoother,
    better images than the raw generator (used by ProGAN, StyleGAN and diffusion models)."""

    def __init__(self, model, decay=0.999):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay, self.n = decay, 0

    @torch.no_grad()
    def update(self, model):
        # warm-up: early on, follow the model closely (otherwise the first sample grids show the random init)
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.lerp_(p.detach(), 1 - d)
        for e, b in zip(self.model.buffers(), model.buffers()):
            e.copy_(b)


def diff_augment(x, policy="color,translation,cutout"):
    """DiffAugment (Zhao et al., NeurIPS 2020): the SAME random, differentiable augmentation is applied to real
    and fake images before the discriminator. Stops D from memorizing a small dataset (1-5k images)."""
    if "color" in policy:
        x = x + (torch.rand(x.size(0), 1, 1, 1, device=x.device) - 0.5)                       # brightness
        m = x.mean(dim=1, keepdim=True)
        x = (x - m) * (torch.rand(x.size(0), 1, 1, 1, device=x.device) * 2) + m                # saturation
        m = x.mean(dim=[1, 2, 3], keepdim=True)
        x = (x - m) * (torch.rand(x.size(0), 1, 1, 1, device=x.device) + 0.5) + m              # contrast
    if "translation" in policy:
        n, _, h, w = x.shape
        sh, sw = int(h * 0.125 + 0.5), int(w * 0.125 + 0.5)
        tx = torch.randint(-sh, sh + 1, (n, 1, 1), device=x.device)
        ty = torch.randint(-sw, sw + 1, (n, 1, 1), device=x.device)
        gb, gx, gy = torch.meshgrid(torch.arange(n, device=x.device), torch.arange(h, device=x.device),
                                    torch.arange(w, device=x.device), indexing="ij")
        gx = torch.clamp(gx + tx + 1, 0, h + 1)
        gy = torch.clamp(gy + ty + 1, 0, w + 1)
        x = F.pad(x, [1, 1, 1, 1])[gb, :, gx, gy].permute(0, 3, 1, 2)
    if "cutout" in policy:
        n, _, h, w = x.shape
        ch, cw = int(h * 0.5 + 0.5), int(w * 0.5 + 0.5)
        ox = torch.randint(0, h + (1 - ch % 2), (n, 1, 1), device=x.device)
        oy = torch.randint(0, w + (1 - cw % 2), (n, 1, 1), device=x.device)
        gb, gx, gy = torch.meshgrid(torch.arange(n, device=x.device), torch.arange(ch, device=x.device),
                                    torch.arange(cw, device=x.device), indexing="ij")
        gx = torch.clamp(gx + ox - ch // 2, 0, h - 1)
        gy = torch.clamp(gy + oy - cw // 2, 0, w - 1)
        mask = torch.ones(n, h, w, device=x.device)
        mask[gb, gx, gy] = 0
        x = x * mask.unsqueeze(1)
    return x


def plot_curves(curves, path, title, ylabel="loss", xlabel="iteration", smooth=50):
    """curves: {label: list of values}. Draws the raw values faintly and a moving average on top."""
    plt.figure(figsize=(9, 4.5))
    for lab, v in curves.items():
        v = torch.tensor(v, dtype=torch.float32)
        line, = plt.plot(v.numpy(), alpha=0.25)
        if len(v) > smooth:
            k = torch.ones(smooth) / smooth
            sm = torch.conv1d(v[None, None], k[None, None]).squeeze()
            plt.plot(range(smooth - 1, len(v)), sm.numpy(), color=line.get_color(), label=lab)
        else:
            line.set_label(lab)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title); plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()
