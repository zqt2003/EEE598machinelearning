"""FID and KID between real images and each model's samples, plus a nearest-neighbour (memorization) check.

    python fid.py --real data/animal256 --size 64 --fake samples/dcgan samples/progan samples/diffusion \
        --labels DCGAN ProGAN Diffusion --out results/fid

Lower is better for both. FID needs many images (use >= 5,000 samples) and is biased for small sets;
KID is unbiased and more reliable with a few thousand images, so report both. All models must be compared
at the SAME resolution (--size), so ProGAN's 128x128 samples are downsampled to 64x64 here.
Needs:  pip install torchmetrics torch-fidelity   (downloads the Inception weights once)
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torchvision.utils as vutils  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torchmetrics.image.fid import FrechetInceptionDistance  # noqa: E402
from torchmetrics.image.kid import KernelInceptionDistance  # noqa: E402

from common import ImageDir  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--real", required=True)
ap.add_argument("--fake", nargs="+", required=True)
ap.add_argument("--labels", nargs="+", default=None)
ap.add_argument("--size", type=int, default=64)
ap.add_argument("--max", type=int, default=5000)
ap.add_argument("--out", default="results/fid")
args = ap.parse_args()
labels = args.labels or [os.path.basename(f.rstrip("/")) for f in args.fake]
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(args.out, exist_ok=True)


def batches(folder):
    ds = ImageDir(folder, args.size)
    n = min(args.max, len(ds))
    dl = DataLoader(torch.utils.data.Subset(ds, range(n)), batch_size=100, num_workers=4)
    for x in dl:
        yield ((x + 1) / 2).clamp(0, 1)                  # [0, 1] floats (normalize=True below)


real_imgs = torch.cat(list(batches(args.real)))
rows = []
for folder, lab in zip(args.fake, labels):
    fake_imgs = torch.cat(list(batches(folder)))
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    kid = KernelInceptionDistance(subset_size=min(1000, len(fake_imgs), len(real_imgs)), normalize=True).to(device)
    for i in range(0, len(real_imgs), 100):
        fid.update(real_imgs[i:i + 100].to(device), real=True); kid.update(real_imgs[i:i + 100].to(device), real=True)
    for i in range(0, len(fake_imgs), 100):
        fid.update(fake_imgs[i:i + 100].to(device), real=False); kid.update(fake_imgs[i:i + 100].to(device), real=False)
    kid_mean, kid_std = kid.compute()
    rows.append(dict(model=lab, n_fake=len(fake_imgs), n_real=len(real_imgs), FID=fid.compute().item(),
                     KID_x1000=1000 * kid_mean.item(), KID_std_x1000=1000 * kid_std.item()))
    print(rows[-1], flush=True)

    # memorization check: the closest TRAINING image (pixel L2) to each of 8 samples
    q = fake_imgs[:8]
    d = torch.cdist(q.flatten(1), real_imgs.flatten(1))
    nn = real_imgs[d.argmin(1)]
    grid = vutils.make_grid(torch.cat([q, nn]), nrow=8, padding=2)
    plt.figure(figsize=(12, 3.6)); plt.imshow(grid.permute(1, 2, 0)); plt.xticks([])
    plt.yticks([args.size / 2, args.size * 1.5 + 2], ["generated", "nearest\ntraining image"])
    plt.xlabel("sample"); plt.title(f"{lab}: generated samples and their nearest training images (pixel L2)")
    plt.tight_layout(); plt.savefig(os.path.join(args.out, f"nearest_{lab}.png"), dpi=120); plt.close()

df = pd.DataFrame(rows)
df.to_csv(os.path.join(args.out, "fid_kid.csv"), index=False)
print(df.to_string(index=False))
