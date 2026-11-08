"""Does the GAN draw squares?  A simple, objective "squareness" score for 64x64 images.

python squares_score.py --real data/squares_1k --ckpt runs/dcgan_squares_1k/checkpoint.pt --out runs/dcgan_squares_1k

For every image: threshold the non-black pixels, keep the largest blob and test it:
    fill   = blob area / area of its minimum-area rotated rectangle  (1.0 for a perfect square, ~0.785 for a disk)
    aspect = short side / long side of that rectangle                 (1.0 for a square)
    color  = std of the blob's color                                  (a square has ONE color -> small)
An image counts as "a square" if there is exactly one blob that does not touch the border, the background
around it is black, fill >= 0.85, aspect >= 0.85 and color std <= 0.08. (A flat gray image would otherwise
pass as one image-sized "square", which is what an untrained GAN produces.)
The same test on the REAL images gives the reference (should be close to 100%).
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy import ndimage  # noqa: E402

from common import ImageDir  # noqa: E402


def min_area_rect(ys, xs):
    """Smallest rotated bounding rectangle of a pixel set, by trying angles 0..89 degrees."""
    pts = np.stack([xs, ys], 1).astype(np.float64)
    best = (np.inf, 0, 0)
    for a in np.deg2rad(np.arange(0, 90, 1.0)):
        r = pts @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        w, h = np.ptp(r[:, 0]) + 1, np.ptp(r[:, 1]) + 1
        if w * h < best[0]:
            best = (w * h, w, h)
    return best


def score(img):
    """img: (H, W, 3) in [0, 1] -> dict of measurements."""
    fg = img.max(axis=2) > 0.25
    lab, n = ndimage.label(fg)
    if n == 0:
        return dict(blobs=0, fill=0.0, aspect=0.0, color_std=1.0, dark_bg=False, is_square=False)
    sizes = ndimage.sum(fg, lab, range(1, n + 1))
    big = lab == (1 + int(np.argmax(sizes)))
    n_real = int((sizes >= 0.1 * sizes.max()).sum())                 # ignore specks smaller than 10% of the blob
    ys, xs = np.nonzero(big)
    area, w, h = min_area_rect(ys, xs)
    fill, aspect = big.sum() / area, min(w, h) / max(w, h)
    interior = ndimage.binary_erosion(big, iterations=2)
    color_std = float(img[interior].std(axis=0).mean()) if interior.sum() > 5 else 1.0
    touches = big[0].any() or big[-1].any() or big[:, 0].any() or big[:, -1].any()
    outside = ~ndimage.binary_dilation(big, iterations=2)
    dark_bg = bool(not touches and img.max(axis=2)[outside].mean() < 0.1)
    return dict(blobs=n_real, fill=float(fill), aspect=float(aspect), color_std=color_std, dark_bg=dark_bg,
                is_square=bool(n_real == 1 and dark_bg and fill >= 0.85 and aspect >= 0.85 and color_std <= 0.08))


def evaluate(images):
    return pd.DataFrame([score(im) for im in images])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True, help="folder of real square images")
    ap.add_argument("--ckpt", default=None, help="GAN checkpoint (or use --fake-dir)")
    ap.add_argument("--fake-dir", default=None)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    to_np = lambda t: ((t.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).numpy()   # noqa: E731
    real_ds = ImageDir(args.real, 64)
    real = to_np(torch.stack([real_ds[i] for i in range(min(args.n, len(real_ds)))]))
    if args.ckpt:
        from models_io import load_generator
        G, z_dim, _ = load_generator(args.ckpt)
        with torch.no_grad():
            fake = to_np(G(torch.randn(args.n, z_dim, generator=torch.Generator().manual_seed(0))))
    else:
        fake_ds = ImageDir(args.fake_dir, 64)
        fake = to_np(torch.stack([fake_ds[i] for i in range(min(args.n, len(fake_ds)))]))
    r, f = evaluate(real), evaluate(fake)
    summary = pd.DataFrame({k: [v.is_square.mean(), (v.blobs == 1).mean(), v.dark_bg.mean(), v.fill.median(),
                                v.aspect.median(), v.color_std.median()] for k, v in [("real", r), ("generated", f)]},
                           index=["frac judged squares", "frac one blob", "frac black background", "median fill",
                                  "median aspect", "median color std"]).T
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))
    os.makedirs(args.out, exist_ok=True)
    summary.to_csv(os.path.join(args.out, "squares_score.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    for ax, col, xl in zip(axes, ["fill", "aspect", "color_std"],
                           ["fill of min-area rectangle", "aspect ratio (short / long side)", "color std inside blob"]):
        bins = np.linspace(0, 1 if col != "color_std" else 0.3, 31)
        ax.hist(r[col], bins, alpha=0.6, label="real", density=True)
        ax.hist(f[col], bins, alpha=0.6, label="generated", density=True)
        ax.set_xlabel(xl); ax.set_ylabel("density"); ax.legend(); ax.grid(alpha=0.3)
    axes[0].set_title("Squareness: fill"); axes[1].set_title("Squareness: aspect"); axes[2].set_title("Single color")
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "squares_score.png"), dpi=130)


if __name__ == "__main__":
    main()
