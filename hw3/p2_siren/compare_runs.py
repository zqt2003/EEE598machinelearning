"""Compare several fit_image.py runs: PSNR curves, reconstructions with a zoomed crop, and error spectra.

python compare_runs.py runs/sine runs/parawave runs/relu --out comparison
Writes: psnr_curves.png, reconstructions.png, error_spectrum.png, summary.csv  (in --out)

The error spectrum is the radially averaged power spectrum of (target - reconstruction): it shows WHICH
spatial frequencies a model fails to fit. A curve that rises at high frequency = fine detail is missing.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--labels", default=None, help="comma-separated labels (default: folder names)")
ap.add_argument("--crop", type=int, default=128, help="size of the zoomed crop (pixels)")
ap.add_argument("--out", default="comparison")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
labels = args.labels.split(",") if args.labels else [os.path.basename(r.rstrip("/")) for r in args.runs]


def load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def radial_spectrum(err):
    """Radially averaged power spectrum of a (H, W) array; returns frequency (cycles/pixel) and power."""
    f = np.fft.fftshift(np.fft.fft2(err - err.mean()))
    p = np.abs(f) ** 2
    h, w = err.shape
    yy, xx = np.indices((h, w))
    r = np.hypot(yy - h // 2, xx - w // 2).astype(int)
    radial = np.bincount(r.ravel(), p.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    n = min(h, w) // 2
    return np.arange(n) / min(h, w), radial[:n]


target = load(os.path.join(args.runs[0], "target.png"))
H = target.shape[0]
c = args.crop
y0, x0 = H // 2 - c // 2, H // 2 - c // 2                 # zoom on the image center

# ---- PSNR curves ----
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
rows = []
for run, lab in zip(args.runs, labels):
    h = pd.read_csv(os.path.join(run, "history.csv"))
    axes[0].plot(h.step, h.psnr, label=lab)
    axes[1].plot(h.time_s, h.psnr, label=lab)
    s = json.load(open(os.path.join(run, "summary.json")))
    rows.append(dict(run=lab, **s))
for ax, xl in zip(axes, ["training step", "training time (s)"]):
    ax.set_xlabel(xl); ax.set_ylabel("PSNR (dB)"); ax.grid(alpha=0.3); ax.legend()
axes[0].set_title("Image-fitting PSNR vs. step"); axes[1].set_title("Image-fitting PSNR vs. wall-clock time")
plt.tight_layout(); plt.savefig(os.path.join(args.out, "psnr_curves.png"), dpi=140); plt.close()
pd.DataFrame(rows).to_csv(os.path.join(args.out, "summary.csv"), index=False)

# ---- reconstructions (full + zoom) ----
n = len(args.runs) + 1
fig, axes = plt.subplots(2, n, figsize=(4 * n, 8.3))
recons = [load(os.path.join(r, "recon_final.png")) for r in args.runs]
for j, (im, t) in enumerate(zip([target] + recons, ["Ground truth"] + [
        f"{lab}: {r['final_psnr']:.2f} dB" for lab, r in zip(labels, rows)])):
    axes[0, j].imshow(im); axes[0, j].set_title(t)
    axes[0, j].add_patch(plt.Rectangle((x0, y0), c, c, fill=False, ec="yellow", lw=1.5))
    axes[1, j].imshow(im[y0:y0 + c, x0:x0 + c], interpolation="nearest")
    axes[1, j].set_title(f"zoom ({c}x{c} px)")
    for ax in axes[:, j]:
        ax.set_xlabel("x (pixels)"); ax.set_ylabel("y (pixels)")
plt.tight_layout(); plt.savefig(os.path.join(args.out, "reconstructions.png"), dpi=120); plt.close()

# ---- error spectrum ----
plt.figure(figsize=(7, 4.8))
for im, lab in zip(recons, labels):
    f, p = radial_spectrum((target - im).mean(axis=2))
    plt.semilogy(f[1:], p[1:], label=lab)
plt.xlabel("spatial frequency (cycles / pixel)"); plt.ylabel("error power (log scale)")
plt.title("Which frequencies are missing? (spectrum of target - reconstruction)")
plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(args.out, "error_spectrum.png"), dpi=140)
print(pd.DataFrame(rows)[["run", "params", "final_psnr", "best_psnr", "train_time_s"]].to_string(index=False))
