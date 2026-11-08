"""Fit a coordinate MLP to one high-resolution image:  f_theta(x, y) -> (R, G, B).

Example (Sol GPU):
    python fit_image.py --image photo.jpg --size 1024 --act sine     --out runs/sine
    python fit_image.py --image photo.jpg --size 1024 --act parawave --out runs/parawave
    python fit_image.py --image photo.jpg --size 1024 --act relu     --out runs/relu --lr 1e-3
Quick CPU test:
    python fit_image.py --image photo.jpg --size 64 --hidden 64 --steps 50 --out runs/test

Writes to --out: history.csv (step, loss, PSNR, time), recon_final.png, recon_step*.png, summary.json,
model.pt, and with --derivatives a figure comparing image gradients/Laplacians with the network's.
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

from siren import CoordMLP, get_grid  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True)
ap.add_argument("--size", type=int, default=1024, help="the image is center-cropped to a square and resized to this")
ap.add_argument("--act", default="sine", help="sine | parawave | triangle | relu")
ap.add_argument("--hidden", type=int, default=512)
ap.add_argument("--hidden-layers", type=int, default=3)
ap.add_argument("--omega0", type=float, default=30.0, help="frequency factor of the periodic activations")
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--batch-pixels", type=int, default=0, help="random pixels per step (0 = whole image each step)")
ap.add_argument("--log-every", type=int, default=100)
ap.add_argument("--snapshots", default="100,500,1000", help="steps at which to save the reconstruction")
ap.add_argument("--derivatives", action="store_true", help="also compare gradient and Laplacian maps")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="runs/sine")
args = ap.parse_args()

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(args.out, exist_ok=True)

# ---- image -> (pixels, 3) in [-1, 1] ----
img = ImageOps.exif_transpose(Image.open(args.image)).convert("RGB")
s = min(img.size)
img = img.crop(((img.width - s) // 2, (img.height - s) // 2, (img.width + s) // 2, (img.height + s) // 2))
img = img.resize((args.size, args.size), Image.LANCZOS)
img.save(os.path.join(args.out, "target.png"))
H = W = args.size
target = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).reshape(-1, 3).to(device) * 2 - 1
coords = get_grid(H, W, device)

model = CoordMLP(args.act, hidden=args.hidden, hidden_layers=args.hidden_layers,
                 first_omega0=args.omega0, hidden_omega0=args.omega0).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"{args.act}: {n_params:,} parameters for {H * W * 3:,} pixel values "
      f"({H * W * 3 / n_params:.1f} values per parameter), device {device}")
opt = torch.optim.Adam(model.parameters(), lr=args.lr)


@torch.no_grad()
def render(chunk=2 ** 18):
    model.eval()
    out = torch.cat([model(coords[i:i + chunk]) for i in range(0, len(coords), chunk)])
    model.train()
    return out


def psnr(pred):  # on the [0, 1] intensity scale
    mse = (((pred.clamp(-1, 1) - target) / 2) ** 2).mean().item()
    return -10 * np.log10(mse)


def save_img(pred, path):
    arr = ((pred.clamp(-1, 1) + 1) / 2).reshape(H, W, 3).cpu().numpy()
    Image.fromarray((arr * 255).round().astype(np.uint8)).save(path)


snapshots = {int(x) for x in args.snapshots.split(",") if x}
history, t_start = [], time.time()
for step in range(1, args.steps + 1):
    if args.batch_pixels:
        idx = torch.randint(0, len(coords), (args.batch_pixels,), device=device)
        loss = ((model(coords[idx]) - target[idx]) ** 2).mean()
    else:
        loss = ((model(coords) - target) ** 2).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if step % args.log_every == 0 or step == 1 or step == args.steps or step in snapshots:
        pred = render()
        history.append(dict(step=step, loss=loss.item(), psnr=psnr(pred), time_s=time.time() - t_start))
        print(f"step {step:6d}  loss {loss.item():.6f}  PSNR {history[-1]['psnr']:.2f} dB  "
              f"({history[-1]['time_s']:.0f}s)", flush=True)
        if step in snapshots:
            save_img(pred, os.path.join(args.out, f"recon_step{step}.png"))

pred = render()
save_img(pred, os.path.join(args.out, "recon_final.png"))
pd.DataFrame(history).to_csv(os.path.join(args.out, "history.csv"), index=False)
torch.save({"model": model.state_dict(), "args": vars(args)}, os.path.join(args.out, "model.pt"))
summary = dict(act=args.act, omega0=args.omega0, size=args.size, params=n_params, steps=args.steps,
               final_psnr=psnr(pred), best_psnr=max(h["psnr"] for h in history),
               train_time_s=history[-1]["time_s"])
json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=2)
print(summary)

# ---- optional: does the network's DERIVATIVE look like the image's derivative? ----
if args.derivatives:
    from scipy import ndimage
    c = 256 if H >= 256 else H                     # a c x c crop in the middle, at full resolution
    y0 = x0 = (H - c) // 2
    grid = get_grid(H, W, device).reshape(H, W, 2)[y0:y0 + c, x0:x0 + c].reshape(-1, 2).clone().requires_grad_(True)
    gray = model(grid).mean(dim=1)                 # grayscale output
    g, = torch.autograd.grad(gray.sum(), grid, create_graph=True)
    lap = sum(torch.autograd.grad(g[:, i].sum(), grid, retain_graph=True)[0][:, i] for i in range(2))
    net_grad = g.norm(dim=1).reshape(c, c).detach().cpu().numpy()
    net_lap = lap.reshape(c, c).detach().cpu().numpy()
    gt = np.asarray(img.convert("L"), dtype=np.float32)[y0:y0 + c, x0:x0 + c] / 255.0
    gt_grad = np.hypot(ndimage.sobel(gt, 0), ndimage.sobel(gt, 1))
    gt_lap = ndimage.laplace(ndimage.gaussian_filter(gt, 0.5))
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
    for ax, m, t in zip(axes, [gt_grad, net_grad, gt_lap, net_lap],
                        ["Image gradient magnitude (Sobel)", f"Network gradient |grad f| ({args.act})",
                         "Image Laplacian", f"Network Laplacian ({args.act})"]):
        lim = np.percentile(np.abs(m), 99)
        im = ax.imshow(m, cmap="gray" if "gradient" in t.lower() else "RdBu",
                       vmin=0 if "gradient" in t.lower() else -lim, vmax=lim)
        ax.set_title(t); ax.set_xlabel("x (pixels)"); ax.set_ylabel("y (pixels)")
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "derivatives.png"), dpi=130)
    print("saved derivatives.png")
