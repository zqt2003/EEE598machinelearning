"""Train a Progressive GAN on a folder of images, growing 4x4 -> 8x8 -> ... -> --max-res.

python train_progan.py --data data/animal256 --max-res 128 --out runs/progan_animal128
    (about 2 h on one A100 with the defaults; resumes automatically from checkpoint.pt)
Quick test: python train_progan.py --data data/squares_1k --max-res 16 --fade-kimg 2 --stable-kimg 2 --out runs/pg_test

Each resolution has a FADE phase (alpha 0 -> 1, new layers blended in) and a STABLE phase (alpha = 1).
Writes: samples/res<R>_<phase>.png, final_samples.png, losses.png, history.csv, checkpoint.pt
"""
import argparse
import math
import os
import time

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import EMA, ImageDir, plot_curves, save_grid
from progan import Discriminator, Generator

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", default="runs/progan")
ap.add_argument("--max-res", type=int, default=128)
ap.add_argument("--z-dim", type=int, default=512)
ap.add_argument("--fmap-max", type=int, default=512)
ap.add_argument("--fade-kimg", type=float, default=150, help="thousands of real images shown while fading in")
ap.add_argument("--stable-kimg", type=float, default=150, help="thousands of real images at each stable resolution")
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--gp", type=float, default=10.0, help="gradient-penalty weight (WGAN-GP)")
ap.add_argument("--drift", type=float, default=1e-3)
ap.add_argument("--flip", action="store_true")
ap.add_argument("--ema", type=float, default=0.999)
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
os.makedirs(os.path.join(args.out, "samples"), exist_ok=True)
BATCH = {4: 64, 8: 64, 16: 64, 32: 32, 64: 32, 128: 16, 256: 8}          # smaller batches at high resolution

data = ImageDir(args.data, args.max_res, flip=args.flip)
G = Generator(args.z_dim, args.max_res, fmap_max=args.fmap_max).to(device)
D = Discriminator(args.max_res, fmap_max=args.fmap_max).to(device)
ema = EMA(G, args.ema)
opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.0, 0.99), eps=1e-8)
opt_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.0, 0.99), eps=1e-8)
print(f"{len(data)} images, G {sum(p.numel() for p in G.parameters()) / 1e6:.1f}M, "
      f"D {sum(p.numel() for p in D.parameters()) / 1e6:.1f}M params, device {device}")

# phase list: (depth, kind) -> 4x4 stable, then (fade, stable) for every higher resolution
max_depth = int(math.log2(args.max_res)) - 2
phases = [(0, "stable")] + [(d, k) for d in range(1, max_depth + 1) for k in ("fade", "stable")]
fixed_z = torch.randn(64, args.z_dim, device=device)
ck_path = os.path.join(args.out, "checkpoint.pt")
start_phase, start_img, hist = 0, 0, []
if os.path.exists(ck_path):
    ck = torch.load(ck_path, map_location=device)
    G.load_state_dict(ck["G"]); D.load_state_dict(ck["D"]); ema.model.load_state_dict(ck["G_ema"])
    opt_g.load_state_dict(ck["opt_g"]); opt_d.load_state_dict(ck["opt_d"])
    start_phase, start_img, hist, fixed_z = ck["phase"], ck["img_in_phase"], ck["history"], ck["fixed_z"].to(device)
    print(f"resumed at phase {start_phase} {phases[start_phase]}, image {start_img}")


def real_at(x, depth, alpha):
    """Downsample real images to the current resolution; during a fade, blend with the lower resolution too
    (so D sees real images that match what the fading G can produce)."""
    res = 4 * 2 ** depth
    x = F.adaptive_avg_pool2d(x, res)
    if alpha < 1.0 and depth > 0:
        x = alpha * x + (1 - alpha) * F.interpolate(F.avg_pool2d(x, 2), scale_factor=2, mode="nearest")
    return x


def save(phase, img_in_phase):
    torch.save(dict(G=G.state_dict(), D=D.state_dict(), G_ema=ema.model.state_dict(), opt_g=opt_g.state_dict(),
                    opt_d=opt_d.state_dict(), phase=phase, img_in_phase=img_in_phase, history=hist,
                    fixed_z=fixed_z.cpu(), args=vars(args), kind="progan"), ck_path)


t0 = time.time()
for p in range(start_phase, len(phases)):
    depth, kind = phases[p]
    res, bs = 4 * 2 ** depth, BATCH[4 * 2 ** depth]
    total = int(1000 * (args.fade_kimg if kind == "fade" else args.stable_kimg))
    loader = DataLoader(data, batch_size=bs, shuffle=True, drop_last=True, num_workers=args.workers,
                        pin_memory=device.type == "cuda")
    seen = start_img if p == start_phase else 0
    print(f"--- phase {p}: {res}x{res} {kind}, {total:,} images, batch {bs}", flush=True)
    while seen < total:
        for real in loader:
            alpha = min(1.0, seen / total) if kind == "fade" else 1.0
            real = real_at(real.to(device, non_blocking=True), depth, alpha)
            z = torch.randn(real.size(0), args.z_dim, device=device)

            # Discriminator (critic): WGAN-GP
            fake = G(z, depth, alpha).detach()
            d_real, d_fake = D(real, depth, alpha), D(fake, depth, alpha)
            eps = torch.rand(real.size(0), 1, 1, 1, device=device)
            mix = (eps * real + (1 - eps) * fake).requires_grad_(True)
            grad, = torch.autograd.grad(D(mix, depth, alpha).sum(), mix, create_graph=True)
            gp = (grad.flatten(1).norm(dim=1) - 1).pow(2).mean()
            loss_d = d_fake.mean() - d_real.mean() + args.gp * gp + args.drift * d_real.pow(2).mean()
            opt_d.zero_grad(set_to_none=True); loss_d.backward(); opt_d.step()

            # Generator
            loss_g = -D(G(z, depth, alpha), depth, alpha).mean()
            opt_g.zero_grad(set_to_none=True); loss_g.backward(); opt_g.step()
            ema.update(G)

            seen += real.size(0)
            hist.append(dict(phase=p, res=res, kind=kind, alpha=alpha, loss_d=loss_d.item(), loss_g=loss_g.item(),
                             gp=gp.item(), w_dist=(d_real.mean() - d_fake.mean()).item()))
            if len(hist) % 500 == 0:
                h = hist[-1]
                print(f"  {res}x{res} {kind} {seen / 1000:.0f}/{total / 1000:.0f} kimg  alpha {alpha:.2f}  "
                      f"loss_D {h['loss_d']:.3f}  loss_G {h['loss_g']:.3f}  W-dist {h['w_dist']:.3f}  "
                      f"({(time.time() - t0) / 60:.1f} min)", flush=True)
            if len(hist) % 2000 == 0:
                save(p, seen)
            if seen >= total:
                break
    with torch.no_grad():
        imgs = F.interpolate(ema.model(fixed_z, depth, 1.0), size=args.max_res, mode="nearest")
    save_grid(imgs, os.path.join(args.out, "samples", f"res{res:03d}_{kind}.png"))
    save(p + 1, 0)

# ---- final figures ----
df = pd.DataFrame(hist)
df.to_csv(os.path.join(args.out, "history.csv"), index=False)
plot_curves({"critic (D)": df.loss_d.tolist(), "generator": df.loss_g.tolist()},
            os.path.join(args.out, "losses.png"), "Progressive GAN losses (resolution grows over training)")
plot_curves({"D(real) - D(fake)": df.w_dist.tolist()}, os.path.join(args.out, "wasserstein.png"),
            "Critic's Wasserstein-distance estimate", ylabel="distance estimate")
with torch.no_grad():
    save_grid(ema.model(fixed_z), os.path.join(args.out, "final_samples.png"),
              title=f"Progressive GAN samples ({args.max_res}x{args.max_res})")
print(f"done in {(time.time() - t0) / 60:.1f} min:", args.out)
