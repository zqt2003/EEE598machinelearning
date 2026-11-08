"""Train a DCGAN on a folder of images (CelebA, colored squares, or your animal dataset).

Tutorial settings (CelebA):
    python train_dcgan.py --data data/celeba --size 64 --epochs 10 --out runs/dcgan_celeba
Colored squares:
    python train_dcgan.py --data data/squares_1k --size 64 --epochs 300 --out runs/dcgan_squares_1k
Animals (small dataset -> use the stabilizing tricks):
    python train_dcgan.py --data data/animal256 --size 64 --epochs 400 --diffaug --label-smooth 0.9 \
        --flip --out runs/dcgan_animal64

Writes to --out: samples/epoch_*.png (fixed noise, EMA generator), final_samples.png, real_vs_fake.png,
losses.png, d_outputs.png, history.csv, checkpoint.pt (resume with the same command).
"""
import argparse
import os
import time

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import EMA, ImageDir, diff_augment, plot_curves, save_grid
from dcgan import Discriminator, Generator, weights_init

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", default="runs/dcgan")
ap.add_argument("--size", type=int, default=64)
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--nz", type=int, default=100)
ap.add_argument("--ngf", type=int, default=64)
ap.add_argument("--ndf", type=int, default=64)
ap.add_argument("--lr-g", type=float, default=2e-4)
ap.add_argument("--lr-d", type=float, default=2e-4)
ap.add_argument("--beta1", type=float, default=0.5)
ap.add_argument("--label-smooth", type=float, default=1.0, help="target for real images (1.0 = off, 0.9 = one-sided smoothing)")
ap.add_argument("--diffaug", action="store_true", help="DiffAugment on D's inputs (for small datasets)")
ap.add_argument("--spectral-norm", action="store_true", help="spectral normalization in D instead of BatchNorm")
ap.add_argument("--flip", action="store_true", help="random horizontal flips of the real images")
ap.add_argument("--ema", type=float, default=0.999, help="EMA decay of the generator weights used for samples")
ap.add_argument("--sample-every", type=int, default=1, help="epochs between sample grids")
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--seed", type=int, default=999)
args = ap.parse_args()

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(os.path.join(args.out, "samples"), exist_ok=True)

data = ImageDir(args.data, args.size, flip=args.flip)
loader = DataLoader(data, batch_size=args.bs, shuffle=True, drop_last=True, num_workers=args.workers,
                    pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
print(f"{len(data)} images, {len(loader)} iterations/epoch, device {device}")

G = Generator(args.nz, args.ngf, args.size).to(device)
D = Discriminator(args.ndf, args.size, spectral=args.spectral_norm).to(device)
G.apply(weights_init); D.apply(weights_init)
ema = EMA(G, args.ema)
opt_g = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(args.beta1, 0.999))
opt_d = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(args.beta1, 0.999))
fixed_z = torch.randn(64, args.nz, device=device)
aug = (lambda x: diff_augment(x)) if args.diffaug else (lambda x: x)

ck_path = os.path.join(args.out, "checkpoint.pt")
start_epoch, hist = 1, []
if os.path.exists(ck_path):                                    # resume after a Slurm time limit
    ck = torch.load(ck_path, map_location=device)
    G.load_state_dict(ck["G"]); D.load_state_dict(ck["D"]); ema.model.load_state_dict(ck["G_ema"])
    opt_g.load_state_dict(ck["opt_g"]); opt_d.load_state_dict(ck["opt_d"])
    start_epoch, hist, fixed_z = ck["epoch"] + 1, ck["history"], ck["fixed_z"].to(device)
    print(f"resumed from epoch {ck['epoch']}")

t0 = time.time()
for epoch in range(start_epoch, args.epochs + 1):
    for real in loader:
        real = real.to(device, non_blocking=True)
        b = real.size(0)
        z = torch.randn(b, args.nz, device=device)
        fake = G(z)

        # (1) Discriminator: maximize log D(x) + log(1 - D(G(z)))
        d_real, d_fake = D(aug(real)), D(aug(fake.detach()))
        loss_d = (F.binary_cross_entropy_with_logits(d_real, torch.full_like(d_real, args.label_smooth))
                  + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake)))
        opt_d.zero_grad(set_to_none=True); loss_d.backward(); opt_d.step()

        # (2) Generator: maximize log D(G(z))  (the "non-saturating" loss from the GAN paper)
        d_gen = D(aug(fake))
        loss_g = F.binary_cross_entropy_with_logits(d_gen, torch.ones_like(d_gen))
        opt_g.zero_grad(set_to_none=True); loss_g.backward(); opt_g.step()
        ema.update(G)

        hist.append(dict(epoch=epoch, loss_d=loss_d.item(), loss_g=loss_g.item(),
                         D_x=torch.sigmoid(d_real).mean().item(), D_G_z=torch.sigmoid(d_fake).mean().item()))
    h = hist[-1]
    print(f"epoch {epoch:4d}/{args.epochs}  loss_D {h['loss_d']:.3f}  loss_G {h['loss_g']:.3f}  "
          f"D(x) {h['D_x']:.2f}  D(G(z)) {h['D_G_z']:.2f}  ({time.time() - t0:.0f}s)", flush=True)
    if epoch % args.sample_every == 0 or epoch == args.epochs:
        with torch.no_grad():
            save_grid(ema.model(fixed_z), os.path.join(args.out, "samples", f"epoch_{epoch:04d}.png"))
        torch.save(dict(G=G.state_dict(), D=D.state_dict(), G_ema=ema.model.state_dict(), opt_g=opt_g.state_dict(),
                        opt_d=opt_d.state_dict(), epoch=epoch, history=hist, fixed_z=fixed_z.cpu(),
                        args=vars(args), kind="dcgan"), ck_path)

# ---- final figures ----
df = pd.DataFrame(hist)
df.to_csv(os.path.join(args.out, "history.csv"), index=False)
plot_curves({"Discriminator": df.loss_d.tolist(), "Generator": df.loss_g.tolist()},
            os.path.join(args.out, "losses.png"), "DCGAN losses during training")
plot_curves({"D(x), real images": df.D_x.tolist(), "D(G(z)), generated images": df.D_G_z.tolist()},
            os.path.join(args.out, "d_outputs.png"), "Discriminator output (probability of 'real')",
            ylabel="mean D output")
with torch.no_grad():
    fake = ema.model(fixed_z)
save_grid(fake, os.path.join(args.out, "final_samples.png"), title=f"DCGAN samples after {args.epochs} epochs")
real = torch.stack([data[i] for i in range(min(64, len(data)))])
save_grid(torch.cat([real[:32], fake[:32].cpu()]), os.path.join(args.out, "real_vs_fake.png"),
          title="Top 4 rows: real images. Bottom 4 rows: generated images")
print("done:", args.out)
