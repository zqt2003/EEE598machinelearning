"""Write N generated images to a folder (for FID/KID with fid.py).

python sample_to_folder.py --ckpt runs/dcgan_animal64/checkpoint.pt --n 5000 --out samples/dcgan
(For the diffusion model use:  python train_diffusion.py ... --sample-only 5000 --sample-dir samples/diffusion)
"""
import argparse
import os

import torch
import torchvision.utils as vutils

from models_io import load_generator

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--n", type=int, default=5000)
ap.add_argument("--bs", type=int, default=100)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=123)
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
G, z_dim, kind = load_generator(args.ckpt, device)
os.makedirs(args.out, exist_ok=True)
torch.manual_seed(args.seed)
with torch.no_grad():
    for i in range(0, args.n, args.bs):
        imgs = (G(torch.randn(min(args.bs, args.n - i), z_dim, device=device)).clamp(-1, 1) + 1) / 2
        for j, im in enumerate(imgs):
            vutils.save_image(im, os.path.join(args.out, f"{i + j:05d}.png"))
print(f"wrote {args.n} {kind} samples to {args.out}")
