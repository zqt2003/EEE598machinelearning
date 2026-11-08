"""Latent-space interpolation with a trained GAN.

python interpolate.py --ckpt runs/progan_animal128/checkpoint.pt --out runs/progan_animal128/interp

For each of --pairs random latent pairs (z_a, z_b) we decode the path between them in --steps steps.
    linear: z(t) = (1 - t) z_a + t z_b
    slerp : walk along the great circle between z_a and z_b. Gaussian latents of dimension 100-512 lie
            near a sphere of radius sqrt(dim); the linear midpoint is much closer to the origin (an atypical
            latent), so slerp usually gives cleaner middle images.
Writes interp_linear.png, interp_slerp.png, interp_2d.png (a 2-D grid between 4 corner latents) and a GIF.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torchvision.utils as vutils  # noqa: E402
from PIL import Image  # noqa: E402

from models_io import load_generator  # noqa: E402


def slerp(a, b, t):
    an, bn = a / a.norm(dim=-1, keepdim=True), b / b.norm(dim=-1, keepdim=True)
    omega = torch.acos((an * bn).sum(-1, keepdim=True).clamp(-1, 1))
    so = torch.sin(omega)
    return (torch.sin((1 - t) * omega) / so) * a + (torch.sin(t * omega) / so) * b


def to_img(x):
    return ((x.clamp(-1, 1) + 1) / 2).cpu()


def figure(rows, path, title, steps):
    """rows: list of (steps, 3, H, W) tensors -> one labelled figure."""
    grid = vutils.make_grid(torch.cat(rows), nrow=steps, padding=2)
    h, w = rows[0].shape[-2:]
    plt.figure(figsize=(steps * 1.3 + 1, len(rows) * 1.3 + 1))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.xticks([(w + 2) * i + w / 2 for i in range(steps)], [f"{i / (steps - 1):.2f}" for i in range(steps)])
    plt.yticks([(h + 2) * i + h / 2 for i in range(len(rows))], [f"pair {i + 1}" for i in range(len(rows))])
    plt.xlabel("interpolation weight t  (0 = z_a, 1 = z_b)"); plt.ylabel("latent pair")
    plt.title(title); plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--steps", type=int, default=9)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    G, z_dim, kind = load_generator(args.ckpt, device)
    out = args.out or os.path.join(os.path.dirname(args.ckpt), "interp")
    os.makedirs(out, exist_ok=True)
    g = torch.Generator().manual_seed(args.seed)
    za, zb = torch.randn(args.pairs, z_dim, generator=g), torch.randn(args.pairs, z_dim, generator=g)
    t = torch.linspace(0, 1, args.steps).view(-1, 1)
    for name, fn in [("linear", lambda a, b: (1 - t) * a + t * b), ("slerp", lambda a, b: slerp(a, b, t))]:
        rows = [to_img(G(fn(za[i], zb[i]).to(device))) for i in range(args.pairs)]
        figure(rows, os.path.join(out, f"interp_{name}.png"),
               f"{kind.upper()} latent interpolation ({name})", args.steps)
        if name == "slerp":                                     # GIF of the first pair, there and back
            frames = [Image.fromarray((im.permute(1, 2, 0).numpy() * 255).astype("uint8")).resize((256, 256))
                      for im in torch.cat([rows[0], rows[0].flip(0)])]
            frames[0].save(os.path.join(out, "interp.gif"), save_all=True, append_images=frames[1:], duration=120, loop=0)
    # 2-D grid: bilinear mix of 4 corner latents
    c = torch.randn(4, z_dim, generator=g)
    s = torch.linspace(0, 1, args.steps)
    zs = torch.stack([(1 - u) * ((1 - v) * c[0] + v * c[1]) + u * ((1 - v) * c[2] + v * c[3]) for u in s for v in s])
    imgs = to_img(G(zs.to(device)))
    rows = [imgs[i * args.steps:(i + 1) * args.steps] for i in range(args.steps)]
    figure(rows, os.path.join(out, "interp_2d.png"), f"{kind.upper()}: 2-D walk between four latent corners",
           args.steps)
    print("saved to", out)


if __name__ == "__main__":
    main()
