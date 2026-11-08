"""DDPM diffusion model (Ho et al., 2020) for the animal dataset, built on Hugging Face diffusers
(UNet2DModel + DDPMScheduler; cite: von Platen et al., "Diffusers", github.com/huggingface/diffusers).

Train:   python train_diffusion.py --data data/animal256 --size 64 --steps 60000 --out runs/diffusion64
Figures (also made at the end of training):
         python train_diffusion.py --data data/animal256 --size 64 --out runs/diffusion64 --figures-only
Samples for FID:
         python train_diffusion.py --data data/animal256 --size 64 --out runs/diffusion64 --sample-only 5000 \
             --sample-dir samples/diffusion

How it works
  Forward (encoding) process: x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps,  eps ~ N(0, I),  t = 0..999.
      abar_t goes from ~1 (clean image) to ~0 (pure noise). No learning involved.
  Training: pick a random t, make x_t, and train the U-Net to predict the noise eps from (x_t, t) with MSE.
  Reverse (decoding/sampling): start from pure noise x_999 and repeatedly remove the predicted noise.
      DDPM uses all 1000 steps; DDIM (Song et al., 2021) reuses the same network with e.g. 100 larger steps.

Writes: fig_forward_process.png, fig_reverse_process.png, samples/step_*.png, final_samples.png, loss.png,
history.csv, checkpoint.pt (resumes automatically).
"""
import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torchvision.utils as vutils  # noqa: E402
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from common import EMA, ImageDir, plot_curves, save_grid  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", default="runs/diffusion64")
ap.add_argument("--size", type=int, default=64)
ap.add_argument("--base", type=int, default=64, help="U-Net channels at full resolution (x1, x2, x4, x4 deeper)")
ap.add_argument("--steps", type=int, default=60000, help="training iterations")
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--warmup", type=int, default=1000)
ap.add_argument("--schedule", default="linear", choices=["linear", "squaredcos_cap_v2"],
                help="noise schedule: linear (DDPM paper) or cosine (Nichol & Dhariwal 2021)")
ap.add_argument("--ema", type=float, default=0.9995)
ap.add_argument("--flip", action="store_true")
ap.add_argument("--sample-every", type=int, default=5000)
ap.add_argument("--ddim-steps", type=int, default=100, help="sampling steps for sample grids / FID samples")
ap.add_argument("--figures-only", action="store_true")
ap.add_argument("--sample-only", type=int, default=0)
ap.add_argument("--sample-dir", default=None)
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(os.path.join(args.out, "samples"), exist_ok=True)
amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")

data = ImageDir(args.data, args.size, flip=args.flip)
b = args.base
model = UNet2DModel(
    sample_size=args.size, in_channels=3, out_channels=3, layers_per_block=2,
    block_out_channels=(b, 2 * b, 4 * b, 4 * b),
    down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),   # self-attention at 1/4 res
    up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    norm_num_groups=min(32, b),                                  # GroupNorm groups must divide the channels
).to(device)
noise_sched = DDPMScheduler(num_train_timesteps=1000, beta_schedule=args.schedule)
ema = EMA(model, args.ema)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup))
print(f"{len(data)} images, U-Net {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, device {device}")

ck_path = os.path.join(args.out, "checkpoint.pt")
step, hist = 0, []
if os.path.exists(ck_path):
    ck = torch.load(ck_path, map_location=device)
    model.load_state_dict(ck["model"]); ema.model.load_state_dict(ck["ema"])
    opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
    step, hist = ck["step"], ck["history"]
    print(f"resumed at step {step}")


@torch.no_grad()
def sample(net, n, steps=100, keep_at=(), seed=0):
    """DDIM sampling with `steps` steps (steps=1000 -> exact DDPM schedule). Returns final images and, for every
    timestep in keep_at, the current noisy x_t and the network's current estimate of the clean image x_0."""
    s = DDIMScheduler.from_config(noise_sched.config) if steps < 1000 else noise_sched
    s.set_timesteps(steps)
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n, 3, args.size, args.size, generator=g).to(device)
    snaps = {}
    for t in s.timesteps:
        with torch.autocast(**amp):
            eps = net(x, t).sample
        out = s.step(eps.float(), t, x)
        if int(t) in keep_at:
            snaps[int(t)] = (x.clone(), out.pred_original_sample.clone())
        x = out.prev_sample
    return x, snaps


def figure_forward():
    """Encoding process: the same images getting progressively noisier."""
    x0 = torch.stack([data[i] for i in range(4)]).to(device)
    ts = [0, 50, 100, 200, 300, 500, 700, 999]
    g = torch.Generator(device="cpu").manual_seed(0)
    eps = torch.randn(x0.shape, generator=g).to(device)
    cols = [noise_sched.add_noise(x0, eps, torch.full((4,), t, device=device, dtype=torch.long)) for t in ts]
    grid = torch.stack(cols, 1).flatten(0, 1)                     # row = image, column = timestep
    abar = noise_sched.alphas_cumprod
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0.05, 0.36, 0.9, 0.6])
    img = vutils.make_grid(grid.cpu(), nrow=len(ts), normalize=True, value_range=(-1, 1), padding=2)
    ax.imshow(img.permute(1, 2, 0)); ax.set_yticks([])
    ax.set_xticks([(args.size + 2) * (i + 0.5) for i in range(len(ts))])
    ax.set_xticklabels([f"t = {t}\nsignal {abar[t].sqrt():.2f}" for t in ts])
    ax.set_xlabel("diffusion step t (forward / encoding process)"); ax.set_ylabel("training image")
    ax.set_title("Forward process: x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) noise")
    ax2 = fig.add_axes([0.08, 0.07, 0.86, 0.15])
    ax2.plot(abar.sqrt(), label="sqrt(abar_t): image weight"); ax2.plot((1 - abar).sqrt(), label="sqrt(1 - abar_t): noise weight")
    ax2.set_xlabel("diffusion step t"); ax2.set_ylabel("weight"); ax2.grid(alpha=0.3)
    ax2.legend(title=f"{args.schedule} noise schedule", loc="center right")
    plt.savefig(os.path.join(args.out, "fig_forward_process.png"), dpi=120); plt.close()


def figure_reverse(net):
    """Decoding process: from pure noise to an image, showing x_t and the predicted clean image."""
    s = DDIMScheduler.from_config(noise_sched.config); s.set_timesteps(args.ddim_steps)
    all_t = [int(t) for t in s.timesteps]                         # e.g. 990, 980, ..., 0 for 100 steps
    keep = all_t[:: max(1, len(all_t) // 10)]                     # about 10 snapshots along the way
    x, snaps = sample(net, 4, args.ddim_steps, keep_at=set(keep), seed=1)
    ts = sorted(snaps, reverse=True)
    for which, name in [(0, "x_t (current noisy image)"), (1, "predicted clean image x_0")]:
        cols = [snaps[t][which] for t in ts] + [x]
        grid = torch.stack(cols, 1).flatten(0, 1).clamp(-1, 1)
        img = vutils.make_grid(grid.cpu(), nrow=len(cols), normalize=True, value_range=(-1, 1), padding=2)
        plt.figure(figsize=(1.45 * len(cols) + 1, 6.5))
        plt.imshow(img.permute(1, 2, 0)); plt.yticks([])
        plt.xticks([(args.size + 2) * (i + 0.5) for i in range(len(cols))], [f"t={t}" for t in ts] + ["final"])
        plt.xlabel(f"reverse (decoding) step, DDIM with {args.ddim_steps} steps"); plt.ylabel("sample")
        plt.title(f"Reverse process from pure noise: {name}")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "fig_reverse_process.png" if which == 0 else "fig_reverse_x0_estimate.png"),
                    dpi=120)
        plt.close()


def save_ck():
    torch.save(dict(model=model.state_dict(), ema=ema.model.state_dict(), opt=opt.state_dict(),
                    sched=sched.state_dict(), step=step, history=hist, args=vars(args)), ck_path)


if args.sample_only:
    os.makedirs(args.sample_dir, exist_ok=True)
    for i in range(0, args.sample_only, 100):
        x, _ = sample(ema.model, min(100, args.sample_only - i), args.ddim_steps, seed=1000 + i)
        for j, im in enumerate((x.clamp(-1, 1) + 1) / 2):
            vutils.save_image(im, os.path.join(args.sample_dir, f"{i + j:05d}.png"))
    print(f"wrote {args.sample_only} samples to {args.sample_dir}")
    raise SystemExit

if not args.figures_only:
    loader = DataLoader(data, batch_size=args.bs, shuffle=True, drop_last=True, num_workers=args.workers,
                        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    t0 = time.time()
    while step < args.steps:
        for x0 in loader:
            x0 = x0.to(device, non_blocking=True)
            noise = torch.randn_like(x0)
            t = torch.randint(0, 1000, (x0.size(0),), device=device)
            xt = noise_sched.add_noise(x0, noise, t)                     # forward process, in one shot
            with torch.autocast(**amp):
                pred = model(xt, t).sample                               # U-Net predicts the noise
            loss = F.mse_loss(pred.float(), noise)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); ema.update(model)
            step += 1
            hist.append(dict(step=step, loss=loss.item()))
            if step % 500 == 0:
                print(f"step {step}/{args.steps}  loss {loss.item():.4f}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
            if step % args.sample_every == 0 or step == args.steps:
                x, _ = sample(ema.model, 64, args.ddim_steps)
                save_grid(x, os.path.join(args.out, "samples", f"step_{step:06d}.png"))
                save_ck()
            if step >= args.steps:
                break
    df = pd.DataFrame(hist)
    df.to_csv(os.path.join(args.out, "history.csv"), index=False)
    plot_curves({"noise-prediction MSE": df.loss.tolist()}, os.path.join(args.out, "loss.png"),
                "Diffusion training loss", smooth=200)

figure_forward()
figure_reverse(ema.model)
x, _ = sample(ema.model, 64, args.ddim_steps, seed=2)
save_grid(x, os.path.join(args.out, "final_samples.png"),
          title=f"Diffusion samples ({args.size}x{args.size}, DDIM {args.ddim_steps} steps)")
print("done:", args.out)
