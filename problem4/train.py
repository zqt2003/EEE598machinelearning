"""Train the turbulence-robust feature head.

Examples:
    python train.py --data data --out ckpt/turbfeat.pt                       # proposed method
    python train.py --data data --out ckpt/no_antialias.pt --no-antialias    # ablation
    python train.py --data data --out ckpt/inv_only.pt --loss invariance     # ablation (collapses)
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from p4lib import PairDataset, TurbFeature, dense_infonce, invariance_only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="ckpt/turbfeat.pt")
    ap.add_argument("--loss", choices=["infonce", "invariance"], default="infonce")
    ap.add_argument("--no-antialias", action="store_true", help="use VGG's original max-pooling")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-pretrained", action="store_true", help="random VGG weights (debugging only)")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = PairDataset(args.data, "train")
    dl = torch.utils.data.DataLoader(ds, batch_size=args.bs, shuffle=True, drop_last=True,
                                     num_workers=args.workers, pin_memory=device.type == "cuda")
    model = TurbFeature(antialias=not args.no_antialias, dim=args.dim,
                        pretrained=not args.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(dl))
    print(f"{len(ds)} training images, {len(dl)} batches/epoch, device {device}, "
          f"{sum(p.numel() for p in model.head.parameters()) / 1e6:.2f}M trainable params")

    history = []
    for epoch in range(args.epochs):
        model.head.train()
        t0, losses = time.time(), []
        for clean, ta, tb in dl:
            clean, ta, tb = clean.to(device), ta.to(device), tb.to(device)
            z = model(torch.cat([clean, ta, tb]))
            zc, za, zb = z.chunk(3)
            if args.loss == "infonce":
                # clean <-> turbulent and turbulent <-> another turbulence realization
                loss = (dense_infonce(zc, za, tau=args.tau) + dense_infonce(za, zb, tau=args.tau)) / 2
            else:
                loss = (invariance_only(zc, za) + invariance_only(za, zb)) / 2
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        history.append(float(np.mean(losses)))
        print(f"epoch {epoch + 1:3d}/{args.epochs}  loss {history[-1]:.4f}  ({time.time() - t0:.1f}s)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = dict(antialias=not args.no_antialias, dim=args.dim, loss=args.loss)
    torch.save({"head": model.head.state_dict(), "config": cfg, "history": history}, out)
    out.with_suffix(".json").write_text(json.dumps({"config": cfg, "history": history}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
