"""Train the GRU note model on MAESTRO.

Example (Sol GPU):  python train.py --num-files 300 --epochs 40 --out runs/gru
Quick test:         python train.py --num-files 5 --epochs 1 --out runs/test

Writes to --out: best.pt (lowest validation loss), history.csv, learning_curves.png
"""
import argparse
import json
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from data import build_windows, download_maestro  # noqa: E402
from model import MusicGRU, music_loss  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data", help="where MAESTRO is (downloaded there if missing)")
ap.add_argument("--out", default="runs/gru")
ap.add_argument("--num-files", type=int, default=300, help="MIDI files to use (MAESTRO has 1,282)")
ap.add_argument("--val-frac", type=float, default=0.1, help="fraction of FILES held out for validation")
ap.add_argument("--seq-len", type=int, default=32)
ap.add_argument("--stride", type=int, default=16)
ap.add_argument("--hidden", type=int, default=256)
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--dropout", type=float, default=0.2)
ap.add_argument("--rnn", choices=["ours", "torch"], default="ours", help="our GRU (gru.py) or torch.nn.GRU")
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--w-pitch", type=float, default=0.05, help="loss weight of the pitch cross-entropy (tutorial: 0.05)")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(args.out, exist_ok=True)

# ---- data: split by FILE, so validation pieces were never seen in training ----
files = download_maestro(args.data)
random.Random(args.seed).shuffle(files)
files = files[:args.num_files]
n_val = max(1, round(args.val_frac * len(files)))
val_files, train_files = files[:n_val], files[n_val:]
print("train:"); tr = build_windows(train_files, args.seq_len, args.stride)
print("val:");   va = build_windows(val_files, args.seq_len, args.stride)
with open(os.path.join(args.out, "val_files.txt"), "w") as f:     # generate.py takes its seed melody from here
    f.write("\n".join(val_files))


def batches(arrs, bs, shuffle):
    idx = np.random.permutation(len(arrs[0])) if shuffle else np.arange(len(arrs[0]))
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        p, s, d = (torch.from_numpy(a[j]).to(device) for a in arrs)
        # input = notes 0..T-1, target = notes 1..T  (predict the next note at every position)
        yield (p[:, :-1], s[:, :-1], d[:, :-1]), (p[:, 1:], s[:, 1:], d[:, 1:])


model = MusicGRU(args.hidden, args.layers, dropout=args.dropout, rnn=args.rnn).to(device)
print(f"model: {sum(p.numel() for p in model.parameters()):,} parameters, rnn={args.rnn}, device={device}")
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)

history, best = [], float("inf")
for epoch in range(1, args.epochs + 1):
    t0 = time.time()
    sums = {}
    model.train()
    for x, y in batches(tr, args.bs, True):
        loss, parts = music_loss(model(*x), y, w_pitch=args.w_pitch)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # RNNs can have exploding gradients
        opt.step()
        for k, v in dict(total=loss.item(), **parts).items():
            sums.setdefault("train_" + k, []).append(v)
    model.eval()
    with torch.no_grad():
        for x, y in batches(va, 512, False):
            out = model(*x)
            loss, parts = music_loss(out, y, w_pitch=args.w_pitch)
            acc = (out[0].argmax(-1) == y[0]).float().mean().item()
            for k, v in dict(total=loss.item(), pitch_acc=acc, **parts).items():
                sums.setdefault("val_" + k, []).append(v)
    row = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], "time_s": time.time() - t0}
    row.update({k: float(np.mean(v)) for k, v in sums.items()})
    history.append(row)
    sched.step(row["val_total"])
    print(f"epoch {epoch:3d}  train {row['train_total']:.4f}  val {row['val_total']:.4f}  "
          f"(pitch CE {row['val_pitch']:.3f}, acc {row['val_pitch_acc']:.3f}, step {row['val_step']:.4f}, "
          f"dur {row['val_duration']:.4f})  {row['time_s']:.0f}s", flush=True)
    if row["val_total"] < best:
        best = row["val_total"]
        torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch}, os.path.join(args.out, "best.pt"))
    pd.DataFrame(history).to_csv(os.path.join(args.out, "history.csv"), index=False)

# ---- learning curves ----
h = pd.DataFrame(history)
fig, axes = plt.subplots(1, 4, figsize=(20, 4.2))
for ax, key, title in zip(axes, ["total", "pitch", "step", "duration"],
                          ["Total loss", "Pitch cross-entropy", "Step loss (MSE + positive pressure)",
                           "Duration loss (MSE + positive pressure)"]):
    ax.plot(h.epoch, h["train_" + key], "-o", ms=3, label="train")
    ax.plot(h.epoch, h["val_" + key], "-o", ms=3, label="validation")
    ax.set_title(title); ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.grid(alpha=0.3); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(args.out, "learning_curves.png"), dpi=140)
json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"), indent=2)
print(f"best validation loss {best:.4f}; saved to {args.out}/")
