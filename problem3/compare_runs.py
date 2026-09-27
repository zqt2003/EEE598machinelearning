"""Compare training runs: results table + learning curves.

Example:  python compare_runs.py runs/resnet34 runs/resnet36 --out runs/comparison.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--out", default="runs/comparison.png")
args = ap.parse_args()

rows, hists = [], {}
for r in args.runs:
    r = Path(r)
    if (r / "results.json").exists():
        rows.append(json.loads((r / "results.json").read_text()))
    if (r / "history.csv").exists():
        hists[r.name] = pd.read_csv(r / "history.csv")

if rows:
    t = pd.DataFrame(rows)[["arch", "layers", "params", "gpus", "epochs", "best_epoch", "train_top1", "val_top1",
                            "val_top5", "test_top1", "test_top5", "train_time_min", "avg_images_per_s"]]
    t["params"] = (t["params"] / 1e6).round(2).astype(str) + "M"
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    t.to_csv(Path(args.out).with_suffix(".csv"), index=False)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for name, h in hists.items():
    axes[0].plot(h.epoch, h.train_loss, label=f"{name} train")
    axes[0].plot(h.epoch, h.val_loss, "--", label=f"{name} val")
    axes[1].plot(h.epoch, h.train_top1, label=f"{name} train")
    axes[1].plot(h.epoch, h.val_top1, "--", label=f"{name} val")
    axes[2].plot(h.epoch, h.val_top5, label=name)
for ax, title in zip(axes, ["Loss", "Top-1 accuracy", "Validation top-5 accuracy"]):
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(args.out, dpi=140)
print(f"saved {args.out}")
