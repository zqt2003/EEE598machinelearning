"""Training/validation loss curves, training time and peak HPU memory from a run's trainer_state.json.

python plot_losses.py --run runs/llava_chartqa_lora --out results
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="runs/llava_chartqa_lora")
ap.add_argument("--out", default="results")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

path = os.path.join(args.run, "trainer_state.json")
if not os.path.exists(path):                           # training not finished: use the newest checkpoint
    cks = sorted(glob.glob(os.path.join(args.run, "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[1]))
    path = os.path.join(cks[-1], "trainer_state.json")
log = pd.DataFrame(json.load(open(path))["log_history"])
print("read", path)

train = log.dropna(subset=["loss"]) if "loss" in log else pd.DataFrame()
evals = log.dropna(subset=["eval_loss"]) if "eval_loss" in log else pd.DataFrame()
plt.figure(figsize=(9, 5))
if len(train):
    plt.plot(train.step, train.loss, alpha=0.35, label="training loss (every 10 steps)")
    plt.plot(train.step, train.loss.rolling(20, min_periods=1).mean(), label="training loss (moving average)")
if len(evals):
    plt.plot(evals.step, evals.eval_loss, "o-", lw=2, label="validation loss")
plt.xlabel("optimizer step"); plt.ylabel("cross-entropy loss"); plt.yscale("log")
plt.title("LLaVA-1.6 LoRA fine-tuning on ChartQA (Gaudi 2)"); plt.grid(alpha=0.3, which="both"); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(args.out, "loss_curves.png"), dpi=140)

summary = {}
end = log.dropna(subset=["train_runtime"]) if "train_runtime" in log else pd.DataFrame()
if len(end):
    r = end.iloc[-1]
    summary.update(train_runtime_h=r.train_runtime / 3600, train_samples_per_s=r.get("train_samples_per_second"),
                   steps=int(r.step))
mem_cols = [c for c in log.columns if "max_memory_allocated" in c]   # optimum-habana logs HPU memory (GB)
if mem_cols:
    summary["peak_hpu_memory_GB"] = float(log[mem_cols[0]].max())
for c in log.columns:
    if "total_memory_available" in c:
        summary["hpu_memory_total_GB"] = float(log[c].max())
if len(train):
    summary.update(first_train_loss=float(train.loss.iloc[0]), last_train_loss=float(train.loss.iloc[-20:].mean()))
if len(evals):
    summary.update(first_eval_loss=float(evals.eval_loss.iloc[0]), last_eval_loss=float(evals.eval_loss.iloc[-1]))
json.dump(summary, open(os.path.join(args.out, "train_summary.json"), "w"), indent=2)
print(json.dumps(summary, indent=2))
