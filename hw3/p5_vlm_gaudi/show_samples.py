"""Step 2: show two ChartQA charts with their questions and answers, and count the training samples the
fine-tuning script uses after its 80/20 train/validation split.  (CPU is enough.)

python show_samples.py --out results
"""
import argparse
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from datasets import load_dataset  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results")
ap.add_argument("--idx", type=int, nargs=2, default=[0, 1], help="which training samples to show")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

ds = load_dataset("HuggingFaceM4/ChartQA")
print(ds)                                            # splits: train / val / test
# run_lora_vlm.py: no split named "validation" -> it splits the TRAIN split 80/20 with seed 42
split = ds["train"].train_test_split(test_size=0.2, seed=42)
n_train, n_val = len(split["train"]), len(split["test"])
print(f"ChartQA train split: {len(ds['train']):,} -> script trains on {n_train:,} and validates on {n_val:,}")
print(f"(test split: {len(ds['test']):,} questions, used for the accuracy evaluation)")

fig, axes = plt.subplots(1, 2, figsize=(15, 7))
for ax, i in zip(axes, args.idx):
    ex = ds["train"][i]
    ans = ex["label"][0] if isinstance(ex["label"], list) else ex["label"]
    ax.imshow(ex["image"])
    ax.set_title("\n".join(textwrap.wrap(f"Q: {ex['query']}", 60)) + f"\nA: {ans}", fontsize=11)
    ax.set_xlabel("x (pixels)"); ax.set_ylabel("y (pixels)")
plt.suptitle("Two ChartQA training samples (chart, question, answer)")
plt.tight_layout()
plt.savefig(os.path.join(args.out, "chartqa_samples.png"), dpi=120)
with open(os.path.join(args.out, "split_counts.txt"), "w") as f:
    f.write(f"ChartQA train {len(ds['train'])}, val {len(ds['val']) if 'val' in ds else 'n/a'}, "
            f"test {len(ds['test'])}\nafter 80/20 split (seed 42): train {n_train}, validation {n_val}\n")
print("saved", os.path.join(args.out, "chartqa_samples.png"))
