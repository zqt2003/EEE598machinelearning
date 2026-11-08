"""Compare baseline and fine-tuned ChartQA answers: accuracy table (overall and by question type) and example
figures (questions the fine-tuned model now gets right, and ones it still gets wrong).

python compare_eval.py --base results/eval_baseline.csv --ft results/eval_finetuned.csv --out results
"""
import argparse
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="results/eval_baseline.csv")
ap.add_argument("--ft", default="results/eval_finetuned.csv")
ap.add_argument("--out", default="results")
ap.add_argument("--k", type=int, default=3, help="examples per figure")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

b, f = pd.read_csv(args.base), pd.read_csv(args.ft)
n = min(len(b), len(f))
b, f = b.iloc[:n], f.iloc[:n]
assert (b.question == f.question).all(), "the two files must contain the same questions in the same order"


def acc_table(groups):
    rows = []
    for name, mask in groups:
        if mask.sum():
            rows.append({"subset": name, "questions": int(mask.sum()),
                         "baseline (relaxed)": b.correct[mask].mean(), "fine-tuned (relaxed)": f.correct[mask].mean(),
                         "baseline (lenient)": b.lenient[mask].mean(), "fine-tuned (lenient)": f.lenient[mask].mean()})
    return pd.DataFrame(rows)


all_ = pd.Series(True, index=b.index)
groups = [("all", all_), ("human-written", b.human_written), ("machine-generated", ~b.human_written),
          ("numeric answer", b.numeric), ("text answer", ~b.numeric)]
groups += [(f"type: {t}", b.qtype == t) for t in sorted(b.qtype.unique())]
table = acc_table(groups)
pd.set_option("display.width", 160)
print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
table.to_csv(os.path.join(args.out, "accuracy_table.csv"), index=False)
with open(os.path.join(args.out, "accuracy_table.tex"), "w") as fh:
    fh.write(table.to_latex(index=False, float_format=lambda x: f"{100 * x:.1f}\\%"))

fixed = b.index[(~b.correct) & f.correct]
still = b.index[(~f.correct)]
print(f"\nfine-tuning fixed {len(fixed)} questions, broke {int((b.correct & ~f.correct).sum())}, "
      f"{len(still)} still wrong")


def examples(idx, path, title):
    if len(idx) == 0 or args.k == 0:
        return
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceM4/ChartQA", split="test").shuffle(seed=0)    # same order as eval_chartqa.py
    idx = list(idx[:args.k])
    fig, axes = plt.subplots(1, len(idx), figsize=(7 * len(idx), 7.5))
    axes = [axes] if len(idx) == 1 else axes
    for ax, i in zip(axes, idx):
        ax.imshow(ds[int(b.i[i])]["image"])
        txt = (f"Q: {b.question[i]}\nTrue answer: {b.answer[i]}\n"
               f"Baseline: {str(b.prediction[i])[:80]}\nFine-tuned: {str(f.prediction[i])[:80]}")
        ax.set_title("\n".join(textwrap.fill(line, 55) for line in txt.split("\n")), fontsize=10, loc="left")
        ax.set_xlabel("x (pixels)"); ax.set_ylabel("y (pixels)")
    plt.suptitle(title, fontsize=14); plt.tight_layout()
    plt.savefig(path, dpi=110); plt.close(fig)


examples(fixed, os.path.join(args.out, "examples_fixed.png"), "Questions the fine-tuned model now answers correctly")
examples(still, os.path.join(args.out, "examples_still_wrong.png"), "Questions the fine-tuned model still gets wrong")
with open(os.path.join(args.out, "still_wrong.txt"), "w") as fh:
    for i in still:
        fh.write(f"[{f.qtype[i]}, {'human' if f.human_written[i] else 'machine'}] {f.question[i]} | "
                 f"true: {f.answer[i]} | predicted: {f.prediction[i]}\n")
print("saved accuracy_table.csv/.tex, examples_fixed.png, examples_still_wrong.png, still_wrong.txt")
