"""Problem 3.1: build train/val/test splits of the ImageNet *train* folder, plus a prototype subset.

Nothing is copied: the script writes CSV manifests (image path, class, label, split) that the
training code reads, so the ~140 GB of images stay where they are on Sol.

Outputs (in --out):
  full_split.csv          every image of all classes, split per class into train/val/test
  proto_split.csv         the prototype subset (fewer classes, fewer images per class)
  proto_classes.csv       the prototype classes with their human-readable names
  proto_stats.csv         per-class counts in the prototype (train/val/test)
  proto_summary.txt       dataset statistics for the report
  fig_proto_samples.png   one example image per prototype class (first 40 classes)
  fig_proto_stats.png     class availability, split sizes, image sizes and aspect ratios
  (the prototype test images are always part of the full test split, so results stay comparable)

Example:
  python make_prototype.py --mapping LOC_synset_mapping.txt --n-classes 100 --per-class 300
"""
import argparse
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/data/datasets/community/deeplearning/imagenet/train")
ap.add_argument("--mapping", default=None, help="synset -> name file from Canvas (e.g. 'n01440764 tench, Tinca tinca')")
ap.add_argument("--out", default="imagenet_splits")
ap.add_argument("--n-classes", type=int, default=100, help="classes in the prototype")
ap.add_argument("--per-class", type=int, default=300, help="images per class in the prototype")
ap.add_argument("--val-frac", type=float, default=0.1)
ap.add_argument("--test-frac", type=float, default=0.1)
ap.add_argument("--stats-sample", type=int, default=3000, help="images to open for size / color statistics")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

rng = random.Random(args.seed)
root, out = Path(args.root), Path(args.out)
out.mkdir(parents=True, exist_ok=True)

# ---- 1. scan the ImageNet train folder -------------------------------------------------
wnids = sorted(d.name for d in os.scandir(root) if d.is_dir())
names = {w: w for w in wnids}
if args.mapping:
    for line in open(args.mapping):
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            names[parts[0]] = parts[1].split(",")[0].strip()          # keep the first name only
files = {}
for i, w in enumerate(wnids):
    files[w] = sorted(e.name for e in os.scandir(root / w) if e.name.lower().endswith((".jpeg", ".jpg", ".png")))
    if (i + 1) % 100 == 0:
        print(f"scanned {i + 1}/{len(wnids)} classes", flush=True)
counts = pd.Series({w: len(f) for w, f in files.items()})
print(f"{len(wnids)} classes, {counts.sum():,} images (min {counts.min()}, max {counts.max()} per class)", flush=True)

# ---- 2. full train/val/test split (per class, so every class is in every split) ---------
rows = []
for label, w in enumerate(wnids):
    f = files[w][:]
    rng.shuffle(f)
    n_te, n_va = round(len(f) * args.test_frac), round(len(f) * args.val_frac)
    for j, name in enumerate(f):
        split = "test" if j < n_te else ("val" if j < n_te + n_va else "train")
        rows.append((f"{w}/{name}", w, label, split))
full = pd.DataFrame(rows, columns=["path", "wnid", "label", "split"])
full.to_csv(out / "full_split.csv", index=False)
print("full split:", full.split.value_counts().to_dict(), flush=True)

# ---- 3. prototype subset ------------------------------------------------------------------
# Classes: evenly spaced through the sorted synset list. Sorted WordNet IDs roughly group
# related concepts (animals, then objects...), so even spacing covers all the super-categories
# instead of e.g. 100 dog breeds. Images: sampled from each split separately, keeping the
# 80/10/10 ratio, so prototype test images are also full-dataset test images.
step = len(wnids) / args.n_classes
proto_wnids = [wnids[int(i * step)] for i in range(args.n_classes)]
quota = {"test": round(args.per_class * args.test_frac), "val": round(args.per_class * args.val_frac)}
quota["train"] = args.per_class - quota["test"] - quota["val"]
parts = []
for new_label, w in enumerate(proto_wnids):
    for split, q in quota.items():
        pool = full[(full.wnid == w) & (full.split == split)]
        parts.append(pool.sample(min(q, len(pool)), random_state=args.seed).assign(label=new_label))
proto = pd.concat(parts, ignore_index=True)
proto["name"] = proto.wnid.map(names)
proto.to_csv(out / "proto_split.csv", index=False)
pd.DataFrame({"label": range(len(proto_wnids)), "wnid": proto_wnids, "name": [names[w] for w in proto_wnids],
              "available_in_imagenet": [counts[w] for w in proto_wnids]}).to_csv(out / "proto_classes.csv", index=False)
stats = proto.pivot_table(index=["label", "wnid", "name"], columns="split", values="path", aggfunc="count").fillna(0).astype(int)
stats = stats[["train", "val", "test"]]
stats["total"] = stats.sum(axis=1)
stats.to_csv(out / "proto_stats.csv")

# ---- 4. image statistics on a random sample of the prototype ------------------------------
sample = proto.sample(min(args.stats_sample, len(proto)), random_state=args.seed)
sizes, modes, means, sqs = [], [], [], []
for p in sample.path:
    with Image.open(root / p) as im:
        sizes.append(im.size)
        modes.append(im.mode)
        a = np.asarray(im.convert("RGB").resize((64, 64)), dtype=np.float64) / 255.0
    means.append(a.reshape(-1, 3).mean(0))
    sqs.append((a.reshape(-1, 3) ** 2).mean(0))
sizes = np.array(sizes)
mean = np.mean(means, 0)
std = np.sqrt(np.mean(sqs, 0) - mean ** 2)
modes = pd.Series(modes).value_counts()

summary = [
    "ImageNet (train folder) on Sol",
    f"  classes: {len(wnids)}   images: {counts.sum():,}   per class: min {counts.min()}, median {int(counts.median())}, max {counts.max()}",
    f"  full split: train {int((full.split == 'train').sum()):,} / val {int((full.split == 'val').sum()):,} / test {int((full.split == 'test').sum()):,}",
    "",
    f"Prototype subset (seed {args.seed})",
    f"  classes: {len(proto_wnids)} (every {step:g}-th synset of the sorted list)",
    f"  images per class: {args.per_class} -> train {quota['train']} / val {quota['val']} / test {quota['test']}",
    f"  total: {len(proto):,} images = {len(proto) / counts.sum():.2%} of ImageNet train",
    f"  split sizes: train {int((proto.split == 'train').sum()):,} / val {int((proto.split == 'val').sum()):,} / test {int((proto.split == 'test').sum()):,}",
    "",
    f"Image statistics (random sample of {len(sample)} prototype images)",
    f"  width : min {sizes[:, 0].min()}, median {int(np.median(sizes[:, 0]))}, max {sizes[:, 0].max()}",
    f"  height: min {sizes[:, 1].min()}, median {int(np.median(sizes[:, 1]))}, max {sizes[:, 1].max()}",
    f"  most common size: {pd.Series([str(w) + 'x' + str(h) for w, h in sizes]).value_counts().index[0]}",
    f"  color modes: {modes.to_dict()}",
    f"  RGB mean {mean.round(3).tolist()}   RGB std {std.round(3).tolist()}",
    "  (standard ImageNet values: mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225])",
]
(out / "proto_summary.txt").write_text("\n".join(summary) + "\n")
print("\n".join(summary), flush=True)

# ---- 5. figures ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
ax = axes[0, 0]
ax.bar(range(len(wnids)), counts.values, width=1.0, color="lightgray", label="all ImageNet classes")
idx = [wnids.index(w) for w in proto_wnids]
ax.bar(idx, counts[proto_wnids].values, width=3.0, color="tab:blue", label="prototype classes")
ax.axhline(args.per_class, color="tab:red", ls="--", label=f"prototype: {args.per_class} per class")
ax.set_xlabel("class index (sorted synset ID)"); ax.set_ylabel("images available")
ax.set_title("Images per class in ImageNet train, and the chosen classes"); ax.legend(fontsize=8)

ax = axes[0, 1]
sp = proto.split.value_counts()[["train", "val", "test"]]
bars = ax.bar(sp.index, sp.values, color=["tab:blue", "tab:orange", "tab:green"])
ax.bar_label(bars, labels=[f"{v:,}\n({v / len(proto):.0%})" for v in sp.values])
ax.set_title(f"Prototype split sizes ({len(proto_wnids)} classes, {len(proto):,} images)")
ax.margins(y=0.15)
ax.set_ylabel("images")

ax = axes[1, 0]
ax.scatter(sizes[:, 0], sizes[:, 1], s=4, alpha=0.3)
ax.set_xlabel("width (px)"); ax.set_ylabel("height (px)")
ax.set_title(f"Original image sizes (sample of {len(sample)})")

ax = axes[1, 1]
ax.hist(sizes[:, 0] / sizes[:, 1], bins=60, color="tab:purple")
ax.set_xlabel("aspect ratio (width / height)"); ax.set_ylabel("images")
ax.set_title("Aspect ratios: images are resized/cropped to 224x224 for training")
plt.tight_layout()
plt.savefig(out / "fig_proto_stats.png", dpi=130)
plt.close()

n_show = min(40, len(proto_wnids))
fig, axes = plt.subplots(5, 8, figsize=(16, 11))
for ax, w in zip(axes.ravel(), proto_wnids[:n_show]):
    p = proto[(proto.wnid == w) & (proto.split == "train")].path.iloc[0]
    with Image.open(root / p) as im:
        ax.imshow(im.convert("RGB").resize((160, 160)))
    ax.set_title(names[w][:22], fontsize=8)
for ax in axes.ravel():
    ax.axis("off")
plt.suptitle(f"Prototype dataset: one training image from each of the first {n_show} of {len(proto_wnids)} classes")
plt.tight_layout()
plt.savefig(out / "fig_proto_samples.png", dpi=110)
plt.close()
print(f"\nSaved manifests, statistics and figures to {out}/", flush=True)
