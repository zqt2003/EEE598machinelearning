"""Draw the report's charts from the numbers measured on Sol (all values copied from the run outputs).

Run:  python make_figures.py   (writes PDFs/PNGs into figs/)
Optional inputs for the PV figures: --pv-meta <InfraredSolarModules folder with module_metadata.json>
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="figs")
ap.add_argument("--pv-meta", default=None, help="folder containing module_metadata.json (RaptorMaps dataset)")
ap.add_argument("--daatsim-examples", default=None, help="folder with clean.png and turb_<r0>.png")
args = ap.parse_args()
OUT = Path(args.out)
OUT.mkdir(exist_ok=True)

# categorical slots (fixed order) + neutral ink
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e3e3e3"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
COL = 3.25  # CVPR column width in inches


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", OUT / name)


# ---------------- Problem 1.2: 1-pixel shift in pixel and VGG feature space ----------------
layers = ["relu1_2", "relu2_2", "relu3_3", "relu4_3"]
rel_1px = {"pixels": 0.0898, "relu1_2": 0.5904, "relu2_2": 0.5569, "relu3_3": 0.3785, "relu4_3": 0.3169}
shifts = [1, 2, 4, 8, 16]
raw = {"relu1_2": [0.613, 0.881, 0.992, 1.021, 1.044], "relu2_2": [0.595, 0.918, 1.146, 1.188, 1.212],
       "relu3_3": [0.400, 0.602, 0.915, 1.172, 1.240], "relu4_3": [0.342, 0.400, 0.488, 0.707, 1.004]}
stride = {"relu1_2": 1, "relu2_2": 2, "relu3_3": 4, "relu4_3": 8}
aligned = {"relu1_2": {1: 0.0, 2: 0.0, 4: 0.0, 8: 0.0, 16: 0.0}, "relu2_2": {2: 0.0, 4: 0.0, 8: 0.0, 16: 0.0},
           "relu3_3": {4: 0.002, 8: 0.002, 16: 0.002}, "relu4_3": {8: 0.009, 16: 0.009}}
lc = dict(zip(layers, [C1, C2, C3, C4]))
mk = dict(zip(layers, ["o", "s", "^", "D"]))

fig, ax = plt.subplots(1, 2, figsize=(2 * COL + 0.3, 2.0), gridspec_kw={"width_ratios": [1, 1.25]})
names = list(rel_1px)
bars = ax[0].bar(names, [rel_1px[n] for n in names], width=0.6,
                 color=["#9a9a9a"] + [lc[n] for n in layers], edgecolor="white", linewidth=1)
ax[0].bar_label(bars, labels=[f"{v:.3f}" for v in rel_1px.values()], fontsize=6.5, padding=1.5, color=INK)
ax[0].set_ylabel(r"relative change $\|\phi(x')-\phi(x)\|_2/\|\phi(x)\|_2$")
ax[0].set_title("(a) 1-px left shift of the 4032$\\times$3024 photo")
ax[0].set_ylim(0, 0.7)
ax[0].grid(axis="x", visible=False)
for n in layers:
    ax[1].plot(shifts, raw[n], marker=mk[n], ms=4, lw=1.6, color=lc[n], label=f"{n} (stride {stride[n]})")
    s = sorted(aligned[n])
    ax[1].plot(s, [aligned[n][k] for k in s], marker=mk[n], ms=4, lw=1.0, ls="--", color=lc[n], mfc="white")
ax[1].set_xscale("log", base=2)
ax[1].set_xticks(shifts, [str(s) for s in shifts])
ax[1].set_xlabel("horizontal shift (pixels)")
ax[1].set_ylabel("relative feature change")
ax[1].set_title("(b) raw (solid) vs. re-aligned by shift/stride (dashed)")
ax[1].set_ylim(-0.08, 1.62)
ax[1].legend(loc="upper left", ncol=2, fontsize=6.3)
save(fig, "p1_shift.pdf")

# ---------------- Problem 1.3: additive Gaussian noise ----------------
sig = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
noise = {"pixels": [0, 0.0196, 0.0391, 0.0967, 0.1901, 0.3597, 0.4990],
         "relu1_2": [0, 0.1135, 0.2225, 0.4994, 0.8743, 1.4845, 1.9575],
         "relu2_2": [0, 0.2171, 0.3780, 0.7459, 1.1974, 1.8867, 2.4211],
         "relu3_3": [0, 0.2506, 0.4035, 0.6880, 0.9235, 1.1796, 1.3699],
         "relu4_3": [0, 0.2960, 0.4580, 0.7331, 0.9291, 1.0828, 1.1608]}
fig, ax = plt.subplots(figsize=(COL, 2.25))
ax.plot(sig, noise["pixels"], marker="o", ms=3.5, lw=1.4, ls="--", color="#7a7a7a", label="pixels")
for n in layers:
    ax.plot(sig, noise[n], marker=mk[n], ms=3.5, lw=1.7, color=lc[n], label=n)
ax.set_xlabel(r"noise standard deviation $\sigma$ (pixel range [0, 1])")
ax.set_ylabel(r"relative change $\|\Delta\|_2/\|\phi\|_2$")
ax.set_title("Relative change under additive Gaussian noise")
ax.set_xticks([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
ax.set_ylim(-0.05, 2.6)
ax.legend(loc="upper left", ncol=2, fontsize=6.5)
save(fig, "p1_noise.pdf")

# ---------------- Problem 1.4: DAATSim examples (optional) ----------------
if args.daatsim_examples:
    from PIL import Image
    d = Path(args.daatsim_examples)
    panels = [("clean", d / "clean.png")] + [(f"$r_0$ = {r}", d / f"turb_{r}.png") for r in ["0.1", "0.05", "0.02", "0.01"]]
    fig, ax = plt.subplots(1, 5, figsize=(2 * COL + 0.3, 1.55))
    for a, (t, p) in zip(ax, panels):
        a.imshow(Image.open(p))
        a.set_title(t, fontsize=7.5)
        a.axis("off")
    fig.suptitle("DAATSim turbulence (tilt + blur): smaller $r_0$ = stronger turbulence", fontsize=8, y=1.02)
    save(fig, "p1_daatsim.pdf")

# ---------------- Problem 2.1: class balance (needs the dataset metadata) ----------------
if args.pv_meta:
    from PIL import Image
    root = Path(args.pv_meta)
    meta = json.load(open(root / "module_metadata.json"))
    counts = {}
    for v in meta.values():
        counts[v["anomaly_class"]] = counts.get(v["anomaly_class"], 0) + 1
    order = sorted(counts, key=counts.get)
    fig, ax = plt.subplots(figsize=(COL, 2.55))
    y = np.arange(len(order))
    kept = [min(counts[c], 2500) for c in order]
    added = [max(0, 2500 - counts[c]) for c in order]
    ax.barh(y, kept, height=0.62, color=C1, edgecolor="white", linewidth=1, label="original images kept")
    ax.barh(y, added, left=kept, height=0.62, color=C2, edgecolor="white", linewidth=1, label="augmented images added")
    over = [(i, counts[c]) for i, c in enumerate(order) if counts[c] > 2500]
    for i, n in over:
        ax.text(2560, i, f"{n:,} originals (down-sampled)", va="center", fontsize=6, color=MUTED)
    for i, c in enumerate(order):
        if counts[c] < 2500:
            ax.text(2560, i, f"{counts[c]:,} originals", va="center", fontsize=6, color=MUTED)
    ax.set_yticks(y, order)
    ax.set_xlim(0, 4300)
    ax.set_xticks([0, 1000, 2000, 2500])
    ax.axvline(2500, color=INK, lw=0.8)
    ax.set_xlabel("images per class after balancing")
    ax.set_title("PV dataset: balancing every class to 2,500 images", pad=16)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6.3)
    ax.grid(axis="y", visible=False)
    save(fig, "p2_counts.pdf")
    json.dump(counts, open(OUT / "p2_counts.json", "w"), indent=1)

    rng = np.random.default_rng(42)
    show = ["Cell", "Diode", "Hot-Spot", "Shadowing", "Vegetation"]
    by_cls = {}
    for v in meta.values():
        by_cls.setdefault(v["anomaly_class"], []).append(v["image_filepath"])
    fig, ax = plt.subplots(1, 5, figsize=(COL, 1.5), gridspec_kw={"wspace": 0.45})
    for a, c in zip(ax, show):
        p = by_cls[c][rng.integers(len(by_cls[c]))]
        a.imshow(np.asarray(Image.open(root / p)), cmap="inferno", interpolation="nearest")
        a.set_title(c, fontsize=6.2)
        a.set_xticks([]); a.set_yticks([])
        a.grid(False)
    fig.suptitle("Sample infrared images (24$\\times$40 px) with labels; brighter = hotter", fontsize=7.5, y=0.99)
    save(fig, "p2_samples.pdf")

# ---------------- Problem 3.3: ArcGate vs. ReLU ----------------
x = np.linspace(-6, 4, 2001)
arc = x * (0.5 + np.arctan(x) / np.pi)
darc = 0.5 + np.arctan(x) / np.pi + x / (np.pi * (1 + x ** 2))
relu, drelu = np.maximum(x, 0), (x > 0).astype(float)
sil = x / (1 + np.exp(-x)); s = 1 / (1 + np.exp(-x)); dsil = s * (1 + x * (1 - s))
fig, ax = plt.subplots(1, 2, figsize=(2 * COL + 0.3, 1.95))
for a, ys, title, yl in [(ax[0], (arc, relu, sil), "(a) activation $f(x)$", (-1.0, 4.0)),
                         (ax[1], (darc, drelu, dsil), "(b) derivative $f'(x)$ used by back-propagation", (-0.2, 1.25))]:
    a.plot(x, ys[1], color=INK, lw=1.3, label="ReLU")
    a.plot(x, ys[2], color=C1, lw=1.1, ls="--", label="SiLU (reference)")
    a.plot(x, ys[0], color=C2, lw=2.2, label="ArcGate (ours)")
    a.set_title(title); a.set_xlabel("pre-activation $x$"); a.set_ylim(*yl); a.set_xlim(-6, 4)
ax[0].axhline(-1 / math.pi, color=C2, lw=0.7, ls=":")
ax[0].text(-5.9, -1 / math.pi - 0.33, r"lower bound $-1/\pi$", color=MUTED, fontsize=6.5)
ax[0].set_ylabel("$f(x)$"); ax[1].set_ylabel("$f'(x)$")
ax[0].legend(loc="upper left")
save(fig, "p3_activation.pdf")

# ---------------- Problem 3.2/3.3: prototype results (100 epochs, 2 seeds) ----------------
models = ["ResNet-34\nReLU", "ResNet-36\nReLU", "ResNet-36\nArcGate", "ResNet-36\nKAN head"]
train = np.array([[91.92, 91.61], [91.60, 92.05], [88.56, 88.73], [93.41, 92.84]])
test = np.array([[62.47, 62.47], [62.10, 64.03], [62.67, 62.40], [60.40, 61.20]])
fig, ax = plt.subplots(figsize=(COL, 2.25))
xi = np.arange(len(models)); w = 0.36
for off, data, col, lab in [(-w / 2, train, C1, "train top-1"), (w / 2, test, C2, "test top-1")]:
    b = ax.bar(xi + off, data.mean(1), width=w, color=col, edgecolor="white", linewidth=1, label=lab + " (mean)")
    ax.bar_label(b, labels=[f"{v:.1f}" for v in data.mean(1)], fontsize=6.3, padding=1.5)
    ax.scatter(np.repeat(xi + off, 2), data.ravel(), s=9, color=INK, zorder=3, lw=0)
gap = train.mean(1) - test.mean(1)
for i, g in enumerate(gap):
    ax.text(i, 99.5, f"gap {g:.1f}", ha="center", fontsize=6.5, color=MUTED)
ax.set_xticks(xi, models)
ax.set_ylim(50, 112)
ax.set_yticks([50, 60, 70, 80, 90, 100])
ax.set_ylabel("top-1 accuracy (%)")
ax.set_title("Prototype (100 classes), 100 epochs; dots = the two seeds")
ax.legend(loc="upper center", ncol=2, fontsize=6.3)
ax.grid(axis="x", visible=False)
save(fig, "p3_proto_results.pdf")

# ---------------- Problem 3.4: full ImageNet learning curve (A100 run) ----------------
ep = np.arange(1, 11)
tr = [3.02, 15.26, 25.48, 32.32, 37.23, 41.41, 45.20, 48.93, 52.36, 54.43]
v1 = [8.98, 23.03, 31.92, 39.42, 42.98, 48.60, 51.95, 56.25, 59.19, 59.97]
v5 = [23.14, 45.94, 57.00, 65.34, 68.60, 73.78, 76.86, 79.90, 82.05, 82.56]
fig, ax = plt.subplots(figsize=(COL, 2.05))
ax.plot(ep, v5, marker="^", ms=3.5, lw=1.7, color=C3, label="val top-5")
ax.plot(ep, v1, marker="o", ms=3.5, lw=1.7, color=C1, label="val top-1")
ax.plot(ep, tr, marker="s", ms=3.5, lw=1.7, color=C2, label="train top-1 (augmented)")
ax.annotate("test 60.4 / 82.9", (10, 60.4), xytext=(-62, 6), textcoords="offset points", fontsize=6.5, color=INK,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
ax.set_xticks(ep)
ax.set_xlabel("epoch"); ax.set_ylabel("accuracy (%)")
ax.set_title("ResNet-36 + ArcGate, full ImageNet (650 classes), 2$\\times$A100")
ax.legend(loc="lower right")
save(fig, "p3_full_curve.pdf")

# ---------------- Problem 3.4: A100 vs. Gaudi 2 ----------------
fig, ax = plt.subplots(1, 3, figsize=(2 * COL + 0.3, 1.75), gridspec_kw={"wspace": 0.42})
plat = ["2$\\times$A100", "2$\\times$Gaudi 2"]
for a, vals, title, fmt in [(ax[0], [53.85, 46.76], "(a) training time, 10 epochs", "{:.1f}"),
                            (ax[1], [2121, 2381], "(b) training throughput", "{:,.0f}"),
                            (ax[2], None, "(c) inference latency, 1 device", "{:.2f}")]:
    if vals is not None:
        b = a.bar(plat, vals, width=0.55, color=[C1, C2], edgecolor="white", linewidth=1)
        a.bar_label(b, labels=[fmt.format(v) for v in vals], fontsize=6.5, padding=1.5)
        a.set_ylim(0, max(vals) * 1.18)
    a.set_title(title)
    a.grid(axis="x", visible=False)
xi = np.arange(2); w = 0.36
for off, vals, col, lab in [(-w / 2, [6.207, 7.724], C1, "A100"), (w / 2, [9.154, 9.351], C2, "Gaudi 2")]:
    b = ax[2].bar(xi + off, vals, width=w, color=col, edgecolor="white", linewidth=1, label=lab)
    ax[2].bar_label(b, labels=[f"{v:.2f}" for v in vals], fontsize=6.3, padding=1.5)
ax[2].set_xticks(xi, ["batch 1", "batch 64"]); ax[2].set_ylim(0, 13)
ax[2].legend(loc="upper left", ncol=2, fontsize=6.3)
ax[0].set_ylabel("minutes"); ax[1].set_ylabel("images / s"); ax[2].set_ylabel("ms per forward pass")
save(fig, "p3_gpu_gaudi.pdf")

# ---------------- Problem 3.5: ResNet vs. ViT ----------------
fig, ax = plt.subplots(figsize=(COL, 2.1))
groups = ["prototype\n(100 cl., 100 ep.)", "full ImageNet\n(650 cl., 10 ep.)"]
res_tr, res_te = [88.56, 54.43], [62.67, 60.38]
vit_tr, vit_te = [87.52, 44.48], [35.63, 45.43]
xi = np.arange(2); w = 0.2
for off, vals, col, hatch, lab in [(-1.5 * w, res_tr, C1, "", "ResNet-36+ArcGate, train"),
                                   (-0.5 * w, res_te, C1, "////", "ResNet-36+ArcGate, test"),
                                   (0.5 * w, vit_tr, C2, "", "ViT-S/16, train"),
                                   (1.5 * w, vit_te, C2, "////", "ViT-S/16, test")]:
    b = ax.bar(xi + off, vals, width=w, color=col if not hatch else "white", edgecolor=col, hatch=hatch,
               linewidth=1, label=lab)
    ax.bar_label(b, labels=[f"{v:.1f}" for v in vals], fontsize=6, padding=1.2)
ax.set_xticks(xi, groups)
ax.set_ylim(0, 128)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_ylabel("top-1 accuracy (%)")
ax.set_title("Custom ResNet vs. ViT trained from scratch (seed 0)")
ax.legend(loc="upper center", ncol=2, fontsize=5.8)
ax.grid(axis="x", visible=False)
save(fig, "p3_vit_vs_resnet.pdf")
