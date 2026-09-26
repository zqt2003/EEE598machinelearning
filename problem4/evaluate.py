"""Compare standard features with the trained turbulence-robust features on the held-out test set.

For every test image x we compute distances d(x, .) to
  * T(x):      the same image under turbulence, at each strength r0 (should be SMALL)
  * edit(x):   x with a square patch replaced by another image's content (a real change; should be LARGE)
  * y:         a completely different image (sets the scale of "very different")
and report
  * AUC_edit = P(d(x, T(x)) < d(x, edit(x)))   -> 1.0 means turbulence never looks like a real change
  * ratio    = median d(x, T(x)) / median d(x, y) -> lower means more turbulence-invariant

Example:
    python evaluate.py --data data --ckpt "Proposed=ckpt/turbfeat.pt" \
        --ckpt "No anti-alias=ckpt/no_antialias.pt" --ckpt "Invariance only=ckpt/inv_only.pt"
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchvision.transforms.functional as TF  # noqa: E402

from p4lib import (TurbFeature, VGGTrunk, auc_smaller, content_edit, distance_map,  # noqa: E402
                   feature_distance, load_image, read_pairs)

TAPS = ["relu1_2", "relu2_2", "relu3_3", "relu4_3"]
PIXEL_METHODS = {"Pixels (MSE)", "Blurred pixels (σ=2)"}
D_APERTURE = 0.2   # DAATSim's aperture diameter, used to report D/r0


def load_learned(ckpts, device, pretrained):
    learned = {}
    for spec in ckpts:
        name, path = spec.split("=", 1)
        ck = torch.load(path, map_location=device)
        m = TurbFeature(antialias=ck["config"]["antialias"], dim=ck["config"]["dim"], pretrained=pretrained)
        m.head.load_state_dict(ck["head"])
        learned[name] = m.to(device).eval()
    return learned


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--ckpt", action="append", default=[], help='"Name=path/to/ckpt.pt" (repeatable)')
    ap.add_argument("--out", default="results")
    ap.add_argument("--patch", type=int, default=64, help="side of the pasted patch in the content edit")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--qual-index", type=int, default=0, help="which test image to show in the qualitative figure")
    ap.add_argument("--no-pretrained", action="store_true", help="random VGG weights (debugging only)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained = not args.no_pretrained
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    root = Path(args.data)
    table = read_pairs(root, "test")
    ids = list(table)
    levels = [r0 for _, r0, _ in table[ids[0]]]
    print(f"{len(ids)} test images, turbulence levels r0 = {levels}")

    vgg = VGGTrunk(TAPS, antialias=False, pretrained=pretrained).to(device)
    aa_vgg = VGGTrunk(("relu3_3",), antialias=True, pretrained=pretrained).to(device)
    learned = load_learned(args.ckpt, device, pretrained)

    def features(x):
        v, a = vgg(x), aa_vgg(x)
        f = {"Pixels (MSE)": [x], "Blurred pixels (σ=2)": [TF.gaussian_blur(x, 13, 2.0)]}
        for t in TAPS:
            f[f"VGG {t}"] = [v[t]]
        f["VGG 4-layer (LPIPS-style)"] = [v[t] for t in TAPS]
        f["AA-VGG relu3_3 (untrained)"] = [a["relu3_3"]]
        for name, m in learned.items():
            f[name] = [m(x)]
        return f

    def dists(fa, fb):
        return {n: feature_distance(fa[n], fb[n], normalize=n not in PIXEL_METHODS).cpu().numpy() for n in fa}

    gen = torch.Generator().manual_seed(0)
    d_turb, d_edit, d_diff = {}, {}, {}
    for s in range(0, len(ids), args.bs):
        batch = ids[s:s + args.bs]
        donors = [ids[(ids.index(c) + len(ids) // 2) % len(ids)] for c in batch]   # a different scene
        X = torch.stack([load_image(root / "clean" / f"{c}.png") for c in batch]).to(device)
        Y = torch.stack([load_image(root / "clean" / f"{c}.png") for c in donors]).to(device)
        E = content_edit(X, Y, args.patch, gen)
        fX = features(X)
        for n, d in dists(fX, features(Y)).items():
            d_diff.setdefault(n, []).extend(d)
        for n, d in dists(fX, features(E)).items():
            d_edit.setdefault(n, []).extend(d)
        for li in range(len(levels)):
            T = torch.stack([load_image(table[c][li][2]) for c in batch]).to(device)
            for n, d in dists(fX, features(T)).items():
                d_turb.setdefault((n, li), []).extend(d)
        print(f"  {min(s + args.bs, len(ids))}/{len(ids)} test images")

    methods = list(d_diff)
    rows = []
    for n in methods:
        for li, r0 in enumerate(levels):
            dt = np.array(d_turb[(n, li)])
            rows.append(dict(method=n, r0=r0, D_over_r0=D_APERTURE / r0,
                             auc_edit=auc_smaller(dt, d_edit[n]), auc_diff=auc_smaller(dt, d_diff[n]),
                             ratio=float(np.median(dt) / np.median(d_diff[n]))))
    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    def get(n, key):
        return [r[key] for r in rows if r["method"] == n]

    dr0 = [D_APERTURE / r0 for r0 in levels]
    print("\nAUC_edit = P(turbulent copy closer than a content edit)   [higher = better]")
    print(f"{'method':<30}" + "".join(f"{'D/r0=' + format(d, '.3g'):>11}" for d in dr0) + f"{'  ratio@strongest':>18}")
    for n in methods:
        print(f"{n:<30}" + "".join(f"{a:>11.3f}" for a in get(n, "auc_edit")) + f"{get(n, 'ratio')[-1]:>18.3f}")

    # ---- Figure 1: quantitative curves
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for n in methods:
        style = "-o" if n in learned else ("--s" if n in PIXEL_METHODS else ":^")
        lw = 2.5 if n in learned else 1.3
        axes[0].plot(dr0, get(n, "auc_edit"), style, lw=lw, label=n)
        axes[1].plot(dr0, get(n, "ratio"), style, lw=lw, label=n)
    for ax in axes:
        ax.set_xscale("log"); ax.set_xlabel("turbulence strength D/r0  (log scale)")
        ax.axvspan(D_APERTURE / 0.1, D_APERTURE / 0.02, color="gray", alpha=0.12, label="training range")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("AUC_edit  (higher = better)")
    axes[0].set_title("Is the turbulent copy judged closer than a real content change?")
    axes[1].set_yscale("log"); axes[1].set_ylabel("median d(x,T(x)) / median d(x,y)  (lower = better)")
    axes[1].set_title("Turbulence distance relative to a completely different image")
    axes[1].legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    plt.tight_layout(); plt.savefig(out / "fig_quantitative.png", dpi=150, bbox_inches="tight"); plt.close()

    # ---- Figure 2: qualitative distance maps (shared color scale per column)
    q = ids[args.qual_index]
    X = load_image(root / "clean" / f"{q}.png")[None].to(device)
    Y = load_image(root / "clean" / f"{ids[(args.qual_index + len(ids) // 2) % len(ids)]}.png")[None].to(device)
    show_levels = [li for li, r0 in enumerate(levels) if r0 in (0.05, 0.02)] or [0, len(levels) - 1]
    inputs = [(f"turbulence r0={levels[li]} (D/r0={D_APERTURE / levels[li]:g})",
               load_image(table[q][li][2])[None].to(device)) for li in show_levels]
    inputs.append((f"content edit ({args.patch}px patch)", content_edit(X, Y, args.patch, torch.Generator().manual_seed(1))))
    shown = ["Pixels (MSE)", "VGG relu3_3", "VGG 4-layer (LPIPS-style)"] + list(learned)
    fX = features(X)
    maps = {}
    for n in shown:
        scale = float(np.median(d_diff[n]))                   # express change relative to "different image"
        for label, img in inputs:
            m = distance_map(fX[n], features(img)[n], normalize=n not in PIXEL_METHODS)[0] / scale
            maps[(n, label)] = torch.nn.functional.interpolate(m[None, None], size=X.shape[-2:], mode="bilinear")[0, 0].cpu().numpy()
    fig, axes = plt.subplots(len(inputs), len(shown) + 1, figsize=(3.3 * (len(shown) + 1), 3.4 * len(inputs)))
    for j, n in enumerate(shown):
        vmax = max(np.percentile(maps[(n, label)], 99.5) for label, _ in inputs)
        for i, (label, img) in enumerate(inputs):
            im = axes[i, j + 1].imshow(maps[(n, label)], cmap="magma", vmin=0, vmax=vmax)
            axes[i, j + 1].set_title(f"{n}\nrel. dist {maps[(n, label)].mean():.3f}", fontsize=9)
        fig.colorbar(im, ax=axes[:, j + 1], shrink=0.6)
    for i, (label, img) in enumerate(inputs):
        axes[i, 0].imshow(img[0].permute(1, 2, 0).cpu().numpy()); axes[i, 0].set_title(label, fontsize=9)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("Where each feature changes (divided by that feature's distance to a different image; "
                 "shared color scale per column). Ideal: dark for turbulence, bright on the edited patch.")
    plt.savefig(out / "fig_qualitative.png", dpi=130, bbox_inches="tight"); plt.close()

    # ---- Figure 3: dataset examples
    n_show = min(4, len(ids))
    fig, axes = plt.subplots(n_show, len(levels) + 1, figsize=(2.6 * (len(levels) + 1), 2.6 * n_show), squeeze=False)
    for i in range(n_show):
        c = ids[i * len(ids) // n_show]
        axes[i, 0].imshow(load_image(root / "clean" / f"{c}.png").permute(1, 2, 0).numpy()); axes[i, 0].set_title("clean")
        for li, (_, r0, p) in enumerate(table[c]):
            axes[i, li + 1].imshow(load_image(p).permute(1, 2, 0).numpy())
            axes[i, li + 1].set_title(f"r0={r0}  D/r0={D_APERTURE / r0:g}")
    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout(); plt.savefig(out / "fig_dataset.png", dpi=110); plt.close()
    print(f"\nSaved metrics.csv and figures to {out}/")


if __name__ == "__main__":
    main()
