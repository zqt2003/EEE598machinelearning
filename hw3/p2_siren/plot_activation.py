"""Graph the activations and their 1st and 2nd derivatives (computed with autograd).

python plot_activation.py --out activation.png
"""
import argparse
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from activations import ACTIVATIONS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="activation.png")
args = ap.parse_args()

x = torch.linspace(-2 * math.pi, 2 * math.pi, 4001, dtype=torch.float64, requires_grad=True)
styles = {"sine": dict(color="black", lw=2), "parawave": dict(color="tab:red", lw=2.5),
          "triangle": dict(color="tab:blue", lw=1.2, ls="--")}
fig, axes = plt.subplots(1, 3, figsize=(18, 4.6))
for name, st in styles.items():
    y = ACTIVATIONS[name](x)
    d1, = torch.autograd.grad(y.sum(), x, create_graph=True)
    d2, = torch.autograd.grad(d1.sum(), x)
    label = "parawave (ours)" if name == "parawave" else name
    for ax, v in zip(axes, [y, d1, d2]):
        ax.plot(x.detach(), v.detach(), label=label, **st)
for ax, t, yl in zip(axes, ["Activation f(x)", "First derivative f'(x)", "Second derivative f''(x)"],
                     ["f(x)", "f'(x)", "f''(x)"]):
    ax.set_title(t); ax.set_xlabel("x"); ax.set_ylabel(yl); ax.grid(alpha=0.3); ax.legend(loc="upper right")
    ax.set_xticks([k * math.pi / 2 for k in range(-4, 5)])
    ax.set_xticklabels([f"{k / 2:g}$\\pi$" if k else "0" for k in range(-4, 5)])
axes[2].set_ylim(-1.5, 1.5)
plt.tight_layout(); plt.savefig(args.out, dpi=140)

xs = torch.linspace(-10, 10, 200001, dtype=torch.float64)
print(f"max |parawave - sin| = {(ACTIVATIONS['parawave'](xs) - torch.sin(xs)).abs().max():.4f}")
print(f"max |triangle - sin| = {(ACTIVATIONS['triangle'](xs) - torch.sin(xs)).abs().max():.4f}")
print("saved", args.out)
