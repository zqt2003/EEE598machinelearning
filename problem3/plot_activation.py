"""Plot the ArcGate activation and its derivative next to ReLU (and SiLU/GELU for context).

Example:  python plot_activation.py --out runs/activation.png
"""
import argparse
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from resnet_custom import ArcGate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="runs/activation.png")
args = ap.parse_args()

x = torch.linspace(-6, 6, 2001, dtype=torch.float64, requires_grad=True)
curves = {"ArcGate (ours)": ArcGate()(x), "ReLU": F.relu(x), "SiLU / Swish": F.silu(x), "GELU": F.gelu(x)}
styles = {"ArcGate (ours)": dict(lw=3, color="tab:red"), "ReLU": dict(lw=2, color="black"),
          "SiLU / Swish": dict(lw=1.2, ls="--", color="tab:blue"), "GELU": dict(lw=1.2, ls=":", color="tab:green")}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for name, y in curves.items():
    g, = torch.autograd.grad(y.sum(), x, retain_graph=True)
    axes[0].plot(x.detach(), y.detach(), label=name, **styles[name])
    axes[1].plot(x.detach(), g, label=name, **styles[name])
axes[0].axhline(-1 / math.pi, color="tab:red", lw=0.8, ls="--")
axes[0].text(-5.9, -1 / math.pi - 0.35, "ArcGate lower bound  -1/pi = -0.318", color="tab:red", fontsize=8)
axes[0].set_title("Activation f(x)"); axes[0].set_ylim(-1, 4); axes[0].set_xlim(-6, 4)
axes[1].set_title("Derivative f'(x)  (what backprop multiplies by)"); axes[1].set_ylim(-0.2, 1.2)

# Right panel: the negative side on a log scale -> ArcGate's gradient decays like 1/|x|^3 (polynomial), SiLU's exponentially
xn = torch.linspace(-20, -0.5, 500, dtype=torch.float64, requires_grad=True)
for name, fn in {"ArcGate (ours)": ArcGate(), "SiLU / Swish": F.silu, "GELU": F.gelu}.items():
    g, = torch.autograd.grad(fn(xn).sum(), xn)
    axes[2].semilogy(-xn.detach(), g.abs().clamp_min(1e-30), label=name, **styles[name])
axes[2].set_title("|f'(x)| for negative inputs (ReLU: exactly 0)")
axes[2].set_xlabel("-x"); axes[2].set_ylim(1e-12, 1)
for ax in axes:
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
axes[0].set_xlabel("x"); axes[1].set_xlabel("x")
plt.tight_layout()
plt.savefig(args.out, dpi=140)

# Key numbers for the report
xs = torch.linspace(-10, 0, 200001, dtype=torch.float64)
ys = ArcGate()(xs)
i = int(ys.argmin())
print(f"ArcGate minimum: f({xs[i]:.3f}) = {ys[i]:.4f}   (ReLU min 0; SiLU min {F.silu(torch.tensor(-1.2785)).item():.4f})")
print(f"f'(0) = 0.5, f(-1) = {ArcGate()(torch.tensor(-1.0)).item():.4f}, f(1) = {ArcGate()(torch.tensor(1.0)).item():.4f}")
print(f"saved {args.out}")
