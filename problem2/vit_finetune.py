"""Problem 2: fine-tune ViT-B/32 on the balanced PV (InfraredSolarModules) dataset.

Runs the three fine-tuning strategies and saves, in --out:
  vit_results.csv           train / val / test accuracy and training time per strategy
  history_<strategy>.csv    per-epoch loss and accuracy
  test_preds_<strategy>.csv test-set predictions (for confusion matrices later)
  vit_b32_<strategy>.pt     best model weights (chosen by validation accuracy)
  learning_curves.png

Examples:
  python vit_finetune.py                              # all three strategies
  python vit_finetune.py --strategies head            # only one strategy
  python vit_finetune.py --epochs 20 --bs 64
"""
import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision.models import ViT_B_32_Weights, vit_b_32  # noqa: E402

LRS = {"head": 1e-3, "full": 5e-5, "last": 1e-4}
NAMES = {"head": "1) Classifier head only", "full": "2) Entire model", "last": "3) Last N blocks + head"}

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="PV_balanced", help="folder made by the balancing step (contains manifest.csv)")
ap.add_argument("--out", default="vit_results")
ap.add_argument("--strategies", nargs="+", default=["head", "full", "last"], choices=["head", "full", "last"])
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--n-last", type=int, default=4, help="number of transformer blocks to unfreeze in 'last'")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--subset", type=int, default=0, help="debug: max images per class and split (0 = all)")
ap.add_argument("--no-pretrained", action="store_true", help="debug: random weights instead of ImageNet")
args = ap.parse_args()

OUT_DIR, OUT = Path(args.data), Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "", flush=True)

# ---- split by original image (an original and its augmented copies stay together) ----
manifest = pd.read_csv(OUT_DIR / "manifest.csv", dtype={"source_id": str})
classes = sorted(manifest.cls.unique())
cls2idx = {c: i for i, c in enumerate(classes)}
rng = np.random.default_rng(args.seed)
split_of = {}
for c in classes:
    srcs = rng.permutation(manifest.loc[manifest.cls == c, "source_id"].unique().tolist())
    n_te = n_va = round(0.15 * len(srcs))
    for i, s in enumerate(srcs):
        split_of[s] = "test" if i < n_te else ("val" if i < n_te + n_va else "train")
manifest["split"] = manifest.source_id.map(split_of)
if args.subset:
    manifest = manifest.groupby(["cls", "split"]).head(args.subset)
print(pd.crosstab(manifest.cls, manifest.split, margins=True), flush=True)


def load_split(name):
    d = manifest[manifest.split == name]
    X = torch.from_numpy(np.stack([np.array(Image.open(OUT_DIR / c / f)) for c, f in zip(d.cls, d.file)]))
    y = torch.tensor([cls2idx[c] for c in d.cls])
    is_orig = torch.from_numpy((d.kind == "original").to_numpy().copy())
    return X, y, is_orig, d.file.to_numpy()


data = {s: load_split(s) for s in ["train", "val", "test"]}
MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")


def prep(xb):
    """uint8 N x 40 x 24 grayscale -> normalized N x 3 x 224 x 224."""
    x = xb.to(device, non_blocking=True).float().div(255).unsqueeze(1)
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    return (x.repeat(1, 3, 1, 1) - MEAN) / STD


def build(strategy):
    m = vit_b_32(weights=None if args.no_pretrained else ViT_B_32_Weights.IMAGENET1K_V1)
    m.heads.head = nn.Linear(m.heads.head.in_features, len(classes))
    for p in m.parameters():
        p.requires_grad = False
    parts = {"head": [m.heads], "full": [m],
             "last": [m.heads, m.encoder.ln, *list(m.encoder.layers)[-args.n_last:]]}[strategy]
    for part in parts:
        for p in part.parameters():
            p.requires_grad = True
    return m.to(device)


@torch.no_grad()
def evaluate(model, X, y, bs=512):
    model.eval()
    preds = []
    for i in range(0, len(X), bs):
        with torch.autocast(**amp):
            preds.append(model(prep(X[i:i + bs])).argmax(1).cpu())
    preds = torch.cat(preds)
    return (preds == y).float().mean().item(), preds


def train(strategy):
    lr, bs = LRS[strategy], args.bs
    model = build(strategy)
    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    n_all = sum(p.numel() for p in model.parameters())
    Xtr, ytr = data["train"][:2]
    Xva, yva = data["val"][:2]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, pct_start=0.1,
                                                total_steps=args.epochs * ((len(Xtr) + bs - 1) // bs))
    best_val, best_state, hist, train_time = -1.0, None, [], 0.0
    print(f"\n=== {NAMES[strategy]}: {n_train:,} trainable params ({100 * n_train / n_all:.2f}%), lr {lr} ===", flush=True)

    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        tot_loss = correct = 0
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i + bs]
            yb = ytr[idx].to(device)
            with torch.autocast(**amp):
                out = model(prep(Xtr[idx]))
                loss = F.cross_entropy(out, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot_loss += loss.item() * len(idx)
            correct += (out.argmax(1) == yb).sum().item()
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_time += time.time() - t0            # training time only (no evaluation)

        val_acc, _ = evaluate(model, Xva, yva)
        hist.append(dict(epoch=ep + 1, loss=tot_loss / len(Xtr), train_acc=correct / len(Xtr), val_acc=val_acc))
        print(f"[{strategy}] epoch {ep + 1:2d}/{args.epochs}  loss {hist[-1]['loss']:.4f}  "
              f"train {hist[-1]['train_acc']:.4f}  val {val_acc:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    torch.save(best_state, OUT / f"vit_b32_{strategy}.pt")
    Xte, yte, orig, files = data["test"]
    test_acc, test_preds = evaluate(model, Xte, yte)
    pd.DataFrame(dict(file=files, true=[classes[i] for i in yte], pred=[classes[i] for i in test_preds],
                      original=orig.numpy())).to_csv(OUT / f"test_preds_{strategy}.csv", index=False)
    pd.DataFrame(hist).to_csv(OUT / f"history_{strategy}.csv", index=False)
    res = dict(strategy=NAMES[strategy], trainable_params=n_train, trainable_pct=round(100 * n_train / n_all, 2),
               lr=lr, epochs=args.epochs,
               train_acc=evaluate(model, *data["train"][:2])[0],
               val_acc=evaluate(model, Xva, yva)[0],
               test_acc=test_acc,
               test_acc_originals_only=(test_preds[orig] == yte[orig]).float().mean().item(),
               train_time_min=train_time / 60)
    print(f"[{strategy}] best: train {res['train_acc']:.4f}  val {res['val_acc']:.4f}  test {res['test_acc']:.4f}  "
          f"time {res['train_time_min']:.1f} min", flush=True)
    return res, pd.DataFrame(hist)


results, histories = [], {}
for s in args.strategies:
    res, hist = train(s)
    results.append(res)
    histories[s] = hist
    # save after every strategy, so a crash or timeout later doesn't lose finished results
    pd.DataFrame(results).to_csv(OUT / "vit_results.csv", index=False)

print("\n" + pd.DataFrame(results).to_string(index=False), flush=True)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for s, h in histories.items():
    axes[0].plot(h.epoch, h.loss, "-o", label=NAMES[s])
    axes[1].plot(h.epoch, h.val_acc, "-o", label=NAMES[s])
axes[0].set_title("Training loss")
axes[1].set_title("Validation accuracy")
for ax in axes:
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.legend()
plt.tight_layout()
plt.savefig(OUT / "learning_curves.png", dpi=150)
print(f"Saved results to {OUT}/", flush=True)
