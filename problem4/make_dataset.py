"""Build a paired clean / turbulent dataset with DAATSim.

For every source image we take random square crops (resized to --size). Each crop
is one "clean" image. Train crops get --n-turb turbulent versions with random r0;
test crops get one turbulent version at every fixed strength in R0_TEST_LEVELS,
so evaluation is deterministic. Train/test are split by *source image*, so no
scene appears in both.

Example:
    python make_dataset.py --src photos/ --daatsim DAATSim --out data --crops-per-image 8
"""
import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from turbulence import DAATSimTurbulence

R0_TEST_LEVELS = [0.1, 0.05, 0.03, 0.02, 0.01]   # smaller r0 = stronger turbulence
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def random_crop(img, size, rng):
    """Random square crop covering 40-100% of the shorter side, resized to size x size."""
    w, h = img.size
    side = int(min(w, h) * rng.uniform(0.4, 1.0))
    side = max(side, min(size, w, h))
    x0, y0 = rng.randint(0, w - side), rng.randint(0, h - side)
    return img.crop((x0, y0, x0 + side, y0 + side)).resize((size, size), Image.BICUBIC)


def to_tensor(img):
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)


def save(t, path):
    Image.fromarray((t.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder of source photos (searched recursively)")
    ap.add_argument("--daatsim", required=True, help="path to the cloned DAATSim repository")
    ap.add_argument("--out", default="data")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--crops-per-image", type=int, default=8)
    ap.add_argument("--max-clean", type=int, default=1000, help="cap on the number of clean crops")
    ap.add_argument("--n-turb", type=int, default=3, help="turbulent versions per train crop")
    ap.add_argument("--r0-min", type=float, default=0.02)
    ap.add_argument("--r0-max", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    sources = sorted(p for p in Path(args.src).rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not sources:
        raise SystemExit(f"No images found in {args.src}")
    rng.shuffle(sources)
    n_test_src = max(1, round(len(sources) * args.test_frac))
    split_of = {p: ("test" if i < n_test_src else "train") for i, p in enumerate(sources)}

    out = Path(args.out)
    (out / "clean").mkdir(parents=True, exist_ok=True)
    (out / "turb").mkdir(parents=True, exist_ok=True)
    sim = DAATSimTurbulence(args.daatsim, size=args.size)
    print(f"{len(sources)} source images ({n_test_src} held out for test); device {sim.device}")

    rows, n_clean = [], 0
    for src in sources:
        img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
        if min(img.size) < args.size // 2:
            continue
        for _ in range(args.crops_per_image):
            if n_clean >= args.max_clean:
                break
            cid = f"{n_clean:05d}"
            clean = to_tensor(random_crop(img, args.size, rng))
            save(clean, out / "clean" / f"{cid}.png")
            split = split_of[src]
            if split == "train":
                r0s = [float(np.exp(rng.uniform(np.log(args.r0_min), np.log(args.r0_max))))
                       for _ in range(args.n_turb)]
            else:
                r0s = R0_TEST_LEVELS
            for k, r0 in enumerate(r0s):
                save(sim(clean, r0), out / "turb" / f"{cid}_{k}.png")
                rows.append(dict(id=cid, k=k, r0=f"{r0:.4f}", split=split, source=src.name))
            n_clean += 1
        if n_clean % 50 == 0 or n_clean >= args.max_clean:
            print(f"  {n_clean} clean images done")
        if n_clean >= args.max_clean:
            break

    with open(out / "pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "k", "r0", "split", "source"])
        w.writeheader()
        w.writerows(rows)
    n_train = len({r["id"] for r in rows if r["split"] == "train"})
    print(f"Done: {n_clean} clean images ({n_train} train, {n_clean - n_train} test), "
          f"{len(rows)} turbulent images -> {out}")


if __name__ == "__main__":
    main()
