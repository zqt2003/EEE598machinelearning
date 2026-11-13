"""Build the animal dataset: 1,000-5,000 images, center-cropped to a square and resized to 256x256.

Option A, no download (ImageNet on Sol):
    python prepare_animal.py --imagenet /data/datasets/community/deeplearning/imagenet/train --preset cat \
        --max 5000 --out data/animal256
Option B, any folder(s) of images you downloaded (Kaggle AFHQ, icrawler, ...):
    python prepare_animal.py --src downloads/cat downloads/cat2 --max 5000 --out data/animal256
Which ImageNet classes of an animal does Sol have? (needs the Canvas synset -> name file)
    python prepare_animal.py --imagenet /data/datasets/community/deeplearning/imagenet/train \
        --mapping ~/EE598_Assignment2/LOC_synset_mapping.txt --find cat

Removes images smaller than --min-side and exact/near duplicates (8x8 average hash), and writes
data/animal256/img/*.jpg. The montage figure for the report goes NEXT TO that folder
(data/animal256_montage.png), so the training scripts never mistake it for a training image.
"""
import argparse
import glob
import os
import random
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

# ImageNet synset IDs. The Sol copy has 650 of the 1,000 classes: the script reports which are missing.
PRESETS = {
    "cat": ["n02123045", "n02123159", "n02123394", "n02123597", "n02124075"],   # tabby, tiger cat, Persian, Siamese, Egyptian
    "fox": ["n02119022", "n02119789", "n02120079", "n02120505"],               # red, kit, Arctic, grey fox
    "panda": ["n02510455"],                                                     # giant panda (~1,300 images)
    "golden_retriever": ["n02099601"],
    "dog": ("range", "n02085620", "n02113978"),   # ALL dog breeds ImageNet has (118, Chihuahua ... Mexican hairless)
}
EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".JPEG")


def ahash(img):
    a = np.asarray(img.convert("L").resize((8, 8), Image.BILINEAR), dtype=np.float32)
    return np.packbits(a > a.mean()).tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="*", default=[], help="folders of images")
    ap.add_argument("--imagenet", default=None, help="ImageNet train folder (with one sub-folder per synset)")
    ap.add_argument("--preset", choices=list(PRESETS), default=None)
    ap.add_argument("--wnids", nargs="*", default=[], help="ImageNet synset IDs (instead of / in addition to --preset)")
    ap.add_argument("--max", type=int, default=5000)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--min-side", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/animal256")
    ap.add_argument("--find", default=None, help="list the ImageNet classes whose name contains this word, then exit")
    ap.add_argument("--mapping", default=None, help="synset -> name file (LOC_synset_mapping.txt from Canvas)")
    args = ap.parse_args()

    if args.find:                                        # which classes of this animal exist on this copy?
        if not (args.imagenet and args.mapping):
            sys.exit("--find needs --imagenet <train folder> and --mapping <LOC_synset_mapping.txt>")
        hits = []
        for line in open(os.path.expanduser(args.mapping)):
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2 and re.search(rf"\b{re.escape(args.find)}\b", parts[1], re.IGNORECASE):
                n = len(glob.glob(os.path.join(args.imagenet, parts[0], "*")))
                hits.append((n, parts[0], parts[1]))
        for n, w, name in sorted(hits, reverse=True):
            print(f"  {w}  {n:5d} images  {name}" + ("" if n else "   <-- not on this copy"))
        print(f"use the ones with images:  --wnids " + " ".join(w for n, w, _ in sorted(hits, reverse=True) if n))
        return

    files = []
    for d in args.src:
        files += [f for f in glob.glob(os.path.join(d, "**", "*"), recursive=True) if f.endswith(EXT)]
    if args.imagenet:
        preset = PRESETS[args.preset] if args.preset else []
        if isinstance(preset, tuple):                  # a range of synset IDs: take every one this copy has
            _, lo, hi = preset
            preset = sorted(d for d in os.listdir(args.imagenet) if lo <= d <= hi)
            print(f"preset {args.preset}: {len(preset)} classes found on this copy")
        wnids = preset + args.wnids
        for w in wnids:
            got = sorted(glob.glob(os.path.join(args.imagenet, w, "*")))
            print(f"  {w}: {len(got)} images" + ("   <-- not on this copy of ImageNet" if not got else ""))
            files += got
    random.Random(args.seed).shuffle(files)
    print(f"{len(files)} candidate images")

    os.makedirs(os.path.join(args.out, "img"), exist_ok=True)
    seen, kept, skipped_small, skipped_dup = set(), [], 0, 0
    for f in files:
        if len(kept) >= args.max:
            break
        try:
            img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
        except Exception:
            continue
        if min(img.size) < args.min_side:
            skipped_small += 1
            continue
        s = min(img.size)
        img = img.crop(((img.width - s) // 2, (img.height - s) // 2, (img.width + s) // 2, (img.height + s) // 2))
        img = img.resize((args.size, args.size), Image.LANCZOS)
        h = ahash(img)
        if h in seen:
            skipped_dup += 1
            continue
        seen.add(h)
        path = os.path.join(args.out, "img", f"{len(kept):05d}.jpg")
        img.save(path, quality=95)
        kept.append(path)
    print(f"kept {len(kept)} images ({skipped_small} too small, {skipped_dup} duplicates) -> {args.out}/img")
    if not kept:
        sys.exit("ERROR: no images were kept. If you used --preset/--wnids, those classes are not in this ImageNet "
                 "copy: run with --find <animal> --mapping <LOC_synset_mapping.txt> to see which ones are.")
    if len(kept) < 1000:
        print("WARNING: fewer than 1,000 images; add more sources (the assignment asks for 1,000-5,000).")

    show = kept[:48]
    fig, axes = plt.subplots(6, 8, figsize=(16, 12.5))
    for ax, p in zip(axes.flat, show):
        ax.imshow(Image.open(p)); ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.flat[len(show):]:
        ax.axis("off")
    fig.suptitle(f"Animal dataset: {len(kept)} images, {args.size}x{args.size} (48 random examples)", fontsize=16)
    montage = args.out.rstrip("/") + "_montage.png"       # outside the dataset folder (see docstring)
    plt.tight_layout(); plt.savefig(montage, dpi=90)
    print("montage:", montage)


if __name__ == "__main__":
    main()
