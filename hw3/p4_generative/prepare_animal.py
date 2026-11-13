"""Build the animal dataset: 1,000-5,000 images, center-cropped to a square and resized to 256x256.

Which animals does Sol's ImageNet copy have? (it has 650 of the 1,000 classes; classes 0-397 are animals)
    python prepare_animal.py --imagenet /data/datasets/community/deeplearning/imagenet/train --list-animals
Option A, no download (ImageNet on Sol), using classes that list shows:
    python prepare_animal.py --imagenet /data/datasets/community/deeplearning/imagenet/train \
        --wnids n02510455 --max 5000 --out data/animal256
Option B, any folder(s) of images you downloaded (Kaggle AFHQ, icrawler, ...):
    python prepare_animal.py --src downloads/cat downloads/cat2 --max 5000 --out data/animal256
Search the class names for a word:
    python prepare_animal.py --imagenet /data/datasets/community/deeplearning/imagenet/train --find monkey

Removes images smaller than --min-side and exact/near duplicates (8x8 average hash), and writes
data/animal256/img/*.jpg. The montage figure for the report goes NEXT TO that folder
(data/animal256_montage.png), so the training scripts never mistake it for a training image.
"""
import argparse
import glob
import json
import os
import random
import re
import sys
import urllib.request

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
# the standard ImageNet class list (index -> [synset ID, name]); same file as problem3/make_mapping.py
CLASS_INDEX_URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"


def imagenet_classes(mapping=None, cache="imagenet_class_index.json"):
    """[(index, synset ID, name)] for all 1,000 classes (index -1 when read from a mapping file)."""
    if mapping:
        rows = [line.strip().split(maxsplit=1) for line in open(os.path.expanduser(mapping))]
        return [(-1, r[0], r[1]) for r in rows if len(r) == 2]
    if not os.path.exists(cache):
        urllib.request.urlretrieve(CLASS_INDEX_URL, cache)
    idx = json.load(open(cache))
    return [(int(i), w, n.replace("_", " ")) for i, (w, n) in sorted(idx.items(), key=lambda kv: int(kv[0]))]


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
    ap.add_argument("--list-animals", action="store_true", help="list the animal classes (0-397) this copy has, then exit")
    ap.add_argument("--mapping", default=None, help="optional synset -> name file; default: the standard class list")
    args = ap.parse_args()

    if args.find or args.list_animals:                   # what does this ImageNet copy contain?
        if not args.imagenet:
            sys.exit("--find / --list-animals need --imagenet <train folder>")
        present = set(os.listdir(args.imagenet))
        classes = imagenet_classes(args.mapping)
        idx_present = [i for i, w, _ in classes if w in present and i >= 0]
        if idx_present:
            print(f"this copy has {len(present)} class folders; standard class indices {min(idx_present)}..{max(idx_present)}")
        if args.list_animals:
            hits = [(i, w, n) for i, w, n in classes if 0 <= i <= 397 and w in present]
            print(f"{len(hits)} of the 398 animal classes are on this copy:")
        else:
            hits = [(i, w, n) for i, w, n in classes if re.search(rf"\b{re.escape(args.find)}\b", n, re.IGNORECASE)]
        ok = []
        for i, w, n in hits:
            k = len(os.listdir(os.path.join(args.imagenet, w))) if w in present else 0
            print(f"  {i:4d}  {w}  {k:5d} images  {n}" + ("" if k else "   <-- not on this copy"))
            if k:
                ok.append(w)
        if ok and not args.list_animals:
            print("use the ones with images:  --wnids " + " ".join(ok))
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
