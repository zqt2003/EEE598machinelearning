"""Make a dataset of 64x64 images: black background, one colored square of random color, size, angle, position.

python make_squares.py --n 1000 --out data/squares_1k
python make_squares.py --n 5000 --out data/squares_5k      # if the GAN does not learn squares, make more

Writes PNGs to <out>/img/ and the parameters of every square to <out>/squares.csv.
Squares are drawn 4x larger and downsampled, so their edges are anti-aliased (smooth, not jagged).
"""
import argparse
import colorsys
import csv
import math
import os
import random

from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--size", type=int, default=64)
ap.add_argument("--min-side", type=float, default=12, help="smallest square side (pixels)")
ap.add_argument("--max-side", type=float, default=36, help="largest square side (pixels)")
ap.add_argument("--max-squares", type=int, default=1, help="squares per image (1..max-squares)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="data/squares_1k")
args = ap.parse_args()

rng = random.Random(args.seed)
os.makedirs(os.path.join(args.out, "img"), exist_ok=True)
SS = 4                                                   # supersampling factor
rows = []
for i in range(args.n):
    big = Image.new("RGB", (args.size * SS, args.size * SS), "black")
    draw = ImageDraw.Draw(big)
    for j in range(rng.randint(1, args.max_squares)):
        side = rng.uniform(args.min_side, args.max_side)
        angle = rng.uniform(0, 90)                       # a square looks the same every 90 degrees
        half_diag = side / math.sqrt(2)
        cx = rng.uniform(half_diag, args.size - half_diag)     # keep the whole square inside the image
        cy = rng.uniform(half_diag, args.size - half_diag)
        hue = rng.random()                               # bright, saturated random color
        r, g, b = (int(255 * c) for c in colorsys.hsv_to_rgb(hue, rng.uniform(0.6, 1.0), rng.uniform(0.7, 1.0)))
        corners = [(SS * (cx + half_diag * math.cos(math.radians(angle + 45 + 90 * k))),
                    SS * (cy + half_diag * math.sin(math.radians(angle + 45 + 90 * k)))) for k in range(4)]
        draw.polygon(corners, fill=(r, g, b))
        rows.append(dict(file=f"{i:05d}.png", square=j, cx=round(cx, 2), cy=round(cy, 2), side=round(side, 2),
                         angle=round(angle, 2), r=r, g=g, b=b))
    big.resize((args.size, args.size), Image.LANCZOS).save(os.path.join(args.out, "img", f"{i:05d}.png"))

with open(os.path.join(args.out, "squares.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print(f"wrote {args.n} images to {args.out}/img")
