"""All Problem 3 figures from one selfie.

python run_p3.py --image selfie.jpg --out results
    [--seg-model facebook/mask2former-swin-large-coco-panoptic] [--depth-model depth-anything/Depth-Anything-V2-Large-hf]

Writes to --out:
    input_512.png, mask.png, background_blur.png, depth.npy, lens_blur.png and the figures
    fig1_segmentation.png  fig2_background_blur.png  fig3_depth.png  fig4_lens_blur.png
    fig5_halo_comparison.png (extra: why the mask-aware blur matters)
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import blur_lib as bl  # noqa: E402


def show(ax, img, title, cmap=None, cbar_label=None):
    im = ax.imshow(img, cmap=cmap)
    ax.set_title(title); ax.set_xlabel("x (pixels)"); ax.set_ylabel("y (pixels)")
    if cbar_label:
        plt.colorbar(im, ax=ax, fraction=0.046).set_label(cbar_label)


def pair(a, b, ta, tb, path, cmap_b=None, cbar_b=None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    show(axes[0], a, ta)
    show(axes[1], b, tb, cmap=cmap_b, cbar_label=cbar_b)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seg-model", default=bl.DEFAULT_SEG_MODEL)
    ap.add_argument("--depth-model", default=bl.DEFAULT_DEPTH_MODEL)
    ap.add_argument("--depth-output", choices=["auto", "disparity", "depth"], default="auto")
    ap.add_argument("--bg-sigma", type=float, default=15.0)
    ap.add_argument("--max-sigma", type=float, default=15.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # 1. preprocess + segmentation
    img_pil = bl.load_square(args.image, args.size)
    img_pil.save(os.path.join(args.out, "input_512.png"))
    img = bl.to_float(img_pil)
    mask, labels = bl.person_mask(bl.load_segmenter(args.seg_model), img_pil)
    print("segments found:", labels, f"| person covers {mask.mean():.1%} of the image")
    Image.fromarray((mask * 255).astype(np.uint8)).save(os.path.join(args.out, "mask.png"))
    pair(img, mask, f"Input ({args.size}x{args.size})", f"Person mask ({args.seg_model.split('/')[-1]})",
         os.path.join(args.out, "fig1_segmentation.png"), cmap_b="gray")

    # 2. background blur, sigma = 15
    bgb = bl.background_blur(img, mask, sigma=args.bg_sigma)
    Image.fromarray(bl.to_uint8(bgb)).save(os.path.join(args.out, "background_blur.png"))
    pair(img, bgb, "Input", f"Background blurred (Gaussian, sigma = {args.bg_sigma:g})",
         os.path.join(args.out, "fig2_background_blur.png"))

    # extra: naive vs mask-aware blur, zoomed on the silhouette edge
    naive = bl.background_blur(img, mask, sigma=args.bg_sigma, mask_aware=False)
    ys, xs = np.nonzero(mask ^ np.roll(mask, 1, axis=1))               # a point on the silhouette edge
    cy, cx = (int(np.median(ys)), int(np.median(xs))) if len(ys) else (args.size // 2, args.size // 2)
    r = 64
    y0, x0 = np.clip(cy - r, 0, args.size - 2 * r), np.clip(cx - r, 0, args.size - 2 * r)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    show(axes[0], naive[y0:y0 + 2 * r, x0:x0 + 2 * r], "Naive: blur(image), then paste person")
    show(axes[1], bgb[y0:y0 + 2 * r, x0:x0 + 2 * r], "Mask-aware: blur(image * bg) / blur(bg)")
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "fig5_halo_comparison.png"), dpi=130); plt.close(fig)

    # 3. depth
    depth = bl.estimate_depth(bl.load_depth_model(args.depth_model), img_pil, args.depth_output)
    np.save(os.path.join(args.out, "depth.npy"), depth)
    pair(img, depth, "Input", f"Depth ({args.depth_model.split('/')[-1]}), bright = far",
         os.path.join(args.out, "fig3_depth.png"), cmap_b="magma", cbar_b="relative distance (larger = farther)")

    # 4. depth-based lens blur: sigma(x, y) = depth normalized to [0, 15]
    sigma_map = bl.normalize(depth, 0, args.max_sigma)
    lens = bl.depth_lens_blur(img, sigma_map, args.max_sigma)
    Image.fromarray(bl.to_uint8(lens)).save(os.path.join(args.out, "lens_blur.png"))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    show(axes[0], img, "Input")
    show(axes[1], sigma_map, "Blur map: depth normalized to [0, 15]", cmap="viridis", cbar_label="Gaussian sigma (px)")
    show(axes[2], lens, "Depth-based lens blur (far = more blur)")
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "fig4_lens_blur.png"), dpi=130); plt.close(fig)
    pair(img, lens, "Input", "Depth-based lens blur", os.path.join(args.out, "fig4b_lens_blur_pair.png"))
    print("saved figures to", args.out)


if __name__ == "__main__":
    main()
