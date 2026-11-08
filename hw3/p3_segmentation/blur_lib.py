"""Selfie segmentation, background blur, monocular depth and depth-based lens blur.

Shared by run_p3.py (report figures), app.py (Hugging Face Space) and the Colab notebook.
Images are float32 arrays in [0, 1], shape (H, W, 3); masks are bool (H, W); depth maps float (H, W).
"""
import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

DEFAULT_SEG_MODEL = "facebook/mask2former-swin-large-coco-panoptic"
DEFAULT_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Large-hf"


# ----------------------------------------------------------------------------------------- image
def load_square(path_or_img, size=512):
    """Open, apply the phone's EXIF rotation, resize so the SHORTER side is `size`, center-crop to size x size."""
    img = path_or_img if isinstance(path_or_img, Image.Image) else Image.open(path_or_img)
    img = ImageOps.exif_transpose(img).convert("RGB")       # phone photos are often stored sideways
    scale = size / min(img.size)
    img = img.resize((max(size, round(img.width * scale)), max(size, round(img.height * scale))), Image.LANCZOS)
    left, top = (img.width - size) // 2, (img.height - size) // 2
    return img.crop((left, top, left + size, top + size))


def to_float(img):
    return np.asarray(img, dtype=np.float32) / 255.0


def to_uint8(arr):
    return (np.clip(arr, 0, 1) * 255).round().astype(np.uint8)


def gaussian_blur(img, sigma):
    """Gaussian blur with standard deviation `sigma` pixels in x and y (not across the color channels)."""
    if sigma <= 0:
        return img.copy()
    return ndimage.gaussian_filter(img, sigma=(sigma, sigma, 0), mode="reflect", truncate=4.0)


def _device():
    try:
        import torch
        return 0 if torch.cuda.is_available() else -1
    except ImportError:
        return -1


# ---------------------------------------------------------------------------------- segmentation
def load_segmenter(model=DEFAULT_SEG_MODEL):
    from transformers import pipeline
    return pipeline("image-segmentation", model=model, device=_device())


def person_mask(segmenter, img, keep_largest=True):
    """Run the segmentation model and return a bool mask of the person (True = person).

    The pipeline returns one {label, mask} entry per segment. We keep segments whose label contains
    "person"; for human-parsing models (labels like hair, face, upper-clothes) we keep everything that is
    not "background". With keep_largest, only the biggest connected region is kept (the selfie subject,
    not people far in the background)."""
    results = segmenter(img)
    labels = [r["label"] for r in results]
    keep = [r for r in results if "person" in r["label"].lower()]
    if not keep and any(r["label"].lower() == "background" for r in results):
        keep = [r for r in results if r["label"].lower() != "background"]
    if not keep:
        raise RuntimeError(f"no person found; the model returned labels {labels}")
    mask = np.zeros((img.height, img.width), bool)
    for r in keep:
        mask |= np.asarray(r["mask"]) > 127
    mask = ndimage.binary_fill_holes(ndimage.binary_closing(mask, iterations=2))
    if keep_largest:
        lab, n = ndimage.label(mask)
        if n > 1:
            sizes = ndimage.sum(mask, lab, range(1, n + 1))
            mask = lab == (1 + int(np.argmax(sizes)))
    return mask, labels


def background_blur(img, mask, sigma=15.0, feather=1.5, mask_aware=True):
    """Blur the background (mask == False) with a Gaussian of std `sigma`; keep the person sharp.

    mask_aware: blur only background pixels and renormalize, i.e. blur(img * bg) / blur(bg). Without it the
    person's colors leak into the blurred background and leave a bright/dark halo around the silhouette.
    feather: soften the mask edge by a small Gaussian so the cut-out does not look jagged."""
    bg = (~mask).astype(np.float32)
    if mask_aware:
        num = gaussian_blur(img * bg[..., None], sigma)
        den = gaussian_blur(np.repeat(bg[..., None], 3, axis=2), sigma)
        blurred = np.where(den > 1e-3, num / np.maximum(den, 1e-3), gaussian_blur(img, sigma))
    else:
        blurred = gaussian_blur(img, sigma)
    alpha = mask.astype(np.float32)
    if feather > 0:
        alpha = ndimage.gaussian_filter(alpha, feather)
    alpha = alpha[..., None]
    return alpha * img + (1 - alpha) * blurred


# ----------------------------------------------------------------------------------------- depth
def load_depth_model(model=DEFAULT_DEPTH_MODEL):
    from transformers import pipeline
    return pipeline("depth-estimation", model=model, device=_device())


def estimate_depth(depth_pipe, img, output="auto"):
    """Return a map where LARGER = FARTHER from the camera, same size as img.

    Careful: most monocular models (Depth Anything, DPT/MiDaS) predict relative INVERSE depth ("disparity"):
    large values are CLOSE. Those are flipped here. Metric-depth models (names with "metric", ZoeDepth)
    already predict distance in meters."""
    import torch
    import torch.nn.functional as F
    pred = depth_pipe(img)["predicted_depth"]
    d = torch.as_tensor(pred).float().squeeze()
    if d.shape != (img.height, img.width):          # some versions return the model's internal resolution
        d = F.interpolate(d[None, None], size=(img.height, img.width), mode="bicubic", align_corners=False)[0, 0]
    d = d.cpu().numpy()
    if output == "auto":
        name = depth_pipe.model.name_or_path.lower()
        output = "depth" if ("metric" in name or "zoedepth" in name) else "disparity"
    return (d.max() - d) if output == "disparity" else d


def normalize(x, lo=0.0, hi=15.0):
    """Min-max normalize to [lo, hi]."""
    return lo + (hi - lo) * (x - x.min()) / max(x.max() - x.min(), 1e-8)


def depth_lens_blur(img, sigma_map, max_sigma=15):
    """Spatially varying Gaussian blur: pixel (y, x) gets blur std sigma_map[y, x] (0 = sharp).

    A per-pixel kernel would be slow, so we blur the whole image at sigma = 0, 1, ..., max_sigma once and,
    for every pixel, interpolate linearly between the two levels around its sigma."""
    levels = np.arange(0, int(np.ceil(max_sigma)) + 1)
    stack = np.stack([gaussian_blur(img, s) for s in levels])          # (L, H, W, 3)
    s = np.clip(sigma_map, 0, levels[-1])
    k0 = np.minimum(np.floor(s).astype(int), len(levels) - 2)
    w = (s - k0)[..., None]
    yy, xx = np.indices(s.shape)
    return (1 - w) * stack[k0, yy, xx] + w * stack[k0 + 1, yy, xx]
