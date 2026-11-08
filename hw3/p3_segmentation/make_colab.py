"""Build HW3_P3_colab.ipynb: a self-contained Colab notebook for Problem 3 (extra credit a).

The notebook embeds blur_lib.py (written to disk with %%writefile), installs dependencies, downloads the
selfie from IMAGE_URL (put your photo on GitHub or a public Drive link) or lets you upload one, and makes
all four figures.   Run:  python make_colab.py   (needs: pip install nbformat)
"""
import nbformat as nbf

lib = open("blur_lib.py").read()
nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
    md("# EEE 598 HW3, Problem 3: segmentation, background blur, depth and lens blur\n"
       "Run all cells (Runtime > Run all). A GPU runtime is faster but not required."),
    code("!pip -q install transformers scipy pillow"),
    code("%%writefile blur_lib.py\n" + lib),
    md("## Input image\nSet `IMAGE_URL` to a public link to your selfie (e.g. a raw GitHub URL). "
       "Leave it empty to upload a file instead."),
    code('IMAGE_URL = ""  # e.g. "https://raw.githubusercontent.com/<user>/<repo>/main/selfie.jpg"\n'
         "import io, urllib.request\nfrom PIL import Image\n"
         "if IMAGE_URL:\n"
         "    raw = Image.open(io.BytesIO(urllib.request.urlopen(IMAGE_URL).read()))\n"
         "else:\n"
         "    from google.colab import files\n"
         "    up = files.upload()\n"
         "    raw = Image.open(io.BytesIO(next(iter(up.values()))))\n"
         "import blur_lib as bl\n"
         "img_pil = bl.load_square(raw, 512)\nimg = bl.to_float(img_pil)\nimg_pil"),
    code("import matplotlib.pyplot as plt, numpy as np\n"
         "def pair(a, b, ta, tb, cmap=None, cbar=None):\n"
         "    fig, ax = plt.subplots(1, 2, figsize=(11, 5.2))\n"
         "    for x, im, t, c in [(ax[0], a, ta, None), (ax[1], b, tb, cmap)]:\n"
         "        h = x.imshow(im, cmap=c); x.set_title(t); x.set_xlabel('x (pixels)'); x.set_ylabel('y (pixels)')\n"
         "    if cbar: plt.colorbar(h, ax=ax[1], fraction=0.046).set_label(cbar)\n"
         "    plt.tight_layout(); plt.show()"),
    md("## 1. Person segmentation"),
    code("segmenter = bl.load_segmenter(bl.DEFAULT_SEG_MODEL)\n"
         "mask, labels = bl.person_mask(segmenter, img_pil)\nprint(labels)\n"
         "pair(img, mask, 'Input (512x512)', 'Person mask', cmap='gray')"),
    md("## 2. Background blur (sigma = 15)"),
    code("bgb = bl.background_blur(img, mask, sigma=15)\npair(img, bgb, 'Input', 'Background blurred, sigma = 15')"),
    md("## 3. Monocular depth"),
    code("depth = bl.estimate_depth(bl.load_depth_model(bl.DEFAULT_DEPTH_MODEL), img_pil)\n"
         "pair(img, depth, 'Input', 'Depth, bright = far', cmap='magma', cbar='relative distance')"),
    md("## 4. Depth-based lens blur (sigma = depth normalized to [0, 15])"),
    code("sigma_map = bl.normalize(depth, 0, 15)\nlens = bl.depth_lens_blur(img, sigma_map, 15)\n"
         "pair(img, lens, 'Input', 'Depth-based lens blur')"),
]
nb.metadata["accelerator"] = "GPU"
nbf.write(nb, "HW3_P3_colab.ipynb")
print("wrote HW3_P3_colab.ipynb")
