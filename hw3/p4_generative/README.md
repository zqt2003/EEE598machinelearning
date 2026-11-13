# Problem 4: Generative models (DCGAN, Progressive GAN, diffusion)

Submit every job from this folder: `cd hw3/p4_generative && sbatch jobs/<name>.sbatch`.

## Files

| File | What it does |
|---|---|
| `dcgan.py`, `train_dcgan.py` | DCGAN (the tutorial's network, generalized to 32-256 px) + training with optional tricks |
| `make_squares.py` | Colored-squares dataset (color, size, angle, position all random; anti-aliased) |
| `squares_score.py` | Objective test: what fraction of generated images are a single, one-colored square? |
| `prepare_animal.py` | Animal dataset at 256x256 (from ImageNet on Sol or from any downloaded folder) |
| `progan.py`, `train_progan.py` | Progressive GAN, 4x4 up to 128x128, WGAN-GP loss |
| `interpolate.py` | Latent-space interpolation figures (linear, slerp, 2-D grid) + GIF |
| `train_diffusion.py` | DDPM diffusion model (diffusers U-Net), forward- and reverse-process figures |
| `sample_to_folder.py`, `fid.py` | 5,000 samples per model -> FID / KID + nearest-training-image check |
| `common.py` | Image loading, sample grids, EMA, DiffAugment, loss plots |
| `jobs/*.sbatch` | One Sol job per step (below) |

Every training script saves `checkpoint.pt` regularly and **resumes automatically** if you resubmit the
same job after a time limit.

---

## 4.1 DCGAN on CelebA (the tutorial)

**How a GAN trains.** The generator G turns random noise z into an image. The discriminator D outputs the
probability that an image is real. They alternate: D learns to tell real from fake (`log D(x) + log(1 - D(G(z)))`),
and G learns to fool D (`log D(G(z))`). `train_dcgan.py` is the tutorial's loop. One change:
`BCEWithLogitsLoss` (D outputs a logit, no Sigmoid layer), which is numerically more stable.

**Data.** The tutorial uses aligned CelebA (`img_align_celeba`, 202,599 face images):
1. First check whether Sol already has it: `ls /data/datasets/community/deeplearning/ | grep -i celeb`.
2. Otherwise download it from Kaggle (`jessicali9530/celeba-dataset`) and unzip it into `data/celeba/`.
   The torchvision/Google Drive download often fails with a quota error.

Any folder structure works: the loader takes every image under `--data`.

`sbatch jobs/dcgan_celeba.sbatch` runs the tutorial settings (64x64, batch 128, Adam 2e-4, beta1 0.5,
10 epochs) and takes under an hour on one GPU. "As good as possible": more epochs (15-25) help a bit. After that, DCGAN
quality plateaus; the loss curves will show the two networks oscillating rather than converging. That's
normal for GANs.

Outputs: `samples/epoch_*.png` (same noise every epoch, so you can watch faces form), `final_samples.png`,
`real_vs_fake.png`, `losses.png`, `d_outputs.png` (D(x) and D(G(z)) over time).

## 4.2-4.3 Colored squares

`sbatch jobs/squares.sbatch` makes datasets of 1,000, 5,000 and 20,000 images and trains a DCGAN on each,
**with the same number of generator updates (6,000)**, so only the dataset size changes. Then it scores them.

**"Does it learn squares?" with a number, not just by eye.** `squares_score.py` thresholds each image,
takes the largest blob, and fits the smallest rotated rectangle around it:
- *fill* = blob area / rectangle area: 1.0 for a square, 0.785 for a disk
- *aspect* = short side / long side: 1.0 for a square
- *color std* inside the blob: about 0 for a single color

An image counts as a square if it has one blob on a black background, fill >= 0.85, aspect >= 0.85 and
color std <= 0.08. On the real images the score is ~98%, which is the upper reference.

What to expect (from a short CPU test): early on, the GAN draws round, multi-colored blobs on a dark
background (fill ~0.73, i.e. disk-like). Corners and a single flat color take longer to learn. Report the score table per
dataset size, the sample grids, and `squares_score.png` (histograms of fill/aspect, real vs. generated).
If 20,000 is still not enough, run `make_squares.py --n 50000` and train again. The assignment asks you to keep
increasing the size until it works.

## 4.4 Animal dataset (1,000-5,000 images, 256x256)

`prepare_animal.py` center-crops, resizes to 256x256, drops tiny images and near-duplicates, and saves
a montage figure for the report, saved next to the dataset folder (`data/animal256_montage.png`), not
inside it, so it never ends up in training. Pick a source:

- **ImageNet on Sol (no download, used by `jobs/animal_data.sbatch`).** `--preset cat` uses the 5 domestic
  cat classes (~6,500 images, capped at 5,000). Other presets: `fox`, `panda`, `golden_retriever`, or
  `--wnids n0xxxxxxx ...`. Sol's copy has 650 of the 1,000 classes; the script tells you if a class is missing,
  and stops if it finds no images at all. To see which classes of an animal Sol has (uses the Canvas mapping
  file from HW2):
  `python prepare_animal.py --imagenet /data/datasets/community/deeplearning/imagenet/train --mapping ~/EE598_Assignment2/LOC_synset_mapping.txt --find dog`
  then pass the listed IDs with `--wnids`.
  Caveat: these are cluttered photos (cats in boxes, on laps). GANs find that much harder than faces.
- **AFHQ (Animal Faces HQ), recommended for nicer results.** About 5,150 cat, 4,700 dog and 4,700 wildlife
  *face* photos (training split), already aligned and 512x512. Get it from Kaggle (`andrewmvd/animal-faces`) or search the
  Hugging Face Hub for "AFHQ". Then
  `python prepare_animal.py --src afhq/train/cat --max 5000 --out data/animal256`.
- **Bulk web download:** `pip install icrawler`, then in Python:
  `BingImageCrawler(storage={"root_dir": "dl/cat"}).crawl(keyword="cat face photo", max_num=1000)`. Use a
  few keywords, then run `prepare_animal.py --src dl`. Inspect the montage and delete junk.

## 4.5 DCGAN vs. Progressive GAN on the animals

- `sbatch jobs/dcgan_animal.sbatch`: two DCGANs at 64x64. A plain one, and one with the small-dataset tricks:
  - **DiffAugment** (Zhao et al. 2020): the same random color/translation/cutout is applied to real and
    fake images before D. This stops D from memorizing a 5,000-image dataset, the main cause of GAN
    collapse on small data.
  - **One-sided label smoothing** (real target 0.9), so D is less overconfident.
  - **Random flips**, which effectively doubles the data.
  - **EMA** of G's weights for sampling (always on).
  Other knobs to try: `--lr-d`/`--lr-g` (e.g. a slower D), `--spectral-norm`, `--ngf/--ndf`, `--size 128`.
- `sbatch jobs/progan_animal.sbatch`: Progressive GAN, 4x4 -> 128x128, about 2-4 h on an A100.

**What makes ProGAN different (for the compare-and-contrast part).**

| | DCGAN | Progressive GAN |
|---|---|---|
| Resolution | fixed (64x64 here) | grows 4 -> 8 -> ... -> 128; new layers fade in (alpha 0 -> 1) |
| Loss | binary cross-entropy (non-saturating) | WGAN-GP: Wasserstein loss + gradient penalty |
| Normalization | BatchNorm | pixel norm in G, none in D; equalized learning rate |
| Anti-collapse | none (we add tricks) | minibatch standard-deviation feature in D |
| Typical behavior | fast, OK at 64 px, can mode-collapse | slower, more stable, higher resolution |

**Mode collapse:** all samples look alike, or a few images repeat. Check the sample grids and the
nearest-neighbor figure from `fid.py`. If you see it, say so and show it; the assignment explicitly accepts
documenting it.

## 4.6 Latent-space interpolation

`python interpolate.py --ckpt runs/progan_animal128/checkpoint.pt` (or the better GAN) writes
`interp_linear.png`, `interp_slerp.png`, `interp_2d.png` and `interp.gif`.
**Linear vs. slerp:** random 512-dimensional Gaussian vectors all have length about sqrt(512) ~ 22.6, so they
live near a sphere. The straight-line midpoint between two of them is much shorter (~16), an atypical latent,
which often gives washed-out images in the middle of the path. Slerp walks along the sphere instead.
What to look for: smooth changes (pose, color, background morph gradually) mean the GAN learned a
continuous space. Sudden jumps or garbage in the middle mean holes in the latent space.

## 4.7 Diffusion model

`sbatch jobs/diffusion_animal.sbatch` trains a DDPM at 64x64 (U-Net from Hugging Face `diffusers`,
~25M parameters, 60,000 steps, about 3-5 h on an A100; cite the diffusers library).

**How it works.**
- **Forward (encoding):** `x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) noise`. `abar_t` goes from 1 (clean,
  t = 0) to ~0 (pure noise, t = 999). There is nothing to learn here. → `fig_forward_process.png`
  (images getting noisier, plus the noise schedule).
- **Training:** pick a random t, noise the image, and train the U-Net to predict the noise (MSE).
  → `loss.png`
- **Reverse (decoding):** start from pure noise and repeatedly subtract the predicted noise. We use DDIM
  with 100 steps instead of 1,000 (same network, 10x faster). → `fig_reverse_process.png` (x_t along
  the way) and `fig_reverse_x0_estimate.png` (the network's current guess of the final image, which shows
  coarse layout first and details last).

## Comparing the three models (the "justify your claims" part)

`sbatch jobs/evaluate.sbatch` writes 5,000 samples per model, computes **FID** and **KID** against the real
images at the same 64x64 resolution (lower is better), and makes nearest-training-image figures.
- KID is more reliable than FID with only ~5,000 images; report both.
- The nearest-neighbor figure checks for **memorization**: if a sample is nearly identical to a training
  image, the model copied rather than generated.

Expected pattern, which the literature supports but your numbers must confirm: diffusion has the best
diversity (no mode collapse) and often the best FID, but it is much slower to sample (100 network calls vs. 1);
ProGAN gives the sharpest high-resolution images; plain DCGAN is weakest on a small dataset.

## For the report

- 4.1: tutorial settings, sample grid at the end, loss and D-output curves.
- 4.2-4.3: dataset examples, samples for each dataset size, squareness table, your conclusion.
- 4.4: data source, number of images, montage.
- 4.5: DCGAN (plain vs. tricks) and ProGAN samples; loss curves; table of differences; mode collapse if any.
- 4.6: interpolation figures with your observations.
- 4.7: the forward and reverse figures, samples, loss; the FID/KID table and nearest-neighbor figures.
- Cite: Radford et al. 2016 (DCGAN), Karras et al. 2018 (ProGAN), Gulrajani et al. 2017 (WGAN-GP),
  Zhao et al. 2020 (DiffAugment), Ho et al. 2020 (DDPM), Song et al. 2021 (DDIM), diffusers,
  Heusel et al. 2017 (FID), Binkowski et al. 2018 (KID), and your dataset source.
