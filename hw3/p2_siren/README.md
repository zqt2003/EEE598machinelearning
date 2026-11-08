# Problem 2: SIREN and your own activation

## The idea

An image can be stored as a **function** instead of a pixel grid: a small MLP takes a coordinate (x, y)
and outputs the color (R, G, B). Training it on all pixels makes it "memorize" the image.
With ReLU this fails: the MLP can only draw smooth, blurry shapes (it is biased towards low frequencies).
SIREN (Sitzmann et al., NeurIPS 2020) uses **sin** as the activation, `y = sin(omega0 (W x + b))`, with
omega0 = 30 and a special initialization, and fits fine detail easily. A bonus: the derivative of a sine
network is again a sine network, so its gradients and Laplacians are also clean.

## Your activation: the "parabolic wave" (parawave)

Defined in `activations.py`:

```
k = floor(x / pi),   r = x / pi - k          (r in [0, 1))
f(x) = (-1)^k * 4 r (1 - r)
```

It is a chain of parabolic arches with the same period (2 pi), peaks (+-1 at pi/2 + k pi) and zeros
(k pi) as sin(x), and it differs from sin(x) by at most 0.056. But it is built only from `floor` and a
polynomial (no transcendental function), and it is **less smooth**:

- f'(x) = (-1)^k 4 (1 - 2r) / pi: continuous, a triangle wave.
- f''(x) = (-1)^k (-8/pi^2): piecewise constant, it **jumps** at every zero crossing.

So parawave is C^1, while sine is C^infinity. `plot_activation.py` draws f, f' and f'' (with autograd)
for sine, parawave and the triangle wave (which is only C^0). Put that figure in the report.

You can design a different one. Add a function to `ACTIVATIONS` in `activations.py` and pass its name
with `--act`. Keep it periodic and roughly in [-1, 1] so the SIREN initialization still makes sense.

## Files

| File | What it does |
|---|---|
| `activations.py` | sine, parawave, triangle, relu |
| `siren.py` | Coordinate MLP, SIREN initialization (first layer U(-1/n, 1/n), others U(-sqrt(6/n)/omega0, +)) |
| `fit_image.py` | Fits one image; logs PSNR; saves reconstructions; `--derivatives` compares gradient/Laplacian maps |
| `plot_activation.py` | Graph of the activations and their 1st and 2nd derivatives |
| `compare_runs.py` | PSNR curves, side-by-side reconstructions with a zoom, error spectrum, summary table |
| `run_siren.sbatch` | Main comparison + the hypothesis experiments |

## Steps

1. Copy a photo of at least 1024x1024 to `p2_siren/`, e.g. your HW2 lake photo (4032x3024) or a DIV2K
   image. It is center-cropped and resized to 1024x1024.
2. Edit the `IMG=` line in `run_siren.sbatch`, then `sbatch run_siren.sbatch` (A100, ~1-2 h for all runs).

The network: 3 hidden layers of 512 units (~800k parameters) for 3.1M pixel values. A 256-wide network
(~200k parameters) can't hold a 1024x1024 image well, so we use 512. Each step uses the whole image (full batch, as in the paper's
Colab). On a smaller GPU, add `--batch-pixels 262144` to use random pixel batches instead.

## Expected outcome and the hypothesis experiment

In a quick test (96x96 image, 300 steps), SIREN and parawave reached the same PSNR (31.9 vs 32.0 dB)
and ReLU only 12.7 dB. **At 1024x1024 you have to see for yourself.** Two outcomes are possible:

**A. Parawave is as good as sine.** Then the conclusion is: what matters is the periodicity (many
oscillations that let the network represent high frequencies), not the exact sine shape. Back it up with
the triangle-wave run: if even the C^0 triangle works, smoothness is not needed for fitting **values**.
The `--derivatives` figure will still differ, because parawave's Laplacian is built from a piecewise-constant
f''. That is where the exact sine matters: when the network's derivatives are used (e.g. fitting
gradients, or solving PDEs as in the SIREN paper).

**B. Parawave is worse.** Hypothesis: the jumps in f'' (and the kinks in the triangle's f') make the loss
landscape rougher and limit how finely the network can represent high frequencies. Evidence:
1. *Smoothness ladder* (`comparison_smoothness/`): if PSNR orders as triangle (C^0) < parawave (C^1) <
   sine (C^infinity), smoothness explains the gap.
2. *Error spectrum* (`error_spectrum.png`): if parawave's error is concentrated at high spatial
   frequencies, it is missing fine detail, not getting colors wrong.
3. *omega0 sweep* (`comparison_omega/`): if a different omega0 closes the gap, the problem is the
   frequency scale that the SIREN initialization assumes, not the shape. Sine's initialization was
   derived for sine's output distribution.

Either way you get a clear, evidence-backed paragraph. That is what the assignment asks for.

## For the report

- The SIREN equations and the initialization (two sentences), and the network size vs. number of pixels.
- `comparison/reconstructions.png` (ground truth, SIREN, parawave, ReLU, each with a zoom) and
  `comparison/psnr_curves.png`, plus the final PSNR table (`summary.csv`).
- The definition of your activation (the formula above), and `activation.png` (f, f', f'').
- Your hypothesis and the experiment's evidence (smoothness ladder, error spectrum, omega0 sweep,
  `runs/*/derivatives.png`).
- Cite SIREN (Sitzmann et al., 2020). If you mention it, also cite Tancik et al. 2020 (Fourier features) on
  ReLU's spectral bias.
