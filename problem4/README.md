# Problem 4: a turbulence-robust perceptual feature

## Files

| File | What it does |
|---|---|
| `turbulence.py` | Wraps the DAATSim simulator so it can turn any image into one turbulent image (tilt + blur, uniform/no-depth mode). |
| `make_dataset.py` | Builds the paired clean/turbulent dataset. Train and test are split by source photo. |
| `p4lib.py` | Feature extractors (VGG baselines, anti-aliased VGG, proposed head), losses, metrics. |
| `train.py` | Trains the proposed feature head (and the ablations). |
| `evaluate.py` | Compares all features on the held-out test set. Writes `metrics.csv` and 3 figures. |

## How to run (Sol, GPU Jupyter session)

Run these in a terminal, or in notebook cells with a leading `!`.

```bash
# 0. Code and simulator
git clone https://github.com/zqt2003/EEE598machinelearning.git
cd EEE598machinelearning/problem4
git clone https://github.com/Riponcs/DAATSim.git
pip install --user opencv-python scipy matplotlib pillow   # torch/torchvision must already be installed

# 1. Source photos: DIV2K validation set (100 diverse high-res photos, ~450 MB)
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip && unzip -q DIV2K_valid_HR.zip

# 2. Dataset: 100 photos x 8 crops = 800 clean 256x256 images
#    train: 3 turbulent versions each (random r0 in [0.02, 0.1])
#    test:  one version at each r0 in {0.1, 0.05, 0.03, 0.02, 0.01}
python make_dataset.py --src DIV2K_valid_HR --daatsim DAATSim --out data --crops-per-image 8

# 3. Proposed method + two ablations
python train.py --data data --out ckpt/proposed.pt
python train.py --data data --out ckpt/no_antialias.pt --no-antialias
python train.py --data data --out ckpt/inv_only.pt --loss invariance

# 4. Evaluation against the standard features
python evaluate.py --data data --out results \
    --ckpt "Proposed=ckpt/proposed.pt" \
    --ckpt "Proposed w/o anti-alias=ckpt/no_antialias.pt" \
    --ckpt "Invariance-only (ablation)=ckpt/inv_only.pt"
```

In Jupyter, show the figures with `from IPython.display import Image; Image("results/fig_quantitative.png")`.

## The idea

**What turbulence does:** DAATSim models it as two effects.
1. **Tilt:** a smooth random warp. Every region shifts by a different, small, sub-pixel-to-few-pixel amount.
2. **Blur:** spatially varying blur from Zernike-based PSFs.

The scene content does not change.

**Why standard features fail (Problems 1 and 3):** VGG features are not invariant to small shifts. A 1-pixel shift changed relu4_3 by 34%, and an aligned 8-pixel shift changed it by only 0.9%. The cause is aliasing in max-pooling and strided downsampling. Turbulence tilt consists entirely of small, non-stride-aligned local shifts, so VGG should treat it as a big change. Problem 3 also showed VGG features react strongly to high-frequency perturbations.

**Proposed feature (three parts):**
1. **Anti-aliased trunk:** a frozen ImageNet VGG-16 with every `MaxPool2d(2)` replaced by *max → blur → subsample* (BlurPool, Zhang 2019). This targets the aliasing measured in Problem 1.
2. **Multi-scale head:** relu2_2 (anti-aliased downsample) and relu3_3 are concatenated. Two 3×3 conv layers produce a 128-dim, unit-normalized feature map at 1/4 resolution. The 3×3 receptive field lets the head learn to tolerate the local displacements that tilt causes.
3. **Dense contrastive training (InfoNCE):**
   - At a sampled location p, the feature of the clean image must match the feature of the turbulent image at the *same location p*. The same holds between two different turbulence realizations.
   - It must *not* match any other sampled location, in the same image or in other images in the batch.
   - The positive pairs teach invariance to turbulence. The negatives force every location to keep identifying its own content, so the feature cannot collapse and still responds to real changes.

**Why this should work:** turbulence is a *nuisance* transformation. The simulator gives unlimited (clean, turbulent) pairs, so we can teach the feature exactly which changes to ignore. The negatives keep it a useful, spatially resolved perceptual feature, not a trivial constant.

## How robustness is measured

A feature that outputs a constant is perfectly "invariant" and useless, so measuring invariance alone isn't enough. For each held-out test image x:

- `d(x, T(x))` is the distance to the same image under turbulence. It **should be small**.
- `d(x, edit(x))` is the distance to x with a 64×64 patch replaced by another scene's content, a real, visible change. It **should be large**.
- `d(x, y)` is the distance to a completely different image. It sets the scale.

Reported:

- **AUC_edit** = P(d(x,T(x)) < d(x,edit(x))). The main metric: 1.0 means turbulence is always judged a smaller change than a real content change, and 0.5 means chance.
- **ratio** = median d(x,T(x)) / median d(x,y). Lower means more invariant. This makes different features comparable.
- Both are reported **per turbulence strength**, including r0 = 0.01 (D/r0 = 20), which is *stronger than anything in training*. That tests generalization.

Baselines are the standard features from earlier parts: pixel MSE, blurred pixels, VGG relu1_2 / 2_2 / 3_3 / 4_3, an LPIPS-style 4-layer VGG distance, and anti-aliased VGG without training.

Ablations:
- **No anti-alias:** does BlurPool matter?
- **Invariance-only loss (no negatives):** shows why the contrastive negatives are needed. This version is expected to collapse.

## Figures produced

- `fig_dataset.png`: dataset examples, clean vs. 5 turbulence strengths.
- `fig_quantitative.png`: AUC_edit and ratio vs. turbulence strength for every method. The gray band is the training range.
- `fig_qualitative.png`: *where* each feature changes, for two turbulence strengths and one content edit. Each column uses one shared color scale, and values are divided by that feature's distance to a different image. Ideal: dark for turbulence, bright only on the edited patch.

## Things to discuss honestly

- **Turbulence blur vs. real blur:** a turbulence-invariant feature is *less sensitive to blur and small misalignments in general*. That is a real trade-off for restoration tasks, where you *want* to penalize blur.
- **Blurred pixels are a strong hand-made baseline:** low-pass filtering removes much of the tilt and blur difference. Say whether the learned feature beats it, and on which strengths.
- **Sim-to-real gap:** training and testing both use DAATSim in uniform mode, so real turbulence (depth-dependent, correlated in time) may differ.

## References

- R. K. Saha, Y. Zhang, J. Ye, S. Jayasuriya. *DAATSim: Depth-Aware Atmospheric Turbulence Simulation for Fast Image Rendering.* Computer Graphics Forum (Pacific Graphics), 2025. Code: https://github.com/Riponcs/DAATSim
- K. Simonyan, A. Zisserman. *Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG).* ICLR 2015.
- J. Johnson, A. Alahi, L. Fei-Fei. *Perceptual Losses for Real-Time Style Transfer and Super-Resolution.* ECCV 2016.
- R. Zhang, P. Isola, A. Efros, E. Shechtman, O. Wang. *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (LPIPS).* CVPR 2018.
- R. Zhang. *Making Convolutional Networks Shift-Invariant Again (BlurPool).* ICML 2019.
- A. van den Oord, Y. Li, O. Vinyals. *Representation Learning with Contrastive Predictive Coding (InfoNCE).* 2018.
- X. Wang, R. Zhang, C. Shen, T. Kong, L. Li. *Dense Contrastive Learning for Self-Supervised Visual Pre-Training.* CVPR 2021.
- E. Agustsson, R. Timofte. *NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study (DIV2K).* CVPRW 2017.
