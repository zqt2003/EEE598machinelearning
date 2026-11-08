# EEE 598 Assignment 3: code and how-to

One folder per problem. Each has its own README with the ideas behind the code, the exact commands, what
figures/tables come out, and what to write about them.

| Folder | Problem | Runs on | Rough compute |
|---|---|---|---|
| `p1_music/` | 1. Music generation with a hand-written GRU | 1 GPU | ~30 min |
| `p2_siren/` | 2. SIREN and your own activation for 1024x1024 image fitting | 1 A100 | ~1-2 h for all runs |
| `p3_segmentation/` | 3. Selfie segmentation, background blur, depth, lens blur (+ Colab, + HF Space) | GPU or CPU | minutes |
| `p4_generative/` | 4. DCGAN, colored squares, animal dataset, Progressive GAN, interpolation, diffusion | 1 GPU / A100 | ~12-16 h in total |
| `p5_vlm_gaudi/` | 5. LoRA fine-tuning of LLaVA-1.6 on ChartQA on Gaudi | Gaudi 2 cards | ~3-10 h |

**Start the long jobs first:** Problem 5 (Gaudi queue + hours of training) and Problem 4's ProGAN and
diffusion jobs. Work on Problems 1-3 while they run.

## 1. Get the code onto Sol

```bash
ssh <asurite>@login.sol.rc.asu.edu
git clone -b claude/confident-shannon-t0bddv https://github.com/zqt2003/EEE598machinelearning.git
cd EEE598machinelearning/hw3
# already cloned?   cd EEE598machinelearning && git fetch && git checkout claude/confident-shannon-t0bddv && git pull
```

## 2. Install the extra packages (once)

Problems 1-4 use your own `qtz` environment (it already has PyTorch):

```bash
module load mamba/latest
source activate qtz
pip install -r requirements.txt
```

Problem 5 needs the Gaudi software stack; it has its own setup script (`p5_vlm_gaudi/setup_env.sh`).

## 3. Run things

Every folder has `.sbatch` job files. Submit them **from inside that folder**, for example:

```bash
cd p1_music
sbatch run_music.sbatch
squeue -u $USER                 # PD = waiting, R = running
tail -f music.*.out             # watch it (Ctrl+C stops watching, not the job)
```

To try something interactively first (recommended, so a typo doesn't waste a queue slot):

```bash
srun -p public -q class --gres=gpu:1 -c 8 --mem=32G -t 0-01:00:00 --pty bash
module load mamba/latest && source activate qtz
cd ~/EEE598machinelearning/hw3/p1_music
python train.py --num-files 5 --epochs 1 --out runs/test      # tiny smoke test, then the real run
```

All scripts were tested on small inputs on a CPU before being given to you. The Hugging Face downloads
(Problems 3 and 5) and the Gaudi runs could not be tested there, so watch the first minutes of those jobs.

## 4. The report

The assignment wants a LaTeX report (you can reuse the CVPR template from HW2 in `../report/`), one
section per problem, with labeled figures, short code snippets (not whole files), and results.
Every plotting script here already gives its figures a title and labeled axes. Each README ends with a
"For the report" list of what to show and what to discuss.
Remember the acknowledgements section: cite the papers, datasets, libraries and any help you used.
