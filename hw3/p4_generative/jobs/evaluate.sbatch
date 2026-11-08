#!/bin/bash
# 4.6-4.7 Compare the three models (FID/KID, nearest neighbours) and make interpolation figures.
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -t 0-02:00:00
#SBATCH -p public
#SBATCH -q class
#SBATCH -o evaluate.%j.out
#SBATCH -e evaluate.%j.err
#SBATCH --export=NONE

source ~/.bashrc
module load mamba/latest
source activate qtz
cd "$SLURM_SUBMIT_DIR"                     # submit from hw3/p4_generative:  sbatch jobs/<name>.sbatch
nvidia-smi -L 2>/dev/null
export PYTHONUNBUFFERED=1

# 5,000 samples per model -> FID/KID at 64x64 -> latent interpolation for the GANs
python sample_to_folder.py --ckpt runs/dcgan_animal64_tricks/checkpoint.pt --n 5000 --out samples/dcgan
python sample_to_folder.py --ckpt runs/progan_animal128/checkpoint.pt --n 5000 --out samples/progan
python train_diffusion.py --data data/animal256 --size 64 --out runs/diffusion64 --sample-only 5000 --sample-dir samples/diffusion
python fid.py --real data/animal256 --size 64 --fake samples/dcgan samples/progan samples/diffusion \
    --labels DCGAN ProGAN Diffusion --out results/fid
python interpolate.py --ckpt runs/progan_animal128/checkpoint.pt
python interpolate.py --ckpt runs/dcgan_animal64_tricks/checkpoint.pt
