#!/bin/bash
# 4.7 DDPM diffusion model on the animal dataset (64x64); resumes if resubmitted.
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1
#SBATCH -t 0-06:00:00
#SBATCH -p public
#SBATCH -q class
#SBATCH -o diffusion_animal.%j.out
#SBATCH -e diffusion_animal.%j.err
#SBATCH --export=NONE

source ~/.bashrc
module load mamba/latest
source activate qtz
cd "$SLURM_SUBMIT_DIR"                     # submit from hw3/p4_generative:  sbatch jobs/<name>.sbatch
nvidia-smi -L 2>/dev/null
export PYTHONUNBUFFERED=1

python train_diffusion.py --data data/animal256 --size 64 --steps 60000 --flip --out runs/diffusion64
