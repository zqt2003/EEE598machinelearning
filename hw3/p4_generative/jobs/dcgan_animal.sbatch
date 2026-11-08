#!/bin/bash
# 4.5 DCGAN on the animal dataset (64x64).
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -t 0-04:00:00
#SBATCH -p public
#SBATCH -q class
#SBATCH -o dcgan_animal.%j.out
#SBATCH -e dcgan_animal.%j.err
#SBATCH --export=NONE

source ~/.bashrc
module load mamba/latest
source activate qtz
cd "$SLURM_SUBMIT_DIR"                     # submit from hw3/p4_generative:  sbatch jobs/<name>.sbatch
nvidia-smi -L 2>/dev/null
export PYTHONUNBUFFERED=1

# plain DCGAN vs. DCGAN with the small-dataset tricks (DiffAugment, one-sided label smoothing, flips)
python train_dcgan.py --data data/animal256 --size 64 --epochs 300 --out runs/dcgan_animal64_plain
python train_dcgan.py --data data/animal256 --size 64 --epochs 300 --diffaug --label-smooth 0.9 --flip \
    --out runs/dcgan_animal64_tricks
