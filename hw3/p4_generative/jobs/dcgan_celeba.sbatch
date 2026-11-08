#!/bin/bash
# 4.1 DCGAN tutorial on CelebA (put the aligned CelebA images under data/celeba first, see README).
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -t 0-02:00:00
#SBATCH -p public
#SBATCH -q class
#SBATCH -o dcgan_celeba.%j.out
#SBATCH -e dcgan_celeba.%j.err
#SBATCH --export=NONE

source ~/.bashrc
module load mamba/latest
source activate qtz
cd "$SLURM_SUBMIT_DIR"                     # submit from hw3/p4_generative:  sbatch jobs/<name>.sbatch
nvidia-smi -L 2>/dev/null
export PYTHONUNBUFFERED=1

python train_dcgan.py --data data/celeba --size 64 --epochs 10 --bs 128 --out runs/dcgan_celeba
python interpolate.py --ckpt runs/dcgan_celeba/checkpoint.pt
