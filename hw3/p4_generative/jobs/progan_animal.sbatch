#!/bin/bash
# 4.5 Progressive GAN on the animal dataset.
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1
#SBATCH -t 0-08:00:00
#SBATCH -p public
#SBATCH -q class
#SBATCH -o progan_animal.%j.out
#SBATCH -e progan_animal.%j.err
#SBATCH --export=NONE

source ~/.bashrc
module load mamba/latest
source activate qtz
cd "$SLURM_SUBMIT_DIR"                     # submit from hw3/p4_generative:  sbatch jobs/<name>.sbatch
nvidia-smi -L 2>/dev/null
export PYTHONUNBUFFERED=1

# grows 4x4 -> 128x128; resubmit the same job if it hits the time limit (it resumes)
python train_progan.py --data data/animal256 --max-res 128 --flip --out runs/progan_animal128
