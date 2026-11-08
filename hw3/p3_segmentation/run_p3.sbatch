#!/bin/bash
# Problem 3: all figures from your selfie (a GPU is optional; on CPU it takes a few minutes).
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -t 0-00:45:00
#SBATCH -p public
#SBATCH -q class
#SBATCH -o p3.%j.out
#SBATCH -e p3.%j.err
#SBATCH --export=NONE

source ~/.bashrc
module load mamba/latest
source activate qtz
cd "$SLURM_SUBMIT_DIR"                     # submit from hw3/p3_segmentation
export PYTHONUNBUFFERED=1
IMG=selfie.jpg                            # <-- EDIT: your selfie, in this folder
python run_p3.py --image $IMG --out results
