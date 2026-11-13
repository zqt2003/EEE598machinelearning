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
# Epochs are set from the dataset size so both runs get ~12,000 generator updates (batch 128),
# whether the dataset has 1,300 or 5,000 images.
N=$(find data/animal256 -name "*.jpg" | wc -l)
EPOCHS=$(( 12000 * 128 / N ))
echo "$N images -> $EPOCHS epochs"
python train_dcgan.py --data data/animal256 --size 64 --epochs $EPOCHS --sample-every $(( EPOCHS / 20 + 1 )) \
    --out runs/dcgan_animal64_plain
python train_dcgan.py --data data/animal256 --size 64 --epochs $EPOCHS --sample-every $(( EPOCHS / 20 + 1 )) \
    --diffaug --label-smooth 0.9 --flip --out runs/dcgan_animal64_tricks
