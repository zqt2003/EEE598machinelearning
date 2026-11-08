#!/bin/bash
# 4.4 Build the 256x256 animal dataset (CPU only).
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=48G

#SBATCH -t 0-01:00:00
#SBATCH -p public
#SBATCH -q class
#SBATCH -o animal_data.%j.out
#SBATCH -e animal_data.%j.err
#SBATCH --export=NONE

source ~/.bashrc
module load mamba/latest
source activate qtz
cd "$SLURM_SUBMIT_DIR"                     # submit from hw3/p4_generative:  sbatch jobs/<name>.sbatch
nvidia-smi -L 2>/dev/null
export PYTHONUNBUFFERED=1

# Option A: ImageNet classes already on Sol (no download). Change --preset (cat, fox, panda, golden_retriever)
# or use --src <folder> for images you downloaded yourself (see README).
python prepare_animal.py --imagenet /data/datasets/community/deeplearning/imagenet/train --preset cat \
    --max 5000 --out data/animal256
