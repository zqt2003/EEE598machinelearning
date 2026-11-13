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

# Option A: ImageNet classes already on Sol (no download). Sol's copy has 650 of the 1,000 classes and no cats
# or dog breeds. First see which animals it has (interactive, takes seconds):
#     python prepare_animal.py --imagenet /data/datasets/community/deeplearning/imagenet/train --list-animals
# then put the synset IDs you want here (one class has ~1,300 images; 1,000-5,000 are needed):
WNIDS="n02510455"                         # <-- EDIT: example = giant panda (if it is on the list)
python prepare_animal.py --imagenet /data/datasets/community/deeplearning/imagenet/train --wnids $WNIDS \
    --max 5000 --out data/animal256
# Option B: your own downloaded images instead:  python prepare_animal.py --src <folder> --max 5000 --out data/animal256
