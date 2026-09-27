#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4                         # CPU only: this job just lists files and reads a few thousand images
#SBATCH --mem=16G
#SBATCH -t 0-01:00:00
#SBATCH -p public
#SBATCH -q class
#SBATCH -o proto_output.%j.out
#SBATCH -e proto_output.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=qitongzh@asu.edu
#SBATCH --export=NONE

source ~/.bashrc
echo "Job started at: $(date)   Job ID: $SLURM_JOB_ID"

module load mamba/latest
source activate qtz                  # or: PY=/packages/apps/jupyter/2026-05-21/bin/python3.13

cd ~/EE598_Assignment2               # folder with make_prototype.py and the Canvas mapping file

python -u make_prototype.py \
    --root /data/datasets/community/deeplearning/imagenet/train \
    --mapping LOC_synset_mapping.txt \
    --out imagenet_splits \
    --n-classes 100 --per-class 300 --seed 0

echo "Job finished at: $(date)"
