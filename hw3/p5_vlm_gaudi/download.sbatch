#!/bin/bash
# Download the model (~15 GB), ChartQA (~1 GB) and the Gaudi config ONCE, into your scratch space, so the
# Gaudi jobs don't spend accelerator time downloading. CPU only.
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 0-02:00:00
#SBATCH -p public
#SBATCH -q class
#SBATCH -o download.%j.out
#SBATCH -e download.%j.err
#SBATCH --export=NONE

source ~/.bashrc
module load mamba/latest
source activate gaudi-pytorch-diffusion-1.22.0.740
source $HOME/venvs/hw3-vlm/bin/activate
export HF_HOME=/scratch/$USER/hf_cache          # big files go to scratch, not your 100 GB home
mkdir -p $HF_HOME
cd "$SLURM_SUBMIT_DIR"                    # submit from hw3/p5_vlm_gaudi

python check_downloads.py      # downloads what is missing, verifies every file's size, builds ChartQA
du -sh $HF_HOME
