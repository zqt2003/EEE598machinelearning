#!/bin/bash
#SBATCH -N 1                         # one node
#SBATCH -c 8                         # CPU cores
#SBATCH --mem=64G                    # RAM
#SBATCH --gres=gpu:1                 # one GPU  (use --gres=gpu:a100:1 to require an A100)
#SBATCH -t 0-02:00:00                # time limit: 2 hours (the job is killed after this)
#SBATCH -p public
#SBATCH -q public
#SBATCH -o vit_output.%j.out         # printed output  (%j = job ID)
#SBATCH -e vit_output.%j.err         # errors
#SBATCH --mail-type=ALL
#SBATCH --mail-user=qitongzh@asu.edu
#SBATCH --export=NONE

source ~/.bashrc

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID   Node: $(hostname)"
echo "=========================================="

module load mamba/latest
source activate YOUR_ENV_NAME        # <-- the environment you use in Jupyter (see `mamba env list`)

cd ~/EE598_Assignment2               # <-- the folder that contains PV_balanced/ and vit_finetune.py

nvidia-smi

# -u = print immediately (unbuffered), so the .out file updates while the job runs
python -u vit_finetune.py --data PV_balanced --out vit_results --epochs 10 --bs 128 2>&1 | tee vit_finetune.log

echo "Job finished at: $(date)"
