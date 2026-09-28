#!/bin/bash
# Gaudi smoke test: the course's Lab 14 example (run.sh), adapted to YOUR account.
# Lines that differ from the instructor's copy are marked  <-- CHANGED.
#   Check before submitting:  sbatch --test-only run_gaudi_test.sh   (validates, submits nothing)
#SBATCH -N 1
#SBATCH --mem=40G
#SBATCH -c 10
#SBATCH -t 0-00:15:00                # <-- CHANGED (was 5 min): the first Gaudi run spends a while compiling graphs
#SBATCH -p gaudi
#SBATCH -q class_gaudi               # <-- CHANGED (was public): your account only has QOS "class,htc", so "public" is
                                     #     rejected. Use the QOS the instructor grants you (class_gaudi or public).
#SBATCH --gres=gpu:hl225:1           # one Gaudi 2 card (HL-225)
#SBATCH -o gaudi_lazy_output.%j.out
#SBATCH -e gaudi_lazy_output.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=qitongzh@asu.edu # <-- CHANGED: your email, not the instructor's
#SBATCH --export=NONE
source ~/.bashrc
echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID   Node: $(hostname)"
echo "=========================================="
# Load mamba module (cluster-provided)
module load mamba/latest
# Move to working directory                <-- CHANGED: your own folder (you cannot read /home/dpujara1)
cd "/home/qitongzh/EEE598_Lab14/" || exit 1
ls -l mnist_gaudi_lazy.py || { echo "copy mnist_gaudi_lazy.py into $(pwd) first"; exit 1; }
hl-smi                                     # lists the Gaudi card you got (like nvidia-smi)
# CRITICAL: Set environment variable for HPU Lazy Mode
export PT_HPU_LAZY_MODE=1
export PYTHONUNBUFFERED=1                  # print each line immediately
# Run the training script and capture full stdout+stderr
# `tee` writes a copy to persistent log file while SLURM captures stdout
mamba run -n gaudi-pytorch-diffusion-1.22.0.740 \
    python mnist_gaudi_lazy.py 2>&1 | tee mnist_gaudi_lazy.log

# Extra, for Problem 4: does this environment have everything train_imagenet.py imports?
echo "---------- package check for train_imagenet.py ----------"
mamba run -n gaudi-pytorch-diffusion-1.22.0.740 python -c "
import torch, torchvision, pandas, PIL
import habana_frameworks.torch as ht
print('torch', torch.__version__, '| torchvision', torchvision.__version__, '| pandas', pandas.__version__)
print('HPU available:', ht.hpu.is_available(), '| cards visible:', ht.hpu.device_count())
" 2>&1
echo "Job finished at: $(date)"
