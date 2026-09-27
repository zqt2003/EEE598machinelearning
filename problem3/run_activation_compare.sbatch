#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:a100:1            # one GPU: only the ArcGate model needs training (ReLU run is reused from 3.2)
#SBATCH -c 16                        # CPU cores decode the JPEGs (the real bottleneck)
#SBATCH --mem=64G
#SBATCH -t 0-04:00:00
#SBATCH -p public
#SBATCH -q class
#SBATCH -o activation_compare.%j.out
#SBATCH -e activation_compare.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=qitongzh@asu.edu
#SBATCH --export=NONE

source ~/.bashrc

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID   Node: $(hostname)"
echo "=========================================="

module load mamba/latest
source activate qtz
cd "/home/qitongzh/EE598_Assignment2/"
nvidia-smi -L

# Same settings as the ResNet-36 (ReLU) run of problem 3.2 -- only the activation function differs
python -u train_imagenet.py --arch resnet36 --act arcgate --out runs/resnet36_arcgate \
    --csv imagenet_splits/proto_split.csv --epochs 30 --bs 128 --img-size 224 --workers 14 --seed 0 \
    2>&1 | tee train_resnet36_arcgate.log

python plot_activation.py --out runs/activation.png
python compare_runs.py runs/resnet36 runs/resnet36_arcgate --out runs/activation_comparison.png

echo "Job finished at: $(date)"
