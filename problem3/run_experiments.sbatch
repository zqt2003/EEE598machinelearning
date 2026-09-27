#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:a100:2            # 2 experiments at once, 1 GPU each -> only ONE wait in the queue
#SBATCH -c 32                        # CPU cores decode the JPEGs (the real bottleneck): 14 workers per experiment
#SBATCH --mem=128G
#SBATCH -t 0-03:00:00
#SBATCH -p public
#SBATCH -q class
#SBATCH -o experiments.%j.out
#SBATCH -e experiments.%j.err
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

IFS=',' read -ra G <<< "$CUDA_VISIBLE_DEVICES"
# Same settings as the ResNet-36 (ReLU, linear head) run of problem 3.2: each experiment changes ONE thing
COMMON="--arch resnet36 --csv imagenet_splits/proto_split.csv --epochs 30 --bs 128 --img-size 224 --workers 14 --seed 0"

# Experiment 1 (problem 3.3): ArcGate activation instead of ReLU
CUDA_VISIBLE_DEVICES=${G[0]} python -u train_imagenet.py $COMMON --act arcgate --out runs/resnet36_arcgate \
    > train_resnet36_arcgate.log 2>&1 &
# Experiment 2 (extra): KAN classifier head instead of the linear fc layer
CUDA_VISIBLE_DEVICES=${G[1]} python -u train_imagenet.py $COMMON --head kan --out runs/resnet36_kan \
    > train_resnet36_kan.log 2>&1 &
wait

python plot_activation.py --out runs/activation.png
python compare_runs.py runs/resnet36 runs/resnet36_arcgate --out runs/activation_comparison.png
python compare_runs.py runs/resnet34 runs/resnet36 runs/resnet36_arcgate runs/resnet36_kan --out runs/all_comparison.png

echo "Job finished at: $(date)"
