#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:a100:4            # 4 full A100s: 2 for ResNet-34, 2 for ResNet-36 (also works with 2 GPUs: 1 each)
#SBATCH -c 32                        # CPU cores decode the JPEGs; they feed the GPUs
#SBATCH --mem=128G
#SBATCH -t 0-04:00:00
#SBATCH -p public
#SBATCH -q public
#SBATCH -o resnet_compare.%j.out
#SBATCH -e resnet_compare.%j.err
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

# Split the GPUs of this job between the two models
IFS=',' read -ra G <<< "$CUDA_VISIBLE_DEVICES"
N=${#G[@]}
if [ "$N" -ge 4 ]; then GPUS_A="${G[0]},${G[1]}"; GPUS_B="${G[2]},${G[3]}"; P=2
else                    GPUS_A="${G[0]}";         GPUS_B="${G[1]}";         P=1; fi
WORKERS=$(( (SLURM_CPUS_PER_TASK - 4) / (2 * P) ))
echo "GPUs: $N -> ResNet-34 on [$GPUS_A], ResNet-36 on [$GPUS_B], $P GPU(s) per model, $WORKERS loader workers per GPU"

COMMON="--csv imagenet_splits/proto_split.csv --epochs 30 --bs 128 --img-size 224 --workers $WORKERS --seed 0"

# Train both models at the same time, each on its own GPUs (& = run in the background)
CUDA_VISIBLE_DEVICES=$GPUS_A python -u -m torch.distributed.run --nproc_per_node $P --master_port 29500 \
    train_imagenet.py --arch resnet34 --out runs/resnet34 $COMMON > train_resnet34.log 2>&1 &
CUDA_VISIBLE_DEVICES=$GPUS_B python -u -m torch.distributed.run --nproc_per_node $P --master_port 29501 \
    train_imagenet.py --arch resnet36 --out runs/resnet36 $COMMON > train_resnet36.log 2>&1 &
wait                                  # until both trainings are finished

python compare_runs.py runs/resnet34 runs/resnet36 --out runs/comparison.png

echo "Job finished at: $(date)"
