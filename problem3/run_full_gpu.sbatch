#!/bin/bash
# Problem 4 on Sol NVIDIA GPUs: ResNet-36 + ArcGate trained from scratch on the FULL ImageNet split (650 classes)
#SBATCH -N 1
#SBATCH --gres=gpu:a100:2            # 2 GPUs, one DDP process each
#SBATCH -c 32                        # CPU cores decode the JPEGs: 14 loader workers per GPU
#SBATCH --mem=160G
#SBATCH -t 0-12:00:00                # if the class QOS allows less, lower this: the run resumes when resubmitted
#SBATCH -p public
#SBATCH -q class
#SBATCH -o full_gpu.%j.out
#SBATCH -e full_gpu.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=qitongzh@asu.edu
#SBATCH --export=NONE

source ~/.bashrc
echo "=========================================="
echo "Job started at: $(date)   Job ID: $SLURM_JOB_ID   Node: $(hostname)"
echo "=========================================="

module load mamba/latest
source activate qtz
cd "/home/qitongzh/EE598_Assignment2/"
nvidia-smi -L
NGPU=$(nvidia-smi -L | wc -l)

python -u -m torch.distributed.run --nproc_per_node $NGPU --master_port 29600 train_imagenet.py \
    --device cuda --arch resnet36 --act arcgate \
    --csv imagenet_splits/full_split.csv --out runs_full/resnet36_arcgate_gpu \
    --epochs 10 --bs 128 --lr 0.1 --wd 1e-4 --warmup 1 --label-smoothing 0.1 \
    --img-size 224 --workers 14 --seed 0 \
    2>&1 | tee full_gpu_train.log

echo "Job finished at: $(date)"
