#!/bin/bash
# Problem 4 on Intel Gaudi 2 (HPU): same model, data split and hyperparameters as run_full_gpu.sbatch
# (2 devices x batch 128 = global batch 256, lr 0.1, 10 epochs), so the GPU and Gaudi runs are comparable.
# Partition, card type and environment come from the course's Gaudi example (Lab 14 run.sh).
# Run run_gaudi_test.sh first: if that works, this one will be accepted too.
#SBATCH -N 1
#SBATCH -p gaudi
#SBATCH -q class_gaudi               # the QOS the instructor grants you for Gaudi (class_gaudi or public)
#SBATCH --gres=gpu:hl225:2           # 2 Gaudi 2 cards; keep NHPU below equal to this
#SBATCH -c 32                        # CPU cores decode the JPEGs: 14 loader workers per card
#SBATCH --mem=160G
#SBATCH -t 0-06:00:00                # the GPU run took ~1 h; if the job is cut off, resubmit and it resumes
#SBATCH -o full_gaudi.%j.out
#SBATCH -e full_gaudi.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=qitongzh@asu.edu
#SBATCH --export=NONE

source ~/.bashrc
echo "=========================================="
echo "Job started at: $(date)   Job ID: $SLURM_JOB_ID   Node: $(hostname)"
echo "=========================================="

module load mamba/latest
cd "/home/qitongzh/EE598_Assignment2/" || exit 1   # contains train_imagenet.py, resnet_custom.py, imagenet_splits/
hl-smi                                    # lists the Gaudi cards (like nvidia-smi)

NHPU=2                                    # number of Gaudi cards = the --gres count above
export PT_HPU_LAZY_MODE=1                 # Gaudi "lazy mode": ops are collected into graphs, run at mark_step()
export PYTHONUNBUFFERED=1

mamba run -n gaudi-pytorch-diffusion-1.22.0.740 \
    python -u -m torch.distributed.run --nproc_per_node $NHPU --master_port 29601 train_imagenet.py \
    --device hpu --arch resnet36 --act arcgate \
    --csv imagenet_splits/full_split.csv --out runs_full/resnet36_arcgate_gaudi \
    --root /data/datasets/community/deeplearning/imagenet/train \
    --epochs 10 --bs 128 --lr 0.1 --wd 1e-4 --warmup 1 --label-smoothing 0.1 \
    --img-size 224 --workers 14 --seed 0 \
    2>&1 | tee full_gaudi_train.log

echo "Job finished at: $(date)"
