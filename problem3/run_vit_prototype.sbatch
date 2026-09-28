#!/bin/bash
# ViT-S/16 from scratch on the PROTOTYPE (100 classes), 100 epochs -> compare with runs100/resnet36_arcgate
#SBATCH -N 1
#SBATCH --gres=gpu:a100:1
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 0-02:00:00
#SBATCH -p public
#SBATCH -q class
#SBATCH -o vit_proto.%j.out
#SBATCH -e vit_proto.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=qitongzh@asu.edu
#SBATCH --export=NONE

source ~/.bashrc
echo "Job started at: $(date)   Job ID: $SLURM_JOB_ID   Node: $(hostname)"
module load mamba/latest
source activate qtz
cd "/home/qitongzh/EE598_Assignment2/"
nvidia-smi -L

# Same data, epochs, batch, augmentation, label smoothing, seed as the ResNet runs; DeiT-style optimizer for the ViT
python -u train_imagenet.py --arch vit_s16 --opt adamw --lr 5e-4 --wd 0.05 --warmup 5 --clip 1.0 \
    --csv imagenet_splits/proto_split.csv --out runs100/vit_s16 \
    --epochs 100 --bs 128 --label-smoothing 0.1 --img-size 224 --workers 28 --seed 0 \
    2>&1 | tee runs100/vit_s16.log

python compare_runs.py runs100/resnet36_arcgate runs100/vit_s16 --out runs100/resnet_vs_vit.png
echo "Job finished at: $(date)"
