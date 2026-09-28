#!/bin/bash
# ViT-S/16 from scratch on the FULL ImageNet split (650 classes) -> compare with runs_full/resnet36_arcgate_gpu
#SBATCH -N 1
#SBATCH --gres=gpu:a100:1            # 1 GPU because of the GPU-minutes limit (MaxGRESRunMinsPerUser)
#SBATCH -c 32
#SBATCH --mem=160G
#SBATCH -t 0-08:00:00                # resubmit if cut off: training resumes from the last finished epoch
#SBATCH -p public
#SBATCH -q class
#SBATCH -o vit_full.%j.out
#SBATCH -e vit_full.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=qitongzh@asu.edu
#SBATCH --export=NONE

source ~/.bashrc
echo "Job started at: $(date)   Job ID: $SLURM_JOB_ID   Node: $(hostname)"
module load mamba/latest
source activate qtz
cd "/home/qitongzh/EE598_Assignment2/"
nvidia-smi -L
NGPU=$(nvidia-smi -L | wc -l)

# Same data split, epochs, batch, augmentation, label smoothing, seed as the full ResNet-36 + ArcGate run
python -u -m torch.distributed.run --nproc_per_node $NGPU --master_port 29610 train_imagenet.py \
    --device cuda --arch vit_s16 --opt adamw --lr 5e-4 --wd 0.05 --warmup 1 --clip 1.0 \
    --csv imagenet_splits/full_split.csv --out runs_full/vit_s16_gpu \
    --epochs 10 --bs 128 --label-smoothing 0.1 --img-size 224 --workers 28 --seed 0 \
    2>&1 | tee -a full_vit_train.log

python compare_runs.py runs_full/resnet36_arcgate_gpu runs_full/vit_s16_gpu --out runs_full/resnet_vs_vit.png
echo "Job finished at: $(date)"
