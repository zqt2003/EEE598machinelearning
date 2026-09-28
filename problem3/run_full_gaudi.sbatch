#!/bin/bash
# Problem 4 on Intel Gaudi (HPU): same model, data split and hyperparameters as run_full_gpu.sbatch.
#
# !!! Fill in the four lines marked  <-- FROM YOUR GAUDI LAB  using the Gaudi example your course provided
#     (partition / QOS / how Gaudi cards are requested / how the Habana PyTorch environment is loaded).
#     These differ between systems, so they are NOT guessed here.
#SBATCH -N 1
#SBATCH -p GAUDI_PARTITION           # <-- FROM YOUR GAUDI LAB  (e.g. the partition name of the Gaudi nodes)
#SBATCH -q class                     # <-- FROM YOUR GAUDI LAB  (QOS)
#SBATCH --gres=GAUDI_GRES            # <-- FROM YOUR GAUDI LAB  (how Gaudi cards are requested, if via gres)
#SBATCH -c 32
#SBATCH --mem=160G
#SBATCH -t 0-12:00:00
#SBATCH -o full_gaudi.%j.out
#SBATCH -e full_gaudi.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=qitongzh@asu.edu
#SBATCH --export=NONE

source ~/.bashrc
echo "=========================================="
echo "Job started at: $(date)   Job ID: $SLURM_JOB_ID   Node: $(hostname)"
echo "=========================================="

# <-- FROM YOUR GAUDI LAB: load the environment that has the Habana PyTorch bridge (habana_frameworks), e.g.
#     a module, a conda env, or a container. Check it with:  python -c "import habana_frameworks.torch"
module load mamba/latest
source activate GAUDI_ENV

cd "/home/qitongzh/EE598_Assignment2/"     # must contain train_imagenet.py, resnet_custom.py, imagenet_splits/
hl-smi                                    # lists the Gaudi cards (like nvidia-smi)

NHPU=${NHPU:-8}                           # number of Gaudi cards to use (set to what your allocation has)
export PT_HPU_LAZY_MODE=1                 # Gaudi "lazy mode": ops are collected into graphs, run at mark_step()

python -u -m torch.distributed.run --nproc_per_node $NHPU --master_port 29601 train_imagenet.py \
    --device hpu --arch resnet36 --act arcgate \
    --csv imagenet_splits/full_split.csv --out runs_full/resnet36_arcgate_gaudi \
    --root /data/datasets/community/deeplearning/imagenet/train \
    --epochs 10 --bs 128 --lr 0.1 --wd 1e-4 --warmup 1 --label-smoothing 0.1 \
    --img-size 224 --workers 3 --seed 0 \
    2>&1 | tee full_gaudi_train.log

echo "Job finished at: $(date)"
