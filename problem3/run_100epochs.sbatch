#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:a100:2            # 2 runs at a time, 1 GPU each (4 runs = 2 rounds)
#SBATCH -c 32                        # CPU cores decode the JPEGs: 14 loader workers per run
#SBATCH --mem=128G
#SBATCH -t 0-04:00:00
#SBATCH -p public
#SBATCH -q class
#SBATCH -o exp100.%j.out
#SBATCH -e exp100.%j.err
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
EPOCHS=100
OUT=runs100                          # new folder: does NOT touch the 30-epoch results in runs/
COMMON="--csv imagenet_splits/proto_split.csv --epochs $EPOCHS --bs 128 --img-size 224 --workers 14 --seed 0"

# run NAME GPU ARGS... : train one model on one GPU, log to NAME.log
run() {
    local name=$1 gpu=$2; shift 2
    CUDA_VISIBLE_DEVICES=$gpu python -u train_imagenet.py $COMMON --out $OUT/$name "$@" > $OUT/$name.log 2>&1
}
mkdir -p $OUT

# Round 1: depth comparison (problem 3.2)
run resnet34 ${G[0]} --arch resnet34 &
run resnet36 ${G[1]} --arch resnet36 &
wait
echo "round 1 done: $(date)"

# Round 2: activation (problem 3.3) and KAN head (extra); both change ONE thing vs. resnet36
run resnet36_arcgate ${G[0]} --arch resnet36 --act arcgate &
run resnet36_kan     ${G[1]} --arch resnet36 --head kan &
wait
echo "round 2 done: $(date)"

python plot_activation.py --out $OUT/activation.png
python compare_runs.py $OUT/resnet34 $OUT/resnet36 --out $OUT/depth_comparison.png
python compare_runs.py $OUT/resnet36 $OUT/resnet36_arcgate --out $OUT/activation_comparison.png
python compare_runs.py $OUT/resnet36 $OUT/resnet36_kan --out $OUT/kan_comparison.png
python compare_runs.py $OUT/resnet34 $OUT/resnet36 $OUT/resnet36_arcgate $OUT/resnet36_kan --out $OUT/all_comparison.png

echo "Job finished at: $(date)"
