#!/bin/bash
# 4.2-4.3 Colored squares: make 1k / 5k / 20k-image datasets, train DCGAN on each, score squareness.
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -t 0-04:00:00
#SBATCH -p public
#SBATCH -q class
#SBATCH -o squares.%j.out
#SBATCH -e squares.%j.err
#SBATCH --export=NONE

source ~/.bashrc
module load mamba/latest
source activate qtz
cd "$SLURM_SUBMIT_DIR"                     # submit from hw3/p4_generative:  sbatch jobs/<name>.sbatch
nvidia-smi -L 2>/dev/null
export PYTHONUNBUFFERED=1

# Same number of generator updates for every dataset size, so only the dataset size changes.
ITERS=6000
for N in 1000 5000 20000; do
    python make_squares.py --n $N --out data/squares_$N
    EPOCHS=$(( ITERS * 128 / N ))
    python train_dcgan.py --data data/squares_$N --size 64 --epochs $EPOCHS --sample-every $(( EPOCHS / 20 + 1 )) \
        --out runs/dcgan_squares_$N
    python squares_score.py --real data/squares_$N --ckpt runs/dcgan_squares_$N/checkpoint.pt --out runs/dcgan_squares_$N
done
