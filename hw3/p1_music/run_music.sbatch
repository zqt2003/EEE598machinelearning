#!/bin/bash
# Problem 1: train the GRU on MAESTRO, then generate music with several parameter settings.
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -t 0-03:00:00
#SBATCH -p public
#SBATCH -q class
#SBATCH -o music.%j.out
#SBATCH -e music.%j.err
#SBATCH --export=NONE

source ~/.bashrc
module load mamba/latest
source activate qtz
cd "$SLURM_SUBMIT_DIR"                     # submit from hw3/p1_music
nvidia-smi -L
export PYTHONUNBUFFERED=1

python test_gru.py                                              # our GRU == torch.nn.GRU
python train.py --num-files 300 --epochs 40 --out runs/gru      # hand-written GRU (the one to report)

# generation: the tutorial's parameter (temperature) and ours (top-p, scale, tempo)
python generate.py --ckpt runs/gru/best.pt --temperature 1.0 --top-p 1.0                     # tutorial-style
python generate.py --ckpt runs/gru/best.pt --temperature 1.0 --top-p 0.9 --scale C:major
python generate.py --ckpt runs/gru/best.pt --temperature 1.3 --top-p 0.9 --scale A:minor_pentatonic --tempo 1.2
python generate.py --ckpt runs/gru/best.pt --temperature 0.7 --top-p 0.9 --scale none --tempo 0.8
python compare_params.py --ckpt runs/gru/best.pt
echo "listen to runs/gru/generated/*/clip.wav and pick the best one"
