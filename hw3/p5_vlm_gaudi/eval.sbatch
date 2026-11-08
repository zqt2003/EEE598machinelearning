#!/bin/bash
# Step 4: baseline vs. fine-tuned accuracy on the same 500 ChartQA test questions (1 Gaudi card).
#SBATCH -N 1
#SBATCH -p gaudi
#SBATCH -q class_gaudi
#SBATCH --gres=gpu:hl225:1
#SBATCH -c 10
#SBATCH --mem=120G
#SBATCH -t 0-04:00:00
#SBATCH -o eval.%j.out
#SBATCH -e eval.%j.err
#SBATCH --export=NONE

source ~/.bashrc
module load mamba/latest
source activate gaudi-pytorch-diffusion-1.22.0.740
source $HOME/venvs/hw3-vlm/bin/activate
cd "$SLURM_SUBMIT_DIR"                    # submit from hw3/p5_vlm_gaudi
export HF_HOME=/scratch/$USER/hf_cache
export PT_HPU_LAZY_MODE=0
export PYTHONUNBUFFERED=1
N=500
ADAPTER=runs/llava_chartqa_lora/final_model
mkdir -p results

python eval_chartqa.py --n $N --out results/eval_baseline.csv
python eval_chartqa.py --n $N --adapter $ADAPTER --out results/eval_finetuned.csv
# fairness check: the base model with an explicit "be brief" instruction
python eval_chartqa.py --n $N --prompt-suffix " Answer with a single word or number." \
    --out results/eval_baseline_short.csv
python compare_eval.py --base results/eval_baseline.csv --ft results/eval_finetuned.csv --out results
python compare_eval.py --base results/eval_baseline_short.csv --ft results/eval_finetuned.csv --out results/vs_short_prompt
