#!/bin/bash
# Step 3: LoRA fine-tuning of LLaVA-1.6 (Mistral-7B) on ChartQA, ALL 22,639 training samples, 1 epoch,
# on NHPU Gaudi 2 cards. If the job hits the time limit, just `sbatch train_lora.sbatch` again: it resumes
# from the last checkpoint (saved every 200 steps).
#   Rough time: about 1 s per sample per card -> 4 cards ~ 1.5-2 h, 1 card ~ 6-8 h (check your first log lines).
#SBATCH -N 1
#SBATCH -p gaudi
#SBATCH -q class_gaudi
#SBATCH --gres=gpu:hl225:4           # 4 Gaudi 2 cards; keep NHPU below equal to this
#SBATCH -c 32
#SBATCH --mem=256G
#SBATCH -t 0-08:00:00
#SBATCH -o train_lora.%j.out
#SBATCH -e train_lora.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=qitongzh@asu.edu
#SBATCH --export=NONE

source ~/.bashrc
echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
module load mamba/latest
source activate gaudi-pytorch-diffusion-1.22.0.740
source $HOME/venvs/hw3-vlm/bin/activate
cd "$SLURM_SUBMIT_DIR"                    # submit from hw3/p5_vlm_gaudi
hl-smi                                    # which Gaudi cards we got

NHPU=4
export HF_HOME=/scratch/$USER/hf_cache
export PT_HPU_LAZY_MODE=0                 # eager mode, as the example README recommends
export PYTHONUNBUFFERED=1
OUT=runs/llava_chartqa_lora

# gaudi_spawn.py starts NHPU copies of the script (torch.distributed.run), one per card (data parallel).
# --do_eval makes the script split ChartQA's train split 80/20 -> 22,639 training samples.
# Validation loss is computed every --save_steps on --max_eval_samples held-out samples.
python gaudi_spawn.py --world_size $NHPU run_lora_vlm_patched.py \
    --model_name_or_path llava-hf/llava-v1.6-mistral-7b-hf \
    --dataset_name HuggingFaceM4/ChartQA \
    --output_dir $OUT \
    --do_train --do_eval --max_eval_samples 400 \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
    --num_train_epochs 1 --learning_rate 1e-4 --lora_rank 16 \
    --logging_steps 10 --save_steps 200 \
    --gradient_checkpointing \
    --use_habana --bf16 --gaudi_config_name Habana/gpt2 \
    2>&1 | tee -a train_lora.log

python plot_losses.py --run $OUT --out results
echo "finished $(date)"
