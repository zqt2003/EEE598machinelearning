# Problem 5: LoRA fine-tuning of LLaVA-1.6 on ChartQA, on Gaudi

## The ideas

- **LLaVA-1.6 (Mistral-7B)** is a vision-language model. A CLIP vision encoder turns the image into tokens,
  a small projector maps them into the language model's embedding space, and Mistral-7B reads
  "[image tokens] + question" and writes the answer. LLaVA-1.6 cuts high-resolution images into tiles
  ("AnyRes"), so one chart becomes ~2,000-2,900 tokens.
- **LoRA** (Hu et al., ICLR 2022) freezes the 7B weights and learns a low-rank update for some weight
  matrices: `W x + (alpha/r) B A x`, with A (r x d) and B (d x r), r = 16. Only A and B are trained. Here that is about 44M parameters, ~0.6% of the model, so it fits in one
  Gaudi card's memory, and the result is a small adapter (~0.1-0.2 GB) instead of a new 15 GB model.
- **ChartQA** (Masry et al., ACL Findings 2022): questions about bar/line/pie charts. Answers are short
  (numbers, names, yes/no), and many need reading values and doing arithmetic.
- **Gaudi** runs through `optimum-habana` (`GaudiTrainer` instead of the Hugging Face `Trainer`). We use
  **eager mode** (`PT_HPU_LAZY_MODE=0`), as the example recommends: operations run immediately, with no graph
  compilation.

## Steps

All commands are run from `~/EEE598machinelearning/hw3/p5_vlm_gaudi`.

**1. Setup (login node, once, ~10 min)**
```bash
bash setup_env.sh
```
This builds a venv on top of the course's Gaudi conda env (`gaudi-pytorch-diffusion-1.22.0.740`, the one
from Lab 14). It installs the `optimum-habana` version that matches the cluster's Gaudi software (1.19.1 for
Gaudi SW 1.22) plus PEFT 0.12. It downloads the example's `run_lora_vlm.py` and `gaudi_spawn.py`, and writes
`run_lora_vlm_patched.py` (see "The patch" below). Check the version table it prints at the end.

**2. Download model and data (CPU job, ~30 min)**
```bash
sbatch download.sbatch           # llava-v1.6-mistral-7b-hf (~15 GB), ChartQA, Habana/gpt2 -> /scratch/$USER/hf_cache
```

**3. Look at the data (assignment step 2)**
```bash
module load mamba/latest && source activate gaudi-pytorch-diffusion-1.22.0.740 && source ~/venvs/hw3-vlm/bin/activate
export HF_HOME=/scratch/$USER/hf_cache
python show_samples.py --out results       # results/chartqa_samples.png + split counts
```
ChartQA's train split has 28,299 questions. The script finds no split called "validation", so it splits
train 80/20 (seed 42): **22,639 for training and 5,660 for validation**. `show_samples.py` repeats that
split so you can report the numbers.

**4. Fine-tune (assignment step 3)**
```bash
sbatch train_lora.sbatch
tail -f train_lora.log
```
Settings: all 22,639 samples, 1 epoch, 4 Gaudi 2 cards (data parallel, global batch 4), batch 1 per card,
learning rate 1e-4, LoRA rank 16, bf16, gradient checkpointing, validation loss on 400 held-out samples
every 200 steps. Check the first lines of the log for the step speed. If 8 hours won't be enough,
just resubmit the job: it continues from the last checkpoint. To use a different number of cards, change
both `--gres=gpu:hl225:4` and `NHPU=4`. Fewer cards means a longer run, and you may need to ask the
instructor how many cards the `class_gaudi` QOS allows.

At the end it runs `plot_losses.py`: `results/loss_curves.png`, and `results/train_summary.json` with
training time (hours), samples/s, and **peak HPU memory** (optimum-habana logs `max_memory_allocated (GB)`;
a Gaudi 2 card has 96 GB).

**5. Evaluate (assignment step 4)**
```bash
sbatch eval.sbatch
```
It answers the same 500 test questions (a fixed random subset of the 2,500) with the base model and with
the fine-tuned model, then with the base model again with an extra instruction to be brief. Results are
scored with ChartQA's **relaxed accuracy**: a number is correct within 5% of the true value, anything else
must match exactly. `compare_eval.py` writes:
- `results/accuracy_table.csv` / `.tex`: overall and by subset (human-written vs. machine-generated
  questions, numeric vs. text answers, and question type: arithmetic, comparison, counting, yes/no, retrieval)
- `results/examples_fixed.png`: questions the fine-tuned model now gets right (baseline got them wrong)
- `results/examples_still_wrong.png` and `still_wrong.txt`: questions it still gets wrong

## The patch (`patch_example.py`)

Four small fixes to the example script, each explained in the file:
1. **Multi-card epoch length.** The script computes `max_steps = samples / (batch x grad_accum) x epochs`
   and ignores the number of cards. With 4 cards, each step uses 4 samples, so "1 epoch" would silently
   be 4 epochs. The patch divides by the world size.
2. **Resume after a Slurm time limit**, from the newest `checkpoint-*`.
3. **`trainer.save_state()`**, so the full loss/memory log ends up in `trainer_state.json`.
4. **Non-reentrant gradient checkpointing** (`use_reentrant=False`). With the default ("reentrant")
   checkpointing, multi-card training crashes on the first step with `Expected to mark a variable ready only
   once`, because the script turns on DDP's `find_unused_parameters` for LLaVA. The old method also gives no
   gradients to LoRA layers whose inputs don't require grad (the vision tower). The newer method has neither
   problem and saves the same memory.

## Things to know (good material for the report)

- **What LoRA actually trains here.** The script's comments say the vision tower is excluded. But PEFT
  0.12 converts the target-module list into a *set*, and the script's LLaVA filter only runs on a *list*,
  so the filter is skipped. LoRA then goes on every module named q/k/v/o/gate/up/down_proj. That means all
  attention and MLP projections of Mistral, and also q/k/v of the CLIP vision tower. I checked this on a
  small random LLaVA-NeXT model. Your log line `trainable params: ... || all params: ...` gives the real
  count; report it.
- **The training loss covers the whole sequence**, question and image-placeholder tokens included, not just
  the answer (the collate function copies `input_ids` to `labels` without masking). So the loss values
  are not "answer loss". Validation loss still tells you whether the model is improving on unseen charts.
- **The baseline is penalized for being chatty.** The base model often answers "The value in 2015 was 25%."
  while the fine-tuned one learns to answer "25". The relaxed metric counts the sentence as wrong. That is
  why the table also has a *lenient* score (finds the number or word inside the sentence) and why
  `eval.sbatch` also runs the baseline with "Answer with a single word or number." Discuss how much of the
  gain is answer **format** and how much is real chart understanding.
- **A quirk of the official metric:** 5% tolerance means "2015" counts as correct for "2016". Mention it
  if you see it.
- **What stays hard:** expect arithmetic questions (differences, sums, averages over several bars) and
  human-written questions to remain the weakest. Machine-generated questions are mostly "read one value".
  Check your own table before writing this. In the HuggingFaceM4/ChartQA data, `human_or_machine` is 0 for
  human-written; confirm this on the dataset card.

## For the report

- Setup: model, LoRA config (rank 16, alpha 16, dropout 0.05, target modules), Gaudi cards, eager mode,
  the patch.
- Two charts with Q&A (`chartqa_samples.png`), and the split sizes (28,299 -> 22,639 / 5,660).
- Training time, peak HPU memory, `loss_curves.png`, and what you observe (fast initial drop, then a plateau?
  is validation loss still falling, i.e. no overfitting after 1 epoch?).
- The accuracy table (baseline vs. fine-tuned), 2-3 fixed and 2-3 still-wrong examples, and which question
  types remain difficult.
- Cite LLaVA-NeXT (Liu et al. 2024), LoRA (Hu et al. 2022), ChartQA (Masry et al. 2022), optimum-habana.
