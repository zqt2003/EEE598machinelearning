"""Make three small fixes to optimum-habana's run_lora_vlm.py and save the result as a new file.

python patch_example.py run_lora_vlm.py run_lora_vlm_patched.py

1. Multi-card step count. The script sets max_steps = N / (batch x grad_accum) x epochs, ignoring how many
   Gaudi cards share the work. With 4 cards each step uses 4 samples, so "1 epoch" would really be 4 epochs.
   We divide by the world size.
2. Resume. If a Slurm job hits its time limit, resubmitting continues from the last checkpoint-* folder.
3. Save the training log. trainer.save_state() writes trainer_state.json (all losses, memory, timing) to the
   output folder; plot_losses.py reads it.
4. Gradient checkpointing that works on several cards. The default ("reentrant") checkpointing crashes with
   multi-card DDP when find_unused_parameters is on ("Expected to mark a variable ready only once"), and it
   gives no gradients to LoRA layers whose inputs don't require grad (the vision tower). Non-reentrant
   checkpointing (use_reentrant=False) has neither problem and saves the same memory.
Every patch checks that the original line is there, so a changed upstream file fails loudly, not silently.
"""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
patches = [
    ("            steps_per_epoch = max(1, len(self.train_dataset) // eff_bs)",
     "            world = int(os.environ.get(\"WORLD_SIZE\", 1))  # PATCH 1: samples per step = bs x ga x cards\n"
     "            steps_per_epoch = max(1, len(self.train_dataset) // (eff_bs * world))"),
    ("            train_result = trainer.train()",
     "            from transformers.trainer_utils import get_last_checkpoint  # PATCH 2: resume after time limit\n"
     "            last = get_last_checkpoint(self.args.output_dir) if os.path.isdir(self.args.output_dir) else None\n"
     "            if last:\n"
     "                logger.info(f\"Resuming from {last}\")\n"
     "            train_result = trainer.train(resume_from_checkpoint=last)\n"
     "            trainer.save_state()  # PATCH 3: trainer_state.json with the full loss / memory log\n"
     "            trainer.save_metrics(\"train\", train_result.metrics)"),
    ("            gradient_checkpointing=self.args.gradient_checkpointing,  # Configurable gradient checkpointing\n",
     "            gradient_checkpointing=self.args.gradient_checkpointing,  # Configurable gradient checkpointing\n"
     "            gradient_checkpointing_kwargs={\"use_reentrant\": False},  # PATCH 4: works with multi-card DDP\n"),
]
for old, new in patches:
    if old not in s:
        sys.exit(f"patch failed, line not found (did the upstream script change?):\n{old}")
    s = s.replace(old, new)
open(dst, "w").write(s)
print(f"wrote {dst} (4 fixes in {len(patches)} places)")
