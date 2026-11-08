"""Step 4: answer ChartQA test questions with the base model or the fine-tuned (LoRA) model and score them.

Baseline:    python eval_chartqa.py --n 500 --out results/eval_baseline.csv
Fine-tuned:  python eval_chartqa.py --n 500 --adapter runs/llava_chartqa_lora/final_model --out results/eval_finetuned.csv
Both use the SAME first --n questions of the test split (seeded shuffle), so the comparison is paired.

Scoring = ChartQA's standard "relaxed accuracy": a numeric answer counts as correct if it is within 5% of
the true number; any other answer must match exactly (ignoring case). We also report a "lenient" score that
first pulls the number or the answer word out of a longer sentence, because the base model tends to answer
with full sentences ("The value in 2015 was 25%.") while the fine-tuned model learned to answer tersely.
"""
import argparse
import os
import re
import time

import pandas as pd
import torch

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def to_float(s):
    s = s.strip().rstrip(".").replace(",", "").replace("$", "")
    try:
        return float(s[:-1]) / 100 if s.endswith("%") else float(s)
    except ValueError:
        return None


def relaxed_correct(pred, target, tol=0.05):
    p, t = to_float(pred), to_float(target)
    if p is not None and t is not None:
        return abs(p - t) <= tol * abs(t) if t != 0 else p == 0
    return pred.strip().rstrip(".").lower() == target.strip().lower()


def lenient_correct(pred, target):
    if relaxed_correct(pred, target):
        return True
    if to_float(target) is not None:                      # numeric target: try every number in the answer
        return any(relaxed_correct(m, target) or relaxed_correct(m + "%", target) for m in NUM.findall(pred))
    return re.search(rf"\b{re.escape(target.strip().lower())}\b", pred.lower()) is not None


def question_type(q):
    q = q.lower()
    if re.search(r"\b(difference|sum|total|average|mean|ratio|times|median|add|subtract|product|divided)\b", q):
        return "arithmetic"
    if re.search(r"\b(highest|lowest|largest|smallest|maximum|minimum|most|least|greater|more than|less than|"
                 r"compare|bigger|exceed)\b", q):
        return "comparison"
    if q.startswith("how many"):
        return "counting"
    if re.match(r"(is|are|does|do|was|were|did) ", q):
        return "yes/no"
    return "retrieval"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llava-hf/llava-v1.6-mistral-7b-hf")
    ap.add_argument("--adapter", default=None, help="LoRA folder (output_dir/final_model); omit for the baseline")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--split", default="test")
    ap.add_argument("--prompt-suffix", default="", help='e.g. " Answer with a single word or number."')
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--device", default="auto", help="auto | hpu | cuda | cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForVision2Seq, AutoProcessor
    if args.device == "auto":
        try:
            import habana_frameworks.torch.core  # noqa: F401  (registers the "hpu" device)
            args.device = "hpu"
        except ImportError:
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "hpu":
        import habana_frameworks.torch.core  # noqa: F401  (registers the "hpu" device)
    print("device:", args.device, flush=True)

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForVision2Seq.from_pretrained(args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()   # fold LoRA into the weights
    model = model.to(args.device).eval()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    ds = load_dataset("HuggingFaceM4/ChartQA", split=args.split).shuffle(seed=0)
    ds = ds.select(range(min(args.n, len(ds))))
    rows, t0 = [], time.time()
    for i, ex in enumerate(ds):
        target = ex["label"][0] if isinstance(ex["label"], list) else str(ex["label"])
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": ex["query"] + args.prompt_suffix}]}]
        prompt = processor.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = processor(images=ex["image"].convert("RGB"), text=prompt, return_tensors="pt")
        inputs = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v).to(args.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        pred = processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        rows.append(dict(i=i, question=ex["query"], answer=target, prediction=pred,
                         correct=relaxed_correct(pred, target), lenient=lenient_correct(pred, target),
                         human_written=bool(ex.get("human_or_machine", 0) == 0),
                         numeric=to_float(target) is not None, qtype=question_type(ex["query"])))
        if (i + 1) % 25 == 0:
            df = pd.DataFrame(rows)
            print(f"{i + 1}/{len(ds)}  relaxed acc {df.correct.mean():.3f}  lenient {df.lenient.mean():.3f}  "
                  f"({(time.time() - t0) / (i + 1):.2f} s/question)", flush=True)
            df.to_csv(args.out, index=False)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"relaxed accuracy {df.correct.mean():.3f}, lenient {df.lenient.mean():.3f} on {len(df)} questions")


if __name__ == "__main__":
    main()
