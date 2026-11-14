"""Download (if missing) and VERIFY the model, the Gaudi config and ChartQA in the Hugging Face cache.

    export HF_HOME=/scratch/$USER/hf_cache
    python check_downloads.py

For every file of each repository it compares the size on disk with the size the Hub reports, so a file that
was cut off (e.g. two processes downloading into the same cache at once) is found here, not hours later in
training. Then it builds ChartQA's local Arrow copy and prints the splits.
Run only ONE copy at a time, and nothing else that loads ChartQA while it runs.
"""
import os
import sys

from huggingface_hub import HfApi, snapshot_download

REPOS = [("llava-hf/llava-v1.6-mistral-7b-hf", "model"), ("Habana/gpt2", "model"), ("HuggingFaceM4/ChartQA", "dataset")]
api = HfApi()
all_ok = True
for repo, rtype in REPOS:
    print(f"checking {repo} ...", flush=True)
    local = snapshot_download(repo, repo_type=rtype)              # downloads only what is missing
    info = api.repo_info(repo, repo_type=rtype, files_metadata=True)
    bad = []
    for f in info.siblings:
        path = os.path.join(local, f.rfilename)
        if f.size is not None and (not os.path.exists(path) or os.path.getsize(path) != f.size):
            bad.append((f.rfilename, os.path.getsize(path) if os.path.exists(path) else 0, f.size))
    if bad:
        all_ok = False
        print(f"  BROKEN files in {local}:")
        for name, have, want in bad:
            print(f"    {name}: {have:,} bytes on disk, {want:,} expected")
    else:
        print(f"  OK: {len(info.siblings)} files, sizes match ({local})")
if not all_ok:
    sys.exit("Delete the broken repository folder(s) under $HF_HOME/hub (datasets--... or models--...) and run again.")

from datasets import load_dataset  # noqa: E402

print(load_dataset("HuggingFaceM4/ChartQA"))
print("all downloads verified")
