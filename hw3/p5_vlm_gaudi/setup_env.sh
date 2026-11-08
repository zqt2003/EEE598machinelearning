#!/bin/bash
# One-time setup for Problem 5. Run on a Sol LOGIN node (needs internet, not a Gaudi card):
#     cd ~/EEE598machinelearning/hw3/p5_vlm_gaudi && bash setup_env.sh
#
# What it does:
#  1. Makes a Python virtual environment ON TOP OF the course's Gaudi conda env (--system-site-packages), so
#     the Gaudi build of PyTorch is reused and only the missing packages are added. You can't pip-install
#     into the shared env, and a plain `pip install --user` would leak into every other env.
#  2. Installs optimum-habana in the version that matches the cluster's Gaudi software, plus PEFT (LoRA).
#  3. Downloads the official example script run_lora_vlm.py and the launcher gaudi_spawn.py, and patches
#     the script (see patch_example.py).
set -e
GAUDI_ENV=${GAUDI_ENV:-gaudi-pytorch-diffusion-1.22.0.740}     # the env from Lab 14 / HW2
VENV=${VENV:-$HOME/venvs/hw3-vlm}

module load mamba/latest
source activate "$GAUDI_ENV"
GPY=$(python -c "import sys; print(sys.executable)")
echo "Gaudi env python: $GPY"
[ -d "$VENV" ] || "$GPY" -m venv --system-site-packages "$VENV"
source "$VENV/bin/activate"

# Gaudi software version -> matching optimum-habana release
SW=$(pip show habana-torch-plugin 2>/dev/null | awk '/^Version/{print $2}')
echo "Gaudi software (habana-torch-plugin): ${SW:-not found}"
case "$SW" in
    1.22*) OH=1.19.1 ;;      # Gaudi SW 1.22 <-> optimum-habana 1.19.x (transformers 4.51)
    1.23*|1.24*) OH=1.21.2 ;;  # Gaudi SW 1.23/1.24 <-> optimum-habana 1.20/1.21 (transformers 4.55)
    *) OH=1.19.1; echo "unknown Gaudi version; trying optimum-habana $OH" ;;
esac
pip install --upgrade pip
pip install "optimum-habana==$OH" "peft==0.12.0" "datasets>=3.6" evaluate "sentencepiece!=0.1.92" protobuf \
    matplotlib pandas

# The vision-language example only exists from v1.21.0 on; it uses nothing newer than GaudiTrainer, so it
# also runs with 1.19.x.
BASE=https://raw.githubusercontent.com/huggingface/optimum-habana
curl -sSfL -o run_lora_vlm.py "$BASE/v1.21.2/examples/vision-language-modeling/run_lora_vlm.py"
curl -sSfL -o gaudi_spawn.py "$BASE/v$OH/examples/gaudi_spawn.py"
python patch_example.py run_lora_vlm.py run_lora_vlm_patched.py

python - <<'EOF'
import importlib
for m in ["torch", "habana_frameworks.torch", "optimum.habana", "transformers", "peft", "datasets"]:
    mod = importlib.import_module(m)
    print(f"{m:25s} {getattr(mod, '__version__', 'ok')}")
EOF
echo "Setup done. Next: sbatch download.sbatch"
