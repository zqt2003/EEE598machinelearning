#!/bin/bash
# Problem 2: fit a 1024x1024 image with SIREN (sine), our parawave activation, and baselines.
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1            # full-image batches of 1M pixels need a big GPU (else add --batch-pixels 262144)
#SBATCH -t 0-03:00:00
#SBATCH -p public
#SBATCH -q class
#SBATCH -o siren.%j.out
#SBATCH -e siren.%j.err
#SBATCH --export=NONE

source ~/.bashrc
module load mamba/latest
source activate qtz
cd "$SLURM_SUBMIT_DIR"                     # submit from hw3/p2_siren
nvidia-smi -L
export PYTHONUNBUFFERED=1
IMG=my_photo.jpg                          # <-- EDIT: your photo (at least 1024 px on the short side), in this folder
COMMON="--image $IMG --size 1024 --hidden 512 --hidden-layers 3 --steps 3000 --derivatives"

python plot_activation.py --out activation.png
python fit_image.py $COMMON --act sine     --out runs/sine        # SIREN
python fit_image.py $COMMON --act parawave --out runs/parawave    # our activation
python fit_image.py $COMMON --act relu --lr 1e-3 --out runs/relu  # plain ReLU MLP baseline
python compare_runs.py runs/sine runs/parawave runs/relu --labels "SIREN (sine),parawave (ours),ReLU" --out comparison

# hypothesis experiment: does smoothness matter? triangle (C0) < parawave (C1) < sine (C-infinity)
python fit_image.py $COMMON --act triangle --out runs/triangle
for w in 10 60; do python fit_image.py $COMMON --act parawave --omega0 $w --out runs/parawave_w$w; done
python compare_runs.py runs/triangle runs/parawave runs/sine --labels "triangle (C0),parawave (C1),sine (C-inf)" \
    --out comparison_smoothness
python compare_runs.py runs/parawave_w10 runs/parawave runs/parawave_w60 --labels "omega0=10,omega0=30,omega0=60" \
    --out comparison_omega
