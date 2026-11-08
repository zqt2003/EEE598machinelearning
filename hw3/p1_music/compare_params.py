"""Measure how the generation parameters change the music (for the "parameters selected" part of the report).

For every setting we generate notes from the SAME seed melody and the SAME random seed, then compute:
    distinct pitches, pitch entropy (bits; higher = more varied), mean |interval| between consecutive notes
    (semitones; large = jumpy), fraction of notes in C major, and notes per second.

Example:  python compare_params.py --ckpt runs/gru/best.pt
Writes params_table.csv and params_comparison.png next to the checkpoint.
"""
import argparse
import itertools
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from data import midi_to_notes  # noqa: E402
from generate import allowed_pitch_mask, generate  # noqa: E402
from model import MusicGRU  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="runs/gru/best.pt")
ap.add_argument("--num-notes", type=int, default=200)
ap.add_argument("--temperatures", default="0.5,1.0,1.5,2.0")
ap.add_argument("--top-ps", default="1.0,0.9")
ap.add_argument("--scales", default="none,C:major")
args = ap.parse_args()

ck = torch.load(args.ckpt, map_location="cpu")
a = ck["args"]
model = MusicGRU(a["hidden"], a["layers"], dropout=0.0, rnn=a["rnn"])
model.load_state_dict(ck["model"])
model.eval()
seed_file = open(os.path.join(os.path.dirname(args.ckpt), "val_files.txt")).read().split("\n")[0]
sp, ss, sd = midi_to_notes(seed_file)
seed = (sp[:32], np.clip(ss[:32], 0, 4), np.clip(sd[:32], 0, 4))
c_major = allowed_pitch_mask("C:major", (0, 127)).numpy()

rows = []
for T, p, sc in itertools.product([float(x) for x in args.temperatures.split(",")],
                                  [float(x) for x in args.top_ps.split(",")], args.scales.split(",")):
    pitch, step, dur = generate(model, seed, args.num_notes, T, p, sc, (21, 108), seed=0)
    counts = np.bincount(pitch, minlength=128) / len(pitch)
    nz = counts[counts > 0]
    rows.append(dict(temperature=T, top_p=p, scale=sc, distinct_pitches=len(nz),
                     pitch_entropy_bits=float(-(nz * np.log2(nz)).sum()),
                     mean_abs_interval=float(np.abs(np.diff(pitch)).mean()),
                     in_C_major=float(c_major[pitch].mean()),
                     notes_per_s=float(len(pitch) / max(step.sum() + dur[-1], 1e-6))))
    print(rows[-1])
df = pd.DataFrame(rows)
out = os.path.dirname(args.ckpt)
df.to_csv(os.path.join(out, "params_table.csv"), index=False)

fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
for (p, sc), g in df.groupby(["top_p", "scale"]):
    lab = f"top-p={p}, scale={sc}"
    axes[0].plot(g.temperature, g.pitch_entropy_bits, "-o", label=lab)
    axes[1].plot(g.temperature, g.mean_abs_interval, "-o", label=lab)
    axes[2].plot(g.temperature, g.in_C_major, "-o", label=lab)
for ax, yl, t in zip(axes, ["pitch entropy (bits)", "mean |interval| (semitones)", "fraction of notes in C major"],
                     ["Variety of pitches", "Melodic jumps", "Notes inside the C-major scale"]):
    ax.set_xlabel("temperature"); ax.set_ylabel(yl); ax.set_title(t); ax.grid(alpha=0.3)
axes[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(out, "params_comparison.png"), dpi=140)
print(df.to_string(index=False))
