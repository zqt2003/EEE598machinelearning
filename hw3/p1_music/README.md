# Problem 1: Music generation with a GRU

## The idea in 3 sentences

A piece of piano music is a sequence of notes, and each note is 3 numbers: **pitch** (which key, 0-127),
**step** (seconds since the previous note started) and **duration** (how long it is held).
A recurrent network reads the notes one by one and, at every position, predicts the next note: pitch as a
128-way classification, step and duration as regression.
To generate music, we give it a few real notes ("seed"), sample the next note, feed it back in, and repeat.

## Files

| File | What it does |
|---|---|
| `gru.py` | **The GRU written from scratch** (the part the assignment asks you to explain). |
| `test_gru.py` | Proves our GRU gives exactly the same output and gradients as `torch.nn.GRU`. |
| `data.py` | Downloads MAESTRO v2 (1,282 piano performances, MIDI), converts MIDI to (pitch, step, duration). |
| `model.py` | Pitch embedding + step + duration -> 2-layer GRU -> 3 heads. Loss = 0.05 x pitch CE + step MSE + duration MSE. |
| `train.py` | Training, validation on held-out pieces, learning curves. |
| `generate.py` | Generation with the note-generation parameters, MIDI + WAV output, picks the best 10-second clip. |
| `compare_params.py` | Measures how each parameter changes the music (table + plot for the report). |
| `run_music.sbatch` | Everything above as one Sol job. |

## How the GRU works (for the report)

For each time step, with input `x_t` and previous state `h`:

```
r = sigmoid(W_ir x + b_ir + W_hr h + b_hr)        reset gate: how much of the past to use for the new content
z = sigmoid(W_iz x + b_iz + W_hz h + b_hz)        update gate: how much of the old state to keep
n = tanh(W_in x + b_in + r * (W_hn h + b_hn))     candidate new state
h_new = (1 - z) * n + z * h                        mix old and new
```

In `gru.py`, the three gates' weights are stacked into one `nn.Linear(input, 3*hidden)` for the input and
one `nn.Linear(hidden, 3*hidden)` for the state. `.chunk(3)` splits the result into the r, z, n parts.
`GRU` loops over time and stacks layers. `test_gru.py` copies our weights into `nn.GRU` and checks that the
outputs match to ~1e-7, which shows the implementation is correct. Put a short snippet of
`GRUCell.forward` plus this check in the report.

Why a GRU and not an LSTM: it has 2 gates instead of 3 and no separate cell state, so it has 25% fewer
parameters and is a bit faster, and it usually works as well on sequences this short (32 notes).

## Note-generation parameters

| Parameter | In tutorial? | What it does |
|---|---|---|
| `--temperature T` | yes | Divides the pitch logits by T. T < 1: safer, more repetitive. T > 1: more random. |
| `--top-p p` | **new** | Nucleus sampling: keep only the most likely pitches whose probabilities add up to p, then sample. This removes the long tail of unlikely "wrong" notes but keeps variety. |
| `--scale ROOT:MODE` | **new** | Forces every note into a musical scale (`C:major`, `A:minor`, `C:pentatonic`, `E:blues`, ...) by setting out-of-scale logits to minus infinity. Makes it sound more consonant. |
| `--tempo x` | **new** (sequencing) | Divides all steps and durations by x: 1.5 = 50% faster. |
| `--pitch-range LO-HI` | new | Allowed keys (default 21-108, the 88 piano keys). |

`compare_params.py` generates from the same seed with every combination and measures pitch entropy
(variety), mean melodic jump (semitones) and the fraction of notes in C major. Use its plot to justify
your chosen settings. What to expect: higher temperature means more entropy and bigger jumps; top-p
reduces jumps at high temperature; the scale constraint makes the "in key" fraction 100%. Check what
your own run shows rather than assuming this.

## Steps

```bash
cd ~/EEE598machinelearning/hw3/p1_music
sbatch run_music.sbatch          # ~30 min on a GPU: trains, generates 4 settings, compares parameters
```

Or step by step (interactive GPU session):

```bash
python test_gru.py
python train.py --num-files 300 --epochs 40 --out runs/gru
python generate.py --ckpt runs/gru/best.pt --temperature 1.0 --top-p 0.9 --scale C:major
python compare_params.py --ckpt runs/gru/best.pt
```

Each `generate.py` call makes a folder `runs/gru/generated/<settings>/` with `generated.mid`,
`generated.wav`, `clip.wav` (the most varied 10 seconds), `clip.mid` and `piano_roll.png`.

**The 10-second clip.** Listen to the `clip.wav` files (copy them to your laptop with `scp`). The WAV is
made with simple sine-wave synthesis, so it sounds like a basic synthesizer. For real piano sound,
open `clip.mid` in GarageBand/MuseScore, or an online MIDI player, and export an MP3. Upload it to
Google Drive, set "Anyone with the link can view", and put the link in the report.

**Optional comparison.** `python train.py --rnn torch --out runs/torch_gru` trains the same model with
PyTorch's built-in GRU. It should reach about the same loss but run faster, because cuDNN fuses the loop
over time. If you run it, that comparison makes a nice sentence for the report.

## For the report

- How the GRU is implemented (equations, the `GRUCell.forward` snippet, the equivalence test).
- Model and training settings: 300 MAESTRO files split 90/10 by piece, 32-note windows, hidden 256,
  2 layers, Adam 2e-3, loss weights.
- `runs/gru/learning_curves.png` (4 panels: total, pitch, step, duration; train vs. validation). Say whether
  it overfits (validation loss rising while training loss keeps falling).
- The parameters: table above, `params_comparison.png`, and which settings you chose and why.
- A piano roll of your chosen piece (`piano_roll.png`) and the Google Drive link to the 10-s clip.
- Cite the TensorFlow tutorial and MAESTRO (Hawthorne et al., ICLR 2019).
