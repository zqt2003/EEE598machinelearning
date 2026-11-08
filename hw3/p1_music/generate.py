"""Generate music with a trained GRU, write MIDI + WAV, and cut the most interesting 10-second clip.

Note-generation parameters (the tutorial only has the first one):
    --temperature  T   divide the pitch logits by T before softmax. T<1: safer, more repetitive; T>1: more random.
    --top-p        p   nucleus sampling: only sample among the most likely pitches whose probabilities add up to p.
                       Removes the long tail of unlikely (usually "wrong-sounding") notes while keeping variety.
    --scale  ROOT:MODE force every note into a musical scale, e.g. C:major, A:minor, C:pentatonic, E:blues.
                       Implemented by setting the logits of all out-of-scale pitches to -infinity.
    --pitch-range LO-HI  allowed MIDI pitches (default 21-108 = the 88 piano keys).
    --tempo        x   sequencing parameter: all steps and durations are divided by x (x=1.5 -> 50% faster).

Example:  python generate.py --ckpt runs/gru/best.pt --temperature 1.0 --top-p 0.9 --scale C:major --tempo 1.0
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.io.wavfile  # noqa: E402
import torch  # noqa: E402

from data import midi_to_notes, notes_to_midi  # noqa: E402
from model import MusicGRU  # noqa: E402

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MODES = {"major": [0, 2, 4, 5, 7, 9, 11], "minor": [0, 2, 3, 5, 7, 8, 10], "pentatonic": [0, 2, 4, 7, 9],
         "minor_pentatonic": [0, 3, 5, 7, 10], "blues": [0, 3, 5, 6, 7, 10], "chromatic": list(range(12))}


def allowed_pitch_mask(scale="none", pitch_range=(21, 108)):
    """Boolean mask over the 128 MIDI pitches that are allowed."""
    pitches = np.arange(128)
    ok = (pitches >= pitch_range[0]) & (pitches <= pitch_range[1])
    if scale and scale != "none":
        root, mode = scale.split(":")
        degrees = {(NOTE_NAMES.index(root.upper()) + d) % 12 for d in MODES[mode]}
        ok &= np.isin(pitches % 12, list(degrees))
    return torch.from_numpy(ok)


def sample_pitch(logits, temperature=1.0, top_p=1.0, mask=None, generator=None):
    """logits: (128,) -> one sampled pitch (int)."""
    logits = logits / max(temperature, 1e-6)
    if mask is not None:
        logits = logits.masked_fill(~mask.to(logits.device), float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    if top_p < 1.0:                                      # nucleus (top-p) sampling
        sorted_p, order = probs.sort(descending=True)
        keep = sorted_p.cumsum(0) - sorted_p < top_p     # keep a pitch if the mass BEFORE it is < p
        sorted_p = sorted_p * keep
        probs = torch.zeros_like(probs).scatter(0, order, sorted_p)
        probs = probs / probs.sum()
    return int(torch.multinomial(probs, 1, generator=generator))


@torch.no_grad()
def generate(model, seed_notes, num_notes=300, temperature=1.0, top_p=1.0, scale="none",
             pitch_range=(21, 108), seed=0):
    """Autoregressive generation. The GRU state is carried from note to note (no re-reading of the window)."""
    device = next(model.parameters()).device
    g = torch.Generator().manual_seed(seed)
    mask = allowed_pitch_mask(scale, pitch_range)
    p, s, d = (torch.as_tensor(a, device=device)[None] for a in seed_notes)
    logits, step, dur, h = model(p, s, d)                 # read the seed melody once
    out = []
    for _ in range(num_notes):
        pitch = sample_pitch(logits[0, -1].cpu(), temperature, top_p, mask, g)
        st = max(0.0, float(step[0, -1]))                 # step and duration are regression outputs
        du = max(0.05, float(dur[0, -1]))
        out.append((pitch, st, du))
        x = [torch.tensor([[v]], device=device, dtype=t) for v, t in
             ((pitch, torch.long), (st, torch.float32), (du, torch.float32))]
        logits, step, dur, h = model(*x, h=h)             # feed the new note, keep the hidden state
    pitch, step, dur = map(np.array, zip(*out))
    return pitch, step, dur


def render_wav(pm, fs=22050):
    """MIDI -> audio. Uses FluidSynth (real piano sound) if installed, otherwise simple sine-wave synthesis."""
    try:
        audio = pm.fluidsynth(fs=fs)
    except Exception:
        audio = pm.synthesize(fs=fs)
    audio = audio / (np.abs(audio).max() + 1e-9) * 0.9
    return (audio * 32767).astype(np.int16)


def most_interesting_window(pitch, step, dur, length=10.0, hop=0.5):
    """Start time of the `length`-second window with the most variety: (distinct pitches) x log(1 + notes)."""
    start = np.cumsum(step)
    best, best_t = -1, 0.0
    for t in np.arange(0, max(start[-1] - length, 0) + 1e-9, hop):
        sel = (start >= t) & (start < t + length)
        score = len(np.unique(pitch[sel])) * np.log1p(sel.sum())
        if score > best:
            best, best_t = score, t
    return best_t


def piano_roll(ax, pitch, step, dur, title, color="tab:blue", t0=0.0):
    start = t0 + np.cumsum(step)
    ax.hlines(pitch, start, start + dur, color=color, lw=3)
    ax.set_xlabel("time (s)"); ax.set_ylabel("MIDI pitch"); ax.set_title(title)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/gru/best.pt")
    ap.add_argument("--seed-midi", default=None, help="MIDI file for the seed melody (default: a validation file)")
    ap.add_argument("--seed-notes", type=int, default=32, help="notes of the seed melody given to the model")
    ap.add_argument("--num-notes", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--scale", default="none", help="none, or ROOT:MODE with MODE in " + ", ".join(MODES))
    ap.add_argument("--pitch-range", default="21-108")
    ap.add_argument("--tempo", type=float, default=1.0)
    ap.add_argument("--clip-seconds", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output folder (default: next to the checkpoint)")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    a = ck["args"]
    model = MusicGRU(a["hidden"], a["layers"], dropout=0.0, rnn=a["rnn"])
    model.load_state_dict(ck["model"])
    model.eval()

    seed_file = args.seed_midi or open(os.path.join(os.path.dirname(args.ckpt), "val_files.txt")).read().split("\n")[0]
    sp, ss, sd = midi_to_notes(seed_file)
    sp, ss, sd = sp[:args.seed_notes], np.clip(ss[:args.seed_notes], 0, 4), np.clip(sd[:args.seed_notes], 0, 4)
    lo, hi = map(int, args.pitch_range.split("-"))
    pitch, step, dur = generate(model, (sp, ss, sd), args.num_notes, args.temperature, args.top_p,
                                args.scale, (lo, hi), args.seed)
    step, dur = step / args.tempo, dur / args.tempo

    tag = f"T{args.temperature}_p{args.top_p}_{args.scale.replace(':', '-')}_tempo{args.tempo}_s{args.seed}"
    out = args.out or os.path.join(os.path.dirname(args.ckpt), "generated", tag)
    os.makedirs(out, exist_ok=True)
    pm = notes_to_midi(pitch, step, dur, os.path.join(out, "generated.mid"))
    scipy.io.wavfile.write(os.path.join(out, "generated.wav"), 22050, render_wav(pm))

    # the most interesting 10 s -> clip.mid / clip.wav (upload this one to Google Drive)
    t0 = most_interesting_window(pitch, step, dur, args.clip_seconds)
    start = np.cumsum(step)
    sel = (start >= t0) & (start < t0 + args.clip_seconds)
    cs = np.diff(start[sel], prepend=t0)
    cpm = notes_to_midi(pitch[sel], cs, np.minimum(dur[sel], t0 + args.clip_seconds - start[sel]),
                        os.path.join(out, "clip.mid"))
    scipy.io.wavfile.write(os.path.join(out, "clip.wav"), 22050, render_wav(cpm))

    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    piano_roll(axes[0], sp, ss, sd, "Seed melody (from a validation piece)", "gray")
    piano_roll(axes[0], pitch, step, dur, "Seed (gray) and generated continuation (blue)", "tab:blue",
               t0=np.cumsum(ss)[-1])
    piano_roll(axes[1], pitch, step, dur, f"Generated music, T={args.temperature}, top-p={args.top_p}, "
               f"scale={args.scale}, tempo x{args.tempo}")
    axes[1].axvspan(t0, t0 + args.clip_seconds, color="orange", alpha=0.2, label=f"{args.clip_seconds:g}-s clip")
    axes[1].legend(loc="upper right")
    plt.tight_layout(); plt.savefig(os.path.join(out, "piano_roll.png"), dpi=130)
    print(f"{len(pitch)} notes, {start[-1] + dur[-1]:.1f} s, {len(np.unique(pitch))} distinct pitches; "
          f"clip starts at {t0:.1f} s -> {out}/clip.wav")


if __name__ == "__main__":
    main()
