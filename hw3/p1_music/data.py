"""MAESTRO piano MIDI -> note sequences (pitch, step, duration), as in the TensorFlow music tutorial.

Every note is described by 3 numbers:
    pitch     MIDI note number 0..127 (60 = middle C)               -> predicted as a class (128-way softmax)
    step      seconds since the START of the previous note           -> predicted as a real number
    duration  seconds the note is held                               -> predicted as a real number
"""
import glob
import os
import urllib.request
import zipfile

import numpy as np
import pretty_midi

MAESTRO_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip"


def download_maestro(root="data"):
    """Download + unzip MAESTRO v2 (MIDI only, ~57 MB, 1,282 piano performances). Returns the list of MIDI files."""
    folder = os.path.join(root, "maestro-v2.0.0")
    if not os.path.isdir(folder):
        os.makedirs(root, exist_ok=True)
        zpath = os.path.join(root, "maestro-v2.0.0-midi.zip")
        if not os.path.exists(zpath):
            print("downloading", MAESTRO_URL)
            urllib.request.urlretrieve(MAESTRO_URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(root)
    files = sorted(glob.glob(os.path.join(folder, "**", "*.mid*"), recursive=True))
    print(f"{len(files)} MIDI files in {folder}")
    return files


def midi_to_notes(path):
    """Read one MIDI file -> (pitch int array, step float array, duration float array), sorted by start time."""
    pm = pretty_midi.PrettyMIDI(path)
    notes = sorted((n for inst in pm.instruments if not inst.is_drum for n in inst.notes), key=lambda n: n.start)
    if not notes:
        return None
    start = np.array([n.start for n in notes])
    pitch = np.array([n.pitch for n in notes], dtype=np.int64)
    step = np.diff(start, prepend=start[0])            # first note: step 0
    duration = np.array([n.end - n.start for n in notes])
    return pitch, step.astype(np.float32), duration.astype(np.float32)


def notes_to_midi(pitch, step, duration, path=None, velocity=90, program=0):
    """Inverse of midi_to_notes: build (and optionally save) a one-instrument PrettyMIDI. program 0 = piano."""
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=program)
    t = 0.0
    for p, s, d in zip(pitch, step, duration):
        t += float(s)
        inst.notes.append(pretty_midi.Note(velocity=velocity, pitch=int(p), start=t, end=t + float(d)))
    pm.instruments.append(inst)
    if path:
        pm.write(path)
    return pm


def build_windows(files, seq_len=32, stride=16, max_seconds=4.0):
    """Cut every file into windows of seq_len+1 notes (input = first seq_len notes, target = the next note
    at every position). Windows never cross a file boundary. step/duration are clipped to max_seconds
    (a few notes are held for very long and would dominate the MSE)."""
    P, S, D = [], [], []
    n_notes = 0
    for f in files:
        out = midi_to_notes(f)
        if out is None:
            continue
        p, s, d = out
        s, d = np.clip(s, 0, max_seconds), np.clip(d, 0, max_seconds)
        n_notes += len(p)
        for i in range(0, len(p) - seq_len - 1, stride):
            P.append(p[i:i + seq_len + 1]); S.append(s[i:i + seq_len + 1]); D.append(d[i:i + seq_len + 1])
    print(f"  {len(files)} files, {n_notes:,} notes -> {len(P):,} windows of {seq_len + 1} notes")
    return np.stack(P), np.stack(S), np.stack(D)
