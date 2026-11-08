"""Note-prediction network: pitch embedding + (step, duration) -> GRU -> 3 output heads."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from gru import GRU


class MusicGRU(nn.Module):
    def __init__(self, hidden=256, layers=2, emb=32, dropout=0.2, rnn="ours"):
        super().__init__()
        self.pitch_emb = nn.Embedding(128, emb)          # a learned vector per MIDI pitch (better than pitch/128)
        n_in = emb + 2                                   # + step + duration
        if rnn == "ours":
            self.rnn = GRU(n_in, hidden, num_layers=layers, dropout=dropout)      # hand-written GRU (gru.py)
        else:
            self.rnn = nn.GRU(n_in, hidden, num_layers=layers, dropout=dropout, batch_first=True)  # cuDNN, faster
        self.pitch_head = nn.Linear(hidden, 128)         # logits over the 128 pitches
        self.step_head = nn.Linear(hidden, 1)
        self.dur_head = nn.Linear(hidden, 1)

    def forward(self, pitch, step, dur, h=None):
        """pitch: (B, T) long, step/dur: (B, T) float.
        Returns predictions for the NEXT note at every position: logits (B, T, 128), step (B, T), dur (B, T), h."""
        x = torch.cat([self.pitch_emb(pitch), step.unsqueeze(-1), dur.unsqueeze(-1)], dim=-1)
        out, h = self.rnn(x, h)
        return self.pitch_head(out), self.step_head(out).squeeze(-1), self.dur_head(out).squeeze(-1), h


def mse_with_positive_pressure(pred, target):
    """MSE plus a penalty on negative predictions (a time can't be negative), as in the TF tutorial."""
    return F.mse_loss(pred, target) + 10 * F.relu(-pred).mean()


def music_loss(out, target, w_pitch=0.05, w_step=1.0, w_dur=1.0):
    logits, step, dur, _ = out
    tp, ts, td = target
    l_pitch = F.cross_entropy(logits.reshape(-1, 128), tp.reshape(-1))
    l_step = mse_with_positive_pressure(step, ts)
    l_dur = mse_with_positive_pressure(dur, td)
    total = w_pitch * l_pitch + w_step * l_step + w_dur * l_dur
    return total, dict(pitch=l_pitch.item(), step=l_step.item(), duration=l_dur.item())
