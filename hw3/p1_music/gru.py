"""A GRU layer written from scratch (no nn.GRU / nn.GRUCell inside).

GRU cell, for one time step (x_t = input, h = previous hidden state, * = element-wise product):

    r_t = sigmoid(W_ir x_t + b_ir + W_hr h + b_hr)        reset gate:  how much of the old state to use
    z_t = sigmoid(W_iz x_t + b_iz + W_hz h + b_hz)        update gate: how much of the old state to keep
    n_t = tanh   (W_in x_t + b_in + r_t * (W_hn h + b_hn)) candidate state (new content)
    h_t = (1 - z_t) * n_t + z_t * h                       new hidden state

These are exactly the equations PyTorch uses, so test_gru.py can copy the weights into nn.GRU and
check that both give the same output.

Compared with an LSTM, a GRU has no separate cell state and only 2 gates instead of 3, so it has
3/4 of the parameters and is a bit faster, and it usually learns about as well on short sequences.
"""
import math

import torch
import torch.nn as nn


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # One matrix for the 3 gates at once (r, z, n stacked), for the input and for the hidden state.
        self.x2h = nn.Linear(input_size, 3 * hidden_size)    # W_i* and b_i*
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size)   # W_h* and b_h*
        bound = 1.0 / math.sqrt(hidden_size)                 # same initialization as nn.GRU
        for p in self.parameters():
            nn.init.uniform_(p, -bound, bound)

    def forward(self, x, h):
        """x: (batch, input_size), h: (batch, hidden_size) -> new h: (batch, hidden_size)"""
        gx = self.x2h(x)
        gh = self.h2h(h)
        xr, xz, xn = gx.chunk(3, dim=1)
        hr, hz, hn = gh.chunk(3, dim=1)
        r = torch.sigmoid(xr + hr)          # reset gate
        z = torch.sigmoid(xz + hz)          # update gate
        n = torch.tanh(xn + r * hn)         # candidate state
        return (1 - z) * n + z * h


class GRU(nn.Module):
    """Multi-layer GRU over a whole sequence (batch_first), built from GRUCell.

    Layer k reads the hidden states of layer k-1 at every time step. Dropout is applied between layers.
    """

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_size, self.num_layers = hidden_size, num_layers
        self.cells = nn.ModuleList(
            [GRUCell(input_size if k == 0 else hidden_size, hidden_size) for k in range(num_layers)])
        self.drop = nn.Dropout(dropout)

    def forward(self, x, h0=None):
        """x: (batch, time, input_size)
        returns  out: (batch, time, hidden_size) = last layer's state at every step,
                 h_n: (num_layers, batch, hidden_size) = every layer's final state"""
        batch, steps, _ = x.shape
        if h0 is None:
            h0 = x.new_zeros(self.num_layers, batch, self.hidden_size)
        layer_input, h_n = x, []
        for k, cell in enumerate(self.cells):
            h, outs = h0[k], []
            for t in range(steps):                       # unroll over time
                h = cell(layer_input[:, t], h)
                outs.append(h)
            layer_out = torch.stack(outs, dim=1)
            h_n.append(h)
            layer_input = self.drop(layer_out) if k < self.num_layers - 1 else layer_out
        return layer_out, torch.stack(h_n)
