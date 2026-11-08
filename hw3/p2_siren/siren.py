"""SIREN-style coordinate MLP with a choice of activation.

Every hidden layer computes  y = act(omega0 * (W x + b)).
Initialization (SIREN paper, Sec. 3.2), which keeps the pre-activations roughly N(0,1)-like through depth:
    first layer : W ~ U(-1/n_in, 1/n_in)                         (omega0 = 30 then spreads the input over
                                                                  several periods of the activation)
    other layers: W ~ U(-sqrt(6/n_in)/omega0, sqrt(6/n_in)/omega0)
The last layer is linear (no activation). For ReLU we use PyTorch's default init and no omega0.
"""
import math

import torch
import torch.nn as nn

from activations import ACTIVATIONS, PERIODIC


class Layer(nn.Module):
    def __init__(self, n_in, n_out, act, omega0, first):
        super().__init__()
        self.linear = nn.Linear(n_in, n_out)
        self.act, self.omega0 = ACTIVATIONS[act], (omega0 if act in PERIODIC else 1.0)
        if act in PERIODIC:
            bound = 1 / n_in if first else math.sqrt(6 / n_in) / omega0
            nn.init.uniform_(self.linear.weight, -bound, bound)

    def forward(self, x):
        return self.act(self.omega0 * self.linear(x))


class CoordMLP(nn.Module):
    def __init__(self, act="sine", in_features=2, hidden=512, hidden_layers=3, out_features=3,
                 first_omega0=30.0, hidden_omega0=30.0):
        super().__init__()
        layers = [Layer(in_features, hidden, act, first_omega0, first=True)]
        layers += [Layer(hidden, hidden, act, hidden_omega0, first=False) for _ in range(hidden_layers)]
        self.net = nn.Sequential(*layers)
        self.last = nn.Linear(hidden, out_features)
        if act in PERIODIC:
            bound = math.sqrt(6 / hidden) / hidden_omega0
            nn.init.uniform_(self.last.weight, -bound, bound)

    def forward(self, coords):
        return self.last(self.net(coords))


def get_grid(h, w, device="cpu"):
    """Pixel-center coordinates in [-1, 1]^2, shape (h*w, 2), row-major (y, x)."""
    ys = torch.linspace(-1, 1, h, device=device)
    xs = torch.linspace(-1, 1, w, device=device)
    return torch.stack(torch.meshgrid(ys, xs, indexing="ij"), dim=-1).reshape(-1, 2)
