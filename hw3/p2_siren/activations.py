"""Periodic activations for coordinate MLPs (implicit neural representations).

sine       f(x) = sin(x)                       SIREN (Sitzmann et al., 2020). Smooth: every derivative is
                                               again a shifted sine (C-infinity).
parawave   OUR activation, the "parabolic wave":
               k = floor(x / pi),  r = x / pi - k  (r in [0, 1))
               f(x) = (-1)^k * 4 r (1 - r)
           A chain of parabolic arches with the same period (2 pi), peaks (+-1 at x = pi/2 + k pi) and zeros
           (x = k pi) as sin(x). It differs from sin by at most 0.056, but it is only C^1:
               f'(x)  = (-1)^k * 4 (1 - 2r) / pi          continuous
               f''(x) = (-1)^k * (-8 / pi^2)              piecewise constant, jumps at every zero crossing
           and it needs no transcendental function (only floor and a polynomial), so it is cheap to compute.
triangle   f(x) = (2/pi) * (-1)^k * (x - k pi), k = floor(x/pi + 1/2): same zeros/peaks, piecewise LINEAR,
           only C^0 (f'' = 0 almost everywhere). Used in the smoothness experiment.
relu       standard ReLU MLP (no positional encoding) - the baseline SIREN beats.

All periodic ones are written with plain torch ops, so autograd gives correct 1st and 2nd derivatives.
To try your own idea, add a function to ACTIVATIONS.
"""
import math

import torch


def parawave(x):
    u = x / math.pi
    k = torch.floor(u)
    r = u - k
    sign = 1.0 - 2.0 * torch.remainder(k, 2.0)          # (-1)^k
    return sign * 4.0 * r * (1.0 - r)


def triangle(x):
    k = torch.floor(x / math.pi + 0.5)
    sign = 1.0 - 2.0 * torch.remainder(k, 2.0)
    return sign * (2.0 / math.pi) * (x - k * math.pi)


ACTIVATIONS = {
    "sine": torch.sin,
    "parawave": parawave,
    "triangle": triangle,
    "relu": torch.relu,
}
PERIODIC = {"sine", "parawave", "triangle"}
