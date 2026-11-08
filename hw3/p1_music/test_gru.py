"""Check that our hand-written GRU computes exactly what torch.nn.GRU computes.

We copy our weights into nn.GRU (PyTorch stores the 3 gates stacked in the same r, z, n order)
and compare outputs and gradients.   Run:  python test_gru.py
"""
import torch
import torch.nn as nn

from gru import GRU

torch.manual_seed(0)
batch, steps, n_in, n_hid, layers = 4, 25, 10, 32, 2
ours = GRU(n_in, n_hid, num_layers=layers)
ref = nn.GRU(n_in, n_hid, num_layers=layers, batch_first=True)
with torch.no_grad():
    for k, cell in enumerate(ours.cells):
        getattr(ref, f"weight_ih_l{k}").copy_(cell.x2h.weight)
        getattr(ref, f"bias_ih_l{k}").copy_(cell.x2h.bias)
        getattr(ref, f"weight_hh_l{k}").copy_(cell.h2h.weight)
        getattr(ref, f"bias_hh_l{k}").copy_(cell.h2h.bias)

x = torch.randn(batch, steps, n_in, requires_grad=True)
out1, h1 = ours(x)
out2, h2 = ref(x)
print("max |output difference|       :", (out1 - out2).abs().max().item())
print("max |final state difference|  :", (h1 - h2).abs().max().item())
g1, = torch.autograd.grad(out1.sum(), x)
g2, = torch.autograd.grad(out2.sum(), x)
print("max |input-gradient difference|:", (g1 - g2).abs().max().item())
assert torch.allclose(out1, out2, atol=1e-5) and torch.allclose(g1, g2, atol=1e-5)
print("OK: our GRU matches torch.nn.GRU")
