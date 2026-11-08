"""Load a trained generator (DCGAN or Progressive GAN) from its checkpoint.pt."""
import torch

import dcgan
import progan


def load_generator(ckpt_path, device="cpu", use_ema=True):
    """Returns (G, z_dim, kind). G(z) gives images in [-1, 1] at the model's final resolution."""
    ck = torch.load(ckpt_path, map_location=device)
    a, kind = ck["args"], ck["kind"]
    if kind == "dcgan":
        G = dcgan.Generator(a["nz"], a["ngf"], a["size"])
        z_dim = a["nz"]
    elif kind == "progan":
        G = progan.Generator(a["z_dim"], a["max_res"], fmap_max=a["fmap_max"])
        z_dim = a["z_dim"]
    else:
        raise ValueError(kind)
    G.load_state_dict(ck["G_ema"] if use_ema else ck["G"])
    return G.to(device).eval(), z_dim, kind
