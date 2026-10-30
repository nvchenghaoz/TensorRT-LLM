"""Branch 54 / idea 84: realistic FLUX.3 RoPE cos_sin tables for the served 2048^2 shapes, built
through the model's own code (Flux3PosEmbed -> _pe_to_block_freqs -> rope_cos_sin_table).

M=16384: image stream (t=0, h, w in 128x128, l=0); also the i2i image_cond stream (l=-1).
M=16464: joint t2i = text (80 tokens, l=0..79) + image.
M=32848: joint i2i = text + image + image_cond.
"""

import glob
import json
import os

import torch


def rope_config(checkpoint=None):
    axes, theta = [32, 32, 32, 32], 10000
    if checkpoint:
        for path in sorted(glob.glob(os.path.join(checkpoint, "**", "*.json"), recursive=True))[:50]:
            try:
                with open(path) as f:
                    cfg = json.load(f)
            except Exception:
                continue
            dit = cfg.get("dit") if isinstance(cfg, dict) else None
            if isinstance(dit, dict) and "axes_dim" in dit:
                return list(dit["axes_dim"]), int(dit.get("theta", theta)), path
    return axes, theta, None


def _img_ids(h, w, t=0, l=0):
    hh, ww = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    n = h * w
    return torch.stack([torch.full((n,), t), hh.reshape(-1), ww.reshape(-1), torch.full((n,), l)], 1)


def _txt_ids(n):
    return torch.stack([torch.zeros(n, dtype=torch.long)] * 3 + [torch.arange(n)], 1)


def make_cos_sin(m, dev, checkpoint=None, cond=False):
    from tensorrt_llm._torch.visual_gen.models.flux3.primsts import rope_cos_sin_table
    from tensorrt_llm._torch.visual_gen.models.flux3.transformer_flux3 import (
        Flux3PosEmbed,
        _pe_to_block_freqs,
    )

    axes, theta, _ = rope_config(checkpoint)
    if m == 16384:
        ids = _img_ids(128, 128, l=-1 if cond else 0)
    elif m == 16464:
        ids = torch.cat([_txt_ids(80), _img_ids(128, 128)])
    elif m == 32848:
        ids = torch.cat([_txt_ids(80), _img_ids(128, 128), _img_ids(128, 128, l=-1)])
    else:
        raise ValueError(m)
    pe = Flux3PosEmbed(dim=128, theta=theta, axes_dim=axes)(ids.to(dev).unsqueeze(0))
    cos, sin = _pe_to_block_freqs(pe)
    return rope_cos_sin_table(cos, sin).float().contiguous()
