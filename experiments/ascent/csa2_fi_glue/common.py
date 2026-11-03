"""Shared synthetic setup for the CSA2FlashInfer adapter glue benchmark/check.

One TP4 rank of DeepSeek-V4.1-Flash: 16 query heads, head_dim 512, 128 SWA
staging rows + 512 index_topk rows (num_sparse_topk 640), FP32 sinks.
"""

import math
import os
import sys

import torch

HEADS = 16
DIM = 512
SWA = 128
EXTRA = 512
SCALE = 1.0 / math.sqrt(DIM)


def log(*args):
    print(*args, flush=True)
    sys.stdout.flush()


def make_lengths(count, width, gen, step=0):
    """Mixed prepared_lens in [1, width], always including 1 and width."""
    choices = [1, width, 7, 129, 131, 255, 384, 511, 600, 639, 64, 8, 9, 300]
    picked = [choices[(i * 5 + step * 3) % len(choices)] for i in range(count)]
    picked = [min(max(v, 1), width) for v in picked]
    return torch.tensor(picked, dtype=torch.int32, device="cuda")


def make_layer(count, gen, extra_width=EXTRA, step=0):
    """Staging pools with stale (non-zero) data beyond each query's length."""
    swa = torch.randn(count, SWA, DIM, generator=gen, device="cuda").to(torch.bfloat16)
    extra = (
        torch.randn(count, extra_width, DIM, generator=gen, device="cuda").to(torch.bfloat16)
        if extra_width
        else None
    )
    q = (torch.randn(count, HEADS, DIM, generator=gen, device="cuda") * 2).to(torch.bfloat16)
    sink = torch.randn(HEADS, generator=gen, device="cuda", dtype=torch.float32)
    lens = make_lengths(count, SWA + extra_width, gen, step)
    return dict(q=q, swa=swa, extra=extra, sink=sink, lens=lens)


def new_adapter():
    from tensorrt_llm._torch.attention.backends.sparse.csa2.backend import CSA2FlashInfer

    return CSA2FlashInfer()


def reference_call(inst, layer, out):
    """The current adapter glue: prepare_selected's cat/valid + _forward + output.copy_."""
    q, swa, extra, lens = layer["q"], layer["swa"], layer["extra"], layer["lens"]
    count = q.shape[0]
    kv = swa if extra is None else torch.cat((swa, extra), dim=1)
    valid = torch.arange(kv.shape[1], device=q.device)[None, :] < lens[:count, None]
    result = inst(q, kv, valid, layer["sink"], SCALE, protect=True).flatten(1)
    out.copy_(result)
    return out


def has_fused(inst):
    return hasattr(inst, "_forward_pools") and getattr(inst, "fused_glue", False)


def fused_call(inst, layer, out):
    q = layer["q"]
    with torch.cuda.device(q.device):
        inst._forward_pools(
            q,
            layer["swa"],
            layer["extra"],
            layer["lens"],
            layer["sink"],
            SCALE,
            protect=True,
            out=out.view_as(q),
        )
    return out


def adapter_call(inst, layer, out):
    """What the tree under test runs (fused on the candidate, reference on base)."""
    if has_fused(inst) and os.environ.get("CSA2_GLUE_FORCE_REFERENCE") != "1":
        return fused_call(inst, layer, out)
    return reference_call(inst, layer, out)
