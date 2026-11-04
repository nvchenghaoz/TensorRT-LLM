"""Branch 89 (idea 175): CSA2FlashInfer eager plan LRU (bound 2 per key).

Drives the REAL CSA2FlashInfer (one instance per CSA2 layer, as csa2/module.py
builds it, so each instance shares ONE float workspace per device across its
plans) through c4's alternating eager EXTEND steps: 3-ctx steps (6144 rows at
width 640 in L0-19, 384 rows at width 128 in L20-39) and 1-ctx steps (2048 /
128). Each layer call is _evict_eager_plans(metadata, key) + run_shared(...),
exactly what _forward_selected does on the shared path.
"""

import json
import os
import types

import torch

CKPT = os.environ.get("CKPT", "")
TP = 4
DIM = 512
BANK_ROWS = 8192
LAYERS = int(os.environ.get("CSA2_LAYERS", "40"))
# (count_3ctx, count_1ctx, width) per layer band.
BANDS = ((6144, 2048, 640), (384, 128, 128))


def log(*a):
    print(*a, flush=True)


def heads_per_rank():
    cfg = {}
    if CKPT and os.path.exists(os.path.join(CKPT, "config.json")):
        with open(os.path.join(CKPT, "config.json")) as f:
            cfg = json.load(f)
    heads = int(cfg.get("num_attention_heads", 64))
    return max(heads // TP, 1), cfg


def layer_band(layer):
    return BANDS[0] if layer < LAYERS // 2 else BANDS[1]


def ctx_meta(num_contexts=1):
    return types.SimpleNamespace(num_contexts=num_contexts)


class Inputs:
    """Seeded synthetic inputs, regenerated per (count, width, variant)."""

    def __init__(self, heads, device="cuda", seed=1234):
        self.heads = heads
        self.device = device
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.bank = (torch.randn(BANK_ROWS, DIM, generator=g) * 0.5).to(torch.bfloat16).to(device)
        self.sink = (torch.randn(heads, generator=g) * 0.1).float().to(device)
        self.scale = DIM ** -0.5
        self._cache = {}

    def get(self, count, width, variant=0):
        k = (count, width, variant)
        if k not in self._cache:
            g = torch.Generator(device="cpu").manual_seed(hash(k) & 0xFFFFFFFF)
            q = (torch.randn(count, self.heads, DIM, generator=g) * 0.5).to(torch.bfloat16)
            idx = torch.randint(0, BANK_ROWS, (count, width), generator=g, dtype=torch.int32)
            drop = torch.rand(count, width, generator=g) < 0.15
            idx[drop] = -1
            idx[:: max(count // 7, 1)] = -1  # some queries select nothing
            self._cache[k] = (q.to(self.device), idx.to(self.device).contiguous())
        return self._cache[k]


def make_layers(n=LAYERS):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.backend import CSA2FlashInfer

    return [CSA2FlashInfer() for _ in range(n)]


def layer_call(inst, inp, count, width, out=None, variant=0, protect=False, meta=None):
    q, idx = inp.get(count, width, variant)
    key = (count, inp.heads, width, q.device, inp.scale)
    inst._evict_eager_plans(meta or ctx_meta(), key)
    with torch.cuda.device(q.device):
        return inst.run_shared(q, inp.bank, idx, inp.sink, inp.scale, protect=protect, out=out)


def step_counts(layer, three_ctx):
    c3, c1, w = layer_band(layer)
    return (c3 if three_ctx else c1), w


def pinned_bytes(layers):
    total = 0
    for inst in layers:
        for cache in (inst._plans, inst._paged_plans):
            for plan in cache.values():
                buf = getattr(plan.wrapper, "_pin_memory_int_workspace_buffer", None)
                if buf is not None:
                    total += buf.numel() * buf.element_size()
    return total


def int_ws_bytes(layers):
    """Device bytes of the plans' int workspaces (FlashInfer default: 8 MiB each)."""
    total = 0
    for inst in layers:
        for cache in (inst._plans, inst._paged_plans):
            for plan in cache.values():
                buf = getattr(plan.wrapper, "_int_workspace_buffer", None)
                if buf is not None:
                    total += buf.numel() * buf.element_size()
    return total


def default_buffer_instance():
    """A CSA2FlashInfer whose plans keep FlashInfer's default 8 MiB int buffer (base)."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.backend import CSA2FlashInfer

    inst = CSA2FlashInfer()
    inst._RIGHT_SIZE_INT_WORKSPACE = False
    return inst


def plan_counts(layers):
    return sum(len(inst._plans) + len(inst._paged_plans) for inst in layers)
