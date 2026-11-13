#!/usr/bin/env python3
"""Toy check of the REAL MiniMax-H3 Cache-DiT enabler (cache_dit_enablers.enable_cache_dit_for_minimax_h3).

1 GPU, no model load. A 50-block toy with H3's block signature (hidden_states, temb, adaln_indices,
rotary_emb, key_padding_mask, timestep, valid_seq_len=) -> hidden_states is wrapped by the enabler with
each arm's CacheDiTConfig (built through the real pydantic config, bins validated). Then:
  1. server warmup: refresh(2) + 2 forwards  -> forward 1 cached (blocks 8.. not called)
  2. request A (seed a): refresh(49) + 49 forwards -> cached set == mask zeros
  3. client warmup: refresh(1) + 1 forward  -> nothing cached
  4. request B (seed b), 5. request A again -> bitwise equal to 2, and to a fresh-enable run of A.
Prints the resolved live config after each refresh. Last stdout line: {"passed": ...}.
"""
from __future__ import annotations

import json
import math
import os
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h3_cache_arms_table import MASKS  # noqa: E402

from tensorrt_llm._torch.visual_gen.cache import cache_dit_enablers as ce  # noqa: E402
from tensorrt_llm.visual_gen.args import CacheDiTConfig  # noqa: E402

torch.set_grad_enabled(False)
DEV = "cuda"
NB, D, B, S, N = 50, 128, 1, 64, 49


def bins_of(mask):
    comp, cache = [], []
    i = 0
    while i < len(mask):
        j = i
        while j < len(mask) and mask[j] == mask[i]:
            j += 1
        (comp if mask[i] == 1 else cache).append(j - i)
        i = j
    return comp, cache


ARMS = {
    "w14_c2": dict(max_warmup_steps=14, max_continuous_cached_steps=2),
    "E": dict(max_warmup_steps=8, max_continuous_cached_steps=-1),
    "D": dict(max_warmup_steps=8, max_continuous_cached_steps=-1),
    "C": dict(max_warmup_steps=8, max_continuous_cached_steps=-1),
    "fallback": dict(max_warmup_steps=10, max_continuous_cached_steps=-1),
    "W": dict(max_warmup_steps=14, max_continuous_cached_steps=-1),
}


def cache_cfg(arm):
    comp, cache = bins_of(MASKS[arm])
    d = dict(cache_backend="cache_dit", Fn_compute_blocks=8, Bn_compute_blocks=0,
             residual_diff_threshold=0.08, scm_compute_bins=comp, scm_cache_bins=cache,
             scm_steps_policy="static", **ARMS[arm])
    return d


g = torch.Generator(device="cpu").manual_seed(1234)
W1 = [(torch.randn(D, D, generator=g) / math.sqrt(D)).to(DEV, torch.bfloat16) for _ in range(NB)]
W2 = [(torch.randn(D, D, generator=g) / math.sqrt(D)).to(DEV, torch.bfloat16) for _ in range(NB)]
TEMB = torch.randn(B, 1, D, generator=g).to(DEV, torch.bfloat16)
ADA = torch.arange(S, device=DEV)
ROPE = (torch.randn(S, 64, generator=g).to(DEV), torch.randn(S, 64, generator=g).to(DEV))
KPM = torch.ones(B, S, dtype=torch.bool, device=DEV)
TS = torch.full((B,), 500.0, device=DEV)


class ToyBlock(nn.Module):
    def __init__(self, i):
        super().__init__()
        self.w1 = nn.Parameter(W1[i], requires_grad=False)
        self.w2 = nn.Parameter(W2[i], requires_grad=False)
        self.calls = 0

    def forward(self, hidden_states, temb, adaln_indices, rotary_emb, key_padding_mask=None,
                timestep=None, valid_seq_len=None):
        self.calls += 1
        x = hidden_states
        x1 = x + 0.5 * torch.tanh(F.linear(x, self.w1)) + 0.01 * temb
        return x1 + 0.3 * torch.tanh(F.linear(x1, self.w2))


class ToyH3(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([ToyBlock(i) for i in range(NB)])

    def forward(self, x):
        h = x
        for blk in self.transformer_blocks:
            h = blk(h, TEMB, ADA, ROPE, KPM, TS, valid_seq_len=S)
        return h


def inputs(seed, n):
    gg = torch.Generator(device="cpu").manual_seed(seed)
    x0 = torch.randn(B, S, D, generator=gg).to(DEV, torch.bfloat16)
    return [(x0 * (1.0 - 0.015 * i)).contiguous() for i in range(n)]


def make(arm):
    toy = ToyH3().to(DEV)
    cfg = CacheDiTConfig(**cache_cfg(arm))
    pipe = types.SimpleNamespace(transformer=toy, rank=0)
    res = ce.enable_cache_dit_for_minimax_h3(pipe, cfg)
    return toy, res


def request(toy, res, seed, n):
    res.refresh(n)
    last = toy.transformer_blocks[-1]
    cached, outs = [], []
    xs = inputs(seed, n)
    for i in range(n):
        before = last.calls
        outs.append(toy(xs[i]).clone())
        if last.calls == before:
            cached.append(i)
    stats = ce.cache_dit_context_stats(toy)
    return cached, outs, stats


def main():
    fails, report = [], {}
    for arm in ARMS:
        mask = MASKS[arm]
        want = [i for i, v in enumerate(mask) if v == 0]
        print(f"== arm {arm}: bins={bins_of(mask)} cached={len(want)}", flush=True)
        toy, res = make(arm)
        c_w, _, st_w = request(toy, res, 7, 2)
        cA, oA, stA = request(toy, res, 42, N)
        c_cw, _, _ = request(toy, res, 7, 1)
        cB, oB, _ = request(toy, res, 43, N)
        cA2, oA2, _ = request(toy, res, 42, N)
        res.disable()
        toy2, res2 = make(arm)
        cF, oF, _ = request(toy2, res2, 42, N)
        res2.disable()
        r = {
            "warmup_cached": c_w, "warmup_stats": [(s["executed_steps"], s["cached_steps"]) for s in st_w],
            "A_cached_ok": cA == want, "A_stats_cached": [s["cached_steps"] for s in stA],
            "client_warmup_cached": c_cw, "B_cached_ok": cB == want, "A2_cached_ok": cA2 == want,
            "A2_bitwise_A": all(torch.equal(a, b) for a, b in zip(oA, oA2)),
            "fresh_bitwise_A": all(torch.equal(a, b) for a, b in zip(oA, oF)),
            "B_differs": not all(torch.equal(a, b) for a, b in zip(oA, oB)),
        }
        ok = (c_w == [1] and r["A_cached_ok"] and c_cw == [] and r["B_cached_ok"] and r["A2_cached_ok"]
              and r["A2_bitwise_A"] and r["fresh_bitwise_A"] and r["B_differs"]
              and stA and sorted(stA[0]["cached_steps"]) == want)
        r["ok"] = bool(ok)
        if not ok:
            fails.append(arm)
        print(f"   {json.dumps(r)}", flush=True)
        report[arm] = r
    print(json.dumps({"passed": not fails, "failed_arms": fails,
                      "arms": {k: v["ok"] for k, v in report.items()}}), flush=True)


if __name__ == "__main__":
    main()
