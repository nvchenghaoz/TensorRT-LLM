"""Idea 184: kernel-only A/B of the routed fmha_v2 flux3 kernel, BF16 O store (main) vs the
nvfp4_out epilogue (nfo), plus main's attn_out quantize alone and FMHA + quantize.

Graphs (one per arm, each = 28 x S4176 + 4 x S4096 calls on 2 cycled input sets per S):
  bf16   : routed module run() -> BF16 O
  nfo    : nfo module run_nvfp4_out -> FP4 + SF
  quant  : nvfp4_quantize_prepared of a BF16 O (the served attn_out quantize) alone
  bf16q  : bf16 then quant per call (main's two kernels)
  bf16_2 / nfo_2 : repeats (A/A, B/B)
Interleaved rounds behind a spin; median us per call (graph time / 32).
"""

import json
import os
import time

import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.visual_gen.attention.backends import fmha_v2_flux3 as fx
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp
from tensorrt_llm._torch.visual_gen.models.flux3.primsts import PreparedNVFP4

H, D, HID = 24, 128, 3072
ROUNDS = int(os.environ.get("IDEA184_ROUNDS", "80"))
SPIN = 6_000_000


def log(*a):
    print(*a, flush=True)


def qkv(S, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    t = [(torch.randn((1, S, H, D), device="cuda", generator=g) * 32).to(torch.float8_e4m3fn)
         for _ in range(3)]
    s1 = torch.tensor([(1 / 32) ** 2 / D ** 0.5], device="cuda")
    s2 = torch.tensor([1 / 32], device="cuda")
    return (*t, s1, s2)


def main():
    levers = tuple(fx.ROUTED_LEVERS)
    mod_b = fx.get_flux3_d128_module(levers)
    mod_n = fx.get_flux3_d128_module(levers + ("nvfp4_out",))
    w = torch.zeros((HID, HID // 2), device="cuda", dtype=torch.uint8)
    wsf = torch.zeros((HID * HID // 16,), device="cuda", dtype=torch.uint8)
    out_ad = PreparedNVFP4(w, wsf, 1.0, 1.0)
    sets = {S: [qkv(S, S + i) for i in range(2)] for S in (4176, 4096)}
    plan = [sets[4176][i % 2] for i in range(28)] + [sets[4096][i % 2] for i in range(4)]
    outs = {id(c): torch.empty((1, c[0].shape[1], H, D), device="cuda", dtype=torch.bfloat16)
            for c in plan}
    fp4s = {id(c): torch.empty((c[0].shape[1], HID // 2), device="cuda", dtype=torch.uint8)
            for c in plan}
    sfs = {id(c): torch.empty((fx.nvfp4_scale_bytes(c[0].shape[1], HID),), device="cuda",
                              dtype=torch.uint8) for c in plan}

    def bf16(c):
        S = c[0].shape[1]
        fx.fmha_v2_flux3_d128(*c[:3], outs[id(c)], H, D, S, scale_bmm1_d=c[3], scale_bmm2_d=c[4],
                              causal=False, lse=None, module=mod_b)

    def quant(c):
        S = c[0].shape[1]
        fp.nvfp4_quantize_prepared(outs[id(c)].view(S, HID), out_ad.weight)

    def nfo(c):
        S = c[0].shape[1]
        fx.fmha_v2_flux3_d128_nvfp4_out(*c[:3], fp4s[id(c)], sfs[id(c)], out_ad.inverse_scale, H,
                                        D, S, scale_bmm1_d=c[3], scale_bmm2_d=c[4], module=mod_n)

    arms = {
        "bf16": lambda: [bf16(c) for c in plan],
        "nfo": lambda: [nfo(c) for c in plan],
        "quant": lambda: [quant(c) for c in plan],
        "bf16q": lambda: [(bf16(c), quant(c)) for c in plan],
        "bf16_2": lambda: [bf16(c) for c in plan],
        "nfo_2": lambda: [nfo(c) for c in plan],
    }
    for fn in arms.values():
        fn()
    torch.cuda.synchronize()
    st = torch.cuda.Stream()
    graphs = {}
    for k, fn in arms.items():
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=st):
            fn()
        graphs[k] = g
    torch.cuda.synchronize()
    t0 = time.time()
    while time.time() - t0 < 12:
        for g in graphs.values():
            g.replay()
        torch.cuda.synchronize()
    keys = list(graphs)
    times = {k: [] for k in keys}
    for r in range(ROUNDS):
        order = keys if r % 2 == 0 else keys[::-1]
        evs = []
        for k in order:
            torch.cuda.synchronize()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda._sleep(SPIN)
            a.record()
            graphs[k].replay()
            b.record()
            evs.append((k, a, b))
        torch.cuda.synchronize()
        for k, a, b in evs:
            times[k].append(a.elapsed_time(b) * 1e3 / 32)
        if r % 20 == 0:
            log(f"round {r}: " + " ".join(f"{k}={times[k][-1]:.2f}" for k in keys))
    res = {}
    for k in keys:
        v = sorted(times[k])
        n = len(v)
        res[k] = {"us_per_call": round(v[n // 2], 2), "p10": round(v[n // 10], 2),
                  "p90": round(v[(9 * n) // 10], 2)}
    pr = sorted(b / a for a, b in zip(times["bf16"], times["nfo"]))
    res["nfo_minus_bf16_us"] = round((res["nfo"]["us_per_call"] + res["nfo_2"]["us_per_call"]
                                      - res["bf16"]["us_per_call"] - res["bf16_2"]["us_per_call"])
                                     / 2, 2)
    res["bf16q_minus_nfo_us"] = round(res["bf16q"]["us_per_call"] - (
        res["nfo"]["us_per_call"] + res["nfo_2"]["us_per_call"]) / 2, 2)
    res["aa_us"] = round(res["bf16_2"]["us_per_call"] - res["bf16"]["us_per_call"], 2)
    res["bb_us"] = round(res["nfo_2"]["us_per_call"] - res["nfo"]["us_per_call"], 2)
    res["paired_nfo_over_bf16_median"] = pr[len(pr) // 2]
    print(json.dumps(res))


if __name__ == "__main__":
    main()
