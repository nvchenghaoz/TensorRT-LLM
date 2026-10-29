"""Idea 6 side arms (top-lead request): per-call and per-pass times of three attention-out routes
on ONE GPU, at the served shapes S = 16384 (mode blocks), 16464 (t2i single), 32848 (i2i single).

  nfo      the tree's served nvfp4_out module ((*load_trim(ROUTED_LEVERS), nvfp4_out)): on main the
           4-warp nfo kernel, on branch 1 G3 + NVFP4 epilogue
  g3bf16   the served G3 BF16-out module alone (BF16 O, no quantize)
  g3bf16q  g3bf16 + FlashInfer nvfp4_quantize(O, gs, layout_128x4, cute-dsl) as a separate launch
           (the ATTN_NVFP4_OUT=0 route's quantize; in serving it sits in the LPT pair, so this is
           its standalone cost)
Per-pass sums (served call map): t2i 1 x g3bf16@16384 + 4 x R@16384 + 28 x R@16464; i2i
6 x g3bf16@16384 + 4 x R@16384 + 28 x R@32848, R in {nfo, g3bf16q}. CUDA-graph timed, medians.
Also checks nfo fp4+sf == nvfp4_quantize(g3bf16 O) bitwise at each S (the epilogue's contract).
Last line JSON (no "batch" key: diagnostics, not the evidence pair).
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import b186_nfo as b  # noqa: E402
from bench import stats, time_graph  # noqa: E402

log = b.log
SIZES = (16384, 16464, 32848)
PASSES = {"t2i": (1, 16384, 16464), "i2i": (6, 16384, 32848)}


def main():
    from flashinfer import SfLayout, nvfp4_quantize

    from tensorrt_llm._torch.visual_gen.attention.backends import fmha_v2_flux3 as lib

    sample = b.make_qkv(4176, 7, False)
    lev = (*lib.load_trim_levers(tuple(lib.ROUTED_LEVERS), sample["k"], sample["v"]), "nvfp4_out")
    mod = lib.get_flux3_d128_module(tuple(lev))
    bmod = lib.get_flux3_d128_module(tuple(lev[:-1]))
    gscale = torch.tensor([448.0 * 6.0 / 8.0], dtype=torch.float32, device="cuda")
    res = {"levers": list(lev), "per_call_ms": {}, "per_pass_ms": {}, "bitwise_nfo_vs_bf16q": {},
           "census": {}}
    inps = {S: b.make_qkv(S, 700 + S, False) for S in SIZES}
    nouts = {S: b.NfoOut(S, post=4096) for S in SIZES}
    obf = {S: torch.empty((1, S, b.H, b.D), device="cuda", dtype=torch.bfloat16) for S in SIZES}

    def nfo(S):
        b.call_nfo(lib, mod, inps[S], nouts[S], gscale)

    def g3bf16(S):
        b.call_bf16(lib, bmod, inps[S], obf[S])

    def g3bf16q(S):
        g3bf16(S)
        return nvfp4_quantize(obf[S].view(S, b.W), gscale, sfLayout=SfLayout.layout_128x4,
                              do_shuffle=False, backend="cute-dsl")

    arms = {"nfo": nfo, "g3bf16": g3bf16, "g3bf16q": g3bf16q}
    for S in SIZES:
        for name, fn in arms.items():
            fn(S)
        res["census"][f"nfo_S{S}"] = b.names(lambda: nfo(S))
        nouts[S].reset()
        nfo(S)
        q, sf = g3bf16q(S)
        torch.cuda.synchronize()
        fp4_eq = torch.equal(nouts[S].fp4.reshape(-1), q.view(torch.uint8).reshape(-1))
        sf_eq = torch.equal(nouts[S].sf.reshape(-1), sf.view(torch.uint8).reshape(-1))
        res["bitwise_nfo_vs_bf16q"][f"S{S}"] = {"fp4": fp4_eq, "sf": sf_eq}
        log(f"[S{S}] nfo vs bf16+quantize fp4 {fp4_eq} sf {sf_eq}")
    iters = 30
    for r in range(3):
        for S in SIZES:
            for name, fn in arms.items():
                n = iters if S < 30000 else 12
                m = stats([t / 4 for t in time_graph(lambda f=fn, s=S: [f(s) for _ in range(4)],
                                                     n)])[0]
                res["per_call_ms"].setdefault(f"{name}_S{S}", []).append(round(m, 4))
        log(f"round {r} per-call {json.dumps({k: v[-1] for k, v in res['per_call_ms'].items()})}")
        for case, (nb, si, sfull) in PASSES.items():
            for name in ("nfo", "g3bf16q"):
                fn = arms[name]

                def body(fn=fn, nb=nb, si=si, sfull=sfull):
                    for _ in range(nb):
                        g3bf16(si)
                    for _ in range(4):
                        fn(si)
                    for _ in range(28):
                        fn(sfull)

                n = 20 if case == "t2i" else 8
                m = stats(time_graph(body, n))[0]
                res["per_pass_ms"].setdefault(f"{case}_{name}", []).append(round(m, 3))
        log(f"round {r} per-pass {json.dumps({k: v[-1] for k, v in res['per_pass_ms'].items()})}")
    res["median"] = {k: sorted(v)[len(v) // 2] for d in ("per_call_ms", "per_pass_ms")
                     for k, v in res[d].items()}
    res["passed"] = all(v["fp4"] and v["sf"] for v in res["bitwise_nfo_vs_bf16q"].values())
    print(json.dumps(res))


if __name__ == "__main__":
    main()
