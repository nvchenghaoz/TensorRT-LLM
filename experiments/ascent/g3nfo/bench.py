"""Idea 6 timing: the served nvfp4_out FMHA calls, same script in both trees (run_gpu_pair).

Module = attn_nvfp4_out_module's: (*load_trim_levers(ROUTED_LEVERS), "nvfp4_out"). On main the G3
guard sends it to the 4-warp kernel; on the branch it runs G3 with the NVFP4 epilogue.

Labels (2048^2, T=512 text):
  t2i-2048 / i2i-2048  CUDA graph of the step's FP8 attention calls (knowledge 316 call map):
                       n_bf16 G3 BF16-out calls at 16384 (unforked; the same kernel in both
                       trees), 4 nvfp4_out at 16384 (image mode blocks 1-4) and 28 nvfp4_out at
                       S_full (single blocks): t2i 1 + 4 @16384 + 28 @16896; i2i 6 + 4 @16384 +
                       28 @33280.
  fork-t2i / fork-i2i  served-order fork pair at S_full, 4 pairs per graph replay, reported per
                       pair: fork event, FMHA (current stream), then the side-stream CUTLASS
                       SwiGLU gate_up (M=S_full, N=18432, K=3072, NVFP4) waiting on the event,
                       join. (fmha_first order, flux3_primsts._side_gate_up_fork.)
  s4176-1024           the old 1024^2 chain (4096 / 4176), regression only.
kernel_ms = median per replay (per pair for fork labels); noise = (p90 - p10) / median.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import b186_nfo as b  # noqa: E402

log = b.log

# label -> (G3 BF16-out calls at S_img (unchanged by the branch), S_img, S_full); per step every
# label also runs 4 nfo calls at S_img + 28 nfo calls at S_full (knowledge 316 call map:
# t2i 5 x 16384 + 28 x 16896, i2i 10 x 16384 + 28 x 33280).
CHAINS = {"t2i-2048": (1, 16384, 16896), "i2i-2048": (6, 16384, 33280),
          "s4176-1024": (1, 4096, 4176)}
FORKS = {"fork-t2i": 16896, "fork-i2i": 33280}
HIDDEN, MLP2 = 3072, 2 * 9216


def stats(ts):
    ts = sorted(ts)
    n = len(ts)
    med = ts[n // 2] if n % 2 else 0.5 * (ts[n // 2 - 1] + ts[n // 2])
    p10 = ts[int(0.1 * (n - 1))]
    p90 = ts[int(round(0.9 * (n - 1)))]
    return med, (p90 - p10) / med


def time_graph(body, iters, warm=3):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        body()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        body()
    for _ in range(warm):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        e1.synchronize()
        ts.append(e0.elapsed_time(e1))
    del g
    return ts


def gate_up_inputs(rows, gen):
    """Synthetic NVFP4 gate_up operands at the served shapes (random E2M1 bytes, E4M3 scales)."""
    dev = "cuda"
    pr = (rows + 127) // 128 * 128
    a = torch.randint(0, 256, (rows, HIDDEN // 2), dtype=torch.uint8, device=dev, generator=gen)
    w = torch.randint(0, 256, (MLP2, HIDDEN // 2), dtype=torch.uint8, device=dev, generator=gen)
    sfa = torch.randint(0x30, 0x40, (pr * HIDDEN // 16,), dtype=torch.uint8, device=dev,
                        generator=gen)
    sfb = torch.randint(0x30, 0x40, (MLP2 * HIDDEN // 16,), dtype=torch.uint8, device=dev,
                        generator=gen)
    alpha = torch.ones(1, dtype=torch.float32, device=dev)
    inv = torch.ones(1, dtype=torch.float32, device=dev)
    return a, w, sfa, sfb, alpha, inv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", default="t2i-2048,i2i-2048,fork-t2i,fork-i2i,s4176-1024")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--rounds", type=int, default=3)
    args = p.parse_args()
    from tensorrt_llm._torch.visual_gen.attention.backends import fmha_v2_flux3 as lib

    labels = [x for x in args.labels.split(",") if x]
    sample = b.make_qkv(4176, 7, False)
    lev = (*lib.load_trim_levers(tuple(lib.ROUTED_LEVERS), sample["k"], sample["v"]), "nvfp4_out")
    t0 = time.time()
    mod = lib.get_flux3_d128_module(tuple(lev))
    bmod = lib.get_flux3_d128_module(tuple(lev[:-1]))  # the served G3 BF16-out module
    log(f"[build] {lev} {time.time() - t0:.1f}s")
    gscale = torch.tensor([448.0 * 6.0 / 8.0], dtype=torch.float32, device="cuda")
    out = {"batch": {}, "single_call_ms": {}, "census": {}, "levers": list(lev)}
    gen = torch.Generator(device="cuda").manual_seed(1234)
    for lab in labels:
        if lab in CHAINS:
            n_bf16, s_img, s_full = CHAINS[lab]
            inps = {S: b.make_qkv(S, 900 + S, False) for S in (s_img, s_full)}
            outs = {S: b.NfoOut(S, post=4096) for S in (s_img, s_full)}
            call = {S: (lambda S=S: b.call_nfo(lib, mod, inps[S], outs[S], gscale)) for S in inps}
            obf = torch.empty((1, s_img, b.H, b.D), device="cuda", dtype=torch.bfloat16)

            def bf16_call(S=s_img, o=obf):
                b.call_bf16(lib, bmod, inps[S], o)

            bf16_call()
            out["census"][f"bf16_S{s_img}"] = b.names(bf16_call)
            for S in inps:
                call[S]()
                out["census"][f"S{S}"] = b.names(call[S])

            def body(c=call, si=s_img, sf=s_full, nb=n_bf16):
                for _ in range(nb):
                    bf16_call()
                for _ in range(4):
                    c[si]()
                for _ in range(28):
                    c[sf]()

            iters = args.iters if s_full < 30000 else max(20, args.iters * 2 // 5)
            allts = []
            for r in range(args.rounds):
                ts = time_graph(body, iters)
                allts += ts
                m, nf = stats(ts)
                log(f"[{lab}] round {r} chain32 median {m:.3f} ms noise {nf:.4f}")
                for S in (s_img, s_full):
                    ts1 = time_graph(lambda S=S: [call[S]() for _ in range(4)], max(10, iters // 2))
                    m1, n1 = stats([t / 4 for t in ts1])
                    out["single_call_ms"].setdefault(f"S{S}", []).append(round(m1, 4))
                    log(f"[{lab}] round {r} single S{S} {m1:.4f} ms noise {n1:.4f}")
            m, nf = stats(allts)
            out["batch"][lab] = {"kernel_ms": round(m, 4), "noise_floor": round(nf, 4),
                                 "bf16_calls": n_bf16, "nfo_calls": 32, "S_img": s_img,
                                 "S_full": s_full}
            del inps, outs, call, obf
        else:
            from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_swiglu_fp4_cutlass as swc

            S = FORKS[lab]
            inp = b.make_qkv(S, 900 + S, False)
            nout = b.NfoOut(S, post=4096)
            a, w, sfa, sfb, alpha, inv = gate_up_inputs(S, gen)
            go, gsf = swc.allocate_outputs(S, MLP2, a.device)
            gmod = swc.get_module()
            side = torch.cuda.Stream()

            def gate_up():
                swc.swiglu_packed(a, w, sfa, sfb, alpha, inv, module=gmod, output=go,
                                  output_scale=gsf)

            gate_up()
            b.call_nfo(lib, mod, inp, nout, gscale)
            out["census"][f"fork_S{S}"] = b.names(lambda: b.call_nfo(lib, mod, inp, nout, gscale))

            def body():
                cur = torch.cuda.current_stream()
                for _ in range(4):
                    side.wait_stream(cur)
                    b.call_nfo(lib, mod, inp, nout, gscale)
                    with torch.cuda.stream(side):
                        gate_up()
                    cur.wait_stream(side)

            iters = args.iters if S < 30000 else max(20, args.iters * 2 // 5)
            allts = []
            for r in range(args.rounds):
                ts = [t / 4 for t in time_graph(body, iters)]
                allts += ts
                m, nf = stats(ts)
                tg = stats([t / 4 for t in time_graph(lambda: [gate_up() for _ in range(4)],
                                                      max(10, iters // 2))])[0]
                tf = stats([t / 4 for t in time_graph(
                    lambda: [b.call_nfo(lib, mod, inp, nout, gscale) for _ in range(4)],
                    max(10, iters // 2))])[0]
                out["single_call_ms"].setdefault(f"fork_S{S}_gate_up", []).append(round(tg, 4))
                out["single_call_ms"].setdefault(f"fork_S{S}_fmha", []).append(round(tf, 4))
                log(f"[{lab}] round {r} pair {m:.4f} ms noise {nf:.4f} (gate_up alone {tg:.4f}, "
                    f"fmha alone {tf:.4f})")
            m, nf = stats(allts)
            out["batch"][lab] = {"kernel_ms": round(m, 4), "noise_floor": round(nf, 4), "S": S}
            del inp, nout, a, w, sfa, sfb, go, gsf
        torch.cuda.empty_cache()
    out["output"] = {"class": "exact"}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
