"""Idea 307 microbenchmark: DeepSeek-V4 embed_tokens lookup (+ the mHC seed repeat that
consumes it in DeepseekV4Model.forward), as this tree's model builds it.

Decode M = 1, 2, 4, 8 and prefill M = 2048, synthetic ids/table at the checkpoint's
vocab_size x hidden_size (bf16, full table on every rank: Embedding has no mapping).
Each configuration is captured in a CUDA graph of K back-to-back chains
(embed -> unsqueeze(1).repeat(1, hc_mult, 1)) and timed with CUDA events over many
replays; kernel_ms is per chain. noise_floor = (p90 - p10) / median of the samples.
Usage: bench.py <checkpoint>
"""
import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ckpt_arg, log, make_embedding, model_dims, nvsmi, tree_flag  # noqa: E402

K = 16
REPLAYS = 20
SAMPLES = 30


def pct(xs, q):
    xs = sorted(xs)
    i = (len(xs) - 1) * q
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def time_graph(fn, chains):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    keep = []
    with torch.cuda.graph(g):
        for _ in range(chains):
            keep.append(fn())
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(SAMPLES):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(REPLAYS):
            g.replay()
        b.record()
        b.synchronize()
        samples.append(a.elapsed_time(b) / (REPLAYS * chains))
    med = statistics.median(samples)
    return med, (pct(samples, 0.9) - pct(samples, 0.1)) / med, g, keep


def main():
    nvsmi()
    flag, path, call = tree_flag()
    V, H, hc, dt = model_dims(ckpt_arg())
    log(f"tree file {path}")
    log(f"ctor call: Embedding{call}")
    log(f"use_gather_kernel={flag} V={V} H={H} hc_mult={hc} dtype={dt} "
        f"TRTLLM_ENABLE_PDL={os.environ.get('TRTLLM_ENABLE_PDL', '(unset=1)')}")
    torch.manual_seed(0)
    w = torch.randn(V, H, dtype=dt, device="cuda")
    emb = make_embedding(V, H, dt, flag, weight=w)
    out = {"batch": {}, "output": {"class": "exact"},
           "use_gather_kernel": flag, "V": V, "H": H, "embed_only_ms": {}, "cold": {}}
    flush = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    for M in (1, 2, 4, 8, 2048):
        ids = torch.randint(0, V, (M, ), dtype=torch.int32, device="cuda")
        chains = K if M <= 8 else 4

        def chain():
            h = emb(ids)
            return h.unsqueeze(1).repeat(1, hc, 1)

        def only():
            return emb(ids)

        med, noise, g, _ = time_graph(chain, chains)
        del g
        med_e, noise_e, g2, _ = time_graph(only, chains)
        del g2
        # cold variant: one chain per replay after a 256 MB L2 flush (ids and table rows cold)
        _, _, g3, _ = time_graph(chain, 1)
        cold = []
        for _ in range(60):
            flush.zero_()
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record()
            g3.replay()
            b.record()
            b.synchronize()
            cold.append(a.elapsed_time(b))
        del g3
        cmed = statistics.median(cold)
        out["cold"][str(M)] = {"kernel_ms": round(cmed, 6),
                               "noise_floor": round((pct(cold, 0.9) - pct(cold, 0.1)) / cmed, 4)}
        log(f"M={M}: cold single replay {cmed * 1e3:.2f} us")
        out["batch"][str(M)] = {"kernel_ms": round(med, 6), "noise_floor": round(noise, 4)}
        out["embed_only_ms"][str(M)] = {"kernel_ms": round(med_e, 6),
                                        "noise_floor": round(noise_e, 4)}
        log(f"M={M}: chain {med * 1e3:.2f} us (noise {noise:.3f}), "
            f"embed only {med_e * 1e3:.2f} us (noise {noise_e:.3f})")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
