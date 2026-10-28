"""CUDA-event timing: unfused (silu_and_mul -> warp_fp8_per_token_quant) vs fused op.

Runs in either tree. In the base tree (no fused op) the arm reports the unfused pair; in the
branch tree it reports the fused op. Both arms also time the unfused pair twice (interleaved)
as an A/A row, and the whole FP8R MLP chain (gate_up GEMM + middle + down GEMM).
Each timed call of a middle op is preceded by a 256 MB L2 flush (the real input is the 154 MB
gate_up output, larger than L2). Median of ITERS interleaved iterations after WARMUP.
usage: bench.py [num_warps,...]   (warps sweep is a candidate-only diagnostic)
Last stdout line: evidence JSON.
"""
import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

WARMUP = 10
ITERS = 50
MS = (4176, 4096)


def stats(xs):
    xs = sorted(xs)
    med = statistics.median(xs)
    p10 = xs[int(0.1 * (len(xs) - 1))]
    p90 = xs[int(round(0.9 * (len(xs) - 1)))]
    return {"median_us": round(med * 1e3, 2), "p10_us": round(p10 * 1e3, 2),
            "p90_us": round(p90 * 1e3, 2), "noise": round((p90 - p10) / med, 4), "n": len(xs)}


def main():
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.visual_gen.models.flux3 import fp8

    fused_ok = C.has_fused()
    warps = [int(v) for v in sys.argv[1].split(",")] if len(sys.argv) > 1 else [None]
    C.log(f"[bench] fused op available: {fused_ok}; warps {warps}")
    torch.manual_seed(0)
    w_up, w_down = C.weights()
    flush = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    table = {}
    for m in MS:
        x, (a_d, a_s) = C.gate_up_input(m, w_up, seed=m)
        pristine = x.clone()
        variants = {"unfused_A": C.unfused, "unfused_A2": C.unfused}
        if fused_ok:
            for wv in warps:
                def f(t, wv=wv):
                    if wv is not None:
                        fp8._SWIGLU_FP8R_NUM_WARPS = wv
                    return C.fused(t)
                variants["fused" if wv is None else f"fused_w{wv}"] = f
        chains = {"chain_unfused": C.unfused}
        if fused_ok:
            chains["chain_fused"] = variants[[k for k in variants if k.startswith("fused")][0]]
        times = {k: [] for k in list(variants) + list(chains)}
        for it in range(WARMUP + ITERS):
            for name, fn in variants.items():
                flush.zero_()
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                fn(x)
                e.record()
                torch.cuda.synchronize()
                if it >= WARMUP:
                    times[name].append(s.elapsed_time(e))
            for name, fn in chains.items():
                flush.zero_()
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                gu = C.gemm(a_d, a_s, w_up)
                d, sc = fn(gu)
                C.gemm(d, sc.reshape(-1), w_down)
                e.record()
                torch.cuda.synchronize()
                if it >= WARMUP:
                    times[name].append(s.elapsed_time(e))
            if it % 20 == 0:
                C.log(f"[bench] M={m} it {it}")
        assert torch.equal(x.view(torch.int16), pristine.view(torch.int16)), "input changed"
        table[m] = {k: stats(v) for k, v in times.items()}
        for k, v in table[m].items():
            C.log(f"[table] M={m} {k:14s} median {v['median_us']:8.2f} us  "
                  f"p10 {v['p10_us']:8.2f} p90 {v['p90_us']:8.2f} noise {v['noise']:.4f}")
        a, a2 = table[m]["unfused_A"]["median_us"], table[m]["unfused_A2"]["median_us"]
        C.log(f"[table] M={m} A/A delta {a2 - a:+.2f} us ({(a2 - a) / a * 100:+.2f}%)")
    key = "fused" if fused_ok and "fused" in table[4176] else "unfused_A"
    if fused_ok and key not in table[4176]:
        key = [k for k in table[4176] if k.startswith("fused")][0]
    main_row = table[4176][key]
    print(json.dumps({
        "batch": {"1": {"kernel_ms": main_row["median_us"] / 1e3, "noise_floor": main_row["noise"]}},
        "output": {"class": "exact"},
        "reported": f"{key} at M=4176 (single_blocks.24-27); M=4096 (image.0) in table",
        "table_us": {str(m): v for m, v in table.items()},
    }))


if __name__ == "__main__":
    main()
