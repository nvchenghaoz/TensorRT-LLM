"""Idea 201 ncu probe (acceptance item 4): DRAM bytes of the fused fc1+fc2 kernel vs main's
fc1_act_q1 + fc2_finalize at fixed routings, one launch each between cudaProfilerStart/Stop.

Run under profile_kernels(capture="region", kernel_regex="cutlass"): only the three CuTe-DSL
GEMM kernels match (the route kernels are Triton).  Region launch order, per case:
    fused, base fc1_act_q1, base fc2_finalize
Cases (EP4 rank by default): (M, D) = (1,1) (1,3) (1,6) (2,6) (4,6).  ncu flushes the caches
before every profiled launch, so each launch reads cold.
Bytes the routing requires: D x (fc1 12.53 MB + fc2 6.27 MB) of weights (MXFP4 + UE8M0 SF)
plus activations (q0 + sf0 in, q1/sf1 out and back in, the BF16 output).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from common12 import H, I, SHAPE, TOPK, Runner, log, make_state, routing  # noqa: E402

CASES = ((1, 1), (1, 3), (1, 6), (2, 6), (4, 6))


def main():
    import tensorrt_llm  # noqa: F401
    import tensorrt_llm._torch.custom_ops  # noqa: F401

    leaf, st = make_state()
    run = Runner(leaf, st)
    flush = torch.ones(512 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    data = []
    for m, d in CASES:
        x = torch.randn(m, H, dtype=torch.bfloat16, device="cuda")
        ids, w = routing(m, d)
        run(x, ids, w, "fused")  # compile + warm outside the region
        run(x, ids, w, "base")
        data.append((x, ids, w))
    torch.cuda.synchronize()
    fc1_b = 2 * I * H // 2 + 2 * I * H // 32
    fc2_b = H * I // 2 + H * I // 32
    expect = []
    for (m, d), (x, ids, w) in zip(CASES, data):
        pairs = m * TOPK
        # activations: q0 (fp8) + sf0 in; q1 (fp8) + sf1 written by fc1 (and read by fc2 in
        # the chain); BF16 output written.  Upper bound: all pairs routed here.
        act_in = min(pairs, d * m) * H * (1 + 1 / 32)
        q1 = min(pairs, d * m) * I * (1 + 1 / 32)
        outb = m * H * 2
        e = {"case": f"M{m}_D{d}", "weights_MB": round(d * (fc1_b + fc2_b) / 1e6, 3),
             "fc1_weights_MB": round(d * fc1_b / 1e6, 3), "fc2_weights_MB": round(d * fc2_b / 1e6, 3),
             "act_in_MB": round(act_in / 1e6, 4), "q1_sf1_MB": round(q1 / 1e6, 4),
             "out_MB": round(outb / 1e6, 4)}
        expect.append(e)
        log(f"{SHAPE} M{m} D{d}: weight bytes {e['weights_MB']:.2f} MB "
            f"(fc1 {e['fc1_weights_MB']:.2f} + fc2 {e['fc2_weights_MB']:.2f}); activations "
            f"q0 {e['act_in_MB']} MB, q1/sf1 {e['q1_sf1_MB']} MB, out {e['out_MB']} MB")
    order = []
    torch.cuda.profiler.start()
    for (m, d), (x, ids, w) in zip(CASES, data):
        for mode in ("fused", "base"):
            flush.sum()
            torch.cuda.synchronize()
            run(x, ids, w, mode)
            torch.cuda.synchronize()
            order += [f"M{m}_D{d}:fused"] if mode == "fused" else [f"M{m}_D{d}:base_fc1",
                                                                  f"M{m}_D{d}:base_fc2"]
    torch.cuda.profiler.stop()
    log(f"region launch order (cutlass kernels): {order}")
    print(json.dumps({"done": True, "shape": SHAPE, "order": order, "expect": expect}))


if __name__ == "__main__":
    main()
