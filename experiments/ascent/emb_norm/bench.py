"""Paired microbenchmark: main's embedding chain + layer-0 input norm + final
norm (all rows, then gather) vs the fused embed_scale_rmsnorm + final norm
on the gathered rows. The arm is picked by whether the tree has the fused
kernel. CUDA events, 256 MB L2 flush before every timed iteration."""
import json
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import CASES, build_modules, fused_chain, have_fused, main_chain, make_inputs, read_dims  # noqa: E402

WARMUP, ITERS = 5, 30


def main():
    dims = read_dims(sys.argv[1])
    fused = have_fused()
    print("arm", "candidate(fused)" if fused else "baseline(main chain)", flush=True)
    emb, norm0, normf = build_modules(dims)
    fn = fused_chain if fused else main_chain
    flush = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    out = {}
    for label in CASES:
        ids, x_last, seq_lens = make_inputs(label, dims, seed=1)
        for _ in range(WARMUP):
            fn(emb, norm0, normf, ids, x_last, seq_lens)
        torch.cuda.synchronize()
        times = []
        for _ in range(ITERS):
            flush.fill_(1)
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            fn(emb, norm0, normf, ids, x_last, seq_lens)
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        t = torch.tensor(times)
        med = float(t.median())
        p10, p90 = float(t.quantile(0.1)), float(t.quantile(0.9))
        out[label] = {"kernel_ms": med, "p10_ms": p10, "p90_ms": p90,
                      "noise_floor": (p90 - p10) / med, "M": int(ids.numel())}
        print(label, out[label], flush=True)
    res = {"batch": out, "output": {"class": "exact"}, "arm": "fused" if fused else "main"}
    if fused:
        # Untimed correctness in the same process (queue hold: one job): bitwise
        # vs main's chain, repeated-call sequence (5 warmup + 30 calls, as
        # timed), FP32 reference, out-of-range ids, A,B,A order.
        from check import run as run_check
        print("correctness (untimed)", flush=True)
        chk = run_check(dims, (emb, norm0, normf))
        res["correctness"] = chk
        res["passed"] = chk["passed"]
    print(json.dumps(res))


if __name__ == "__main__":
    main()
