"""Idea 201 ncu probe: one fused fc1+fc2 launch per case between cudaProfilerStart/Stop.

Cases (EP4 rank by default): M1 D3 and M4 D6, cold L2 (512 MB read-only flush before each).
Weight bytes the launch must read: D x (fc1 12.53 MB + fc2 6.27 MB) at EP4.
Run under profile_kernels(capture="region").
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from common12 import H, I, SHAPE, Runner, log, make_state, routing  # noqa: E402

CASES = ((1, 3), (4, 6))


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
        data.append((x, ids, w))
    torch.cuda.synchronize()
    fc1_b = 2 * I * H // 2 + 2 * I * H // 32
    fc2_b = H * I // 2 + H * I // 32
    for (m, d), (x, ids, w) in zip(CASES, data):
        log(f"{SHAPE} M{m} D{d}: weight bytes {d * (fc1_b + fc2_b) / 1e6:.2f} MB "
            f"(fc1 {d * fc1_b / 1e6:.2f} + fc2 {d * fc2_b / 1e6:.2f})")
    torch.cuda.profiler.start()
    for (m, d), (x, ids, w) in zip(CASES, data):
        flush.sum()
        torch.cuda.synchronize()
        run(x, ids, w, "fused")
        torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print('{"done": true}')


if __name__ == "__main__":
    main()
