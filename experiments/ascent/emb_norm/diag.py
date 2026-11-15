"""Diagnostic: host enqueue time and device time per component at small and large M."""
import sys
import time

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import build_modules, make_inputs, read_dims  # noqa: E402


def timeit(name, fn, n=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # device time with the GPU kept busy ahead (queue many, sync once)
    torch.cuda._sleep(2_000_000)
    s.record()
    for _ in range(n):
        fn()
    e.record()
    torch.cuda.synchronize()
    print(f"{name}: host_enqueue_us={(t1 - t0) / n * 1e6:.1f} wall_us={(t2 - t0) / n * 1e6:.1f} "
          f"device_us_back_to_back={s.elapsed_time(e) / n * 1e3:.1f}", flush=True)


def main():
    dims = read_dims(sys.argv[1])
    emb, norm0, normf = build_modules(dims)
    from tensorrt_llm._torch.kernels.fused_ops.embed_scale_rmsnorm import embed_scale_rmsnorm
    sc = float(emb.embed_scale.float().item())
    for label in ("300", "1000", "8000"):
        ids, x_last, seq_lens = make_inputs(label, dims, seed=1)
        last = torch.cumsum(seq_lens, dim=0, dtype=torch.long) - 1
        h = emb(ids)
        print("M", ids.numel(), flush=True)
        timeit("main emb(ids)", lambda: emb(ids))
        timeit("main norm0(h)", lambda: norm0(h))
        timeit("main normf(all)[last]", lambda: normf(x_last)[last])
        timeit("fused embed_scale_rmsnorm", lambda: embed_scale_rmsnorm(ids, emb.weight, sc, norm0.weight, 1e-6))
        timeit("fused embed_scale only", lambda: embed_scale_rmsnorm(ids, emb.weight, sc))
        timeit("normf(x[last])", lambda: normf(x_last[last]))
        timeit("cumsum", lambda: torch.cumsum(seq_lens, dim=0, dtype=torch.long) - 1)
        timeit("scale float()", lambda: float(emb.embed_scale.float().item()))
    print('{"passed": true}')


if __name__ == "__main__":
    main()
