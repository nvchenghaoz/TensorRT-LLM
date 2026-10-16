# Config sweep for the small-N BF16 GEMV (tensorrt_llm/_torch/modules/small_n_gemv.py)
# against cuBLAS F.linear at the GDN in_proj_ba shape (N=96, K=5120), M=1..16.
# Cold L2: each CUDA graph holds 48 calls on 48 distinct weight/activation
# copies (one per GDN layer), and a 256 MiB read evicts L2 before every replay
# (outside the timed region). 10 warmup + 50 timed replays, CUDA events.
import itertools
import json
import sys

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modules import small_n_gemv as sng


def log(*a):
    print(*a, flush=True)


dev = torch.device("cuda")
LAYERS, WARMUP, ITERS = 48, 10, 50
flush_buf = torch.ones(64 * 1024 * 1024, dtype=torch.float32, device=dev)


def time_graph(fn, xs, ws):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for x, w in zip(xs, ws):
            fn(x, w)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for x, w in zip(xs, ws):
            fn(x, w)
    torch.cuda.synchronize()
    evs = [(torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True))
           for _ in range(WARMUP + ITERS)]
    for e0, e1 in evs:
        flush_buf.sum()
        e0.record()
        g.replay()
        e1.record()
    torch.cuda.synchronize()
    us = sorted(e0.elapsed_time(e1) * 1e3 / len(xs) for e0, e1 in evs[WARMUP:])
    del g
    return us[len(us) // 2], us[int(0.1 * len(us))], us[int(0.9 * len(us))]


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 96
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 5120
    Ms = [1, 2, 3, 4, 5, 6, 8, 12, 16]
    log("gpu", torch.cuda.get_device_name(), "N", N, "K", K)
    gen = torch.Generator(device=dev)
    gen.manual_seed(7)
    ws = [(torch.randn(N, K, device=dev, generator=gen) * 0.02).to(
        torch.bfloat16) for _ in range(LAYERS)]
    best = {}
    for M in Ms:
        xs = [torch.randn(M, K, device=dev, generator=gen).to(torch.bfloat16)
              for _ in range(LAYERS)]
        cub = time_graph(lambda x, w: F.linear(x, w), xs, ws)
        dflt = time_graph(lambda x, w: sng.small_n_gemv(x, w), xs, ws)
        log(f"M={M} cublas med {cub[0]:.2f} p90 {cub[2]:.2f} us | default "
            f"{sng._default_config(M, N, K)} med {dflt[0]:.2f} p90 {dflt[2]:.2f}")
        res = []
        bms = [b for b in (1, 2, 4, 8, 16) if b <= triton_np2(M)]
        for bm, bn, bk, nw in itertools.product(bms, (1, 2, 4), (256, 512, 1024),
                                                (2, 4, 8)):
            if bm * bn * bk // (32 * nw) > 128:
                continue
            cfg = (bm, bn, bk, nw)
            try:
                t = time_graph(lambda x, w: sng.small_n_gemv(x, w, cfg), xs, ws)
            except Exception as e:  # noqa
                log("cfg", cfg, "failed", repr(e)[:200])
                continue
            res.append((t[0], t[2], cfg))
        res.sort()
        for t, p90, cfg in res[:6]:
            log(f"  M={M} cfg {cfg} med {t:.2f} p90 {p90:.2f} us")
        best[str(M)] = {"cublas_us": cub[0], "default_us": dflt[0],
                        "best_us": res[0][0], "best_cfg": res[0][2]}
    print(json.dumps({"N": N, "K": K, "best": best}), flush=True)


def triton_np2(x):
    p = 1
    while p < x:
        p *= 2
    return p


main()
