"""Stage 0 (idea 243): isolated cost of torch.argmax(...).to(int32) + scatter_ on [B,129280] fp32."""
import json
import statistics

import torch

V = 129280
dev = "cuda"
torch.manual_seed(0)


def trio(logits, new_tokens, dest, mbw=1):
    nt = torch.argmax(logits, dim=-1).to(dtype=new_tokens.dtype)
    di = dest.unsqueeze(1).expand(-1, mbw)
    ne = nt.unsqueeze(1).expand(-1, mbw)
    new_tokens.view(-1, *new_tokens.shape[2:]).scatter_(0, di, ne)
    return nt


def make_dummy():
    # ~6 ms of elementwise work captured in a graph (no cuBLAS) to hide host launch cost
    a = torch.ones(16 * 1024 * 1024, device=dev)
    a.mul_(1.0)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(68):
            a.mul_(1.0)
    g._buf = a
    return g


def time_fn(fn, dummy, iters=200):
    ts = []
    for _ in range(iters):
        dummy.replay()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)
    ts.sort()
    return statistics.median(ts), (ts[int(0.9 * len(ts))] - ts[int(0.1 * len(ts))]) / statistics.median(ts)


def main():
    dummy = make_dummy()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record(); dummy.replay(); e.record(); e.synchronize()
    print("dummy graph ms", s.elapsed_time(e), flush=True)
    res = {}
    for B in (1, 2, 4, 8):
        logits = torch.randn(B, V, device=dev, dtype=torch.float32)
        new_tokens = torch.zeros(4, 64, 1, dtype=torch.int32, device=dev)
        dest = torch.arange(B, device=dev, dtype=torch.int64) * 3
        for _ in range(10):
            torch.argmax(logits, -1)
            trio(logits, new_tokens, dest)
        torch.cuda.synchronize()
        am_e = time_fn(lambda: torch.argmax(logits, -1), dummy)
        trio_e = time_fn(lambda: trio(logits, new_tokens, dest), dummy)
        # graphed
        ga = torch.cuda.CUDAGraph()
        with torch.cuda.graph(ga):
            torch.argmax(logits, -1)
        gt = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gt):
            trio(logits, new_tokens, dest)
        am_g = time_fn(ga.replay, dummy)
        trio_g = time_fn(gt.replay, dummy)
        # kernel names
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            trio(logits, new_tokens, dest)
            torch.cuda.synchronize()
        kn = []
        for ev in prof.events():
            if ev.device_type == torch.autograd.DeviceType.CUDA:
                kn.append((ev.name[:120], round(ev.device_time_total, 2)))
        res[B] = dict(argmax_eager_us=am_e, trio_eager_us=trio_e, argmax_graph_us=am_g,
                      trio_graph_us=trio_g, kernels=kn)
        print(B, json.dumps(res[B]), flush=True)
    print(json.dumps({"stage0": res}))


if __name__ == "__main__":
    main()
