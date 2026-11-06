"""Shared helpers for the fast-K1 (TRTLLM_MHC_K1_FAST) probes: inputs and cold-L2 graph timing."""

import statistics

import torch

HIDDEN, NHC, EPS = 5120, 4, 1e-6


def log(*a):
    print(*a, flush=True)


def make_set(M, gen, dev, hidden=HIDDEN):
    x = torch.randn(M, hidden, generator=gen, device=dev).to(torch.bfloat16)
    res = torch.randn(M, NHC, hidden, generator=gen, device=dev).to(torch.bfloat16)
    post = (2.0 * torch.rand(M, NHC, generator=gen, device=dev)).float().contiguous()
    comb = torch.rand(M, NHC, NHC, generator=gen, device=dev)
    comb = (comb / comb.sum(-2, keepdim=True)).float().contiguous()
    pre = (torch.rand(M, NHC, generator=gen, device=dev) + 1e-6).float().contiguous()
    nw = (1.0 + 0.1 * torch.randn(hidden, generator=gen, device=dev)).to(torch.bfloat16)
    return x, res, post, comb, pre, nw


def stats(v):
    v = sorted(v)
    med = statistics.median(v)
    p10 = v[int(0.1 * (len(v) - 1))]
    p90 = v[int(0.9 * (len(v) - 1))]
    return {"median": round(med, 4), "p10": round(p10, 4), "p90": round(p90, 4),
            "noise": round((p90 - p10) / med, 4) if med else 0.0}


class ColdGraph:
    """A CUDA graph of `n` K1 launches on distinct input sets (distinct norm weights too, as the
    80 boundaries of a step have); every timed replay follows an L2 flush."""

    def __init__(self, call, sets):
        self.call, self.sets = call, sets
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                for st in sets:
                    call(st)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g):
            for st in sets:
                call(st)
        torch.cuda.synchronize()
        for _ in range(3):
            self.g.replay()
        torch.cuda.synchronize()

    def time_us(self, flush):
        flush.zero_()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        self.g.replay()
        b.record()
        b.synchronize()
        return a.elapsed_time(b) * 1000.0 / len(self.sets)


def abba(graphs, flush, rounds):
    """graphs: {name: ColdGraph}. ABBA order over rounds; returns {name: [us per launch]}."""
    names = list(graphs)
    out = {k: [] for k in names}
    for r in range(rounds):
        order = names if r % 2 == 0 else names[::-1]
        for k in order:
            out[k].append(graphs[k].time_us(flush))
    return out
