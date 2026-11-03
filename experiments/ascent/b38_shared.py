"""Branch 38: CSA2FlashInfer.run_shared prefill glue (bench + bitwise check).

Usage: python b38_shared.py bench|check
Synthetic one-rank shapes: 16 heads, hd 512, BF16, bank 36864 rows.
"""

import json
import math
import sys
import time
from types import SimpleNamespace

import torch

from tensorrt_llm._torch.attention.backends.sparse.csa2.backend import CSA2FlashInfer
from tensorrt_llm._torch.attention.backends.interface import AttentionInputType

HEADS, DIM, BANK = 16, 512, 36864
SCALE = 1.0 / math.sqrt(DIM)
WARMUP, ITERS = 3, 30
dev = torch.device("cuda")


def log(*a):
    print(*a, flush=True)


def make_indices(count, width, gen):
    idx = torch.randint(0, BANK, (count, width), generator=gen, dtype=torch.int32)
    kind = torch.arange(count) % 5
    # 0: random holes, 1: suffix holes, 2: fully empty, 3: full, 4: holes + suffix
    holes = torch.rand(count, width, generator=gen) < 0.15
    lens = torch.randint(0, width + 1, (count,), generator=gen)
    suffix = torch.arange(width)[None, :] >= lens[:, None]
    m = torch.zeros(count, width, dtype=torch.bool)
    m[kind == 0] = holes[kind == 0]
    m[kind == 1] = suffix[kind == 1]
    m[kind == 2] = True
    m[kind == 4] = (holes | suffix)[kind == 4]
    idx[m] = -1
    return idx.to(dev)


class Case:
    def __init__(self, count, width, bank, sink, seed):
        gen = torch.Generator().manual_seed(seed)
        self.count, self.width = count, width
        self.q0 = (torch.randn(count, HEADS, DIM, generator=gen) * 2).to(torch.bfloat16).to(dev)
        self.i0 = make_indices(count, width, gen)
        self.q = self.q0.clone()
        self.idx = self.i0.clone()
        self.out = torch.empty(count, HEADS * DIM, dtype=torch.bfloat16, device=dev)
        self.bank, self.sink = bank, sink

    def reset(self):
        self.q.copy_(self.q0)
        self.idx.copy_(self.i0)
        self.out.fill_(7.0)


class FakeAttention:
    inputs = None

    def prepare_selected(self, q, metadata, forward_args):
        return self.inputs


def make_helper(fused=None):
    attn = FakeAttention()
    fi = CSA2FlashInfer(attn)
    if fused is not None:
        fi.fused_glue = fused
    return fi, attn


MD = SimpleNamespace(shared_plan=object(), num_contexts=1, _csa2_graph_reserved=False)


def call(fi, attn, case):
    attn.inputs = (case.q, case.bank, case.idx, case.sink, SCALE)
    fa = SimpleNamespace(output=case.out, attention_input_type=AttentionInputType.context_only)
    return fi._forward_selected(case.q.view(case.count, -1), MD, fa)


def plan_of(fi, case):
    return fi._paged_plans[(case.count, HEADS, case.width, dev, SCALE)]


def common():
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(1)
    bank = torch.randn(BANK, DIM, generator=gen).to(torch.bfloat16).to(dev)
    sink = (torch.randn(HEADS, generator=gen) * 2).float().to(dev)
    return bank, sink


def sm_clock():
    try:
        return torch.cuda.clock_rate()
    except Exception:
        return None


def stats(xs):
    xs = sorted(xs)
    n = len(xs)
    med = xs[n // 2]
    p10, p90 = xs[int(0.1 * (n - 1))], xs[int(0.9 * (n - 1))]
    return {"median": med, "p10": p10, "p90": p90, "noise": (p90 - p10) / med}


def bench():
    bank, sink = common()
    fi, attn = make_helper()
    widths = (640, 128)
    # One request: layers 0-19 at 2048 rows, 20-39 at 128 rows, widths alternating.
    layers = [(2048 if l < 20 else 128, widths[l % 2]) for l in range(40)]
    cases = {}
    for s, (c, w) in enumerate(sorted(set(layers))):
        cases[(c, w)] = Case(c, w, bank, sink, 100 + s)
    log("cases built", list(cases))

    def reset_all():
        for cs in cases.values():
            cs.reset()

    res = {}
    # Per-request sequence (production eviction: re-plans at each count change).
    gpu, wall = [], []
    for it in range(WARMUP + ITERS):
        reset_all()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        e0.record()
        for key in layers:
            call(fi, attn, cases[key])
        e1.record()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if it >= WARMUP:
            gpu.append(e0.elapsed_time(e1))
            wall.append((t1 - t0) * 1e3)
    res["request_gpu"] = stats(gpu)
    res["request_wall"] = stats(wall)
    log("request", res["request_gpu"], res["request_wall"])
    # Single calls (plan warm, no eviction between calls).
    for c, w in sorted(cases):
        times = []
        cs = cases[(c, w)]
        for it in range(WARMUP + ITERS):
            cs.reset()
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            call(fi, attn, cs)
            e1.record()
            torch.cuda.synchronize()
            if it >= WARMUP:
                times.append(e0.elapsed_time(e1))
        res[f"single_{c}x{w}"] = stats(times)
        log("single", c, w, res[f"single_{c}x{w}"])
    clock = sm_clock()
    out = {
        "batch": {
            "1": {"kernel_ms": res["request_gpu"]["median"], "noise_floor": res["request_gpu"]["noise"],
                  "wall_ms": res["request_wall"]["median"], "wall_noise": res["request_wall"]["noise"]},
            "2048": {"kernel_ms": res["single_2048x640"]["median"],
                     "noise_floor": res["single_2048x640"]["noise"]},
            "128": {"kernel_ms": res["single_128x640"]["median"],
                    "noise_floor": res["single_128x640"]["noise"]},
        },
        "detail": res,
        "sm_clock_mhz": clock,
        "output": {"class": "exact"},
    }
    print(json.dumps(out))


def check():
    bank, sink = common()
    old, old_attn = make_helper(fused=False)
    new, new_attn = make_helper(fused=True)
    failures, report = [], {}
    shapes = [(c, w) for c in (2048, 128) for w in (640, 128, 400, 101)]
    for s, (c, w) in enumerate(shapes):
        ca = Case(c, w, bank, sink, 300 + s)
        cb = Case(c, w, bank, sink, 300 + s)
        diffs = []
        for it in range(WARMUP + ITERS):
            ca.reset()
            cb.reset()
            ra = call(old, old_attn, ca)
            rb = call(new, new_attn, cb)
            torch.cuda.synchronize()
            pa, pb = plan_of(old, ca), plan_of(new, cb)
            d = (ra.float() - rb.float()).abs().max().item()
            ok = (
                torch.equal(ra, rb)
                and torch.equal(pa.indices, pb.indices)
                and torch.equal(pa.packed_mask, pb.packed_mask)
                and torch.equal(cb.idx, cb.i0)
                and torch.equal(cb.q, cb.q0)
                and bool(torch.isfinite(rb.float()).all())
            )
            diffs.append(d)
            if not ok:
                failures.append({"shape": [c, w], "iter": it, "maxdiff": d})
        # Empty rows must be zero.
        empty = (ca.i0 < 0).all(-1)
        zero_ok = bool((rb.view(c, HEADS, DIM)[empty] == 0).all())
        if not zero_ok:
            failures.append({"shape": [c, w], "empty_rows_nonzero": True})
        report[f"{c}x{w}"] = {"iters": len(diffs), "max_abs_diff_per_iter_max": max(diffs),
                              "empty_rows": int(empty.sum())}
        log(c, w, report[f"{c}x{w}"], "failures", len(failures))
    # Direct run_shared API: old signature and out=.
    ca = Case(128, 640, bank, sink, 900)
    ca.reset()
    r_old_sig = new.run_shared(ca.q, bank, ca.idx, sink, SCALE)
    r_out = torch.empty_like(ca.q)
    new.run_shared(ca.q, bank, ca.idx, sink, SCALE, out=r_out)
    r_ref = old.run_shared(ca.q, bank, ca.idx, sink, SCALE)
    api_ok = torch.equal(r_old_sig, r_ref) and torch.equal(r_out, r_ref)
    if not api_ok:
        failures.append({"api": "run_shared"})
    # CUDA graph capture/replay at a warmed shape, inputs changing between replays.
    cg = Case(128, 640, bank, sink, 901)
    cg.reset()
    gout = torch.empty_like(cg.q)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            new.run_shared(cg.q, bank, cg.idx, sink, SCALE, out=gout)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        new.run_shared(cg.q, bank, cg.idx, sink, SCALE, out=gout)
    graph_ok = True
    for r in range(5):
        gen = torch.Generator().manual_seed(1000 + r)
        cg.q.copy_((torch.randn(128, HEADS, DIM, generator=gen) * 2).to(torch.bfloat16))
        cg.idx.copy_(make_indices(128, 640, gen))
        graph.replay()
        torch.cuda.synchronize()
        eager_new = new.run_shared(cg.q.clone(), bank, cg.idx.clone(), sink, SCALE)
        eager_old = old.run_shared(cg.q.clone(), bank, cg.idx.clone(), sink, SCALE)
        torch.cuda.synchronize()
        ok = torch.equal(gout, eager_new) and torch.equal(gout, eager_old)
        graph_ok &= ok
        log("graph replay", r, ok)
    if not graph_ok:
        failures.append({"graph": "replay != eager"})
    print(json.dumps({"passed": not failures, "compared": "fused run_shared vs ATen run_shared "
                      "(fused_glue=False, unmodified code path): out, plan.indices, packed_mask "
                      "torch.equal every iteration (3 warmup + 30)", "cases": report,
                      "api_ok": api_ok, "graph_ok": graph_ok, "failures": failures[:20]}))


if __name__ == "__main__":
    {"bench": bench, "check": check}[sys.argv[1]]()
