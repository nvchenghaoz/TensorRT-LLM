"""Step-time pair bench: 40 CSA2 layers through c4's alternating eager steps.

Cases (one step = the 40 layers' evict + run_shared, wall time, synced):
  c4_3ctx / c4_1ctx: alternating 3-ctx and 1-ctx steps (base re-plans every
                     layer every step; candidate hits after warmup).
  c1_1ctx:           repeated 1-ctx steps only (both arms hit; must stay flat).
Also counts plan() calls (wrapper constructions) and memory per arm.
"""

import json
import statistics
import sys
import time

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import common as C  # noqa: E402

WARM = 4
TIMED = 16  # alternating -> 8 samples per case
STEADY_WARM = 2
STEADY = 8


def spread(xs):
    xs = sorted(xs)
    n = len(xs)
    p10 = xs[max(int(0.1 * (n - 1)), 0)]
    p90 = xs[min(int(round(0.9 * (n - 1))), n - 1)]
    med = statistics.median(xs)
    return med, (p90 - p10) / med


def main():
    torch.manual_seed(0)
    heads, _ = C.heads_per_rank()
    C.log(f"heads/rank={heads} layers={C.LAYERS} bands={C.BANDS}")
    inp = C.Inputs(heads)
    layers = C.make_layers()
    outs = {}
    for c3, c1, w in C.BANDS:
        for c in (c3, c1):
            outs[c] = torch.empty(c, heads, C.DIM, dtype=torch.bfloat16, device="cuda")

    from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper

    plans = {"n": 0}
    orig_plan = BatchPrefillWithPagedKVCacheWrapper.plan

    def counting_plan(self, *a, **k):
        plans["n"] += 1
        return orig_plan(self, *a, **k)

    BatchPrefillWithPagedKVCacheWrapper.plan = counting_plan

    def step(three):
        for li, inst in enumerate(layers):
            c, w = C.step_counts(li, three)
            C.layer_call(inst, inp, c, w, out=outs[c])

    def timed_step(three):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        step(three)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1e3
        mem["alloc"] = max(mem["alloc"], torch.cuda.memory_allocated())
        mem["pinned"] = max(mem["pinned"], C.pinned_bytes(layers))
        return ms

    mem = {"alloc": 0, "pinned": 0}

    for i in range(WARM):
        step(i % 2 == 0)
        torch.cuda.synchronize()
        C.log(f"warm {i} plans so far {plans['n']}")
    torch.cuda.synchronize()
    p0 = plans["n"]
    t3, t1 = [], []
    for i in range(TIMED):
        three = i % 2 == 0
        ms = timed_step(three)
        (t3 if three else t1).append(ms)
        C.log(f"step {i} {'3ctx' if three else '1ctx'} {ms:.2f} ms")
    alt_plans = plans["n"] - p0
    for i in range(STEADY_WARM):
        step(False)
    p1 = plans["n"]
    ts = []
    for i in range(STEADY):
        ms = timed_step(False)
        ts.append(ms)
        C.log(f"steady {i} 1ctx {ms:.2f} ms")
    steady_plans = plans["n"] - p1
    m3, n3 = spread(t3)
    m1, n1 = spread(t1)
    ms_, ns = spread(ts)
    res = {
        "batch": {
            "c4_3ctx": {"step_ms": m3, "noise_floor": n3},
            "c4_1ctx": {"step_ms": m1, "noise_floor": n1},
            "c1_1ctx": {"step_ms": ms_, "noise_floor": ns},
        },
        "output": {"class": "exact"},
        "plan_calls_alternating_timed": alt_plans,
        "plan_calls_steady_timed": steady_plans,
        "device_allocated_after_step_max_bytes": mem["alloc"],
        "device_reserved_bytes": torch.cuda.memory_reserved(),
        "device_max_allocated_bytes": torch.cuda.max_memory_allocated(),
        "pinned_int_workspace_max_bytes": mem["pinned"],
        "device_int_workspace_bytes": C.int_ws_bytes(layers),
        "plans_cached": C.plan_counts(layers),
        "heads_per_rank": heads,
        "raw_ms": {"c4_3ctx": t3, "c4_1ctx": t1, "c1_1ctx": ts},
    }
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
