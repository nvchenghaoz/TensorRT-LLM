"""Branch 15 (idea 24) paired microbenchmark: one NVFP4 single block's head + gated residual
chain at the served shapes (i2i S=32848 K=3, t2i S=16464 K=2), as THIS tree routes it.

Baseline tree: the warp head refuses a u8 index, so i2i runs main's per-token route
(Inductor LN+mod over materialized [1,S,D] shift/scale, CuTe-DSL NVFP4 quantize,
per-token-gate residual). Candidate: [1,3,D] tables + u8 index -> warp head + table-gate
residual. t2i (bool K=2) is the same route in both trees; its output sha256 is printed so the
two arms can be compared bitwise.

Protocol: fixed seed; pristine copy of every input; restore before the timed loop and before
the untimed correctness pass; CUDA-event timing per iteration, median of ITERS after WARMUP;
the untimed pass repeats the same WARMUP + ITERS sequence and compares every iteration's
outputs with iteration 0 (bitwise) and checks inputs unchanged before each reset.
Last stdout line: {"batch": {case: {"kernel_ms", "noise_floor"}}, "output": {...}}.
"""

import hashlib
import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CASES, Chains, log, make_inputs, restore, same_as, snapshot, table_route_ok  # noqa: E402

WARMUP = int(os.environ.get("B15_WARMUP", "10"))
ITERS = int(os.environ.get("B15_ITERS", "60"))


def sha(ts):
    h = hashlib.sha256()
    for t in ts:
        h.update(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def time_fn(fn, n):
    evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
           for _ in range(n)]
    for s, e in evs:
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in evs]


def stats(ts):
    ts = sorted(ts)
    med = statistics.median(ts)
    p10 = ts[int(0.1 * (len(ts) - 1))]
    p90 = ts[int(0.9 * (len(ts) - 1))]
    return med, (p90 - p10) / med


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    result = {"batch": {}, "output": {"class": "reordered"}, "routes": {}, "sub_ms": {},
              "sha256": {}, "repeat_ok": {}, "inputs_unchanged": {}}
    for case in CASES:
        inp = make_inputs(case)
        route = "table" if (inp["k"] == 2 or table_route_ok(inp)) else "pertoken"
        result["routes"][case] = route
        log(case, "M", inp["m"], "K", inp["k"], "route", route)
        ch = Chains(inp)
        snap = snapshot(inp)
        fn = lambda: ch.chain(route)  # noqa: E731
        # compile / build
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        restore(inp, snap)
        for _ in range(WARMUP):
            fn()
        ts = time_fn(fn, ITERS)
        med, noise = stats(ts)
        result["batch"][case] = {"kernel_ms": med, "noise_floor": noise}
        log(case, f"chain median {med:.4f} ms noise {noise:.4f}")
        # diagnostics: sub-parts
        if route == "table":
            hf = lambda: ch.head_table(store_x_mod=False)  # noqa: E731
            rf = ch.resid_table
        else:
            hf = ch.head_pertoken
            rf = ch.resid_pertoken
        for _ in range(WARMUP):
            hf()
        hm, _ = stats(time_fn(hf, ITERS))
        for _ in range(WARMUP):
            rf()
        rm, _ = stats(time_fn(rf, ITERS))
        result["sub_ms"][case] = {"head": hm, "residual": rm}
        log(case, f"head {hm:.4f} ms residual {rm:.4f} ms")
        result["inputs_unchanged"][case] = same_as(inp, snap)
        # untimed matching sequence
        restore(inp, snap)
        ref = None
        ok = True
        for it in range(WARMUP + ITERS):
            outs = fn()
            if ref is None:
                ref = [o.clone() for o in outs]
            else:
                ok &= all(torch.equal(a, b) for a, b in zip(outs, ref))
        torch.cuda.synchronize()
        ok &= same_as(inp, snap)
        result["repeat_ok"][case] = bool(ok)
        data, sf, out = ref
        m = inp["m"]
        valid_sf = sf  # flat 128x4 layout (padded rows included)
        result["sha256"][case] = {"fp4": sha([data]), "sf": sha([valid_sf]), "resid": sha([out])}
        log(case, "repeat_ok", ok, "sha", result["sha256"][case])
        del ch, inp, snap, ref
        torch.cuda.empty_cache()
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
