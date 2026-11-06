"""Correctness + host-cost check for idea 372 (FI sparse-prefill direct PreparedCall.execute
and cached index-prep launch), no model load.

Reference = branch 197's route, i.e. the same production dispatch with
TRTLLM_CSA2_FI_DIRECT_EXECUTE=0 (wrapper.run + Triton JIT launch), which is main's code path.
Per shape (128/384/2048/6144 rows x widths 640/128, holes, empty rows, out-of-range slots):
  * 3 warmup + 30 iterations through CSA2FlashInfer._forward_selected, output NaN-filled
    before every call, each bitwise equal to the reference; two eager reference runs equal;
  * steady state under torch.cuda.set_sync_debug_mode('error'); no new FlashInfer plans;
    no new CUDA caching-allocator allocations; every steady call took the direct path;
  * index scratch (cached-handle launch) bitwise equal to the JIT-launched reference.
LRU: one FISharedPrefill with a 2-entry LRU over 128 -> 2048 -> 128 rows, 640 <-> 128
widths (hits and misses), plus a forced calibration-epoch bump; every call bitwise equal.
Host us per call (perf_counter, 500 back-to-back calls, one sync at the end) and GPU us
per call (CUDA graph of the FI call) at 128x640 / 128x128, reference route vs branch.
Prints {"passed": ...} last.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pair import Case  # noqa: E402
from stage0 import Scenario, log, time_graph  # noqa: E402

from tensorrt_llm._torch.attention.backends.sparse.csa2 import fi_sparse_prefill as fsp  # noqa: E402

WARMUP, ITERS, HOST_N = 3, 30, 500


def make_case(s, direct):
    os.environ["TRTLLM_CSA2_FI_DIRECT_EXECUTE"] = "1" if direct else "0"
    c = Case(s)
    c.out.fill_(float("nan"))
    c()
    torch.cuda.synchronize()
    assert c.helper._fi_prefill is not None, "FI route not taken"
    assert c.helper._fi_prefill._direct == direct
    return c


def allocs():
    return torch.cuda.memory_stats().get("allocation.all.allocated", 0)


# Allocation attribution: device allocations inside FISharedPrefill.__call__ and inside the
# index prep alone (the rest of a Case call is the harness's fake attention / backend glue).
ALLOC = {"on": False, "fi_call": 0, "prep": 0}
_orig_call = fsp.FISharedPrefill.__call__
_orig_prep = fsp.prepare_shared_prefill_indices


def _counted_call(self, *a, **k):
    if not ALLOC["on"]:
        return _orig_call(self, *a, **k)
    b = allocs()
    try:
        return _orig_call(self, *a, **k)
    finally:
        ALLOC["fi_call"] += allocs() - b


def _counted_prep(*a, **k):
    if not ALLOC["on"]:
        return _orig_prep(*a, **k)
    b = allocs()
    try:
        return _orig_prep(*a, **k)
    finally:
        ALLOC["prep"] += allocs() - b


fsp.FISharedPrefill.__call__ = _counted_call
fsp.prepare_shared_prefill_indices = _counted_prep


def count_allocs(case, n):
    """Steady device allocations over n calls: whole Case call, FI call, index prep."""
    torch.cuda.synchronize()
    ALLOC.update(fi_call=0, prep=0)
    ALLOC["on"] = True
    total = 0
    for _ in range(n):
        b = allocs()
        case()
        total += allocs() - b
    ALLOC["on"] = False
    torch.cuda.synchronize()
    return {"case_call": total, "fi_call": ALLOC["fi_call"], "index_prep": ALLOC["prep"]}


def mismatch(a, b):
    d = a != b
    n = int(d.sum())
    out = {"count": n, "numel": a.numel()}
    if n:
        idx = d.nonzero()[:6].tolist()
        out["first"] = [[i, int(a[tuple(i)]), int(b[tuple(i)])] for i in idx]
        out["rows"] = int(d.any(-1).sum())
    return out


def index_diag(s, bb, rb, hm):
    """Index scratch: JIT vs cached-handle launch into fresh buffers, and whether the
    FI attention writes the scratch in place (post-call state vs a fresh re-prep)."""
    count = s.count
    res = {}
    mp = s.main_pool if hm else None
    ml = s.main_logical if hm else None
    mw = ml.shape[1] if hm else 1
    fresh = {}
    for direct in (False, True):
        swa = torch.full((count, s.swa_slots.shape[1]), -7, dtype=torch.int32, device=s.q.device)
        main = torch.full((count, mw), -7, dtype=torch.int32, device=s.q.device)
        _orig_prep(s.swa_pool, s.swa_slots, mp, ml, s.mapping, s.plan, swa, main if hm else None,
                   direct=direct)
        fresh[direct] = (swa, main)
    torch.cuda.synchronize()
    res["fresh_jit_vs_cached_launch_swa"] = mismatch(fresh[False][0], fresh[True][0])
    if hm:
        res["fresh_jit_vs_cached_launch_main"] = mismatch(fresh[False][1], fresh[True][1])
    res["fresh_jit_vs_branch_scratch_after_calls_swa"] = mismatch(fresh[False][0], bb["swa"])
    res["fresh_jit_vs_base_scratch_after_calls_swa"] = mismatch(fresh[False][0], rb["swa"])
    if hm:
        res["fresh_jit_vs_branch_scratch_after_calls_main"] = mismatch(fresh[False][1], bb["main"])
        res["fresh_jit_vs_base_scratch_after_calls_main"] = mismatch(fresh[False][1], rb["main"])
    return res


def plans(fi):
    return {str(k): len(w._prepared_calls) for k, w in fi._wrappers.items()}


def host_us(fn, n=HOST_N):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t = time.perf_counter()
    s.record()
    for _ in range(n):
        fn()
    e.record()
    host = (time.perf_counter() - t) / n * 1e6
    torch.cuda.synchronize()
    return host, s.elapsed_time(e) * 1e3 / n


def sm_clock():
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"nvidia-smi failed: {e}"


def main():
    report = {"cases": {}, "lru": {}, "host": {}}
    ok = True
    shapes = [
        (1, 128, True), (1, 128, False), (1, 2048, True), (1, 2048, False),
        (3, 128, True), (3, 128, False), (3, 2048, True), (3, 2048, False),
    ]
    scen, refs = {}, {}
    for i, (r, t, hm) in enumerate(shapes):
        s = Scenario(r, t, hm, seed=3000 + i)
        base = make_case(s, direct=False)
        ref = base.out.clone()
        base.out.fill_(float("nan"))
        base()
        torch.cuda.synchronize()
        eager_repeat = bool(torch.equal(ref, base.out))
        br = make_case(s, direct=True)
        fi = br.helper._fi_prefill
        first_equal = bool(torch.equal(br.out, ref))
        per_iter, steady_direct = [], True
        p0 = d0 = None
        new_allocs = 0
        for it in range(WARMUP + ITERS):
            br.out.fill_(float("nan"))
            if it == WARMUP:
                torch.cuda.synchronize()
                p0, d0 = plans(fi), fi.direct_calls
            # Allocations counted around the production call only (not the harness's
            # fill_/torch.equal).
            a_before = allocs()
            if it >= WARMUP:
                torch.cuda.set_sync_debug_mode("error")
            res = br()
            if it >= WARMUP:
                torch.cuda.set_sync_debug_mode("default")
                new_allocs += allocs() - a_before
            torch.cuda.synchronize()
            per_iter.append(bool(torch.equal(br.out, ref)) and res is br.out)
        p1, d1 = plans(fi), fi.direct_calls
        a0, a1 = 0, new_allocs
        steady_direct = (d1 - d0) == ITERS
        bkey = next(iter(br.helper._fi_prefill._buffers))
        bb, rb = br.helper._fi_prefill._buffers[bkey], base.helper._fi_prefill._buffers[bkey]
        # The main scratch is written only on dual (SWA+main) tiles.
        idiag = index_diag(s, bb, rb, hm)
        # The cached-handle launch must write exactly what the JIT launch writes.
        index_equal = idiag["fresh_jit_vs_cached_launch_swa"]["count"] == 0 and (
            not hm or idiag["fresh_jit_vs_cached_launch_main"]["count"] == 0
        )
        empty_zero = bool((br.out.view_as(s.q)[s.empty] == 0).all())
        alloc_branch = count_allocs(br, ITERS)
        alloc_base = count_allocs(base, ITERS)
        # No new allocations: the branch's FI call allocates nothing in steady state, and the
        # whole production call allocates no more than the wrapper.run route.
        alloc_ok = alloc_branch["fi_call"] == 0 and alloc_branch["case_call"] <= alloc_base["case_call"]
        case_ok = (
            all(per_iter) and first_equal and eager_repeat and p0 == p1 and alloc_ok
            and steady_direct and index_equal and empty_zero
            and base.helper._fi_prefill.direct_calls == 0
        )
        ok &= case_ok
        report["cases"][s.name] = {
            "iterations": WARMUP + ITERS,
            "all_iters_bitwise_vs_wrapper_route": all(per_iter),
            "bad_iters": [i for i, v in enumerate(per_iter) if not v],
            "first_call_bitwise": first_equal,
            "reference_eager_repeat_bitwise": eager_repeat,
            "steady_plans_before_after": [p0, p1],
            "steady_new_allocations_whole_case_call_in_loop": a1 - a0,
            "steady_allocations_30_calls_branch": alloc_branch,
            "steady_allocations_30_calls_wrapper_route": alloc_base,
            "allocations_ok": alloc_ok,
            "steady_calls_direct": d1 - d0,
            "index_scratch_bitwise_vs_jit_launch": index_equal,
            "index_diag": idiag,
            "empty_rows_zero": empty_zero,
            "kill_switch_direct_calls": base.helper._fi_prefill.direct_calls,
            "kill_switch_wrapper_calls": base.helper._fi_prefill.wrapper_calls,
            "max_abs_vs_ref": float((br.out.float() - ref.float()).abs().nan_to_num(1e9).max()),
            "passed": case_ok,
        }
        log(s.name, json.dumps(report["cases"][s.name]))
        if r == 1:
            scen[s.name] = s
            refs[s.name] = ref.view_as(s.q)
            if t == 128:
                hb, gb = host_us(base)
                hd, gd = host_us(br)
                g_ref = time_graph(lambda: s.run_fi(base.helper._fi_prefill))
                g_dir = time_graph(lambda: s.run_fi(br.helper._fi_prefill))
                report["host"][s.name] = {
                    "wrapper_route_host_us_per_call": hb,
                    "direct_host_us_per_call": hd,
                    "wrapper_route_event_us_per_call_500_back_to_back": gb,
                    "direct_event_us_per_call_500_back_to_back": gd,
                    "wrapper_route_gpu_us_per_call_graph": g_ref.get("median_ms", float("nan")) * 1e3,
                    "direct_gpu_us_per_call_graph": g_dir.get("median_ms", float("nan")) * 1e3,
                    "host_saved_us_per_call": hb - hd,
                }
                log("host", s.name, json.dumps(report["host"][s.name]))
        del base, br
        torch.cuda.empty_cache()
    # LRU hit/miss alternation with a 2-entry LRU, then a calibration-epoch bump.
    os.environ["TRTLLM_CSA2_FI_DIRECT_EXECUTE"] = "1"
    fi = fsp.FISharedPrefill()
    fi._MAX_SHAPES = 2
    seq = ["128x640", "2048x640", "128x640", "128x640", "128x128", "2048x128", "128x128", "128x128",
           "128x640", "2048x640", "2048x640", "128x640", "128x128"]
    trace, lru_ok = [], True
    for nm in seq:
        s = scen[nm]
        s.out_fi.fill_(float("nan"))
        d, w = fi.direct_calls, fi.wrapper_calls
        s.run_fi(fi)
        torch.cuda.synchronize()
        eq = bool(torch.equal(s.out_fi, refs[nm]))
        lru_ok &= eq
        trace.append([nm, "hit" if fi.direct_calls > d else "miss", eq])
    from flashinfer.mla._sparse_mla_sm120 import _calibration as cal

    cal._constants_version += 1
    s = scen["128x640"]
    s.out_fi.fill_(float("nan"))
    d = fi.direct_calls
    s.run_fi(fi)
    torch.cuda.synchronize()
    eq = bool(torch.equal(s.out_fi, refs["128x640"]))
    trace.append(["128x640 after epoch bump", "hit" if fi.direct_calls > d else "miss(re-prepare)", eq])
    lru_ok &= eq and fi.direct_calls == d
    s.out_fi.fill_(float("nan"))
    s.run_fi(fi)
    torch.cuda.synchronize()
    eq = bool(torch.equal(s.out_fi, refs["128x640"]))
    trace.append(["128x640 next", "hit" if fi.direct_calls > d else "miss", eq])
    lru_ok &= eq and fi.direct_calls == d + 1
    hits = sum(1 for x in trace if x[1] == "hit")
    misses = len(trace) - hits
    lru_ok &= hits > 0 and misses > 0
    report["lru"] = {"trace": trace, "hits": hits, "misses": misses, "passed": lru_ok}
    log("lru", json.dumps(report["lru"]))
    ok &= lru_ok
    report["sm_clock"] = sm_clock()
    report["passed"] = bool(ok)
    print(json.dumps(report))


if __name__ == "__main__":
    sys.exit(main())
