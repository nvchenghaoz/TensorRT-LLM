"""Paired microbenchmark for idea 212 S1: mhc_fused_hc (pmap/GEMM kernel + bigfuse) at prefill M, cold L2.

For each M the AutoTuner (cache cleared) picks the tactic exactly as production does (V4.1-lagged call:
norm_weight + pre_mix_ext), then that tactic's full op is timed: REPS x ITERS calls, 256 MB L2 flush
before each, CUDA events around the op only. Same script runs in both arms; the base arm has no
mtile_v2 tactics. Also prints the best timed tactic of every one offered (diagnostic).
Last stdout line: {"batch": {M: {"kernel_ms", "noise_floor", "tactic"}}, "output": {"class": "exact"}}.
"""
import json
import os
import sys

import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.modules.mhc import mhc_cuda

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mhc_fhc_bench import make_inputs  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from guard import dirty_guard  # noqa: E402

dirty_guard()


def lib_receipt():
    """Pitfall 3271 receipt: the libtensorrt_llm.so actually loaded (path, mtime, kernel symbol counts)."""
    import mmap
    import time as _t
    rec = {}
    for ln in open("/proc/self/maps"):
        if "libtensorrt_llm.so" in ln:
            p = ln.split()[-1]
            if p in rec:
                continue
            with open(p, "rb") as fh:
                mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)

                def cnt(pat):
                    n, i = 0, mm.find(pat)
                    while i != -1:
                        n += 1
                        i = mm.find(pat, i + 1)
                    return n
                rec[p] = {"mtime": _t.ctime(os.path.getmtime(p)),
                          "mtile_v3_all": cnt(b"fused_pmap_gemm_fma_mtile_v3"),
                          "mtile_v3_TM8": cnt(b"fused_pmap_gemm_fma_mtile_v3ILi8ELi8ELi1E"),
                          "mtile_v3_TM11": cnt(b"fused_pmap_gemm_fma_mtile_v3ILi11ELi8ELi1E")}
                mm.close()
    return rec


LIB = lib_receipt()
print("[lib receipt]", json.dumps(LIB), flush=True)

HC, HID = 4, 5120
ARGS = (HC, HID, 1e-6, 1e-6, 1e-6, 2.0, 20)
MS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "128,512,1024,2048,4096,8192").split(",")]
ITERS = int(os.environ.get("ITERS", "20"))
REPS = int(os.environ.get("REPS", "5"))
FIX_MS = [int(x) for x in os.environ.get("FIX_MS", "2048").split(",")]

flush = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")


def time_fn(fn, iters):
    st = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        flush.zero_()
        st[i].record()
        fn()
        en[i].record()
    torch.cuda.synchronize()
    return sorted(s.elapsed_time(e) for s, e in zip(st, en))


out = {}
for M in MS:
    inputs, nw = make_inputs(M)
    pre_ext = torch.rand((M, HC), device="cuda").float() + 0.5
    kw = dict(norm_weight=nw, norm_eps=1e-6, pre_mix_ext=pre_ext)
    AutoTuner.get().clear_cache()
    with autotune(skip_dynamic_tuning_buckets=True):
        mhc_cuda.mhc_fused_hc(*inputs, *ARGS, **kw)
    torch.cuda.synchronize()
    tac = tuple(mhc_cuda.mhc_fused_hc_tactic(*inputs, *ARGS, **kw))
    fn = lambda: mhc_cuda.mhc_fused_hc(*inputs, *ARGS, **kw)  # noqa: E731  (production entry, tuned tactic)
    for _ in range(5):
        fn()
    meds, noises = [], []
    for _ in range(REPS):
        ts = time_fn(fn, ITERS)
        med = ts[len(ts) // 2]
        meds.append(med)
        noises.append((ts[int(0.9 * (len(ts) - 1))] - ts[int(0.1 * (len(ts) - 1))]) / med)
    meds.sort()
    med = meds[len(meds) // 2]
    out[str(M)] = {"kernel_ms": med, "noise_floor": max(noises), "tactic": list(tac),
                   "rep_medians_us": [round(m * 1000, 1) for m in meds]}
    print(f"M={M}: tactic {tac} {med*1000:.1f} us noise {max(noises):.3f} reps {out[str(M)]['rep_medians_us']}",
          flush=True)
    if M in FIX_MS:
        # Diagnostic (area lead 13:3x): fixed-tactic full-op time of v1 <24,3>, v1 <24,4> and, when the
        # tree offers it, the mtile_v2 id (tile_m = 16 + 4), same cold-L2 timing, V4.1-lagged call.
        rn = mhc_cuda.MhcFusedHcRunner(*ARGS)
        offered = set(tuple(t) for t in rn.get_valid_tactics(inputs, None))
        fixed = {}
        for t in [("fused_half_fma", 24, 1, 256, 3), ("fused_half_fma", 24, 1, 256, 4),
                  ("fused_half_fma", 24, 1, 256, 20), ("fused_half_fma", 24, 1, 256, 72)]
                  + [("fused_half_fma", 24, 1, 256, 64 + tm) for tm in range(9, 17)]:
            if t not in offered:
                continue
            ftn = lambda t=t: rn.forward(inputs, tactic=list(t), **kw)  # noqa: E731
            for _ in range(5):
                ftn()
            fm = sorted(time_fn(ftn, ITERS)[ITERS // 2] for _ in range(REPS))
            fixed[str(t)] = round(fm[len(fm) // 2] * 1000, 1)
        out[str(M)]["fixed_tactic_us"] = fixed
        print(f"M={M}: fixed tactics {fixed}", flush=True)
    del inputs
    torch.cuda.empty_cache()
print(json.dumps({"batch": out, "output": {"class": "exact"}, "lib_receipt": LIB}))
