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

HC, HID = 4, 5120
ARGS = (HC, HID, 1e-6, 1e-6, 1e-6, 2.0, 20)
MS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "128,512,1024,2048,4096,8192").split(",")]
ITERS = int(os.environ.get("ITERS", "20"))
REPS = int(os.environ.get("REPS", "5"))

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
    del inputs
    torch.cuda.empty_cache()
print(json.dumps({"batch": out, "output": {"class": "exact"}}))
