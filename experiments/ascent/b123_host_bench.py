"""Idea 228: host-side (CPU enqueue) cost of one CUTLASS fused_moe call, EP4 rank (96 x 2304) vs EP2xTP2 rank
(192 x 1152), prefill M=2048 (and 1024/4096), synthetic weights, 1 GPU.

Host cost is timed while the GPU is held busy by torch.cuda._sleep, so every launch is asynchronous and the
host loop's wall time is pure enqueue + host-side setup (python op dispatch, runner setup, tactic lookup).
Per layout and M: N_CALLS calls per trial, TRIALS trials; prints median host us per call and (p90-p10)/median.
Also the GPU time per call (CUDA events, no sleep) for reference.
"""
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "b100_moe_prefill"))
os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")

import torch  # noqa: E402

import bench as b100  # noqa: E402
from tensorrt_llm._torch.autotuner import AutoTuner, autotune  # noqa: E402

N_CALLS = 16
TRIALS = 15


def host_time(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(TRIALS):
        torch.cuda._sleep(int(1.5e8))  # ~60+ ms of GPU busy: all launches below are queued, never waited on
        t0 = time.perf_counter()
        for _ in range(N_CALLS):
            fn()
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        out.append(1e6 * (t1 - t0) / N_CALLS)
    out.sort()
    med = statistics.median(out)
    return med, (out[int(0.9 * (len(out) - 1))] - out[int(0.1 * (len(out) - 1))]) / med


def gpu_time(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(10):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(1e3 * a.elapsed_time(b))
    return statistics.median(ts)


def main():
    ckpt = sys.argv[1]
    ms = [int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "1024,2048,4096").split(",")]
    tune = os.environ.get("B123_TUNE", "1") == "1"
    hidden, inter, n_exp, topk, n_layers = b100.read_cfg(ckpt)
    print("cfg", hidden, inter, n_exp, topk, n_layers, "tune", tune, torch.cuda.get_device_name(), flush=True)
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import MoERunner
    res = {}
    for name in ("ep4", "ep2tp2", "ep4"):
        tp, ep = b100.LAYOUTS[name]
        MoERunner.runner_dict.clear()
        AutoTuner.get().profiling_cache.cache.clear()
        torch.cuda.empty_cache()
        r = b100.Rank(hidden, inter, n_exp, topk, tp, ep)
        if tune:
            x, sel, sc = r.inputs(max(ms))
            with autotune():
                r.run(x, sel, sc)
            torch.cuda.synchronize()
        for m in ms:
            x, sel, sc = r.inputs(m)
            fn = lambda: r.run(x, sel, sc)  # noqa: E731
            h, hn = host_time(fn)
            g = gpu_time(fn)
            key = f"{name}_M{m}"
            if key in res:
                key += "_again"
            res[key] = {"host_us": round(h, 1), "host_noise": round(hn, 3), "gpu_us": round(g, 1)}
            print(f"LAYOUT {name} M={m}: host {h:.1f} us/call (noise {hn:.3f}), gpu {g:.1f} us/call", flush=True)
        del r
    print(json.dumps({"passed": True, "results": res}))


if __name__ == "__main__":
    main()
