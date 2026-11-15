"""Shared helpers for the down_proj wave-split microbenchmarks (idea 64)."""
import json
import os
import statistics
import sys
import time

import torch

CKPT = os.environ.get(
    "CKPT",
    "/home/scratch.chengzhang/trtllm-dev-gemma4-31b-pf-20261114-163048/models/Gemma-4-31B-IT-NVFP4"
)


def log(*a):
    print(*a, flush=True)


def read_dims():
    cfg = json.load(open(os.path.join(CKPT, "config.json")))
    tc = cfg.get("text_config", cfg)
    hidden = tc["hidden_size"]
    inter = tc["intermediate_size"]
    nl = tc["num_hidden_layers"]
    q = {}
    try:
        q = json.load(open(os.path.join(CKPT, "hf_quant_config.json")))
    except Exception:
        pass
    log(f"config: hidden={hidden} intermediate={inter} layers={nl} "
        f"quant={json.dumps(q.get('quantization', q))[:200]}")
    # down_proj: N = hidden, K = intermediate
    return hidden, inter, nl


def receipts(tag=""):
    out = {}
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        out = dict(
            sm_clock=pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
            mem_clock=pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM),
            temp=pynvml.nvmlDeviceGetTemperature(h,
                                                 pynvml.NVML_TEMPERATURE_GPU),
            power_w=pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0,
            throttle=hex(
                pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(h)),
        )
    except Exception as e:  # noqa
        out = dict(err=str(e)[:100])
    log(f"receipts{tag}: {out}")
    return out


def make_weight(n, k, seed, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    w = torch.randn(n, k, generator=g, device=device,
                    dtype=torch.float32).to(torch.bfloat16) * 0.02
    amax = w.abs().amax().float()
    gs = (448.0 * 6.0 / amax).reshape(1).to(torch.float32)
    w_fp4, w_sf = torch.ops.trtllm.fp4_quantize(w, gs, 16, False)
    del w
    return w_fp4, w_sf, gs


def make_act(m, k, seed, kind="randn", device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(m, k, generator=g, device=device, dtype=torch.float32)
    if kind == "heavy":
        # per-row heavy tail: randn * 2^U(-8, 8)
        e = (torch.rand(m, 1, generator=g, device=device) * 16 - 8)
        x = x * torch.exp2(e)
    x = x.to(torch.bfloat16)
    amax = x.abs().amax().float()
    gs = (448.0 * 6.0 / amax).reshape(1).to(torch.float32)
    a_fp4, a_sf = torch.ops.trtllm.fp4_quantize(x, gs, 16, False)
    del x
    return a_fp4, a_sf, gs


_FI = None


def fi_module():
    global _FI
    if _FI is None:
        from flashinfer.jit.gemm.core import gen_gemm_sm120_module_cutlass_fp4
        t0 = time.time()
        _FI = gen_gemm_sm120_module_cutlass_fp4().build_and_load()
        log(f"FI sm120 fp4 module loaded in {time.time()-t0:.1f}s; "
            f"tactics={_FI.fp4_gemm_tactic_num()}")
    return _FI


def fi_gemm(a_fp4, w_fp4, a_sf, w_sf, alpha, out, ws, tactic):
    fi_module().fp4_gemm(a_fp4, w_fp4, a_sf, w_sf, alpha, out, ws, tactic)
    return out


def int16_diff(a, b):
    ai = a.view(torch.int16)
    bi = b.view(torch.int16)
    ne = (ai != bi)
    n = int(ne.sum().item())
    first = None
    if n:
        idx = ne.flatten().nonzero()[0].item()
        r, c = divmod(idx, a.shape[1])
        first = (r, c, float(a[r, c]), float(b[r, c]))
    return n, first


def time_fn(fn, iters=20, warmup=5, pre=None):
    ts = []
    for i in range(warmup + iters):
        if pre is not None:
            pre(i)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn(i)
        e.record()
        e.synchronize()
        if i >= warmup:
            ts.append(s.elapsed_time(e) * 1000.0)
    ts.sort()
    med = statistics.median(ts)
    p10 = ts[int(0.1 * (len(ts) - 1))]
    p90 = ts[int(0.9 * (len(ts) - 1))]
    return dict(med=med, p10=p10, p90=p90, noise=(p90 - p10) / med)


def kernel_names(fn):
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = []
    for ev in prof.events():
        if ev.device_type is not None and "CUDA" in str(ev.device_type):
            names.append(ev.name[:140])
    return names
