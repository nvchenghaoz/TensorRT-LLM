import json, os, statistics, sys, time
import torch
from tensorrt_llm._torch.modules.linear import FP8QDQLinearMethod
from tensorrt_llm._torch.autotuner import autotune, AutoTuner
try:
    from tensorrt_llm._torch.modules import fp8_prefill_nvfp4 as fpn
except ImportError:
    fpn = None
torch.manual_seed(0)
dev = "cuda"
N, K = 16384, 5120
W = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
ws = (W.float().abs().amax() / 448.0).reshape(())
wq = (W.float() / ws).to(torch.float8_e4m3fn)
class M(torch.nn.Module):
    pass
m = M()
m.quant_method = FP8QDQLinearMethod()
m.weight = torch.nn.Parameter(wq, requires_grad=False)
m.weight_scale = torch.nn.Parameter(ws.float().clone(), requires_grad=False)
m.force_dynamic_quantization = False
m.enable_cuda_core = True
m.all_reduce = None
m.mapping = None
m.dtype = torch.bfloat16
built = False
xs = {}
for Mr in (2048, 8192):
    x = (torch.randn(Mr, K, device=dev) * 1.5).to(torch.bfloat16)
    xs[Mr] = x
amax = max(float(x.float().abs().amax()) for x in xs.values())
m.input_scale = torch.nn.Parameter(torch.tensor(1.1 * amax / 448.0, device=dev), requires_grad=False)
if fpn is not None:
    built = fpn.maybe_build_prefill_nvfp4_copy(m)
arm = "candidate" if built else "baseline"
print(f"arm={arm} nvfp4_copy_built={built}", flush=True)
Wdq = (wq.float() * ws.float())
apply = lambda x: m.quant_method.apply(m, x, None)
res = {}
errs = {}
WARM, ITERS = 10, 50
for Mr, x in xs.items():
    with autotune():
        for _ in range(3):
            apply(x)
    torch.cuda.synchronize()
    # untimed repeated-execution check (same counts as timing)
    ref1 = None
    nbad = 0
    for i in range(WARM + ITERS):
        y = apply(x)
        if ref1 is None:
            ref1 = y.clone()
        elif not torch.equal(y, ref1):
            nbad += 1
    torch.cuda.synchronize()
    # error vs FP32 reference on the FP8-dequantized weight and BF16 input
    yref = x.float() @ Wdq.t()
    d = (ref1.float() - yref)
    errs[Mr] = {"rel_rms": float(d.norm() / yref.norm()), "max_abs": float(d.abs().max()),
                "mean_abs": float(d.abs().mean()), "finite": bool(torch.isfinite(ref1).all()),
                "repeat_mismatches": nbad}
    if built:
        # also vs the original FP8 op (static quant + cublas_scaled_mm)
        q, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(x, m.input_scale)
        y8 = torch.ops.trtllm.cublas_scaled_mm(q, m.weight.t(), scale_a=m.input_scale,
                                               scale_b=m.weight_scale, bias=None, out_dtype=torch.bfloat16)
        d8 = ref1.float() - y8.float()
        y8e = y8.float() - yref
        errs[Mr]["vs_fp8_op_rel_rms"] = float(d8.norm() / y8.float().norm())
        errs[Mr]["fp8_op_rel_rms_vs_ref"] = float(y8e.norm() / yref.norm())
    del yref
    for _ in range(WARM):
        apply(x)
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(ITERS)]
    for e0, e1 in ev:
        e0.record(); apply(x); e1.record()
    torch.cuda.synchronize()
    ts = sorted(e0.elapsed_time(e1) for e0, e1 in ev)
    med = statistics.median(ts)
    p10, p90 = ts[int(0.1 * len(ts))], ts[int(0.9 * len(ts)) - 1]
    tflops = 2.0 * Mr * N * K / (med * 1e-3) / 1e12
    res[str(Mr)] = {"kernel_ms": med, "p90_ms": p90, "noise_floor": (p90 - p10) / med, "tflops": tflops}
    print(f"[{arm}] M={Mr} median {med*1e3:.1f} us p90 {p90*1e3:.1f} us {tflops:.1f} TFLOPS err {errs[Mr]}", flush=True)
picks = []
try:
    cache = AutoTuner.get().profiling_cache.cache
    for k, v in cache.items():
        if "nvfp4" in str(k):
            picks.append(f"{str(k)[:160]} -> {str(v)[:120]}")
except Exception as e:
    picks.append(repr(e))
for p in picks:
    print("autotuner pick:", p, flush=True)
ok = all(e["repeat_mismatches"] == 0 and e["finite"] for e in errs.values())
print(json.dumps({"batch": res, "output": {"class": "precision" if built else "exact", "errors": errs,
                  "repeat_bitwise_identical": ok, "autotuner_picks": picks[:6]}, "arm": arm, "passed": ok}))
