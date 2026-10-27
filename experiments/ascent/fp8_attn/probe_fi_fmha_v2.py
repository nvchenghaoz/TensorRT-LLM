"""Stage-2 probe: FlashInfer fmha_v2_prefill_sm120 (fmha_v2 e4m3 64x64 tiled, separate
Q/K/V, BF16 out) vs VANILLA (torch SDPA) at FLUX.3 klein attention shapes.

B=1, H=24, d=128, S in {4096, 4608, 512}. FP8 op time includes per-tensor amax+quantize
of Q, K, V and the scale product. Errors vs an fp32 reference.
"""
import json
import math
import sys
import time

import torch
import torch.nn.functional as F

print("importing", flush=True)
import tensorrt_llm  # noqa: F401,E402  (registers torch.ops.trtllm)
from flashinfer.prefill import fmha_v2_prefill_sm120  # noqa: E402

H, D = 24, 128
SM_SCALE = 1.0 / math.sqrt(D)
SHAPES = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["4096", "4608", "512"])]
REPS = 100
print("device", torch.cuda.get_device_name(), torch.cuda.get_device_capability(), flush=True)


def make(S, qk_outlier=False):
    g = torch.Generator(device="cuda").manual_seed(S)
    q = torch.randn(1, S, H, D, device="cuda", dtype=torch.float32, generator=g)
    k = torch.randn(1, S, H, D, device="cuda", dtype=torch.float32, generator=g)
    v = torch.randn(1, S, H, D, device="cuda", dtype=torch.float32, generator=g)
    if qk_outlier:
        # A peakier softmax: per-channel gains as after learned QK-norm weights.
        gain = torch.rand(D, device="cuda", generator=g) * 3 + 0.25
        q, k = q * gain, k * gain
    return q.bfloat16(), k.bfloat16(), v.bfloat16()


def vanilla(q, k, v):
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=SM_SCALE
    ).transpose(1, 2)


def ref32(q, k, v):
    return F.scaled_dot_product_attention(
        q.float().transpose(1, 2), k.float().transpose(1, 2), v.float().transpose(1, 2),
        scale=SM_SCALE,
    ).transpose(1, 2)


def quant(x):
    xq, ds = torch.ops.trtllm.quantize_e4m3_per_tensor(x)
    return xq, ds.float().reshape(1)


def fp8_attn(q, k, v):
    S = q.shape[1]
    qq, dq = quant(q)
    kq, dk = quant(k)
    vq, dv = quant(v)
    s1 = dq * dk * SM_SCALE
    out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    fmha_v2_prefill_sm120(qq, kq, vq, out, H, D, S, scale_softmax=1.0, scale_bmm1=1.0,
                          scale_bmm2=1.0, scale_bmm1_d=s1, scale_bmm2_d=dv, causal=False)
    return out


_pre = {}


def fp8_kernel_only(q, k, v):
    key = q.data_ptr()
    if key not in _pre:
        qq, dq = quant(q)
        kq, dk = quant(k)
        vq, dv = quant(v)
        _pre[key] = (qq, kq, vq, (dq * dk * SM_SCALE).contiguous(), dv.contiguous(),
                     torch.empty(q.shape, dtype=torch.bfloat16, device=q.device))
    qq, kq, vq, s1, dv, out = _pre[key]
    S = q.shape[1]
    fmha_v2_prefill_sm120(qq, kq, vq, out, H, D, S, scale_softmax=1.0, scale_bmm1=1.0,
                          scale_bmm2=1.0, scale_bmm1_d=s1, scale_bmm2_d=dv, causal=False)
    return out


def err(a, r):
    d = (a.float() - r).abs()
    return {"max_abs": d.max().item(), "rel_l2": (d.norm() / r.norm()).item()}


def bench(fns, args):
    for f in fns.values():
        for _ in range(5):
            f(*args)
    torch.cuda.synchronize()
    times = {n: [] for n in fns}
    for _ in range(REPS):
        for n, f in fns.items():
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            f(*args)
            e.record()
            e.synchronize()
            times[n].append(s.elapsed_time(e))
    out = {}
    for n, t in times.items():
        t = sorted(t)
        med = t[len(t) // 2]
        out[n] = {"median_ms": round(med, 5),
                  "noise": round((t[int(0.9 * len(t))] - t[int(0.1 * len(t))]) / med, 4)}
    return out


results = {}
t0 = time.time()
q, k, v = make(128)
fp8_attn(q, k, v)
torch.cuda.synchronize()
print(f"JIT/load done in {time.time() - t0:.1f}s", flush=True)
for S in SHAPES:
    res = {}
    for outlier in (False, True):
        q, k, v = make(S, outlier)
        r = ref32(q, k, v)
        tag = "peaky" if outlier else "randn"
        res[f"{tag}_vanilla_err"] = err(vanilla(q, k, v), r)
        o1 = fp8_attn(q, k, v)
        res[f"{tag}_fp8_err"] = err(o1, r)
        res[f"{tag}_fp8_repeat_bitwise"] = all(torch.equal(o1, fp8_attn(q, k, v)) for _ in range(5))
    q, k, v = make(S)
    res["timing"] = bench({"vanilla": vanilla, "fp8_total": fp8_attn, "fp8_kernel": fp8_kernel_only},
                          (q, k, v))
    print(S, json.dumps(res), flush=True)
    results[S] = res
print(json.dumps({"results": results}))
