"""Stage-1 probe: cuDNN sdpa_fp8 on SM120 at FLUX.3 klein attention shapes.

B=1, H=24, d=128; S in {4096 (image self), 4608 (joint, T=512), 512 (text)}.
Compares VANILLA (torch SDPA, HND views of NHD planes) vs cuDNN FP8 (incl. quantize)
and cuDNN BF16. Errors vs an fp32 reference. Interleaved CUDA-event timing.
"""
import json
import math
import sys
import time

import torch
import torch.nn.functional as F

print("importing tensorrt_llm", flush=True)
import tensorrt_llm  # noqa: F401,E402
from tensorrt_llm._torch.visual_gen.attention.backends import cudnn as cudnn_be  # noqa: E402
from tensorrt_llm.visual_gen.args import QuantAttentionConfig  # noqa: E402

print("cudnn backend version", cudnn_be.cudnn.backend_version(), flush=True)
print("device", torch.cuda.get_device_name(), torch.cuda.get_device_capability(), flush=True)

# Lift the SM gate for the probe only.
cudnn_be.CuDNNAttention.check_hardware_compatibility = classmethod(lambda cls, d, q=None: None)

H, D = 24, 128
SHAPES = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["4096", "4608", "512"])]
REPS = 100
torch.manual_seed(0)


def make(S):
    g = torch.Generator(device="cuda").manual_seed(S)
    q = torch.randn(1, S, H, D, device="cuda", dtype=torch.bfloat16, generator=g)
    k = torch.randn(1, S, H, D, device="cuda", dtype=torch.bfloat16, generator=g)
    v = torch.randn(1, S, H, D, device="cuda", dtype=torch.bfloat16, generator=g)
    return q, k, v


def vanilla(q, k, v):
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=1 / math.sqrt(D)
    ).transpose(1, 2)


def ref32(q, k, v):
    return F.scaled_dot_product_attention(
        q.float().transpose(1, 2), k.float().transpose(1, 2), v.float().transpose(1, 2),
        scale=1 / math.sqrt(D),
    ).transpose(1, 2)


fp8_be = cudnn_be.CuDNNAttention(num_heads=H, head_dim=D, dtype=torch.bfloat16,
                                 quant_attention_config=QuantAttentionConfig(qk_dtype="fp8", v_dtype="fp8"))
bf16_be = cudnn_be.CuDNNAttention(num_heads=H, head_dim=D, dtype=torch.bfloat16)


def quant_only(q, k, v):
    return (torch.ops.trtllm.quantize_e4m3_per_tensor(q),
            torch.ops.trtllm.quantize_e4m3_per_tensor(k),
            torch.ops.trtllm.quantize_e4m3_per_tensor(v))


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
        out[n] = {"median_ms": med, "noise": (t[int(0.9 * len(t))] - t[int(0.1 * len(t))]) / med}
    return out


results = {}
for S in SHAPES:
    print(f"== S={S}", flush=True)
    q, k, v = make(S)
    r = ref32(q, k, v)
    res = {}
    o_v = vanilla(q, k, v)
    res["vanilla_err"] = err(o_v, r)
    fns = {"vanilla": vanilla, "quant_only": quant_only}
    try:
        o_b = bf16_be.forward(q, k, v)
        res["cudnn_bf16_err"] = err(o_b, r)
        fns["cudnn_bf16"] = bf16_be.forward
    except Exception as ex:  # noqa: BLE001
        res["cudnn_bf16_error"] = repr(ex)[:500]
        print("cudnn bf16 failed:", repr(ex)[:500], flush=True)
    try:
        t0 = time.time()
        o_f = fp8_be.forward(q, k, v)
        torch.cuda.synchronize()
        print(f"fp8 graph built in {time.time() - t0:.1f}s", flush=True)
        res["cudnn_fp8_err"] = err(o_f, r)
        o_f2 = fp8_be.forward(q, k, v)
        res["cudnn_fp8_repeat_bitwise"] = bool(torch.equal(o_f, o_f2))
        fns["cudnn_fp8"] = fp8_be.forward
    except Exception as ex:  # noqa: BLE001
        res["cudnn_fp8_error"] = repr(ex)[:800]
        print("cudnn fp8 failed:", repr(ex)[:800], flush=True)
    res["timing"] = bench(fns, (q, k, v))
    print(json.dumps(res), flush=True)
    results[S] = res

print(json.dumps({"results": results}))
