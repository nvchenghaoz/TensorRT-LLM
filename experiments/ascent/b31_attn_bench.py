"""Branch 31 / idea 58: FLUX.3 VAE mid-block attention options, compiled, synthetic q/k/v.

Shapes: q/k/v [1, 512, 128, 128] (the decoder mid block at a 1024^2 output), 1 head,
16384 tokens, d=512. Variants (each torch.compile(dynamic=False) like compile_decoder):
  base : rearrange 'b c h w -> b 1 (h w) c' + SDPA (takes MATH: stride(-1) != 1)
  d1   : same + .contiguous() on q/k/v (lets SDPA pick EFFICIENT if it accepts d=512)
  d3   : S = bmm(q^T, k, out_dtype=fp32) (bf16 tensor cores, fp32 scores), fp32 softmax
         with the d^-0.5 scale, P cast to bf16, O = bmm(v, P^T) (bf16, fp32 accumulate)
  d2   : (fp32 only) base under TF32 matmul
  d3x  : (fp32 only) q/k/v cast to bf16, then d3, output back to fp32
Reports per variant: CUDA-event median of ITERS calls (interleaved), noise, max/mean abs
vs base and vs an fp64 reference, kernel census of one call, dynamo frame counters.
usage: python b31_attn_bench.py [--dtypes bf16,fp32] [--iters 60] [--warmup 5]
"""

import argparse
import json
import math
import time
from collections import Counter

import torch
import torch.nn.functional as F
from einops import rearrange

p = argparse.ArgumentParser()
p.add_argument("--dtypes", default="bf16,fp32")
p.add_argument("--iters", type=int, default=60)
p.add_argument("--warmup", type=int, default=5)
p.add_argument("--variants", default="")
args = p.parse_args()


def log(*a):
    print(*a, flush=True)


def base(q, k, v):
    b, c, h, w = q.shape
    q = rearrange(q, "b c h w -> b 1 (h w) c")
    k = rearrange(k, "b c h w -> b 1 (h w) c")
    v = rearrange(v, "b c h w -> b 1 (h w) c")
    o = F.scaled_dot_product_attention(q, k, v)
    return rearrange(o, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)


def d1(q, k, v):
    b, c, h, w = q.shape
    q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
    k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
    v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
    o = F.scaled_dot_product_attention(q, k, v)
    return rearrange(o, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)


def _d3_core(q, k, v):
    b, c, h, w = q.shape
    q = q.reshape(b, c, h * w)
    k = k.reshape(b, c, h * w)
    v = v.reshape(b, c, h * w)
    s = torch.bmm(q.transpose(1, 2), k, out_dtype=torch.float32)  # [b, hw, hw] fp32
    pr = torch.softmax(s * (c ** -0.5), dim=-1).to(v.dtype)
    o = torch.bmm(v, pr.transpose(1, 2))  # [b, c, hw]
    return o.reshape(b, c, h, w)


def d3(q, k, v):
    return _d3_core(q, k, v)


def d3x(q, k, v):
    dt = q.dtype
    return _d3_core(q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)).to(dt)


def ref64(q, k, v):
    b, c, h, w = q.shape
    q = q.double().reshape(c, h * w)
    k = k.double().reshape(c, h * w)
    v = v.double().reshape(c, h * w)
    out = torch.empty(c, h * w, dtype=torch.float64, device=q.device)
    for i in range(0, h * w, 4096):
        s = (q[:, i:i + 4096].t() @ k) / math.sqrt(c)
        pr = torch.softmax(s, dim=-1)
        out[:, i:i + 4096] = (pr @ v.t()).t()
    return out.reshape(1, c, h, w)


def frames():
    import torch._dynamo.utils as du

    return {k: dict(v) for k, v in du.counters.items() if k in ("frames", "recompiles", "graph_break")}


def census(fn, *a):
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn(*a)
        torch.cuda.synchronize()
    cnt, dur = Counter(), Counter()
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            cnt[ev.name] += 1
            dur[ev.name] += ev.device_time_total
    return [(k[:100], cnt[k], round(dur[k] / 1e3, 3)) for k, _ in dur.most_common(8)]


class TF32:
    def __enter__(self):
        self.old = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True

    def __exit__(self, *a):
        torch.backends.cuda.matmul.allow_tf32 = self.old


def errs(a, r):
    d = (a.double() - r.double()).abs()
    return {"max_abs": d.max().item(), "mean_abs": d.mean().item()}


results = {}
for dname in args.dtypes.split(","):
    dt = {"bf16": torch.bfloat16, "fp32": torch.float32}[dname]
    torch.manual_seed(0)
    # GN-normalized input through 1x1 convs: roughly unit-variance q/k/v per channel.
    q, k, v = (torch.randn(1, 512, 128, 128, device="cuda").to(dt) for _ in range(3))
    q0, k0, v0 = q.clone(), k.clone(), v.clone()
    ref = ref64(q, k, v)
    opts = {} if dt == torch.float32 else {"emulate_precision_casts": True}
    variants = {"base": base, "d1": d1, "d3": d3}
    if dt == torch.float32:
        variants.update({"d2": base, "d3x": d3x})
    if args.variants:
        variants = {k_: v_ for k_, v_ in variants.items() if k_ in args.variants.split(",")}
    comp, out, info = {}, {}, {}
    for name, fn in variants.items():
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        c = torch.compile(fn, dynamic=False, options=dict(opts))
        ctx = TF32() if name == "d2" else None
        try:
            t0 = time.time()
            with torch.inference_mode():
                if ctx:
                    with ctx:
                        o = c(q, k, v)
                else:
                    o = c(q, k, v)
            torch.cuda.synchronize()
            f0 = frames()
            log(f"[{dname}] {name} compiled in {time.time() - t0:.1f}s frames={f0}")
            comp[name] = (c, ctx)
            out[name] = o.clone()
            with torch.inference_mode():
                if ctx:
                    with ctx:
                        cen = census(c, q, k, v)
                else:
                    cen = census(c, q, k, v)
            info[name] = {"frames_first": f0, "census": cen}
        except Exception as e:  # record the refusal
            log(f"[{dname}] {name} FAILED: {type(e).__name__}: {str(e)[:500]}")
            info[name] = {"error": f"{type(e).__name__}: {str(e)[:500]}"}
    # SDPA eager backend probe for d1 (what does SDPA pick on contiguous q/k/v?)
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        qq = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        probe = {}
        for be in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION):
            try:
                with sdpa_kernel([be]), torch.inference_mode():
                    F.scaled_dot_product_attention(qq, qq, qq)
                probe[be.name] = "ok"
            except Exception as e:
                probe[be.name] = str(e)[:300]
        info["sdpa_backend_probe_contiguous"] = probe
    except Exception as e:
        info["sdpa_backend_probe_contiguous"] = str(e)[:300]

    def call(name):
        c, ctx = comp[name]
        with torch.inference_mode():
            if ctx:
                with ctx:
                    return c(q, k, v)
            return c(q, k, v)

    names = list(comp)
    for name in names:
        for _ in range(args.warmup):
            call(name)
    torch.cuda.synchronize()
    ts = {n: [] for n in names}
    for i in range(args.iters):
        r = i % len(names)
        for n in names[r:] + names[:r]:
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            call(n)
            e.record()
            e.synchronize()
            ts[n].append(s.elapsed_time(e))
        if i % 20 == 0:
            log(f"[{dname}] iter {i} " + " ".join(f"{n}={ts[n][-1]:.3f}" for n in names))
    # untimed repeat sequence: same warmup + iters, bitwise vs first output, inputs unchanged
    for n in names:
        f0 = frames()
        first, rep, inp = None, True, True
        for i in range(args.warmup + args.iters):
            inp &= torch.equal(q, q0) and torch.equal(k, k0) and torch.equal(v, v0)
            o = call(n)
            if first is None:
                first = o.clone()
            elif not torch.equal(o, first):
                rep = False
        info[n].update({"repeat_bitwise": rep, "inputs_unchanged": inp,
                        "bitwise_vs_first_call": torch.equal(first, out[n]),
                        "no_recompile": frames() == f0})
    for n in names:
        tt = sorted(ts[n])
        m = tt[len(tt) // 2]
        info[n].update({"median_ms": m, "p10_ms": tt[len(tt) // 10], "p90_ms": tt[int(0.9 * len(tt)) - 1],
                        "noise_floor": (tt[int(0.9 * len(tt)) - 1] - tt[len(tt) // 10]) / m,
                        "vs_fp64": errs(out[n], ref),
                        "vs_base": errs(out[n], out["base"]) if "base" in out else None,
                        "bitwise_vs_base": torch.equal(out[n], out["base"]) if "base" in out else None})
        log(f"[{dname}] {n}: {json.dumps(info[n])}")
    results[dname] = info
    del comp, out, ref
    torch.cuda.empty_cache()

log(json.dumps({"results": {d: {n: {kk: vv for kk, vv in i.items() if kk != "census"} if isinstance(i, dict) else i
                                   for n, i in r.items()} for d, r in results.items()}}))
