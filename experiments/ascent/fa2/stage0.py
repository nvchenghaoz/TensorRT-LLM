"""Stage 0: bitwise + timing of FA2-rebuild tile variants vs torch SDPA FA2.

Per-rank TP2 shape of MiniMax-H3: B=1, H=28, D=128, BF16, non-causal, no mask.
Q/K: per-head RMSNorm (non-unit weight) + RoPE, contiguous [B,S,H,D].
V: strided view into a fused qkv buffer (row stride 3*H*D = 10752).
"""
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fa2_build import build  # noqa: E402

H, D = 28, 128
SCALE = 1.0 / math.sqrt(D)


def make_inputs(S, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    qkv = torch.randn(1, S, 3 * H * D, device="cuda", dtype=torch.bfloat16, generator=g)
    wq = (1.0 + 0.2 * torch.randn(D, device="cuda", generator=g)).to(torch.bfloat16)
    wk = (1.0 + 0.2 * torch.randn(D, device="cuda", generator=g)).to(torch.bfloat16)
    pos = torch.arange(S, device="cuda", dtype=torch.float32)
    inv = 1.0 / (10000 ** (torch.arange(0, D, 2, device="cuda", dtype=torch.float32) / D))
    ang = pos[:, None] * inv[None, :]
    cos = torch.cat([ang.cos(), ang.cos()], -1)[None, :, None, :]
    sin = torch.cat([ang.sin(), ang.sin()], -1)[None, :, None, :]

    def norm_rope(x, w):
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()
        x1, x2 = x[..., : D // 2], x[..., D // 2:]
        rot = torch.cat([-x2, x1], -1)
        return (x * cos + rot * sin).to(torch.bfloat16).contiguous()

    q = norm_rope(qkv[..., : H * D].view(1, S, H, D), wq)
    k = norm_rope(qkv[..., H * D: 2 * H * D].view(1, S, H, D), wk)
    v = qkv[..., 2 * H * D:].view(1, S, H, D)  # strided view, row stride 10752
    assert v.stride(1) == 3 * H * D
    return q, k, v, qkv


def ref_sdpa(q, k, v):
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=SCALE)


def ours(mod, q, k, v, variant, out=None, lse=None):
    B, S, Hh, Dd = q.shape
    if out is None:
        out = torch.empty(B, S, Hh, Dd, device=q.device, dtype=q.dtype)
    if lse is None:
        lse = torch.empty(B, Hh, S, device=q.device, dtype=torch.float32)
    mod.fwd(q, k, v, out, lse, SCALE, variant)
    return out.transpose(1, 2)


def first_diff(a, b):
    d = (a.float() - b.float()).abs()
    idx = (a != b).nonzero()
    n = idx.shape[0]
    return {"n_diff": int(n), "max_abs": float(d.max()), "first": idx[0].tolist() if n else None}


def time_fn(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    med = ts[len(ts) // 2] if len(ts) % 2 else 0.5 * (ts[len(ts) // 2 - 1] + ts[len(ts) // 2])
    p10 = ts[int(0.1 * (len(ts) - 1))]
    p90 = ts[int(round(0.9 * (len(ts) - 1)))]
    return {"median_ms": med, "p10_ms": p10, "p90_ms": p90, "min_ms": ts[0], "noise": (p90 - p10) / med}


def main():
    variants = [int(x) for x in os.environ.get("VARIANTS", "0,1,2,3,4,5,6,7").split(",")]
    sizes = [int(x) for x in os.environ.get("SIZES", "1000,4097,4096,37966").split(",")]
    do_time = os.environ.get("TIME", "1") == "1"
    splits = [tuple(int(a) for a in s.split(":")) for s in os.environ.get("SPLITS", "").split(",") if s]
    print("device", torch.cuda.get_device_name(), flush=True)
    mod = build(variants=range(8))
    infos = {}
    for vv in variants:
        try:
            infos[vv] = dict(zip(["block_m", "nthreads", "smem", "regs", "max_blocks_per_sm"], mod.info(vv)))
        except Exception as e:  # noqa: BLE001
            infos[vv] = {"err": str(e)}
        print("info", vv, infos[vv], flush=True)

    # confirm torch's kernel
    q, k, v, _ = make_inputs(1000)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        ref_sdpa(q, k, v)
        torch.cuda.synchronize()
    names = sorted({e.name for e in prof.events() if "flash" in e.name})
    print("torch kernels:", names, flush=True)

    results = {"info": infos, "bitwise": {}, "timing": {}}
    for S in sizes:
        q, k, v, qkv = make_inputs(S, seed=S)
        ref = ref_sdpa(q, k, v)
        torch.cuda.synchronize()
        for vb, vt in splits:
            try:
                o = torch.empty(1, S, H, D, device="cuda", dtype=torch.bfloat16)
                lse = torch.empty(1, H, S, device="cuda", dtype=torch.float32)
                nl = mod.fwd_split(q, k, v, o, lse, SCALE, vb, vt, 1.0)
                torch.cuda.synchronize()
                o = o.transpose(1, 2)
                eq = torch.equal(o, ref)
                rec = {"equal": eq, "launches": nl}
                if not eq:
                    rec.update(first_diff(o, ref))
            except Exception as e:  # noqa: BLE001
                rec = {"err": str(e)[:300]}
            results["bitwise"][f"{S}/s{vb}_{vt}"] = rec
            print("bitwise S", S, "split", vb, vt, rec, flush=True)
        for vv in variants:
            try:
                o = ours(mod, q, k, v, vv)
                torch.cuda.synchronize()
                eq = torch.equal(o, ref)
                rec = {"equal": eq}
                if not eq:
                    rec.update(first_diff(o, ref))
            except Exception as e:  # noqa: BLE001
                rec = {"err": str(e)[:300]}
            results["bitwise"][f"{S}/v{vv}"] = rec
            print("bitwise S", S, "v", vv, rec, flush=True)
        del q, k, v, qkv, ref
        torch.cuda.empty_cache()

    if do_time:
        S = int(os.environ.get("TIME_S", "37966"))
        q, k, v, _ = make_inputs(S, seed=S)
        flops = 4.0 * S * S * D * H
        out = torch.empty(1, S, H, D, device="cuda", dtype=torch.bfloat16)
        lse = torch.empty(1, H, S, device="cuda", dtype=torch.float32)
        order = ["sdpa"] + variants + [f"s{a}_{b}" for a, b in splits] + ["sdpa"]
        frac = float(os.environ.get("MAX_REM_FRAC", "0.5"))
        for name in order:
            if name == "sdpa":
                fn = lambda: ref_sdpa(q, k, v)  # noqa: E731
            elif isinstance(name, str):
                a, b = (int(x) for x in name[1:].split("_"))
                fn = (lambda a, b: (lambda: mod.fwd_split(q, k, v, out, lse, SCALE, a, b, frac)))(a, b)
            else:
                if "err" in infos.get(name, {}):
                    continue
                fn = (lambda vv: (lambda: mod.fwd(q, k, v, out, lse, SCALE, vv)))(name)
            r = time_fn(fn, 5, 20)
            r["tflops"] = flops / (r["median_ms"] * 1e-3) / 1e12
            key = name if isinstance(name, str) else f"v{name}"
            if key in results["timing"]:
                key += "_2"
            results["timing"][key] = r
            print("time", key, {kk: round(vv, 3) for kk, vv in r.items()}, flush=True)
    print(json.dumps(results))


if __name__ == "__main__":
    main()
