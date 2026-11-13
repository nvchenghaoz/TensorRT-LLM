"""Idea 127 Stage 0/1 probe: CuTe DSL SM120 BF16 gate_up GEMM (+ SwiGLU epilogue).

Loads the kernel module and swiglu.py by file path (no full tensorrt_llm import).
Usage: python stage0.py [--quick] [--cfgs all|default] [--time] [--shapes ...]
"""
import argparse
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G = _load("gate_up_swiglu_sm120", "tensorrt_llm/_torch/modules/gate_up_swiglu_sm120.py")
SW = _load("h3_swiglu", "tensorrt_llm/_torch/modules/swiglu.py")


def log(*a):
    print(*a, flush=True)


def swiglu_ref(h):
    """Exactly torch.ops.trtllm.silu_and_mul(h) (same Triton kernel, same launch args)."""
    b, n = h.shape
    d = n // 2
    o = torch.empty((b, d), dtype=h.dtype, device=h.device)
    grid = lambda meta: (b, (d + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"])
    SW.silu_and_mul_kernel[grid](o_ptr=o, o_stride=o.stride(0), o_scale_ptr=None, x_ptr=h,
                                 x_stride=h.stride(0), d=d, swiglu_limit=0.0, swiglu_alpha=1.0,
                                 swiglu_beta=0.0, BLOCK_SIZE=1024, HAS_O_SCALE=False,
                                 HAS_SWIGLU_LIMIT=False)
    return o


def dump_triton_ptx():
    h = torch.randn(4, 2048, device="cuda", dtype=torch.bfloat16)
    swiglu_ref(h)
    torch.cuda.synchronize()
    try:
        cache = SW.silu_and_mul_kernel.device_caches
        found = None
        for dev, entry in cache.items():
            kc = entry[0] if isinstance(entry, tuple) else entry
            for k, v in kc.items():
                found = v
        ptx = found.asm["ptx"]
    except Exception as e:  # noqa
        log("ptx dump failed:", repr(e))
        return
    keep = [ln.strip() for ln in ptx.splitlines()
            if any(t in ln for t in ("ex2", "div", "rcp", "mul.f32", "add.f32", "neg", "cvt",
                                     "fma", "sub.f32", "mov.b32 %f", "0f"))]
    log("=== triton silu_and_mul PTX (math lines) ===")
    for ln in keep[:80]:
        log("  ", ln)


def bitcmp(a, b):
    ai, bi = a.view(torch.int16), b.view(torch.int16)
    both_nan = torch.isnan(a) & torch.isnan(b)
    diff = (ai != bi) & ~both_nan
    return int(diff.sum().item())


def epilogue_exhaustive(cfg):
    """All finite BF16 gate values x 3 up distributions through the fused kernel (identity W)."""
    bits = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16)
    g = bits.view(torch.bfloat16).cuda()
    finite = torch.isfinite(g.float())
    g = g[finite]
    d = 256
    m = (g.numel() + d - 1) // d
    pad = m * d - g.numel()
    g = torch.cat([g, torch.zeros(pad, dtype=g.dtype, device="cuda")]).view(m, d)
    W = torch.eye(2 * d, device="cuda", dtype=torch.bfloat16)
    fails = 0
    for kind in range(3):
        torch.manual_seed(kind)
        if kind == 0:
            u = torch.randn(m, d, device="cuda").to(torch.bfloat16)
        elif kind == 1:
            u = (torch.randn(m, d, device="cuda") * 100).to(torch.bfloat16)
        else:
            u = g.flip(0).clone()
        x = torch.cat([g, u], dim=1).contiguous()
        ref = swiglu_ref(F.linear(x, W))
        out = G.gate_up_sm120(x, W, fused=True, cfg=cfg)
        torch.cuda.synchronize()
        nd = bitcmp(out, ref)
        fails += nd
        log(f"  epilogue exhaustive kind={kind}: M={m} mismatches={nd}")
    return fails


def make_inputs(m, k, kind, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed + kind)
    x = torch.randn(m, k, device="cuda", generator=gen)
    if kind == 1:
        x = x * torch.exp(torch.randn(1, k, device="cuda", generator=gen))
    elif kind == 2:
        x[:, :8] *= 50
    return x.to(torch.bfloat16).contiguous()


def make_weight(n, k, seed=123):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(n, k, device="cuda", generator=gen) * 0.02).to(torch.bfloat16).contiguous()


def check_shape(m, n, k, cfg, fused_too=True, kinds=(0, 1, 2), control=True):
    W = make_weight(n, k)
    res = {}
    for kind in kinds:
        x = make_inputs(m, k, kind)
        ref = F.linear(x, W)
        out = torch.full((m, n), float("nan"), device="cuda", dtype=torch.bfloat16)
        G.gate_up_sm120(x, W, fused=False, out=out, cfg=cfg)
        torch.cuda.synchronize()
        r = {"plain_eq": bool(torch.equal(out, ref)), "plain_mis": bitcmp(out, ref)}
        if fused_too:
            ref2 = swiglu_ref(ref)
            o2 = torch.full((m, n // 2), float("nan"), device="cuda", dtype=torch.bfloat16)
            G.gate_up_sm120(x, W, fused=True, out=o2, cfg=cfg)
            torch.cuda.synchronize()
            r["fused_eq"] = bool(torch.equal(o2, ref2))
            r["fused_mis"] = bitcmp(o2, ref2)
        if control and kind == 0:
            outr = G.gate_up_sm120(x.flip(1).contiguous(), W.flip(1).contiguous(), fused=False,
                                   cfg=cfg)
            torch.cuda.synchronize()
            r["revK_mis"] = bitcmp(outr, ref)
        res[kind] = r
    return res


def bench(fns, warm=5, iters=20, passes=2):
    """Interleaved CUDA-event timing; returns {name: [ms...]}."""
    times = {k: [] for k in fns}
    for _ in range(passes):
        for k, f in fns.items():
            for _ in range(warm):
                f()
        torch.cuda.synchronize()
        for _ in range(iters):
            for k, f in fns.items():
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                f()
                e.record()
                e.synchronize()
                times[k].append(s.elapsed_time(e))
    return times


def stats(v):
    t = torch.tensor(v)
    return dict(med=round(t.median().item(), 4), p10=round(t.quantile(0.1).item(), 4),
                p90=round(t.quantile(0.9).item(), 4))


CFGS = {
    "128x256x64s2_w2x4": dict(tile_m=128, tile_n=256, tile_k=64, stages=2, atom_mn=(2, 4), group_m=8),
    "128x256x32s4_w2x4": dict(tile_m=128, tile_n=256, tile_k=32, stages=4, atom_mn=(2, 4), group_m=8),
    "256x128x64s2_w4x2": dict(tile_m=256, tile_n=128, tile_k=64, stages=2, atom_mn=(4, 2), group_m=8),
    "256x128x32s4_w4x2": dict(tile_m=256, tile_n=128, tile_k=32, stages=4, atom_mn=(4, 2), group_m=8),
    "128x128x64s3_w2x2": dict(tile_m=128, tile_n=128, tile_k=64, stages=3, atom_mn=(2, 2), group_m=8),
}
for _base in ("128x256x64s2_w2x4", "128x256x32s4_w2x4", "256x128x64s2_w4x2", "256x128x32s4_w4x2"):
    CFGS["F_" + _base] = dict(CFGS[_base], prefetch_next=True)
    CFGS["P_" + _base] = dict(CFGS[_base], prefetch_next=True, persistent=True)
    CFGS["Pn_" + _base] = dict(CFGS[_base], persistent=True)
    CFGS["S_" + _base] = dict(CFGS[_base], use_setmaxnreg=True)
    CFGS["SP_" + _base] = dict(CFGS[_base], use_setmaxnreg=True, prefetch_next=True,
                               persistent=True)
    CFGS["I_" + _base] = dict(CFGS[_base], inline_producer=True)
for _gm in (4, 16):
    CFGS[f"I_256x128x32s4_w4x2_g{_gm}"] = dict(CFGS["256x128x32s4_w4x2"], inline_producer=True,
                                              group_m=_gm)
for _gm in (8, 16, 32):
    CFGS[f"IP_256x128x32s4_w4x2_g{_gm}"] = dict(CFGS["256x128x32s4_w4x2"], inline_producer=True,
                                               persistent=True, group_m=_gm)
CFGS["IP_128x256x32s4_w2x4_g16"] = dict(CFGS["128x256x32s4_w2x4"], inline_producer=True,
                                        persistent=True, group_m=16)
CFGS["IP_256x128x64s2_w4x2_g16"] = dict(CFGS["256x128x64s2_w4x2"], inline_producer=True,
                                        persistent=True, group_m=16)
CFGS["SHIP"] = dict(G.DEFAULT_CFG)
CFGS["I_256x128x32s3_w4x2"] = dict(CFGS["256x128x32s4_w4x2"], inline_producer=True, stages=3)
CFGS["P_128x128x64s3_w2x2"] = dict(CFGS["128x128x64s3_w2x2"], prefetch_next=True, persistent=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfgs", default="all")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--ptx", action="store_true")
    args = ap.parse_args()
    log("device", torch.cuda.get_device_name(), torch.cuda.get_device_capability())
    if args.ptx:
        dump_triton_ptx()
    names = list(CFGS) if args.cfgs == "all" else args.cfgs.split(",")
    summary = {}
    for name in names:
        cfg = CFGS[name]
        log(f"=== cfg {name} ===")
        try:
            t0 = time.time()
            xs = make_inputs(256, 512, 0)
            Ws = make_weight(512, 512)
            log("  compiling small plain ...")
            o = G.gate_up_sm120(xs, Ws, fused=False, cfg=cfg)
            log(f"  launched ({time.time() - t0:.1f}s), syncing ...")
            torch.cuda.synchronize()
            log(f"  small plain eq={torch.equal(o, F.linear(xs, Ws))} mis={bitcmp(o, F.linear(xs, Ws))}")
            r = check_shape(1000, 14336, 5376, cfg, kinds=(0,))
            log(f"  quick M=1000 gate_up: {r} (incl compile {time.time() - t0:.1f}s)")
            ep = epilogue_exhaustive(cfg)
            summary[name] = {"quick": r, "epilogue_mis": ep}
        except Exception:
            log(traceback.format_exc())
            summary[name] = {"error": True}
            continue
        if args.time:
            n, k, I = 14336, 5376, 7168
            W = make_weight(n, k)
            for m in (9600, 9166):
                x = make_inputs(m, k, 0)
                h = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
                o = torch.empty(m, I, device="cuda", dtype=torch.bfloat16)
                o2 = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
                fns = {
                    "nvjet": lambda: torch.matmul(x, W.t(), out=h),
                    "pair": lambda: swiglu_ref(F.linear(x, W)),
                    "cute_plain": lambda: G.gate_up_sm120(x, W, fused=False, out=o2, cfg=cfg),
                    "cute_fused": lambda: G.gate_up_sm120(x, W, fused=True, out=o, cfg=cfg),
                }
                t = bench(fns)
                st = {kk: stats(v) for kk, v in t.items()}
                st["plain_pct_nvjet"] = round(100 * st["cute_plain"]["med"] / st["nvjet"]["med"], 1)
                st["fused_pct_pair"] = round(100 * st["cute_fused"]["med"] / st["pair"]["med"], 1)
                log(f"  timing M={m}: {json.dumps(st)}")
                summary[name][f"t{m}"] = st
    if args.full:
        for name in names:
            if summary.get(name, {}).get("error"):
                continue
            cfg = CFGS[name]
            for m in (9600, 9166, 9472, 9296):
                r = check_shape(m, 14336, 5376, cfg)
                log(f"  [{name}] gate_up M={m}: {r}")
            r = check_shape(37966, 10752, 5376, cfg, fused_too=False)
            log(f"  [{name}] qkv M=37966: {r}")
    log(json.dumps({"summary": summary, "compile_count": G.COMPILE_COUNT}))


if __name__ == "__main__":
    main()
