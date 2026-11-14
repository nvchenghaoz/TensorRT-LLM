"""Idea 127 Stage 0/1 probe: CuTe DSL SM120 BF16 gate_up GEMM (+ SwiGLU epilogue).

Bitwise: plain kernel vs F.linear, fused kernel vs torch.ops.trtllm.silu_and_mul(F.linear), at the
gate_up chunk rows and QKV 37966, 3 input kinds, NaN-poisoned outputs, reversed-K control.
Timing (CUDA events, 5 warmup + 20 timed, two interleaved passes): F.linear (heuristic nvjet),
torch.ops.trtllm.cublas_mm (main's SM120 pin, landing 44), the unfused pairs with silu_and_mul,
and the CuTe plain / fused kernels, all in the shipped dynamic-M form (one compile per config).
Usage: python stage0.py [--cfgs a,b] [--time] [--full] [--epi]
"""
import argparse
import json
import time
import traceback

import torch
import torch.nn.functional as F

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.modules import gate_up_swiglu_sm120 as G


def log(*a):
    print(*a, flush=True)


def swiglu_ref(h):
    return torch.ops.trtllm.silu_and_mul(h)


def pin_mm(x, W):
    return torch.ops.trtllm.cublas_mm(x, W.t(), None, None)


def bitcmp(a, b):
    ai, bi = a.view(torch.int16), b.view(torch.int16)
    both_nan = torch.isnan(a) & torch.isnan(b)
    return int(((ai != bi) & ~both_nan).sum().item())


def epilogue_exhaustive(cfg):
    """All finite BF16 gate values x 3 up distributions through the fused kernel (identity W)."""
    bits = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16)
    g = bits.view(torch.bfloat16).cuda()
    g = g[torch.isfinite(g.float())]
    d = 320
    m = (g.numel() + d - 1) // d
    g = torch.cat([g, torch.zeros(m * d - g.numel(), dtype=g.dtype, device="cuda")]).view(m, d)
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
        out = torch.full((m, d), float("nan"), device="cuda", dtype=torch.bfloat16)
        G.gate_up_sm120(x, W, fused=True, out=out, cfg=cfg)
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
        r = {"pin_eq_flinear": bool(torch.equal(pin_mm(x, W), ref))}
        out = torch.full((m, n), float("nan"), device="cuda", dtype=torch.bfloat16)
        G.gate_up_sm120(x, W, fused=False, out=out, cfg=cfg)
        torch.cuda.synchronize()
        r["plain_eq"] = bool(torch.equal(out, ref))
        r["plain_mis"] = bitcmp(out, ref)
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


def kernel_names(fn):
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        fn()
        torch.cuda.synchronize()
    return sorted({e.name for e in p.events() if e.device_type.name == "CUDA"})[:6]


CFGS = {
    "SHIP": dict(G.DEFAULT_CFG),
    "SHIP_persistent": dict(G.DEFAULT_CFG, persistent=True),
    "T128_32s4": dict(tile_m=256, tile_n=128, tile_k=32, stages=4, atom_mn=(4, 2), group_m=16),
    "T160_32s3": dict(tile_m=256, tile_n=160, tile_k=32, stages=3, atom_mn=(4, 2), group_m=16),
    "T160_32s3_nofp": dict(tile_m=256, tile_n=160, tile_k=32, stages=3, atom_mn=(4, 2),
                           group_m=16, frag_prefetch=False),
    "T160_32s3_g8": dict(tile_m=256, tile_n=160, tile_k=32, stages=3, atom_mn=(4, 2), group_m=8),
    "T128_nofp": dict(G.DEFAULT_CFG, frag_prefetch=False),
    "T192x160_32s4_nofp": dict(tile_m=192, tile_n=160, tile_k=32, stages=4, atom_mn=(4, 2),
                               group_m=16, frag_prefetch=False),
    "T192x160_32s3_nofp": dict(tile_m=192, tile_n=160, tile_k=32, stages=3, atom_mn=(4, 2),
                               group_m=16, frag_prefetch=False),
    "T192x160_32s4": dict(tile_m=192, tile_n=160, tile_k=32, stages=4, atom_mn=(4, 2),
                          group_m=16),
    "T192x128_64s2_nofp": dict(tile_m=192, tile_n=128, tile_k=64, stages=2, atom_mn=(4, 2),
                               group_m=16, frag_prefetch=False),
    "T192x160_32s4_nofp_g8": dict(tile_m=192, tile_n=160, tile_k=32, stages=4, atom_mn=(4, 2),
                                  group_m=8, frag_prefetch=False),
    "T160_32s3_np": dict(tile_m=256, tile_n=160, tile_k=32, stages=3, atom_mn=(4, 2),
                         group_m=16, persistent=False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfgs", default="SHIP")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--epi", action="store_true")
    args = ap.parse_args()
    log("device", torch.cuda.get_device_name(), torch.cuda.get_device_capability())
    names = args.cfgs.split(",")
    summary = {}
    n, k, I = 14336, 5376, 7168
    Wg = make_weight(n, k)
    xk = make_inputs(9600, k, 0)
    log("pinned gate_up kernel:", kernel_names(lambda: pin_mm(xk, Wg)))
    log("F.linear gate_up kernel:", kernel_names(lambda: F.linear(xk, Wg)))
    for name in names:
        cfg = CFGS[name]
        log(f"=== cfg {name} {cfg} ===")
        try:
            t0 = time.time()
            r = check_shape(1000, n, k, cfg, kinds=(0,))
            log(f"  quick M=1000 gate_up: {r} ({time.time() - t0:.1f}s incl compile)")
            summary[name] = {"quick": r}
            if args.epi:
                summary[name]["epilogue_mis"] = epilogue_exhaustive(cfg)
        except Exception:
            log(traceback.format_exc()[-3000:])
            summary[name] = {"error": True}
            continue
        if args.time:
            for m in (9600, 9166):
                x = make_inputs(m, k, 0)
                h = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
                o = torch.empty(m, I, device="cuda", dtype=torch.bfloat16)
                o2 = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
                fns = {
                    "nvjet": lambda: torch.matmul(x, Wg.t(), out=h),
                    "pin": lambda: pin_mm(x, Wg),
                    "pair": lambda: swiglu_ref(F.linear(x, Wg)),
                    "pair_pin": lambda: swiglu_ref(pin_mm(x, Wg)),
                    "cute_plain": lambda: G.gate_up_sm120(x, Wg, fused=False, out=o2, cfg=cfg),
                    "cute_fused": lambda: G.gate_up_sm120(x, Wg, fused=True, out=o, cfg=cfg),
                }
                t = bench(fns)
                st = {kk: stats(v) for kk, v in t.items()}
                st["plain_pct_nvjet"] = round(100 * st["cute_plain"]["med"] / st["nvjet"]["med"], 1)
                st["plain_pct_pin"] = round(100 * st["cute_plain"]["med"] / st["pin"]["med"], 1)
                st["fused_pct_pair"] = round(100 * st["cute_fused"]["med"] / st["pair"]["med"], 1)
                st["fused_pct_pair_pin"] = round(
                    100 * st["cute_fused"]["med"] / st["pair_pin"]["med"], 1)
                log(f"  timing M={m}: {json.dumps(st)}")
                summary[name][f"t{m}"] = {kk: st[kk] for kk in (
                    "plain_pct_nvjet", "plain_pct_pin", "fused_pct_pair", "fused_pct_pair_pin")}
                summary[name][f"t{m}"].update({kk: st[kk]["med"] for kk in fns})
    if args.full:
        for name in names:
            if summary.get(name, {}).get("error"):
                continue
            cfg = CFGS[name]
            full = {}
            for m in (9600, 9166, 9472, 9296):
                r = check_shape(m, n, k, cfg)
                log(f"  [{name}] gate_up M={m}: {r}")
                full[f"gate_up_{m}"] = r
            r = check_shape(37966, 10752, k, cfg, fused_too=False)
            log(f"  [{name}] qkv M=37966: {r}")
            full["qkv_37966"] = r
            summary[name]["full"] = full
            # QKV timing: pinned vs F.linear vs the plain kernel
            if args.time:
                Wq = make_weight(10752, k)
                xq = make_inputs(37966, k, 0)
                oq = torch.empty(37966, 10752, device="cuda", dtype=torch.bfloat16)
                hq = torch.empty(37966, 10752, device="cuda", dtype=torch.bfloat16)
                log("pinned qkv kernel:", kernel_names(lambda: pin_mm(xq, Wq)))
                t = bench({"nvjet": lambda: torch.matmul(xq, Wq.t(), out=hq),
                           "pin": lambda: pin_mm(xq, Wq),
                           "cute_plain": lambda: G.gate_up_sm120(xq, Wq, fused=False, out=oq,
                                                                 cfg=cfg)})
                st = {kk: stats(v) for kk, v in t.items()}
                st["plain_pct_nvjet"] = round(100 * st["cute_plain"]["med"] / st["nvjet"]["med"], 1)
                st["plain_pct_pin"] = round(100 * st["cute_plain"]["med"] / st["pin"]["med"], 1)
                log(f"  [{name}] qkv timing: {json.dumps(st)}")
                summary[name]["qkv_time"] = st
    log(json.dumps({"summary": summary, "compile_count": G.COMPILE_COUNT}))


if __name__ == "__main__":
    main()
