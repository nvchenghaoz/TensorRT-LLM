"""Idea 68 paired microbenchmark + correctness: MiniMax-H3 DiT block's 4 BF16 Linears per rank (TP2).

Runs the 13 GEMMs one block issues on one rank, through tensorrt_llm Linear modules exactly as the
model calls them: QKV on the full packed sequence ([1, M, 5376] input, M = 37,966), then to_out,
gate_up and down on the 4 TP-overlap row chunks (_row_chunks(M, 4) = 9600/9600/9600/9166 rows).

The module flag comes from the tree under test: on a tree whose transformer_minimax_h3 has
_h3_sm120_cublas_pin (the candidate), the Linears get use_custom_cublas_mm=True exactly when the
model would set it; on main (baseline) they run F.linear.

--mode bench: CUDA-event timing of the whole block (rounds x iters, medians), plus per-shape.
--mode check: untimed sequence with the same warmup / iters; every output compared bitwise with
  F.linear on the same inputs (main's op), inputs checked unchanged; randn / realistic / outlier
  inputs at two packed lengths; error vs an FP32 torch.matmul reference reported.
Last stdout line: JSON.
"""
import argparse
import json
import statistics
import types

import torch
import torch.nn.functional as F

HID, QKV_N, ATT, FFN = 5376, 10752, 3584, 14336


def log(*a):
    print(*a, flush=True)


def row_chunks(rows, n=4, align=128):
    step = -(-rows // n)
    step = -(-step // align) * align
    return [(s, min(rows, s + step)) for s in range(0, rows, step)]


def pin_enabled():
    try:
        from tensorrt_llm._torch.visual_gen.models.minimax_h3 import transformer_minimax_h3 as h3
    except Exception as e:  # noqa
        log("import h3 failed", e)
        return False, "import_failed"
    fn = getattr(h3, "_h3_sm120_cublas_pin", None)
    if fn is None:
        return False, "absent (main)"
    return bool(fn(types.SimpleNamespace(torch_dtype=torch.bfloat16))), "present"


def make_linear(K, N, w, pin):
    from tensorrt_llm._torch.modules.linear import Linear
    lin = Linear(K, N, bias=False, dtype=torch.bfloat16).cuda()
    with torch.no_grad():
        lin.weight.copy_(w)
    lin.use_custom_cublas_mm = pin
    return lin


def make_act(M, K, kind, dist, gen):
    if dist == "randn":
        return torch.randn(M, K, generator=gen, device="cuda").to(torch.bfloat16)
    if kind == "swiglu":
        a = torch.randn(M, K, generator=gen, device="cuda") * 1.5
        b = torch.randn(M, K, generator=gen, device="cuda") * 1.5
        x = F.silu(a) * b
    else:
        x = torch.randn(M, K, generator=gen, device="cuda")
        x = x * torch.exp(0.5 * torch.randn(K, generator=gen, device="cuda")) + 0.3 * torch.randn(
            K, generator=gen, device="cuda")
    if dist == "outlier":
        idx = torch.randperm(K, generator=gen, device="cuda")[:8]
        x[:, idx] *= 50.0
    return x.to(torch.bfloat16).contiguous()


class Block:

    def __init__(self, pin, seed=1234):
        gen = torch.Generator(device="cuda")
        gen.manual_seed(seed)

        def w(N, K):
            t = torch.randn(N, K, generator=gen, device="cuda") / K ** 0.5
            return (t * torch.exp(0.3 * torch.randn(N, 1, generator=gen, device="cuda"))).to(
                torch.bfloat16)

        self.w = {"qkv": w(QKV_N, HID), "out": w(HID, ATT), "gate_up": w(FFN, HID),
                  "down": w(HID, FFN // 2)}
        self.lin = {"qkv": make_linear(HID, QKV_N, self.w["qkv"], pin),
                    "out": make_linear(ATT, HID, self.w["out"], pin),
                    "gate_up": make_linear(HID, FFN, self.w["gate_up"], pin),
                    "down": make_linear(FFN // 2, HID, self.w["down"], pin)}
        self.gen = gen

    def inputs(self, M, dist):
        g = self.gen
        return {"qkv": make_act(M, HID, "act", dist, g).view(1, M, HID),
                "out": make_act(M, ATT, "act", dist, g),
                "gate_up": make_act(M, HID, "act", dist, g),
                "down": make_act(M, FFN // 2, "swiglu", dist, g)}

    def run(self, x, M, ref=False):
        """The block's 13 GEMMs in model order; ref=True runs F.linear (main's op)."""
        f = (lambda name, a: F.linear(a, self.w[name])) if ref else (lambda name, a: self.lin[name](a))
        outs = [f("qkv", x["qkv"])]
        bounds = row_chunks(M)
        for s, e in bounds:
            outs.append(f("out", x["out"][s:e]))
        for s, e in bounds:
            outs.append(f("gate_up", x["gate_up"][s:e]))
            outs.append(f("down", x["down"][s:e]))
        return outs


def ev_times(fn, n):
    s = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    e = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    for i in range(n):
        s[i].record()
        fn()
        e[i].record()
    torch.cuda.synchronize()
    return [a.elapsed_time(b) for a, b in zip(s, e)]


def st(ts):
    ts = sorted(ts)
    med = statistics.median(ts)
    p10 = ts[int(0.1 * (len(ts) - 1))]
    p90 = ts[int(round(0.9 * (len(ts) - 1)))]
    return {"median_ms": med, "p10_ms": p10, "p90_ms": p90, "spread": (p90 - p10) / med, "n": len(ts)}


def bench(args):
    pin, why = pin_enabled()
    log(f"pin {pin} ({why}); device {torch.cuda.get_device_name()}")
    M = args.M
    blk = Block(pin)
    x = blk.inputs(M, "realistic")
    for _ in range(args.warmup):
        blk.run(x, M)
    torch.cuda.synchronize()
    block_t = []
    shape_t = {k: [] for k in ("qkv", "out", "gate_up", "down")}
    bounds = row_chunks(M)
    for r in range(args.rounds):
        block_t += ev_times(lambda: blk.run(x, M), args.iters)
        shape_t["qkv"] += ev_times(lambda: blk.lin["qkv"](x["qkv"]), args.iters)
        for name in ("out", "gate_up", "down"):
            shape_t[name] += ev_times(
                lambda name=name: [blk.lin[name](x[name][s:e]) for s, e in bounds], args.iters)
        log(f"round {r}: block median so far {statistics.median(block_t):.4f} ms")
    res = {"block": st(block_t)}
    for k, v in shape_t.items():
        res[k] = st(v)
        log(f"{k:8s} {res[k]['median_ms']:.4f} ms spread {res[k]['spread']:.4f}")
    log(f"block    {res['block']['median_ms']:.4f} ms spread {res['block']['spread']:.4f}")
    try:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            blk.run(x, M)
            torch.cuda.synchronize()
        names = sorted({e.name for e in prof.events() if e.device_type.name == "CUDA"})
        log("kernels:", names)
    except Exception as e:  # noqa
        names = [str(e)]
    out = {"batch": {"1": {"kernel_ms": res["block"]["median_ms"], "noise_floor": res["block"]["spread"]}},
           "output": {"class": "exact"}, "pin": pin, "pin_source": why, "M": M,
           "per_shape_ms": {k: res[k]["median_ms"] for k in shape_t},
           "per_shape_spread": {k: res[k]["spread"] for k in shape_t}, "kernels": names}
    print(json.dumps(out))


def check(args):
    pin, why = pin_enabled()
    log(f"pin {pin} ({why})")
    blk = Block(pin)
    passed = True
    cases = []
    for M in (args.M, args.M2):
        for dist in ("randn", "realistic", "outlier"):
            x = blk.inputs(M, dist)
            pristine = {k: v.clone() for k, v in x.items()}
            ref = blk.run(x, M, ref=True)  # main's F.linear on the same inputs
            bad = []
            n = args.warmup + args.iters
            for i in range(n):
                outs = blk.run(x, M)
                for j, (o, r) in enumerate(zip(outs, ref)):
                    if o.shape != r.shape or not torch.equal(o, r):
                        bad.append((i, j))
                for k in x:
                    if not torch.equal(x[k], pristine[k]):
                        bad.append((i, "input_changed", k))
            # error vs FP32 reference (info): qkv and the first gate_up chunk
            e = {}
            for name, a, o in (("qkv", x["qkv"].view(M, HID), outs[0].view(M, -1)),
                               ("gate_up0", x["gate_up"][:row_chunks(M)[0][1]], outs[5])):
                wname = "qkv" if name == "qkv" else "gate_up"
                r32 = torch.matmul(a.float(), blk.w[wname].float().t())
                e[name] = {"max_abs": (o.float() - r32).abs().max().item(),
                           "rel_l2": ((o.float() - r32).norm() / r32.norm()).item()}
            ok = not bad
            passed &= ok
            cases.append({"M": M, "dist": dist, "iterations": n, "gemms_per_iter": len(ref),
                          "mismatches": bad[:10], "passed": ok, "err_vs_fp32": e})
            log(f"M={M} {dist}: {n} iters x {len(ref)} GEMMs, bitwise vs F.linear "
                f"{'OK' if ok else 'FAIL ' + str(bad[:5])}; err vs fp32 {e}")
            del x, pristine, ref
            torch.cuda.empty_cache()
    # a shape outside the LUT keeps the heuristic: still bitwise vs F.linear
    extra = []
    for M in (1024, 4096):
        a = make_act(M, HID, "act", "realistic", blk.gen)
        o = blk.lin["gate_up"](a)
        r = F.linear(a, blk.w["gate_up"])
        extra.append({"M": M, "bitwise": bool(torch.equal(o, r))})
    log("non-LUT M:", extra)
    print(json.dumps({"passed": passed, "pin": pin, "pin_source": why,
                      "compared": "every output of the block's 13 GEMMs vs main's F.linear, bitwise",
                      "cases": cases, "non_lut_m": extra}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="bench")
    ap.add_argument("--M", type=int, default=37966)
    ap.add_argument("--M2", type=int, default=37838)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=5)
    args = ap.parse_args()
    import tensorrt_llm  # noqa: F401
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.mode == "bench":
        bench(args)
    else:
        check(args)


if __name__ == "__main__":
    main()
