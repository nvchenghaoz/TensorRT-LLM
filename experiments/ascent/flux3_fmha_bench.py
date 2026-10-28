"""FLUX.3 d128 FP8 attention microbenchmark / correctness for idea 48 (branch 23).

Shapes: B=1, H=24, D=128, non-causal; Q/K/V are strided views of ONE packed BF16
[1, S, 9216] buffer (real call form), per-tensor E4M3 quantized like main's
_quantize_e4m3, then fmha_v2 with device scale_bmm1_d / scale_bmm2_d.

Modes
  step0   stock kernel only: CUDA-event us/call per S (also used under ncu).
  ab      in-process interleaved A/B: stock vs variants (--variants ms,rq,ms+rq,ms+rq@3),
          plus bitwise check of each variant vs stock (randn + peaky, S=4176/4096/4256).
  pair    one arm for run_gpu_pair: the branch's default variant if this tree has it,
          else the stock kernel. Prints the evidence JSON.
  check   untimed correctness of the default variant: same warmup/iteration count as
          'pair', each iteration bitwise vs the stock reference and vs iteration 0,
          inputs unchanged, interleaved S, CUDA-graph replay == eager.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import threading
import time
from pathlib import Path

import torch

H, D = 24, 128
CALLS = {4176: 28, 4096: 5}  # per transformer pass
ROOT = Path(__file__).resolve().parents[2]
VARIANT_FILE = ROOT / "tensorrt_llm/_torch/visual_gen/attention/backends/fmha_v2_flux3.py"


def log(*a):
    print(*a, flush=True)


def load_variant_lib():
    if not VARIANT_FILE.exists():
        return None
    spec = importlib.util.spec_from_file_location("fmha_v2_flux3", VARIANT_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def with_heartbeat(label, fn):
    done = threading.Event()
    t0 = time.time()

    def beat():
        while not done.wait(30):
            log(f"[heartbeat] {label} building... {time.time() - t0:.0f}s")

    th = threading.Thread(target=beat, daemon=True)
    th.start()
    try:
        return fn()
    finally:
        done.set()
        log(f"[build] {label} ready in {time.time() - t0:.1f}s")


def parse_variant(s):
    min_blocks = 0
    if "@" in s:
        s, mb = s.split("@")
        min_blocks = int(mb)
    names = {"ms": "mask_skip", "rq": "resident_q"}
    levers = tuple(sorted(names[p] for p in s.split("+") if p and p != "base"))
    return levers, min_blocks


def quant(x):
    amax = x.float().abs().amax().clamp_min(1e-12)
    descale = (amax / 448.0).reshape(1)
    xq = (x.float() / descale).to(torch.float8_e4m3fn).contiguous()
    return xq, descale.float().contiguous()


def make_inputs(S, kind, seed, H=H):
    g = torch.Generator(device="cuda").manual_seed(seed)
    qkv = torch.randn((1, S, 3 * H * D), generator=g, device="cuda", dtype=torch.float32)
    if kind == "peaky":
        gain = torch.ones(D, device="cuda")
        idx = torch.randperm(D, generator=torch.Generator().manual_seed(seed))[:8].cuda()
        gain[idx] = 8.0
        qkv[..., 0 : H * D].view(1, S, H, D).mul_(gain)
        qkv[..., H * D : 2 * H * D].view(1, S, H, D).mul_(gain)
        qkv[..., 0 : 2 * H * D] *= 1.5
    qkv = qkv.to(torch.bfloat16)
    q = qkv[..., 0 : H * D].view(1, S, H, D)
    k = qkv[..., H * D : 2 * H * D].view(1, S, H, D)
    v = qkv[..., 2 * H * D :].view(1, S, H, D)
    qf, qd = quant(q)
    kf, kd = quant(k)
    vf, vd = quant(v)
    s1 = (qd * kd * (1.0 / math.sqrt(D))).contiguous()
    return dict(S=S, H=H, q=qf, k=kf, v=vf, s1=s1, s2=vd, qd=qd, kd=kd, vd=vd)


def run(module, inp, out):
    module.run(inp["q"], inp["k"], inp["v"], out, None, inp["s1"], inp["s2"], inp["H"], D,
               inp["S"], 1.0, 1.0, 1.0, False, True, True)
    return out


def mode_hstair(args):
    """Stock and default-variant us/call vs head count (wave-quantization probe)."""
    stock = stock_module()
    lib = load_variant_lib()
    impls = {"stock": stock}
    if lib is not None:
        impls["variant"] = variant_module(lib, lib.DEFAULT_LEVERS, lib.DEFAULT_MIN_BLOCKS)
    res = {}
    for S in (4176, 4096):
        for h in (20, 22, 23, 24, 25, 26, 28):
            inp = make_inputs(S, "randn", 1, H=h)
            out = torch.empty((1, S, h, D), device="cuda", dtype=torch.bfloat16)
            for name, m in impls.items():
                for _ in range(args.warmup):
                    run(m, inp, out)
                xs = []
                for _ in range(args.reps):
                    a = torch.cuda.Event(enable_timing=True)
                    b = torch.cuda.Event(enable_timing=True)
                    a.record()
                    run(m, inp, out)
                    b.record()
                    xs.append((a, b))
                torch.cuda.synchronize()
                med, nf = stats([a.elapsed_time(b) * 1e3 for a, b in xs])
                ctas = ((S + 63) // 64) * h
                res[f"{name}/S{S}/H{h}"] = dict(us=round(med, 2), noise=round(nf, 4), ctas=ctas,
                                                us_per_head=round(med / h, 3))
                log(f"{name:8s} S={S} H={h:2d} CTAs={ctas:5d}  {med:8.1f} us  {med / h:6.2f} us/head")
    print(json.dumps({"hstair": res}))


def stock_module():
    import flashinfer.prefill as fp

    return with_heartbeat("stock trtllm_fmha_v2", fp.get_trtllm_fmha_v2_sm120_module)


def variant_module(lib, levers, mb):
    return with_heartbeat(f"variant {levers} mb={mb}",
                          lambda: lib.get_flux3_d128_module(levers, mb))


def time_interleaved(impls, inputs, warmup, reps):
    """impls: name -> module; inputs: list of input dicts. Returns name -> S -> list[us]."""
    outs = {S: torch.empty((1, S, H, D), device="cuda", dtype=torch.bfloat16)
            for S in {i["S"] for i in inputs}}
    for _ in range(warmup):
        for m in impls.values():
            for inp in inputs:
                run(m, inp, outs[inp["S"]])
    torch.cuda.synchronize()
    ev = {n: {i["S"]: [] for i in inputs} for n in impls}
    for r in range(reps):
        order = list(impls.items())
        if r % 2:
            order.reverse()
        for name, m in order:
            for inp in inputs:
                a = torch.cuda.Event(enable_timing=True)
                b = torch.cuda.Event(enable_timing=True)
                a.record()
                run(m, inp, outs[inp["S"]])
                b.record()
                ev[name][inp["S"]].append((a, b))
        if r % 50 == 49:
            log(f"[timing] rep {r + 1}/{reps}")
    torch.cuda.synchronize()
    return {n: {S: [a.elapsed_time(b) * 1e3 for a, b in lst] for S, lst in d.items()}
            for n, d in ev.items()}


def stats(xs):
    t = torch.tensor(xs, dtype=torch.float64)
    med = t.median().item()
    p10 = t.quantile(0.1).item()
    p90 = t.quantile(0.9).item()
    return med, (p90 - p10) / med


def pristine(inp):
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in inp.items()}


def same_inputs(a, b):
    return all(torch.equal(a[k].view(torch.uint8) if a[k].dtype == torch.float8_e4m3fn else a[k],
                           b[k].view(torch.uint8) if b[k].dtype == torch.float8_e4m3fn else b[k])
               for k in ("q", "k", "v", "s1", "s2"))


def fp32_ref(inp):
    q = inp["q"].float() * inp["qd"]
    k = inp["k"].float() * inp["kd"]
    v = inp["v"].float() * inp["vd"]
    o = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=1.0 / math.sqrt(D))
    return o.transpose(1, 2)


def err(o, ref):
    d = (o.float() - ref)
    return (d.norm() / ref.norm()).item(), d.abs().max().item()


def mode_step0(args):
    stock = stock_module()
    inputs = [make_inputs(S, "randn", 1) for S in (4176, 4096)]
    if args.launches:
        out = {i["S"]: torch.empty((1, i["S"], H, D), device="cuda", dtype=torch.bfloat16)
               for i in inputs}
        for _ in range(args.launches):
            for inp in inputs:
                run(stock, inp, out[inp["S"]])
        torch.cuda.synchronize()
        log("done launches")
        return
    t = time_interleaved({"stock": stock}, inputs, args.warmup, args.reps)
    res = {}
    for S, xs in t["stock"].items():
        med, nf = stats(xs)
        flops = 4.0 * H * S * S * D
        res[S] = dict(us=med, noise=nf, tflops=flops / med / 1e6)
        log(f"stock S={S}: median {med:.1f} us  noise {nf:.3f}  {flops / med / 1e6:.0f} TFLOPS")
    print(json.dumps({"step0": res}))


def mode_ab(args):
    lib = load_variant_lib()
    stock = stock_module()
    impls = {"stock": stock}
    for vs in args.variants.split(","):
        levers, mb = parse_variant(vs)
        impls[vs] = variant_module(lib, levers, mb)
        log(f"{vs}: kernel {lib.flux3_d128_kernel_name(levers, mb)}")
    # Bitwise vs stock.
    bitwise = {}
    for kind in ("randn", "peaky"):
        for S in (4176, 4096, 4256):
            inp = make_inputs(S, kind, 7)
            ref = run(stock, inp, torch.empty((1, S, H, D), device="cuda", dtype=torch.bfloat16))
            for name, m in impls.items():
                if name == "stock":
                    continue
                o = run(m, inp, torch.empty_like(ref))
                bitwise[f"{name}/{kind}/{S}"] = bool(torch.equal(o, ref))
    torch.cuda.synchronize()
    log("bitwise:", json.dumps(bitwise))
    inputs = [make_inputs(S, "randn", 1) for S in (4176, 4096)]
    t = time_interleaved(impls, inputs, args.warmup, args.reps)
    table = {}
    for name, d in t.items():
        row = {}
        for S, xs in d.items():
            med, nf = stats(xs)
            row[S] = dict(us=round(med, 2), noise=round(nf, 4))
        row["pass_ms"] = round(sum(CALLS[S] * row[S]["us"] for S in CALLS) / 1e3, 4)
        table[name] = row
    for name, row in table.items():
        rel = row["pass_ms"] / table["stock"]["pass_ms"] - 1
        log(f"{name:12s} S4176 {row[4176]['us']:8.1f}us ({row[4176]['noise']:.3f})  "
            f"S4096 {row[4096]['us']:8.1f}us  pass {row['pass_ms']:.3f} ms  {rel * 100:+.2f}%")
    print(json.dumps({"ab": table, "bitwise": bitwise, "all_bitwise": all(bitwise.values())}))


def get_arm():
    lib = load_variant_lib()
    if lib is None:
        return "stock", stock_module()
    levers, mb = lib.DEFAULT_LEVERS, lib.DEFAULT_MIN_BLOCKS
    log(f"variant kernel {lib.flux3_d128_kernel_name(levers, mb)}")
    return "variant", variant_module(lib, levers, mb)


def mode_pair(args):
    arm, m = get_arm()
    inputs = [make_inputs(S, "randn", 1) for S in (4176, 4096)]
    snaps = [pristine(i) for i in inputs]
    t = time_interleaved({arm: m}, inputs, args.warmup, args.reps)[arm]
    assert all(same_inputs(i, s) for i, s in zip(inputs, snaps)), "inputs mutated"
    per = {S: stats(xs) for S, xs in t.items()}
    series = [sum(CALLS[S] * t[S][r] for S in CALLS) / 1e3 for r in range(args.reps)]
    med, nf = stats(series)
    for S, (m_, n_) in per.items():
        log(f"{arm} S={S}: {m_:.1f} us (noise {n_:.3f})")
    log(f"{arm} per-pass (28x4176 + 5x4096): {med:.4f} ms noise {nf:.4f}")
    print(json.dumps({"batch": {"1": {"kernel_ms": med, "noise_floor": max(nf, *[n for _, n in per.values()])}},
                      "output": {"class": "exact"}, "arm": arm,
                      "per_call_us": {str(S): v[0] for S, v in per.items()}}))


def mode_check(args):
    lib = load_variant_lib()
    stock = stock_module()
    levers, mb = lib.DEFAULT_LEVERS, lib.DEFAULT_MIN_BLOCKS
    var = variant_module(lib, levers, mb)
    log(f"variant kernel {lib.flux3_d128_kernel_name(levers, mb)}")
    failures = []
    report = {}
    # 1) Bitwise vs stock: randn + peaky, S=4176/4096/4256; fp32 error for scale.
    for kind in ("randn", "peaky"):
        for S in (4176, 4096, 4256):
            inp = make_inputs(S, kind, 11)
            ref = run(stock, inp, torch.empty((1, S, H, D), device="cuda", dtype=torch.bfloat16))
            o = run(var, inp, torch.empty_like(ref))
            ok = torch.equal(o, ref)
            r32 = fp32_ref(inp)
            report[f"bitwise/{kind}/{S}"] = dict(equal=bool(ok), stock_vs_fp32=err(ref, r32),
                                                 variant_vs_fp32=err(o, r32))
            if not ok:
                failures.append(f"bitwise {kind} {S}")
    log(json.dumps(report))
    # 2) Repeated calls, same warmup/iteration count and reset points as timing.
    inputs = [make_inputs(S, "randn", 1) for S in (4176, 4096)]
    snaps = [pristine(i) for i in inputs]
    refs = {i["S"]: run(stock, i, torch.empty((1, i["S"], H, D), device="cuda", dtype=torch.bfloat16)).clone()
            for i in inputs}
    first = {}
    out = {i["S"]: torch.empty((1, i["S"], H, D), device="cuda", dtype=torch.bfloat16) for i in inputs}
    for it in range(args.warmup + args.reps):
        for inp in inputs:
            o = run(var, inp, out[inp["S"]])
            if it == 0:
                first[inp["S"]] = o.clone()
            if not torch.equal(o, refs[inp["S"]]) or not torch.equal(o, first[inp["S"]]):
                failures.append(f"repeat it={it} S={inp['S']}")
    torch.cuda.synchronize()
    if not all(same_inputs(i, s) for i, s in zip(inputs, snaps)):
        failures.append("inputs mutated")
    log(f"repeated {args.warmup + args.reps} iterations done, failures so far {len(failures)}")
    # 3) Interleaved S: 4176 -> 4096 -> 4256 -> 4176 equal fresh calls.
    seq = [make_inputs(S, "randn", 3) for S in (4176, 4096, 4256)]
    fresh = {i["S"]: run(var, i, torch.empty((1, i["S"], H, D), device="cuda", dtype=torch.bfloat16)).clone()
             for i in seq}
    for inp in seq + [seq[0]]:
        o = run(var, inp, torch.empty((1, inp["S"], H, D), device="cuda", dtype=torch.bfloat16))
        if not torch.equal(o, fresh[inp["S"]]):
            failures.append(f"interleaved S={inp['S']}")
    # 4) CUDA graph capture/replay == eager.
    inp = inputs[0]
    gout = torch.empty((1, inp["S"], H, D), device="cuda", dtype=torch.bfloat16)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        run(var, inp, gout)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    gout.zero_()
    with torch.cuda.graph(g):
        run(var, inp, gout)
    for _ in range(3):
        gout.zero_()
        g.replay()
        torch.cuda.synchronize()
        if not torch.equal(gout, refs[inp["S"]]):
            failures.append("graph replay")
    passed = not failures
    print(json.dumps({"passed": passed, "failures": failures[:20],
                      "compared": "variant vs stock trtllm_fmha_v2 d128 e4m3 kernel (bitwise), "
                                  "repeated calls vs iteration 0, interleaved S, graph replay",
                      "errors_vs_fp32": {k: v["variant_vs_fp32"] for k, v in report.items()}}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["step0", "ab", "pair", "check", "hstair"])
    p.add_argument("--variants", default="ms,rq,ms+rq")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--reps", type=int, default=200)
    p.add_argument("--launches", type=int, default=0)
    args = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    log(f"device {torch.cuda.get_device_name()}  root {ROOT}")
    {"step0": mode_step0, "ab": mode_ab, "pair": mode_pair, "check": mode_check,
     "hstair": mode_hstair}[args.mode](args)


if __name__ == "__main__":
    main()
