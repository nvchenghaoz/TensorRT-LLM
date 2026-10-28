"""Idea 90 (branch 45): rqms_b2 (64x128 tile + base-2 softmax) routed inside
run_fmha_v2_fp8_sm12x. Synthetic FLUX.3 shapes (B=1, H=24, D=128, S=4176/4096), per-tensor
E4M3 inputs like main's _quantize_e4m3, device scale_bmm1_d / scale_bmm2_d.

Modes
  ab     in-process interleaved CUDA-event us/call: ms_rq (main) vs routed levers, direct modules,
         plus error vs fp32 SDPA (randn + peaky, S=4176/4096/4256).
  pair   one arm for run_gpu_pair, timed THROUGH run_fmha_v2_fp8_sm12x (base tree -> ms_rq,
         branch tree -> rqms_b2). Prints the evidence JSON.
  check  untimed: route check (kernel names via torch.profiler through the helper: non-causal
         no-LSE -> rqms_b2; causal and LSE -> ms_rq bitwise vs direct ms_rq), tolerance vs fp32
         (rel L2 <= 1.05x ms_rq's, max abs <= 2x ms_rq's), repeated calls (same warmup/reps as
         pair) vs iteration 0, interleaved S, CUDA-graph replay == eager, inputs unchanged,
         ms_rq still bitwise vs stock.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import torch

H, D = 24, 128
CALLS = {4176: 28, 4096: 5}


def log(*a):
    print(*a, flush=True)


def helper_mod():
    t0 = time.time()
    from tensorrt_llm._torch.visual_gen.attention.backends import flashinfer as fib
    log(f"[import] tensorrt_llm flashinfer backend in {time.time() - t0:.1f}s")
    return fib


def lib_mod():
    from tensorrt_llm._torch.visual_gen.attention.backends import fmha_v2_flux3 as lib
    return lib


def quant(x):
    amax = x.float().abs().amax().clamp_min(1e-12)
    descale = (amax / 448.0).reshape(1)
    xq = (x.float() / descale).to(torch.float8_e4m3fn).contiguous()
    return xq, descale.float().contiguous()


def make_inputs(S, kind, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    qkv = torch.randn((1, S, 3 * H * D), generator=g, device="cuda", dtype=torch.float32)
    if kind == "peaky":
        gain = torch.ones(D, device="cuda")
        idx = torch.randperm(D, generator=torch.Generator().manual_seed(seed))[:8].cuda()
        gain[idx] = 8.0
        qkv[..., 0:H * D].view(1, S, H, D).mul_(gain)
        qkv[..., H * D:2 * H * D].view(1, S, H, D).mul_(gain)
        qkv[..., 0:2 * H * D] *= 1.5
    qkv = qkv.to(torch.bfloat16)
    q = qkv[..., 0:H * D].view(1, S, H, D)
    k = qkv[..., H * D:2 * H * D].view(1, S, H, D)
    v = qkv[..., 2 * H * D:].view(1, S, H, D)
    qf, qd = quant(q)
    kf, kd = quant(k)
    vf, vd = quant(v)
    s1 = (qd * kd * (1.0 / math.sqrt(D))).contiguous()
    return dict(S=S, q=qf, k=kf, v=vf, s1=s1, s2=vd, qd=qd, kd=kd, vd=vd)


def new_out(S):
    return torch.empty((1, S, H, D), device="cuda", dtype=torch.bfloat16)


def run_module(m, inp, out, causal=False, lse=None):
    m.run(inp["q"], inp["k"], inp["v"], out, lse, inp["s1"], inp["s2"], H, D, inp["S"],
          1.0, 1.0, 1.0, causal, True, True)
    return out


def run_helper(fib, inp, out, causal=False, lse=None):
    return fib.run_fmha_v2_fp8_sm12x(inp["q"], inp["k"], inp["v"], out, H, D, inp["S"],
                                     scale_bmm1_d=inp["s1"], scale_bmm2_d=inp["s2"],
                                     causal=causal, return_lse=lse is not None, lse=lse)


def fp32_ref(inp):
    q = inp["q"].float() * inp["qd"]
    k = inp["k"].float() * inp["kd"]
    v = inp["v"].float() * inp["vd"]
    o = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=1.0 / math.sqrt(D))
    return o.transpose(1, 2)


def err(o, ref):
    d = o.float() - ref
    return (d.norm() / ref.norm()).item(), d.abs().max().item()


def stats(xs):
    t = torch.tensor(xs, dtype=torch.float64)
    med = t.median().item()
    return med, (t.quantile(0.9).item() - t.quantile(0.1).item()) / med


def same_inputs(a, b):
    return all(torch.equal(a[k].view(torch.uint8) if a[k].dtype == torch.float8_e4m3fn else a[k],
                           b[k].view(torch.uint8) if b[k].dtype == torch.float8_e4m3fn else b[k])
               for k in ("q", "k", "v", "s1", "s2"))


def pristine(inp):
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in inp.items()}


def build(label, fn):
    t0 = time.time()
    log(f"[build] {label} ...")
    m = fn()
    log(f"[build] {label} ready in {time.time() - t0:.1f}s")
    return m


def time_interleaved(fns, inputs, warmup, reps):
    outs = {i["S"]: new_out(i["S"]) for i in inputs}
    for _ in range(warmup):
        for f in fns.values():
            for inp in inputs:
                f(inp, outs[inp["S"]])
    torch.cuda.synchronize()
    ev = {n: {i["S"]: [] for i in inputs} for n in fns}
    for r in range(reps):
        order = list(fns.items())
        if r % 2:
            order.reverse()
        for name, f in order:
            for inp in inputs:
                a = torch.cuda.Event(enable_timing=True)
                b = torch.cuda.Event(enable_timing=True)
                a.record()
                f(inp, outs[inp["S"]])
                b.record()
                ev[name][inp["S"]].append((a, b))
        if r % 50 == 49:
            log(f"[timing] rep {r + 1}/{reps}")
    torch.cuda.synchronize()
    return {n: {S: [a.elapsed_time(b) * 1e3 for a, b in l] for S, l in d.items()}
            for n, d in ev.items()}


def mode_ab(args):
    lib = lib_mod()
    msrq = build("ms_rq", lambda: lib.get_flux3_d128_module(lib.DEFAULT_LEVERS))
    variants = {"ms_rq": msrq}
    for spec in args.variants.split(","):
        lev = tuple(sorted(spec.split("+")))
        variants[spec] = build(spec, lambda lev=lev: lib.get_flux3_d128_module(lev))
    errs = {}
    for kind in ("randn", "peaky"):
        for S in (4176, 4096, 4256):
            inp = make_inputs(S, kind, 7)
            r32 = fp32_ref(inp)
            for n, m in variants.items():
                errs[f"{n}/{kind}/{S}"] = err(run_module(m, inp, new_out(S)), r32)
    log("errors vs fp32 (rel_l2, max_abs):", json.dumps(errs))
    inputs = [make_inputs(S, "randn", 1) for S in (4176, 4096)]
    fns = {n: (lambda inp, out, m=m: run_module(m, inp, out)) for n, m in variants.items()}
    t = time_interleaved(fns, inputs, args.warmup, args.reps)
    table = {}
    for n, d in t.items():
        row = {S: dict(zip(("us", "noise"), map(lambda x: round(x, 4), stats(xs))))
               for S, xs in d.items()}
        row["pass_ms"] = sum(CALLS[S] * row[S]["us"] for S in CALLS) / 1e3
        table[n] = row
    for n, row in table.items():
        rel = row["pass_ms"] / table["ms_rq"]["pass_ms"] - 1
        log(f"{n:28s} S4176 {row[4176]['us']:8.1f}us S4096 {row[4096]['us']:8.1f}us "
            f"pass {row['pass_ms']:.4f} ms {rel * 100:+.2f}% vs ms_rq")
    print(json.dumps({"ab": {n: {str(k): v for k, v in r.items()} for n, r in table.items()},
                      "errors": errs}))


def mode_pair(args):
    fib = helper_mod()
    inputs = [make_inputs(S, "randn", 1) for S in (4176, 4096)]
    snaps = [pristine(i) for i in inputs]
    t0 = time.time()
    for inp in inputs:  # builds the routed module (JIT) outside timing
        run_helper(fib, inp, new_out(inp["S"]))
    torch.cuda.synchronize()
    log(f"[build] helper first call {time.time() - t0:.1f}s")
    t = time_interleaved({"arm": lambda inp, out: run_helper(fib, inp, out)}, inputs,
                         args.warmup, args.reps)["arm"]
    assert all(same_inputs(i, s) for i, s in zip(inputs, snaps)), "inputs mutated"
    per = {S: stats(xs) for S, xs in t.items()}
    series = [sum(CALLS[S] * t[S][r] for S in CALLS) / 1e3 for r in range(args.reps)]
    med, nf = stats(series)
    for S, (m_, n_) in per.items():
        log(f"S={S}: {m_:.1f} us (noise {n_:.3f})")
    log(f"per-pass (28x4176 + 5x4096): {med:.4f} ms noise {nf:.4f}")
    print(json.dumps({"batch": {"1": {"kernel_ms": med,
                                      "noise_floor": max(nf, *[n for _, n in per.values()])}},
                      "output": {"class": "reordered"},
                      "per_call_us": {str(S): v[0] for S, v in per.items()}}))


def kernel_names(fn):
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sorted({e.key for e in prof.key_averages() if "fmha" in e.key or "flash" in e.key})


def mode_check(args):
    fib = helper_mod()
    lib = lib_mod()
    import flashinfer.prefill as fp
    stock = build("stock", fp.get_trtllm_fmha_v2_sm120_module)
    msrq = build("ms_rq", lambda: lib.get_flux3_d128_module(lib.DEFAULT_LEVERS))
    routed_name = lib.flux3_d128_kernel_name(lib.ROUTED_LEVERS)
    msrq_name = lib.flux3_d128_kernel_name(lib.DEFAULT_LEVERS)
    failures, report = [], {}
    # 1) Route check through the helper.
    inp = make_inputs(4176, "randn", 5)
    names_nc = kernel_names(lambda: run_helper(fib, inp, new_out(4176)))
    names_c = kernel_names(lambda: run_helper(fib, inp, new_out(4176), causal=True))
    st = torch.empty((1, 4176, H, 2), device="cuda", dtype=torch.float32)
    names_l = kernel_names(lambda: run_helper(fib, inp, new_out(4176), lse=st))
    report["route"] = dict(noncausal_nolse=names_nc, causal=names_c, lse=names_l,
                           routed_kernel=routed_name, msrq_kernel=msrq_name)
    log("route:", json.dumps(report["route"]))
    if not any(routed_name in n for n in names_nc):
        failures.append("non-causal no-LSE did not launch rqms_b2")
    if not any(msrq_name in n for n in names_c) or any(routed_name in n for n in names_c):
        failures.append("causal not on ms_rq")
    if not any(msrq_name in n for n in names_l) or any(routed_name in n for n in names_l):
        failures.append("LSE not on ms_rq")
    # causal / LSE outputs bitwise vs direct ms_rq; ms_rq bitwise vs stock (non-causal).
    for S in (4176, 4096):
        i2 = make_inputs(S, "randn", 9)
        a = run_helper(fib, i2, new_out(S), causal=True)
        b = run_module(msrq, i2, new_out(S), causal=True)
        if not torch.equal(a, b):
            failures.append(f"causal helper != ms_rq S={S}")
        la = torch.empty((1, S, H, 2), device="cuda", dtype=torch.float32)
        lb = torch.empty_like(la)
        a = run_helper(fib, i2, new_out(S), lse=la)
        b = run_module(msrq, i2, new_out(S), lse=lb)
        if not (torch.equal(a, b) and torch.equal(la, lb)):
            failures.append(f"LSE helper != ms_rq S={S}")
        if not torch.equal(run_module(msrq, i2, new_out(S)), run_module(stock, i2, new_out(S))):
            failures.append(f"ms_rq not bitwise vs stock S={S}")
    # 2) Tolerance vs fp32 SDPA (declared before the run).
    for kind in ("randn", "peaky"):
        for S in (4176, 4096, 4256):
            i3 = make_inputs(S, kind, 11)
            r32 = fp32_ref(i3)
            e_new = err(run_helper(fib, i3, new_out(S)), r32)
            e_ref = err(run_module(msrq, i3, new_out(S)), r32)
            ok = e_new[0] <= 1.05 * e_ref[0] and e_new[1] <= 2.0 * e_ref[1]
            report[f"tol/{kind}/{S}"] = dict(rqms_b2=e_new, ms_rq=e_ref,
                                             rel_l2_ratio=e_new[0] / e_ref[0],
                                             max_abs_ratio=e_new[1] / e_ref[1], ok=ok)
            if not ok:
                failures.append(f"tolerance {kind} {S}")
    log(json.dumps({k: v for k, v in report.items() if k.startswith("tol")}))
    # 3) Repeated calls, same warmup/reps as pair, vs iteration 0; inputs unchanged.
    inputs = [make_inputs(S, "randn", 1) for S in (4176, 4096)]
    snaps = [pristine(i) for i in inputs]
    first, out = {}, {i["S"]: new_out(i["S"]) for i in inputs}
    for it in range(args.warmup + args.reps):
        for i4 in inputs:
            o = run_helper(fib, i4, out[i4["S"]])
            if it == 0:
                first[i4["S"]] = o.clone()
            elif not torch.equal(o, first[i4["S"]]):
                failures.append(f"repeat it={it} S={i4['S']}")
    torch.cuda.synchronize()
    if not all(same_inputs(i, s) for i, s in zip(inputs, snaps)):
        failures.append("inputs mutated")
    log(f"repeated {args.warmup + args.reps} iterations, failures {len(failures)}")
    # 4) Interleaved S equals fresh calls.
    seq = [make_inputs(S, "randn", 3) for S in (4176, 4096, 4256)]
    fresh = {i["S"]: run_helper(fib, i, new_out(i["S"])).clone() for i in seq}
    for i5 in seq + [seq[0], seq[1]]:
        if not torch.equal(run_helper(fib, i5, new_out(i5["S"])), fresh[i5["S"]]):
            failures.append(f"interleaved S={i5['S']}")
    # 5) CUDA graph capture/replay == eager.
    i6 = inputs[0]
    gout = new_out(i6["S"])
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        run_helper(fib, i6, gout)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run_helper(fib, i6, gout)
    for _ in range(3):
        gout.zero_()
        g.replay()
        torch.cuda.synchronize()
        if not torch.equal(gout, first[i6["S"]]):
            failures.append("graph replay")
    print(json.dumps({"passed": not failures, "failures": failures[:20],
                      "compared": "rqms_b2 via run_fmha_v2_fp8_sm12x vs fp32 SDPA (tolerance "
                                  "relative to main's ms_rq), route by kernel name, causal/LSE "
                                  "bitwise vs ms_rq, repeats vs it0, interleaved S, graph replay",
                      "route": report["route"],
                      "tolerance": {k: v for k, v in report.items() if k.startswith("tol")}}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["ab", "pair", "check"])
    p.add_argument("--variants", default="base2+kv128+mask_skip+resident_q,kv128+mask_skip,"
                                         "base2+mask_skip+resident_q")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--reps", type=int, default=200)
    args = p.parse_args()
    log(f"device {torch.cuda.get_device_name()}")
    {"ab": mode_ab, "pair": mode_pair, "check": mode_check}[args.mode](args)


if __name__ == "__main__":
    main()
