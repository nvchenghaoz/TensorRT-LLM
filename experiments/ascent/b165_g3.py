"""Branch 165 (idea 245): G3 shared-K/V fmha_v2 flux3 kernel -- timing and bitwise checks.

Synthetic E4M3 Q/K/V (B=1, H=24, D=128) from main's quantize_qkv_e4m3 on one BF16 buffer per S,
fixed seed; pristine copies restored before timing and before every correctness sequence.

Modes
  pair   one arm for run_gpu_pair: the tree's own routing through
         flashinfer.run_fmha_v2_fp8_sm12x (base -> main's ROUTED_LEVERS, branch -> + g3).
         One sample = CUDA-event-timed replay of a graph holding the per-pass d128 set
         (5 x S4096 + 28 x S4176); kernel_ms = graph / 33. noise = (p90-p10)/p50.
  check  branch tree: module A = ROUTED_LEVERS without g3 (main's kernel), module B = A + g3.
         For every S in the list: B == A bitwise (eager), 100 repeated B calls == B call 0,
         graph(B) == eager(B), inputs unchanged. Plus resource usage (REG/STACK/smem) and the
         SASS hash of the 4-warp kernel in A and the G3 kernel in B.
  ab     branch tree: graphs A and B interleaved (same per-pass set), paired ratios.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import subprocess
import time

import torch

H, D = 24, 128
SEQ = [4096] * 5 + [4176] * 28
K = len(SEQ)


def log(*a):
    print(*a, flush=True)


def make_inp(S, seed):
    from tensorrt_llm._torch.visual_gen.attention.backends.fp8_sm12x import quantize_qkv_e4m3

    g = torch.Generator(device="cuda").manual_seed(seed)
    buf = torch.randn((3, 1, S, H, D), generator=g, device="cuda", dtype=torch.float32)
    buf = (buf * 2.0).to(torch.bfloat16)
    q, k, v = buf.unbind(0)
    q8, k8, v8, scales = quantize_qkv_e4m3(q, k, v, 1.0 / math.sqrt(D))
    return dict(S=S, q=q8, k=k8, v=v8, scales=scales)


def snap(inp):
    return {n: inp[n].clone() for n in ("q", "k", "v", "scales")}


def same(a, b):
    if a.dtype == torch.float8_e4m3fn:
        a, b = a.view(torch.uint8), b.view(torch.uint8)
    return torch.equal(a, b)


def unchanged(inp, s):
    return all(same(inp[n], s[n]) for n in s)


def restore(inp, s):
    for n in s:
        inp[n].copy_(s[n])


def call_lib(lib, mod, inp, out):
    sc = inp["scales"]
    lib.fmha_v2_flux3_d128(inp["q"], inp["k"], inp["v"], out, H, D, inp["S"],
                           scale_bmm1_d=sc[3:4], scale_bmm2_d=sc[2:3], causal=False, lse=None,
                           module=mod)


def call_fib(fib, inp, out):
    sc = inp["scales"]
    fib.run_fmha_v2_fp8_sm12x(inp["q"], inp["k"], inp["v"], out, H, D, inp["S"],
                              scale_bmm1_d=sc[3:4], scale_bmm2_d=sc[2:3], causal=False,
                              return_lse=False)


def pct(xs, p):
    return torch.tensor(xs, dtype=torch.float64).quantile(p).item()


def summary(xs):
    p10, p50, p90 = pct(xs, 0.1), pct(xs, 0.5), pct(xs, 0.9)
    return dict(n=len(xs), p10=p10, p50=p50, p90=p90, noise=(p90 - p10) / p50)


def sm_clock():
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
    except Exception:  # noqa: BLE001
        return None


def pass_inputs():
    return {S: make_inp(S, 1 + S) for S in sorted(set(SEQ))}


def make_outputs():
    flat = torch.empty(sum(SEQ) * H * D, device="cuda", dtype=torch.bfloat16)
    outs, off = [], 0
    for S in SEQ:
        outs.append(flat[off:off + S * H * D].view(1, S, H, D))
        off += S * H * D
    return flat, outs


def capture(fn, inputs):
    flat, outs = make_outputs()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            for S, o in zip(SEQ, outs):
                fn(inputs[S], o)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for S, o in zip(SEQ, outs):
            fn(inputs[S], o)
    torch.cuda.synchronize()
    return g, flat


def graph_kernels(g):
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        g.replay()
        torch.cuda.synchronize()
    return {e.key: e.count for e in prof.key_averages() if "fmha" in e.key}


def soak(graphs, seconds):
    t0, n = time.time(), 0
    while time.time() - t0 < seconds:
        for g in graphs:
            g.replay()
        n += 1
        if n % 20 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    return n


def timed(arms, blocks, block):
    samples = {n: [] for n, _ in arms}
    for b in range(blocks):
        c0 = sm_clock()
        ev = []
        for i in range(block):
            order = arms if (b + i) % 2 == 0 else arms[::-1]
            for n, g in order:
                a, z = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                g.replay()
                z.record()
                ev.append((n, a, z))
        torch.cuda.synchronize()
        for n, a, z in ev:
            samples[n].append(a.elapsed_time(z))
        med = {n: round(sorted(v[-block:])[block // 2], 4) for n, v in samples.items()}
        log(f"[block {b + 1}/{blocks}] sm_mhz {c0}->{sm_clock()} median ms {med}")
    return samples


def mode_pair(args):
    from tensorrt_llm._torch.visual_gen.attention.backends import flashinfer as fib
    from tensorrt_llm._torch.visual_gen.attention.backends import fmha_v2_flux3 as lib

    log(f"[tree] ROUTED_LEVERS {lib.ROUTED_LEVERS}")
    inputs = pass_inputs()
    snaps = {S: snap(inputs[S]) for S in inputs}
    g, flat = capture(lambda inp, o: call_fib(fib, inp, o), inputs)
    names = graph_kernels(g)
    log(f"[graph] {json.dumps(names)}")
    n = soak([g], args.soak_s)
    log(f"[soak] {n} replays")
    xs = timed([("arm", g)], args.blocks, args.block)["arm"]
    s = summary(xs)
    ok = all(unchanged(inputs[S], snaps[S]) for S in inputs)
    # untimed repeat check: same soak/sample counts, every replay == replay 0
    for S in inputs:
        restore(inputs[S], snaps[S])
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    ref = flat.clone()
    bad = 0
    for i in range(args.blocks * args.block):
        g.replay()
        if (i + 1) % 25 == 0:
            torch.cuda.synchronize()
            bad += int(not torch.equal(flat, ref))
    torch.cuda.synchronize()
    bad += int(not torch.equal(flat, ref))
    log(f"[result] per-pass p50 {s['p50']:.4f} ms noise {s['noise']:.4f} per-call "
        f"{s['p50'] / K * 1e3:.1f} us; repeat mismatches {bad}; inputs unchanged {ok}")
    print(json.dumps({"batch": {"1": {"kernel_ms": s["p50"] / K, "noise_floor": s["noise"]}},
                      "output": {"class": "exact"},
                      "sample": f"one CUDA-event replay of a graph of {K} routed d128 FP8 "
                                f"calls (5xS4096 + 28xS4176); kernel_ms = graph/{K}",
                      "per_pass_ms": s, "kernels": names, "routed_levers": list(lib.ROUTED_LEVERS),
                      "repeat_mismatches": bad, "inputs_unchanged": ok,
                      "output_sha": hashlib.sha256(ref.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()}))


def modules(sample=None):
    from tensorrt_llm._torch.visual_gen.attention.backends import fmha_v2_flux3 as lib

    lev_b = tuple(lib.ROUTED_LEVERS)
    if sample is not None and hasattr(lib, "load_trim_levers"):
        lev_b = tuple(lib.load_trim_levers(lev_b, sample["k"], sample["v"]))
    if "g3" not in lev_b:
        lev_b = (*lev_b, "g3")
    lev_a = tuple(x for x in lev_b if x != "g3")
    t0 = time.time()
    ma = lib.get_flux3_d128_module(lev_a)
    mb = lib.get_flux3_d128_module(lev_b)
    log(f"[build] A {lib._tag(lev_a, 0)} B {lib._tag(lev_b, 0)} in {time.time() - t0:.1f}s")
    return lib, lev_a, lev_b, ma, mb


def resource_usage(lib, levers):
    from flashinfer.jit import env as jit_env

    uri = f"trtllm_fmha_v2_flux3_d128_{lib._tag(levers, 0)}"
    sos = [p for p in glob.glob(str(jit_env.FLASHINFER_JIT_DIR / uri / "**" / "*.so"),
                                recursive=True)]
    sos += [p for p in glob.glob(str(jit_env.FLASHINFER_JIT_DIR / "**" / f"{uri}*.so"),
                                 recursive=True)]
    sos += [p for p in glob.glob(str(jit_env.FLASHINFER_JIT_DIR / "**" / uri / "*.o"),
                                 recursive=True)]
    out = {"uri": uri, "files": sos, "functions": {}}
    for so in sos:
        try:
            r = subprocess.run(["cuobjdump", "-res-usage", so], capture_output=True, text=True,
                               timeout=120)
        except Exception as exc:  # noqa: BLE001
            out["error"] = str(exc)
            continue
        cur = None
        for line in r.stdout.splitlines():
            m = re.search(r"Function (\S+):", line)
            if m:
                cur = m.group(1)
                continue
            if cur and "REG:" in line:
                out["functions"][cur] = line.strip()
                cur = None
        try:
            r = subprocess.run(["cuobjdump", "-sass", so], capture_output=True, text=True,
                               timeout=300)
            funcs, name, buf = {}, None, []
            for line in r.stdout.splitlines():
                m = re.search(r"Function : (\S+)", line)
                if m:
                    if name:
                        funcs[name] = buf
                    name, buf = m.group(1), []
                    continue
                if name:
                    buf.append(re.sub(r"/\*[0-9a-f]{4,}\*/", "", line).strip())
            if name:
                funcs[name] = buf
            out["sass_sha"] = {
                n: hashlib.sha256("\n".join(b).encode()).hexdigest()[:16] for n, b in funcs.items()
                if "padding" not in n and "causal" not in n and "custom" not in n}
            out["sass_stl_ldl"] = {n: sum(("STL" in x or "LDL" in x) for x in b)
                                   for n, b in funcs.items() if "_g3" in n}
            ctx = []
            for n, b in funcs.items():
                if "_g3" not in n:
                    continue
                out["sass_g3_lines"] = len(b)
                for i, x in enumerate(b):
                    if "STL" in x or "LDL" in x:
                        ctx.append(f"--- line {i}")
                        ctx += b[max(0, i - 6):i + 4]
            out["sass_stl_ldl_ctx"] = ctx[:400]
        except Exception as exc:  # noqa: BLE001
            out["sass_error"] = str(exc)
        break
    return out


def mode_check(args):
    lib, lev_a, lev_b, ma, mb = modules(make_inp(4176, 7))
    failures = []
    res = {"levers_A": list(lev_a), "levers_B": list(lev_b), "per_S": {}}
    for S in args.seqs:
        inp = make_inp(S, 1000 + S)
        sn = snap(inp)
        oa = torch.empty((1, S, H, D), device="cuda", dtype=torch.bfloat16)
        ob = torch.empty_like(oa)
        oa.fill_(7.0)
        ob.fill_(-3.0)
        call_lib(lib, ma, inp, oa)
        call_lib(lib, mb, inp, ob)
        torch.cuda.synchronize()
        eq = torch.equal(oa.view(torch.int16), ob.view(torch.int16))
        nd = int((oa.view(torch.int16) != ob.view(torch.int16)).sum().item())
        # repeats
        ref = ob.clone()
        rep_bad = 0
        for i in range(args.repeats):
            ob.fill_(1.0)
            call_lib(lib, mb, inp, ob)
            if (i + 1) % 10 == 0 or i == args.repeats - 1:
                torch.cuda.synchronize()
            rep_bad += int(not torch.equal(ob.view(torch.int16), ref.view(torch.int16)))
        # graph == eager
        og = torch.empty_like(ob)
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            call_lib(lib, mb, inp, og)
        torch.cuda.current_stream().wait_stream(st)
        torch.cuda.synchronize()
        og.fill_(5.0)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            call_lib(lib, mb, inp, og)
        g.replay()
        g.replay()
        torch.cuda.synchronize()
        geq = torch.equal(og.view(torch.int16), ref.view(torch.int16))
        finite = bool(torch.isfinite(ob.float()).all().item())
        inok = unchanged(inp, sn)
        # fp32 reference error (scale)
        sc = inp["scales"]
        q = inp["q"].float() * sc[0]
        k = inp["k"].float() * sc[1]
        v = inp["v"].float() * sc[2]
        r32 = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            scale=1.0 / math.sqrt(D)).transpose(1, 2)
        rel = ((ob.float() - r32).norm() / r32.norm()).item()
        q_tiles = (S + 63) // 64
        row = dict(bitwise_B_eq_A=eq, ndiff=nd, repeat_mismatch=rep_bad, graph_eq_eager=geq,
                   finite=finite, inputs_unchanged=inok, rel_l2_vs_fp32=rel,
                   q_tiles=q_tiles, ctas_per_head=(q_tiles + 2) // 3,
                   last_cta_valid_groups=q_tiles - 3 * ((q_tiles + 2) // 3 - 1))
        res["per_S"][str(S)] = row
        log(f"[S={S}] {json.dumps(row)}")
        if not (eq and rep_bad == 0 and geq and finite and inok):
            failures.append(f"S={S}")
        del g
    res["resources_B"] = resource_usage(lib, lev_b)
    res["resources_A"] = resource_usage(lib, lev_a)
    log(f"[res B] {json.dumps(res['resources_B'])}")
    log(f"[res A] {json.dumps(res['resources_A'])}")
    res["passed"] = not failures
    res["failures"] = failures
    res["compared"] = ("G3 module (ROUTED_LEVERS+g3) vs the same levers without g3 (main's routed "
                       "kernel), bitwise per S; 100 repeats == call 0; graph replay == eager; "
                       "inputs unchanged")
    print(json.dumps(res))


def mode_ab(args):
    inputs = pass_inputs()
    lib, lev_a, lev_b, ma, mb = modules(inputs[4176])
    snaps = {S: snap(inputs[S]) for S in inputs}
    gA, fA = capture(lambda inp, o: call_lib(lib, ma, inp, o), inputs)
    gA2, fA2 = capture(lambda inp, o: call_lib(lib, ma, inp, o), inputs)
    gB, fB = capture(lambda inp, o: call_lib(lib, mb, inp, o), inputs)
    kA, kB = graph_kernels(gA), graph_kernels(gB)
    log(f"[graph A] {json.dumps(kA)}")
    log(f"[graph B] {json.dumps(kB)}")
    for S in inputs:
        restore(inputs[S], snaps[S])
    n = soak([gA, gA2, gB], args.soak_s)
    log(f"[soak] {n}")
    smp = timed([("A", gA), ("A2", gA2), ("B", gB)], args.blocks, args.block)
    sA, sA2, sB = summary(smp["A"]), summary(smp["A2"]), summary(smp["B"])
    sP = summary([b / a for a, b in zip(smp["A"], smp["B"])])
    sAA = summary([b / a for a, b in zip(smp["A"], smp["A2"])])
    # untimed replica: same soak count and sample count, every replay == replay 0
    for S in inputs:
        restore(inputs[S], snaps[S])
    for _ in range(n):
        for g in (gA, gA2, gB):
            g.replay()
    torch.cuda.synchronize()
    refs, bad = {}, 0
    for i in range(args.blocks * args.block):
        for nm, g, f in (("A", gA, fA), ("A2", gA2, fA2), ("B", gB, fB)):
            g.replay()
            if nm not in refs:
                torch.cuda.synchronize()
                refs[nm] = f.clone()
            elif (i + 1) % 25 == 0:
                torch.cuda.synchronize()
                bad += int(not torch.equal(f.view(torch.int16), refs[nm].view(torch.int16)))
    torch.cuda.synchronize()
    eq = torch.equal(refs["A"].view(torch.int16), refs["B"].view(torch.int16))
    inok = all(unchanged(inputs[S], snaps[S]) for S in inputs)
    res = {"levers_A": list(lev_a), "levers_B": list(lev_b), "kernels_A": kA, "kernels_B": kB,
           "A_per_call_us": sA["p50"] / K * 1e3, "A2_per_call_us": sA2["p50"] / K * 1e3,
           "B_per_call_us": sB["p50"] / K * 1e3,
           "A": sA, "A2": sA2, "B": sB, "paired_ratio_B_over_A": sP,
           "paired_ratio_AA": sAA, "graph_outputs_bitwise_equal": eq,
           "replica_mismatches": bad, "inputs_unchanged": inok}
    log(f"[ab] B/A {sP['p50']:.4f} [{sP['p10']:.4f}, {sP['p90']:.4f}]  A/A {sAA['p50']:.4f} "
        f"[{sAA['p10']:.4f}, {sAA['p90']:.4f}]")
    # per-S graphs
    for S in sorted(set(SEQ)):
        inp = inputs[S]
        oa = torch.empty((1, S, H, D), device="cuda", dtype=torch.bfloat16)
        ob = torch.empty_like(oa)
        gs = []
        for mod, o in ((ma, oa), (mb, ob)):
            call_lib(lib, mod, inp, o)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(10):
                    call_lib(lib, mod, inp, o)
            gs.append(g)
        soak(gs, 3.0)
        sm = timed([("A", gs[0]), ("B", gs[1])], 6, 25)
        a, b = summary(sm["A"]), summary(sm["B"])
        res[f"S{S}"] = {"A_us": a["p50"] / 10 * 1e3, "B_us": b["p50"] / 10 * 1e3,
                        "ratio": summary([y / x for x, y in zip(sm["A"], sm["B"])])}
        log(f"[S{S}] {json.dumps(res[f'S{S}'])}")
    log(f"[ab] {json.dumps(res)}")
    print(json.dumps(res))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["pair", "check", "ab"])
    p.add_argument("--soak-s", type=float, default=10.0)
    p.add_argument("--blocks", type=int, default=16)
    p.add_argument("--block", type=int, default=25)
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--seqs", type=int, nargs="+",
                   default=[4176, 4096, 4100, 4160, 4256, 40, 4224])
    args = p.parse_args()
    try:
        lg = subprocess.run(["git", "log", "--oneline", "-3"], capture_output=True, text=True)
        anc = subprocess.run(["git", "merge-base", "--is-ancestor", "0b4693c4", "HEAD"],
                             capture_output=True, text=True)
        log(f"[ancestry] {lg.stdout.strip()!r}; is-ancestor 0b4693c4 rc={anc.returncode}")
    except Exception as exc:  # noqa: BLE001
        log(f"[ancestry] git unavailable: {exc}")
    log(f"device {torch.cuda.get_device_name()} mode {args.mode}")
    {"pair": mode_pair, "check": mode_check, "ab": mode_ab}[args.mode](args)


if __name__ == "__main__":
    main()
