"""Idea 110 (branch 55, finishing branch 45): CUDA-graph-replay kernel A/B of the routed base-2
fmha_v2 variant (fmha_v2_flux3_b2_ms_rq_*) vs main's ms_rq, per top lead ruling 2096.

One SAMPLE = one CUDA-event-timed replay of a CUDA graph holding the full per-pass d128 FP8 set
of FLUX.3 klein at 1024^2: 5 x S=4096 (image ModeBlocks) then 28 x S=4176 (single-stream joint),
B=1 H=24 D=128, every call through tensorrt_llm's run_fmha_v2_fp8_sm12x (the helper
trtllm::visual_gen_fp8_attention_sm120 calls). Per-call time = graph time / 33.
Inputs: E4M3 Q/K/V + device scales from main's K1/K2 (fp8_sm12x.quantize_qkv_e4m3) on the
q/k/v planes of ONE BF16 buffer, fixed seed; pristine copies restored before timing and before
the correctness pass. Clocks are never locked; the SM clock is read (NVML, read-only) per block.

Modes
  pair   one arm, for run_gpu_pair: the tree's own routing (base -> ms_rq, branch -> b2).
         >=10 s warm-soak, then --blocks x --block samples, all kept. noise_floor =
         (p90 - p10) / median over ALL samples. Prints the evidence JSON.
  ab     branch tree, one process: graph A (route OFF, _USE_FMHA_V2_FLUX3_B2=False at capture)
         and graph B (route ON). Kernel names of each graph by torch.profiler on a replay.
         >=10 s warm-soak of both, then samples interleaved ABAB..., SM clock per block,
         per-arm p10/p50/p90. Then the untimed correctness sequence with the same soak count,
         the same interleaving and sample count and the same reset points: every replay of each
         arm bitwise equal to that arm's replay 0, arm B within check 784's tolerance of arm A
         (vs fp32 SDPA: rel L2 <= 1.05x A's, max abs <= 2x A's), inputs unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import torch

H, D = 24, 128
SEQ = [4096] * 5 + [4176] * 28
K = len(SEQ)


def log(*a):
    print(*a, flush=True)


class Clock:
    def __init__(self):
        self.h = None
        try:
            import pynvml

            pynvml.nvmlInit()
            uuid = str(torch.cuda.get_device_properties(0).uuid)
            for cand in (uuid, "GPU-" + uuid):
                try:
                    self.h = pynvml.nvmlDeviceGetHandleByUUID(cand)
                    break
                except Exception:  # noqa: BLE001
                    continue
            self.nv = pynvml
        except Exception as exc:  # noqa: BLE001
            log(f"[clock] NVML unavailable: {exc}")

    def sm_mhz(self):
        if self.h is None:
            return None
        try:
            return self.nv.nvmlDeviceGetClockInfo(self.h, self.nv.NVML_CLOCK_SM)
        except Exception:  # noqa: BLE001
            return None


def make_inputs(seed=1):
    from tensorrt_llm._torch.visual_gen.attention.backends.fp8_sm12x import quantize_qkv_e4m3

    g = torch.Generator(device="cuda").manual_seed(seed)
    out = {}
    for S in sorted(set(SEQ)):
        buf = torch.randn((3, 1, S, H, D), generator=g, device="cuda", dtype=torch.float32)
        buf = buf.to(torch.bfloat16)
        q, k, v = buf.unbind(0)
        q8, k8, v8, scales = quantize_qkv_e4m3(q, k, v, 1.0 / math.sqrt(D))
        out[S] = dict(S=S, q=q8, k=k8, v=v8, scales=scales, bf16=buf)
    torch.cuda.synchronize()
    return out


def snapshot(inputs):
    return {S: {n: inputs[S][n].clone() for n in ("q", "k", "v", "scales")} for S in inputs}


def restore(inputs, snap):
    for S in inputs:
        for n in ("q", "k", "v", "scales"):
            inputs[S][n].copy_(snap[S][n])
    torch.cuda.synchronize()


def unchanged(inputs, snap):
    ok = True
    for S in inputs:
        for n in ("q", "k", "v", "scales"):
            a, b = inputs[S][n], snap[S][n]
            if a.dtype == torch.float8_e4m3fn:
                a, b = a.view(torch.uint8), b.view(torch.uint8)
            ok &= torch.equal(a, b)
    return ok


def call(fib, inp, out):
    sc = inp["scales"]
    fib.run_fmha_v2_fp8_sm12x(inp["q"], inp["k"], inp["v"], out, H, D, inp["S"],
                              scale_bmm1_d=sc[3:4], scale_bmm2_d=sc[2:3], causal=False,
                              return_lse=False)


def make_outputs():
    total = sum(SEQ) * H * D
    flat = torch.empty(total, device="cuda", dtype=torch.bfloat16)
    outs, off = [], 0
    for S in SEQ:
        n = S * H * D
        outs.append(flat[off:off + n].view(1, S, H, D))
        off += n
    return flat, outs


def capture(fib, inputs, route_on):
    """Capture the 33-call pass with the route flag as given (flag read at capture time)."""
    if hasattr(fib, "_USE_FMHA_V2_FLUX3_B2"):
        fib._USE_FMHA_V2_FLUX3_B2 = route_on
    flat, outs = make_outputs()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):  # JIT builds / first-call setup outside capture
            for S, o in zip(SEQ, outs):
                call(fib, inputs[S], o)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for S, o in zip(SEQ, outs):
            call(fib, inputs[S], o)
    torch.cuda.synchronize()
    return g, flat, outs


def graph_kernels(g):
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        g.replay()
        torch.cuda.synchronize()
    names = {}
    for e in prof.key_averages():
        if "fmha" in e.key or "flash" in e.key:
            names[e.key] = e.count
    return names


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


def soak_count(graphs, n):
    for i in range(n):
        for g in graphs:
            g.replay()
        if (i + 1) % 20 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()


def pct(xs, p):
    t = torch.tensor(xs, dtype=torch.float64)
    return t.quantile(p).item()


def summary(xs):
    p10, p50, p90 = pct(xs, 0.1), pct(xs, 0.5), pct(xs, 0.9)
    return dict(n=len(xs), p10=p10, p50=p50, p90=p90, noise=(p90 - p10) / p50,
                mean=sum(xs) / len(xs), min=min(xs), max=max(xs))


def timed(arms, nblocks, block, clock):
    """arms: list of (name, graph). Interleaved per sample in the listed order, all samples kept."""
    samples = {n: [] for n, _ in arms}
    clocks = []
    for b in range(nblocks):
        c0 = clock.sm_mhz()
        ev = []
        for _ in range(block):
            for n, g in arms:
                a = torch.cuda.Event(enable_timing=True)
                z = torch.cuda.Event(enable_timing=True)
                a.record()
                g.replay()
                z.record()
                ev.append((n, a, z))
        torch.cuda.synchronize()
        c1 = clock.sm_mhz()
        blk = {n: [] for n, _ in arms}
        for n, a, z in ev:
            blk[n].append(a.elapsed_time(z))
        for n in blk:
            samples[n] += blk[n]
        med = {n: round(sorted(v)[len(v) // 2], 4) for n, v in blk.items()}
        clocks.append(dict(block=b, sm_mhz_start=c0, sm_mhz_end=c1, block_median_ms=med))
        log(f"[block {b + 1}/{nblocks}] sm_mhz {c0}->{c1} median ms {med}")
    return samples, clocks


def fp32_ref(inp, S):
    sc = inp["scales"]
    q = inp["q"].float() * sc[0]
    k = inp["k"].float() * sc[1]
    v = inp["v"].float() * sc[2]
    o = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=1.0 / math.sqrt(D))
    return o.transpose(1, 2)


def err(o, ref):
    d = o.float() - ref
    return (d.norm() / ref.norm()).item(), d.abs().max().item()


def mode_pair(args):
    from tensorrt_llm._torch.visual_gen.attention.backends import flashinfer as fib

    log(f"[tree] has route flag: {hasattr(fib, '_USE_FMHA_V2_FLUX3_B2')} "
        f"value {getattr(fib, '_USE_FMHA_V2_FLUX3_B2', None)}")
    clock = Clock()
    inputs = make_inputs()
    snap = snapshot(inputs)
    g, flat, _ = capture(fib, inputs, route_on=True)
    names = graph_kernels(g)
    log(f"[graph] kernels per replay: {json.dumps(names)}")
    restore(inputs, snap)
    n_soak = soak([g], args.soak_s)
    log(f"[soak] {n_soak} replays in >= {args.soak_s}s, sm_mhz {clock.sm_mhz()}")
    samples, clocks = timed([("arm", g)], args.blocks, args.block, clock)
    xs = samples["arm"]
    s = summary(xs)
    ok_inputs = unchanged(inputs, snap)
    per_call = s["p50"] / K
    log(f"[result] per-pass p10/p50/p90 {s['p10']:.4f}/{s['p50']:.4f}/{s['p90']:.4f} ms, "
        f"noise {s['noise']:.4f}, per-call {per_call * 1e3:.1f} us, inputs unchanged {ok_inputs}")
    print(json.dumps({"batch": {"1": {"kernel_ms": per_call, "noise_floor": s["noise"]}},
                      "output": {"class": "reordered"},
                      "sample": f"one CUDA-event-timed replay of a CUDA graph of {K} calls "
                                f"(5xS4096 + 28xS4176); kernel_ms = per-call = graph/{K}",
                      "per_pass_ms": s, "soak_replays": n_soak, "kernels": names,
                      "clocks": clocks, "inputs_unchanged": ok_inputs}))


def mode_ab(args):
    from tensorrt_llm._torch.visual_gen.attention.backends import flashinfer as fib
    from tensorrt_llm._torch.visual_gen.attention.backends import fmha_v2_flux3 as lib

    assert hasattr(fib, "_USE_FMHA_V2_FLUX3_B2"), "ab mode needs the branch tree"
    routed = "fmha_v2_flux3_" + lib._tag(lib.ROUTED_LEVERS, 0) + "_"
    msrq = "fmha_v2_flux3_" + lib._tag(lib.DEFAULT_LEVERS, 0) + "_"
    clock = Clock()
    inputs = make_inputs()
    snap = snapshot(inputs)
    gA, flatA, outsA = capture(fib, inputs, route_on=False)
    gB, flatB, outsB = capture(fib, inputs, route_on=True)
    fib._USE_FMHA_V2_FLUX3_B2 = True
    kA, kB = graph_kernels(gA), graph_kernels(gB)
    log(f"[graph A] {json.dumps(kA)}")
    log(f"[graph B] {json.dumps(kB)}")
    failures = []
    cntA = sum(c for n, c in kA.items() if msrq in n and routed not in n)
    cntB = sum(c for n, c in kB.items() if routed in n)
    if cntA != K or any(routed in n for n in kA):
        failures.append(f"graph A does not hold {K} ms_rq kernels: {kA}")
    if cntB != K or any(msrq in n and routed not in n for n in kB):
        failures.append(f"graph B does not hold {K} b2 kernels: {kB}")
    arms = [("A_msrq", gA), ("B_b2", gB)]
    # ---- timed ----
    restore(inputs, snap)
    n_soak = soak([gA, gB], args.soak_s)
    log(f"[soak] {n_soak} AB replays in >= {args.soak_s}s, sm_mhz {clock.sm_mhz()}")
    samples, clocks = timed(arms, args.blocks, args.block, clock)
    sA, sB = summary(samples["A_msrq"]), summary(samples["B_b2"])
    paired = [b / a - 1 for a, b in zip(samples["A_msrq"], samples["B_b2"])]
    sP = summary([1 + x for x in paired])
    log(f"[A ms_rq] per-pass p10/p50/p90 {sA['p10']:.4f}/{sA['p50']:.4f}/{sA['p90']:.4f} "
        f"noise {sA['noise']:.4f}")
    log(f"[B b2   ] per-pass p10/p50/p90 {sB['p10']:.4f}/{sB['p50']:.4f}/{sB['p90']:.4f} "
        f"noise {sB['noise']:.4f}")
    log(f"[B/A] p50 {sB['p50'] / sA['p50'] - 1:+.4%}; paired-ratio p10/p50/p90 "
        f"{sP['p10']:.4f}/{sP['p50']:.4f}/{sP['p90']:.4f}")
    if not unchanged(inputs, snap):
        failures.append("inputs changed during timing")
    # ---- untimed correctness, same sequence ----
    restore(inputs, snap)
    soak_count([gA, gB], n_soak)
    ref = {}
    nrep = 0
    for b in range(args.blocks):
        for _ in range(args.block):
            for n, g, flat in (("A", gA, flatA), ("B", gB, flatB)):
                g.replay()
                if n not in ref:
                    ref[n] = flat.clone()
                elif not torch.equal(flat, ref[n]):
                    failures.append(f"arm {n} replay {nrep} differs from replay 0")
            nrep += 1
        torch.cuda.synchronize()
        if len(failures) > 20:
            break
    log(f"[check] {nrep} replays/arm after {n_soak} soak replays, failures {len(failures)}")
    if not unchanged(inputs, snap):
        failures.append("inputs changed during the correctness sequence")
    # arm B vs arm A within check 784's tolerance (relative to fp32 SDPA)
    tol = {}
    _, oA = make_outputs()
    for idx in (0, 5):  # first S=4096 call, first S=4176 call
        S = SEQ[idx]
        r32 = fp32_ref(inputs[S], S)
        offs = sum(SEQ[:idx]) * H * D
        n = S * H * D
        a = ref["A"][offs:offs + n].view(1, S, H, D)
        bb = ref["B"][offs:offs + n].view(1, S, H, D)
        eA, eB = err(a, r32), err(bb, r32)
        dAB = (bb.float() - a.float())
        ok = eB[0] <= 1.05 * eA[0] and eB[1] <= 2.0 * eA[1]
        tol[str(S)] = dict(A_vs_fp32=eA, B_vs_fp32=eB, rel_l2_ratio=eB[0] / eA[0],
                           max_abs_ratio=eB[1] / eA[1],
                           B_vs_A_rel_l2=(dAB.norm() / a.float().norm()).item(),
                           B_vs_A_max_abs=dAB.abs().max().item(), ok=ok)
        if not ok:
            failures.append(f"tolerance S={S}")
    log(f"[tolerance] {json.dumps(tol)}")
    res = {"passed": not failures, "failures": failures[:20],
           "compared": "per arm: every graph replay bitwise vs replay 0 (same soak count, "
                       "interleaving and sample count as the timed run, pristine inputs restored "
                       "first); arm B vs arm A via fp32 SDPA reference (check 784 tolerance: "
                       "rel L2 <= 1.05x, max abs <= 2x of arm A); inputs unchanged",
           "kernels": {"A": kA, "B": kB},
           "A_msrq_per_pass_ms": sA, "B_b2_per_pass_ms": sB,
           "B_over_A_p50": sB["p50"] / sA["p50"] - 1, "paired_ratio": sP,
           "per_call_us": {"A": sA["p50"] / K * 1e3, "B": sB["p50"] / K * 1e3},
           "soak_replays": n_soak, "clocks": clocks, "tolerance": tol}
    print(json.dumps(res))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["pair", "ab"])
    p.add_argument("--soak-s", type=float, default=12.0)
    p.add_argument("--blocks", type=int, default=20)
    p.add_argument("--block", type=int, default=25)
    args = p.parse_args()
    log(f"device {torch.cuda.get_device_name()} K={K} calls/graph")
    {"pair": mode_pair, "ab": mode_ab}[args.mode](args)


if __name__ == "__main__":
    main()
