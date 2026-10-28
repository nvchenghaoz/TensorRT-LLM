"""Idea 116 (branch 77): exact unit-alpha O-rescale skip (FMHA_FLUX3_SKIP_UNIT_ALPHA, lever
'unit_alpha', tag ua) in the routed base-2 fmha_v2 flux3 kernel. Copied from
flux3_b2_graph_ab.py (method 881); the original stays untouched.

One SAMPLE = one CUDA-event-timed replay of a CUDA graph holding the per-pass d128 FP8 set of
FLUX.3 klein at 1024^2: 5 x S=4096 then 28 x S=4176, B=1 H=24 D=128, every call through
tensorrt_llm's run_fmha_v2_fp8_sm12x. Per-call time = graph time / 33.

Modes
  pair  one arm, for run_gpu_pair: the tree's own routing (base -> b2_ms_rq, branch ->
        b2_ms_rq_ua), randn inputs (fp8_sm12x.quantize_qkv_e4m3 on one BF16 buffer, fixed seed).
        12 s warm-soak, then --blocks x --block samples, all kept. Prints the evidence JSON.
  ab    branch tree, one process. Input sets:
          real  : the 33 d128 calls of denoise step --step of ONE eager in-process request
                  (FLASHINFER fp8/fp8 attention, torch.compile off), captured by wrapping
                  flashinfer.run_fmha_v2_fp8_sm12x; the pipeline is freed afterwards.
          randn : as pair mode.
          peaky : randn plus a per-head bias on q and 8 spike keys per head aligned with it.
        For each set: unit-alpha share of (warp, 64-key step) pairs (fp32 torch emulation of the
        raw-accumulator running max); graph A (ROUTED_LEVERS without unit_alpha = main's b2) and
        graph B (with unit_alpha); kernel names by torch.profiler. Warm-soak, then samples
        interleaved per set A,B,..., SM clock per block, per-arm p10/p50/p90, nothing dropped.
        Then the untimed correctness replica (same soak count, interleaving, sample count, reset
        points): every replay bitwise vs that arm's replay 0, B bitwise == A, inputs unchanged.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
import types

import torch

H, D = 24, 128
SEQ = [4096] * 5 + [4176] * 28
K = len(SEQ)
UA = "unit_alpha"


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


# ---------------------------------------------------------------- inputs (list of 33 call dicts)
def _quant(q, k, v):
    from tensorrt_llm._torch.visual_gen.attention.backends.fp8_sm12x import quantize_qkv_e4m3

    return quantize_qkv_e4m3(q, k, v, 1.0 / math.sqrt(D))


def make_synth(kind, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    per_s = {}
    for S in sorted(set(SEQ)):
        buf = torch.randn((3, 1, S, H, D), generator=g, device="cuda", dtype=torch.float32)
        if kind == "peaky":
            bias = torch.randn((H, D), generator=g, device="cuda", dtype=torch.float32)
            buf[0] += 1.5 * bias
            idx = torch.randint(0, S, (8,), generator=g, device="cuda")
            buf[1][0, idx] = 3.0 * bias + 0.3 * buf[1][0, idx]
        buf = buf.to(torch.bfloat16)
        q, k, v = buf.unbind(0)
        q8, k8, v8, sc = _quant(q, k, v)
        per_s[S] = dict(S=S, q=q8, k=k8, v=v8, scales=sc)
    torch.cuda.synchronize()
    return [per_s[S] for S in SEQ]  # shared per S, as the original bench


def capture_real(ckpt, step):
    from tensorrt_llm import VisualGenArgs
    from tensorrt_llm._torch.visual_gen.attention.backends import flashinfer as fib
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
    from tensorrt_llm.visual_gen.args import (AttentionConfig, QuantAttentionConfig,
                                              TorchCompileConfig)
    from tensorrt_llm.visual_gen.params import VisualGenParams

    cfg = VisualGenArgs()
    att = AttentionConfig(backend="FLASHINFER",
                          quant_attention_config=QuantAttentionConfig(qk_dtype="fp8", v_dtype="fp8"))
    cfg = cfg.model_copy(update={"attention_config": att, "torch_compile_config":
                                 TorchCompileConfig(enable=False, enable_autotune=False)})
    cfg.pipeline_config["variant"] = "nvfp4"
    t0 = time.time()
    pipe = PipelineLoader(cfg, device="cuda:0").load(checkpoint_dir=ckpt)
    log(f"[real] pipeline loaded in {time.time() - t0:.0f}s")
    orig = fib.run_fmha_v2_fp8_sm12x
    st = {"n": 0, "keep": [], "other": 0}
    lo, hi = step * K, (step + 1) * K

    def wrapped(q8, k8, v8, out, nh, hd, S, *, scale_bmm1_d, scale_bmm2_d, causal,
                return_lse=False, **kw):
        if hd == 128 and not causal and not return_lse:
            if lo <= st["n"] < hi:
                st["keep"].append(dict(S=int(S), q=q8.clone(), k=k8.clone(), v=v8.clone(),
                                       scales=scale_bmm1_d.new_empty(0)))
                st["keep"][-1]["b1"] = scale_bmm1_d.clone()
                st["keep"][-1]["b2"] = scale_bmm2_d.clone()
            st["n"] += 1
        else:
            st["other"] += 1
        return orig(q8, k8, v8, out, nh, hd, S, scale_bmm1_d=scale_bmm1_d,
                    scale_bmm2_d=scale_bmm2_d, causal=causal, return_lse=return_lse, **kw)

    fib.run_fmha_v2_fp8_sm12x = wrapped
    try:
        params = VisualGenParams(height=1024, width=1024, seed=1000, num_inference_steps=16)
        req = types.SimpleNamespace(prompt="A red fox standing in fresh snow at sunrise, "
                                           "photorealistic", params=params)
        t0 = time.time()
        pipe.infer(req)
        torch.cuda.synchronize()
        log(f"[real] eager request {time.time() - t0:.1f}s; d128 non-causal calls {st['n']} "
            f"(other {st['other']}); kept {len(st['keep'])} from step {step}")
    finally:
        fib.run_fmha_v2_fp8_sm12x = orig
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    calls = st["keep"]
    for c in calls:  # scales layout used by call(): [.., .., b2, b1]
        c["scales"] = torch.cat([c["b1"], c["b1"], c["b2"], c["b1"]]).contiguous()
    info = {"calls_per_request": st["n"], "other_calls": st["other"],
            "seq": [c["S"] for c in calls]}
    return calls, info


# ---------------------------------------------------------------- helpers
def uniq(calls):
    seen, out = set(), []
    for c in calls:
        if id(c) not in seen:
            seen.add(id(c))
            out.append(c)
    return out


def snapshot(calls):
    return [{n: c[n].clone() for n in ("q", "k", "v", "scales")} for c in uniq(calls)]


def restore(calls, snap):
    for c, s in zip(uniq(calls), snap):
        for n in s:
            c[n].copy_(s[n])
    torch.cuda.synchronize()


def unchanged(calls, snap):
    ok = True
    for c, s in zip(uniq(calls), snap):
        for n in s:
            a, b = c[n], s[n]
            if a.dtype == torch.float8_e4m3fn:
                a, b = a.view(torch.uint8), b.view(torch.uint8)
            ok &= torch.equal(a, b)
    return bool(ok)


def call(fib, c, out):
    sc = c["scales"]
    fib.run_fmha_v2_fp8_sm12x(c["q"], c["k"], c["v"], out, H, D, c["S"],
                              scale_bmm1_d=sc[3:4], scale_bmm2_d=sc[2:3], causal=False,
                              return_lse=False)


def make_outputs(calls):
    total = sum(c["S"] for c in calls) * H * D
    flat = torch.empty(total, device="cuda", dtype=torch.bfloat16)
    outs, off = [], 0
    for c in calls:
        n = c["S"] * H * D
        outs.append(flat[off:off + n].view(1, c["S"], H, D))
        off += n
    return flat, outs


def capture(fib, calls):
    flat, outs = make_outputs(calls)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            for c, o in zip(calls, outs):
                call(fib, c, o)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for c, o in zip(calls, outs):
            call(fib, c, o)
    torch.cuda.synchronize()
    return g, flat


def graph_kernels(g):
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        g.replay()
        torch.cuda.synchronize()
    return {e.key: e.count for e in prof.key_averages() if "fmha" in e.key or "flash" in e.key}


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


def summary(xs):
    t = torch.tensor(xs, dtype=torch.float64)
    p10, p50, p90 = (t.quantile(p).item() for p in (0.1, 0.5, 0.9))
    return dict(n=len(xs), p10=p10, p50=p50, p90=p90, noise=(p90 - p10) / p50,
                mean=sum(xs) / len(xs), min=min(xs), max=max(xs))


def timed(arms, nblocks, block, clock):
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


# ---------------------------------------------------------------- unit-alpha share (emulation)
@torch.no_grad()
def unit_alpha_share(calls):
    """Share of (warp of 16 q rows, KV step >= 1) pairs where both alphas of every lane are
    exactly 1.0f, emulating the kernel's running max of raw QK accumulators in fp32 (64 q rows per
    CTA, 4 warps x 16 rows, 64-key steps, padded keys masked). Products of e4m3 values are exact in
    fp32; the sum order differs from the MMA's, so near-ties can differ (estimate)."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    tot_u = tot_n = tot_eq = 0
    per_s = {}
    try:
        for c in uniq(calls):
            S = c["S"]
            s_log2e = float(c["scales"][3].float().item()) * 1.4426950408889634
            nsteps = (S + 63) // 64
            nw = (S + 15) // 16
            u_c = n_c = eq_c = 0
            for h in range(H):
                q = c["q"][0, :, h].float()
                k = c["k"][0, :, h].float()
                sc = q @ k.t()
                pad = nsteps * 64 - S
                if pad:
                    sc = torch.nn.functional.pad(sc, (0, pad), value=float("-inf"))
                m = sc.view(S, nsteps, 64).amax(-1).cummax(-1).values  # running max per step
                alpha = torch.exp2((m[:, :-1] - m[:, 1:]) * s_log2e)
                unit = alpha == 1.0
                eq = m[:, :-1] == m[:, 1:]
                rpad = nw * 16 - S
                if rpad:
                    unit = torch.cat([unit, unit.new_ones(rpad, nsteps - 1)])
                    eq = torch.cat([eq, eq.new_ones(rpad, nsteps - 1)])
                wu = unit.view(nw, 16, nsteps - 1).all(1)
                we = eq.view(nw, 16, nsteps - 1).all(1)
                u_c += int(wu.sum())
                eq_c += int(we.sum())
                n_c += wu.numel()
                del sc
            tot_u += u_c
            tot_n += n_c
            tot_eq += eq_c
            d = per_s.setdefault(str(S), [0, 0])
            d[0] += u_c
            d[1] += n_c
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    return {"share_unit_alpha": tot_u / tot_n, "share_max_unchanged": tot_eq / tot_n,
            "pairs": tot_n, "per_S": {s: v[0] / v[1] for s, v in per_s.items()}}


# ---------------------------------------------------------------- modes
def mode_pair(args):
    from tensorrt_llm._torch.visual_gen.attention.backends import flashinfer as fib
    from tensorrt_llm._torch.visual_gen.attention.backends import fmha_v2_flux3 as lib

    log(f"[tree] ROUTED_LEVERS {lib.ROUTED_LEVERS} tag {lib._tag(lib.ROUTED_LEVERS, 0)}")
    clock = Clock()
    calls = make_synth("randn")
    snap = snapshot(calls)
    g, _ = capture(fib, calls)
    names = graph_kernels(g)
    log(f"[graph] kernels per replay: {json.dumps(names)}")
    restore(calls, snap)
    n_soak = soak([g], args.soak_s)
    log(f"[soak] {n_soak} replays, sm_mhz {clock.sm_mhz()}")
    samples, clocks = timed([("arm", g)], args.blocks, args.block, clock)
    s = summary(samples["arm"])
    ok_inputs = unchanged(calls, snap)
    per_call = s["p50"] / K
    log(f"[result] per-pass p10/p50/p90 {s['p10']:.4f}/{s['p50']:.4f}/{s['p90']:.4f} ms, "
        f"noise {s['noise']:.4f}, per-call {per_call * 1e3:.1f} us, inputs unchanged {ok_inputs}")
    print(json.dumps({"batch": {"1": {"kernel_ms": per_call, "noise_floor": s["noise"]}},
                      "output": {"class": "exact"},
                      "sample": f"one CUDA-event-timed replay of a CUDA graph of {K} calls "
                                f"(5xS4096 + 28xS4176, randn); kernel_ms = per-call = graph/{K}",
                      "routed_levers": list(lib.ROUTED_LEVERS),
                      "per_pass_ms": s, "soak_replays": n_soak, "kernels": names,
                      "clocks": clocks, "inputs_unchanged": ok_inputs}))


def mode_ab(args):
    from tensorrt_llm._torch.visual_gen.attention.backends import flashinfer as fib
    from tensorrt_llm._torch.visual_gen.attention.backends import fmha_v2_flux3 as lib

    failures = []
    lev_b = tuple(lib.ROUTED_LEVERS)
    if UA not in lev_b:
        failures.append(f"ROUTED_LEVERS has no {UA}: {lev_b}")
        lev_b = lev_b + (UA,)
    lev_a = tuple(x for x in lev_b if x != UA)
    tag_a = "fmha_v2_flux3_" + lib._tag(lev_a, 0) + "_"
    tag_b = "fmha_v2_flux3_" + lib._tag(lev_b, 0) + "_"
    log(f"[levers] A {lev_a} ({tag_a}) B {lev_b} ({tag_b})")
    sets = {}
    if not args.no_real:
        real, info = capture_real(args.ckpt, args.step)
        log(f"[real] {json.dumps(info)}")
        if [c["S"] for c in real] != SEQ:
            failures.append(f"real step-{args.step} calls do not match SEQ: {info['seq']}")
        if real:
            sets["real"] = real
    sets["randn"] = make_synth("randn", seed=1)
    sets["peaky"] = make_synth("peaky", seed=2)
    clock = Clock()
    shares = {}
    for name, calls in sets.items():
        t0 = time.time()
        shares[name] = unit_alpha_share(calls)
        log(f"[share {name}] {json.dumps(shares[name])} ({time.time() - t0:.0f}s)")
    snaps = {n: snapshot(c) for n, c in sets.items()}
    graphs, flats, kern = {}, {}, {}
    for name, calls in sets.items():
        for arm, lev in (("A", lev_a), ("B", lev_b)):
            lib.ROUTED_LEVERS = lev  # flashinfer.py reads it at call (capture) time
            g, flat = capture(fib, calls)
            graphs[(name, arm)], flats[(name, arm)] = g, flat
            kern[f"{name}/{arm}"] = graph_kernels(g)
    lib.ROUTED_LEVERS = lev_b
    for key, ks in kern.items():
        want = tag_b if key.endswith("/B") else tag_a
        cnt = sum(c for n, c in ks.items() if want in n)
        if cnt != K or sum(ks.values()) != K:
            failures.append(f"graph {key} does not hold {K} {want} kernels: {ks}")
    log(f"[kernels] {json.dumps(kern)}")
    order = [(n, a) for n in sets for a in ("A", "B")]
    arms = [(f"{n}/{a}", graphs[(n, a)]) for n, a in order]
    # ---- timed ----
    for n in sets:
        restore(sets[n], snaps[n])
    n_soak = soak([g for _, g in arms], args.soak_s)
    log(f"[soak] {n_soak} rounds of {len(arms)} replays, sm_mhz {clock.sm_mhz()}")
    samples, clocks = timed(arms, args.blocks, args.block, clock)
    stats = {}
    for n in sets:
        sA, sB = summary(samples[f"{n}/A"]), summary(samples[f"{n}/B"])
        ratio = summary([b / a for a, b in zip(samples[f"{n}/A"], samples[f"{n}/B"])])
        stats[n] = {"A_per_pass_ms": sA, "B_per_pass_ms": sB, "B_over_A_p50": sB["p50"] / sA["p50"] - 1,
                    "paired_ratio": ratio,
                    "per_call_us": {"A": sA["p50"] / K * 1e3, "B": sB["p50"] / K * 1e3}}
        log(f"[{n}] A p10/p50/p90 {sA['p10']:.4f}/{sA['p50']:.4f}/{sA['p90']:.4f} | "
            f"B {sB['p10']:.4f}/{sB['p50']:.4f}/{sB['p90']:.4f} | B/A p50 "
            f"{sB['p50'] / sA['p50'] - 1:+.4%} paired {ratio['p10']:.4f}/{ratio['p50']:.4f}/"
            f"{ratio['p90']:.4f}")
        if not unchanged(sets[n], snaps[n]):
            failures.append(f"{n}: inputs changed during timing")
    # ---- untimed correctness replica ----
    for n in sets:
        restore(sets[n], snaps[n])
    soak_count([g for _, g in arms], n_soak)
    ref = {}
    nrep = 0
    for b in range(args.blocks):
        for _ in range(args.block):
            for key, g in arms:
                g.replay()
                fl = flats[tuple(key.split("/"))]
                if key not in ref:
                    ref[key] = fl.clone()
                elif not torch.equal(fl, ref[key]):
                    failures.append(f"{key} replay {nrep} differs from replay 0")
            nrep += 1
        torch.cuda.synchronize()
        if len(failures) > 20:
            break
    bitwise = {}
    for n in sets:
        eq = torch.equal(ref[f"{n}/A"], ref[f"{n}/B"])
        bitwise[n] = bool(eq)
        if not eq:
            d = (ref[f"{n}/A"].float() - ref[f"{n}/B"].float()).abs()
            failures.append(f"{n}: B != A bitwise (max abs {d.max().item():.3g}, "
                            f"{int((d > 0).sum())} elts)")
        if not unchanged(sets[n], snaps[n]):
            failures.append(f"{n}: inputs changed during the correctness sequence")
    log(f"[check] {nrep} replays/arm after {n_soak} soak rounds; B==A bitwise {bitwise}; "
        f"failures {len(failures)}")
    res = {"passed": not failures, "failures": failures[:20],
           "compared": "per set (real step-8 / randn / peaky, S=4096x5 + 4176x28): every graph "
                       "replay bitwise vs its replay 0 over an untimed replica of the timed "
                       "sequence (same soak count, interleaving, sample count, pristine inputs "
                       "restored first); B (b2+unit_alpha) bitwise == A (main's b2); inputs "
                       "unchanged",
           "levers": {"A": list(lev_a), "B": list(lev_b)}, "kernels": kern,
           "unit_alpha_share": shares, "bitwise_B_eq_A": bitwise, "stats": stats,
           "soak_rounds": n_soak, "clocks": clocks}
    print(json.dumps(res))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["pair", "ab"])
    p.add_argument("--ckpt", default=None)
    p.add_argument("--step", type=int, default=8)
    p.add_argument("--no-real", action="store_true")
    p.add_argument("--soak-s", type=float, default=12.0)
    p.add_argument("--blocks", type=int, default=20)
    p.add_argument("--block", type=int, default=25)
    args = p.parse_args()
    log(f"device {torch.cuda.get_device_name()} K={K} calls/graph pid {os.getpid()}")
    {"pair": mode_pair, "ab": mode_ab}[args.mode](args)


if __name__ == "__main__":
    main()
