"""Idea 128 (branch 80): v_ldsm_x4 (tag vx4) on the routed b2 fmha_v2 flux3 kernel.

One V fragment load of BMM2 per (ni, ki) becomes one LDSM.MT88.4 instead of two LDSM.MT88.2
(smem_tile_v.h overlay, macro FMHA_FLUX3_V_LDSM_X4). Exact: bitwise equal to b2.

Sample design (ruling 2096 / method 881): one SAMPLE = one CUDA-event-timed replay of a CUDA graph
holding the per-pass d128 FP8 set (5 x S=4096 + 28 x S=4176, B=1 H=24 D=128) through
run_fmha_v2_fp8_sm12x; per-call = graph/33. Inputs from fp8_sm12x.quantize_qkv_e4m3 on the q/k/v
planes of one BF16 buffer, fixed seed, pristine copies restored before timing and correctness.

Modes
  pair   one arm for run_gpu_pair: the tree's own routing (base -> b2_ms_rq, branch ->
         b2_ms_rq_vx4). Also prints sha256 of ms_rq (causal + LSE) outputs on fixed inputs, so the
         two arms show the ms_rq module unchanged.
  ab     branch tree, one process: arm A = b2 (ROUTED_LEVERS without v_ldsm_x4), arm B = vx4.
         Warm-soak, ABAB interleaved samples, SM clock per block, per-arm p10/p50/p90, no samples
         dropped. Then the untimed replica (same soak count, interleave, sample count, reset
         points): every replay of B bitwise equal to A's replay 0 and its own replay 0, inputs
         unchanged. Then bitwise B==A per call at S=4176/4096/4256/4100 on randn and peaky inputs,
         cuobjdump LDSM/register counts of both kernels, and the route check (default env ->
         vx4; TRTLLM_FMHA_V2_FLUX3_VX4=0 subprocess -> b2_ms_rq).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
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


def quant(S, seed, peaky=False):
    from tensorrt_llm._torch.visual_gen.attention.backends.fp8_sm12x import quantize_qkv_e4m3

    g = torch.Generator(device="cuda").manual_seed(seed)
    buf = torch.randn((3, 1, S, H, D), generator=g, device="cuda", dtype=torch.float32)
    if peaky:
        # A few large keys/values per head: sharp softmax rows and wide V range.
        buf[0] *= 4.0
        idx = torch.randint(0, S, (32,), generator=g, device="cuda")
        buf[1][:, idx] *= 8.0
        buf[2][:, idx] *= 16.0
    buf = buf.to(torch.bfloat16)
    q, k, v = buf.unbind(0)
    q8, k8, v8, scales = quantize_qkv_e4m3(q, k, v, 1.0 / math.sqrt(D))
    return dict(S=S, q=q8, k=k8, v=v8, scales=scales)


def make_inputs(seed=1):
    out = {S: quant(S, seed + S) for S in sorted(set(SEQ))}
    torch.cuda.synchronize()
    return out


NAMES = ("q", "k", "v", "scales")


def snapshot(inputs):
    return {S: {n: inputs[S][n].clone() for n in NAMES} for S in inputs}


def restore(inputs, snap):
    for S in inputs:
        for n in NAMES:
            inputs[S][n].copy_(snap[S][n])
    torch.cuda.synchronize()


def u8(t):
    return t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t


def unchanged(inputs, snap):
    return all(torch.equal(u8(inputs[S][n]), u8(snap[S][n])) for S in inputs for n in NAMES)


def call(fib, inp, out):
    sc = inp["scales"]
    fib.run_fmha_v2_fp8_sm12x(inp["q"], inp["k"], inp["v"], out, H, D, inp["S"],
                              scale_bmm1_d=sc[3:4], scale_bmm2_d=sc[2:3], causal=False,
                              return_lse=False)


def make_outputs():
    flat = torch.empty(sum(SEQ) * H * D, device="cuda", dtype=torch.bfloat16)
    outs, off = [], 0
    for S in SEQ:
        n = S * H * D
        outs.append(flat[off:off + n].view(1, S, H, D))
        off += n
    return flat, outs


def capture(fib, lib, inputs, levers):
    """Capture the 33-call pass with lib.ROUTED_LEVERS = levers (read at call time)."""
    if lib is not None and levers is not None:
        lib.ROUTED_LEVERS = levers
    flat, outs = make_outputs()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            for S, o in zip(SEQ, outs):
                call(fib, inputs[S], o)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for S, o in zip(SEQ, outs):
            call(fib, inputs[S], o)
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


def msrq_hashes(lib):
    """sha256 of the exact ms_rq module's causal and LSE outputs (paths vx4 must not touch)."""
    mod = lib.get_flux3_d128_module(lib.DEFAULT_LEVERS)
    res = {}
    for S in (4176, 1100):
        inp = quant(S, 77 + S)
        sc = inp["scales"]
        for causal in (False, True):
            out = torch.empty(1, S, H, D, device="cuda", dtype=torch.bfloat16)
            lse = torch.empty(1, S, H, 2, device="cuda", dtype=torch.float32)
            lib.fmha_v2_flux3_d128(inp["q"], inp["k"], inp["v"], out, H, D, S,
                                   scale_bmm1_d=sc[3:4], scale_bmm2_d=sc[2:3], causal=causal,
                                   lse=lse, module=mod)
            torch.cuda.synchronize()
            h = hashlib.sha256(out.view(torch.uint8).cpu().numpy().tobytes())
            h.update(lse.cpu().numpy().tobytes())
            res[f"S{S}_causal{int(causal)}_lse"] = h.hexdigest()[:16]
    return res


def mode_pair(args):
    from tensorrt_llm._torch.visual_gen.attention.backends import flashinfer as fib
    from tensorrt_llm._torch.visual_gen.attention.backends import fmha_v2_flux3 as lib

    log(f"[tree] ROUTED_LEVERS {lib.ROUTED_LEVERS}")
    clock = Clock()
    inputs = make_inputs()
    snap = snapshot(inputs)
    g, flat = capture(fib, None, inputs, None)
    names = graph_kernels(g)
    log(f"[graph] kernels per replay: {json.dumps(names)}")
    restore(inputs, snap)
    n_soak = soak([g], args.soak_s)
    log(f"[soak] {n_soak} replays in >= {args.soak_s}s, sm_mhz {clock.sm_mhz()}")
    samples, clocks = timed([("arm", g)], args.blocks, args.block, clock)
    s = summary(samples["arm"])
    ok_inputs = unchanged(inputs, snap)
    out_hash = hashlib.sha256(flat.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]
    try:
        mh = msrq_hashes(lib)
    except Exception as exc:  # noqa: BLE001
        mh = {"error": repr(exc)[:300]}
    per_call = s["p50"] / K
    log(f"[result] per-pass p10/p50/p90 {s['p10']:.4f}/{s['p50']:.4f}/{s['p90']:.4f} ms, "
        f"noise {s['noise']:.4f}, per-call {per_call * 1e3:.1f} us, inputs unchanged {ok_inputs}")
    print(json.dumps({"batch": {"1": {"kernel_ms": per_call, "noise_floor": s["noise"]}},
                      "output": {"class": "exact"},
                      "sample": f"one CUDA-event-timed replay of a CUDA graph of {K} calls "
                                f"(5xS4096 + 28xS4176); kernel_ms = per-call = graph/{K}",
                      "routed_levers": list(lib.ROUTED_LEVERS),
                      "per_pass_ms": s, "soak_replays": n_soak, "kernels": names,
                      "routed_output_sha256": out_hash, "msrq_output_sha256": mh,
                      "clocks": clocks, "inputs_unchanged": ok_inputs}))


def sass_stats(lib, levers):
    path = lib.gen_flux3_d128_spec(levers).get_library_path()
    kname = lib.flux3_d128_kernel_name(levers)
    try:
        sass = subprocess.run(["cuobjdump", "-sass", str(path)], capture_output=True, text=True,
                              timeout=300).stdout
        res_txt = subprocess.run(["cuobjdump", "-res-usage", str(path)], capture_output=True,
                                 text=True, timeout=300).stdout
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)[:300]}
    # SASS of the d128 kernel(s) named kname (one entry per template instance in the .so)
    blocks = re.split(r"\n\s*Function : ", sass)
    st = {}
    for blk in blocks[1:]:
        fn = blk.split("\n", 1)[0].strip()
        if kname not in fn and "flux3" not in fn:
            continue
        cnt = {k: len(re.findall(re.escape(k) + r"\b", blk)) for k in
               ("LDSM.16.MT88.2", "LDSM.16.MT88.4", "LDSM.16.M88.4", "LDSM.16.M88.2")}
        cnt["STL"] = len(re.findall(r"\bSTL", blk))
        cnt["LDL"] = len(re.findall(r"\bLDL", blk))
        cnt["QMMA_or_HMMA"] = len(re.findall(r"\b(?:QMMA|HMMA|OMMA)", blk))
        st[fn[:160]] = cnt
    regs = {}
    for m in re.finditer(r"Function ([^\s:]+):\s*\n\s*REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+"
                         r"LOCAL:(\d+)", res_txt):
        if "flux3" in m.group(1):
            regs[m.group(1)[:160]] = dict(reg=int(m.group(2)), stack=int(m.group(3)),
                                          shared=int(m.group(4)), local=int(m.group(5)))
    return {"so": str(path), "sass_counts": st, "res_usage": regs}


def occupancy(lib, levers, smem_hint=None):
    """CTAs/SM via the CUDA occupancy API on the compiled kernel."""
    try:
        from cuda.bindings import driver as cu  # noqa: F401
    except Exception:
        cu = None
    return None if cu is None else "see res_usage (smem + regs)"


def bitwise_shapes(lib, levers_a, levers_b):
    ma = lib.get_flux3_d128_module(levers_a)
    mb = lib.get_flux3_d128_module(levers_b)
    res, ok = {}, True
    for peaky in (False, True):
        for S in (4176, 4096, 4256, 4100):
            inp = quant(S, 1000 + S + 7 * int(peaky), peaky=peaky)
            snap = {n: inp[n].clone() for n in NAMES}
            sc = inp["scales"]
            outs = []
            for m in (ma, mb, ma, mb):
                o = torch.full((1, S, H, D), float("nan"), device="cuda", dtype=torch.bfloat16)
                lib.fmha_v2_flux3_d128(inp["q"], inp["k"], inp["v"], o, H, D, S,
                                       scale_bmm1_d=sc[3:4], scale_bmm2_d=sc[2:3], causal=False,
                                       lse=None, module=m)
                outs.append(o)
            torch.cuda.synchronize()
            eq = all(torch.equal(outs[0].view(torch.int16), x.view(torch.int16)) for x in outs[1:])
            same_in = all(torch.equal(u8(inp[n]), u8(snap[n])) for n in NAMES)
            finite = bool(torch.isfinite(outs[1]).all())
            key = f"{'peaky' if peaky else 'randn'}_S{S}"
            res[key] = dict(bitwise=eq, inputs_unchanged=same_in, finite=finite,
                            max_abs_diff=(outs[1].float() - outs[0].float()).abs().max().item())
            ok &= eq and same_in and finite
            log(f"[bitwise] {key}: {res[key]}")
    return ok, res


ROUTE_SNIPPET = r"""
import json, math, torch
from tensorrt_llm._torch.visual_gen.attention.backends import flashinfer as fib
from tensorrt_llm._torch.visual_gen.attention.backends import fmha_v2_flux3 as lib
from tensorrt_llm._torch.visual_gen.attention.backends.fp8_sm12x import quantize_qkv_e4m3
from torch.profiler import ProfilerActivity, profile
S, H, D = 4176, 24, 128
buf = torch.randn((3, 1, S, H, D), device="cuda").to(torch.bfloat16)
q8, k8, v8, sc = quantize_qkv_e4m3(*buf.unbind(0), 1.0 / math.sqrt(D))
out = torch.empty(1, S, H, D, device="cuda", dtype=torch.bfloat16)
f = lambda: fib.run_fmha_v2_fp8_sm12x(q8, k8, v8, out, H, D, S, scale_bmm1_d=sc[3:4],
                                      scale_bmm2_d=sc[2:3], causal=False, return_lse=False)
f(); torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    f(); torch.cuda.synchronize()
names = [e.key for e in prof.key_averages() if "fmha" in e.key]
print(json.dumps({"routed_levers": list(lib.ROUTED_LEVERS), "kernels": names}))
"""


def route_check():
    res = {}
    for label, val in (("default", None), ("VX4=0", "0")):
        env = dict(os.environ)
        env.pop("TRTLLM_FMHA_V2_FLUX3_VX4", None)
        if val is not None:
            env["TRTLLM_FMHA_V2_FLUX3_VX4"] = val
        p = subprocess.run([sys.executable, "-c", ROUTE_SNIPPET], capture_output=True, text=True,
                           env=env, timeout=900)
        last = (p.stdout.strip().splitlines() or ["{}"])[-1]
        try:
            res[label] = json.loads(last)
        except Exception:  # noqa: BLE001
            res[label] = {"error": p.stderr[-800:]}
        log(f"[route] {label}: {res[label]}")
    d = res.get("default", {}).get("kernels", [])
    z = res.get("VX4=0", {}).get("kernels", [])
    ok = (len(d) == 1 and "fmha_v2_flux3_b2_ms_rq_vx4_" in d[0]
          and len(z) == 1 and "fmha_v2_flux3_b2_ms_rq_e4m3" in z[0])
    return ok, res


def mode_ab(args):
    from tensorrt_llm._torch.visual_gen.attention.backends import flashinfer as fib
    from tensorrt_llm._torch.visual_gen.attention.backends import fmha_v2_flux3 as lib

    lev_b = tuple(lib.ROUTED_LEVERS)
    lev_a = tuple(x for x in lev_b if x != "v_ldsm_x4")
    assert "v_ldsm_x4" in lev_b, f"branch tree must route vx4 by default: {lev_b}"
    failures = []
    clock = Clock()
    # ---- 1. SASS ----
    for lev in (lev_a, lev_b):
        lib.get_flux3_d128_module(lev)
    sass = {"A_b2": sass_stats(lib, lev_a), "B_vx4": sass_stats(lib, lev_b)}
    log(f"[sass] {json.dumps(sass)}")
    # ---- 2. bitwise at 4 shapes, randn + peaky ----
    ok_bw, bw = bitwise_shapes(lib, lev_a, lev_b)
    if not ok_bw:
        failures.append("bitwise B==A failed at some shape")
    # ---- 3. graph A/B ----
    inputs = make_inputs()
    snap = snapshot(inputs)
    gA, flatA = capture(fib, lib, inputs, lev_a)
    gB, flatB = capture(fib, lib, inputs, lev_b)
    lib.ROUTED_LEVERS = lev_b
    kA, kB = graph_kernels(gA), graph_kernels(gB)
    log(f"[graph A] {json.dumps(kA)}")
    log(f"[graph B] {json.dumps(kB)}")
    tagA = "fmha_v2_flux3_" + lib._tag(lev_a, 0) + "_e4m3"
    tagB = "fmha_v2_flux3_" + lib._tag(lev_b, 0) + "_e4m3"
    if sum(c for n, c in kA.items() if tagA in n) != K or len(kA) != 1:
        failures.append(f"graph A does not hold {K} b2 kernels: {kA}")
    if sum(c for n, c in kB.items() if tagB in n) != K or len(kB) != 1:
        failures.append(f"graph B does not hold {K} vx4 kernels: {kB}")
    arms = [("A_b2", gA), ("B_vx4", gB)]
    restore(inputs, snap)
    n_soak = soak([gA, gB], args.soak_s)
    log(f"[soak] {n_soak} AB replays in >= {args.soak_s}s, sm_mhz {clock.sm_mhz()}")
    samples, clocks = timed(arms, args.blocks, args.block, clock)
    sA, sB = summary(samples["A_b2"]), summary(samples["B_vx4"])
    sP = summary([b / a for a, b in zip(samples["A_b2"], samples["B_vx4"])])
    log(f"[A b2 ] per-pass p10/p50/p90 {sA['p10']:.4f}/{sA['p50']:.4f}/{sA['p90']:.4f} "
        f"noise {sA['noise']:.4f}")
    log(f"[B vx4] per-pass p10/p50/p90 {sB['p10']:.4f}/{sB['p50']:.4f}/{sB['p90']:.4f} "
        f"noise {sB['noise']:.4f}")
    log(f"[B/A] p50 {sB['p50'] / sA['p50'] - 1:+.4%}; paired-ratio p10/p50/p90 "
        f"{sP['p10']:.4f}/{sP['p50']:.4f}/{sP['p90']:.4f}")
    if not unchanged(inputs, snap):
        failures.append("inputs changed during timing")
    # ---- 4. untimed replica of the timed sequence ----
    restore(inputs, snap)
    soak_count([gA, gB], n_soak)
    ref, nrep = {}, 0
    for b in range(args.blocks):
        for _ in range(args.block):
            for n, g, flat in (("A", gA, flatA), ("B", gB, flatB)):
                g.replay()
                if n not in ref:
                    ref[n] = flat.clone()
                elif not torch.equal(flat.view(torch.int16), ref[n].view(torch.int16)):
                    failures.append(f"arm {n} replay {nrep} differs from its replay 0")
            if not torch.equal(flatB.view(torch.int16), ref["A"].view(torch.int16)):
                failures.append(f"arm B replay {nrep} not bitwise equal to arm A")
            nrep += 1
        torch.cuda.synchronize()
        if not unchanged(inputs, snap):
            failures.append(f"inputs changed in correctness block {b}")
        if len(failures) > 20:
            break
    log(f"[check] {nrep} replays/arm after {n_soak} soak replays, failures {len(failures)}")
    # ---- 5. route check ----
    ok_route, route = route_check()
    if not ok_route:
        failures.append("route check failed")
    res = {"passed": not failures, "failures": failures[:20],
           "compared": "arm B (b2_ms_rq_vx4) vs arm A (b2_ms_rq, the protected reference): "
                       "bitwise at S=4176/4096/4256/4100 randn+peaky; untimed replica of the timed "
                       "ABAB sequence (same soak count, sample count, reset points), every B replay "
                       "bitwise == A replay 0 and every arm replay == its replay 0; inputs unchanged",
           "output": {"class": "exact"},
           "sass": sass, "bitwise_shapes": bw, "route": route,
           "kernels": {"A": kA, "B": kB},
           "A_b2_per_pass_ms": sA, "B_vx4_per_pass_ms": sB,
           "B_over_A_p50": sB["p50"] / sA["p50"] - 1, "paired_ratio": sP,
           "per_call_us": {"A": sA["p50"] / K * 1e3, "B": sB["p50"] / K * 1e3},
           "soak_replays": n_soak, "clocks": clocks}
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
