"""Idea 124 (branch 72): softmax_trim (FMHA_FLUX3_SOFTMAX_TRIM) on the routed base-2 fmha_v2
flux3 kernel. Copy of flux3_b2_graph_ab.py's sample design (ruling 2096 / method 881).

One SAMPLE = one CUDA-event-timed replay of a CUDA graph holding the full per-pass d128 FP8 set
of FLUX.3 klein at 1024^2: 5 x S=4096 then 28 x S=4176, B=1 H=24 D=128. Per-call = graph / 33.
Inputs: E4M3 Q/K/V + device scales from fp8_sm12x.quantize_qkv_e4m3 on the planes of ONE BF16
buffer, fixed seed; pristine copies restored before timing and before the correctness pass.

Modes
  pair  one arm for run_gpu_pair, through run_fmha_v2_fp8_sm12x (the tree's own routing:
        base -> b2_ms_rq, branch -> b2_ms_rq_st). Prints the evidence JSON.
  ab    branch tree, one process: graph A = b2 module (ROUTED_LEVERS minus softmax_trim),
        graph B = b2+softmax_trim module, both via fmha_v2_flux3_d128(module=...). Warm-soak,
        samples interleaved ABAB, SM clock per block, per-arm p10/p50/p90, nothing dropped.
        Then the untimed correctness sequence (same soak count, interleaving, sample count,
        reset points): every replay bitwise vs arm A's replay 0 (B must equal A bitwise),
        inputs unchanged; plus direct calls at S=4176/4096/4256 on randn and peaky inputs,
        B bitwise vs A, repeated.
  sass  cuobjdump the two built modules: opcode census of the kernel + registers/spills.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import shutil
import subprocess
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


def make_one(S, g, peaky=False):
    from tensorrt_llm._torch.visual_gen.attention.backends.fp8_sm12x import quantize_qkv_e4m3

    buf = torch.randn((3, 1, S, H, D), generator=g, device="cuda", dtype=torch.float32)
    if peaky:
        buf[0] *= 3.0
        # a few keys strongly aligned with some queries -> near one-hot rows
        idx = torch.arange(0, S, 97, device="cuda")
        buf[1][:, idx] = buf[0][:, idx] * 4.0
    buf = buf.to(torch.bfloat16)
    q, k, v = buf.unbind(0)
    q8, k8, v8, scales = quantize_qkv_e4m3(q, k, v, 1.0 / math.sqrt(D))
    return dict(S=S, q=q8, k=k8, v=v8, scales=scales)


def make_inputs(seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    out = {S: make_one(S, g) for S in sorted(set(SEQ))}
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


def call_routed(fib, inp, out):
    sc = inp["scales"]
    fib.run_fmha_v2_fp8_sm12x(inp["q"], inp["k"], inp["v"], out, H, D, inp["S"],
                              scale_bmm1_d=sc[3:4], scale_bmm2_d=sc[2:3], causal=False,
                              return_lse=False)


def call_module(lib, mod, inp, out):
    sc = inp["scales"]
    lib.fmha_v2_flux3_d128(inp["q"], inp["k"], inp["v"], out, H, D, inp["S"],
                           scale_bmm1_d=sc[3:4], scale_bmm2_d=sc[2:3], causal=False, lse=None,
                           module=mod)


def make_outputs():
    total = sum(SEQ) * H * D
    flat = torch.empty(total, device="cuda", dtype=torch.bfloat16)
    outs, off = [], 0
    for S in SEQ:
        n = S * H * D
        outs.append(flat[off:off + n].view(1, S, H, D))
        off += n
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
    return g, flat, outs


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


def pct(xs, p):
    return torch.tensor(xs, dtype=torch.float64).quantile(p).item()


def summary(xs):
    p10, p50, p90 = pct(xs, 0.1), pct(xs, 0.5), pct(xs, 0.9)
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


def levers_ab(lib):
    b2 = tuple(x for x in lib.ROUTED_LEVERS if x != "softmax_trim")
    st = tuple(sorted(set(b2) | {"softmax_trim"}))
    return b2, st


def mode_pair(args):
    from tensorrt_llm._torch.visual_gen.attention.backends import flashinfer as fib
    from tensorrt_llm._torch.visual_gen.attention.backends import fmha_v2_flux3 as lib

    log(f"[tree] ROUTED_LEVERS {lib.ROUTED_LEVERS}")
    clock = Clock()
    inputs = make_inputs()
    snap = snapshot(inputs)
    g, flat, _ = capture(lambda inp, o: call_routed(fib, inp, o), inputs)
    names = graph_kernels(g)
    log(f"[graph] kernels per replay: {json.dumps(names)}")
    restore(inputs, snap)
    n_soak = soak([g], args.soak_s)
    log(f"[soak] {n_soak} replays in >= {args.soak_s}s, sm_mhz {clock.sm_mhz()}")
    samples, clocks = timed([("arm", g)], args.blocks, args.block, clock)
    s = summary(samples["arm"])
    ok_inputs = unchanged(inputs, snap)
    per_call = s["p50"] / K
    log(f"[result] per-pass p10/p50/p90 {s['p10']:.4f}/{s['p50']:.4f}/{s['p90']:.4f} ms, "
        f"noise {s['noise']:.4f}, per-call {per_call * 1e3:.1f} us, inputs unchanged {ok_inputs}")
    print(json.dumps({"batch": {"1": {"kernel_ms": per_call, "noise_floor": s["noise"]}},
                      "output": {"class": "exact"},
                      "sample": f"one CUDA-event-timed replay of a CUDA graph of {K} calls "
                                f"(5xS4096 + 28xS4176); kernel_ms = per-call = graph/{K}",
                      "routed_levers": list(lib.ROUTED_LEVERS),
                      "per_pass_ms": s, "soak_replays": n_soak, "kernels": names,
                      "clocks": clocks, "inputs_unchanged": ok_inputs}))


def mode_ab(args):
    from tensorrt_llm._torch.visual_gen.attention.backends import fmha_v2_flux3 as lib

    b2, st = levers_ab(lib)
    log(f"[levers] A {b2} tag {lib._tag(b2, 0)} | B {st} tag {lib._tag(st, 0)}")
    modA, modB = lib.get_flux3_d128_module(b2), lib.get_flux3_d128_module(st)
    tagA, tagB = "fmha_v2_flux3_" + lib._tag(b2, 0) + "_", "fmha_v2_flux3_" + lib._tag(st, 0) + "_"
    clock = Clock()
    inputs = make_inputs()
    snap = snapshot(inputs)
    gA, flatA, _ = capture(lambda inp, o: call_module(lib, modA, inp, o), inputs)
    gB, flatB, _ = capture(lambda inp, o: call_module(lib, modB, inp, o), inputs)
    kA, kB = graph_kernels(gA), graph_kernels(gB)
    log(f"[graph A] {json.dumps(kA)}")
    log(f"[graph B] {json.dumps(kB)}")
    failures = []
    if sum(c for n, c in kA.items() if tagA in n) != K:
        failures.append(f"graph A does not hold {K} {tagA} kernels: {kA}")
    if sum(c for n, c in kB.items() if tagB in n) != K:
        failures.append(f"graph B does not hold {K} {tagB} kernels: {kB}")
    arms = [("A_b2", gA), ("B_b2_st", gB)]
    restore(inputs, snap)
    n_soak = soak([gA, gB], args.soak_s)
    log(f"[soak] {n_soak} AB replays in >= {args.soak_s}s, sm_mhz {clock.sm_mhz()}")
    samples, clocks = timed(arms, args.blocks, args.block, clock)
    sA, sB = summary(samples["A_b2"]), summary(samples["B_b2_st"])
    sP = summary([b / a for a, b in zip(samples["A_b2"], samples["B_b2_st"])])
    log(f"[A b2   ] per-pass p10/p50/p90 {sA['p10']:.4f}/{sA['p50']:.4f}/{sA['p90']:.4f} "
        f"noise {sA['noise']:.4f} per-call {sA['p50'] / K * 1e3:.2f} us")
    log(f"[B b2_st] per-pass p10/p50/p90 {sB['p10']:.4f}/{sB['p50']:.4f}/{sB['p90']:.4f} "
        f"noise {sB['noise']:.4f} per-call {sB['p50'] / K * 1e3:.2f} us")
    log(f"[B/A] p50 {sB['p50'] / sA['p50'] - 1:+.4%}; paired-ratio p10/p50/p90 "
        f"{sP['p10']:.4f}/{sP['p50']:.4f}/{sP['p90']:.4f}")
    if not unchanged(inputs, snap):
        failures.append("inputs changed during timing")
    # ---- untimed correctness, same sequence ----
    restore(inputs, snap)
    soak_count([gA, gB], n_soak)
    ref = None
    nrep = 0
    for b in range(args.blocks):
        for _ in range(args.block):
            for n, g, flat in (("A", gA, flatA), ("B", gB, flatB)):
                g.replay()
                if ref is None:
                    ref = flat.clone()
                elif not torch.equal(flat, ref):
                    failures.append(f"arm {n} replay {nrep} differs from A replay 0")
            nrep += 1
        torch.cuda.synchronize()
        if not unchanged(inputs, snap):
            failures.append(f"inputs changed before block {b} reset point")
        if len(failures) > 20:
            break
    log(f"[check graph] {nrep} replays/arm after {n_soak} soak replays, failures {len(failures)}")
    # ---- direct calls, randn + peaky, S=4176/4096/4256 ----
    direct = {}
    g = torch.Generator(device="cuda").manual_seed(7)
    for peaky in (False, True):
        for S in (4176, 4096, 4256):
            inp = make_one(S, g, peaky)
            sn = {n: inp[n].clone() for n in ("q", "k", "v", "scales")}
            oA = torch.empty(1, S, H, D, device="cuda", dtype=torch.bfloat16)
            oB = torch.empty_like(oA)
            call_module(lib, modA, inp, oA)
            refA = oA.clone()
            ok = True
            for r in range(args.direct_reps):
                call_module(lib, modA, inp, oA)
                call_module(lib, modB, inp, oB)
                ok &= torch.equal(oA, refA) and torch.equal(oB, refA)
            torch.cuda.synchronize()
            same_in = all(torch.equal(inp[n].view(torch.uint8) if inp[n].dtype ==
                                      torch.float8_e4m3fn else inp[n],
                                      sn[n].view(torch.uint8) if sn[n].dtype ==
                                      torch.float8_e4m3fn else sn[n]) for n in sn)
            d = (oB.float() - refA.float()).abs().max().item()
            key = f"{'peaky' if peaky else 'randn'}_S{S}"
            direct[key] = dict(bitwise=bool(ok), max_abs=d, inputs_unchanged=bool(same_in),
                               finite=bool(torch.isfinite(oB.float()).all().item()))
            log(f"[check direct] {key} {direct[key]}")
            if not (ok and same_in):
                failures.append(f"direct {key} {direct[key]}")
    res = {"passed": not failures, "failures": failures[:20],
           "compared": "B (b2+softmax_trim) bitwise vs A (b2): every graph replay of both arms "
                       "vs A replay 0 over the untimed replica of the timed sequence (same soak "
                       "count, interleaving, sample count, pristine inputs restored first, "
                       "inputs checked per block); direct calls at S=4176/4096/4256 randn+peaky",
           "output_class": "exact",
           "kernels": {"A": kA, "B": kB},
           "A_b2_per_pass_ms": sA, "B_b2_st_per_pass_ms": sB,
           "B_over_A_p50": sB["p50"] / sA["p50"] - 1, "paired_ratio": sP,
           "per_call_us": {"A": sA["p50"] / K * 1e3, "B": sB["p50"] / K * 1e3},
           "soak_replays": n_soak, "clocks": clocks, "direct": direct}
    print(json.dumps(res))


def _find_tool(name):
    p = shutil.which(name)
    if p:
        return p
    for c in ("/usr/local/cuda/bin/" + name,):
        if os.path.exists(c):
            return c
    return None


def mode_sass(args):
    from flashinfer.jit import env as jit_env

    from tensorrt_llm._torch.visual_gen.attention.backends import fmha_v2_flux3 as lib

    b2, st = levers_ab(lib)
    cuobj = _find_tool("cuobjdump")
    log(f"[sass] cuobjdump {cuobj}")
    out = {}
    for name, lv in (("b2", b2), ("b2_st", st)):
        lib.get_flux3_d128_module(lv)
        uri = f"trtllm_fmha_v2_flux3_d128_{lib._tag(lv, 0)}"
        cands = []
        for root in {str(jit_env.FLASHINFER_WORKSPACE_DIR), str(jit_env.FLASHINFER_JIT_DIR)}:
            cands += glob.glob(os.path.join(root, "**", f"{uri}.so"), recursive=True)
        cands += glob.glob(os.path.join(str(jit_env.FLASHINFER_JIT_DIR), uri, "**", "*.o"),
                           recursive=True)
        log(f"[sass] {name} candidates {cands[:6]}")
        so = [c for c in cands if c.endswith(".so")] or cands
        # the SASS function is <prefix>_sm120_kernel_nl_tiled (the padding-mask kernel)
        kern = lib.flux3_d128_kernel_name(lv).replace("_sm120_nl_tiled", "_sm120_kernel_nl_tiled")
        res = subprocess.run([cuobj, "-res-usage", so[0]], capture_output=True, text=True)
        resline = ""
        lines = res.stdout.splitlines()
        for i, ln in enumerate(lines):
            if kern in ln and i + 1 < len(lines):
                resline = lines[i + 1].strip()
                break
        log(f"[sass] res-usage rc {res.returncode} lines {len(lines)} err {res.stderr[:400]!r} "
            f"head {lines[:4]}")
        sp = subprocess.run([cuobj, "-sass", so[0]], capture_output=True, text=True)
        sass = sp.stdout
        funcs = [ln.strip() for ln in sass.splitlines() if "Function :" in ln]
        log(f"[sass] -sass rc {sp.returncode} chars {len(sass)} err {sp.stderr[:400]!r} "
            f"functions {len(funcs)} {[f[:160] for f in funcs[:6]]}")
        if not funcs:
            # fall back: extract the cubins and disassemble them
            tmp = os.path.join(os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp"), f"elf_{name}")
            os.makedirs(tmp, exist_ok=True)
            x = subprocess.run([cuobj, "-xelf", "all", os.path.abspath(so[0])], cwd=tmp,
                               capture_output=True, text=True)
            log(f"[sass] -xelf rc {x.returncode} {x.stdout[:300]!r} {x.stderr[:300]!r}")
            nvdis = _find_tool("nvdisasm")
            sass = ""
            for cub in sorted(glob.glob(os.path.join(tmp, "*.cubin"))):
                if "flux3" not in cub and len(glob.glob(os.path.join(tmp, "*.cubin"))) > 1:
                    pass
                for tool in ([cuobj, "-sass", cub], [nvdis, "-c", cub] if nvdis else None):
                    if tool is None:
                        continue
                    r = subprocess.run(tool, capture_output=True, text=True)
                    if "Function :" in r.stdout or ".text." in r.stdout:
                        sass += r.stdout
                        break
            funcs = [ln.strip() for ln in sass.splitlines() if "Function :" in ln]
            log(f"[sass] fallback chars {len(sass)} functions {len(funcs)}")
            if not resline:
                for cub in sorted(glob.glob(os.path.join(tmp, "*.cubin"))):
                    r = subprocess.run([cuobj, "-res-usage", cub], capture_output=True, text=True)
                    ls = r.stdout.splitlines()
                    for i, ln in enumerate(ls):
                        if kern in ln and i + 1 < len(ls):
                            resline = ls[i + 1].strip()
        # isolate the kernel's function body
        body, on = [], False
        for ln in sass.splitlines():
            if "Function :" in ln:
                on = ln.strip().endswith(kern)
                continue
            if on:
                body.append(ln)
        ops = {}
        for ln in body:
            m = re.search(r"/\*[0-9a-f]{4,}\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Z0-9_]*)(\.[A-Z0-9_.]+)?", ln)
            if m:
                op = m.group(2)
                ops[op] = ops.get(op, 0) + 1
        # KV loop = the widest backward branch (target < branch address) in the kernel
        loop_ops, loop_span = {}, None
        insts = []
        for ln in body:
            m = re.search(r"/\*([0-9a-f]{4,})\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Z0-9_]*)(\.[A-Z0-9_.]+)?"
                          r"(.*)", ln)
            if m:
                insts.append((int(m.group(1), 16), m.group(3), m.group(5)))
        best = None
        for addr, op, rest in insts:
            if op == "BRA":
                t = re.search(r"0x([0-9a-f]+)", rest)
                if t and int(t.group(1), 16) < addr:
                    span = addr - int(t.group(1), 16)
                    if best is None or span > best[1] - best[0]:
                        best = (int(t.group(1), 16), addr)
        if best:
            loop_span = [hex(best[0]), hex(best[1])]
            for addr, op, _ in insts:
                if best[0] <= addr <= best[1]:
                    loop_ops[op] = loop_ops.get(op, 0) + 1
        keep = ("FMUL", "FMNMX", "FMNMX3", "FADD", "FFMA", "MUFU", "F2FP", "PRMT", "BAR", "SHFL",
                "HMMA", "QMMA", "OMMA", "F2F")
        sel = {k: v for k, v in sorted(ops.items()) if any(k.startswith(x) for x in keep)}
        lsel = {k: v for k, v in sorted(loop_ops.items()) if any(k.startswith(x) for x in keep)}
        out[name] = dict(kernel=kern, so=so[0], resource=resline, total=sum(ops.values()),
                         ops=sel, lines=len(body), loop_span=loop_span,
                         loop_total=sum(loop_ops.values()), loop_ops=lsel)
        log(f"[sass] {name} {kern}\n  {resline}\n  total {sum(ops.values())} {json.dumps(sel)}\n"
            f"  loop {loop_span} total {sum(loop_ops.values())} {json.dumps(lsel)}")
        art = os.environ.get("AUTO_DEV_ARTIFACTS")
        if art:
            with open(os.path.join(art, f"sass_{name}.txt"), "w") as f:
                f.write("\n".join(body))
    diff = {k: out["b2_st"]["ops"].get(k, 0) - out["b2"]["ops"].get(k, 0)
            for k in set(out["b2"]["ops"]) | set(out["b2_st"]["ops"])}
    ldiff = {k: out["b2_st"]["loop_ops"].get(k, 0) - out["b2"]["loop_ops"].get(k, 0)
             for k in set(out["b2"]["loop_ops"]) | set(out["b2_st"]["loop_ops"])}
    log(f"[sass] st - b2: {json.dumps(diff)} | loop: {json.dumps(ldiff)}")
    print(json.dumps({"passed": True, "sass": out, "diff_st_minus_b2": diff,
                      "loop_diff_st_minus_b2": ldiff}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["pair", "ab", "sass"])
    p.add_argument("--soak-s", type=float, default=12.0)
    p.add_argument("--blocks", type=int, default=20)
    p.add_argument("--block", type=int, default=25)
    p.add_argument("--direct-reps", type=int, default=5)
    args = p.parse_args()
    log(f"device {torch.cuda.get_device_name()} K={K} calls/graph")
    {"pair": mode_pair, "ab": mode_ab, "sass": mode_sass}[args.mode](args)


if __name__ == "__main__":
    main()
