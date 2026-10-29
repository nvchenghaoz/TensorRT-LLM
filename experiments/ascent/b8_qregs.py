"""Branch 8 (idea 7): register-resident Q fragments (lever q_regs) in the fmha_v2 flux3 4-warp and
G3 kernels -- timing, bitwise checks and resource usage.

Synthetic E4M3 Q/K/V (B=1, H=24, D=128) from main's quantize_qkv_e4m3, laid out as the served
[3, S, 3072] qkv8 planes (so load_trim applies), fixed seeds; pristine copies kept and compared.

The served attention pass at 2048^2 (per denoise step; text T = 512, top-lead ruling):
  t2i : 1 x BF16-out (routed, G3) S16384 + 4 x nvfp4_out S16384 + 28 x nvfp4_out S16896
  i2i : the same with S32768 / S33280
nvfp4_out calls run the routed levers + nvfp4_out (4-warp: G3 refuses nvfp4_out).

Modes
  pair   one arm of run_gpu_pair: the tree's own ROUTED_LEVERS (base = main, branch = + q_regs).
         Per case, one sample = CUDA-event-timed replay of a graph holding the 33-call pass;
         kernel_ms = graph / 33; noise = (p90 - p10) / p50. Then an untimed replica of the timed
         sequence (same warmup and replay count), every replay == replay 0, inputs unchanged.
  ab     branch tree: A = ROUTED_LEVERS without q_regs, B = ROUTED_LEVERS; graphs interleaved.
  check  branch tree: A vs B bitwise for BF16 G3, BF16 4-warp and nvfp4_out (fp4 + sf) at
         S in --seqs; 100 repeats == call 0; graph replay x3 == eager; served-pass chain graph
         (t2i shapes) A == B; inputs unchanged; REG/STACK/STL-LDL of every module.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import re
import subprocess
import time

import torch

H, D = 24, 128
W = H * D
CASES = {"t2i-2048": (16384, 16896), "i2i-2048": (32768, 33280)}


def log(*a):
    print(*a, flush=True)


def lib():
    from tensorrt_llm._torch.visual_gen.attention.backends import fmha_v2_flux3 as fx

    return fx


def make_qkv(S, seed, peaky=False):
    from tensorrt_llm._torch.visual_gen.attention.backends.fp8_sm12x import quantize_qkv_e4m3

    g = torch.Generator(device="cuda").manual_seed(seed)
    buf = torch.randn((3, 1, S, H, D), generator=g, device="cuda", dtype=torch.float32) * 2.0
    if peaky:
        m = torch.rand((3, 1, S, H, 1), generator=g, device="cuda") < 0.01
        buf = torch.where(m, buf * 40.0, buf)
    q, k, v = buf.to(torch.bfloat16).unbind(0)
    q8, k8, v8, scales = quantize_qkv_e4m3(q, k, v, 1.0 / math.sqrt(D))
    del buf, q, k, v
    qkv8 = torch.empty((3, S, W), dtype=torch.float8_e4m3fn, device="cuda")
    for i, t in enumerate((q8, k8, v8)):
        qkv8[i].copy_(t.reshape(S, W))
    planes = [p.view(1, S, H, D) for p in qkv8.unbind(0)]
    return dict(S=S, qkv8=qkv8, q=planes[0], k=planes[1], v=planes[2], scales=scales,
                pristine=(qkv8.view(torch.uint8).clone(), scales.clone()))


def unchanged(inp):
    return bool(torch.equal(inp["qkv8"].view(torch.uint8), inp["pristine"][0])
                and torch.equal(inp["scales"], inp["pristine"][1]))


def restore(inp):
    inp["qkv8"].view(torch.uint8).copy_(inp["pristine"][0])
    inp["scales"].copy_(inp["pristine"][1])


GSCALE = None


def gscale():
    global GSCALE
    if GSCALE is None:
        GSCALE = torch.tensor([448.0 * 6.0 / 8.0], dtype=torch.float32, device="cuda")
    return GSCALE


class NfoOut:
    def __init__(self, S):
        self.fp4 = torch.empty((1, S, W // 2), dtype=torch.uint8, device="cuda")
        self.sf = torch.empty((lib().nvfp4_scale_bytes(S, W),), dtype=torch.uint8, device="cuda")

    def tensors(self):
        return [self.fp4, self.sf]


def call_bf16(mod, inp, o):
    sc = inp["scales"]
    lib().fmha_v2_flux3_d128(inp["q"], inp["k"], inp["v"], o, H, D, inp["S"],
                             scale_bmm1_d=sc[3:4], scale_bmm2_d=sc[2:3], causal=False, lse=None,
                             module=mod)


def call_nfo(mod, inp, out):
    sc = inp["scales"]
    lib().fmha_v2_flux3_d128_nvfp4_out(inp["q"], inp["k"], inp["v"], out.fp4, out.sf, gscale(),
                                       H, D, inp["S"], scale_bmm1_d=sc[3:4],
                                       scale_bmm2_d=sc[2:3], module=mod)


def routed(sample, levers=None):
    fx = lib()
    lev = tuple(fx.ROUTED_LEVERS) if levers is None else tuple(levers)
    return tuple(fx.load_trim_levers(lev, sample["k"], sample["v"]))


def pct(xs, p):
    return torch.tensor(xs, dtype=torch.float64).quantile(p).item()


def summary(xs):
    p10, p50, p90 = pct(xs, 0.1), pct(xs, 0.5), pct(xs, 0.9)
    return dict(n=len(xs), p10=p10, p50=p50, p90=p90, noise=(p90 - p10) / p50)


def pass_fns(levers, inps, outs, s_mode, s_single):
    """The 33-call served pass for one case with modules built from `levers`."""
    fx = lib()
    m_bf = fx.get_flux3_d128_module(levers)
    m_nfo = fx.get_flux3_d128_module((*levers, "nvfp4_out"))
    fns = [lambda: call_bf16(m_bf, inps[s_mode], outs["bf16"])]
    fns += [lambda: call_nfo(m_nfo, inps[s_mode], outs[s_mode])] * 4
    fns += [lambda: call_nfo(m_nfo, inps[s_single], outs[s_single])] * 28
    return fns


def case_buffers(s_mode, s_single, seed):
    inps = {s_mode: make_qkv(s_mode, seed + s_mode), s_single: make_qkv(s_single, seed + s_single)}
    outs = {"bf16": torch.empty((1, s_mode, H, D), device="cuda", dtype=torch.bfloat16),
            s_mode: NfoOut(s_mode), s_single: NfoOut(s_single)}
    return inps, outs


def out_tensors(outs):
    t = [outs["bf16"]]
    for k, v in outs.items():
        if k != "bf16":
            t += v.tensors()
    return t


def capture(fns):
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for f in fns:
            f()
    torch.cuda.current_stream().wait_stream(st)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in fns:
            f()
    torch.cuda.synchronize()
    return g


def graph_kernels(g):
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        g.replay()
        torch.cuda.synchronize()
    return {e.key: e.count for e in prof.key_averages() if "fmha" in e.key}


def timed(arms, samples_n, warm):
    for _ in range(warm):
        for _, g in arms:
            g.replay()
    torch.cuda.synchronize()
    samples = {n: [] for n, _ in arms}
    for i in range(samples_n):
        order = arms if i % 2 == 0 else arms[::-1]
        for n, g in order:
            a, z = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            g.replay()
            z.record()
            z.synchronize()
            samples[n].append(a.elapsed_time(z))
        if (i + 1) % 10 == 0:
            log(f"  [{i + 1}/{samples_n}] " + " ".join(
                f"{n}={sorted(v)[len(v) // 2]:.3f}ms" for n, v in samples.items()))
    return samples


def replica(g, tensors, inps, n_warm, n):
    """Untimed sequence matching the timed one; every replay == the first."""
    for inp in inps.values():
        restore(inp)
    g.replay()
    torch.cuda.synchronize()
    ref = [t.clone() for t in tensors]
    bad = 0
    for i in range(n_warm + n - 1):
        g.replay()
        torch.cuda.synchronize()
        bad += int(not all(torch.equal(a, b) for a, b in zip(tensors, ref)))
    ok = all(unchanged(inp) for inp in inps.values())
    sha = hashlib.sha256()
    for t in ref:
        sha.update(t.view(torch.uint8).flatten().cpu().numpy().tobytes())
    return bad, ok, sha.hexdigest()


def mode_pair(args):
    fx = lib()
    log(f"[tree] ROUTED_LEVERS {fx.ROUTED_LEVERS}")
    res = {"batch": {}, "output": {"class": "exact"}, "routed_levers": list(fx.ROUTED_LEVERS),
           "cases": {}}
    for case, (sm, ss) in CASES.items():
        n = args.samples if case.startswith("t2i") else args.samples_i2i
        inps, outs = case_buffers(sm, ss, 1)
        lev = routed(inps[ss])
        t0 = time.time()
        fns = pass_fns(lev, inps, outs, sm, ss)
        g = capture(fns)
        names = graph_kernels(g)
        log(f"[{case}] levers {lev} built+captured {time.time() - t0:.1f}s kernels {names}")
        s = summary(timed([("arm", g)], n, args.warm)["arm"])
        bad, ok, sha = replica(g, out_tensors(outs), inps, args.warm, n)
        log(f"[{case}] pass p50 {s['p50']:.3f} ms noise {s['noise']:.4f} per-call "
            f"{s['p50'] / 33:.4f} ms; replica mismatches {bad}; inputs unchanged {ok}")
        res["batch"][case] = {"kernel_ms": s["p50"] / 33, "noise_floor": s["noise"]}
        res["cases"][case] = dict(pass_ms=s, kernels=names, levers=list(lev),
                                  replica_mismatches=bad, inputs_unchanged=ok, output_sha=sha)
        del g, inps, outs
        torch.cuda.empty_cache()
    # SASS identity: the base tree's routed modules vs the candidate's modules without q_regs.
    sass = {}
    for lv in (lev, (*lev, "nvfp4_out"), tuple(x for x in lev if x != "q_regs"),
               tuple(x for x in (*lev, "nvfp4_out") if x != "q_regs")):
        tag = fx._tag(lv, 0)
        if tag in sass:
            continue
        fx.get_flux3_d128_module(lv)
        r = resource_usage(lv)
        sass[tag] = {n.split("_e4m3_")[-1]: dict(res=d.get("res"), sha=d.get("sass_sha"),
                                                 stl_ldl=d.get("stl_ldl"))
                     for n, d in r["functions"].items()}
    res["sass"] = sass
    log(f"[sass] {json.dumps(sass)}")
    print(json.dumps(res))


def mode_ab(args):
    fx = lib()
    res = {"cases": {}}
    for case, (sm, ss) in CASES.items():
        n = args.samples if case.startswith("t2i") else args.samples_i2i
        inps, outs = case_buffers(sm, ss, 1)
        lev_b = routed(inps[ss])
        if "q_regs" not in lev_b:
            lev_b = (*lev_b, "q_regs")
        lev_a = tuple(x for x in lev_b if x != "q_regs")
        gA = capture(pass_fns(lev_a, inps, outs, sm, ss))
        gB = capture(pass_fns(lev_b, inps, outs, sm, ss))
        gA2 = capture(pass_fns(lev_a, inps, outs, sm, ss))
        smp = timed([("A", gA), ("B", gB), ("A2", gA2)], n, args.warm)
        row = {k: summary(v) for k, v in smp.items()}
        row["B_over_A"] = summary([b / a for a, b in zip(smp["A"], smp["B"])])
        row["A2_over_A"] = summary([b / a for a, b in zip(smp["A"], smp["A2"])])
        log(f"[{case}] {json.dumps(row)}")
        res["cases"][case] = row
        del gA, gB, gA2, inps, outs
        torch.cuda.empty_cache()
    print(json.dumps(res))


def resource_usage(levers):
    from flashinfer.jit import env as jit_env

    fx = lib()
    uri = f"trtllm_fmha_v2_flux3_d128_{fx._tag(levers, 0)}"
    sos = glob.glob(str(jit_env.FLASHINFER_JIT_DIR / uri / "**" / "*.so"), recursive=True)
    sos += glob.glob(str(jit_env.FLASHINFER_JIT_DIR / "**" / f"{uri}*.so"), recursive=True)
    out = {"uri": uri, "functions": {}}
    for so in sos[:1]:
        r = subprocess.run(["cuobjdump", "-res-usage", so], capture_output=True, text=True,
                           timeout=300)
        cur = None
        for line in r.stdout.splitlines():
            m = re.search(r"Function (\S+):", line)
            if m:
                cur = m.group(1)
                continue
            if cur and "REG:" in line and "flux3" in cur:
                out["functions"].setdefault(cur, {})["res"] = line.strip()
                cur = None
        r = subprocess.run(["cuobjdump", "-sass", so], capture_output=True, text=True,
                           timeout=600)
        name, buf, funcs = None, [], {}
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
        for n, b in funcs.items():
            if n not in out["functions"]:
                continue
            d = out["functions"][n]
            d["stl_ldl"] = sum(("STL" in x or "LDL" in x) for x in b)
            d["ldsm"] = sum("LDSM" in x for x in b)
            d["lines"] = len(b)
            d["sass_sha"] = hashlib.sha256("\n".join(b).encode()).hexdigest()[:16]
    return out


def seq_check(fns, tensors, repeats):
    for f in fns:
        f()
    torch.cuda.synchronize()
    ref = [t.clone() for t in tensors]
    rep_bad = 0
    for r in range(repeats):
        for t in tensors:
            t.fill_(0x5A if t.dtype == torch.uint8 else 3)
        for f in fns:
            f()
        torch.cuda.synchronize()
        rep_bad += int(not all(torch.equal(a, b) for a, b in zip(tensors, ref)))
    g = capture(fns)
    gbad = 0
    for _ in range(3):
        for t in tensors:
            t.fill_(0x11 if t.dtype == torch.uint8 else 1)
        g.replay()
        torch.cuda.synchronize()
        gbad += int(not all(torch.equal(a, b) for a, b in zip(tensors, ref)))
    del g
    return ref, rep_bad, gbad


def eq_all(a, b):
    return all(torch.equal(x.view(torch.uint8), y.view(torch.uint8)) for x, y in zip(a, b))


def mode_check(args):
    fx = lib()
    sample = make_qkv(4176, 7)
    lev_b = routed(sample)
    if "q_regs" not in lev_b:
        lev_b = (*lev_b, "q_regs")
    lev_a = tuple(x for x in lev_b if x != "q_regs")
    mods = {}
    t0 = time.time()
    for arm, lev in (("A", lev_a), ("B", lev_b)):
        no_g3 = tuple(x for x in lev if x != "g3")
        mods[arm] = {"bf16_g3": fx.get_flux3_d128_module(lev),
                     "bf16_4w": fx.get_flux3_d128_module(no_g3),
                     "nfo": fx.get_flux3_d128_module((*lev, "nvfp4_out"))}
        mods[arm + "_levers"] = {"bf16_g3": lev, "bf16_4w": no_g3, "nfo": (*lev, "nvfp4_out")}
    log(f"[build] {time.time() - t0:.1f}s A={lev_a} B={lev_b}")
    failures, rows = [], {}
    for peaky in (False, True):
        tag = "peaky" if peaky else "randn"
        for S in args.seqs:
            inp = make_qkv(S, 300 + S + int(peaky), peaky)
            for kind in ("bf16_g3", "bf16_4w", "nfo"):
                refs = {}
                for arm in ("A", "B"):
                    m = mods[arm][kind]
                    if kind == "nfo":
                        o = NfoOut(S)
                        tens = o.tensors()
                        fns = [lambda m=m, o=o: call_nfo(m, inp, o)]
                    else:
                        o = torch.empty((1, S, H, D), device="cuda", dtype=torch.bfloat16)
                        tens = [o]
                        fns = [lambda m=m, o=o: call_bf16(m, inp, o)]
                    reps = args.repeats if arm == "B" else 3
                    ref, rb, gb = seq_check(fns, tens, reps)
                    refs[arm] = ref
                    if rb or gb:
                        failures.append(f"{tag}_{kind}_{arm}_S{S}_repeat{rb}_graph{gb}")
                    rows[f"{tag}_{kind}_{arm}_S{S}"] = dict(repeats=reps, repeat_mismatch=rb,
                                                            graph_mismatch=gb)
                eq = eq_all(refs["A"], refs["B"])
                fin = True
                if kind != "nfo":
                    fin = bool(torch.isfinite(refs["B"][0].float()).all())
                rows[f"{tag}_{kind}_S{S}_bitwise_B_eq_A"] = eq
                if not (eq and fin):
                    failures.append(f"{tag}_{kind}_S{S}_bitwise{eq}_finite{fin}")
                log(f"[{tag} {kind} S{S}] bitwise {eq} finite {fin} "
                    f"B {rows[f'{tag}_{kind}_B_S{S}']}")
            if not unchanged(inp):
                failures.append(f"{tag}_inputs_changed_S{S}")
            del inp
            torch.cuda.empty_cache()
    # served-pass chain (t2i 2048 shapes), A vs B, eager and graph
    sm, ss = CASES["t2i-2048"]
    inps, outs = case_buffers(sm, ss, 5)
    chain = {}
    for arm, lev in (("A", lev_a), ("B", lev_b)):
        fns = pass_fns(lev, inps, outs, sm, ss)
        ref, rb, gb = seq_check(fns, out_tensors(outs), 2)
        chain[arm] = ref
        rows[f"chain_t2i_{arm}"] = dict(repeat_mismatch=rb, graph_mismatch=gb)
        if rb or gb:
            failures.append(f"chain_{arm}")
    rows["chain_t2i_bitwise_B_eq_A"] = eq_all(chain["A"], chain["B"])
    if not rows["chain_t2i_bitwise_B_eq_A"]:
        failures.append("chain_bitwise")
    if not all(unchanged(i) for i in inps.values()):
        failures.append("chain_inputs_changed")
    log(f"[chain] {rows['chain_t2i_bitwise_B_eq_A']}")
    del inps, outs, chain
    res = {"levers_A": list(lev_a), "levers_B": list(lev_b), "rows": rows, "resources": {}}
    for arm in ("A", "B"):
        for kind in ("bf16_g3", "nfo"):
            r = resource_usage(mods[arm + "_levers"][kind])
            res["resources"][f"{arm}_{kind}"] = r
            log(f"[res {arm} {kind}] {json.dumps(r)}")
    for kind in ("bf16_g3", "nfo"):
        for fn, d in res["resources"][f"B_{kind}"]["functions"].items():
            if d.get("stl_ldl"):
                failures.append(f"spill_{kind}_{fn[-40:]}")
            m = re.search(r"REG:(\d+)", d.get("res", ""))
            if "_g3" in fn and m and int(m.group(1)) > 168:
                failures.append(f"g3_regs_{m.group(1)}")
            if "STACK:0" not in d.get("res", "STACK:0"):
                failures.append(f"stack_{kind}_{fn[-40:]}")
    res["passed"] = not failures
    res["failures"] = failures
    res["compared"] = ("B = ROUTED_LEVERS (+q_regs) vs A = the same levers without q_regs (main's "
                       "kernels: q_regs is macro-gated) for BF16-out G3, BF16-out 4-warp and "
                       "nvfp4_out (fp4+sf) modules, bitwise per S, randn and peaky; B: repeats == "
                       "call 0, graph replay x3 == eager; t2i-2048 served-pass chain A == B; "
                       "inputs unchanged; REG/STACK/STL-LDL")
    print(json.dumps(res))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="pair", choices=("pair", "ab", "check", "res"))
    p.add_argument("--samples", type=int, default=50)
    p.add_argument("--samples-i2i", dest="samples_i2i", type=int, default=24)
    p.add_argument("--warm", type=int, default=5)
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--seqs", type=int, nargs="+",
                   default=[16384, 16896, 16900, 33280, 4176, 4100])
    args = p.parse_args()
    if args.mode == "pair":
        mode_pair(args)
    elif args.mode == "ab":
        mode_ab(args)
    elif args.mode == "check":
        mode_check(args)
    else:
        fx = lib()
        sample = make_qkv(4176, 7)
        out = {}
        for lev in (routed(sample), tuple(x for x in routed(sample) if x != "q_regs")):
            for extra in ((), ("nvfp4_out",)):
                lv = (*lev, *extra)
                fx.get_flux3_d128_module(lv)
                out[fx._tag(lv, 0)] = resource_usage(lv)
        log(json.dumps(out))
        print(json.dumps({"passed": True, "resources": out}))


if __name__ == "__main__":
    main()
