"""Branch 186 (idea 270) on merged main (69eac46e + G3): the nfo route and the landing-76 CUDA 700.

Modules (all from this tree's fmha_v2_flux3, levers = load_trim_levers(ROUTED_LEVERS)):
  bf16_main  levers - g3                 main's routed BF16-out module (4-warp)
  bf16_g3    levers                      G3 BF16-out module (served for the unforked calls)
  nfo_main   levers - g3 + nvfp4_out     main's nfo module (4-warp + NVFP4 epilogue)
  nfo_fix    levers + nvfp4_out          the served nfo module on this branch (G3 guard active)
  nfo_unfix  nfo_fix built from a copy of the G3 header with the FMHA_FLUX3_NVFP4_OUT guard
             removed (= landing 76 / merge job 1987 as merged), its own URI. --unfixed only.

Checks per S in (4096, 4176, 4100), randn and peaky Q/K/V as [1,S,24,128] views of a guarded
[3,S,3072] qkv8; fp4 / sf / O each carved from a buffer with 0xA5 guards (1 MB, 40 MB after fp4
for the unfixed run so its overrun stays inside the guard):
  census     kernel names of each module call (torch.profiler)
  nfo        fp4 + sf of nfo_fix == nfo_main bitwise; guards clean; 100 repeats == call 0;
             graph replay x3 == eager; sf fully written (no 0xA5 left in the valid region)
  bf16       bf16_g3 == bf16_main bitwise; guards clean; repeats/graph
  chain      served pass: 1 x bf16 S4096 + 4 x nfo S4096 + 28 x nfo S4176, eager + graph x3
  unfixed    nfo_unfix at S4176 / S4096 (eager): first/last dirty guard byte after fp4,
             whether sf was written. Run last (it corrupts memory beyond the guard if any).
  sass       REG/STACK/SASS hash of the 4-warp kernels in nfo_fix vs nfo_main, bf16_g3 vs main.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import b165_g3  # noqa: E402

H, D = 24, 128
W = H * D
MB = 1 << 20
FILL = 0xA5


def log(*a):
    print(*a, flush=True)


class Guarded:
    def __init__(self, nbytes, pre=MB, post=MB):
        self.n, self.pre, self.post = nbytes, pre, post
        self.buf = torch.full((pre + nbytes + post,), FILL, dtype=torch.uint8, device="cuda")
        self.payload = self.buf[pre:pre + nbytes]

    def check(self):
        out = {}
        for name, seg, base in (("pre", self.buf[:self.pre], -self.pre),
                                ("post", self.buf[self.pre + self.n:], self.n)):
            idx = torch.nonzero(seg != FILL).flatten()
            if idx.numel():
                out[name] = dict(count=int(idx.numel()), first=int(idx[0]) + base,
                                 last=int(idx[-1]) + base)
        return out


def probe():
    def sh(*cmd):
        p = subprocess.run(list(cmd), capture_output=True, text=True)
        return p.returncode, (p.stdout + p.stderr).strip()

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    fx_path = os.path.join(root, "tensorrt_llm/_torch/visual_gen/attention/backends/fmha_v2_flux3.py")
    qk_path = os.path.join(root, "tensorrt_llm/_torch/visual_gen")
    res = {"git_log": sh("git", "-C", root, "log", "--oneline", "-3")[1],
           "ancestor_69eac46e_rc": sh("git", "-C", root, "merge-base", "--is-ancestor", "69eac46e",
                                      "HEAD")[0],
           "marker_nvfp4_out_lever": '_LEVER_FLAGS["nvfp4_out"]' in open(fx_path).read(),
           "marker_g3": "TRTLLM_FMHA_V2_FLUX3_G3" in open(fx_path).read(),
           "marker_static_qk": sh("grep", "-rl", "TRTLLM_FLUX3_QKV_FP8_STATIC_QK", qk_path)[1][:300]}
    log(f"[probe] {json.dumps(res)}")
    return res


def make_qkv(S, seed, peaky):
    from tensorrt_llm._torch.visual_gen.attention.backends.fp8_sm12x import quantize_qkv_e4m3

    g = torch.Generator(device="cuda").manual_seed(seed)
    buf = torch.randn((3, 1, S, H, D), generator=g, device="cuda", dtype=torch.float32) * 2.0
    if peaky:
        m = torch.rand((3, 1, S, H, 1), generator=g, device="cuda") < 0.01
        buf = torch.where(m, buf * 40.0, buf)
    q, k, v = buf.to(torch.bfloat16).unbind(0)
    q8, k8, v8, scales = quantize_qkv_e4m3(q, k, v, 1.0 / math.sqrt(D))
    gq = Guarded(3 * S * W)
    qkv8 = gq.payload.view(torch.float8_e4m3fn).view(3, S, W)
    for i, t in enumerate((q8, k8, v8)):
        qkv8[i].copy_(t.reshape(S, W))
    planes = [p.view(1, S, H, D) for p in qkv8.unbind(0)]
    return dict(S=S, q=planes[0], k=planes[1], v=planes[2], scales=scales, guard=gq,
                pristine=gq.payload.clone())


def build_modules(lib, sample, unfixed):
    lev = tuple(lib.load_trim_levers(tuple(lib.ROUTED_LEVERS), sample["k"], sample["v"]))
    assert "g3" in lev, f"ROUTED_LEVERS lacks g3: {lev}"
    no_g3 = tuple(x for x in lev if x != "g3")
    t0 = time.time()
    mods = {"bf16_main": (no_g3, lib.get_flux3_d128_module(no_g3)),
            "bf16_g3": (lev, lib.get_flux3_d128_module(lev)),
            "nfo_main": ((*no_g3, "nvfp4_out"), lib.get_flux3_d128_module((*no_g3, "nvfp4_out"))),
            "nfo_fix": ((*lev, "nvfp4_out"), lib.get_flux3_d128_module((*lev, "nvfp4_out")))}
    if unfixed:
        from flashinfer.jit import env as jit_env

        src = (jit_env.FLASHINFER_CSRC_DIR / "fmha_v2" /
               "fused_multihead_flash_attention_kernel_flux3_g3.h").read_text()
        guard = re.search(r"#if defined\(FMHA_FLUX3_NVFP4_OUT\) && FMHA_FLUX3_NVFP4_OUT\n.*?#endif\n",
                          src, re.S)
        assert guard, "guard not found in the G3 header"
        unfixed_src = src[:guard.start()] + src[guard.end():]
        d = tempfile.mkdtemp(prefix="b186_unfixed_")
        path = os.path.join(d, "flux3_g3_unfixed.h")
        open(path, "w").write(unfixed_src)
        orig = lib._G3_SNIPPET
        lib._G3_SNIPPET = orig.replace("#include <fused_multihead_flash_attention_kernel_flux3_g3.h>",
                                       f'#include "{path}"')
        lib._LEVER_FLAGS["g3_unguarded_probe"] = ("g3ug", "-DFMHA_FLUX3_G3_UNGUARDED_PROBE=1")
        levu = (*lev, "nvfp4_out", "g3_unguarded_probe")
        mods["nfo_unfix"] = (levu, lib.get_flux3_d128_module(levu))
        lib._G3_SNIPPET = orig
        log(f"[unfixed] guard removed ({guard.end() - guard.start()} bytes) -> {path}")
    log(f"[build] {', '.join(f'{k}={lib._tag(v[0], 0)}' for k, v in mods.items())} "
        f"{time.time() - t0:.1f}s")
    return mods


def names(fn):
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return {e.key: e.count for e in prof.key_averages() if "fmha" in e.key}


class NfoOut:
    def __init__(self, S, post=MB):
        rows = S
        self.gf = Guarded(rows * W // 2, post=post)
        self.fp4 = self.gf.payload.view(1, S, W // 2)
        from tensorrt_llm._torch.visual_gen.attention.backends import fmha_v2_flux3 as lib

        self.sf_bytes = lib.nvfp4_scale_bytes(rows, W)
        self.gs = Guarded(self.sf_bytes)
        self.sf = self.gs.payload

    def reset(self):
        self.fp4.fill_(FILL)
        self.sf.fill_(FILL)

    def check(self):
        return {k: v for k, v in (("fp4", self.gf.check()), ("sf", self.gs.check())) if v}


def call_nfo(lib, mod, inp, out, gscale):
    sc = inp["scales"]
    lib.fmha_v2_flux3_d128_nvfp4_out(inp["q"], inp["k"], inp["v"], out.fp4, out.sf, gscale, H, D,
                                     inp["S"], scale_bmm1_d=sc[3:4], scale_bmm2_d=sc[2:3],
                                     module=mod)


def call_bf16(lib, mod, inp, o):
    sc = inp["scales"]
    lib.fmha_v2_flux3_d128(inp["q"], inp["k"], inp["v"], o, H, D, inp["S"],
                           scale_bmm1_d=sc[3:4], scale_bmm2_d=sc[2:3], causal=False, lse=None,
                           module=mod)


def snapshot(outs):
    return [t.clone() for t in outs]


def run_seq(fns, tensors, checks, repeats, failures, name):
    """eager once (reference), `repeats` more eager calls, then graph replay x3; compare + guards."""
    for f in fns:
        f()
    torch.cuda.synchronize()
    ref = snapshot(tensors)
    dirty = [c() for c in checks]
    rep_bad = 0
    for r in range(repeats):
        for f in fns:
            f()
        if (r + 1) % 10 == 0 or r == repeats - 1:
            torch.cuda.synchronize()
            rep_bad += sum(int(not torch.equal(a, b)) for a, b in zip(tensors, ref))
            dirty += [c() for c in checks]
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
    gbad = 0
    for _ in range(3):
        for t in tensors:
            t.fill_(0x11 if t.dtype == torch.uint8 else 1)
        g.replay()
        torch.cuda.synchronize()
        gbad += sum(int(not torch.equal(a, b)) for a, b in zip(tensors, ref))
        dirty += [c() for c in checks]
    d = [x for x in dirty if x]
    row = dict(calls=len(fns), repeat_mismatch=rep_bad, graph_mismatch=gbad, guard_dirty=d[:4])
    if rep_bad or gbad or d:
        failures.append(name)
    del g
    return ref, row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--unfixed", action="store_true")
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--san", action="store_true", help="minimal launches for compute-sanitizer")
    p.add_argument("--heads", type=int, default=24)
    p.add_argument("--case", default="all",
                   help="--san only: s4096|s4100|s4176|graph|chain|census|all")
    args = p.parse_args()
    global H, W
    H = args.heads
    W = H * D
    from tensorrt_llm._torch.visual_gen.attention.backends import fmha_v2_flux3 as lib

    res = {"probe": probe(), "routed_levers": list(lib.ROUTED_LEVERS), "heads": H}
    sample = make_qkv(4176, 7, False)
    mods = build_modules(lib, sample, args.unfixed)
    gscale = torch.tensor([448.0 * 6.0 / 8.0], dtype=torch.float32, device="cuda")
    failures, rows = [], {}

    if args.san and args.case != "all":
        # one case per process (b186_san.py runs each tool x case as its own subprocess).
        # Heartbeat + synchronized wall time after every launch: under a kns= filter only the
        # matched kernel is instrumented, so its launch is orders of magnitude slower than the
        # unmatched ones -- that is the in-process proof the filter matched.
        if args.case.startswith("s"):
            need = (int(args.case[1:]),)
        elif args.case in ("graph", "chain"):
            need = (4096, 4176)
        else:
            need = (4096, 4100, 4176)
        inps = {S: make_qkv(S, 100 + S, True) for S in need}
        outs = {S: NfoOut(S) for S in inps}
        bo = {S: torch.empty((1, S, H, D), device="cuda", dtype=torch.bfloat16) for S in inps}
        torch.cuda.synchronize()
        times = []

        def timed(tag, fn):
            t0 = time.time()
            fn()
            torch.cuda.synchronize()
            dt = time.time() - t0
            times.append((tag, round(dt, 4)))
            log(f"[hb] {tag} {dt:.4f}s")

        def g3(S):
            return lambda: call_bf16(lib, mods["bf16_g3"][1], inps[S], bo[S])

        def nfo(S):
            return lambda: call_nfo(lib, mods["nfo_fix"][1], inps[S], outs[S], gscale)

        out = {"mode": "san", "case": args.case, "heads": H}
        if args.case == "census":
            cen = {}
            for S in need:
                cen[f"bf16_g3_S{S}"] = names(g3(S))
                cen[f"nfo_fix_S{S}"] = names(nfo(S))
            out["census"] = cen
            log(f"[census] {json.dumps(cen)}")
        elif args.case.startswith("s"):
            S = int(args.case[1:])
            timed(f"g3_bf16_S{S}", g3(S))
            timed(f"nfo_S{S}", nfo(S))
        elif args.case == "graph":
            fns = [g3(4176), g3(4096)]
            for f in fns:  # eager warm (also sanitized)
                f()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for f in fns:
                    f()
            log("[hb] graph captured (g3 S4176 + g3 S4096)")
            timed("graph_replay_x1", g.replay)
            del g
        elif args.case == "chain":
            fns = [("g3_bf16_S4096", g3(4096))] + [("nfo_S4096", nfo(4096))] * 4 + \
                [("nfo_S4176", nfo(4176))] * 28
            for i, (tag, f) in enumerate(fns):
                timed(f"eager{i}_{tag}", f)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _, f in fns:
                    f()
            log("[hb] chain graph captured (33 calls)")
            timed("chain_graph_replay_x1", g.replay)
            del g
        out["launch_wall_s"] = times
        print(json.dumps(out))
        return

    if args.san:
        inps = {S: make_qkv(S, 100 + S, True) for S in (4096, 4100, 4176)}
        outs = {S: NfoOut(S) for S in inps}
        bo = {S: torch.empty((1, S, H, D), device="cuda", dtype=torch.bfloat16) for S in inps}
        for S in inps:
            call_nfo(lib, mods["nfo_fix"][1], inps[S], outs[S], gscale)
            call_bf16(lib, mods["bf16_g3"][1], inps[S], bo[S])
            torch.cuda.synchronize()
            log(f"[san] S{S} done")
        fns = [lambda: call_bf16(lib, mods["bf16_g3"][1], inps[4096], bo[4096])]
        fns += [lambda: call_nfo(lib, mods["nfo_fix"][1], inps[4096], outs[4096], gscale)] * 4
        fns += [lambda: call_nfo(lib, mods["nfo_fix"][1], inps[4176], outs[4176], gscale)] * 28
        for f in fns:
            f()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for f in fns:
                f()
        g.replay()
        torch.cuda.synchronize()
        log("[san] chain eager + graph done")
        print(json.dumps({"passed": True, "mode": "san"}))
        return

    # census
    census = {}
    for S in (4096, 4176):
        inp = make_qkv(S, 11 + S, False)
        o = torch.empty((1, S, H, D), device="cuda", dtype=torch.bfloat16)
        for k in ("bf16_main", "bf16_g3"):
            call_bf16(lib, mods[k][1], inp, o)
            census[f"{k}_S{S}"] = names(lambda: call_bf16(lib, mods[k][1], inp, o))
        out = NfoOut(S)
        for k in ("nfo_main", "nfo_fix"):
            call_nfo(lib, mods[k][1], inp, out, gscale)
            census[f"{k}_S{S}"] = names(lambda: call_nfo(lib, mods[k][1], inp, out, gscale))
    res["census"] = census
    log(f"[census] {json.dumps(census)}")

    bitwise = {}
    for peaky in (False, True):
        tag = "peaky" if peaky else "randn"
        inps = {S: make_qkv(S, 200 + S + int(peaky), peaky) for S in (4096, 4100, 4176)}
        for S, inp in inps.items():
            refs = {}
            for k in ("nfo_main", "nfo_fix"):
                out = NfoOut(S)
                out.reset()
                ref, row = run_seq([lambda m=mods[k][1], o=out: call_nfo(lib, m, inp, o, gscale)],
                                   [out.fp4, out.sf], [out.check], args.repeats, failures,
                                   f"{tag}_nfo_{k}_S{S}")
                valid = (S * (W // 16))  # every valid scale byte is written; padding rows zeroed
                row["sf_unwritten_0xA5_bytes"] = int((ref[1] == FILL).sum())
                rows[f"{tag}_nfo_{k}_S{S}"] = row
                log(f"[{tag} nfo {k} S{S}] {json.dumps(row)}")
                refs[k] = ref
            eq = all(torch.equal(a, b) for a, b in zip(refs["nfo_main"], refs["nfo_fix"]))
            bitwise[f"{tag}_nfo_S{S}"] = eq
            if not eq:
                failures.append(f"bitwise_{tag}_nfo_S{S}")
            brefs = {}
            for k in ("bf16_main", "bf16_g3"):
                go = Guarded(S * W * 2)
                o = go.payload.view(torch.bfloat16).view(1, S, H, D)
                ref, row = run_seq([lambda m=mods[k][1], oo=o: call_bf16(lib, m, inp, oo)], [o],
                                   [go.check], args.repeats, failures, f"{tag}_bf16_{k}_S{S}")
                rows[f"{tag}_bf16_{k}_S{S}"] = row
                log(f"[{tag} bf16 {k} S{S}] {json.dumps(row)}")
                brefs[k] = ref[0]
            eq = torch.equal(brefs["bf16_main"].view(torch.int16), brefs["bf16_g3"].view(torch.int16))
            bitwise[f"{tag}_bf16_S{S}"] = eq
            if not eq:
                failures.append(f"bitwise_{tag}_bf16_S{S}")
            inp["guard"].check() and failures.append(f"{tag}_qkv_guard_S{S}")
            if not torch.equal(inp["guard"].payload, inp["pristine"]):
                failures.append(f"{tag}_inputs_changed_S{S}")
        # served pass chain
        crefs = {}
        for arm, nm, bm in (("main", "nfo_main", "bf16_main"), ("fix", "nfo_fix", "bf16_g3")):
            seq = [("b", 4096)] + [("n", 4096)] * 4 + [("n", 4176)] * 28
            fns, tens, checks = [], [], []
            for kind, S in seq:
                if kind == "b":
                    go = Guarded(S * W * 2)
                    o = go.payload.view(torch.bfloat16).view(1, S, H, D)
                    fns.append(lambda m=mods[bm][1], i=inps[S], oo=o: call_bf16(lib, m, i, oo))
                    tens.append(o)
                    checks.append(go.check)
                else:
                    out = NfoOut(S)
                    fns.append(lambda m=mods[nm][1], i=inps[S], oo=out: call_nfo(lib, m, i, oo,
                                                                                  gscale))
                    tens += [out.fp4, out.sf]
                    checks.append(out.check)
            ref, row = run_seq(fns, tens, checks, 2, failures, f"{tag}_chain_{arm}")
            rows[f"{tag}_chain_{arm}"] = row
            log(f"[{tag} chain {arm}] {json.dumps(row)}")
            crefs[arm] = ref
        eq = all(torch.equal(a, b) for a, b in zip(crefs["main"], crefs["fix"]))
        bitwise[f"{tag}_chain"] = eq
        if not eq:
            failures.append(f"bitwise_{tag}_chain")
    res["bitwise_fix_vs_main"] = bitwise
    res["cases"] = rows

    sass = {}
    for k, (lev, _) in mods.items():
        if k == "nfo_unfix":
            continue
        r = b165_g3.resource_usage(lib, lev)
        sass[k] = {"sass_sha": r.get("sass_sha"),
                   "functions": {n: v for n, v in r.get("functions", {}).items() if "flux3" in n}}
    res["sass"] = sass
    log(f"[sass] {json.dumps(sass)}")

    if args.unfixed:
        un = {}
        for S in (4176, 4096):
            inp = make_qkv(S, 300 + S, False)
            out = NfoOut(S, post=40 * MB)
            out.reset()
            un[f"S{S}_kernels"] = names(lambda: call_nfo(lib, mods["nfo_unfix"][1], inp, out, gscale))
            torch.cuda.synchronize()
            d = out.check()
            un[f"S{S}"] = dict(fp4_bytes=out.gf.n, bf16_o_bytes=S * W * 2, guard=d,
                               sf_all_unwritten=bool((out.sf == FILL).all()))
            log(f"[unfixed S{S}] {json.dumps(un[f'S{S}'])} kernels {un[f'S{S}_kernels']}")
        res["unfixed"] = un
    res["passed"] = not failures
    res["failures"] = failures
    res["compared"] = ("nfo_fix (served nfo module, G3 guard) vs nfo_main (main's nfo module): fp4+sf "
                       "bitwise; bf16_g3 vs bf16_main bitwise; 0xA5 guards around fp4/sf/O/qkv after "
                       "every call/replay; 100 repeats == call 0; graph == eager")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
