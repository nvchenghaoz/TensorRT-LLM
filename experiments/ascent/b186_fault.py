"""Branch 186 (idea 270): G3 fault hunt -- route census, guard-band canaries, eager/graph chains.

Modes
  census  kernel names per route via flashinfer.run_fmha_v2_fp8_sm12x (the tree's own router,
          env TRTLLM_FMHA_V2_FLUX3_G3 decides) at S4096/4100/4176, served qkv8 layout.
  canary  module A (ROUTED_LEVERS w/o g3) and B (+g3). Q/K/V are [1,S,H,128] E4M3 views of one
          [3,S,H*128] qkv8 carved from a guarded buffer; O carved from a guarded buffer
          (1 MB guards of 0xA5 before/after). Per S: eager x3, graph replay x3; then the served
          chain 28xS4176 + 4xS4096 eager and graph x3. Guards (O and qkv) checked after every
          call/replay; B == A bitwise; repeats == call 0; graph == eager. Inputs randn and peaky.
  san     minimal launches for compute-sanitizer: each S once eager (B) plus the chain once.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

D = 128
GUARD = 1 << 20
FILL = 0xA5


def log(*a):
    print(*a, flush=True)


class Guarded:
    """A uint8 buffer [GUARD | payload | GUARD]; payload is the tensor under test."""

    def __init__(self, nbytes):
        self.n = nbytes
        self.buf = torch.full((GUARD + nbytes + GUARD,), FILL, dtype=torch.uint8, device="cuda")
        self.payload = self.buf[GUARD:GUARD + nbytes]

    def check(self, row_bytes=None):
        pre = self.buf[:GUARD] != FILL
        post = self.buf[GUARD + self.n:] != FILL
        res = {}
        for name, m, base in (("pre", pre, -GUARD), ("post", post, self.n)):
            idx = torch.nonzero(m).flatten()
            if idx.numel():
                first, last = int(idx[0]) + base, int(idx[-1]) + base
                d = dict(count=int(idx.numel()), first=first, last=last)
                if row_bytes:
                    d["first_row"], d["last_row"] = first // row_bytes, last // row_bytes
                res[name] = d
        return res


def make_qkv(S, H, seed, peaky):
    from tensorrt_llm._torch.visual_gen.attention.backends.fp8_sm12x import quantize_qkv_e4m3

    g = torch.Generator(device="cuda").manual_seed(seed)
    buf = torch.randn((3, 1, S, H, D), generator=g, device="cuda", dtype=torch.float32) * 2.0
    if peaky:
        mask = torch.rand((3, 1, S, H, 1), generator=g, device="cuda") < 0.01
        buf = torch.where(mask, buf * 40.0, buf)
    buf = buf.to(torch.bfloat16)
    q, k, v = buf.unbind(0)
    q8, k8, v8, scales = quantize_qkv_e4m3(q, k, v, 1.0 / math.sqrt(D))
    W = H * D
    gq = Guarded(3 * S * W)
    qkv8 = gq.payload.view(torch.float8_e4m3fn).view(3, S, W)
    for i, t in enumerate((q8, k8, v8)):
        qkv8[i].copy_(t.reshape(S, W))
    planes = [p.view(1, S, H, D) for p in qkv8.unbind(0)]
    assert all(p.is_contiguous() for p in planes)
    return dict(S=S, H=H, q=planes[0], k=planes[1], v=planes[2], scales=scales, guard=gq,
                pristine=gq.payload.clone())


def new_out(S, H):
    go = Guarded(S * H * D * 2)
    o = go.payload.view(torch.bfloat16).view(1, S, H, D)
    o.fill_(0)
    return go, o


def call_mod(lib, mod, inp, out):
    sc = inp["scales"]
    lib.fmha_v2_flux3_d128(inp["q"], inp["k"], inp["v"], out, inp["H"], D, inp["S"],
                           scale_bmm1_d=sc[3:4], scale_bmm2_d=sc[2:3], causal=False, lse=None,
                           module=mod)


def call_fib(fib, inp, out):
    sc = inp["scales"]
    fib.run_fmha_v2_fp8_sm12x(inp["q"], inp["k"], inp["v"], out, inp["H"], D, inp["S"],
                              scale_bmm1_d=sc[3:4], scale_bmm2_d=sc[2:3], causal=False,
                              return_lse=False)


def bits(t):
    return t.view(torch.int16)


def modules(sample):
    from tensorrt_llm._torch.visual_gen.attention.backends import fmha_v2_flux3 as lib

    lev = tuple(lib.load_trim_levers(tuple(lib.ROUTED_LEVERS), sample["k"], sample["v"]))
    lev_b = lev if "g3" in lev else (*lev, "g3")
    lev_a = tuple(x for x in lev_b if x != "g3")
    t0 = time.time()
    ma = lib.get_flux3_d128_module(lev_a)
    mb = lib.get_flux3_d128_module(lev_b)
    log(f"[build] A {lib._tag(lev_a, 0)} B {lib._tag(lev_b, 0)} {time.time() - t0:.1f}s")
    return lib, lev_a, lev_b, ma, mb


def kernel_names(fn):
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return {e.key: e.count for e in prof.key_averages() if "fmha" in e.key}


def mode_census(args):
    from tensorrt_llm._torch.visual_gen.attention.backends import flashinfer as fib
    from tensorrt_llm._torch.visual_gen.attention.backends import fmha_v2_flux3 as lib

    env = os.environ.get("TRTLLM_FMHA_V2_FLUX3_G3", "<unset>")
    res = {"TRTLLM_FMHA_V2_FLUX3_G3": env, "ROUTED_LEVERS": list(lib.ROUTED_LEVERS),
           "has_nvfp4_out_lever": "nvfp4_out" in lib._LEVER_FLAGS, "routes": {}}
    for S in args.seqs:
        inp = make_qkv(S, 24, 11 + S, False)
        _, o = new_out(S, 24)
        call_fib(fib, inp, o)
        res["routes"][f"bf16_out_S{S}"] = kernel_names(lambda: call_fib(fib, inp, o))
    log(json.dumps(res))
    print(json.dumps({"passed": True, **res}))


def run_case(name, fn_list, outs, guards, inps, repeats_graph, failures, rows):
    """fn_list: callables (one per call); outs: outputs; returns eager reference."""
    for o in outs:
        o.fill_(0)
    for f in fn_list:
        f()
    torch.cuda.synchronize()
    eager = [o.clone() for o in outs]
    dirty = [g.check(inp["H"] * D * 2) for g, inp in zip(guards, inps)]
    qd = [inp["guard"].check() for inp in inps]
    # repeats
    rep_bad = 0
    for r in range(2):
        for f in fn_list:
            f()
        torch.cuda.synchronize()
        rep_bad += sum(int(not torch.equal(bits(o), bits(e))) for o, e in zip(outs, eager))
        dirty += [g.check(inp["H"] * D * 2) for g, inp in zip(guards, inps)]
    # graph
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for f in fn_list:
            f()
    torch.cuda.current_stream().wait_stream(st)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in fn_list:
            f()
    graph_bad = 0
    for r in range(repeats_graph):
        for o in outs:
            o.fill_(1)
        g.replay()
        torch.cuda.synchronize()
        graph_bad += sum(int(not torch.equal(bits(o), bits(e))) for o, e in zip(outs, eager))
        dirty += [gd.check(inp["H"] * D * 2) for gd, inp in zip(guards, inps)]
    qd += [inp["guard"].check() for inp in inps]
    inputs_ok = all(torch.equal(inp["guard"].payload, inp["pristine"]) for inp in inps)
    d_any = [d for d in dirty if d]
    q_any = [d for d in qd if d]
    row = dict(calls=len(fn_list), repeat_mismatch=rep_bad, graph_mismatch=graph_bad,
               o_guard_dirty=d_any[:4], qkv_guard_dirty=q_any[:4], inputs_unchanged=inputs_ok)
    rows[name] = row
    log(f"[{name}] {json.dumps(row)}")
    if rep_bad or graph_bad or d_any or q_any or not inputs_ok:
        failures.append(name)
    del g
    return eager


def mode_canary(args):
    H = args.heads
    torch.manual_seed(0)
    sample = make_qkv(4176, H, 7, False)
    lib, lev_a, lev_b, ma, mb = modules(sample)
    failures, rows, bitwise = [], {}, {}
    for peaky in (False, True):
        tag = "peaky" if peaky else "randn"
        inps = {S: make_qkv(S, H, 100 + S + int(peaky), peaky) for S in args.seqs}
        for S in args.seqs:
            inp = inps[S]
            res = {}
            for arm, mod in (("A", ma), ("B", mb)):
                go, o = new_out(S, H)
                e = run_case(f"{tag}_S{S}_{arm}", [lambda m=mod, i=inp, oo=o: call_mod(lib, m, i, oo)],
                             [o], [go], [inp], 3, failures, rows)
                res[arm] = e[0]
            eq = torch.equal(bits(res["A"]), bits(res["B"]))
            fin = bool(torch.isfinite(res["B"].float()).all())
            bitwise[f"{tag}_S{S}"] = dict(B_eq_A=eq, finite=fin)
            log(f"[bitwise {tag} S{S}] B==A {eq} finite {fin}")
            if not eq:
                failures.append(f"bitwise_{tag}_S{S}")
        # served chain 28 x S4176 + 4 x S4096 (distinct outputs, shared inputs per S)
        chain = [4176] * 28 + [4096] * 4
        for S in (4176, 4096):
            if S not in inps:
                inps[S] = make_qkv(S, H, 100 + S + int(peaky), peaky)
        res = {}
        for arm, mod in (("A", ma), ("B", mb)):
            gos = [new_out(S, H) for S in chain]
            fns = [lambda m=mod, i=inps[S], oo=o: call_mod(lib, m, i, oo)
                   for S, (_, o) in zip(chain, gos)]
            e = run_case(f"{tag}_chain_{arm}", fns, [o for _, o in gos], [g for g, _ in gos],
                         [inps[S] for S in chain], 3, failures, rows)
            res[arm] = e
        eq = all(torch.equal(bits(a), bits(b)) for a, b in zip(res["A"], res["B"]))
        bitwise[f"{tag}_chain"] = dict(B_eq_A=eq)
        log(f"[bitwise {tag} chain] B==A {eq}")
        if not eq:
            failures.append(f"bitwise_{tag}_chain")
    out = {"passed": not failures, "failures": failures, "heads": H, "levers_A": list(lev_a),
           "levers_B": list(lev_b), "bitwise": bitwise, "cases": rows,
           "compared": "G3 (B) vs main's routed 4-warp kernel (A) bitwise; 0xA5 1 MB guard bands "
                       "around O and qkv8 after every call/replay; repeats/graph == eager"}
    if args.resources:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import b165_g3

        out["resources_B"] = b165_g3.resource_usage(lib, lev_b)
        out["resources_A"] = b165_g3.resource_usage(lib, lev_a)
        out["resources_B"].pop("sass_stl_ldl_ctx", None)
        out["resources_A"].pop("sass_stl_ldl_ctx", None)
    print(json.dumps(out))


def mode_san(args):
    H = args.heads
    sample = make_qkv(4176, H, 7, False)
    lib, lev_a, lev_b, ma, mb = modules(sample)
    inps = {S: make_qkv(S, H, 100 + S, True) for S in sorted(set(args.seqs) | {4096, 4176})}
    outs = {S: torch.empty((1, S, H, D), device="cuda", dtype=torch.bfloat16) for S in inps}
    for S in args.seqs:
        call_mod(lib, mb, inps[S], outs[S])
        torch.cuda.synchronize()
        log(f"[san] S{S} done")
    chain = [4176] * 28 + [4096] * 4
    for S in chain:
        call_mod(lib, mb, inps[S], outs[S])
    torch.cuda.synchronize()
    log("[san] chain eager done")
    if args.graph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for S in chain:
                call_mod(lib, mb, inps[S], outs[S])
        g.replay()
        torch.cuda.synchronize()
        log("[san] chain graph done")
    print(json.dumps({"passed": True, "mode": "san", "heads": H}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["census", "canary", "san"])
    p.add_argument("--heads", type=int, default=24)
    p.add_argument("--seqs", type=int, nargs="+", default=[4096, 4100, 4176])
    p.add_argument("--resources", action="store_true")
    p.add_argument("--graph", action="store_true")
    args = p.parse_args()
    log(f"device {torch.cuda.get_device_name()} mode {args.mode} heads {args.heads} "
        f"G3env {os.environ.get('TRTLLM_FMHA_V2_FLUX3_G3')} "
        f"nocache {os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING')}")
    {"census": mode_census, "canary": mode_canary, "san": mode_san}[args.mode](args)


if __name__ == "__main__":
    main()
