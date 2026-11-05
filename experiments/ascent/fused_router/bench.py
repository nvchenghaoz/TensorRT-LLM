"""Paired timing for idea 67 (run_gpu_pair: BASE = main tree, CAND = branch tree), plus an in-process
interleaved A/B mode (--both, branch tree only: main's ops are still there).

One GPU, one MoE rank: 40 distinct router weights [384,5120] bf16 + fp32 biases, 40 distinct sets of real mHC K1
inputs (x_prev, residual [M,4,5120], post, comb, pre, norm weight), the real Sm12xW4a8DecodeState.run (FlashInfer
SM12x MoE decode chain) for the MoE arms. Per layer:
  BASE : router_gemm_mma_kernel (main's pre-wait depth) -> gate_forward_kernel -> run() = route_assign_decode +
         _mxfp8_q0_route_direct + fc1_act_q1 + fc2_finalize
  CAND1: dsv4FusedRouterKernel<M,false> (Stage 1: GEMV + gate)         -> run() as BASE
  CAND : dsv4FusedRouterKernel<M,true>  (Stage 2: + route + q0 into the route workspace) -> run() = fc1 + fc2 only
40-layer CUDA graphs, read-only 512 MB flush before every replay, every graph of a case replayed interleaved (ABBA):
  REAL_RUN : real K1 (_mhc_lagged_layer_input_kernel, PDL, early trigger; writes x) -> router row -> run   [JUDGED]
  REAL     : real K1 -> router row only (BASE / CAND1)
  K1_NULL  : real K1 -> null kernel (K1 with nothing behind it)
  P1c1/P1c4: K1 stand-in (spin 4.9/3.3 us, trigger at start) -> router row -> run   (context)
  P0       : router row -> run, no K1 (context)
Cases: EP4 rank 0 (96 x 2304, slot_start 0) at M=1/2/4/8 and EP2xTP2 moe_ep_rank 1 (192 x 1152, slot_start 192)
at M=1/2/4 (labels ep2tp2_M). kernel_ms = REAL_RUN 40-layer graph. Routing of every layer checked bitwise vs main.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (HIDDEN, LAYERS, SPIN_C1_NS, SPIN_C4_NS, biases, capture, fused, fused_route,  # noqa: E402
                    gate_main, has_fused, k1_standin, log, make_decode_state, router_main, router_weights)

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

WARMUP, ITERS = 5, 40
K1_P = 224  # fused_half_fma bigfuse 256 tactic's phase-2 threads (4 warps per K1 program)
CASES = [("ep4", 96, 2304, 0, m) for m in (1, 2, 4, 8)] + [("ep2tp2", 192, 1152, 192, m) for m in (1, 2, 4)]


@triton.jit
def _null_kernel(o_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(o_ptr + offs, offs.to(tl.float32))


def make_k1_inputs(m, gen, dev):
    x = torch.randn(m, HIDDEN, generator=gen, device=dev).to(torch.bfloat16)
    res = torch.randn(m, 4, HIDDEN, generator=gen, device=dev).to(torch.bfloat16)
    post = (2.0 * torch.rand(m, 4, generator=gen, device=dev)).float().contiguous()
    comb = torch.rand(m, 4, 4, generator=gen, device=dev)
    comb = (comb / comb.sum(-2, keepdim=True)).float().contiguous()
    pre = (torch.rand(m, 4, generator=gen, device=dev) + 1e-6).float().contiguous()
    nw = (1.0 + 0.1 * torch.randn(HIDDEN, generator=gen, device=dev)).to(torch.bfloat16)
    return x, res, post, comb, pre, nw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--both", action="store_true", help="time BASE, CAND1 and CAND in one process (branch tree)")
    ap.add_argument("--layouts", default="ep4,ep2tp2")
    ap.add_argument("--no-context", action="store_true")
    args = ap.parse_args()
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.modules.mhc.mhc_lagged_split import mhc_lagged_layer_input

    cand_tree = has_fused()
    has_route = hasattr(torch.ops.trtllm, "dsv4_fused_router_route")
    arms = ["BASE", "CAND1", "CAND"] if args.both else (["CAND1", "CAND"] if cand_tree else ["BASE"])
    assert all(a == "BASE" for a in arms) or (cand_tree and has_route)
    kname = {"BASE": "router_gemm_mma_kernel+gate_forward_kernel | route: _route_assign_decode_kernel+"
                     "_mxfp8_q0_route_direct_kernel",
             "CAND1": "dsv4FusedRouterKernel<M,false> | route: _route_assign_decode_kernel+_mxfp8_q0_route_direct_kernel",
             "CAND": "dsv4FusedRouterKernel<M,true> (route+q0 fused) | route: none"}
    main_arm = arms[-1]
    dev = "cuda"
    log("arms", arms, "judged", main_arm, torch.cuda.get_device_name(), {a: kname[a] for a in arms})
    gen = torch.Generator(device=dev).manual_seed(4242)
    rws = router_weights(gen, LAYERS)
    bs = biases(gen, LAYERS)
    counters = [torch.zeros(1, dtype=torch.int32, device=dev) for _ in range(LAYERS)]
    flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device=dev).fill_(1.0)
    null_out = torch.empty(128, dtype=torch.float32, device=dev)

    out, extra = {}, {}
    layouts = args.layouts.split(",")
    state_key, state = None, None
    for lay, e, inter, slot, m in CASES:
        if lay not in layouts:
            continue
        label = str(m) if lay == "ep4" else f"{lay}_{m}"
        if state_key != (e, inter, slot):
            state = None
            torch.cuda.empty_cache()
            state, _ = make_decode_state(e, inter, slot)
            state_key = (e, inter, slot)
            log(f"decode state E={state.num_experts} inter={state.inter} slot_start={state.slot_start}")

        def row(arm, li, x, with_run):
            if arm == "CAND" and with_run:
                ids, wts, _ = fused_route(state, x, rws[li], bs[li], counters[li])
            elif arm in ("CAND", "CAND1"):
                ids, wts, _ = fused(x, rws[li], bs[li], counters[li])
            else:
                ids, wts = gate_main(router_main(x, rws[li]), bs[li])
            if with_run:
                state.run(x, ids, wts)
            return ids, wts

        k1in = [make_k1_inputs(m, gen, dev) for _ in range(LAYERS)]
        xs = [torch.empty(m, HIDDEN, dtype=torch.bfloat16, device=dev) for _ in range(LAYERS)]
        srcs = [torch.randn((m, HIDDEN), generator=gen, device=dev).to(torch.bfloat16) for _ in range(LAYERS)]

        def k1(li):
            xp, res, post, comb, pre, nw = k1in[li]
            mhc_lagged_layer_input(xp, res, post, comb, pre, nw, 1e-6, K1_P, enable_pdl=True, out=xs[li])

        for li in range(LAYERS):
            k1(li)
        torch.cuda.synchronize()
        # Routing bitwise vs main's pair, and (CAND) the MoE output vs the plain-route run (warms the workspace).
        nbad = 0
        for li in range(LAYERS):
            ri, rw = gate_main(router_main(xs[li], rws[li]), bs[li])
            ref_out = state.run(xs[li], ri, rw).clone()
            for arm in arms:
                ci, cw = row(arm, li, xs[li], False)
                ok = torch.equal(ri, ci) and torch.equal(rw.view(torch.int32), cw.view(torch.int32))
                if arm == "CAND":
                    fi, fw, _ = fused_route(state, xs[li], rws[li], bs[li], counters[li])
                    assert state._prerouted is not None
                    o = state.run(xs[li], fi, fw)
                    ok = ok and bool(torch.isfinite(o.float()).all()) and \
                        (o.float() - ref_out.float()).abs().max().item() <= 1e-2 * max(1.0, ref_out.float().abs().max().item())
                nbad += int(not ok)
        torch.cuda.synchronize()
        log(f"{label}: routing/output mismatches {nbad}")

        def mk(arm, kind):
            def fn():
                for li in range(LAYERS):
                    if kind in ("REAL", "REAL_RUN", "K1_NULL"):
                        k1(li)
                    elif kind == "P1c1":
                        k1_standin(srcs[li], xs[li], SPIN_C1_NS)
                    elif kind == "P1c4":
                        k1_standin(srcs[li], xs[li], SPIN_C4_NS)
                    if kind == "K1_NULL":
                        _null_kernel[(m,)](null_out, BLOCK=128, num_warps=4)
                        continue
                    row(arm, li, xs[li], kind != "REAL")
            return fn

        kinds = ["REAL_RUN"] + ([] if args.no_context or lay != "ep4" else ["REAL", "P1c1", "P1c4", "P0"])
        graphs = {}
        for arm in arms:
            for kind in kinds:
                if arm == "CAND" and kind == "REAL":
                    continue  # Stage 2 without its run() is not a router row; CAND1:REAL is
                graphs[f"{arm}:{kind}"] = capture(mk(arm, kind))[0]
        if lay == "ep4":
            graphs["K1_NULL"] = capture(mk(arms[0], "K1_NULL"))[0]
        names = list(graphs)
        ts = {k: [] for k in names}
        for it in range(WARMUP + ITERS):
            order = names if it % 2 == 0 else names[::-1]  # ABBA
            for name in order:
                flush.sum()
                a = torch.cuda.Event(enable_timing=True)
                b = torch.cuda.Event(enable_timing=True)
                a.record()
                graphs[name].replay()
                b.record()
                torch.cuda.synchronize()
                if it >= WARMUP:
                    ts[name].append(a.elapsed_time(b))
        del graphs
        r = {}
        for k, v in ts.items():
            v.sort()
            med = v[len(v) // 2]
            p10, p90 = v[int(0.1 * (len(v) - 1))], v[int(0.9 * (len(v) - 1))]
            r[k] = {"ms": med, "us_layer": med * 1000.0 / LAYERS, "p10_us_layer": p10 * 1000.0 / LAYERS,
                    "p90_us_layer": p90 * 1000.0 / LAYERS, "noise": (p90 - p10) / med}
        for arm in arms:
            log(f"{arm} {label} E={e} inter={inter} slot={slot} kernels[{kname[arm]}; K1=_mhc_lagged_layer_input_kernel] "
                + " ".join(f"{k.split(':')[1]}={x['us_layer']:.3f}us/layer[{x['p10_us_layer']:.3f},{x['p90_us_layer']:.3f}]"
                           for k, x in r.items() if k.startswith(arm + ":")))
        if "K1_NULL" in r:
            log(f"{label} K1_NULL (K1 -> null) {r['K1_NULL']['us_layer']:.3f} us/layer")
        if args.both:
            for c in ("CAND1", "CAND"):
                d = {k: r[f"{c}:{k}"]["us_layer"] - r[f"BASE:{k}"]["us_layer"] for k in kinds if f"{c}:{k}" in r}
                log(f"{label} {c}-BASE us/layer: " + " ".join(f"{k}={v:+.3f}" for k, v in d.items()))
        out[label] = {"kernel_ms": r[f"{main_arm}:REAL_RUN"]["ms"], "noise_floor": r[f"{main_arm}:REAL_RUN"]["noise"]}
        extra[label] = {k: {kk: round(vv, 4) for kk, vv in x.items()} for k, x in r.items()}
        extra[label]["mismatches"] = nbad
    print(json.dumps({"batch": out, "output": {"class": "exact"}, "arms": arms, "judged_arm": main_arm,
                      "kernels": {a: kname[a] for a in arms}, "chains": extra}))


if __name__ == "__main__":
    with torch.no_grad():
        main()
