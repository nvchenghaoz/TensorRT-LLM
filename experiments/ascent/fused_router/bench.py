"""Paired timing for idea 67 (run_gpu_pair: BASE = main tree, CAND = branch tree), plus an in-process
interleaved A/B mode (--both, branch tree only: main's ops are still there).

One GPU, EP4 rank 0: 40 distinct router weights [384,5120] bf16 + fp32 biases, 40 distinct sets of real mHC K1
inputs (x_prev, residual [M,4,5120], post, comb, pre, norm weight; built as experiments/ascent/mhc_split/k1_check.py),
the real Sm12xW4a8DecodeState.run for the MoE arms. Per layer the router row is
  BASE: router_gemm_mma_kernel (main's pre-wait depth) -> gate_forward_kernel
  CAND: dsv4FusedRouterKernel
40-layer CUDA graphs, read-only 512 MB flush before every replay, every graph of an M replayed interleaved:
  REAL     : real K1 (_mhc_lagged_layer_input_kernel, PDL, early trigger; writes x) -> router row   [JUDGED]
  REAL_RUN : real K1 -> router row -> run (FlashInfer MoE chain)
  K1_NULL  : real K1 -> null kernel (non-PDL, gate-sized)  (K1 with nothing behind it; same in both arms)
  P1c1/P1c4: K1 stand-in (spin 4.9/3.3 us, trigger at start) -> router row -> run   (context)
  P0       : router row -> run, no K1 (context)
kernel_ms = REAL 40-layer graph (per-layer cost x 40). Routing of every layer checked bitwise against main's pair.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (HIDDEN, LAYERS, SPIN_C1_NS, SPIN_C4_NS, biases, capture, fused, gate_main, has_fused,  # noqa: E402
                    k1_standin, log, make_decode_state, router_main, router_weights)

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

WARMUP, ITERS = 5, 40
K1_P = 224  # fused_half_fma bigfuse 256 tactic's phase-2 threads (4 warps per K1 program)


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
    ap.add_argument("--both", action="store_true", help="time BASE and CAND row in one process (branch tree)")
    ap.add_argument("--ms", default="1,2,4,8")
    ap.add_argument("--no-context", action="store_true")
    args = ap.parse_args()
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.modules.mhc.mhc_lagged_split import mhc_lagged_layer_input

    cand_tree = has_fused()
    arms = ["BASE", "CAND"] if args.both else (["CAND"] if cand_tree else ["BASE"])
    assert "CAND" not in arms or cand_tree
    kname = {"BASE": "router_gemm_mma_kernel+gate_forward_kernel", "CAND": "dsv4FusedRouterKernel"}
    dev = "cuda"
    log("arms", arms, torch.cuda.get_device_name(), {a: kname[a] for a in arms})
    state, _ = make_decode_state()
    gen = torch.Generator(device=dev).manual_seed(4242)
    rws = router_weights(gen, LAYERS)
    bs = biases(gen, LAYERS)
    counters = [torch.zeros(1, dtype=torch.int32, device=dev) for _ in range(LAYERS)]
    flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device=dev).fill_(1.0)
    null_out = torch.empty(128, dtype=torch.float32, device=dev)

    def row(arm, li, x):
        if arm == "CAND":
            ids, wts, _ = fused(x, rws[li], bs[li], counters[li])
            return ids, wts
        return gate_main(router_main(x, rws[li]), bs[li])

    out, extra = {}, {}
    for m in [int(v) for v in args.ms.split(",")]:
        k1in = [make_k1_inputs(m, gen, dev) for _ in range(LAYERS)]
        xs = [torch.empty(m, HIDDEN, dtype=torch.bfloat16, device=dev) for _ in range(LAYERS)]
        srcs = [torch.randn((m, HIDDEN), generator=gen, device=dev).to(torch.bfloat16) for _ in range(LAYERS)]

        def k1(li):
            xp, res, post, comb, pre, nw = k1in[li]
            mhc_lagged_layer_input(xp, res, post, comb, pre, nw, 1e-6, K1_P, enable_pdl=True, out=xs[li])

        for li in range(LAYERS):
            k1(li)
        torch.cuda.synchronize()
        nbad = 0
        for arm in arms:
            for li in range(LAYERS):
                ri, rw = gate_main(router_main(xs[li], rws[li]), bs[li])
                ci, cw = row(arm, li, xs[li])
                nbad += int(not (torch.equal(ri, ci) and torch.equal(rw.view(torch.int32), cw.view(torch.int32))))
        torch.cuda.synchronize()

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
                    ids, wts = row(arm, li, xs[li])
                    if kind not in ("REAL",):
                        state.run(xs[li], ids, wts)
            return fn

        kinds = ["REAL", "REAL_RUN"] + ([] if args.no_context else ["P1c1", "P1c4", "P0"])
        graphs = {}
        for arm in arms:
            for kind in kinds:
                graphs[f"{arm}:{kind}"] = capture(mk(arm, kind))[0]
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
            log(f"{arm} M={m} kernels[{kname[arm]}; K1=_mhc_lagged_layer_input_kernel] routing_mismatch_layers={nbad} "
                + " ".join(f"{k.split(':')[1]}={x['us_layer']:.3f}us/layer[{x['p10_us_layer']:.3f},{x['p90_us_layer']:.3f}]"
                           for k, x in r.items() if k.startswith(arm + ":")))
        log(f"M={m} K1_NULL (K1 -> null) {r['K1_NULL']['us_layer']:.3f} us/layer")
        if args.both:
            d = {k: r[f"CAND:{k}"]["us_layer"] - r[f"BASE:{k}"]["us_layer"] for k in kinds}
            log(f"M={m} CAND-BASE us/layer: " + " ".join(f"{k}={v:+.3f}" for k, v in d.items()))
        main_arm = arms[-1]
        out[str(m)] = {"kernel_ms": r[f"{main_arm}:REAL"]["ms"], "noise_floor": r[f"{main_arm}:REAL"]["noise"]}
        extra[str(m)] = {k: {kk: round(vv, 4) for kk, vv in x.items()} for k, x in r.items()}
        extra[str(m)]["routing_mismatch_layers"] = nbad
    print(json.dumps({"batch": out, "output": {"class": "exact"}, "arms": arms,
                      "kernels": {a: kname[a] for a in arms}, "chains": extra}))


if __name__ == "__main__":
    with torch.no_grad():
        main()
