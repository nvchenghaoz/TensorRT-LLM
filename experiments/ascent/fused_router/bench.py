"""Paired timing for idea 67 Stage 1 (run_gpu_pair: BASE = main tree, CAND = branch tree).

One GPU, EP4 rank 0: 40 distinct router weights [384,5120] bf16 + fp32 biases, x [M,5120] bf16, the real
Sm12xW4a8DecodeState.run (FlashInfer SM12x decode MoE chain, 96 local experts) after the router.
Per layer, BASE: router_gemm_mma_kernel -> gate_forward_kernel -> run;  CAND: dsv4FusedRouterKernel -> run.
40-layer CUDA graphs, read-only 512 MB flush before each replay, all graphs of an M replayed interleaved:
  P1c1 : K1 stand-in (PDL, trigger at start, 4.9 us) before every layer  [primary, ruling 2201]
  P1c4 : same with a 3.3 us stand-in
  P0   : no stand-in (router after the previous layer's fc2)
  ROWc1: K1 stand-in -> router row only (no MoE run)
kernel_ms = P1c1 40-layer chain (one decode step's worth of the MoE row). The routing of every layer is checked
bitwise against main's pair (computed eagerly) before timing.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (LAYERS, SPIN_C1_NS, SPIN_C4_NS, biases, capture, fused, gate_main, has_fused,  # noqa: E402
                    k1_standin, log, make_decode_state, router_main, router_weights)

import torch  # noqa: E402

WARMUP, ITERS = 5, 40


def main():
    import tensorrt_llm  # noqa: F401
    cand = has_fused()
    arm = "CAND" if cand else "BASE"
    kn = "dsv4FusedRouterKernel" if cand else "router_gemm_mma_kernel+gate_forward_kernel"
    dev = "cuda"
    log(arm, torch.cuda.get_device_name(), "kernels:", kn)
    state, _ = make_decode_state()
    gen = torch.Generator(device=dev).manual_seed(4242)
    rws = router_weights(gen, LAYERS)
    bs = biases(gen, LAYERS)
    counters = [torch.zeros(1, dtype=torch.int32, device=dev) for _ in range(LAYERS)]
    flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device=dev).fill_(1.0)

    def route(li, x):
        if cand:
            ids, wts, _ = fused(x, rws[li], bs[li], counters[li])
            return ids, wts
        return gate_main(router_main(x, rws[li]), bs[li])

    out = {}
    extra = {}
    for m in (1, 2, 4, 8):
        srcs = [torch.randn((m, HIDDEN_), generator=gen, device=dev).to(torch.bfloat16) for _ in range(LAYERS)]
        xs = [s.clone() for s in srcs]
        # routing parity (the timed graphs route exactly as main)
        nbad = 0
        for li in range(LAYERS):
            ri, rw = gate_main(router_main(xs[li], rws[li]), bs[li])
            ci, cw = route(li, xs[li])
            nbad += int(not (torch.equal(ri, ci) and torch.equal(rw.view(torch.int32), cw.view(torch.int32))))
        torch.cuda.synchronize()

        def mk(spin, with_run):
            def fn():
                for li in range(LAYERS):
                    if spin:
                        k1_standin(srcs[li], xs[li], spin)
                    ids, wts = route(li, xs[li])
                    if with_run:
                        state.run(xs[li], ids, wts)
            return fn

        graphs = {"P1c1": capture(mk(SPIN_C1_NS, True))[0], "P1c4": capture(mk(SPIN_C4_NS, True))[0],
                  "P0": capture(mk(0, True))[0], "ROWc1": capture(mk(SPIN_C1_NS, False))[0],
                  "ROWc4": capture(mk(SPIN_C4_NS, False))[0]}
        ts = {k: [] for k in graphs}
        for it in range(WARMUP + ITERS):
            for name, g in graphs.items():
                flush.sum()
                a = torch.cuda.Event(enable_timing=True)
                b = torch.cuda.Event(enable_timing=True)
                a.record()
                g.replay()
                b.record()
                torch.cuda.synchronize()
                if it >= WARMUP:
                    ts[name].append(a.elapsed_time(b))
        del graphs
        r = {}
        for k, v in ts.items():
            v.sort()
            med = v[len(v) // 2]
            r[k] = {"ms": med, "us_layer": med * 1000.0 / LAYERS,
                    "noise": (v[int(0.9 * (len(v) - 1))] - v[int(0.1 * (len(v) - 1))]) / med}
        log(f"{arm} M={m} kernels[{kn}] routing_mismatch_layers={nbad} " +
            " ".join(f"{k}={x['us_layer']:.3f}us/layer(nf {x['noise']:.3f})" for k, x in r.items()))
        out[str(m)] = {"kernel_ms": r["P1c1"]["ms"], "noise_floor": r["P1c1"]["noise"]}
        extra[str(m)] = {k: {"us_layer": round(x["us_layer"], 4), "noise": round(x["noise"], 4)} for k, x in r.items()}
        extra[str(m)]["routing_mismatch_layers"] = nbad
    print(json.dumps({"batch": out, "output": {"class": "exact"}, "arm": arm, "kernels": kn, "chains": extra}))


HIDDEN_ = 5120

if __name__ == "__main__":
    with torch.no_grad():
        main()
