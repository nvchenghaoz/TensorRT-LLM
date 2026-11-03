"""Paired microbenchmark for idea 35: routed-MoE decode chain of one EP4 rank, M = 1/2/4/8.

Baseline tree (no ``w4a8_mxfp4_mxfp8_sm12x_decode`` module): main's chain,
mxfp8_quantize + torch.ops.trtllm.fused_moe (CUTLASS 128x128 tile).
Candidate tree: the branch's ``Sm12xW4a8DecodeState`` (FlashInfer SM12x chain) built from the
same TRT-LLM-layout weights, the way the leaf builds it at load.
Each M: one CUDA graph of the chain, L2 flushed (512 MB write) before every timed replay.
Last stdout line: {"batch": {M: {"kernel_ms", "noise_floor"}}, "output": {...}}.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from common import (E, H, I, TOPK, CutlassChain, err_stats, log, make_routing,  # noqa: E402
                    make_weights, reference, time_graph, trtllm_layout)

N_LOCAL = {1: 3, 2: 2, 4: 2, 8: 2, 16: 2, 32: 2}
ITERS = int(os.environ.get("BENCH_ITERS", "100"))


def main():
    import tensorrt_llm  # noqa: F401
    import tensorrt_llm._torch.custom_ops  # noqa: F401

    try:
        from tensorrt_llm._torch.moe.fused_moe.cutlass.w4a8_mxfp4_mxfp8_sm12x_decode import (
            Sm12xW4a8DecodeState)
        arm = "candidate_fi_chain"
    except ImportError:
        Sm12xW4a8DecodeState = None
        arm = "baseline_cutlass"
    log("arm", arm)
    torch.manual_seed(0)
    w1, s1, w2, s2 = make_weights()
    w31, s31, w2v, s2s = trtllm_layout(w1, s1, w2, s2)
    if Sm12xW4a8DecodeState is None:
        fn = CutlassChain(w31, s31, w2v, s2s)
    else:
        st = Sm12xW4a8DecodeState(
            w3_w1_weight=w31, w3_w1_weight_scale=s31, w2_weight=w2v, w2_weight_scale=s2s,
            num_local_experts=E, slot_start=0, swiglu_limit=10.0,
            enable_pdl=os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1")
        st.warmup(TOPK)
        fn = st.run
    batch, errs = {}, {}
    for m in (1, 2, 4, 8, 16, 32):
        x = torch.randn(m, H, dtype=torch.bfloat16, device="cuda")
        ids, w = make_routing(m, N_LOCAL[m], seed=m)
        stats, g, out = time_graph(fn, (x, ids, w), iters=ITERS, warmup=20)
        g.replay()
        torch.cuda.synchronize()
        ref = reference(x, ids, w, w1, s1, w2, s2)
        e = err_stats(out, ref)
        errs[str(m)] = {k: e[k] for k in ("max_abs", "max_abs_over_ref_absmax", "calc_diff", "finite")}
        batch[str(m)] = {"kernel_ms": stats["median_ms"], "noise_floor": stats["noise"],
                         "p10_ms": stats["p10_ms"], "p90_ms": stats["p90_ms"]}
        log("M", m, json.dumps(batch[str(m)]), json.dumps(errs[str(m)]))
        del g
    cls = "exact" if arm == "baseline_cutlass" else "mode_switch"
    print(json.dumps({"batch": batch, "output": {"class": cls, "arm": arm,
                                                  "error_vs_fp32_dequant_ref": errs},
                      "units": "ms per layer, routed-MoE chain of one EP4 rank, CUDA graph, cold L2"}))


if __name__ == "__main__":
    main()
