"""Idea 143 B acceptance: the decode leaf with split-K fc1 inside CUDA graphs.

1. Sm12xW4a8DecodeState.warmup(6) JITs every kernel for 1..32 tokens; then cute.compile is
   patched to raise while a stream is capturing, and the leaf is captured at EVERY batch size
   1..32 (random top-6 routing over 384 experts, rank 0 = experts 0..95) and replayed 4x.
   Each replay must equal the eager output (rtol/atol 1.6e-2: fc2's scatter-add order is
   main's own nondeterminism) and the split-K counters must read 0 after every replay.
2. Poisoned scratch: at forced-routing cases where fc1 splits (M = 1/4, D = 1/2, one local
   expert per token, so the leaf output is deterministic), fill the split-K scratch with NaN
   before a replay; the output must stay bitwise equal to the clean replay (every slot is
   written before it is read).
Prints {"passed": ...}.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "w4a8_fi_chain"))
sys.path.insert(0, os.path.join(HERE, "..", "fi_decode_moe_census"))

import torch  # noqa: E402

from common import E, E_GLOBAL, H, TOPK, log, make_weights, trtllm_layout  # noqa: E402


def main():
    import cutlass.cute as cute
    import tensorrt_llm  # noqa: F401
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from tensorrt_llm._torch.moe.fused_moe.cutlass import w4a8_mxfp4_mxfp8_sm12x_decode as leaf
    from census import forced_routing
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x._moe_utils import split_k

    torch.manual_seed(0)
    w1, s1, w2, s2 = make_weights()
    w31, s31, w2v, s2s = trtllm_layout(w1, s1, w2, s2)
    del w1, s1, w2, s2
    st = leaf.Sm12xW4a8DecodeState(
        w3_w1_weight=w31, w3_w1_weight_scale=s31, w2_weight=w2v, w2_weight_scale=s2s,
        num_local_experts=E, slot_start=0, swiglu_limit=10.0, enable_pdl=True)
    st.warmup(TOPK)
    log("warmup done; split-K enabled:", split_k.enabled(), "max", split_k.max_splits(),
        "min", split_k.min_splits())

    orig_compile = cute.compile
    jit_in_capture = []

    def guarded(*a, **k):
        if torch.cuda.is_current_stream_capturing():
            jit_in_capture.append(1)
            raise RuntimeError("cute.compile inside CUDA graph capture")
        return orig_compile(*a, **k)

    cute.compile = guarded
    import flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_mxfp8_mxfp4_fc1_act_q1 as f1mod
    import flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_mxfp8_mxfp4_fc2_finalize as f2mod
    f1mod.cute.compile = guarded
    f2mod.cute.compile = guarded

    scratch, cnt = split_k.workspace(torch.device("cuda"), torch.cuda.get_device_properties(0).multi_processor_count)
    ok = True
    rows = []
    g_cpu = torch.Generator().manual_seed(5)
    cases = []
    for m in range(1, 33):
        x = torch.randn(m, H, dtype=torch.bfloat16, device="cuda")
        ids = torch.stack([torch.randperm(E_GLOBAL, generator=g_cpu)[:TOPK] for _ in range(m)])
        cases.append((m, x, ids.to(torch.int32).cuda(),
                      (torch.rand(m, TOPK, generator=g_cpu) + 0.1).float().cuda()))
    # mode "0" = the unsplit kernel (main's math): its replay-vs-eager deviation is fc2's
    # scatter-add order (main's own nondeterminism), the yardstick for mode "1".
    for mode, (m, x, ids, w) in [(md, c) for md in ("0", "1") for c in cases]:
        os.environ["TRTLLM_SM12X_FI_SPLITK"] = mode
        eager = st.run(x, ids, w).clone()
        torch.cuda.synchronize()
        holder = {}
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                st.run(x, ids, w)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                holder["o"] = st.run(x, ids, w)
            captured = True
        except Exception as e:  # noqa: BLE001
            captured = False
            log(f"M={m} capture FAILED: {e!r}")
        good = captured
        max_err = 0.0
        cnt_zero = True
        amax = max(eager.float().abs().max().item(), 1e-6)
        if captured:
            for _ in range(4):
                g.replay()
                torch.cuda.synchronize()
                cnt_zero &= int(cnt.abs().sum().item()) == 0
                o = holder["o"]
                max_err = max(max_err, (o.float() - eager.float()).abs().max().item())
                good &= bool(torch.isfinite(o).all().item())
            del g
        rel = max_err / amax
        good &= rel <= 1.6e-2
        good &= cnt_zero
        ok &= good
        rows.append({"mode": mode, "M": m, "captured": captured, "counters_zero": cnt_zero,
                     "max_abs_vs_eager": max_err, "rel_to_absmax": round(rel, 5), "pass": good})
        log(f"GRAPH split={mode} M={m}: captured {captured} counters_zero {cnt_zero} "
            f"max|replay-eager| {max_err:.4g} (/absmax {rel:.2e}) pass {good}")
    os.environ["TRTLLM_SM12X_FI_SPLITK"] = "1"

    poison = []
    for m, d in ((1, 1), (4, 1), (4, 2), (2, 2)):
        x = torch.randn(m, H, dtype=torch.bfloat16, device="cuda")
        ids, w = forced_routing(m, d, 0)
        holder = {}
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            st.run(x, ids, w)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            holder["o"] = st.run(x, ids, w)
        g.replay()
        torch.cuda.synchronize()
        clean = holder["o"].clone()
        same = True
        for _ in range(3):
            scratch.fill_(float("nan"))
            g.replay()
            torch.cuda.synchronize()
            same &= torch.equal(holder["o"], clean)
            same &= int(cnt.abs().sum().item()) == 0
        poison.append({"M": m, "D": d, "bitwise_after_nan_scratch": bool(same)})
        ok &= bool(same)
        log(f"POISON M={m} D={d}: bitwise after NaN scratch {same}")
        del g
    scratch.zero_()
    ok &= not jit_in_capture
    print(json.dumps({"passed": bool(ok), "jit_inside_capture": len(jit_in_capture),
                      "graph": rows, "poison": poison}))


if __name__ == "__main__":
    main()
