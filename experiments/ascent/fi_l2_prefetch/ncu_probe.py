"""Idea 143 ncu probe: the leaf (route -> fc1 -> fc2) at M = 4, forced D = 2 (fc1 splits, S = 4)
and D = 3 (S = 1), unsplit build (TRTLLM_SM12X_FI_SPLITK=0, main's kernel) then split build,
each launched once between cudaProfilerStart/Stop after a read-only 512 MB flush.
Launch order: OFF/D2, OFF/D3, ON/D2, ON/D3 (each: fc1 then fc2).
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "w4a8_fi_chain"))
sys.path.insert(0, os.path.join(HERE, "..", "fi_decode_moe_census"))

import torch  # noqa: E402

from common import E, H, log, make_weights, trtllm_layout  # noqa: E402


def main():
    import tensorrt_llm  # noqa: F401
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from tensorrt_llm._torch.moe.fused_moe.cutlass import w4a8_mxfp4_mxfp8_sm12x_decode as leaf
    from census import forced_routing

    torch.manual_seed(0)
    w1, s1, w2, s2 = make_weights()
    w31, s31, w2v, s2s = trtllm_layout(w1, s1, w2, s2)
    del w1, s1, w2, s2
    st = leaf.Sm12xW4a8DecodeState(
        w3_w1_weight=w31, w3_w1_weight_scale=s31, w2_weight=w2v, w2_weight_scale=s2s,
        num_local_experts=E, slot_start=0, swiglu_limit=10.0, enable_pdl=True)
    flush = torch.ones(512 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    m = 4
    x = torch.randn(m, H, dtype=torch.bfloat16, device="cuda")
    cases = [(mode, d, *forced_routing(m, d, 0)) for mode in ("0", "1") for d in (2, 3)]
    for mode, d, ids, w in cases:  # compile everything outside the region
        os.environ["TRTLLM_SM12X_FI_SPLITK"] = mode
        for _ in range(2):
            st.run(x, ids, w)
    torch.cuda.synchronize()
    order = [f"{'ON' if mode == '1' else 'OFF'}/D{d}" for mode, d, *_ in cases]
    log("ncu order (fc1, fc2 each): " + ", ".join(order))
    torch.cuda.profiler.start()
    for mode, d, ids, w in cases:
        os.environ["TRTLLM_SM12X_FI_SPLITK"] = mode
        flush.sum()
        st.run(x, ids, w)
        torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print(json.dumps({"ncu_order": order}))


if __name__ == "__main__":
    main()
