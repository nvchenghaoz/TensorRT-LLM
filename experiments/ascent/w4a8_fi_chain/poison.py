"""Which uninitialized buffer makes the FlashInfer W4A8 decode chain return NaN (idea 35 debug).

For several buffer-fill configurations (FC1 output q1, route activation buffer q_out) and inputs,
compare the chain with main's CUTLASS chain. 0x7F is an e4m3fn NaN byte (poison).
"""

import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from common import (E, H, TOPK, CutlassChain, err_stats, log, make_routing, make_weights,  # noqa: E402
                    trtllm_layout)

CONFIGS = [
    ("q1_empty", None, None),
    ("q1_poison", 0x7F, None),
    ("q1_zero", 0, None),
    ("q1_zero_q0_poison", 0, 0x7F),
    ("q1_zero_q0_zero", 0, 0),
]


def main():
    import tensorrt_llm  # noqa: F401
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from tensorrt_llm._torch.moe.fused_moe.cutlass.w4a8_mxfp4_mxfp8_sm12x_decode import \
        Sm12xW4a8DecodeState

    w1, s1, w2, s2 = make_weights(seed=3)
    w31, s31, w2v, s2s = trtllm_layout(w1, s1, w2, s2)
    cut = CutlassChain(w31, s31, w2v, s2s)
    st = Sm12xW4a8DecodeState(w3_w1_weight=w31, w3_w1_weight_scale=s31, w2_weight=w2v,
                              w2_weight_scale=s2s, num_local_experts=E, slot_start=0,
                              swiglu_limit=10.0, enable_pdl=True)
    st.warmup(TOPK)
    inputs = []
    for m, nl, seed in ((1, 3, 1), (2, 2, 2), (4, 2, 77), (8, 2, 8), (4, 6, 9), (8, 1, 10)):
        torch.manual_seed(seed)
        x = torch.randn(m, H, dtype=torch.bfloat16, device="cuda")
        ids, w = make_routing(m, nl, seed=seed)
        inputs.append((f"M{m}_nl{nl}", x, ids, w))
    refs = {name: cut(x, ids, w).clone() for name, x, ids, w in inputs}
    res = {}
    for cname, q1f, q0f in CONFIGS:
        st.q1_fill, st.q0_fill = q1f, q0f
        rows = {}
        for rep in range(3):
            # dirty the caching allocator with NaN-pattern bytes between reps
            junk = torch.full((64 * 1024 * 1024,), 0x7F, dtype=torch.uint8, device="cuda")
            del junk
            for name, x, ids, w in inputs:
                try:
                    o = st.run(x, ids, w)
                    e = err_stats(o, refs[name].float())
                    key = name
                    prev = rows.get(key, {"finite_all": True, "worst_calc_diff": 0.0})
                    rows[key] = {"finite_all": prev["finite_all"] and e["finite"],
                                 "worst_calc_diff": max(prev["worst_calc_diff"],
                                                        e["calc_diff"] if e["finite"] else 1.0)}
                except Exception as ex:  # noqa: BLE001
                    traceback.print_exc()
                    rows[name] = {"error": repr(ex)}
        res[cname] = rows
        log(cname, json.dumps(rows))
    print(json.dumps({"passed": all(r.get("finite_all") for r in res["q1_zero_q0_poison"].values()),
                      "configs": res}))


if __name__ == "__main__":
    main()
