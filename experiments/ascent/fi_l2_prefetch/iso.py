"""Idea 143 diagnosis: where does the split-K build's ~0.6 us fc1 overhead at S = 1 come from?

fc1-only CUDA graphs (fixed route outputs), M = 4, forced D; three builds in one process:
OFF (TRTLLM_SM12X_FI_SPLITK=0), NOCOUNT (split build, scheduler without the tile count, so it
never splits), ON.  Kineto fc1 duration p50 (read-only 512 MB flush before every replay),
two interleaved rounds.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "w4a8_fi_chain"))
sys.path.insert(0, os.path.join(HERE, "..", "fi_decode_moe_census"))

import torch  # noqa: E402

import bench_pair  # noqa: E402
from common import E, H, I, log, make_weights, trtllm_layout  # noqa: E402

MODES = {"OFF": {"TRTLLM_SM12X_FI_SPLITK": "0", "TRTLLM_SM12X_FI_SPLITK_DEBUG_NOCOUNT": "0"},
         "NOCOUNT": {"TRTLLM_SM12X_FI_SPLITK": "1", "TRTLLM_SM12X_FI_SPLITK_DEBUG_NOCOUNT": "1"},
         "ON": {"TRTLLM_SM12X_FI_SPLITK": "1", "TRTLLM_SM12X_FI_SPLITK_DEBUG_NOCOUNT": "0"}}


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
    bench_pair.FLUSH = torch.ones(512 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    route, _, fc1, out_sf_shape, fc2, fi_act = leaf._load_ops()
    m = 4
    x = torch.randn(m, H, dtype=torch.bfloat16, device="cuda")
    rows = []
    for d in (1, 2, 4, 6, 8):
        ids, w = forced_routing(m, d, 0)
        local = st.local_ids(ids)
        pairs = local.numel()
        ws = leaf._workspace(x, local, E)
        r = route(x, local, w.float(), E, workspace=ws, enable_pdl=True)
        offsets, token_map, rw, a_q, a_sf = [t.clone() for t in r]
        q1 = torch.zeros(pairs, I, dtype=torch.float8_e4m3fn, device="cuda")
        sf1 = torch.zeros(out_sf_shape(pairs, I, E), dtype=torch.int32, device="cuda")
        res = {k: [] for k in MODES}
        for rnd in range(2):
            for name, env in MODES.items():
                os.environ.update(env)
                g = bench_pair.capture(lambda: fc1(
                    a_q, a_sf, st.w1, st.w1_sf, offsets, tune=False, enable_pdl=True, out_q=q1,
                    out_sf=sf1, activation=fi_act.Swiglu, swiglu_limit=10.0))
                (_, _), (f1, f1n), _, n, _ = bench_pair.trace(g)
                res[name].append(f1)
                del g
        row = {"D": d, **{k: round(min(v), 3) for k, v in res.items()}}
        rows.append(row)
        log("ISO " + json.dumps(row))
    print(json.dumps({"rows": rows}))


if __name__ == "__main__":
    main()
