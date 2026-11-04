"""Idea 143 diagnosis: is the FI decode fc1/fc2 per-CTA rate DRAM-latency bound (L2 prefetch can
help) or bound inside the SM (TMA / pipeline; only more CTAs per tile helps)?

fc1-only and fc2-only CUDA graphs (fixed route outputs) at M=4, forced D, timed per replay with
(a) a 512 MB read-only flush before each replay (clean, cold L2) and (b) no flush (weights of
the previous replay still in L2 when they fit), for prefetch modes 0 (off), 1 (bulk), 2 (lsu).
Mean and p50 per replay (CUDA events here tick in ~2 us steps, so the mean is reported too).
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "w4a8_fi_chain"))
sys.path.insert(0, os.path.join(HERE, "..", "fi_decode_moe_census"))

import torch  # noqa: E402

from common import E, H, I, log, make_weights, trtllm_layout  # noqa: E402

FLUSH = None


def capture(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


def time_g(g, do_flush, iters=60, warmup=8):
    ts = []
    for i in range(warmup + iters):
        if do_flush:
            FLUSH.sum()
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        if i >= warmup:
            ts.append(a.elapsed_time(b) * 1000.0)
    ts.sort()
    return round(sum(ts) / len(ts), 2), round(ts[len(ts) // 2], 2)


def main():
    global FLUSH
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
    FLUSH = torch.ones(512 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    route, _, fc1, out_sf_shape, fc2, fi_act = leaf._load_ops()
    m = 4
    x = torch.randn(m, H, dtype=torch.bfloat16, device="cuda")
    modes = os.environ.get("DIAG_MODES", "0,1,2").split(",")
    rows = []
    for d in [int(v) for v in os.environ.get("DIAG_D", "1,2,3,4,6,10").split(",")]:
        ids, w = forced_routing(m, d, 0)
        local = st.local_ids(ids)
        pairs = local.numel()
        ws = leaf._workspace(x, local, E)
        r = route(x, local, w.float(), E, workspace=ws, enable_pdl=True)
        offsets, token_map, rw, a_q, a_sf = [t.clone() for t in r]
        q1 = torch.zeros(pairs, I, dtype=torch.float8_e4m3fn, device="cuda")
        sf1 = torch.zeros(out_sf_shape(pairs, I, E), dtype=torch.int32, device="cuda")
        o2 = torch.zeros(m, H, dtype=torch.bfloat16, device="cuda")
        for mode in modes:
            os.environ["TRTLLM_SM12X_FI_L2_PREFETCH"] = mode
            g1 = capture(lambda: fc1(a_q, a_sf, st.w1, st.w1_sf, offsets, tune=False,
                                     enable_pdl=True, out_q=q1, out_sf=sf1,
                                     activation=fi_act.Swiglu, swiglu_limit=10.0))
            q1_ref = q1.clone()
            sf1_ref = sf1.clone()
            g2 = capture(lambda: fc2(q1, sf1, st.w2, st.w2_sf, offsets, token_map, rw, m,
                                     tune=False, enable_pdl=True, out=o2))
            row = {"D": d, "mode": mode}
            for nm, g in (("fc1", g1), ("fc2", g2)):
                row[nm + "_cold_mean"], row[nm + "_cold_p50"] = time_g(g, True)
                row[nm + "_warm_mean"], row[nm + "_warm_p50"] = time_g(g, False)
            row["q1_sha"] = int(q1_ref.view(torch.uint8).double().sum().item())
            rows.append(row)
            log(json.dumps(row))
            del g1, g2
    print(json.dumps({"rows": rows}))


if __name__ == "__main__":
    main()
