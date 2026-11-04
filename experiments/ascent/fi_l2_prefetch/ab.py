"""Idea 143 stage 0 + A exploration: FI SM12x decode fc1_act_q1 / fc2_finalize L2 bulk prefetch.

One EP4 rank of DeepSeek-V4.1-Flash (96 local experts, H 5120, I 2304), production leaf
Sm12xW4a8DecodeState.run captured in a CUDA graph, CLEAN L2 (512 MB read-only flush before every
replay, pitfall 1356).  In ONE process, alternating OFF/ON builds of the overlay kernels
(TRTLLM_SM12X_FI_L2_PREFETCH is part of the compile key), per forced distinct-expert count D:
chain, fc1-only and fc2-only CUDA-event p50, and bitwise OFF-vs-ON output equality.

Usage: ab.py [M ...]  (default 1 4); env AB_D="1,2,3,..." AB_ITERS=40
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "w4a8_fi_chain"))
sys.path.insert(0, os.path.join(HERE, "..", "fi_decode_moe_census"))

import torch  # noqa: E402

from common import E, H, I, log, make_routing, make_weights, trtllm_layout  # noqa: E402

FC1_BYTES = 2 * I * H // 2 + 2 * I * H // 32
FC2_BYTES = H * I // 2 + H * I // 32
PEAK = 1507.17e9
ITERS = int(os.environ.get("AB_ITERS", "40"))
FLUSH = None


def flush():
    FLUSH.sum()


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


def time_g(g, iters=ITERS, warmup=8):
    ts = []
    for i in range(warmup + iters):
        flush()
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        if i >= warmup:
            ts.append(a.elapsed_time(b) * 1000.0)
    ts.sort()
    n = len(ts)
    med = ts[n // 2]
    return round(med, 2), round((ts[(9 * n) // 10] - ts[n // 10]) / med, 4)


def stage0():
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_mxfp8_mxfp4_fc1_act_q1 as f1
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_mxfp8_mxfp4_fc2_finalize as f2
    from flashinfer.tllm_enums import ActivationType
    out = {}
    t1 = (8, 128, f1.BK)
    op1 = f1._op(I, H, t1, torch.float8_e4m3fn, ActivationType.Swiglu, f1.epi_tactics(t1)[0])
    c1 = op1.cfg
    out["fc1"] = {"tile": list(c1.TILE), "ab_stage": op1.ab_stage, "smem_bytes": c1.smem_bytes,
                  "tiles_per_expert": I // c1.TILE[0],
                  "weight_bytes_per_tile": 2 * c1.TILE[0] * H // 2,
                  "weight_bytes_per_stage": 2 * c1.TILE[0] * c1.TILE[2] // 2}
    t2 = (32, 128, f2.BK)
    op2 = f2.CuteDslSm120GroupedMxfp8Mxfp4Fc2FinalizeOp(H, I, t2)
    c2 = op2.cfg
    out["fc2"] = {"tile": list(c2.TILE), "ab_stage": op2.ab_stage, "smem_bytes": c2.smem_bytes,
                  "tiles_per_expert": H // c2.TILE[1],
                  "weight_bytes_per_tile": c2.TILE[1] * I // 2,
                  "weight_bytes_per_stage": c2.TILE[1] * c2.TILE[2] // 2}
    log("STAGE0", json.dumps(out))
    return out


def main():
    global FLUSH
    ms = [int(a) for a in sys.argv[1:]] or [1, 4]
    dset = [int(d) for d in os.environ.get("AB_D", "1,2,3,4,5,6,8,10,11,12").split(",")]
    import tensorrt_llm  # noqa: F401
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from tensorrt_llm._torch.moe.fused_moe.cutlass import w4a8_mxfp4_mxfp8_sm12x_decode as leaf
    from census import forced_routing
    s0 = stage0()

    torch.manual_seed(0)
    w1, s1, w2, s2 = make_weights()
    w31, s31, w2v, s2s = trtllm_layout(w1, s1, w2, s2)
    del w1, s1, w2, s2
    st = leaf.Sm12xW4a8DecodeState(
        w3_w1_weight=w31, w3_w1_weight_scale=s31, w2_weight=w2v, w2_weight_scale=s2s,
        num_local_experts=E, slot_start=0, swiglu_limit=10.0, enable_pdl=True)
    FLUSH = torch.ones(512 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    route, _, fc1, out_sf_shape, fc2, fi_act = leaf._load_ops()

    rows = []
    all_bitwise = True
    for m in ms:
        x = torch.randn(m, H, dtype=torch.bfloat16, device="cuda")
        cases = [(f"D{d}", *forced_routing(m, d, 0)) for d in dset if d <= 6 * m]
        if m in (1, 2, 4, 8):
            ids_b, w_b = make_routing(m, {1: 3, 2: 2, 4: 2, 8: 2}[m], seed=m)
            cases.append(("bench", ids_b, w_b))
        for label, ids, w in cases:
            d = int(torch.unique(ids[(ids >= 0) & (ids < E)]).numel())
            # fixed fc1/fc2 inputs for the single-kernel probes
            local = st.local_ids(ids)
            pairs = local.numel()
            ws = leaf._workspace(x, local, E)
            offsets, token_map, rw, a_q, a_sf = route(x, local, w.float(), E, workspace=ws,
                                                      enable_pdl=True)
            offsets, token_map, rw, a_q, a_sf = [t.clone() for t in (offsets, token_map, rw, a_q, a_sf)]
            q1 = torch.zeros(pairs, I, dtype=torch.float8_e4m3fn, device="cuda")
            sf1 = torch.zeros(out_sf_shape(pairs, I, E), dtype=torch.int32, device="cuda")
            o2 = torch.zeros(m, H, dtype=torch.bfloat16, device="cuda")
            res = {}
            outs = {}
            for rep in range(2):
                for mode in ("0", "1"):
                    os.environ["TRTLLM_SM12X_FI_L2_PREFETCH"] = mode
                    gc = capture(lambda: st.run(x, ids, w))
                    g1 = capture(lambda: fc1(a_q, a_sf, st.w1, st.w1_sf, offsets, tune=False,
                                             enable_pdl=True, out_q=q1, out_sf=sf1,
                                             activation=fi_act.Swiglu, swiglu_limit=10.0))
                    g2 = capture(lambda: fc2(q1, sf1, st.w2, st.w2_sf, offsets, token_map, rw, m,
                                             tune=False, enable_pdl=True, out=o2))
                    tc, nc = time_g(gc)
                    t1, n1 = time_g(g1)
                    t2, n2 = time_g(g2)
                    r = res.setdefault(mode, {"chain": [], "fc1": [], "fc2": [], "noise": []})
                    r["chain"].append(tc); r["fc1"].append(t1); r["fc2"].append(t2)
                    r["noise"].append(max(nc, n1, n2))
                    # output of the full leaf (eager, after the graph) for bitwise check
                    o = st.run(x, ids, w).clone()
                    torch.cuda.synchronize()
                    outs.setdefault(mode, []).append(o)
                    del gc, g1, g2
            ref = outs["0"][0]
            bit = all(torch.equal(ref, o) for k in outs for o in outs[k])
            all_bitwise &= bit
            off, on = res["0"], res["1"]
            row = {"M": m, "case": label, "D": d,
                   "off": {k: min(v) for k, v in off.items()},
                   "on": {k: min(v) for k, v in on.items()},
                   "bitwise_on_vs_off": bit,
                   "floor_fc1_us": round(d * FC1_BYTES / PEAK * 1e6, 2),
                   "floor_fc2_us": round(d * FC2_BYTES / PEAK * 1e6, 2)}
            rows.append(row)
            log(f"M={m} {label} D={d} | OFF chain {row['off']['chain']} fc1 {row['off']['fc1']} "
                f"fc2 {row['off']['fc2']} | ON chain {row['on']['chain']} fc1 {row['on']['fc1']} "
                f"fc2 {row['on']['fc2']} | floor fc1 {row['floor_fc1_us']} fc2 {row['floor_fc2_us']} "
                f"| bitwise {bit} | noise off {max(off['noise'])} on {max(on['noise'])}")
    out_dir = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
    with open(os.path.join(out_dir, "ab.json"), "w") as f:
        json.dump({"stage0": s0, "rows": rows}, f, indent=1)
    print(json.dumps({"passed": all_bitwise, "stage0": s0,
                      "rows": [{"M": r["M"], "case": r["case"], "D": r["D"],
                                "off_chain": r["off"]["chain"], "on_chain": r["on"]["chain"],
                                "off_fc1": r["off"]["fc1"], "on_fc1": r["on"]["fc1"],
                                "off_fc2": r["off"]["fc2"], "on_fc2": r["on"]["fc2"]}
                               for r in rows]}))


if __name__ == "__main__":
    main()
