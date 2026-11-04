"""Idea 143 stage B: device-side dynamic split-K in FI SM12x decode fc1_act_q1 (overlay).

One EP4 rank of DeepSeek-V4.1-Flash (96 local experts, H 5120, I 2304), production leaf
Sm12xW4a8DecodeState.run in a CUDA graph.  In one process, OFF (TRTLLM_SM12X_FI_SPLITK=0, the
unsplit kernel) and ON (=1) builds, per (M, forced distinct local experts D):

  * determinism: 8 graph replays per build; fc1 outputs (q1, sf1) and the leaf output must be
    bitwise identical across replays; split-K counters must read 0 after every replay;
  * S == 1 cases (T*2 > 188): ON must be bitwise equal to OFF;
  * accuracy: leaf output vs the FP32 dequantized reference (common.reference), for OFF and ON;
  * timing (clean L2: 512 MB read-only flush per replay): chain and fc1-only CUDA-event mean.

Env: SK_M="1,2,4,8,16,32"  SK_D="1,2,3,4,5,6,8,10,12"  SK_ITERS=40  SK_TIME=1
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "w4a8_fi_chain"))
sys.path.insert(0, os.path.join(HERE, "..", "fi_decode_moe_census"))

import torch  # noqa: E402

from common import E, H, I, err_stats, log, make_weights, reference, trtllm_layout  # noqa: E402

FLUSH = None
ITERS = int(os.environ.get("SK_ITERS", "40"))


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
    n = len(ts)
    return round(sum(ts) / n, 2), round((ts[(9 * n) // 10] - ts[n // 10]) / ts[n // 2], 4)


def main():
    global FLUSH
    import tensorrt_llm  # noqa: F401
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from tensorrt_llm._torch.moe.fused_moe.cutlass import w4a8_mxfp4_mxfp8_sm12x_decode as leaf
    from census import forced_routing
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x._moe_utils import split_k

    torch.manual_seed(0)
    w1, s1, w2, s2 = make_weights()
    w31, s31, w2v, s2s = trtllm_layout(w1, s1, w2, s2)
    st = leaf.Sm12xW4a8DecodeState(
        w3_w1_weight=w31, w3_w1_weight_scale=s31, w2_weight=w2v, w2_weight_scale=s2s,
        num_local_experts=E, slot_start=0, swiglu_limit=10.0, enable_pdl=True)
    FLUSH = torch.ones(512 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    route, _, fc1, out_sf_shape, fc2, fi_act = leaf._load_ops()
    do_time = os.environ.get("SK_TIME", "1") == "1"
    ms = [int(v) for v in os.environ.get("SK_M", "1,2,4,8,16,32").split(",")]
    ds = [int(v) for v in os.environ.get("SK_D", "1,2,3,4,5,6,8,10,12").split(",")]
    rows, ok = [], True
    for m in ms:
        x = torch.randn(m, H, dtype=torch.bfloat16, device="cuda")
        for d in ds:
            if d > 6 * m or d > E:
                continue
            ids, w = forced_routing(m, d, 0)
            local = st.local_ids(ids)
            pairs = local.numel()
            ws = leaf._workspace(x, local, E)
            r = route(x, local, w.float(), E, workspace=ws, enable_pdl=True)
            offsets, token_map, rw, a_q, a_sf = [t.clone() for t in r]
            T = int(((offsets[1:] - offsets[:-1] + 7) // 8).sum().item()) * (I // 128)
            q1 = torch.zeros(pairs, I, dtype=torch.float8_e4m3fn, device="cuda")
            sf1 = torch.zeros(out_sf_shape(pairs, I, E), dtype=torch.int32, device="cuda")
            ref = reference(x, ids, w, w1, s1, w2, s2)
            res = {}
            for mode in ("0", "1"):
                os.environ["TRTLLM_SM12X_FI_SPLITK"] = mode
                f1 = lambda: fc1(a_q, a_sf, st.w1, st.w1_sf, offsets, tune=False, enable_pdl=True,
                                 out_q=q1, out_sf=sf1, activation=fi_act.Swiglu, swiglu_limit=10.0)
                g1 = capture(f1)
                out_holder = {}
                gc = capture(lambda: out_holder.__setitem__("o", st.run(x, ids, w)))
                cnt = split_k.workspace(torch.device("cuda"), 188)[1]
                q_list, s_list, o_list, cnt_ok = [], [], [], True
                for rep in range(8):
                    q1.zero_(); sf1.zero_()
                    g1.replay()
                    torch.cuda.synchronize()
                    cnt_ok &= int(cnt.abs().sum().item()) == 0
                    q_list.append(q1.clone()); s_list.append(sf1.clone())
                    gc.replay()
                    torch.cuda.synchronize()
                    cnt_ok &= int(cnt.abs().sum().item()) == 0
                    o_list.append(out_holder["o"].clone())
                det = (all(torch.equal(q_list[0], q) for q in q_list)
                       and all(torch.equal(s_list[0], s) for s in s_list))
                det_out = all(torch.equal(o_list[0], o) for o in o_list)
                e = err_stats(o_list[0], ref)
                rec = {"det_fc1": det, "det_out": det_out, "counters_zero": cnt_ok,
                       "q1": q_list[0], "sf1": s_list[0], "out": o_list[0],
                       "err_max_abs": round(e["max_abs"], 5),
                       "err_rel_absmax": round(e["max_abs_over_ref_absmax"], 5),
                       "calc_diff": e["calc_diff"], "finite": e["finite"]}
                if do_time:
                    rec["chain_us"], rec["chain_noise"] = time_g(gc)
                    rec["fc1_us"], rec["fc1_noise"] = time_g(g1)
                res[mode] = rec
                del g1, gc
            off, on = res["0"], res["1"]
            s_exp = 1  # the kernel's rule: largest c in [S_min, S_max] with c*T <= 188
            for c in range(split_k.min_splits(), split_k.max_splits() + 1):
                if c * T <= 188 and c <= 10:
                    s_exp = c
            fc1_bitwise = torch.equal(off["q1"], on["q1"]) and torch.equal(off["sf1"], on["sf1"])
            q_diff = (off["q1"].float() - on["q1"].float()).abs().max().item()
            row = {"M": m, "D": d, "T": T, "S": s_exp,
                   "fc1_bitwise_on_vs_off": fc1_bitwise, "q1_maxdiff_on_vs_off": q_diff,
                   "out_maxdiff_on_vs_off": (off["out"].float() - on["out"].float()).abs().max().item()}
            for k in ("det_fc1", "det_out", "counters_zero", "err_max_abs", "err_rel_absmax",
                      "calc_diff", "finite", "chain_us", "fc1_us", "chain_noise", "fc1_noise"):
                if k in off:
                    row["off_" + k] = off[k]
                    row["on_" + k] = on[k]
            good = (on["det_fc1"] and on["counters_zero"] and on["finite"] and off["det_fc1"]
                    and (s_exp > 1 or fc1_bitwise)
                    and on["calc_diff"] < 2 * off["calc_diff"] + 1e-4)
            row["pass"] = bool(good)
            ok &= bool(good)
            rows.append(row)
            log("SPLITK " + json.dumps(row))
    out_dir = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
    with open(os.path.join(out_dir, "splitk_check.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print(json.dumps({"passed": bool(ok), "rows": rows}))


if __name__ == "__main__":
    main()
