"""Idea 201 correctness over bench.py's EXACT timed sequence (untimed; evaluator send-back 7219).

What bench.py times (BENCH_MODE=leaf): BENCH_LAYERS=8 Sm12xW4a8DecodeState layers with
distinct weight seeds (make_state(seed=li)), st.warmup(6) each, one CUDA graph of the 8 layers'
st.run back to back (route_assign -> q0 -> fused, PDL on), built by common12.capture (3 eager
calls on a side stream, then capture), then 3 warmup replays and 30 replays, each after the
512 MB READ-ONLY flush (FLUSH.sum()).  All 8 layers share ONE global sync workspace (claim,
done and read counters) and one split-K workspace.  Cases: M1 D0..6, M4 D0..12 (routing(m, d,
seed=li) per layer), M2 D 0/1/6/12, and the uniform busiest-rank M1/M2/M4 mixes
(uniform_routing(m, 1000*s + li), s < BENCH_SEEDS).

Here the same 8-layer graph is built twice per case: BASE = every layer's st._force_chain=True
(main's fc1_act_q1 + fc2_finalize, the protected FI ops), CAND = the fused kernel.  Each graph
runs the 3 warmup + 30 replays (same flush before each).  After EVERY replay (sync, then read
the graph's own buffers, recovered from st.run's returned output: its storage is the one buffer
st.run allocates, q1/sf1 live at the offsets st.run uses):
  * every layer's q1/sf1 bitwise == BASE replay 0's;
  * every layer's output inside main's band: |cand - base_0| <= max(main's replay-to-replay
    band of that layer over its 33 replays, (k-1) bf16 ulps of sum_j |w_j fc2_j| for a row with
    k local experts: the scatter-add order, the only freedom the kernel has); max abs / rel
    per layer per replay reported;
  * output finite;
  * sync workspace (claim + per-expert done/read) and split-K counters all 0.
Also prints the fc2 K-tile count vs fc2 ab_stage (the no-prefetch counter-reset argument needs
k_tiles2 > ab_stage2).
Last line: {"passed": ..., ...}.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from common12 import E, E_GLOBAL, H, SHAPE, SLOT, TOPK, capture, log, routing, uniform_routing  # noqa: E402,I001

import common as _common  # noqa: E402  (on sys.path via common12)

LAYERS = int(os.environ.get("BENCH_LAYERS", "8"))
WARM = 3
REPLAYS = int(os.environ.get("BENCH_REPLAYS", "30"))
SEEDS = int(os.environ.get("BENCH_SEEDS", "6"))
FLUSH = None


def make_layer(seed):
    """common12.make_state, keeping the raw scales (the FP4 codes are shared with the state)."""
    from tensorrt_llm._torch.moe.fused_moe.cutlass import w4a8_mxfp4_mxfp8_sm12x_decode as leaf

    torch.manual_seed(seed)
    w1, s1, w2, s2 = _common.make_weights(seed=seed)
    w31, s31, w2v, s2s = _common.trtllm_layout(w1, s1, w2, s2)
    st = leaf.Sm12xW4a8DecodeState(
        w3_w1_weight=w31, w3_w1_weight_scale=s31, w2_weight=w2v, w2_weight_scale=s2s,
        num_local_experts=E, slot_start=SLOT, swiglu_limit=10.0, enable_pdl=True)
    del s31, s2s
    return leaf, st, (w1, s1, w2, s2)


def views(leaf, st, out, m):
    """q1 / sf1 views of the buffer st.run allocated (out is its tail)."""
    stor = out.untyped_storage()
    buf = torch.empty(0, dtype=torch.uint8, device=out.device)
    buf.set_(stor, 0, (stor.nbytes(),))
    out_sf_shape = leaf._load_ops()[3]
    pairs = m * TOPK
    sf1_shape = out_sf_shape(pairs, st.inter, st.num_experts)
    n_q1 = pairs * st.inter
    n_sf1 = sf1_shape[0] * sf1_shape[1] * 4
    off_sf1 = leaf._align(n_q1, 256)
    off_out = leaf._align(off_sf1 + n_sf1, 256)
    assert out.data_ptr() == buf.data_ptr() + off_out, (out.storage_offset(), off_out)
    assert buf.numel() >= off_out + m * st.hidden * 2
    q1 = buf[:n_q1]
    sf1 = buf[off_sf1:off_sf1 + n_sf1]
    return q1, sf1


def ulp_bf16(x):
    a = x.abs().float().clamp_min(2.0 ** -126)
    return torch.pow(2.0, torch.floor(torch.log2(a)) - 7)


def reorder_slack(x, ids, w, raw):
    """(k-1) bf16 ulps of sum_j |w_j fc2_j(x)| per element (FP32 dequantized)."""
    w1, s1, w2, s2 = raw
    m = x.shape[0]
    lids = ((ids.long() - SLOT) % E_GLOBAL)
    acc = torch.zeros(m, H, dtype=torch.float32, device=x.device)
    loc = torch.zeros(m, dtype=torch.float32, device=x.device)
    cache = {}
    xf = x.float()
    for t in range(m):
        for j in range(TOPK):
            le = int(lids[t, j])
            if not (0 <= le < E):
                continue
            loc[t] += 1
            if le not in cache:
                cache[le] = (_common.dequant_expert(w1[le], s1[le]), _common.dequant_expert(w2[le], s2[le]))
            W1, W2 = cache[le]
            h = W1 @ xf[t]
            lin, gate = h[:_common.I], h[_common.I:]
            gate = gate.clamp(max=_common.LIMIT)
            lin = lin.clamp(-_common.LIMIT, _common.LIMIT)
            acc[t] += (float(w[t, j]) * (W2 @ (gate * torch.sigmoid(gate) * lin))).abs()
    del cache
    return (loc[:, None] - 1).clamp_min(0) * ulp_bf16(acc)


def build(layers, x, routes, chain_mode):
    for _, st, _ in layers:
        st._force_chain = chain_mode
    outs = []

    def chain():
        outs.clear()
        o = None
        for (_, st, _), (ids, w) in zip(layers, routes):
            o = st.run(x, ids, w)
            outs.append(o)
        return o

    g = capture(chain)
    for _, st, _ in layers:
        st._force_chain = False
    leaf = layers[0][0]
    m = x.shape[0]
    bufs = [(*views(leaf, st, o, m), o) for (_, st, _), o in zip(layers, outs)]
    return g, bufs


def replays(g, bufs, on_replay):
    for i in range(WARM + REPLAYS):
        FLUSH.sum()
        g.replay()
        torch.cuda.synchronize()
        on_replay(i, [(q.clone(), s.clone(), o.clone()) for q, s, o in bufs])


def run_case(name, layers, x, routes, sync, cnt):
    m = x.shape[0]
    gb, bb = build(layers, x, routes, True)
    base = []
    replays(gb, bb, lambda i, snap: base.append(snap))
    del gb, bb
    ref0 = base[0]
    q_stable = all(torch.equal(r[li][0], ref0[li][0]) and torch.equal(r[li][1], ref0[li][1])
                   for r in base for li in range(LAYERS))
    bands = [max((r[li][2].float() - ref0[li][2].float()).abs().max().item() for r in base)
             for li in range(LAYERS)]
    slacks = []
    for li, ((_, _, raw), (ids, w)) in enumerate(zip(layers, routes)):
        sl = reorder_slack(x, ids, w, raw)
        slacks.append(torch.maximum(sl, torch.full_like(sl, bands[li])))
    del base
    gf, bf = build(layers, x, routes, False)
    stat = {"ok": q_stable, "q1_sf1_mismatch": 0, "out_band_fail": 0, "nonfinite": 0,
            "counters_nonzero": 0, "max_abs": [0.0] * LAYERS, "max_rel": [0.0] * LAYERS,
            "per_replay_max_abs": []}

    def check(i, snap):
        zero = int((sync != 0).sum().item()) == 0 and int((cnt != 0).sum().item()) == 0
        rep_max = 0.0
        for li in range(LAYERS):
            q, s, o = snap[li]
            if not (torch.equal(q, ref0[li][0]) and torch.equal(s, ref0[li][1])):
                stat["q1_sf1_mismatch"] += 1
            of, bfp = o.float(), ref0[li][2].float()
            d = (of - bfp).abs()
            if not bool((d <= slacks[li]).all().item()):
                stat["out_band_fail"] += 1
            if not bool(torch.isfinite(of).all().item()):
                stat["nonfinite"] += 1
            ma = d.max().item()
            mr = (d / bfp.abs().clamp_min(0.05 * max(bfp.abs().max().item(), 1e-6))).max().item()
            stat["max_abs"][li] = max(stat["max_abs"][li], ma)
            stat["max_rel"][li] = max(stat["max_rel"][li], mr)
            rep_max = max(rep_max, ma)
        if not zero:
            stat["counters_nonzero"] += 1
        stat["per_replay_max_abs"].append(round(rep_max, 6))

    replays(gf, bf, check)
    del gf, bf
    stat["ok"] = (stat["ok"] and stat["q1_sf1_mismatch"] == 0 and stat["out_band_fail"] == 0
                  and stat["nonfinite"] == 0 and stat["counters_nonzero"] == 0)
    stat["main_band_per_layer"] = bands
    stat["base_q1_sf1_stable"] = q_stable
    stat["max_abs"] = [round(v, 6) for v in stat["max_abs"]]
    stat["max_rel"] = [round(v, 6) for v in stat["max_rel"]]
    log(f"{SHAPE} {name}: ok={stat['ok']} replays={WARM}+{REPLAYS} x {LAYERS} layers "
        f"q1/sf1 mism={stat['q1_sf1_mismatch']} band_fail={stat['out_band_fail']} "
        f"nonfinite={stat['nonfinite']} counters_nonzero={stat['counters_nonzero']} "
        f"main band/layer max={max(bands):.4g} cand-vs-base max_abs/layer={stat['max_abs']} "
        f"max_rel/layer={stat['max_rel']}")
    return {"case": name, **stat}


def main():
    global FLUSH
    import tensorrt_llm  # noqa: F401
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import kernel_moe_mxfp8_mxfp4_fc12_fused as fk
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_mxfp8_mxfp4_fc2_finalize as fc2h
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x._moe_utils import split_k
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x._moe_utils.moe_epilogue import EpiMethod

    layers = []
    for li in range(LAYERS):
        leaf, st, raw = make_layer(li)
        st.warmup(6)
        layers.append((leaf, st, raw))
        log(f"layer {li} ready (seed {li}, warmup(6))")
    torch.cuda.synchronize()
    FLUSH = torch.ones(512 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    sync = fk.sync_workspace("cuda")
    _, cnt = split_k.workspace(torch.device("cuda"),
                               torch.cuda.get_device_properties(0).multi_processor_count)
    st0 = layers[0][1]
    stage_info = {}
    try:
        for m in (1, 2, 4, 8, 32):
            t2 = fc2h.select_tile(total_rows=m, n=st0.hidden, num_experts=st0.num_experts,
                                  num_sms=torch.cuda.get_device_properties(0).multi_processor_count)
            op2 = fc2h.CuteDslSm120GroupedMxfp8Mxfp4Fc2FinalizeOp(
                st0.hidden, st0.inter, tuple(t2), EpiMethod.WG_SCATTER, True)
            bk2 = op2.cfg.TILE[2]
            stage_info[str(m)] = {"tile2": list(op2.cfg.TILE), "ab_stage2": op2.cfg.ab_stage,
                                  "k_tiles2": -(-st0.inter // bk2)}
        stage_ok = all(v["k_tiles2"] > v["ab_stage2"] for v in stage_info.values())
    except Exception as e:  # informational only
        stage_info, stage_ok = {"error": repr(e)}, None
    log(f"fc2 K tiles vs ab_stage (K2={st0.inter}): {stage_info} -> k_tiles2 > ab_stage2: {stage_ok}")

    rows = []
    cases = [(1, d) for d in range(0, 7)] + [(4, d) for d in range(0, 13)] + \
            [(2, d) for d in (0, 1, 6, 12)]
    xs = {m: torch.randn(m, H, dtype=torch.bfloat16, device="cuda") for m in (1, 2, 4)}
    for m, d in cases:
        routes = [routing(m, d, seed=li) for li in range(LAYERS)]
        rows.append(run_case(f"M{m}_D{d}", layers, xs[m], routes, sync, cnt))
    for m in (1, 2, 4):
        for s in range(SEEDS):
            rr = [uniform_routing(m, 1000 * s + li) for li in range(LAYERS)]
            routes = [(a, b) for a, b, _ in rr]
            r = run_case(f"uni_M{m}_s{s}", layers, xs[m], routes, sync, cnt)
            r["D_per_layer"] = [c for _, _, c in rr]
            rows.append(r)
    passed = all(r["ok"] for r in rows)
    print(json.dumps({
        "passed": passed, "shape": SHAPE, "slot_start": SLOT, "layers": LAYERS,
        "sequence": f"warmup(6) x{LAYERS}; capture() 3 eager + graph; {WARM} warmup + {REPLAYS} replays, "
                    "512 MB read-only flush before each; checks after every replay",
        "compared": "fused 8-layer graph vs the same graph on main's fc1_act_q1+fc2_finalize "
                    "(st._force_chain): q1/sf1 bitwise, out within main's band, finite, counters 0",
        "fc2_stage_info": stage_info, "k_tiles2_gt_ab_stage2": stage_ok,
        "cases": len(rows), "failed": [r["case"] for r in rows if not r["ok"]],
        "rows": [{k: v for k, v in r.items() if k != "per_replay_max_abs"} for r in rows],
        "per_replay_max_abs": {r["case"]: r["per_replay_max_abs"] for r in rows}}))


if __name__ == "__main__":
    main()
