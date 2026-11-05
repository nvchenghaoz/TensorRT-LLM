"""Idea 201 correctness (untimed): fused fc1+fc2 vs main's fc1_act_q1 + fc2_finalize.

Part A, repeated execution (protocol repeated_execution_check), per (M, D), D = 0..12 at
M = 1/2/4/8 (D <= 6M): the bench harness sequence exactly (3 eager warmup calls inside
capture(), CUDA graph, then REPLAYS replays each after a 512 MB read-only flush).
  * BASE graph (main's chain) replayed REPLAYS times -> reference q1/sf1 and out_0, and main's
    own replay-to-replay band of out.
  * FUSED graph replayed REPLAYS times; after EVERY replay: q1/sf1 bitwise == BASE's, out vs
    BASE out_0 inside the band (rows of tokens with one local expert must be exact; rows with
    k local experts may differ by the atomic order only, |d| <= max(band, (k-1) ulp_bf16)),
    finite, and the sync workspace + split-K counters all 0 (the state the next call expects).
  * both against the FP32 dequantized reference (common.reference): max abs / rel error.
Part B, leaf: Sm12xW4a8DecodeState.warmup, then CUDA-graph capture of st.run at batch 1..32:
no memset node and no JIT during capture (compile cache unchanged), replay == eager.
Part C, aux stream: a large cuBLAS GEMM on a second stream launched right before each fused
call (it holds SMs while the persistent grid launches) for AUX_ITERS rounds: no hang, outputs
bitwise equal to the solo run's q1/sf1, counters 0 after.
Last line: {"passed": ..., ...}.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from common12 import E, E_GLOBAL, H, SHAPE, SLOT, TOPK, Runner, capture, log, make_state, routing  # noqa: E402,I001

import common as _common  # noqa: E402  (on sys.path via common12)

REPLAYS = int(os.environ.get("CHECK_REPLAYS", "20"))
AUX_ITERS = int(os.environ.get("AUX_ITERS", "200"))
FLUSH = None
W_RAW = None


def replay_all(g, bufs, n):
    q1, sf1, out = bufs
    res = []
    for _ in range(n):
        FLUSH.sum()
        g.replay()
        torch.cuda.synchronize()
        res.append((q1.clone(), sf1.clone(), out.clone()))
    return res


def local_ids(ids):
    """Global ids -> this rank's local view (local experts in [0, E)), as common.reference wants."""
    return ((ids.long() - SLOT) % E_GLOBAL).to(torch.int32)


_DQ = {}


def abs_contrib(x, lids, w, w1, s1, w2, s2):
    """sum_j |w_j * fc2_j(x)| per output element (FP32 dequantized): the magnitude that bounds
    what reordering the bf16 scatter-adds of k contributions can change, (k-1) ulps of it."""
    m = x.shape[0]
    acc = torch.zeros(m, H, dtype=torch.float32, device=x.device)
    xf = x.float()
    for t in range(m):
        for j in range(TOPK):
            le = int(lids[t, j])
            if not (0 <= le < E):
                continue
            if le not in _DQ:
                _DQ[le] = (_common.dequant_expert(w1[le], s1[le]), _common.dequant_expert(w2[le], s2[le]))
            W1, W2 = _DQ[le]
            h = W1 @ xf[t]
            lin, gate = h[:_common.I], h[_common.I:]
            gate = gate.clamp(max=_common.LIMIT)
            lin = lin.clamp(-_common.LIMIT, _common.LIMIT)
            acc[t] += (float(w[t, j]) * (W2 @ (gate * torch.sigmoid(gate) * lin))).abs()
    return acc


def ulp_bf16(x):
    a = x.abs().float().clamp_min(2.0 ** -126)
    return torch.pow(2.0, torch.floor(torch.log2(a)) - 7)


def part_a(leaf, st, fk, w_raw):
    sync = fk.sync_workspace("cuda")
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x._moe_utils import split_k
    scratch, cnt = split_k.workspace(torch.device("cuda"), torch.cuda.get_device_properties(0).multi_processor_count)
    rows, ok = [], True
    runner = Runner(leaf, st)
    for m in (1, 2, 4, 8):
        x = torch.randn(m, H, dtype=torch.bfloat16, device="cuda")
        for d in range(0, min(12, TOPK * m) + 1):
            ids, w = routing(m, d)
            # local experts per token -> allowed atomic-order slack per row
            lids = local_ids(ids)
            loc = ((lids >= 0) & (lids < E)).sum(-1).float()
            gb = capture(lambda: runner(x, ids, w, "base"))
            b = runner.buffers(x, ids)[1:]
            base = replay_all(gb, b, REPLAYS)
            del gb
            q1r, sf1r, out0 = base[0]
            band = max((r[2].float() - out0.float()).abs().max().item() for r in base)
            reorder = (loc[:, None] - 1).clamp_min(0) * ulp_bf16(abs_contrib(x, lids, w, *w_raw))
            base_q_stable = all(torch.equal(r[0].view(torch.uint8), q1r.view(torch.uint8))
                                and torch.equal(r[1], sf1r) for r in base)
            gf = capture(lambda: runner(x, ids, w, "fused"))
            fused = replay_all(gf, b, REPLAYS)
            per = []
            case_ok = base_q_stable
            for i, (q, s, o) in enumerate(fused):
                qeq = torch.equal(q.view(torch.uint8), q1r.view(torch.uint8))
                seq = torch.equal(s, sf1r)
                dlt = (o.float() - out0.float()).abs()
                slack = torch.maximum(torch.full_like(dlt, band), reorder)
                inband = bool((dlt <= slack).all().item())
                fin = bool(torch.isfinite(o.float()).all().item())
                zero = int((sync != 0).sum().item()) == 0 and int((cnt != 0).sum().item()) == 0
                per.append({"replay": i, "q1": qeq, "sf1": seq, "max_abs": round(dlt.max().item(), 5),
                            "in_band": inband, "finite": fin, "counters_zero": zero})
                case_ok = case_ok and qeq and seq and inband and fin and zero
            del gf
            ref = _common.reference(x, lids, w, *w_raw)
            eb = _common.err_stats(out0, ref)
            ef = _common.err_stats(fused[-1][2], ref)
            row = {"M": m, "D": d, "ok": case_ok, "base_q_stable": base_q_stable,
                   "base_band": band, "fused_max_abs_vs_base": max(p["max_abs"] for p in per),
                   "all_replays_q1_sf1_bitwise": all(p["q1"] and p["sf1"] for p in per),
                   "all_counters_zero": all(p["counters_zero"] for p in per),
                   "ref_base": {k: eb[k] for k in ("max_abs", "max_rel_floor5pct", "calc_diff")},
                   "ref_fused": {k: ef[k] for k in ("max_abs", "max_rel_floor5pct", "calc_diff")},
                   "per_replay_max_abs": [p["max_abs"] for p in per]}
            rows.append(row)
            ok = ok and case_ok
            log(f"A {SHAPE} M{m} D{d}: ok={case_ok} band={band} fused-vs-base max {row['fused_max_abs_vs_base']} "
                f"ref err base {eb['max_abs']:.4g}/{eb['calc_diff']:.2e} fused {ef['max_abs']:.4g}/{ef['calc_diff']:.2e}")
    return ok, rows


def part_b(leaf, st, fk):
    st.warmup(TOPK)
    n_comp = len(fk._COMPILED)
    ok, rows = True, []
    base_runner = Runner(leaf, st)
    from torch.profiler import ProfilerActivity, profile
    for m in range(1, leaf.FI_DECODE_MAX_TOKENS + 1):
        x = torch.randn(m, H, dtype=torch.bfloat16, device="cuda")
        ids, w = routing(m, min(6 * m, 12))
        eager = st.run(x, ids, w).clone()
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            with torch.cuda.graph(g):
                o = st.run(x, ids, w)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            g.replay()
            torch.cuda.synchronize()
        names = [e.name for e in prof.events() if e.device_type.name == "CUDA"]
        memset = [n for n in names if "memset" in n.lower() or "Memset" in n]
        lids = local_ids(ids)
        loc = ((lids >= 0) & (lids < E)).sum(-1).float()
        # main's own eager run-to-run band at this case (scatter atomics), 8 runs
        bouts = []
        for _ in range(8):
            bouts.append(base_runner(x, ids, w, "base")[2].clone())
        torch.cuda.synchronize()
        bband = max((b.float() - bouts[0].float()).abs().max().item() for b in bouts)
        dlt = (o.float() - bouts[0].float()).abs()
        slack = torch.maximum(torch.full_like(dlt, bband),
                              (loc[:, None] - 1).clamp_min(0) * ulp_bf16(abs_contrib(x, lids, w, *W_RAW)))
        inband = bool((dlt <= slack).all().item())
        fused_used = any("fc12" in n.lower() or "Fc12" in n for n in names) or len(
            [n for n in names if "kernel_cutlass" in n]) == 1
        r = {"M": m, "memset_nodes": len(memset), "kernels": len(names), "fused_kernel_used": fused_used,
             "replay_in_main_band": inband, "main_band": bband, "replay_vs_main_max_abs": dlt.max().item(),
             "replay_vs_eager_max_abs": (o.float() - eager.float()).abs().max().item()}
        rows.append(r)
        ok = ok and not memset and inband and fused_used
        del g
    jit_in_capture = len(fk._COMPILED) != n_comp
    ok = ok and not jit_in_capture
    log(f"B capture 1..{leaf.FI_DECODE_MAX_TOKENS}: ok={ok} jit_in_capture={jit_in_capture} "
        f"memset={sum(r['memset_nodes'] for r in rows)} fused_used_all={all(r['fused_kernel_used'] for r in rows)}")
    return ok, {"jit_in_capture": jit_in_capture, "rows": rows}


def part_c(leaf, st, fk):
    sync = fk.sync_workspace("cuda")
    runner = Runner(leaf, st)
    m, d = 4, 12
    x = torch.randn(m, H, dtype=torch.bfloat16, device="cuda")
    ids, w = routing(m, d)
    q1, sf1, _, _ = runner(x, ids, w, "fused")
    torch.cuda.synchronize()
    q1r, sf1r = q1.clone(), sf1.clone()
    a = torch.randn(4096, 5120, dtype=torch.bfloat16, device="cuda")
    bmat = torch.randn(5120, 5120, dtype=torch.bfloat16, device="cuda")
    side = torch.cuda.Stream()
    t0 = time.time()
    bad = 0
    for i in range(AUX_ITERS):
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            c = a @ bmat  # shared-expert stand-in holding SMs
        q1, sf1, _, _ = runner(x, ids, w, "fused")
        torch.cuda.current_stream().wait_stream(side)
        if i % 20 == 19:
            torch.cuda.synchronize()
            if not (torch.equal(q1.view(torch.uint8), q1r.view(torch.uint8)) and torch.equal(sf1, sf1r)):
                bad += 1
    torch.cuda.synchronize()
    del c
    dt = time.time() - t0
    zero = int((sync != 0).sum().item()) == 0
    ok = bad == 0 and zero
    log(f"C aux-stream {AUX_ITERS} rounds: ok={ok} mismatches={bad} counters_zero={zero} in {dt:.1f}s (no hang)")
    return ok, {"rounds": AUX_ITERS, "mismatch_checks": bad, "counters_zero": zero, "seconds": round(dt, 2)}


def main():
    global FLUSH, W_RAW
    import tensorrt_llm  # noqa: F401
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import kernel_moe_mxfp8_mxfp4_fc12_fused as fk

    torch.manual_seed(0)
    w_raw = _common.make_weights(seed=0)
    W_RAW = w_raw
    leaf, st = make_state(seed=0)
    FLUSH = torch.ones(512 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    oka, ra = part_a(leaf, st, fk, w_raw)
    okb, rb = part_b(leaf, st, fk)
    okc, rc = part_c(leaf, st, fk)
    print(json.dumps({"passed": oka and okb and okc, "shape": SHAPE, "slot_start": SLOT,
                      "compared": "fused fc1+fc2 vs main fc1_act_q1+fc2_finalize (protected FI ops) "
                                  "and FP32 dequantized reference",
                      "A_repeated_execution": {"ok": oka, "replays": REPLAYS, "rows": ra},
                      "B_graph_capture": {"ok": okb, **rb}, "C_aux_stream": {"ok": okc, **rc}}))


if __name__ == "__main__":
    main()
