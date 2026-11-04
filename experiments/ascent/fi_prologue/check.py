"""Idea 135 correctness: fused prologue (global ids + in-kernel zero-fill) vs main's leaf path.

Reference = main's Sm12xW4a8DecodeState.run path in this same process (torch local_ids +
torch.zeros buffer + route with local ids; FlashInfer kernels with their default arguments,
i.e. the pinned code path), toggled by the module flag _ROUTE_FUSES_PROLOGUE.
Checks, per slot_start in 0/96/192/288 and M in 1..32 (plus edge routings):
  * eager: candidate output bitwise vs reference (M=1: within the reference's own
    run-to-run spread), route workspace bitwise (offsets, token_map, token_weights, dst_rows,
    scale_dst_rows, routed q_out rows, scale_out), with the caching allocator poisoned (0xFF)
    before every candidate call so the zero-fill is exercised;
  * graph: capture the candidate at M=1/2/4/8 (+16/32), check no Triton JIT inside capture,
    replay 10x, poisoning the leaf's buffer storage and the route workspace before every
    replay, comparing every replay with the eager reference.
Last line: {"passed": ..., ...}.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                "w4a8_fi_chain"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from bench import routing  # noqa: E402
from common import E, H, TOPK, log, make_weights, trtllm_layout  # noqa: E402

SLOTS = (0, 96, 192, 288)
MS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
EDGE = ((4, 6), (8, 0), (2, 1), (32, 6))
N_LOCAL = {1: 3}
GRAPH_MS = (1, 2, 4, 8, 16, 32)
REPLAYS = 10


def jit_count():
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_mxfp8_q0_route_triton as r
    n = 0
    for fn in (r._route_assign_decode_kernel, r._mxfp8_q0_route_direct_kernel):
        try:
            for v in fn.device_caches.values():
                n += len(v[0])
        except Exception:  # noqa: BLE001
            return -1
    return n


def poison_allocator(nbytes):
    junk = [torch.full((nbytes,), 0xFF, dtype=torch.uint8, device="cuda") for _ in range(64)]
    del junk


def snapshot(ws):
    n = int(ws.offsets[-1].item())
    return {
        "offsets": ws.offsets.clone(),
        "token_map": ws.token_map[:n].clone(),
        "token_weights": ws.token_weights[:n].clone(),
        "dst_rows": ws.dst_rows.clone(),
        "scale_dst_rows": ws.scale_dst_rows.clone(),
        "q_out": ws.q_out[:n].view(torch.uint8).clone(),
        "scale_out": ws.scale_out.clone(),
    }


def ws_equal(a, b):
    bad = [k for k in a if a[k].shape != b[k].shape or not torch.equal(
        a[k].view(torch.uint8) if a[k].dtype != torch.uint8 else a[k],
        b[k].view(torch.uint8) if b[k].dtype != torch.uint8 else b[k])]
    return bad


def poison_ws(ws):
    for t in (ws.offsets, ws.token_map, ws.dst_rows, ws.scale_dst_rows, ws.scale_out):
        t.fill_(-1)
    ws.token_weights.fill_(float("nan"))
    ws.q_out.view(torch.uint8).fill_(0xFF)


def bits_equal(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def maxdiff(a, b):
    return (a.float() - b.float()).abs().max().item()


def main():
    import tensorrt_llm  # noqa: F401
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from tensorrt_llm._torch.moe.fused_moe.cutlass import w4a8_mxfp4_mxfp8_sm12x_decode as mod
    from tensorrt_llm._torch.moe.fused_moe.cutlass.w4a8_mxfp4_mxfp8_sm12x_decode import \
        Sm12xW4a8DecodeState

    assert mod._route_fuses_prologue(), "FI overlay route lacks expert_offset/zero_buf"
    w1, s1, w2, s2 = make_weights(seed=0)
    w31, s31, w2v, s2s = trtllm_layout(w1, s1, w2, s2)
    passed = True
    report = {"eager": {}, "graph": {}}
    for slot in SLOTS:
        st = Sm12xW4a8DecodeState(
            w3_w1_weight=w31, w3_w1_weight_scale=s31, w2_weight=w2v, w2_weight_scale=s2s,
            num_local_experts=E, slot_start=slot, swiglu_limit=10.0, enable_pdl=True)
        mod._ROUTE_FUSES_PROLOGUE = False
        st.warmup(TOPK)
        mod._ROUTE_FUSES_PROLOGUE = True
        mod._WARMED.clear()
        st.warmup(TOPK)
        cases = [(m, N_LOCAL.get(m, 2)) for m in MS] + list(EDGE)
        refs = {}
        for m, nl in cases:
            torch.manual_seed(7 * m + nl + slot)
            x = torch.randn(m, H, dtype=torch.bfloat16, device="cuda")
            ids, w = routing(m, nl, slot, seed=31 * m + nl + slot)
            mod._ROUTE_FUSES_PROLOGUE = False
            ref = st.run(x, ids, w).clone()
            ws = mod._workspace(x, ids, E)
            ref_ws = snapshot(ws)
            spread = 0.0
            for _ in range(3):
                spread = max(spread, maxdiff(st.run(x, ids, w), ref))
            mod._ROUTE_FUSES_PROLOGUE = True
            row = {"bitwise_all": True, "max_abs": 0.0, "ws_bad": [], "ref_spread": spread}
            for it in range(5):
                poison_allocator(ref.numel() * 2 + 2 * 1024 * 1024)
                poison_ws(ws)
                out = st.run(x, ids, w)
                bad = ws_equal(snapshot(ws), ref_ws)
                row["ws_bad"] = sorted(set(row["ws_bad"]) | set(bad))
                row["bitwise_all"] &= bits_equal(out, ref)
                row["max_abs"] = max(row["max_abs"], maxdiff(out, ref))
                row["finite"] = bool(torch.isfinite(out).all().item())
            ok = (not row["ws_bad"]) and row["finite"] and (
                row["bitwise_all"] or (m == 1 and row["max_abs"] <= spread))
            # Zero-pair rank (nl=0) must give exact zeros.
            if nl == 0:
                ok &= bool((ref == 0).all().item())
            row["ok"] = ok
            passed &= ok
            report["eager"][f"s{slot}_M{m}_nl{nl}"] = row
            refs[(m, nl)] = (x, ids, w, ref, ref_ws, spread)
            if not ok:
                log("FAIL", slot, m, nl, json.dumps(row))
        log("slot", slot, "eager done, passed so far", passed)

        for m in GRAPH_MS:
            nl = N_LOCAL.get(m, 2)
            x, ids, w, ref, ref_ws, spread = refs[(m, nl)]
            ws = mod._workspace(x, ids, E)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    st.run(x, ids, w)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            j0 = jit_count()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = st.run(x, ids, w)
            torch.cuda.synchronize()
            j1 = jit_count()
            storage = torch.empty(0, dtype=torch.uint8, device="cuda").set_(
                out.untyped_storage())
            row = {"jit_in_capture": j1 - j0, "replays": [], "ok": True}
            for it in range(REPLAYS):
                storage.fill_(0xFF)
                poison_ws(ws)
                g.replay()
                torch.cuda.synchronize()
                be = bits_equal(out, ref)
                md = maxdiff(out, ref)
                wb = ws_equal(snapshot(ws), ref_ws)
                fin = bool(torch.isfinite(out).all().item())
                okr = fin and not wb and (be or (m == 1 and md <= spread))
                row["replays"].append({"bitwise": be, "max_abs": md, "ws_bad": wb, "ok": okr})
                row["ok"] &= okr
            row["ok"] &= (j1 - j0) == 0
            passed &= row["ok"]
            report["graph"][f"s{slot}_M{m}"] = {
                "ok": row["ok"], "jit_in_capture": row["jit_in_capture"],
                "all_bitwise": all(r["bitwise"] for r in row["replays"]),
                "worst_max_abs": max(r["max_abs"] for r in row["replays"]),
                "ref_spread": spread}
            log("graph", slot, m, json.dumps(report["graph"][f"s{slot}_M{m}"]))
            del g
    nbit = sum(1 for r in report["eager"].values() if r["bitwise_all"])
    print(json.dumps({"passed": bool(passed),
                      "compared": "fused-prologue FI leaf vs main's leaf path (torch local_ids "
                                  "+ torch.zeros + pinned-default FI route) on the same inputs",
                      "eager_cases": len(report["eager"]), "eager_bitwise_cases": nbit,
                      "graph": report["graph"],
                      "eager_failures": {k: v for k, v in report["eager"].items()
                                         if not v["ok"]},
                      "eager_nonbitwise": {k: {"max_abs": v["max_abs"],
                                               "ref_spread": v["ref_spread"]}
                                           for k, v in report["eager"].items()
                                           if not v["bitwise_all"]}}))


if __name__ == "__main__":
    main()
