"""Untimed correctness for idea 35 (candidate tree only), one GPU, one EP4 rank's shapes.

1. Leaf routing: TrtllmCutlassW4a8Mxfp4Mxfp8Impl.quantize_input/run_moe send M <= threshold to
   the FlashInfer chain (BF16 in, no SF) and larger M to CUTLASS, whose output matches main's
   op called directly (M = 33, 64, 256).
2. Repeated execution matching bench.py exactly (graph per M, 20 warmup + 100 replays, L2
   flush before each): every replay's output vs the FP32 dequantized reference and vs main's
   CUTLASS chain; route workspace state (offsets, token_map) inspected after every replay.
3. A token with no pair on this rank stays exactly zero.
4. No JIT inside capture: CuTe-DSL compile caches and Triton kernel caches unchanged by capture.
5. FP4 weight bytes shared (same storage); extra SF bytes per layer reported.
Last stdout line: {"passed": bool, ...}.
"""

import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from common import (E, H, I, TOPK, CutlassChain, err_stats, log, make_routing,  # noqa: E402
                    make_weights, reference, trtllm_layout)

N_LOCAL = {1: 3, 2: 2, 4: 2, 8: 2}
WARMUP, ITERS = 20, 100
# Error budget: the FlashInfer chain must stay within this multiple of main's CUTLASS
# chain error vs the FP32 reference (both are MXFP8 x MXFP4; they round differently).
ERR_FACTOR = 1.5


def jit_counts():
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_mxfp8_mxfp4_fc1_act_q1 as f1
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_mxfp8_mxfp4_fc2_finalize as f2
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_mxfp8_q0_route_triton as r
    tri = {}
    for name in dir(r):
        obj = getattr(r, name)
        cache = getattr(obj, "device_caches", None) or getattr(obj, "cache", None)
        if cache is not None and hasattr(obj, "run"):
            try:
                tri[name] = sum(len(v[0]) if isinstance(v, tuple) else len(v) for v in cache.values())
            except Exception:  # noqa: BLE001
                tri[name] = str(type(cache))
    try:
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x._moe_utils import moe_route_meta as rm
        for name in dir(rm):
            obj = getattr(rm, name)
            cache = getattr(obj, "device_caches", None) or getattr(obj, "cache", None)
            if cache is not None and hasattr(obj, "run"):
                try:
                    tri["meta." + name] = sum(len(v[0]) if isinstance(v, tuple) else len(v)
                                              for v in cache.values())
                except Exception:  # noqa: BLE001
                    tri["meta." + name] = str(type(cache))
    except Exception:  # noqa: BLE001
        pass
    return {"fc1": len(f1._COMPILED), "fc2": len(f2._COMPILED), "triton": tri}


def make_leaf(st, w31, s31, w2v, s2s):
    """A TrtllmCutlassW4a8Mxfp4Mxfp8Impl with just the state its run paths read."""
    from tensorrt_llm._torch.moe.fused_moe.cutlass.w4a8_mxfp4_mxfp8 import \
        TrtllmCutlassW4a8Mxfp4Mxfp8Impl as Leaf
    from tensorrt_llm._torch.moe.fused_moe.quantization import W4A8MXFP4MXFP8CutlassFusedMoEMethod
    from tensorrt_llm._torch.utils import ActivationType

    leaf = Leaf.__new__(Leaf)
    torch.nn.Module.__init__(leaf)
    fake = torch.ones(E, dtype=torch.float32, device="cuda")
    attrs = dict(
        _fi_decode=st, quant_method=W4A8MXFP4MXFP8CutlassFusedMoEMethod(),
        w3_w1_weight=w31, w2_weight=w2v, w3_w1_bias=None, w2_bias=None,
        quant_scales=[s31, fake, s2s, fake], act_alpha=None, act_beta=None,
        act_clamp=torch.full((E,), 10.0, dtype=torch.float32, device="cuda"),
        act_clamp_after_silu=False, tp_size=1, tp_rank=0, ep_size=4, ep_rank=0,
        cluster_size=1, cluster_rank=0, use_fused_finalize=True, tune_max_num_tokens=8192,
        activation_type=ActivationType.Swiglu, unpadded_hidden_size=H,
    )
    for k, v in attrs.items():
        object.__setattr__(leaf, k, v)
    return leaf


def leaf_call(leaf, x, ids, w):
    from tensorrt_llm._torch.moe.fused_moe.impl_contract import MoECommPlan, MoERunContext

    xq, xsf = leaf.quantize_input(x, post_quant_comm=False)
    plan = MoECommPlan(input_sf_swizzled=True, enable_alltoall=False, moe_output=None,
                       payload_in_workspace=False)
    ctx = MoERunContext(token_selected_experts=ids, token_final_scales=w, x=xq, x_sf=xsf,
                        output_dtype=torch.bfloat16, comm_plan=plan)
    return xq, xsf, leaf.run_moe(ctx)


def main():
    import tensorrt_llm  # noqa: F401
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from tensorrt_llm._torch.moe.fused_moe.cutlass.w4a8_mxfp4_mxfp8_sm12x_decode import (
        FI_DECODE_MAX_TOKENS, Sm12xW4a8DecodeState, fi_decode_available)

    res = {"checks": {}}
    ok_all = True

    def record(name, ok, **info):
        nonlocal ok_all
        ok_all = ok_all and bool(ok)
        res["checks"][name] = {"ok": bool(ok), **info}
        log(name, "OK" if ok else "FAIL", json.dumps(info)[:2000])

    avail, why = fi_decode_available()
    record("fi_decode_available", avail, reason=why)
    torch.manual_seed(0)
    w1, s1, w2, s2 = make_weights()
    w31, s31, w2v, s2s = trtllm_layout(w1, s1, w2, s2)
    pdl = os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"
    st = Sm12xW4a8DecodeState(
        w3_w1_weight=w31, w3_w1_weight_scale=s31, w2_weight=w2v, w2_weight_scale=s2s,
        num_local_experts=E, slot_start=0, swiglu_limit=10.0, enable_pdl=pdl)
    record("fp4_weight_bytes_shared",
           st.w1.data_ptr() == w31.data_ptr() and st.w2.data_ptr() == w2v.data_ptr(),
           extra_sf_bytes_per_layer=st.extra_bytes,
           extra_sf_gib_per_rank_40_layers=st.extra_bytes * 40 / 2**30)
    torch.cuda.synchronize()
    m0 = torch.cuda.memory_allocated()
    st.warmup(TOPK)
    record("warmup_done", True, max_tokens=FI_DECODE_MAX_TOKENS,
           warmup_alloc_delta_mib=(torch.cuda.memory_allocated() - m0) / 2**20)
    cut = CutlassChain(w31, s31, w2v, s2s)
    leaf = make_leaf(st, w31, s31, w2v, s2s)

    # 1. leaf routing + CUTLASS unchanged above the threshold
    for m in (1, FI_DECODE_MAX_TOKENS, FI_DECODE_MAX_TOKENS + 1, 64, 256):
        try:
            x = torch.randn(m, H, dtype=torch.bfloat16, device="cuda")
            ids, w = make_routing(m, 2, seed=500 + m)
            xq, xsf, out = leaf_call(leaf, x, ids, w)
            took_fi = xsf is None and xq.dtype == torch.bfloat16
            if m <= FI_DECODE_MAX_TOKENS:
                direct = st.run(x, ids, w)
                e = err_stats(out, direct.float())
                record(f"leaf_M{m}_takes_fi", took_fi and e["max_abs_over_ref_absmax"] < 2e-2, **e)
            else:
                direct = cut(x, ids, w)
                e = err_stats(out, direct.float())
                # main's CUTLASS output is itself run-to-run nondeterministic (atomic finalize)
                record(f"leaf_M{m}_stays_cutlass", (not took_fi) and e["max_abs_over_ref_absmax"] < 2e-2,
                       bitwise=bool(torch.equal(out, direct)), **e)
        except Exception as ex:  # noqa: BLE001
            traceback.print_exc()
            record(f"leaf_M{m}", False, error=repr(ex))

    # 2-4. repeated execution matching bench.py, graph capture without JIT
    flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    cases = [(str(m), m, N_LOCAL[m], None) for m in (1, 2, 4, 8)] + [("4e", 4, 2, 1)]
    for name, m, nl, empty in cases:
        try:
            x = torch.randn(m, H, dtype=torch.bfloat16, device="cuda")
            ids, w = make_routing(m, nl, seed=m if empty is None else 44, empty_token=empty)
            ref = reference(x, ids, w, w1, s1, w2, s2)
            oc = cut(x, ids, w).clone()
            ec = err_stats(oc, ref)
            n_local_pairs = int(((ids >= 0) & (ids < E)).sum())
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    st.run(x, ids, w)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            before = jit_counts()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = st.run(x, ids, w)
            torch.cuda.synchronize()
            after = jit_counts()
            ws = None
            from tensorrt_llm._torch.moe.fused_moe.cutlass import w4a8_mxfp4_mxfp8_sm12x_decode as mod
            for k, v in mod._WORKSPACES.items():
                if k[1] == m:
                    ws = v
            worst, worst_vs_cut, ws_ok, finite = 0.0, 0.0, True, True
            first = None
            n_bitwise_same = 0
            for i in range(WARMUP + ITERS):
                flush.zero_()
                g.replay()
                torch.cuda.synchronize()
                e = err_stats(out, ref)
                worst = max(worst, e["max_abs_over_ref_absmax"])
                worst_vs_cut = max(worst_vs_cut, err_stats(out, oc.float())["max_abs_over_ref_absmax"])
                finite = finite and e["finite"]
                if first is None:
                    first = out.clone()
                elif torch.equal(first, out):
                    n_bitwise_same += 1
                if ws is not None:
                    offs = ws.offsets
                    ws_ok = ws_ok and int(offs[-1]) == n_local_pairs and bool((offs[1:] >= offs[:-1]).all())
                    tm = ws.token_map[:n_local_pairs]
                    ws_ok = ws_ok and bool(((tm >= 0) & (tm < m)).all())
            tol = max(ERR_FACTOR * ec["max_abs_over_ref_absmax"], 1e-3)
            info = dict(cutlass_err=ec["max_abs_over_ref_absmax"], fi_worst_err=worst,
                        fi_worst_vs_cutlass=worst_vs_cut, tol=tol, finite=finite,
                        replays=WARMUP + ITERS, bitwise_same_as_first=n_bitwise_same,
                        workspace_state_ok=ws_ok, jit_before=before, jit_after=after,
                        fi_calc_diff=err_stats(out, ref)["calc_diff"], cutlass_calc_diff=ec["calc_diff"])
            ok = worst <= tol and finite and ws_ok and before == after
            if empty is not None:
                info["empty_token_row_zero"] = bool((out[empty] == 0).all())
                info["cutlass_empty_token_row_zero"] = bool((oc[empty] == 0).all())
                ok = ok and info["empty_token_row_zero"]
            record(f"repeat_M{name}", ok, **info)
            del g
        except Exception as ex:  # noqa: BLE001
            traceback.print_exc()
            record(f"repeat_M{name}", False, error=repr(ex))

    res["passed"] = ok_all
    res["compared_against"] = ("FP32 dequantized reference (BF16 x, FP32 SwiGLU clamp 10) and "
                               "main's CUTLASS chain (mxfp8_quantize + trtllm.fused_moe) on the same inputs")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
