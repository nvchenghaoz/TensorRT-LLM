"""Step 0 + Stage 1 (numerics gate + 1-GPU sizing) for idea 35.

Runs on one GPU with synthetic tensors at one EP4 rank's shapes.  Prints progress and a final
JSON line.  Usage: python stage1.py
"""

import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from common import (E, H, I, TOPK, CutlassChain, FiChain, err_stats, fi_sf_from_raw,  # noqa: E402
                    log, make_routing, make_weights, reference, time_graph, trtllm_layout)

N_LOCAL = {1: 3, 2: 2, 4: 2, 8: 2}
RESULT = {"step0": {}, "jit": {}, "numerics": {}, "timing": {}, "repeat": {}}


def step0():
    import flashinfer
    from flashinfer.cute_dsl import is_cute_dsl_available
    from flashinfer.jit.cpp_ext import get_cuda_version
    from flashinfer.utils import get_compute_capability

    try:
        import cutlass
        dsl_ver = getattr(cutlass, "__version__", "?")
    except Exception as e:  # noqa: BLE001
        dsl_ver = f"import failed: {e}"
    try:
        from importlib.metadata import version
        dsl_ver_pkg = version("nvidia-cutlass-dsl")
    except Exception:  # noqa: BLE001
        dsl_ver_pkg = "?"
    dev = torch.device("cuda")
    s0 = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "flashinfer": flashinfer.__version__,
        "flashinfer_cuda": str(get_cuda_version()),
        "cutlass_dsl": dsl_ver,
        "cutlass_dsl_pkg": dsl_ver_pkg,
        "gpu": torch.cuda.get_device_name(),
        "cc": list(get_compute_capability(dev)),
        "check_support": {
            "cuda_major_ge_13": get_cuda_version().major >= 13,
            "cute_dsl_available": bool(is_cute_dsl_available()),
            "cc_in_120_121": tuple(get_compute_capability(dev)) in ((12, 0), (12, 1)),
            "do_finalize": True,
            "use_fused_finalize": True,
            "per_token_scale_none": True,
            "topk_weights_fp32": True,
            "swiglu_default_alpha_beta_limit_only": True,
        },
    }
    RESULT["step0"] = s0
    log("step0", json.dumps(s0))


def main():
    import tensorrt_llm  # noqa: F401
    import tensorrt_llm._torch.custom_ops  # noqa: F401

    torch.manual_seed(0)
    step0()
    log("weights ...")
    w1, s1, w2, s2 = make_weights()
    w31, s31, w2v, s2s = trtllm_layout(w1, s1, w2, s2)
    # The load-time conversion path the leaf will use: reverse the swizzle, then repack.
    r1 = torch.ops.trtllm.block_scale_interleave_reverse(
        s31.view(torch.uint8).reshape(E, 2 * I, H // 32))
    r2 = torch.ops.trtllm.block_scale_interleave_reverse(
        s2s.view(torch.uint8).reshape(E, H, I // 32))
    RESULT["step0"]["reverse_swizzle_matches_raw"] = bool(torch.equal(r1, s1) and torch.equal(r2, s2))
    w1_sf = fi_sf_from_raw(r1)
    w2_sf = fi_sf_from_raw(r2)
    RESULT["step0"]["fi_sf_shapes"] = [list(w1_sf.shape), list(w2_sf.shape)]
    RESULT["step0"]["fp4_data_shared"] = w31.data_ptr() == w1.data_ptr() and w2v.data_ptr() == w2.data_ptr()
    RESULT["step0"]["extra_sf_bytes_per_layer"] = w1_sf.numel() + w2_sf.numel()
    log("step0 extra", RESULT["step0"]["reverse_swizzle_matches_raw"], RESULT["step0"]["fi_sf_shapes"])

    cut = CutlassChain(w31, s31, w2v, s2s)
    fi = FiChain(w1, w1_sf, w2, w2_sf, pdl=False)
    fi_pdl = FiChain(w1, w1_sf, w2, w2_sf, pdl=True)
    fi_nz = FiChain(w1, w1_sf, w2, w2_sf, pdl=True, zero_sf1=False)

    cases = {}
    for m in (1, 2, 4, 8):
        x = torch.randn(m, H, dtype=torch.bfloat16, device="cuda")
        ids, w = make_routing(m, N_LOCAL[m], seed=m)
        cases[str(m)] = (x, ids, w)
    # extra correctness case: one token with no local pair must stay exactly zero
    x = torch.randn(4, H, dtype=torch.bfloat16, device="cuda")
    ids, w = make_routing(4, 2, seed=44, empty_token=1)
    cases["4e"] = (x, ids, w)

    # JIT cost: first call per M (compiles on the first only if the tile is shared)
    for name, (x, ids, w) in cases.items():
        for tag, ch in (("fi", fi), ("fi_pdl", fi_pdl)):
            try:
                torch.cuda.synchronize()
                t = time.time()
                ch(x, ids, w)
                torch.cuda.synchronize()
                t1 = time.time() - t
                t = time.time()
                ch(x, ids, w)
                torch.cuda.synchronize()
                t2 = time.time() - t
                RESULT["jit"][f"{tag}_M{name}"] = {"first_s": round(t1, 4), "second_s": round(t2, 5)}
                log("jit", tag, name, t1, t2)
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                RESULT["jit"][f"{tag}_M{name}"] = "FAILED"
    try:
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import (
            moe_mxfp8_mxfp4_fc1_act_q1 as f1m, moe_mxfp8_mxfp4_fc2_finalize as f2m)
        RESULT["jit"]["compiled_fc1_keys"] = [str(k) for k in f1m._COMPILED]
        RESULT["jit"]["compiled_fc2_keys"] = [str(k) for k in f2m._COMPILED]
    except Exception:  # noqa: BLE001
        traceback.print_exc()

    # Numerics
    refs = {}
    for name, (x, ids, w) in cases.items():
        ref = reference(x, ids, w, w1, s1, w2, s2)
        refs[name] = ref
        rec = {}
        try:
            oc = cut(x, ids, w).clone()
            rec["cutlass_vs_ref"] = err_stats(oc, ref)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            oc = None
        for tag, ch in (("fi", fi), ("fi_pdl", fi_pdl), ("fi_nozero_sf1", fi_nz)):
            try:
                of = ch(x, ids, w).clone()
                rec[f"{tag}_vs_ref"] = err_stats(of, ref)
                if oc is not None:
                    rec[f"{tag}_vs_cutlass"] = err_stats(of, oc.float())
                if name == "4e":
                    rec[f"{tag}_empty_token_row_zero"] = bool((of[1] == 0).all().item())
                    if oc is not None:
                        rec["cutlass_empty_token_row_zero"] = bool((oc[1] == 0).all().item())
            except Exception:  # noqa: BLE001
                traceback.print_exc()
        RESULT["numerics"][name] = rec
        log("numerics", name, json.dumps(rec))

    # Graph timing (cold L2) + repeated replays vs reference
    for name in ("1", "2", "4", "8"):
        x, ids, w = cases[name]
        ref = refs[name]
        rec = {}
        for tag, ch in (("cutlass", cut), ("fi", fi), ("fi_pdl", fi_pdl), ("fi_pdl_nozero", fi_nz)):
            try:
                st, g, out = time_graph(ch, (x, ids, w))
                rec[tag] = st
                outs = []
                for _ in range(10):
                    g.replay()
                    torch.cuda.synchronize()
                    outs.append(out.clone())
                errs = [err_stats(o, ref)["max_abs_over_ref_absmax"] for o in outs]
                bitwise = all(torch.equal(outs[0], o) for o in outs[1:])
                RESULT["repeat"][f"{tag}_M{name}"] = {"max_err_over_absmax_per_iter_max": max(errs),
                                                      "min": min(errs), "bitwise_repeat": bitwise}
                del g
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                rec[tag] = "FAILED"
        RESULT["timing"][name] = rec
        log("timing", name, json.dumps(rec))
        log("repeat", name, json.dumps({k: v for k, v in RESULT["repeat"].items() if k.endswith(f"M{name}")}))

    summ = {}
    for name, rec in RESULT["timing"].items():
        try:
            summ[name] = {k: round(v["median_ms"] * 1000, 2) for k, v in rec.items() if isinstance(v, dict)}
        except Exception:  # noqa: BLE001
            pass
    RESULT["summary_us"] = summ
    log("summary_us", json.dumps(summ))
    print(json.dumps(RESULT))


if __name__ == "__main__":
    main()
