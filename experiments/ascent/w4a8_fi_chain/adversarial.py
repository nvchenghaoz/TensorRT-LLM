"""Adversarial inputs for the FlashInfer W4A8 decode chain vs main's CUTLASS chain (idea 35 debug).

Cases: activation outliers, strided (non-contiguous) x, bf16 routing weights, a token with all 6 picks
local, -1 ids, wide ue8m0 exponent range in the weights. Prints FI-vs-CUTLASS stats per case.
"""

import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from common import (E, H, I, TOPK, CutlassChain, err_stats, log, make_routing,  # noqa: E402
                    make_weights, reference, trtllm_layout)


def main():
    import tensorrt_llm  # noqa: F401
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from tensorrt_llm._torch.moe.fused_moe.cutlass.w4a8_mxfp4_mxfp8_sm12x_decode import \
        Sm12xW4a8DecodeState

    out = {}
    for wname in ("narrow", "wide"):
        w1, s1, w2, s2 = make_weights(seed=3)
        if wname == "wide":
            g = torch.Generator(device="cuda")
            g.manual_seed(7)
            s1 = torch.randint(100, 135, s1.shape, dtype=torch.uint8, device="cuda", generator=g)
            s2 = torch.randint(100, 135, s2.shape, dtype=torch.uint8, device="cuda", generator=g)
        w31, s31, w2v, s2s = trtllm_layout(w1, s1, w2, s2)
        cut = CutlassChain(w31, s31, w2v, s2s)
        st = Sm12xW4a8DecodeState(w3_w1_weight=w31, w3_w1_weight_scale=s31, w2_weight=w2v,
                                  w2_weight_scale=s2s, num_local_experts=E, slot_start=0,
                                  swiglu_limit=10.0, enable_pdl=True)
        st.warmup(TOPK)
        for case in ("plain", "outlier", "strided", "bf16w", "all_local", "neg_ids", "tiny"):
            try:
                m = 4
                torch.manual_seed(11)
                x = torch.randn(m, H, dtype=torch.bfloat16, device="cuda")
                ids, w = make_routing(m, 2, seed=77)
                if case == "outlier":
                    x = x * 0.3
                    x[:, :8] = 400.0
                    x[:, 100] = -900.0
                if case == "tiny":
                    x = x * 1e-3
                if case == "strided":
                    big = torch.randn(m, 4, H, dtype=torch.bfloat16, device="cuda")
                    x = big[:, 1, :]
                if case == "bf16w":
                    w = w.to(torch.bfloat16)
                if case == "all_local":
                    ids = ids.clone()
                    ids[0] = torch.tensor([3, 7, 11, 50, 60, 95], dtype=torch.int32, device="cuda")
                if case == "neg_ids":
                    ids = ids.clone()
                    ids[1, 0] = -1
                torch.cuda.synchronize()
                of = st.run(x, ids, w).clone()
                try:
                    oc = cut(x.contiguous(), ids, w.float()).clone()
                except Exception as e:  # noqa: BLE001
                    oc = None
                    log("cutlass failed", case, repr(e))
                torch.cuda.synchronize()
                ref = reference(x.contiguous(), ids.clamp(min=0) if case != "neg_ids" else ids, w.float(),
                                w1, s1, w2, s2) if case != "neg_ids" else None
                rec = {"fi_finite": bool(torch.isfinite(of).all())}
                if oc is not None:
                    rec["cut_finite"] = bool(torch.isfinite(oc).all())
                    rec["fi_vs_cut"] = err_stats(of, oc.float())
                if ref is not None:
                    rec["fi_vs_ref"] = err_stats(of, ref)
                    if oc is not None:
                        rec["cut_vs_ref"] = err_stats(oc, ref)
                out[f"{wname}/{case}"] = rec
                log(wname, case, json.dumps(rec))
            except Exception as ex:  # noqa: BLE001
                traceback.print_exc()
                out[f"{wname}/{case}"] = {"error": repr(ex)}
        del w1, s1, w2, s2, w31, s31, w2v, s2s, cut, st
        torch.cuda.empty_cache()
    print(json.dumps({"passed": all(v.get("fi_finite", False) for v in out.values()), "cases": out}))


if __name__ == "__main__":
    main()
