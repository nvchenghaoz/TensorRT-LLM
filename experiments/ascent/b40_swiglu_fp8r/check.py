"""Untimed bitwise check: fused SwiGLU+fp8r quant vs silu_and_mul -> quantize_fp8r.

For M in 4096/4176/4256 and inputs {realistic gate_up GEMM output, special-case random}:
reference = unfused chain (+ down GEMM BF16 output). The candidate runs the same sequence as the
timing script (WARMUP warmups + ITERS iterations); every iteration's e4m3 data, fp32 scales and
down output are compared bitwise with the reference and with iteration 0, and the input is
checked unchanged against a pristine copy. Last stdout line: {"passed": ...}.
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

WARMUP = 10
ITERS = 50


def eq(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype == torch.float8_e4m3fn:
        a, b = a.view(torch.uint8), b.view(torch.uint8)
    elif a.dtype == torch.float32:
        a, b = a.view(torch.int32), b.view(torch.int32)
    elif a.dtype == torch.bfloat16:
        a, b = a.view(torch.int16), b.view(torch.int16)
    return bool(torch.equal(a, b))


def main():
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.visual_gen.models.flux3 import fp8

    if len(sys.argv) > 1:
        fp8._SWIGLU_FP8R_NUM_WARPS = int(sys.argv[1])
    torch.manual_seed(0)
    w_up, w_down = C.weights()
    results = []
    ok_all = True
    for m in (4096, 4176, 4256):
        for kind in ("gemm", "special"):
            if kind == "gemm":
                x, _ = C.gate_up_input(m, w_up, seed=m)
            else:
                x = C.special_input(m, seed=m + 1)
            pristine = x.clone()
            ref_d, ref_s = C.unfused(x)
            ref_s = ref_s.reshape(-1)
            ref_o = C.gemm(ref_d, ref_s, w_down)
            torch.cuda.synchronize()
            first = None
            bad = []
            for it in range(WARMUP + ITERS):
                d, s = C.fused(x)
                o = C.gemm(d, s, w_down)
                torch.cuda.synchronize()
                checks = {
                    "data_vs_ref": eq(d, ref_d),
                    "scales_vs_ref": eq(s, ref_s),
                    "down_vs_ref": eq(o, ref_o),
                    "input_unchanged": eq(x, pristine),
                }
                if first is None:
                    first = (d.clone(), s.clone(), o.clone())
                else:
                    checks["data_vs_it0"] = eq(d, first[0])
                    checks["scales_vs_it0"] = eq(s, first[1])
                    checks["down_vs_it0"] = eq(o, first[2])
                if not all(checks.values()):
                    bad.append({"it": it, **checks})
            # stats for the report
            nd = (first[0].view(torch.uint8) != ref_d.view(torch.uint8)).sum().item()
            negzero = int((ref_d.view(torch.uint8) == 0x80).sum().item())
            row = {"M": m, "input": kind, "iters": WARMUP + ITERS, "failures": len(bad),
                   "first_failures": bad[:3], "data_mismatch_it0": nd,
                   "ref_negzero_e4m3": negzero,
                   "zero_row_scale": float(ref_s[0]) if kind == "special" else None,
                   "scale_min": float(ref_s.min()), "scale_max": float(ref_s.max())}
            ok = not bad
            ok_all &= ok
            results.append(row)
            C.log(f"[check] M={m} {kind}: ok={ok} {json.dumps(row)}")
    print(json.dumps({"passed": ok_all, "compared_against":
                      "torch.ops.trtllm.silu_and_mul -> quantize_fp8r (warp_fp8_per_token_quant) "
                      "and down torch._scaled_mm on its output; bitwise per iteration and vs it0",
                      "num_warps": fp8._SWIGLU_FP8R_NUM_WARPS, "cases": results}))


if __name__ == "__main__":
    main()
