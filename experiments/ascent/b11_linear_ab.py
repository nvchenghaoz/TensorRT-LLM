"""Branch 11: A/B the SM120 CuTe NVFP4 LINEAR kernel against TRT-LLM nvfp4_gemm backends.

Synthetic tensors at the FLUX.3 Klein single-block shapes (M = 4096 + T, T a multiple of 80).
Activation and weight quantized with the same flashinfer nvfp4_quantize (layout_128x4, no shuffle)
that flux3_primsts uses. Prints a per-shape table (warm and cold L2) and the tactic each backend picks.
"""

import json
import os
import sys
import time

import torch

import tensorrt_llm  # noqa: F401  (loads torch.classes.trtllm)
import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.custom_ops.torch_custom_ops import (
    CublasLtFP4GemmRunner,
    FP4GemmRunner,
)
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import flux3_sm120_nvfp4
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_primsts import nvfp4_quantize

ITERS = int(os.environ.get("B11_ITERS", "60"))
SWEEP_ITERS = 10
Ms = [int(x) for x in os.environ.get("B11_MS", "4176,4256").split(",")]
SHAPES = [
    (3072, 3072, "attn_out"),
    (3072, 9216, "down"),
    (9216, 3072, "qkv_plain"),
    (18432, 3072, "gate_up_plain"),
]
if os.environ.get("B11_SHAPES"):
    keep = os.environ["B11_SHAPES"].split(",")
    SHAPES = [s for s in SHAPES if s[2] in keep]


def log(*a):
    print(*a, flush=True)


def quant(x):
    amax = x.abs().amax().float()
    inv = (448.0 * 6.0 / amax).reshape(1).cuda()
    data, sf = nvfp4_quantize(x, inv)
    return data, sf, float(1.0 / inv.item())


flush_buf = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")


def time_fn(fn, iters, cold):
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    for i in range(iters):
        if cold:
            flush_buf.zero_()
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    t = sorted(s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends))
    med = t[len(t) // 2]
    p10, p90 = t[len(t) // 10], t[(len(t) * 9) // 10]
    return med, (p90 - p10) / med


def main():
    torch.manual_seed(0)
    results = {}
    cutlass_runner = FP4GemmRunner(
        fp4_utils.FP4GemmType.W4A4_NVFP4_NVFP4, 0, torch.bfloat16
    )
    cublas_runner = CublasLtFP4GemmRunner(0, torch.bfloat16)
    for M in Ms:
        for N, K, name in SHAPES:
            key = f"M{M}_N{N}_K{K}_{name}"
            log(f"== {key}")
            x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
            a_fp4, a_sf, a_gs = quant(x)
            w_fp4, w_sf, w_gs = quant(w)
            alpha = torch.tensor([a_gs * w_gs], device="cuda", dtype=torch.float32)
            inputs = [a_fp4, w_fp4, a_sf, w_sf, alpha]
            pristine = [t.clone() for t in inputs]

            def cute():
                return flux3_sm120_nvfp4.run("linear", a_fp4, w_fp4, a_sf, w_sf, a_gs, w_gs)

            ref = cute().clone()
            row = {}
            # CuTe
            row["cute"] = {
                "warm_us": time_fn(cute, ITERS, False),
                "cold_us": time_fn(cute, ITERS, True),
            }
            log("  cute", row["cute"])
            # sweep tactics
            for bname, runner in (("cutlass", cutlass_runner), ("cublaslt", cublas_runner)):
                try:
                    tactics = runner.get_valid_tactics(inputs, None)
                except Exception as e:  # noqa: BLE001
                    log(f"  {bname} get_valid_tactics failed: {e}")
                    continue
                sweep = {}
                for t in tactics:
                    try:
                        out = runner(inputs=inputs, tactic=t)
                        torch.cuda.synchronize()
                        med, _ = time_fn(lambda: runner(inputs=inputs, tactic=t), SWEEP_ITERS, False)
                        diff = (out.float() - ref.float()).abs()
                        sweep[t] = (med, bool(torch.equal(out, ref)), float(diff.max()))
                    except Exception as e:  # noqa: BLE001
                        sweep[t] = (float("inf"), False, str(e)[:80])
                log(f"  {bname} sweep:", {k: (round(v[0], 1), v[1], v[2]) for k, v in sweep.items()})
                good = {k: v for k, v in sweep.items() if v[0] != float("inf")}
                if not good:
                    continue
                best = min(good, key=lambda k: good[k][0])
                fn = lambda: runner(inputs=inputs, tactic=best)  # noqa: E731
                out = fn()
                row[bname] = {
                    "tactic": best,
                    "warm_us": time_fn(fn, ITERS, False),
                    "cold_us": time_fn(fn, ITERS, True),
                    "bitwise_vs_cute": bool(torch.equal(out, ref)),
                    "max_abs_vs_cute": float((out.float() - ref.float()).abs().max()),
                    "ref_absmax": float(ref.float().abs().max()),
                }
                log(f"  {bname} best", row[bname])
            # unified autotuned op
            try:
                AutoTuner.get().clear_cache()
                with autotune():
                    torch.ops.trtllm.nvfp4_gemm(
                        a_fp4, w_fp4, a_sf, w_sf, alpha, torch.bfloat16,
                        allowed_backends="cutlass,cublaslt,cuda_core,cutedsl")
                fn = lambda: torch.ops.trtllm.nvfp4_gemm(  # noqa: E731
                    a_fp4, w_fp4, a_sf, w_sf, alpha, torch.bfloat16,
                    allowed_backends="cutlass,cublaslt,cuda_core,cutedsl")
                cache = {str(k): str(v) for k, v in AutoTuner.get().profiling_cache.cache.items()
                         if "nvfp4_gemm" in str(k)}
                out = fn()
                row["nvfp4_gemm_auto"] = {
                    "pick": list(cache.values())[:2],
                    "warm_us": time_fn(fn, ITERS, False),
                    "cold_us": time_fn(fn, ITERS, True),
                    "bitwise_vs_cute": bool(torch.equal(out, ref)),
                }
                log("  auto", row["nvfp4_gemm_auto"])
            except Exception as e:  # noqa: BLE001
                log(f"  auto failed: {e}")
            for p, t in zip(pristine, inputs):
                assert torch.equal(p, t), "input mutated"
            results[key] = row
    log(json.dumps(results, default=str))
    print(json.dumps({"passed": True, "table": results}, default=str))


if __name__ == "__main__":
    main()
