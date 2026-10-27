"""Step-0 survey: BF16 cuBLAS vs FP8 rowwise options on SM120 at the excluded-block shapes."""
import json
import sys

import torch
import torch.nn.functional as F

import tensorrt_llm  # noqa: F401  (registers ops)
from tensorrt_llm._torch.autotuner import autotune
from tensorrt_llm._torch.visual_gen.models.flux3.fp8 import quantize_fp8r

torch.manual_seed(0)
dev = "cuda"
print("device", torch.cuda.get_device_name(), torch.cuda.get_device_capability(), flush=True)


def timeit(fn, iters=50, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2], (ts[int(0.9 * len(ts))] - ts[int(0.1 * len(ts))]) / ts[len(ts) // 2]


res = {}
shapes = [(3072, 3072), (18432, 3072), (3072, 9216)]
for M in (4096, 4256):
    for N, K in shapes:
        tag = f"M{M}_N{N}_K{K}"
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        w = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
        wscale = (w.float().abs().amax(1) / 448.0).clamp_min(1e-12)
        wq = (w.float() / wscale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        xq, xs = quantize_fp8r(x)
        ref = x.float() @ w.float().t()
        r = {}
        r["bf16_ms"], r["bf16_noise"] = timeit(lambda: F.linear(x, w))
        r["quant_ms"], _ = timeit(lambda: quantize_fp8r(x))
        try:
            with autotune():
                torch.ops.trtllm.fp8_rowwise_gemm(xq, wq, xs, wscale, torch.bfloat16)
            out = torch.ops.trtllm.fp8_rowwise_gemm(xq, wq, xs, wscale, torch.bfloat16)
            r["trt_fp8_ms"], r["trt_noise"] = timeit(
                lambda: torch.ops.trtllm.fp8_rowwise_gemm(xq, wq, xs, wscale, torch.bfloat16))
            r["trt_fp8_relerr"] = ((out.float() - ref).norm() / ref.norm()).item()
        except Exception as ex:  # noqa: BLE001
            r["trt_fp8_err"] = repr(ex)[:300]
        try:
            out = torch._scaled_mm(xq, wq.t(), scale_a=xs.reshape(-1, 1), scale_b=wscale.reshape(1, -1),
                                   out_dtype=torch.bfloat16)
            r["scaled_mm_ms"], r["scaled_noise"] = timeit(
                lambda: torch._scaled_mm(xq, wq.t(), scale_a=xs.reshape(-1, 1),
                                         scale_b=wscale.reshape(1, -1), out_dtype=torch.bfloat16))
            r["scaled_mm_relerr"] = ((out.float() - ref).norm() / ref.norm()).item()
        except Exception as ex:  # noqa: BLE001
            r["scaled_mm_err"] = repr(ex)[:300]
        bf = F.linear(x, w)
        r["bf16_relerr"] = ((bf.float() - ref).norm() / ref.norm()).item()
        flop = 2 * M * N * K
        r["bf16_tflops"] = flop / r["bf16_ms"] / 1e9
        if "trt_fp8_ms" in r:
            r["trt_tflops"] = flop / r["trt_fp8_ms"] / 1e9
        if "scaled_mm_ms" in r:
            r["scaled_tflops"] = flop / r["scaled_mm_ms"] / 1e9
        print(tag, json.dumps(r), flush=True)
        res[tag] = r
print(json.dumps({"passed": True, "results": res}))
