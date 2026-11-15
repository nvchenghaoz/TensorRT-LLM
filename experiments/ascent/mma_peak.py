"""Measured tensor-core peaks on this GPU: raw mma.sync microkernels (BF16 m16n8k16
f32-acc, FP16 f16-acc, E4M3 m16n8k32 f32-acc and f16-acc) plus cuBLAS BF16 and FP8
GEMMs at 8192^3. CUDA events, 5 warmup, 20 timed, median/p10/p90."""
import json
import os
import statistics

import torch

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
from torch.utils.cpp_extension import load_inline  # noqa: E402

SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define CH 8
template <int KIND>
__global__ void mma_loop(float* out, int iters) {
  uint32_t a[4], b[2];
  for (int i = 0; i < 4; ++i) a[i] = 0x3c003c00u ^ (threadIdx.x * 2654435761u + i);
  for (int i = 0; i < 2; ++i) b[i] = 0x3c003c00u ^ (threadIdx.x * 40503u + i);
  float c[CH][4];
  uint32_t h[CH][2];
#pragma unroll
  for (int j = 0; j < CH; ++j) { c[j][0]=c[j][1]=c[j][2]=c[j][3]=0.f; h[j][0]=h[j][1]=0u; }
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int j = 0; j < CH; ++j) {
      if (KIND == 0) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])
          : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
      } else if (KIND == 1) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
          : "+r"(h[j][0]), "+r"(h[j][1])
          : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
      } else if (KIND == 2) {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])
          : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
      } else {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
          : "+r"(h[j][0]), "+r"(h[j][1])
          : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
      }
    }
  }
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < CH; ++j) s += c[j][0] + c[j][1] + c[j][2] + c[j][3] + (float)(h[j][0] ^ h[j][1]);
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

void launch(torch::Tensor out, int64_t kind, int64_t blocks, int64_t threads, int64_t iters) {
  float* p = out.data_ptr<float>();
  auto st = at::cuda::getCurrentCUDAStream();
  if (kind == 0) mma_loop<0><<<blocks, threads, 0, st>>>(p, iters);
  else if (kind == 1) mma_loop<1><<<blocks, threads, 0, st>>>(p, iters);
  else if (kind == 2) mma_loop<2><<<blocks, threads, 0, st>>>(p, iters);
  else mma_loop<3><<<blocks, threads, 0, st>>>(p, iters);
}
"""
CPP = "void launch(torch::Tensor out, int64_t kind, int64_t blocks, int64_t threads, int64_t iters);"


def tm(fn, warm=5, iters=20):
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
    return statistics.median(ts), ts[2], ts[-3]


def main():
    p = torch.cuda.get_device_properties(0)
    print("GPU", p.name, p.multi_processor_count, flush=True)
    mod = load_inline("mma_peak_ext", CPP, cuda_sources=SRC.replace(
        "#include <cuda_runtime.h>", "#include <cuda_runtime.h>\n#include <ATen/cuda/CUDAContext.h>"),
        functions=["launch"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a"],
        verbose=False)
    print("built", flush=True)
    sms = p.multi_processor_count
    res = {}
    names = {0: "mma_bf16_f32acc", 1: "mma_f16_f16acc", 2: "mma_e4m3_f32acc", 3: "mma_e4m3_f16acc"}
    flop_per = {0: 16 * 8 * 16 * 2, 1: 16 * 8 * 16 * 2, 2: 16 * 8 * 32 * 2, 3: 16 * 8 * 32 * 2}
    for kind in range(4):
        best = None
        for threads, bpsm in ((128, 4), (256, 4), (512, 2)):
            blocks = sms * bpsm
            iters = 2048
            out = torch.empty(blocks * threads, device="cuda")
            med, p10, p90 = tm(lambda: mod.launch(out, kind, blocks, threads, iters))
            warps = blocks * threads // 32
            fl = warps * iters * 8 * flop_per[kind]
            tf = fl / (med * 1e-3) / 1e12
            print(names[kind], threads, bpsm, f"{med:.3f}ms", f"{tf:.1f} TFLOPS", flush=True)
            if best is None or tf > best["tflops"]:
                best = dict(tflops=tf, ms=med, p10=p10, p90=p90, threads=threads, blocks=blocks)
        res[names[kind]] = best
    N = 8192
    a = torch.randn(N, N, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, N, device="cuda", dtype=torch.bfloat16)
    med, p10, p90 = tm(lambda: a @ b)
    res["cublas_bf16_8192"] = dict(tflops=2 * N ** 3 / (med * 1e-3) / 1e12, ms=med)
    try:
        a8 = a.to(torch.float8_e4m3fn)
        b8 = b.to(torch.float8_e4m3fn).t().contiguous().t()
        one = torch.ones((), device="cuda")
        med, p10, p90 = tm(lambda: torch._scaled_mm(a8, b8, one, one, out_dtype=torch.bfloat16))
        res["cublas_fp8_8192"] = dict(tflops=2 * N ** 3 / (med * 1e-3) / 1e12, ms=med)
    except Exception as ex:
        res["cublas_fp8_8192"] = str(ex)[:200]
    try:
        import pynvml  # noqa
    except Exception:
        pass
    os.system("nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,power.draw,power.limit,temperature.gpu --format=csv")
    print(json.dumps({"passed": True, "peaks": res}))


if __name__ == "__main__":
    main()
