"""Small CUDA helpers (torch load_inline) for the router pre-wait checks and the P1 stand-in predecessor.

producer(src, dst, spin_ns, grid, block, stream_buf, trigger_early):
    plain launch (no PDL attribute, so it is fully ordered after its own predecessor). If trigger_early it
    executes griddepcontrol.launch_dependents first, so a PDL dependent may start at once. It then reads
    stream_buf (read-only DRAM traffic, grid-strided; may be empty), spins until spin_ns after its start
    (globaltimer), and finally copies src -> dst (the dependent's input). If not trigger_early, it triggers last.
consumer_copy(src, out, wait):
    launched WITH the PDL attribute on the current stream; copies src -> out, after griddepcontrol.wait when
    wait=True. wait=False is the negative control: under a live PDL overlap it reads the poisoned src.
"""
import os

import torch

_CPP = r"""
#include <torch/extension.h>
void producer(torch::Tensor src, torch::Tensor dst, int64_t spin_ns, int64_t grid, int64_t block,
              torch::Tensor stream_buf, bool trigger_early);
void consumer_copy(torch::Tensor src, torch::Tensor out, bool wait);
"""

_CU = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ unsigned long long gtimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__global__ void producer_kernel(const uint4* __restrict__ src, uint4* __restrict__ dst, long long n16,
                                long long spin_ns, const uint4* __restrict__ sbuf, long long sn16,
                                float* __restrict__ sink, int trigger_early) {
    unsigned long long t0 = gtimer();
    if (trigger_early) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    unsigned int acc = 0;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < sn16; i += stride) {
        uint4 v;
        asm volatile("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(sbuf + i));
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    while (gtimer() - t0 < (unsigned long long)spin_ns) { }
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += stride) dst[i] = src[i];
    if (acc == 0x9e3779b9u) sink[0] = (float)acc;
    __threadfence();
    if (!trigger_early) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

__global__ void consumer_kernel(const uint4* __restrict__ src, uint4* __restrict__ out, long long n16, int wait) {
    if (wait) asm volatile("griddepcontrol.wait;" ::: "memory");
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += stride) {
        uint4 v;
        asm volatile("ld.global.cv.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(src + i));
        out[i] = v;
    }
}

static float* sink_ptr() {
    static float* p = nullptr;
    if (!p) cudaMalloc(&p, 16);
    return p;
}

void producer(torch::Tensor src, torch::Tensor dst, int64_t spin_ns, int64_t grid, int64_t block,
              torch::Tensor stream_buf, bool trigger_early) {
    TORCH_CHECK(src.nbytes() % 16 == 0 && src.nbytes() == dst.nbytes());
    long long n16 = src.nbytes() / 16;
    long long sn16 = stream_buf.numel() ? (long long)(stream_buf.nbytes() / 16) : 0;
    auto stream = at::cuda::getCurrentCUDAStream();
    producer_kernel<<<(unsigned)grid, (unsigned)block, 0, stream>>>(
        (const uint4*)src.data_ptr(), (uint4*)dst.data_ptr(), n16, spin_ns,
        sn16 ? (const uint4*)stream_buf.data_ptr() : nullptr, sn16, sink_ptr(), trigger_early ? 1 : 0);
}

void consumer_copy(torch::Tensor src, torch::Tensor out, bool wait) {
    TORCH_CHECK(src.nbytes() % 16 == 0 && src.nbytes() == out.nbytes());
    long long n16 = src.nbytes() / 16;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(8);
    cfg.blockDim = dim3(128);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, consumer_kernel, (const uint4*)src.data_ptr(), (uint4*)out.data_ptr(), n16,
                       wait ? 1 : 0);
}
"""

_MOD = None


def ext():
    global _MOD
    if _MOD is None:
        from torch.utils.cpp_extension import load_inline
        build = os.environ.get("ROUTER_PREWAIT_EXT_DIR", "/tmp/router_prewait_ext")
        os.makedirs(build, exist_ok=True)
        _MOD = load_inline(name="router_prewait_pdl_ext", cpp_sources=_CPP, cuda_sources=_CU,
                           functions=["producer", "consumer_copy"], build_directory=build,
                           extra_cuda_cflags=["-O3", "-gencode=arch=compute_120,code=sm_120"], verbose=False)
    return _MOD


_EMPTY = {}


def empty():
    if "e" not in _EMPTY:
        _EMPTY["e"] = torch.empty(0, dtype=torch.uint8, device="cuda")
    return _EMPTY["e"]
