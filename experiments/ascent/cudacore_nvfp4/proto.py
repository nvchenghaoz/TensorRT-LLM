# Branch 36 prototype: cudaCoreGemmFp4 (NVFP4 W4A4 decode GEMV) variants as a standalone
# load_inline extension, timed against main's real op (torch.ops.trtllm.cuda_core_nvfp4_gemm,
# act_sf_swizzled=True) in one process, cold L2 (>= 400 MB rotated), CUDA graphs, PDL on.
# Chain per call: fp4_quantize -> GEMV (what CudaCoreNVFP4Runner runs in serving).
# Modes: bench (default) | check | ncu <main|all>
import hashlib
import json
import os
import sys

import torch

import tensorrt_llm  # noqa: F401


def log(*a):
    print(*a, flush=True)


MODE = sys.argv[1] if len(sys.argv) > 1 else "bench"
dev = torch.device("cuda")
SHAPES = {"gate_up": (34816, 5120), "down": (5120, 17408)}
COPIES = {"gate_up": 4, "down": 8}
ROUNDS = 2
WARMUP, ITERS = 10, 50
FP8_MAX, E2M1_MAX = 448.0, 6.0

CU = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cub/cub.cuh>
#include <cstdint>

namespace {

__device__ __forceinline__ int swz(int row, int col, int ntk) {
    int const tile_offset = ((row / 128) * ntk + col / 4) * 512;
    return tile_offset + (row % 32) * 16 + ((row % 128) / 32) * 4 + col % 4;
}

__device__ __forceinline__ void cvt_fp4x32(int4 const& q, float* out) {
    uint32_t const* w = reinterpret_cast<uint32_t const*>(&q);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint32_t h[4];
        asm volatile(
            "{\n.reg .b8 b0, b1, b2, b3;\n"
            "mov.b32 {b0, b1, b2, b3}, %4;\n"
            "cvt.rn.f16x2.e2m1x2 %0, b0;\n"
            "cvt.rn.f16x2.e2m1x2 %1, b1;\n"
            "cvt.rn.f16x2.e2m1x2 %2, b2;\n"
            "cvt.rn.f16x2.e2m1x2 %3, b3;\n}\n"
            : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3]) : "r"(w[i]));
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 f = __half22float2(reinterpret_cast<__half2 const&>(h[j]));
            out[i * 8 + j * 2] = f.x;
            out[i * 8 + j * 2 + 1] = f.y;
        }
    }
}

__device__ __forceinline__ float e4m3f(uint32_t b) {
    __nv_fp8_e4m3 x;
    x.__x = static_cast<__nv_fp8_storage_t>(b & 0xff);
    return static_cast<float>(x);
}

// LD: 0 plain, 1 nc.L1::no_allocate.L2::256B, 2 nc.L2::256B, 3 nc.L1::no_allocate, 4 nc.L1::no_allocate.L2::128B
template <int LD>
__device__ __forceinline__ int4 ldw(uint8_t const* p) {
    int4 r;
    if constexpr (LD == 1) {
        asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.s32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    } else if constexpr (LD == 2) {
        asm volatile("ld.global.nc.L2::256B.v4.s32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    } else if constexpr (LD == 3) {
        asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    } else if constexpr (LD == 4) {
        asm volatile("ld.global.nc.L1::no_allocate.L2::128B.v4.s32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    } else {
        r = *reinterpret_cast<int4 const*>(p);
    }
    return r;
}

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

template <int TILE_M, int TILE_N, int BLOCK>
__device__ __forceinline__ void reduce_store(float* acc, __nv_bfloat16* __restrict__ out, int n) {
    typedef cub::WarpReduce<float> WarpReduce;
    constexpr int warp_num = BLOCK / 32;
    int const tid = threadIdx.x;
    int warp_id = tid / 32, lane_id = tid % 32;
    __shared__ float shmem[TILE_M * TILE_N * warp_num];
    __shared__ typename WarpReduce::TempStorage temp_storage[warp_num];
#pragma unroll
    for (int mi = 0; mi < TILE_M; ++mi) {
#pragma unroll
        for (int ni = 0; ni < TILE_N; ++ni) {
            float val = WarpReduce(temp_storage[warp_id]).Sum(acc[mi * TILE_N + ni]);
            if (lane_id == 0) shmem[mi * TILE_N + ni + warp_id * TILE_M * TILE_N] = val;
        }
    }
    __syncthreads();
    for (int ii = tid; ii < TILE_M * TILE_N; ii += BLOCK) {
        int mid = ii / TILE_N, nid = ii % TILE_N;
        float val = 0;
#pragma unroll
        for (int jj = 0; jj < warp_num; ++jj) val += shmem[jj * TILE_M * TILE_N + ii];
        out[mid * n + nid] = __float2bfloat16(val);
    }
}

// Copy of main's cudaCoreGemmImpl (TILE_N=2, BLOCK=128, act SF swizzled).
template <int TILE_M, int TILE_N, int BLOCK, int LD, bool PRE_FORCE>
__global__ void __launch_bounds__(BLOCK) gemv_main_copy(uint8_t const* __restrict__ act,
    uint8_t const* __restrict__ weight, uint8_t const* __restrict__ sfa, uint8_t const* __restrict__ sfw,
    float const* alpha_ptr, __nv_bfloat16* __restrict__ out, int m, int n, int k) {
    float const alpha = alpha_ptr[0];
    constexpr bool PRE = TILE_M > 1 || PRE_FORCE;
    constexpr int tile_k = 32 * BLOCK;
    int const tile_m = blockIdx.y * TILE_M, tile_n = blockIdx.x * TILE_N, tid = threadIdx.x;
    float acc[TILE_M * TILE_N];
#pragma unroll
    for (int i = 0; i < TILE_M * TILE_N; ++i) acc[i] = 0;
    act += tile_m * k / 2;
    weight += tile_n * k / 2;
    out += tile_m * n + tile_n;
    int const ntk = (k / 16 + 3) / 4;
    int4 wq[TILE_N];
    uint32_t wsf[TILE_N];
    float tw[TILE_N * 32], tws[TILE_N * 2], ta[32], tas[2];
    auto load_w = [&](int idx_k) {
#pragma unroll
        for (int j = 0; j < TILE_N; ++j) wq[j] = ldw<LD>(weight + (j * k + idx_k) / 2);
#pragma unroll
        for (int j = 0; j < TILE_N; ++j)
            wsf[j] = *reinterpret_cast<uint16_t const*>(sfw + swz(tile_n + j, idx_k / 16, ntk));
    };
    auto cvt_w = [&]() {
#pragma unroll
        for (int j = 0; j < TILE_N; ++j) cvt_fp4x32(wq[j], tw + j * 32);
#pragma unroll
        for (int j = 0; j < TILE_N; ++j) {
            tws[j * 2] = e4m3f(wsf[j]);
            tws[j * 2 + 1] = e4m3f(wsf[j] >> 8);
        }
    };
    auto fma_step = [&](int idx_k) {
#pragma unroll
        for (int i = 0; i < TILE_M; ++i) {
            int4 aq = *reinterpret_cast<int4 const*>(act + (i * k + idx_k) / 2);
            cvt_fp4x32(aq, ta);
            uint32_t s = *reinterpret_cast<uint16_t const*>(sfa + swz(tile_m + i, idx_k / 16, ntk));
            tas[0] = e4m3f(s);
            tas[1] = e4m3f(s >> 8);
#pragma unroll
            for (int j = 0; j < TILE_N; ++j)
#pragma unroll
                for (int l = 0; l < 32; ++l)
                    acc[i * TILE_N + j] = fmaf(alpha * ta[l] * tas[l / 16], tw[j * 32 + l] * tws[j * 2 + l / 16],
                        acc[i * TILE_N + j]);
        }
    };
    int const first = tid * 32;
    if constexpr (!PRE) {
        pdl_wait();
        for (int idx_k = first; idx_k < k; idx_k += tile_k) {
            load_w(idx_k);
            cvt_w();
            fma_step(idx_k);
        }
    } else {
        if (first < k) load_w(first);
        pdl_wait();
        if (first < k) {
            cvt_w();
            fma_step(first);
        }
        for (int idx_k = first + tile_k; idx_k < k; idx_k += tile_k) {
            load_w(idx_k);
            cvt_w();
            fma_step(idx_k);
        }
    }
    reduce_store<TILE_M, TILE_N, BLOCK>(acc, out, n);
    pdl_trigger();
}

// Variant: every K step's weight (+SF) loads issued up front (NS = max steps per thread),
// optionally before the PDL wait; activations prefetched for all steps when registers allow.
// Thread -> K mapping (idx_k = (tid + s * BLOCK) * 32) and FMA order are main's when BLOCK == 128.
template <int TILE_M, int TILE_N, int BLOCK, int NS, bool PRE, bool APF_REQ, int STRM>
__global__ void __launch_bounds__(BLOCK) gemv_var(uint8_t const* __restrict__ act,
    uint8_t const* __restrict__ weight, uint8_t const* __restrict__ sfa, uint8_t const* __restrict__ sfw,
    float const* alpha_ptr, __nv_bfloat16* __restrict__ out, int m, int n, int k) {
    constexpr bool APF = APF_REQ && (TILE_M * NS <= 10);
    float const alpha = alpha_ptr[0];
    int const tile_m = blockIdx.y * TILE_M, tile_n = blockIdx.x * TILE_N, tid = threadIdx.x;
    act += tile_m * k / 2;
    weight += tile_n * k / 2;
    out += tile_m * n + tile_n;
    int const ntk = (k / 16 + 3) / 4;
    int4 wq[NS][TILE_N];
    uint32_t wsf[NS][TILE_N];
    auto load_w = [&]() {
#pragma unroll
        for (int s = 0; s < NS; ++s) {
            int const idx_k = (tid + s * BLOCK) * 32;
            if (idx_k < k) {
#pragma unroll
                for (int j = 0; j < TILE_N; ++j) wq[s][j] = ldw<STRM>(weight + (j * k + idx_k) / 2);
#pragma unroll
                for (int j = 0; j < TILE_N; ++j)
                    wsf[s][j] = *reinterpret_cast<uint16_t const*>(sfw + swz(tile_n + j, idx_k / 16, ntk));
            }
        }
    };
    if constexpr (PRE) load_w();
    pdl_wait();
    if constexpr (!PRE) load_w();
    int4 aq[APF ? NS : 1][TILE_M];
    uint32_t asf[APF ? NS : 1][TILE_M];
    if constexpr (APF) {
#pragma unroll
        for (int s = 0; s < NS; ++s) {
            int const idx_k = (tid + s * BLOCK) * 32;
            if (idx_k < k) {
#pragma unroll
                for (int i = 0; i < TILE_M; ++i) {
                    aq[s][i] = *reinterpret_cast<int4 const*>(act + (i * k + idx_k) / 2);
                    asf[s][i] = *reinterpret_cast<uint16_t const*>(sfa + swz(tile_m + i, idx_k / 16, ntk));
                }
            }
        }
    }
    float acc[TILE_M * TILE_N];
#pragma unroll
    for (int i = 0; i < TILE_M * TILE_N; ++i) acc[i] = 0;
#pragma unroll
    for (int s = 0; s < NS; ++s) {
        int const idx_k = (tid + s * BLOCK) * 32;
        if (idx_k < k) {
            float tw[TILE_N * 32], tws[TILE_N * 2];
#pragma unroll
            for (int j = 0; j < TILE_N; ++j) {
                cvt_fp4x32(wq[s][j], tw + j * 32);
                tws[j * 2] = e4m3f(wsf[s][j]);
                tws[j * 2 + 1] = e4m3f(wsf[s][j] >> 8);
            }
#pragma unroll
            for (int i = 0; i < TILE_M; ++i) {
                int4 a;
                uint32_t sa;
                if constexpr (APF) {
                    a = aq[s][i];
                    sa = asf[s][i];
                } else {
                    a = *reinterpret_cast<int4 const*>(act + (i * k + idx_k) / 2);
                    sa = *reinterpret_cast<uint16_t const*>(sfa + swz(tile_m + i, idx_k / 16, ntk));
                }
                float ta[32], tas[2];
                cvt_fp4x32(a, ta);
                tas[0] = e4m3f(sa);
                tas[1] = e4m3f(sa >> 8);
#pragma unroll
                for (int j = 0; j < TILE_N; ++j)
#pragma unroll
                    for (int l = 0; l < 32; ++l)
                        acc[i * TILE_N + j] = fmaf(alpha * ta[l] * tas[l / 16],
                            tw[j * 32 + l] * tws[j * 2 + l / 16], acc[i * TILE_N + j]);
            }
        }
    }
    reduce_store<TILE_M, TILE_N, BLOCK>(acc, out, n);
    pdl_trigger();
}

struct P {
    uint8_t const* a; uint8_t const* w; uint8_t const* sa; uint8_t const* sw;
    float const* alpha; __nv_bfloat16* o; int m, n, k;
};

template <typename Kern>
void pdl_launch(Kern kern, dim3 grid, dim3 block, P const& p) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = 0;
    cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, p.a, p.w, p.sa, p.sw, p.alpha, p.o, p.m, p.n, p.k));
}

template <int TM, int TN, int LD, bool PF>
void lmain(P const& p) {
    pdl_launch(gemv_main_copy<TM, TN, 128, LD, PF>, dim3(p.n / TN, p.m / TM), dim3(128), p);
}

template <int TN, int LD, bool PF>
bool bym_main(P const& p) {
    switch (p.m) {
    case 1: lmain<1, TN, LD, PF>(p); return true;
    case 2: lmain<2, TN, LD, PF>(p); return true;
    case 3: lmain<3, TN, LD, PF>(p); return true;
    case 4: lmain<4, TN, LD, PF>(p); return true;
    }
    return false;
}

template <int TM, int TN, int B, int NS, bool PRE, bool APF, int STRM>
void lv(P const& p) {
    pdl_launch(gemv_var<TM, TN, B, NS, PRE, APF, STRM>, dim3(p.n / TN, p.m / TM), dim3(B), p);
}

template <int TN, int B, int NS, bool PRE, bool APF, int STRM>
bool bym(P const& p) {
    switch (p.m) {
    case 1: lv<1, TN, B, NS, PRE, APF, STRM>(p); return true;
    case 2: lv<2, TN, B, NS, PRE, APF, STRM>(p); return true;
    case 3: lv<3, TN, B, NS, PRE, APF, STRM>(p); return true;
    case 4: lv<4, TN, B, NS, PRE, APF, STRM>(p); return true;
    }
    return false;
}

bool dispatch(int vid, P const& p) {
    bool const gu = p.k == 5120;
    bool const dn = p.k == 17408;
    if (!gu && !dn) return false;
    switch (vid) {
    case 0: return bym_main<2, 0, false>(p);
    // main's loop structure, weight-load hint only
    case 10: return bym_main<2, 1, false>(p);
    case 11: return bym_main<2, 2, false>(p);
    case 12: return bym_main<2, 3, false>(p);
    case 13: return bym_main<2, 4, false>(p);
    case 14: return bym_main<2, 1, true>(p);
    case 15: return bym_main<4, 1, false>(p);
    // all-steps prefetch + hint (exact)
    case 4: return gu ? bym<2, 128, 2, true, true, 1>(p) : bym<2, 128, 5, true, true, 1>(p);
    case 5: return gu ? bym<4, 128, 2, true, true, 1>(p) : bym<4, 128, 5, true, true, 1>(p);
    }
    return false;
}

}  // namespace

torch::Tensor run(int64_t vid, torch::Tensor aq, torch::Tensor wq, torch::Tensor asf, torch::Tensor wsf,
    torch::Tensor alpha) {
    int m = aq.size(0), n = wq.size(0), k = aq.size(1) * 2;
    auto out = torch::empty({m, n}, aq.options().dtype(torch::kBFloat16));
    P p{aq.data_ptr<uint8_t>(), wq.data_ptr<uint8_t>(), asf.data_ptr<uint8_t>(), wsf.data_ptr<uint8_t>(),
        alpha.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), m, n, k};
    TORCH_CHECK(dispatch(static_cast<int>(vid), p), "variant not dispatched");
    return out;
}
"""

CPP = ("torch::Tensor run(int64_t vid, torch::Tensor aq, torch::Tensor wq, torch::Tensor asf, "
       "torch::Tensor wsf, torch::Tensor alpha);")

VIDS = {
    "copy": 0,
    "C_ld1": 10, "C_ld2": 11, "C_ld3": 12, "C_ld4": 13, "C_ld1_pre": 14, "C_ld1_tn4": 15,
    "E_tn2_pre_strm": 4, "E_tn4_pre_strm": 5,
}
EXACT = {k for k in VIDS if k.startswith("E_") or k == "copy"}


def build_ext():
    from torch.utils.cpp_extension import load_inline
    bd = "/tmp/b36_nvfp4_proto_build"
    os.makedirs(bd, exist_ok=True)
    log("[proto] compiling extension ...")
    ext = load_inline(name="b36_nvfp4_proto", cpp_sources=CPP, cuda_sources=CU, functions=["run"],
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a", "-std=c++20",
                                         "-lineinfo", "-Xptxas=-v"],
                      build_directory=bd, verbose=os.environ.get("PROTO_VERBOSE", "0") == "1")
    log("[proto] extension ready")
    return ext


def sha(t):
    t = t.contiguous()
    return hashlib.sha256(t.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:20]


def make_weight(n, k, seed):
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    w = torch.randn(n, k, device=dev, generator=g).to(torch.bfloat16)
    w.view(n, k // 16, 16)[::5, ::11, :] = 0  # zero-SF blocks
    w.view(n, k // 16, 16)[1::97, 2::13, :] *= 40.0  # large values
    w.view(n, k // 16, 16)[2::7, 5::9, 3] = -0.0
    gs = (FP8_MAX * E2M1_MAX / w.float().abs().amax()).reshape(1).float()
    wq, wsf = torch.ops.trtllm.fp4_quantize(w, gs, 16, False, True)
    del w
    return wq, wsf, gs


def make_act(m, k, seed):
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    x = (torch.randn(m, k, device=dev, generator=g) * 2.0).to(torch.bfloat16)
    x.view(m, k // 16, 16)[:, 3::7, :] = 0  # zero-SF blocks
    x.view(m, k // 16, 16)[:, 4::31, :] *= 50.0
    gs = (FP8_MAX * E2M1_MAX / x.float().abs().amax()).reshape(1).float()
    return x, gs


def quant(x, gsa):
    return torch.ops.trtllm.fp4_quantize(x, gsa, 16, False, True)


def main_gemm(aq, wq, asf, wsf, alpha):
    return torch.ops.trtllm.cuda_core_nvfp4_gemm(aq, wq, asf, wsf, alpha, None, torch.bfloat16,
                                                 act_sf_swizzled=True)


def build():
    data = {}
    for name, (n, k) in SHAPES.items():
        wq, wsf, gsw = make_weight(n, k, seed=17 if name == "gate_up" else 29)
        copies = [(wq, wsf)] + [(wq.clone(), wsf.clone()) for _ in range(COPIES[name] - 1)]
        wbytes = wq.numel() + wsf.numel()
        log(f"[proto] {name} N={n} K={k}: {COPIES[name]} copies x {wbytes / 1e6:.1f} MB")
        data[name] = dict(n=n, k=k, copies=copies, gsw=gsw, wbytes=wbytes)
    return data


def calls_for(d, m):
    out = []
    for i in range(len(d["copies"]) * ROUNDS):
        x, gsa = make_act(m, d["k"], seed=1000 * m + i)
        alpha = (1.0 / (gsa * d["gsw"])).float()
        wq, wsf = d["copies"][i % len(d["copies"])]
        out.append((x, gsa, wq, wsf, alpha))
    return out


def make_fn(ext, vname):
    if vname == "main":
        def f(x, gsa, wq, wsf, alpha):
            aq, asf = quant(x, gsa)
            return main_gemm(aq, wq, asf, wsf, alpha)
    elif vname == "quant":
        def f(x, gsa, wq, wsf, alpha):
            return quant(x, gsa)[0]
    else:
        vid = VIDS[vname]

        def f(x, gsa, wq, wsf, alpha):
            aq, asf = quant(x, gsa)
            return ext.run(vid, aq, wq, asf, wsf, alpha)
    return f


def capture(fn, calls):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for c in calls[:2]:
            fn(*c)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = [fn(*c) for c in calls]
    torch.cuda.synchronize()
    return g, outs


def stats(ts):
    s = sorted(ts)
    q = lambda p: s[min(len(s) - 1, int(round(p * (len(s) - 1))))]  # noqa: E731
    return q(0.5), q(0.9), q(0.1)


def dequant(q, sf, gs, rows, k):
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
                       device=dev, dtype=torch.float32)
    lo = lut[(q & 0xF).long()]
    hi = lut[(q >> 4).long()]
    v = torch.stack([lo, hi], dim=-1).reshape(rows, k)
    ncol = k // 16
    ntk = (ncol + 3) // 4
    r = torch.arange(rows, device=dev).view(-1, 1)
    c = torch.arange(ncol, device=dev).view(1, -1)
    idx = ((r // 128) * ntk + c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4
    s = sf.view(-1)[idx].view(torch.float8_e4m3fn).float()
    return v * s.repeat_interleave(16, dim=1) / gs


def check_variants(ext, data, variants, ms=(1, 2, 3, 4)):
    res = {}
    for name, d in data.items():
        wq, wsf = d["copies"][0]
        wdq = dequant(wq, wsf, d["gsw"], d["n"], d["k"])
        for m in ms:
            x, gsa = make_act(m, d["k"], seed=77 + m)
            alpha = (1.0 / (gsa * d["gsw"])).float()
            aq, asf = quant(x, gsa)
            ref = dequant(aq, asf, gsa, m, d["k"]) @ wdq.t()
            mo = main_gemm(aq, wq, asf, wsf, alpha)
            me = (mo.float() - ref).abs()
            rec = {"main_max": me.max().item(), "main_mean": me.mean().item()}
            for vn in variants:
                try:
                    o = ext.run(VIDS[vn], aq, wq, asf, wsf, alpha)
                except Exception as e:  # not dispatched for this shape
                    rec[vn] = "n/a"
                    continue
                e = (o.float() - ref).abs()
                rec[vn] = {"bitwise": bool(torch.equal(o, mo)),
                           "max": round(e.max().item(), 5), "mean": round(e.mean().item(), 6),
                           "finite": bool(torch.isfinite(o.float()).all())}
            torch.cuda.synchronize()
            res[f"{name}/M{m}"] = rec
            log(f"[proto] check {name} M={m}: " + json.dumps(rec))
        del wdq
    return res


def bench(variants):
    ext = build_ext()
    data = build()
    chk = check_variants(ext, data, [v for v in variants if v not in ("main", "quant")])
    out = {}
    for name, d in data.items():
        for m in (1, 2, 3, 4):
            calls = calls_for(d, m)
            ncall = len(calls)
            graphs = {}
            for vn in ["quant", "main"] + variants:
                try:
                    graphs[vn], _ = capture(make_fn(ext, vn), calls)
                except Exception as e:
                    log(f"[proto] skip {vn} {name} M={m}: {repr(e)[:80]}")
                    torch.cuda.synchronize()
            ts = {vn: [] for vn in graphs}
            for _ in range(WARMUP):
                for g in graphs.values():
                    g.replay()
            torch.cuda.synchronize()
            for _ in range(ITERS):
                for vn, g in graphs.items():
                    e0 = torch.cuda.Event(enable_timing=True)
                    e1 = torch.cuda.Event(enable_timing=True)
                    e0.record()
                    g.replay()
                    e1.record()
                    e1.synchronize()
                    ts[vn].append(e0.elapsed_time(e1) / ncall * 1e3)
            rec = {}
            base = stats(ts["main"])[0]
            for vn, t in ts.items():
                med, p90, p10 = stats(t)
                rec[vn] = {"us": round(med, 2), "p90": round(p90, 2), "nf": round((p90 - p10) / med, 4),
                           "vs_main": round(med / base, 4)}
                if vn != "quant":
                    rec[vn]["TBps"] = round(d["wbytes"] / (med * 1e-6) / 1e12, 3)
            out[f"{name}/M{m}"] = rec
            log(f"[proto] {name} M={m}: " + json.dumps(rec))
            del graphs
            torch.cuda.synchronize()
    log(json.dumps({"timing": out, "check": chk}))


def ncu_mode(which):
    data = {}
    for name, (n, k) in SHAPES.items():
        wq, wsf, gsw = make_weight(n, k, seed=17)
        data[name] = (wq, wsf, gsw, k)
    ext = build_ext() if which != "main" else None
    for name, (wq, wsf, gsw, k) in data.items():
        x, gsa = make_act(1, k, seed=5)
        alpha = (1.0 / (gsa * gsw)).float()
        aq, asf = quant(x, gsa)
        for _ in range(3):
            main_gemm(aq, wq, asf, wsf, alpha)
        if ext is not None:
            for vn in which.split(","):
                for _ in range(3):
                    ext.run(VIDS[vn], aq, wq, asf, wsf, alpha)
        torch.cuda.synchronize()
        log(f"[proto] ncu {name} done")


if MODE == "ncu":
    ncu_mode(sys.argv[2] if len(sys.argv) > 2 else "main")
elif MODE == "check":
    ext = build_ext()
    data = build()
    log(json.dumps(check_variants(ext, data, list(VIDS), ms=(1, 2, 3, 4))))
else:
    vs = sys.argv[2].split(",") if len(sys.argv) > 2 else [v for v in VIDS]
    bench(vs)
