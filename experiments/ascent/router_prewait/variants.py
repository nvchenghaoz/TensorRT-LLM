"""Standalone variants of router_gemm_mma_kernel<M,384,5120,8,640> (copied math, M<=8) to choose the pre-wait scheme.

V0  original (BASE): wait, then per chunk plain C++ loads a0[c], b[c] (compiler free to schedule).
V1  current CAND: B in registers before the wait (plain C++), A plain C++ loads after.
V2  B before the wait via asm volatile ld.global.v4; A after the wait via asm volatile (all 8 issued before any MMA).
V3  wait, then all loads asm volatile (A and B, all issued before any MMA). No pre-wait work.
V4  prefetch.global.L2 of the CTA's weight slice before the wait (one 128 B line per lane), then V3.
V5  V2 but A loads issued first after the wait (same as V2 except B loads issued with .L2::128B hint) -- skipped.

All variants: same MMA order and warp-order shared-memory reduction as dsv3RouterGemm.cu, so outputs must be bitwise
equal to torch.ops.trtllm.dsv3_router_gemm_op. bench prints per-layer router cost in P0 (router -> gate_forward chain)
and P1 (mhc-like producer -> router -> gate_forward), median of per-iteration (chain - chain without router)/40,
40 distinct weights, read-only 512 MB flush before each replay, all variants interleaved in every iteration.
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pdl_ext  # noqa: E402

_CPP = r"""
#include <torch/extension.h>
void router_variant(int64_t variant, torch::Tensor x, torch::Tensor w, torch::Tensor out);
"""

_CU = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t const (&a)[4], uint32_t const (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint4 ldv(const void* p) {
    uint4 v;
    asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

extern __shared__ uint4 dyn_smem[];

template <int M, int V>
__global__ __launch_bounds__(640, 1) void rk(float* out, __nv_bfloat16 const* A, __nv_bfloat16 const* B) {
    constexpr int H = 5120, E = 384, NW = 20, CH = 32, CPW = 8;
    int const warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, q = lane % 4;
    int const n_base = blockIdx.x * 8;
    int const k_warp = warp * (CPW * CH);
    __shared__ float red[NW][1][32][4];
    float acc[1][4] = {};
    uint4 b[CPW];
    uint4 a0[CPW];
    __nv_bfloat16 const* bp = B + static_cast<int64_t>(n_base + g) * H + k_warp + q * 8;
    __nv_bfloat16 const* ap = A + g * H + k_warp + q * 8;
    bool const row0_valid = g < M;
    uint4 const zero4 = make_uint4(0u, 0u, 0u, 0u);
    if constexpr (V == 1) {
#pragma unroll
        for (int c = 0; c < CPW; c++) b[c] = *reinterpret_cast<uint4 const*>(bp + c * CH);
    } else if constexpr (V == 2) {
#pragma unroll
        for (int c = 0; c < CPW; c++) b[c] = ldv(bp + c * CH);
    } else if constexpr (V == 4 || V == 5) {
        // one 128 B line per lane: row lane/4 of the CTA's 8 rows, line lane%4 of the warp's 512 B K-slice
        __nv_bfloat16 const* pp = B + static_cast<int64_t>(n_base + lane / 4) * H + k_warp + (lane % 4) * 64;
        asm volatile("prefetch.global.L2 [%0];" ::"l"(pp));
    } else if constexpr (V == 6) {
#pragma unroll
        for (int c = 0; c < CPW; c++) {
            uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(&dyn_smem[c * 640 + threadIdx.x]));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(bp + c * CH));
        }
        asm volatile("cp.async.commit_group;");
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if constexpr (V == 0 || V == 5) {
#pragma unroll
        for (int c = 0; c < CPW; c++) {
            a0[c] = row0_valid ? *reinterpret_cast<uint4 const*>(ap + c * CH) : zero4;
            b[c] = *reinterpret_cast<uint4 const*>(bp + c * CH);
        }
    } else if constexpr (V == 1) {
#pragma unroll
        for (int c = 0; c < CPW; c++) a0[c] = row0_valid ? *reinterpret_cast<uint4 const*>(ap + c * CH) : zero4;
    } else if constexpr (V == 6) {
#pragma unroll
        for (int c = 0; c < CPW; c++) a0[c] = row0_valid ? *reinterpret_cast<uint4 const*>(ap + c * CH) : zero4;
        asm volatile("cp.async.wait_all;" ::: "memory");
#pragma unroll
        for (int c = 0; c < CPW; c++) b[c] = dyn_smem[c * 640 + threadIdx.x];
    } else if constexpr (V == 2) {
#pragma unroll
        for (int c = 0; c < CPW; c++) a0[c] = zero4;
        if (row0_valid) {
#pragma unroll
            for (int c = 0; c < CPW; c++) a0[c] = ldv(ap + c * CH);
        }
    } else {
#pragma unroll
        for (int c = 0; c < CPW; c++) a0[c] = zero4;
        if (row0_valid) {
#pragma unroll
            for (int c = 0; c < CPW; c++) {
                a0[c] = ldv(ap + c * CH);
                b[c] = ldv(bp + c * CH);
            }
        } else {
#pragma unroll
            for (int c = 0; c < CPW; c++) b[c] = ldv(bp + c * CH);
        }
    }
#pragma unroll
    for (int c = 0; c < CPW; c++) {
        uint32_t const af0[4] = {a0[c].x, 0u, a0[c].y, 0u};
        uint32_t const af1[4] = {a0[c].z, 0u, a0[c].w, 0u};
        uint32_t const bf0[2] = {b[c].x, b[c].y};
        uint32_t const bf1[2] = {b[c].z, b[c].w};
        mma16816(acc[0], af0, bf0);
        mma16816(acc[0], af1, bf1);
    }
#pragma unroll
    for (int i = 0; i < 4; i++) red[warp][0][lane][i] = acc[0][i];
    __syncthreads();
    if (threadIdx.x < 128) {
        int const l = threadIdx.x / 4, i = threadIdx.x % 4;
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < NW; w++) sum += red[w][0][l][i];
        int const row = (l / 4) + (i >= 2 ? 8 : 0);
        int const col = n_base + (l % 4) * 2 + (i & 1);
        if (row < M) out[row * E + col] = sum;
    }
    asm volatile("griddepcontrol.launch_dependents;");
}

// V9: warp-specialised. 4 extra loader warps cp.async the CTA's whole weight slice (80 KB) into shared memory before
// (and without) any griddepcontrol.wait; the 20 compute warps execute the wait with no memory op in flight, load A,
// then meet the loaders at named barrier 1 and read B from shared memory. Same MMA order / reduction as V0.
template <int M>
__global__ __launch_bounds__(768, 1) void rk9(float* out, __nv_bfloat16 const* A, __nv_bfloat16 const* B) {
    constexpr int H = 5120, E = 384, NW = 20, CH = 32, CPW = 8;
    __shared__ float red[NW][32][4];
    int const warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int const n_base = blockIdx.x * 8;
    if (warp >= NW) {
        int const lt = threadIdx.x - NW * 32;
#pragma unroll 4
        for (int i = lt; i < NW * 32 * CPW; i += 128) {
            int const c = i / (NW * 32), t = i % (NW * 32);
            int const w = t / 32, l = t % 32, g = l / 4, q = l % 4;
            __nv_bfloat16 const* src = B + static_cast<int64_t>(n_base + g) * H + w * (CPW * CH) + q * 8 + c * CH;
            uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(&dyn_smem[i]));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(src));
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_all;" ::: "memory");
        asm volatile("bar.arrive 1, 768;" ::: "memory");
    } else {
        int const g = lane / 4, q = lane % 4;
        int const k_warp = warp * (CPW * CH);
        __nv_bfloat16 const* ap = A + g * H + k_warp + q * 8;
        bool const row0_valid = g < M;
        uint4 const zero4 = make_uint4(0u, 0u, 0u, 0u);
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        uint4 a0[CPW];
        asm volatile("griddepcontrol.wait;" ::: "memory");
#pragma unroll
        for (int c = 0; c < CPW; c++) a0[c] = row0_valid ? *reinterpret_cast<uint4 const*>(ap + c * CH) : zero4;
        asm volatile("bar.sync 1, 768;" ::: "memory");
#pragma unroll
        for (int c = 0; c < CPW; c++) {
            uint4 const bv = dyn_smem[c * (NW * 32) + threadIdx.x];
            uint32_t const af0[4] = {a0[c].x, 0u, a0[c].y, 0u};
            uint32_t const af1[4] = {a0[c].z, 0u, a0[c].w, 0u};
            uint32_t const bf0[2] = {bv.x, bv.y};
            uint32_t const bf1[2] = {bv.z, bv.w};
            float (&accr)[4] = acc;
            mma16816(accr, af0, bf0);
            mma16816(accr, af1, bf1);
        }
#pragma unroll
        for (int i = 0; i < 4; i++) red[warp][lane][i] = acc[i];
    }
    __syncthreads();
    if (threadIdx.x < 128) {
        int const l = threadIdx.x / 4, i = threadIdx.x % 4;
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < NW; w++) sum += red[w][l][i];
        int const row = (l / 4) + (i >= 2 ? 8 : 0);
        int const col = n_base + (l % 4) * 2 + (i & 1);
        if (row < M) out[row * E + col] = sum;
    }
    asm volatile("griddepcontrol.launch_dependents;");
}

template <int M>
static void launch9(torch::Tensor x, torch::Tensor w, torch::Tensor out) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(48);
    cfg.blockDim = dim3(768);
    cfg.stream = at::cuda::getCurrentCUDAStream();
    static bool once = false;
    if (!once) {
        cudaFuncSetAttribute(rk9<M>, cudaFuncAttributeMaxDynamicSharedMemorySize, 8 * 640 * 16);
        once = true;
    }
    cfg.dynamicSmemBytes = 8 * 640 * 16;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, rk9<M>, (float*)out.data_ptr(), (__nv_bfloat16 const*)x.data_ptr(),
                       (__nv_bfloat16 const*)w.data_ptr());
}

template <int M, int V>
static void launch(torch::Tensor x, torch::Tensor w, torch::Tensor out) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(48);
    cfg.blockDim = dim3(640);
    cfg.stream = at::cuda::getCurrentCUDAStream();
    if constexpr (V == 6) {
        static bool once = false;
        if (!once) {
            cudaFuncSetAttribute(rk<M, V>, cudaFuncAttributeMaxDynamicSharedMemorySize, 8 * 640 * 16);
            once = true;
        }
        cfg.dynamicSmemBytes = 8 * 640 * 16;
    }
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, rk<M, V>, (float*)out.data_ptr(), (__nv_bfloat16 const*)x.data_ptr(),
                       (__nv_bfloat16 const*)w.data_ptr());
}

template <int M>
static void by_v(int64_t v, torch::Tensor x, torch::Tensor w, torch::Tensor out) {
    switch (v) {
        case 0: launch<M, 0>(x, w, out); break;
        case 1: launch<M, 1>(x, w, out); break;
        case 2: launch<M, 2>(x, w, out); break;
        case 3: launch<M, 3>(x, w, out); break;
        case 4: launch<M, 4>(x, w, out); break;
        case 5: launch<M, 5>(x, w, out); break;
        case 6: launch<M, 6>(x, w, out); break;
        case 9: launch9<M>(x, w, out); break;
        default: TORCH_CHECK(false, "variant");
    }
}

void router_variant(int64_t variant, torch::Tensor x, torch::Tensor w, torch::Tensor out) {
    int m = x.size(0);
    switch (m) {
        case 1: by_v<1>(variant, x, w, out); break;
        case 2: by_v<2>(variant, x, w, out); break;
        case 4: by_v<4>(variant, x, w, out); break;
        case 8: by_v<8>(variant, x, w, out); break;
        default: TORCH_CHECK(false, "M");
    }
}
"""

HIDDEN, EXPERTS, LAYERS, TOPK, SCALE = 5120, 384, 40, 6, 1.5
MS = (1, 2, 4, 8)
VARIANTS = tuple(int(v) for v in os.environ.get("VARIANTS", "0,1,9").split(","))
WARMUP, ITERS = 10, 50
SEED = 22301


def log(*a):
    print(*a, flush=True)


def ext():
    from torch.utils.cpp_extension import load_inline
    build = "/tmp/router_prewait_variants"
    os.makedirs(build, exist_ok=True)
    return load_inline(name="router_prewait_variants", cpp_sources=_CPP, cuda_sources=_CU,
                       functions=["router_variant"], build_directory=build,
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_120,code=sm_120", "-Xptxas", "-v"],
                       verbose=False)


def main():
    import tensorrt_llm  # noqa: F401
    from bench import capture, gate, server_prod, stats, timed
    mod = ext()
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    ws = [(torch.randn((EXPERTS, HIDDEN), generator=gen, device="cuda") * HIDDEN ** -0.5).to(torch.bfloat16)
          for _ in range(LAYERS)]
    bias = (torch.randn((EXPERTS,), generator=gen, device="cuda") * 0.05).float()
    flush = torch.ones(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    ok = True
    res = {}
    for m in MS:
        xs = [torch.randn((m, HIDDEN), generator=gen, device="cuda").to(torch.bfloat16) for _ in range(LAYERS)]
        refs = [torch.ops.trtllm.dsv3_router_gemm_op(x, w.t(), bias=None, out_dtype=torch.float32)
                for x, w in zip(xs, ws)]
        outs = {v: [torch.empty((m, EXPERTS), dtype=torch.float32, device="cuda") for _ in range(LAYERS)]
                for v in VARIANTS}
        for v in VARIANTS:
            for li in range(LAYERS):
                mod.router_variant(v, xs[li], ws[li], outs[v][li])
            torch.cuda.synchronize()
            eq = all(torch.equal(o, r) for o, r in zip(outs[v], refs))
            log(f"M={m} V{v} bitwise vs trtllm op: {eq}")
            ok = ok and eq
        prod = server_prod(m)
        dsts = [torch.empty_like(x) for x in xs]
        ows = [torch.empty((m, TOPK), dtype=torch.float32, device="cuda") for _ in range(LAYERS)]
        ois = [torch.empty((m, TOPK), dtype=torch.int32, device="cuda") for _ in range(LAYERS)]

        def mk_r(v):
            def f():
                for li in range(LAYERS):
                    mod.router_variant(v, xs[li], ws[li], outs[v][li])
                    gate(outs[v][li], bias, ows[li], ois[li])
            return f

        def mk_pr(v):
            def f():
                for li in range(LAYERS):
                    prod(li, xs[li], dsts[li])
                    mod.router_variant(v, dsts[li], ws[li], outs[v][li])
                    gate(outs[v][li], bias, ows[li], ois[li])
            return f

        def g_chain():
            for li in range(LAYERS):
                gate(refs[li], bias, ows[li], ois[li])

        def pg_chain():
            for li in range(LAYERS):
                prod(li, xs[li], dsts[li])
                gate(refs[li], bias, ows[li], ois[li])

        graphs = {"G": capture(g_chain)[0], "PG": capture(pg_chain)[0]}
        for v in VARIANTS:
            graphs[f"R{v}"] = capture(mk_r(v))[0]
            graphs[f"PR{v}"] = capture(mk_pr(v))[0]
        t = {k: [] for k in graphs}
        for it in range(WARMUP + ITERS):
            cur = {k: timed(g, flush) for k, g in graphs.items()}
            if it >= WARMUP:
                for k, v in cur.items():
                    t[k].append(v)
        line = {}
        for v in VARIANTS:
            p0 = [(a - b) / LAYERS for a, b in zip(t[f"R{v}"], t["G"])]
            p1 = [(a - b) / LAYERS for a, b in zip(t[f"PR{v}"], t["PG"])]
            m0, n0 = stats(p0)
            m1, n1 = stats(p1)
            line[f"V{v}"] = {"P0": round(m0, 3), "P0_noise": round(n0, 3), "P1": round(m1, 3),
                             "P1_noise": round(n1, 3)}
            log(f"M={m} V{v}: P0 {m0:.3f} us/layer (noise {n0:.3f}) | P1 {m1:.3f} us/layer (noise {n1:.3f})")
        res[m] = line
        del graphs
    print(json.dumps({"passed": ok, "variants": res}))


if __name__ == "__main__":
    with torch.no_grad():
        main()
