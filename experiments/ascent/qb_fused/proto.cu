// S1 prototype (branch 127, idea 232): SM120 decode attention-input chain.
//   A = wq_a||wkv low-M MXFP8 GEMM (copy of csa2_mxfp8_low_m_gemm_kernel, STAGES=1) with a runtime trigger position.
//   B = q_norm (RMSNorm, rmsnorm_cute order) + MXFP8 quant + wq_b(||index_wq_b) low-M GEMM in ONE launch.
// Built as a torch cpp_extension inside the GPU job; nothing here is part of the TRT-LLM build.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <optional>

namespace cg = cooperative_groups;

namespace
{

constexpr int kRowPad = 32;

__device__ __forceinline__ uint32_t mapa_shared(uint32_t addr, uint32_t target_rank)
{
    uint32_t r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(r) : "r"(addr), "r"(target_rank));
    return r;
}

__device__ __forceinline__ uint32_t smem_u32(void const* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ float rcp_approx_ftz(float a)
{
    float b;
    asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
    return b;
}

__device__ __forceinline__ float rsqrt_approx(float a)
{
    float b;
    asm volatile("rsqrt.approx.f32 %0, %1;\n" : "=f"(b) : "f"(a));
    return b;
}

__device__ __forceinline__ float rsqrt_approx_ftz(float a)
{
    float b;
    asm volatile("rsqrt.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
    return b;
}

__device__ __forceinline__ float pow2_from_ue8m0(uint32_t e)
{
    return __uint_as_float(e == 0 ? 0x00400000u : (e << 23));
}

__device__ __forceinline__ void mma_e4m3(
    float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f));
}

__device__ __forceinline__ void bulk_g2s(void* dst, void const* src, uint32_t bytes, uint64_t* bar)
{
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(
                     smem_u32(dst)),
                 "l"(src), "r"(bytes), "r"(smem_u32(bar))
                 : "memory");
}

__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity)
{
    asm volatile(
        "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra "
        "WAIT_%=;\n}\n" ::"r"(smem_u32(bar)),
        "r"(parity)
        : "memory");
}

struct NormArgs
{
    __nv_bfloat16 const* gamma; // [K] RMSNorm weight
    float eps;
    __nv_bfloat16* qr_out; // [M, K] normed rows (written by CTA 0), or null
    int ld_qr;
    int order; // bits 0-1 in-thread sum order, bit 2 mean by reciprocal multiply, bit 3 rsqrt ftz
};

struct Ctl
{
    int trig;  // 0 after mainloop (landed), 1 at start (before wait), 2 right after wait, 3 at the very end
    int flags; // bit 0: skip griddepcontrol.wait (negative control only)
};

// One warp normalizes one row of K = 256 * NB bf16 values exactly as FlashInfer rmsnorm_cute (H=1280: 32 threads per
// row, lane t holds columns 8t + v + 256b), then MXFP8-quantizes it exactly as trtllm::mxfp8_quantize(qr, True).
// The quantized columns [k_begin, k_begin + kc) go to xq_row / xc_row (relative to k_begin).
template <int NB>
__device__ __forceinline__ void norm_quant_row(__nv_bfloat16 const* xrow, uint4 const (&g)[NB], NormArgs const& na,
    int lane, __nv_bfloat16* qr_row, uint8_t* xq_row, uint8_t* xc_row, int k_begin, int kc)
{
    constexpr int K = 256 * NB;
    float v[NB * 8];
#pragma unroll
    for (int b = 0; b < NB; ++b)
    {
        uint4 const raw = __ldcg(reinterpret_cast<uint4 const*>(xrow + 8 * lane + 256 * b));
        __nv_bfloat162 const* h = reinterpret_cast<__nv_bfloat162 const*>(&raw);
#pragma unroll
        for (int t = 0; t < 4; ++t)
        {
            float2 const f = __bfloat1622float2(h[t]);
            v[b * 8 + 2 * t] = f.x;
            v[b * 8 + 2 * t + 1] = f.y;
        }
    }
    float s = 0.f;
    int const ord = na.order & 3;
    if (ord == 0)
    {
#pragma unroll
        for (int i = 0; i < NB * 8; ++i)
            s = __fmaf_rn(v[i], v[i], s);
    }
    else if (ord == 1)
    {
#pragma unroll
        for (int i = 0; i < NB * 8; ++i)
            s = __fadd_rn(s, __fmul_rn(v[i], v[i]));
    }
    else if (ord == 2)
    {
#pragma unroll
        for (int e = 0; e < 8; ++e)
#pragma unroll
            for (int b = 0; b < NB; ++b)
                s = __fmaf_rn(v[b * 8 + e], v[b * 8 + e], s);
    }
    else
    {
#pragma unroll
        for (int e = 0; e < 8; ++e)
#pragma unroll
            for (int b = 0; b < NB; ++b)
                s = __fadd_rn(s, __fmul_rn(v[b * 8 + e], v[b * 8 + e]));
    }
#pragma unroll
    for (int o = 1; o < 32; o <<= 1)
        s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, o));
    float const mean = (na.order & 4) ? __fmul_rn(s, 1.0f / static_cast<float>(K)) : __fdiv_rn(s, static_cast<float>(K));
    float const t = __fadd_rn(mean, na.eps);
    float const rstd = (na.order & 8) ? rsqrt_approx_ftz(t) : rsqrt_approx(t);
#pragma unroll
    for (int b = 0; b < NB; ++b)
    {
        __nv_bfloat162 const* gh = reinterpret_cast<__nv_bfloat162 const*>(&g[b]);
        float y[8];
        uint32_t yb[4];
#pragma unroll
        for (int e2 = 0; e2 < 4; ++e2)
        {
            float2 const gf = __bfloat1622float2(gh[e2]);
            __nv_bfloat16 const lo
                = __float2bfloat16_rn(__fmul_rn(__fmul_rn(v[b * 8 + 2 * e2], rstd), __fadd_rn(gf.x, 0.f)));
            __nv_bfloat16 const hi
                = __float2bfloat16_rn(__fmul_rn(__fmul_rn(v[b * 8 + 2 * e2 + 1], rstd), __fadd_rn(gf.y, 0.f)));
            y[2 * e2] = __bfloat162float(lo);
            y[2 * e2 + 1] = __bfloat162float(hi);
            yb[e2] = static_cast<uint32_t>(__bfloat16_as_ushort(lo)) | (static_cast<uint32_t>(__bfloat16_as_ushort(hi)) << 16);
        }
        int const col = 8 * lane + 256 * b;
        if (qr_row != nullptr)
        {
            *reinterpret_cast<uint4*>(qr_row + col) = make_uint4(yb[0], yb[1], yb[2], yb[3]);
        }
        float amax = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e)
            amax = fmaxf(amax, fabsf(y[e]));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
        float const sf = amax * rcp_approx_ftz(448.0f);
        __nv_fp8_e8m0 code;
        code.__x = __nv_cvt_float_to_e8m0(sf, __NV_SATFINITE, cudaRoundPosInf);
        float const out_scale
            = amax != 0.f ? (code.__x == 0 ? 1.0f : exp2f(127.0f - static_cast<float>(code.__x))) : 0.0f;
        uint32_t packed[2];
#pragma unroll
        for (int t2 = 0; t2 < 2; ++t2)
        {
            __nv_fp8x2_e4m3 const lo(make_float2(y[4 * t2] * out_scale, y[4 * t2 + 1] * out_scale));
            __nv_fp8x2_e4m3 const hi(make_float2(y[4 * t2 + 2] * out_scale, y[4 * t2 + 3] * out_scale));
            packed[t2] = static_cast<uint32_t>(lo.__x) | (static_cast<uint32_t>(hi.__x) << 16);
        }
        if (col >= k_begin && col < k_begin + kc)
        {
            *reinterpret_cast<uint2*>(xq_row + (col - k_begin)) = make_uint2(packed[0], packed[1]);
            if ((lane & 3) == 0)
            {
                xc_row[(col - k_begin) / 32] = code.__x;
            }
        }
    }
}

template <int WARPS, int CLUSTER_K, int RT, bool NORM>
__global__ void __launch_bounds__(WARPS * 32) qb_kernel(__nv_bfloat16 const* __restrict__ x, int ldx, int M, int N,
    int K, uint8_t const* __restrict__ w, uint8_t const* __restrict__ wsf, int num_k_tiles,
    __nv_bfloat16* __restrict__ out, int ldo, __nv_bfloat16* __restrict__ out2, int ldo2, int n_split,
    int red_aliased, NormArgs na, Ctl ctl)
{
    constexpr int kThreads = WARPS * 32;
    constexpr int kRows = 16 * RT;
    constexpr int kTile = kRows * 8;
    constexpr int NB = 5; // NORM: K == 1280
    extern __shared__ __align__(128) uint8_t smem[];
    __shared__ __align__(8) uint64_t bar;
    __shared__ __align__(8) uint64_t red_bar;

    int rank = 0;
    if constexpr (CLUSTER_K > 1)
    {
        rank = static_cast<int>(cg::this_cluster().block_rank());
    }
    int const tile = blockIdx.x / CLUSTER_K;
    int const row0 = tile * kRows;
    int const kc = K / CLUSTER_K;
    int const kc_blocks = kc / 32;
    int const k_begin = rank * kc;
    int const kb_begin = k_begin / 32;
    int const row_stride = kc + kRowPad;
    int const warp = threadIdx.x / 32;
    int const lane = threadIdx.x % 32;
    int const live = kRows * M;

    uint8_t* w_s = smem;
    uint8_t* sf_s = w_s + kRows * row_stride;
    uint8_t* xq_s = sf_s + RT * (kc_blocks / 4) * 256;
    uint8_t* xc_s = xq_s + M * kc;
    float* cluster_red = reinterpret_cast<float*>(xc_s + ((M * kc_blocks + 15) / 16) * 16);
    float* red = red_aliased ? reinterpret_cast<float*>(w_s) : cluster_red + (CLUSTER_K - 1) * live;

    uint32_t const stage_bytes = kRows * kc + RT * (kc_blocks / 4) * 256u;
    if (threadIdx.x == 0)
    {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(smem_u32(&bar)));
        if (CLUSTER_K > 1 && rank == 0)
        {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(smem_u32(&red_bar)));
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(&bar)),
                     "r"(stage_bytes)
                     : "memory");
        if (CLUSTER_K > 1 && rank == 0)
        {
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(&red_bar)),
                         "r"(static_cast<uint32_t>((CLUSTER_K - 1) * live * sizeof(float)))
                         : "memory");
        }
    }
    __syncthreads();
    if constexpr (CLUSTER_K > 1)
    {
        asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");
    }
    if (warp == 0)
    {
        int const chunks = kc_blocks / 4;
        for (int r = lane; r < kRows; r += 32)
        {
            bulk_g2s(w_s + r * row_stride, w + static_cast<int64_t>(row0 + r) * K + k_begin, kc, &bar);
        }
        for (int t = lane; t < RT * chunks; t += 32)
        {
            int const rt = t / chunks;
            int const c = t % chunks;
            int const rr = row0 + 16 * rt;
            int64_t const sf_base = static_cast<int64_t>(rr / 128) * num_k_tiles * 512 + (rr % 32) * 16;
            bulk_g2s(sf_s + (rt * chunks + c) * 256, wsf + sf_base + static_cast<int64_t>(kb_begin / 4 + c) * 512,
                256, &bar);
        }
    }
    uint4 g[NB];
    if constexpr (NORM)
    {
        // The RMSNorm weight is a constant: load it before the PDL wait.
#pragma unroll
        for (int b = 0; b < NB; ++b)
        {
            g[b] = __ldg(reinterpret_cast<uint4 const*>(na.gamma + 8 * lane + 256 * b));
        }
    }
    if (ctl.trig == 1)
    {
        asm volatile("griddepcontrol.launch_dependents;");
    }
    if ((ctl.flags & 1) == 0)
    {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    if (ctl.trig == 2)
    {
        asm volatile("griddepcontrol.launch_dependents;");
    }
    if constexpr (NORM)
    {
        for (int m = warp; m < M; m += WARPS)
        {
            __nv_bfloat16* qr_row
                = (blockIdx.x == 0 && na.qr_out != nullptr) ? na.qr_out + static_cast<int64_t>(m) * na.ld_qr : nullptr;
            norm_quant_row<NB>(x + static_cast<int64_t>(m) * ldx, g, na, lane, qr_row, xq_s + m * kc,
                xc_s + m * kc_blocks, k_begin, kc);
        }
    }
    else
    {
        int const items = M * kc_blocks;
        for (int p0 = threadIdx.x; p0 < items; p0 += 2 * kThreads)
        {
            uint4 raw[2][4];
#pragma unroll
            for (int u = 0; u < 2; ++u)
            {
                int const p = p0 + u * kThreads;
                if (p < items)
                {
                    int const m = p / kc_blocks;
                    int const b = p % kc_blocks;
                    uint4 const* src
                        = reinterpret_cast<uint4 const*>(x + static_cast<int64_t>(m) * ldx + k_begin + b * 32);
#pragma unroll
                    for (int q = 0; q < 4; ++q)
                    {
                        raw[u][q] = src[q];
                    }
                }
            }
#pragma unroll
            for (int u = 0; u < 2; ++u)
            {
                int const p = p0 + u * kThreads;
                if (p >= items)
                {
                    continue;
                }
                int const m = p / kc_blocks;
                int const b = p % kc_blocks;
                float v[32];
#pragma unroll
                for (int q = 0; q < 4; ++q)
                {
                    __nv_bfloat162 const* h = reinterpret_cast<__nv_bfloat162 const*>(&raw[u][q]);
#pragma unroll
                    for (int t = 0; t < 4; ++t)
                    {
                        float2 const f = __bfloat1622float2(h[t]);
                        v[q * 8 + t * 2] = f.x;
                        v[q * 8 + t * 2 + 1] = f.y;
                    }
                }
                float amax = 0.f;
#pragma unroll
                for (int t = 0; t < 32; ++t)
                {
                    amax = fmaxf(amax, fabsf(v[t]));
                }
                float const sf = amax * rcp_approx_ftz(448.0f);
                __nv_fp8_e8m0 code;
                code.__x = __nv_cvt_float_to_e8m0(sf, __NV_SATFINITE, cudaRoundPosInf);
                float const out_scale = amax != 0.f
                    ? (code.__x == 0 ? 1.0f : exp2f(127.0f - static_cast<float>(code.__x)))
                    : 0.0f;
                uint32_t packed[8];
#pragma unroll
                for (int t = 0; t < 8; ++t)
                {
                    __nv_fp8x2_e4m3 const lo(make_float2(v[4 * t] * out_scale, v[4 * t + 1] * out_scale));
                    __nv_fp8x2_e4m3 const hi(make_float2(v[4 * t + 2] * out_scale, v[4 * t + 3] * out_scale));
                    packed[t] = static_cast<uint32_t>(lo.__x) | (static_cast<uint32_t>(hi.__x) << 16);
                }
                uint4* dst = reinterpret_cast<uint4*>(xq_s + m * kc + b * 32);
                dst[0] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
                dst[1] = make_uint4(packed[4], packed[5], packed[6], packed[7]);
                xc_s[m * kc_blocks + b] = code.__x;
            }
        }
    }
    __syncthreads();

    int const gq = lane >> 2;
    int const j = lane & 3;
    int const col0 = 2 * j;
    int const bps = kc_blocks / WARPS;
    int const chunks = kc_blocks / 4;
    float acc[RT][4] = {};
    mbar_wait(&bar, 0);
#pragma unroll 2
    for (int i = 0; i < bps; ++i)
    {
        int const b = warp * bps + i;
        int const kb = kb_begin + b;
        uint2 const xb = gq < M ? *reinterpret_cast<uint2 const*>(xq_s + gq * kc + b * 32 + 8 * j) : make_uint2(0, 0);
        float const xs0 = col0 < M ? pow2_from_ue8m0(xc_s[col0 * kc_blocks + b]) : 0.f;
        float const xs1 = col0 + 1 < M ? pow2_from_ue8m0(xc_s[(col0 + 1) * kc_blocks + b]) : 0.f;
#pragma unroll
        for (int rt = 0; rt < RT; ++rt)
        {
            uint8_t const* wa_s = w_s + (16 * rt + gq) * row_stride + 8 * j + b * 32;
            uint2 const wa = *reinterpret_cast<uint2 const*>(wa_s);
            uint2 const wb = *reinterpret_cast<uint2 const*>(wa_s + 8 * row_stride);
            int const inner_m = ((row0 + 16 * rt) % 128) / 32;
            uint8_t const* sft = sf_s + (rt * chunks + b / 4) * 256 + inner_m * 4 + kb % 4;
            float const wsa = pow2_from_ue8m0(sft[gq * 16]);
            float const wsb = pow2_from_ue8m0(sft[(gq + 8) * 16]);
            float d[4];
            mma_e4m3(d, wa.x, wb.x, wa.y, wb.y, xb.x, xb.y);
            acc[rt][0] += d[0] * (wsa * xs0);
            acc[rt][1] += d[1] * (wsa * xs1);
            acc[rt][2] += d[2] * (wsb * xs0);
            acc[rt][3] += d[3] * (wsb * xs1);
        }
    }
    if (ctl.trig == 0)
    {
        asm volatile("griddepcontrol.launch_dependents;");
    }
    if (red_aliased)
    {
        __syncthreads();
    }
#pragma unroll
    for (int rt = 0; rt < RT; ++rt)
    {
        int const ra = 16 * rt + gq;
        red[warp * kTile + ra * 8 + col0] = acc[rt][0];
        red[warp * kTile + ra * 8 + col0 + 1] = acc[rt][1];
        red[warp * kTile + (ra + 8) * 8 + col0] = acc[rt][2];
        red[warp * kTile + (ra + 8) * 8 + col0 + 1] = acc[rt][3];
    }
    __syncthreads();

    if constexpr (CLUSTER_K > 1)
    {
        asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
    }
    if (CLUSTER_K == 1 || rank == 0)
    {
        if constexpr (CLUSTER_K > 1)
        {
            uint32_t ready = 0;
            while (!ready)
            {
                asm volatile(
                    "{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\n"
                    " selp.u32 %0, 1, 0, p;\n}\n"
                    : "=r"(ready)
                    : "r"(smem_u32(&red_bar))
                    : "memory");
            }
        }
        __syncthreads();
        for (int idx = threadIdx.x; idx < live; idx += kThreads)
        {
            int const r = idx / M;
            int const m = idx - r * M;
            float total = 0.f;
#pragma unroll
            for (int ww = 0; ww < WARPS; ++ww)
            {
                total += red[ww * kTile + r * 8 + m];
            }
#pragma unroll
            for (int c = 1; c < CLUSTER_K; ++c)
            {
                total += cluster_red[(c - 1) * live + idx];
            }
            int const row = row0 + r;
            if (row < n_split)
            {
                out[static_cast<int64_t>(m) * ldo + row] = __float2bfloat16_rn(total);
            }
            else
            {
                out2[static_cast<int64_t>(m) * ldo2 + (row - n_split)] = __float2bfloat16_rn(total);
            }
        }
    }
    else
    {
        uint32_t const remote_bar = mapa_shared(smem_u32(&red_bar), 0);
        uint32_t const remote_dst = mapa_shared(smem_u32(cluster_red + (rank - 1) * live), 0);
        for (int idx = threadIdx.x; idx < live; idx += kThreads)
        {
            int const r = idx / M;
            int const m = idx - r * M;
            float s = 0.f;
#pragma unroll
            for (int ww = 0; ww < WARPS; ++ww)
            {
                s += red[ww * kTile + r * 8 + m];
            }
            asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];\n" ::"r"(
                             remote_dst + idx * 4),
                         "r"(__float_as_uint(s)), "r"(remote_bar)
                         : "memory");
        }
    }
    __syncthreads();
    if (ctl.trig == 3)
    {
        asm volatile("griddepcontrol.launch_dependents;");
    }
}

// Norm + quant only, one warp per row over many rows (bitwise checks of the device function on >= 8192 rows).
__global__ void norm_test_kernel(__nv_bfloat16 const* x, int ldx, int rows, NormArgs na, __nv_bfloat16* qr,
    uint8_t* xq, uint8_t* xc)
{
    int const warp = threadIdx.x / 32;
    int const lane = threadIdx.x % 32;
    int const r = blockIdx.x * (blockDim.x / 32) + warp;
    if (r >= rows)
        return;
    uint4 g[5];
#pragma unroll
    for (int b = 0; b < 5; ++b)
        g[b] = __ldg(reinterpret_cast<uint4 const*>(na.gamma + 8 * lane + 256 * b));
    norm_quant_row<5>(x + static_cast<int64_t>(r) * ldx, g, na, lane, qr + static_cast<int64_t>(r) * 1280,
        xq + static_cast<int64_t>(r) * 1280, xc + static_cast<int64_t>(r) * 40, 0, 1280);
}

// Stand-in for the rest of the layer: waits for its producer, reads one output value, triggers at its end.
// trig_first: K1-like producer stand-in that triggers its dependents at once, then waits and spins ~spin_ns.
__global__ void glue_kernel(__nv_bfloat16 const* q, float* sink, long long spin_ns, int trig_first)
{
    if (trig_first)
        asm volatile("griddepcontrol.launch_dependents;");
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (spin_ns > 0)
    {
        long long t0, t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
        do
        {
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        } while (t - t0 < spin_ns);
    }
    if (threadIdx.x == 0 && blockIdx.x == 0)
        sink[0] += __bfloat162float(q[0]);
    if (!trig_first)
        asm volatile("griddepcontrol.launch_dependents;");
}

// Live-overlap PDL producer: triggers at once, NaN-poisons dst, spins ~spin_ns, then writes src -> dst.
__global__ void poison_producer_kernel(__nv_bfloat16 const* src, __nv_bfloat16* dst, int n, int ld, int cols,
    long long spin_ns)
{
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;");
    long long t0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x)
        dst[(i / cols) * static_cast<int64_t>(ld) + i % cols] = __ushort_as_bfloat16(0x7fc0);
    long long t;
    do
    {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    } while (t - t0 < spin_ns);
    __threadfence();
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x)
        dst[(i / cols) * static_cast<int64_t>(ld) + i % cols] = src[(i / cols) * static_cast<int64_t>(ld) + i % cols];
}

size_t smem_bytes(int warps, int ck, int rt, int m, int k)
{
    int const kc = k / ck;
    int const kcb = kc / 32;
    size_t const rows = 16 * rt;
    size_t const wb = rows * (kc + kRowPad);
    size_t const red = static_cast<size_t>(warps) * rows * 8 * sizeof(float);
    bool const aliased = red <= wb;
    return wb + rt * (kcb / 4) * 256 + static_cast<size_t>(m) * kc + ((m * kcb + 15) / 16) * 16
        + static_cast<size_t>(ck - 1) * rows * m * sizeof(float) + (aliased ? 0 : red);
}

template <int WARPS, int CK, int RT, bool NORM>
void launch_t(__nv_bfloat16 const* x, int ldx, int M, int N, int K, uint8_t const* w, uint8_t const* wsf,
    __nv_bfloat16* out, int ldo, __nv_bfloat16* out2, int ldo2, int nSplit, NormArgs na, Ctl ctl, bool pdl,
    cudaStream_t stream)
{
    int const kc = K / CK;
    int const kcb = kc / 32;
    TORCH_CHECK(N % (16 * RT) == 0 && K % (CK * 128) == 0 && kcb % WARPS == 0 && (kcb % 4) == 0
            && nSplit % (16 * RT) == 0 && (!NORM || K == 1280),
        "qb proto: unsupported shape");
    size_t const wbytes = 16 * RT * (kc + kRowPad);
    size_t const red = static_cast<size_t>(WARPS) * 16 * RT * 8 * sizeof(float);
    int const red_aliased = red <= wbytes ? 1 : 0;
    size_t const smem = smem_bytes(WARPS, CK, RT, M, K);
    auto kernel = qb_kernel<WARPS, CK, RT, NORM>;
    static bool const configured = [&]()
    {
        int device = 0, max_optin = 0;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
        cudaFuncAttributes attrs = {};
        if (cudaFuncGetAttributes(&attrs, kernel) != cudaSuccess)
            return false;
        bool ok = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                      max_optin - static_cast<int>(attrs.sharedSizeBytes))
            == cudaSuccess;
        ok = ok
            && cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared)
                == cudaSuccess;
        return ok;
    }();
    TORCH_CHECK(configured, "qb proto: configure failed");
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(N / (16 * RT) * CK);
    cfg.blockDim = dim3(WARPS * 32);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute attr[2];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    attr[1].id = cudaLaunchAttributeClusterDimension;
    attr[1].val.clusterDim.x = CK;
    attr[1].val.clusterDim.y = 1;
    attr[1].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 2;
    cudaError_t const err = cudaLaunchKernelEx(&cfg, kernel, x, ldx, M, N, K, w, wsf, (K / 32 + 3) / 4, out, ldo, out2,
        ldo2, nSplit, red_aliased, na, ctl);
    TORCH_CHECK(err == cudaSuccess, "qb proto launch failed: ", cudaGetErrorString(err));
}

#define QB_CONFIGS(X)                                                                                                  \
    X(8, 2, 2, false) X(8, 4, 1, false) X(4, 4, 1, false) X(4, 4, 2, false) X(4, 8, 1, false) X(4, 8, 2, false)       \
        X(4, 2, 2, false) X(4, 1, 1, false) X(4, 1, 2, false) X(4, 2, 1, false) X(8, 1, 2, false)                      \
            X(4, 1, 1, true) X(4, 1, 2, true) X(4, 1, 4, true) X(4, 2, 1, true) X(4, 2, 2, true) X(8, 1, 2, true)      \
                X(8, 1, 4, true)

} // namespace

void gemm(torch::Tensor x, torch::Tensor w, torch::Tensor wsf, torch::Tensor out, std::optional<torch::Tensor> out2,
    int64_t n_split, int64_t warps, int64_t ck, int64_t rt, bool norm, std::optional<torch::Tensor> gamma, double eps,
    std::optional<torch::Tensor> qr_out, int64_t order, int64_t trig, int64_t flags, bool pdl)
{
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.stride(1) == 1 && x.stride(0) % 8 == 0);
    TORCH_CHECK(w.is_contiguous() && w.size(1) == x.size(1));
    int const M = x.size(0), N = w.size(0), K = x.size(1);
    TORCH_CHECK(M >= 1 && M <= 8);
    TORCH_CHECK(out.stride(1) == 1);
    NormArgs na{nullptr, static_cast<float>(eps), nullptr, 0, static_cast<int>(order)};
    if (norm)
    {
        TORCH_CHECK(gamma.has_value() && gamma->numel() == K && gamma->is_contiguous());
        na.gamma = reinterpret_cast<__nv_bfloat16 const*>(gamma->data_ptr());
        if (qr_out.has_value())
        {
            TORCH_CHECK(qr_out->stride(1) == 1 && qr_out->size(1) == K);
            na.qr_out = reinterpret_cast<__nv_bfloat16*>(qr_out->data_ptr());
            na.ld_qr = qr_out->stride(0);
        }
    }
    Ctl ctl{static_cast<int>(trig), static_cast<int>(flags)};
    __nv_bfloat16* o2 = nullptr;
    int ldo2 = 0;
    if (out2.has_value())
    {
        o2 = reinterpret_cast<__nv_bfloat16*>(out2->data_ptr());
        ldo2 = out2->stride(0);
    }
    int const ns = n_split == 0 ? N : static_cast<int>(n_split);
    auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    auto xp = reinterpret_cast<__nv_bfloat16 const*>(x.data_ptr());
    auto wp = reinterpret_cast<uint8_t const*>(w.data_ptr());
    auto sp = reinterpret_cast<uint8_t const*>(wsf.data_ptr());
    auto op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
#define QB_DISPATCH(W, C, R, NM)                                                                                       \
    if (warps == W && ck == C && rt == R && norm == NM)                                                                \
    {                                                                                                                  \
        launch_t<W, C, R, NM>(xp, x.stride(0), M, N, K, wp, sp, op, out.stride(0), o2, ldo2, ns, na, ctl, pdl, stream); \
        return;                                                                                                        \
    }
    QB_CONFIGS(QB_DISPATCH)
#undef QB_DISPATCH
    TORCH_CHECK(false, "qb proto: config not built");
}

int64_t smem(int64_t warps, int64_t ck, int64_t rt, int64_t m, int64_t k)
{
    return static_cast<int64_t>(smem_bytes(warps, ck, rt, m, k));
}

void norm_test(torch::Tensor x, torch::Tensor gamma, double eps, int64_t order, torch::Tensor qr, torch::Tensor xq,
    torch::Tensor xc)
{
    int const rows = x.size(0);
    NormArgs na{reinterpret_cast<__nv_bfloat16 const*>(gamma.data_ptr()), static_cast<float>(eps), nullptr, 0,
        static_cast<int>(order)};
    auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    norm_test_kernel<<<(rows + 3) / 4, 128, 0, stream>>>(reinterpret_cast<__nv_bfloat16 const*>(x.data_ptr()),
        x.stride(0), rows, na, reinterpret_cast<__nv_bfloat16*>(qr.data_ptr()),
        reinterpret_cast<uint8_t*>(xq.data_ptr()), reinterpret_cast<uint8_t*>(xc.data_ptr()));
    TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

void glue(torch::Tensor q, torch::Tensor sink, int64_t spin_ns, int64_t trig_first, int64_t ctas)
{
    auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(ctas);
    cfg.blockDim = dim3(32);
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaError_t const err = cudaLaunchKernelEx(&cfg, glue_kernel,
        reinterpret_cast<__nv_bfloat16 const*>(q.data_ptr()), reinterpret_cast<float*>(sink.data_ptr()),
        static_cast<long long>(spin_ns), static_cast<int>(trig_first));
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
}

void poison_producer(torch::Tensor src, torch::Tensor dst, int64_t spin_ns)
{
    // src/dst: [M, cols] views with the same row stride.
    auto stream = at::cuda::getCurrentCUDAStream(src.get_device());
    int const n = src.size(0) * src.size(1);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(4);
    cfg.blockDim = dim3(256);
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaError_t const err = cudaLaunchKernelEx(&cfg, poison_producer_kernel,
        reinterpret_cast<__nv_bfloat16 const*>(src.data_ptr()), reinterpret_cast<__nv_bfloat16*>(dst.data_ptr()), n,
        static_cast<int>(src.stride(0)), static_cast<int>(src.size(1)), static_cast<long long>(spin_ns));
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gemm", &gemm);
    m.def("smem", &smem);
    m.def("norm_test", &norm_test);
    m.def("glue", &glue);
    m.def("poison_producer", &poison_producer);
}
