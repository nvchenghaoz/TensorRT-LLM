// Idea 233 S1 prototype (torch cpp_extension, no TRT-LLM rebuild).
//
// K2' csa2_lowm_wo_a_kernel: SM120 decode wo_a for DeepSeek-V4.1 (TP4 rank: 2 groups x [1024, 4096] E4M3 with
//     128x128 UE8M0 block scales). One launch = BF16 inverse RoPE (mla_rope_inplace interleaved inverse form, as
//     b12x_wo._inv_rope_quant128_kernel ROPE_VARIANT 0) + 1x128 UE8M0 per-token quant (fp8_utils arithmetic, same
//     PTX ops as the Triton K1) + grouped low-M MXFP8-style GEMM in the structure of the landed
//     csa2_mxfp8_low_m_gemm_kernel (weights bulk-copied to smem BEFORE griddepcontrol.wait, m16n8k32 e4m3 MMAs with
//     C zeroed per 32-k block, pow2 scales accumulated in FP32, fixed-order warp + cluster DSMEM reduction).
//     Writes bf16 out[m, g * R + r] (o_lora layout).
// K3' csa2_port_kernel: verbatim copy of csa2_mxfp8_low_m_gemm_kernel (mxfp8LowMGemm.cu) with the shared
//     plan-v51 hunk: QoutArgs.flags bit 1 = griddepcontrol.launch_dependents right after the pre-wait staging.
// standin_kernel: latency-bound producer/consumer stand-ins (attention merge, lamport AR) for the chain bench.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace cg = cooperative_groups;

namespace
{

constexpr int kRowPad = 32;

__device__ __forceinline__ uint32_t smem_u32(void const* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ uint32_t mapa_shared(uint32_t addr, uint32_t target_rank)
{
    uint32_t r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(r) : "r"(addr), "r"(target_rank));
    return r;
}

__device__ __forceinline__ float rcp_approx_ftz(float a)
{
    float b;
    asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
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

// ---- Triton K1 arithmetic (QV selects PTX variants; QV = 0 is the form read from K1's PTX) ----
template <int QV>
__device__ __forceinline__ float q_div(float a, float b)
{
    float r;
    if constexpr ((QV & 1) == 0)
        asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    else
        asm("div.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

template <int QV>
__device__ __forceinline__ float q_lg2(float a)
{
    float r;
    if constexpr ((QV & 2) == 0)
        asm("lg2.approx.f32 %0, %1;" : "=f"(r) : "f"(a));
    else
        asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
    return r;
}

template <int QV>
__device__ __forceinline__ float q_ex2(float a)
{
    float r;
    if constexpr ((QV & 4) == 0)
        asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(a));
    else
        asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
    return r;
}

__device__ __forceinline__ float q_ceil(float a)
{
    float r;
    asm("cvt.rpi.f32.f32 %0, %1;" : "=f"(r) : "f"(a));
    return r;
}

__device__ __forceinline__ float bf16_round(float a)
{
    return __bfloat162float(__float2bfloat16_rn(a));
}

struct WoaArgs
{
    __nv_bfloat16 const* o; // [M, G * GW] attention output (heads contiguous)
    int ldo_in;
    int const* pos;  // [M]
    float const* cs; // [max_pos, 2 * half_rope]: cos then sin
    int cs_stride;
    uint8_t const* w;   // [G * R, GW] E4M3
    uint8_t const* wsc; // [G, R / 128, GW / 128] UE8M0
    __nv_bfloat16* out; // [M, G * R]
    int ld_out;
    int M, R, GW, head_dim, nope, half_rope;
    uint8_t* xq_dbg;      // [M, G, GW] quantized activation dump (tile 0 of each group) or null
    uint8_t* xs_dbg;      // [M, G, GW / 128] UE8M0 scale dump or null
    uint8_t const* pf;    // optional L2 prefetch range (next kernel's weights) or null
    long long pf_bytes;   //
    int flags;            // bit 0: skip griddepcontrol.wait (negative control); bit 1: trigger at start
    unsigned long long* ts; // debug %globaltimer per CTA [grid][4] or null
};

__device__ __forceinline__ void stamp(unsigned long long* ts, int point)
{
    if (ts != nullptr && threadIdx.x == 0)
    {
        unsigned long long t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        ts[blockIdx.x * 4 + point] = t;
    }
}

template <int WARPS, int CLUSTER_K, int RT, int QV>
__global__ void __launch_bounds__(WARPS * 32) csa2_lowm_wo_a_kernel(WoaArgs a, int red_aliased)
{
    constexpr int kThreads = WARPS * 32;
    constexpr int kRows = 16 * RT;
    constexpr int kTile = kRows * 8;
    extern __shared__ __align__(128) uint8_t smem[];
    __shared__ __align__(8) uint64_t bar;
    __shared__ __align__(8) uint64_t red_bar;

    int rank = 0;
    if constexpr (CLUSTER_K > 1)
    {
        rank = static_cast<int>(cg::this_cluster().block_rank());
    }
    int const M = a.M;
    int const tile = blockIdx.x / CLUSTER_K;
    int const row0 = tile * kRows;   // global row in [0, G * R)
    int const g = row0 / a.R;        // group
    int const lrow0 = row0 - g * a.R; // row within the group
    int const kc = a.GW / CLUSTER_K;
    int const kc_blocks = kc / 32;
    int const kb128 = kc / 128;
    int const k_begin = rank * kc; // within the group
    int const row_stride = kc + kRowPad;
    int const warp = threadIdx.x / 32;
    int const lane = threadIdx.x % 32;
    int const live = kRows * M;
    stamp(a.ts, 0);

    uint8_t* w_s = smem;                                      // [kRows][row_stride]
    uint8_t* xq_s = w_s + kRows * row_stride;                 // [M][kc]
    uint8_t* xs_s = xq_s + M * kc;                            // [M][kb128]
    uint8_t* ws_s = xs_s + ((M * kb128 + 15) / 16) * 16;      // [kb128]
    float* cluster_red = reinterpret_cast<float*>(ws_s + ((kb128 + 15) / 16) * 16); // [CK-1][live]
    float* red = red_aliased ? reinterpret_cast<float*>(w_s) : cluster_red + (CLUSTER_K - 1) * live;

    if (threadIdx.x == 0)
    {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(smem_u32(&bar)));
        if (CLUSTER_K > 1 && rank == 0)
        {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(smem_u32(&red_bar)));
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(&bar)),
                     "r"(static_cast<uint32_t>(kRows * kc))
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
    // Static weights: stage before the PDL wait.
    if (warp == 0)
    {
        for (int r = lane; r < kRows; r += 32)
        {
            bulk_g2s(w_s + r * row_stride, a.w + static_cast<int64_t>(row0 + r) * a.GW + k_begin, kc, &bar);
        }
    }
    {
        int const rb = lrow0 / 128;
        int const kt = a.GW / 128;
        for (int t = threadIdx.x; t < kb128; t += kThreads)
        {
            ws_s[t] = a.wsc[(g * (a.R / 128) + rb) * kt + k_begin / 128 + t];
        }
    }
    if (a.pf != nullptr)
    {
        long long const per = ((a.pf_bytes + gridDim.x - 1) / gridDim.x + 15) / 16 * 16;
        long long const beg = per * blockIdx.x;
        if (threadIdx.x == 0 && beg < a.pf_bytes)
        {
            long long n = a.pf_bytes - beg < per ? a.pf_bytes - beg : per;
            n = n / 16 * 16;
            if (n > 0)
            {
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(a.pf + beg), "r"(
                                 static_cast<uint32_t>(n))
                             : "memory");
            }
        }
    }
    if (a.flags & 2)
    {
        asm volatile("griddepcontrol.launch_dependents;");
    }
    if ((a.flags & 1) == 0)
    {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    // Prologue: inverse RoPE + 1x128 quant of this CTA's K slice for every token (4 lanes x 32 values per block).
    {
        int const items = M * kb128;
        int const sub = threadIdx.x & 3;
        bool const dump = a.xq_dbg != nullptr && lrow0 == 0;
        for (int p = threadIdx.x >> 2; p < (items + kThreads / 4 - 1) / (kThreads / 4) * (kThreads / 4);
             p += kThreads / 4)
        {
            bool const act = p < items;
            int const m = act ? p / kb128 : 0;
            int const b = act ? p - m * kb128 : 0;
            int const kq = k_begin + b * 128 + sub * 32; // within the group
            float v[32] = {};
            if (act)
            {
                uint4 const* src = reinterpret_cast<uint4 const*>(
                    a.o + static_cast<int64_t>(m) * a.ldo_in + static_cast<int64_t>(g) * a.GW + kq);
                uint4 raw[4];
#pragma unroll
                for (int q = 0; q < 4; ++q)
                {
                    raw[q] = src[q];
                }
#pragma unroll
                for (int q = 0; q < 4; ++q)
                {
                    __nv_bfloat162 const* h = reinterpret_cast<__nv_bfloat162 const*>(&raw[q]);
#pragma unroll
                    for (int t = 0; t < 4; ++t)
                    {
                        float2 const f = __bfloat1622float2(h[t]);
                        v[q * 8 + t * 2] = f.x;
                        v[q * 8 + t * 2 + 1] = f.y;
                    }
                }
                int const hd = kq % a.head_dim;
                if (hd >= a.nope)
                {
                    int const ci0 = (hd - a.nope) >> 1;
                    float const* cb = a.cs + static_cast<int64_t>(a.pos[m]) * a.cs_stride;
                    float c[16], s[16];
#pragma unroll
                    for (int q = 0; q < 4; ++q)
                    {
                        float4 const cv = *reinterpret_cast<float4 const*>(cb + ci0 + 4 * q);
                        float4 const sv = *reinterpret_cast<float4 const*>(cb + a.half_rope + ci0 + 4 * q);
                        c[4 * q] = cv.x;
                        c[4 * q + 1] = cv.y;
                        c[4 * q + 2] = cv.z;
                        c[4 * q + 3] = cv.w;
                        s[4 * q] = sv.x;
                        s[4 * q + 1] = sv.y;
                        s[4 * q + 2] = sv.z;
                        s[4 * q + 3] = sv.w;
                    }
#pragma unroll
                    for (int t = 0; t < 16; ++t)
                    {
                        float const xe = v[2 * t];
                        float const xo = v[2 * t + 1];
                        float const ns = -s[t];
                        // K1 ROPE_VARIANT 0: r_even = fma(c, x, -(ns * partner)), r_odd = fma(c, x, ns * partner)
                        float const re = __fmaf_rn(c[t], xe, -__fmul_rn(ns, xo));
                        float const ro = __fmaf_rn(c[t], xo, __fmul_rn(ns, xe));
                        v[2 * t] = bf16_round(re);
                        v[2 * t + 1] = bf16_round(ro);
                    }
                }
            }
            float amax = 0.f;
#pragma unroll
            for (int t = 0; t < 32; ++t)
            {
                amax = fmaxf(amax, fabsf(v[t]));
            }
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
            if (!act)
            {
                continue;
            }
            amax = fmaxf(amax, 1e-10f);
            float sc = q_div<QV>(amax, 448.0f);
            sc = q_ex2<QV>(q_ceil(q_lg2<QV>(fabsf(sc))));
            uint32_t packed[8];
#pragma unroll
            for (int t = 0; t < 8; ++t)
            {
                float f[4];
#pragma unroll
                for (int u = 0; u < 4; ++u)
                {
                    f[u] = fminf(fmaxf(q_div<QV>(v[4 * t + u], sc), -448.0f), 448.0f);
                }
                __nv_fp8x2_e4m3 const lo(make_float2(f[0], f[1]));
                __nv_fp8x2_e4m3 const hi(make_float2(f[2], f[3]));
                packed[t] = static_cast<uint32_t>(lo.__x) | (static_cast<uint32_t>(hi.__x) << 16);
            }
            uint8_t const byte = static_cast<uint8_t>((__float_as_uint(sc) >> 23) & 0xFF);
            uint4* dst = reinterpret_cast<uint4*>(xq_s + m * kc + b * 128 + sub * 32);
            dst[0] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
            dst[1] = make_uint4(packed[4], packed[5], packed[6], packed[7]);
            if (sub == 0)
            {
                xs_s[m * kb128 + b] = byte;
            }
            if (dump)
            {
                uint4* dd = reinterpret_cast<uint4*>(
                    a.xq_dbg + (static_cast<int64_t>(m) * (gridDim.x / CLUSTER_K * kRows / a.R) + g) * a.GW + kq);
                dd[0] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
                dd[1] = make_uint4(packed[4], packed[5], packed[6], packed[7]);
                if (sub == 0)
                {
                    a.xs_dbg[(static_cast<int64_t>(m) * (gridDim.x / CLUSTER_K * kRows / a.R) + g) * (a.GW / 128)
                        + kq / 128]
                        = byte;
                }
            }
        }
    }
    __syncthreads();
    stamp(a.ts, 1);

    int const gq = lane >> 2;
    int const j = lane & 3;
    int const col0 = 2 * j;
    int const bps = kc_blocks / WARPS;
    float acc[RT][4] = {};
    mbar_wait(&bar, 0);
    stamp(a.ts, 2);
#pragma unroll 2
    for (int i = 0; i < bps; ++i)
    {
        int const b = warp * bps + i;
        int const b4 = b >> 2;
        uint2 const xb
            = gq < M ? *reinterpret_cast<uint2 const*>(xq_s + gq * kc + b * 32 + 8 * j) : make_uint2(0, 0);
        float const ws = pow2_from_ue8m0(ws_s[b4]);
        float const xs0 = col0 < M ? pow2_from_ue8m0(xs_s[col0 * kb128 + b4]) : 0.f;
        float const xs1 = col0 + 1 < M ? pow2_from_ue8m0(xs_s[(col0 + 1) * kb128 + b4]) : 0.f;
#pragma unroll
        for (int rt = 0; rt < RT; ++rt)
        {
            uint8_t const* wa_s = w_s + (16 * rt + gq) * row_stride + 8 * j + b * 32;
            uint2 const wa = *reinterpret_cast<uint2 const*>(wa_s);
            uint2 const wb = *reinterpret_cast<uint2 const*>(wa_s + 8 * row_stride);
            float d[4];
            mma_e4m3(d, wa.x, wb.x, wa.y, wb.y, xb.x, xb.y);
            acc[rt][0] += d[0] * (ws * xs0);
            acc[rt][1] += d[1] * (ws * xs1);
            acc[rt][2] += d[2] * (ws * xs0);
            acc[rt][3] += d[3] * (ws * xs1);
        }
    }
    stamp(a.ts, 3);
    asm volatile("griddepcontrol.launch_dependents;");
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
            a.out[static_cast<int64_t>(m) * a.ld_out + row0 + r] = __float2bfloat16_rn(total);
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
    stamp(a.ts, 3);
}

size_t woaSmem(int warps, int ck, int rt, int m, int gw)
{
    int const kc = gw / ck;
    int const kb128 = kc / 128;
    size_t const rows = 16 * rt;
    size_t const w_bytes = rows * (kc + kRowPad);
    size_t const red_bytes = static_cast<size_t>(warps) * rows * 8 * sizeof(float);
    bool const aliased = red_bytes <= w_bytes;
    return w_bytes + static_cast<size_t>(m) * kc + ((m * kb128 + 15) / 16) * 16 + ((kb128 + 15) / 16) * 16
        + static_cast<size_t>(ck - 1) * rows * m * sizeof(float) + (aliased ? 0 : red_bytes);
}

template <typename K>
void configure(K kernel)
{
    static bool done = false; // per instantiation (template static)
    (void) done;
    int device = 0, max_optin = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    cudaFuncAttributes attrs = {};
    TORCH_CHECK(cudaFuncGetAttributes(&attrs, kernel) == cudaSuccess);
    TORCH_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                    max_optin - static_cast<int>(attrs.sharedSizeBytes))
        == cudaSuccess);
    TORCH_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                    cudaSharedmemCarveoutMaxShared)
        == cudaSuccess);
}

template <typename K, typename... Args>
void launch_ex(K kernel, int grid, int block, size_t smem, int cluster, bool pdl, cudaStream_t stream, Args... args)
{
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(grid);
    cfg.blockDim = dim3(block);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute attr[2];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    attr[1].id = cudaLaunchAttributeClusterDimension;
    attr[1].val.clusterDim.x = cluster;
    attr[1].val.clusterDim.y = 1;
    attr[1].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 2;
    cudaError_t const err = cudaLaunchKernelEx(&cfg, kernel, args...);
    TORCH_CHECK(err == cudaSuccess, "launch failed: ", cudaGetErrorString(err));
}

template <int W, int C, int R, int QV>
void launchWoa(WoaArgs const& a, int G, cudaStream_t stream, bool pdl)
{
    auto kernel = csa2_lowm_wo_a_kernel<W, C, R, QV>;
    static bool const configured = [&]()
    {
        configure(kernel);
        return true;
    }();
    (void) configured;
    int const kc = a.GW / C;
    TORCH_CHECK(a.R % (16 * R) == 0 && a.GW % (C * 128) == 0 && (kc / 32) % W == 0, "bad wo_a config");
    size_t const smem = woaSmem(W, C, R, a.M, a.GW);
    size_t const w_bytes = 16 * R * (kc + kRowPad);
    int const red_aliased = static_cast<size_t>(W) * 16 * R * 8 * 4 <= w_bytes ? 1 : 0;
    launch_ex(kernel, G * a.R / (16 * R) * C, W * 32, smem, C, pdl, stream, a, red_aliased);
}

#define WOA_CONFIGS(X)                                                                                                 \
    X(4, 4, 1) X(4, 8, 1) X(4, 4, 2) X(4, 2, 1) X(4, 8, 2) X(8, 4, 1) X(8, 8, 2) X(4, 2, 2) X(8, 4, 2) X(4, 1, 1)

} // namespace

// o: [M, G*GW] bf16; pos int32 [M]; cs fp32 [P, 2*half]; w e4m3 [G*R, GW] (or [G, R, GW]); wsc uint8 [G, R/128, GW/128]
void wo_a(torch::Tensor o, torch::Tensor pos, torch::Tensor cs, torch::Tensor w, torch::Tensor wsc,
    torch::Tensor out, int64_t groups, int64_t head_dim, int64_t nope, int64_t warps, int64_t ck, int64_t rt,
    int64_t qv, int64_t flags, bool pdl, std::optional<torch::Tensor> xq_dbg, std::optional<torch::Tensor> xs_dbg,
    std::optional<torch::Tensor> pf, std::optional<torch::Tensor> ts)
{
    TORCH_CHECK(o.scalar_type() == at::kBFloat16 && o.dim() == 2 && o.stride(1) == 1);
    TORCH_CHECK(pos.scalar_type() == at::kInt && cs.scalar_type() == at::kFloat);
    int const G = static_cast<int>(groups);
    WoaArgs a{};
    a.o = reinterpret_cast<__nv_bfloat16 const*>(o.data_ptr());
    a.ldo_in = static_cast<int>(o.stride(0));
    a.pos = pos.data_ptr<int>();
    a.cs = cs.data_ptr<float>();
    a.cs_stride = static_cast<int>(cs.stride(0));
    a.w = reinterpret_cast<uint8_t const*>(w.data_ptr());
    a.wsc = reinterpret_cast<uint8_t const*>(wsc.data_ptr());
    a.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    a.ld_out = static_cast<int>(out.stride(0));
    a.M = static_cast<int>(o.size(0));
    a.GW = static_cast<int>(o.size(1) / G);
    a.R = static_cast<int>(w.numel() / a.GW / G);
    a.head_dim = static_cast<int>(head_dim);
    a.nope = static_cast<int>(nope);
    a.half_rope = static_cast<int>((head_dim - nope) / 2);
    a.xq_dbg = xq_dbg.has_value() ? reinterpret_cast<uint8_t*>(xq_dbg->data_ptr()) : nullptr;
    a.xs_dbg = xs_dbg.has_value() ? reinterpret_cast<uint8_t*>(xs_dbg->data_ptr()) : nullptr;
    a.pf = pf.has_value() ? reinterpret_cast<uint8_t const*>(pf->data_ptr()) : nullptr;
    a.pf_bytes = pf.has_value() ? pf->numel() * pf->element_size() : 0;
    a.flags = static_cast<int>(flags);
    a.ts = ts.has_value() ? reinterpret_cast<unsigned long long*>(ts->data_ptr()) : nullptr;
    TORCH_CHECK(a.M >= 1 && a.M <= 8);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(o.get_device());
#define WOA_DISPATCH(W, C, R)                                                                                          \
    if (warps == W && ck == C && rt == R)                                                                              \
    {                                                                                                                  \
        if (qv == 0)                                                                                                   \
            launchWoa<W, C, R, 0>(a, G, stream, pdl);                                                                  \
        else if (qv == 1)                                                                                              \
            launchWoa<W, C, R, 1>(a, G, stream, pdl);                                                                  \
        else if (qv == 2)                                                                                              \
            launchWoa<W, C, R, 2>(a, G, stream, pdl);                                                                  \
        else if (qv == 4)                                                                                              \
            launchWoa<W, C, R, 4>(a, G, stream, pdl);                                                                  \
        else                                                                                                           \
            launchWoa<W, C, R, 6>(a, G, stream, pdl);                                                                  \
        return;                                                                                                        \
    }
    WOA_CONFIGS(WOA_DISPATCH)
#undef WOA_DISPATCH
    TORCH_CHECK(false, "unsupported wo_a config");
}

int64_t wo_a_smem(int64_t warps, int64_t ck, int64_t rt, int64_t m, int64_t gw)
{
    return static_cast<int64_t>(woaSmem(warps, ck, rt, m, gw));
}

// ------------------------------------------------------------------------------------------------------------------
// K3': copy of csa2_mxfp8_low_m_gemm_kernel (mxfp8LowMGemm.cu, main 2fa8e034) + plan-v51 trigger hunk (flags bit 1).
// ------------------------------------------------------------------------------------------------------------------
namespace
{
struct PreqArgs
{
    uint8_t const* xq;
    uint8_t const* xsf;
    int sf_k_tiles;
};

struct QoutArgs
{
    uint8_t* xq;
    uint8_t* xsf;
    int sf_k_tiles;
    unsigned long long* ts;
    int flags; // bit 0 skips griddepcontrol.wait (negative control of the PDL checks only); bit 1: trigger dependents at start
};

__device__ __forceinline__ int csa2_sf_offset(int m, int kb, int sf_k_tiles)
{
    return (m / 128) * sf_k_tiles * 512 + (kb / 4) * 512 + (m % 32) * 16 + ((m % 128) / 32) * 4 + kb % 4;
}

template <int WARPS, int CLUSTER_K, int RT>
__global__ void __launch_bounds__(WARPS * 32) csa2_port_kernel(__nv_bfloat16 const* __restrict__ x, int ldx, int M,
    int N, int K, uint8_t const* __restrict__ w, uint8_t const* __restrict__ wsf, int num_k_tiles,
    __nv_bfloat16* __restrict__ out, int ldo, int red_aliased, QoutArgs qout)
{
    constexpr int STAGES = 1;
    constexpr int kThreads = WARPS * 32;
    constexpr int kRows = 16 * RT;
    constexpr int kTile = kRows * 8;
    extern __shared__ __align__(128) uint8_t smem[];
    __shared__ __align__(8) uint64_t bar[STAGES];
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

    int const kcs = kc / STAGES;
    int const kcbs = kc_blocks / STAGES;
    int const chunks_s = kcbs / 4;
    uint32_t const stage_bytes = kRows * kcs + RT * chunks_s * 256u;
    if (threadIdx.x == 0)
    {
#pragma unroll
        for (int s = 0; s < STAGES; ++s)
        {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(smem_u32(&bar[s])));
        }
        if (CLUSTER_K > 1 && rank == 0)
        {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(smem_u32(&red_bar)));
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#pragma unroll
        for (int s = 0; s < STAGES; ++s)
        {
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(&bar[s])),
                         "r"(stage_bytes)
                         : "memory");
        }
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
#pragma unroll
        for (int s = 0; s < STAGES; ++s)
        {
            for (int r = lane; r < kRows; r += 32)
            {
                bulk_g2s(w_s + r * row_stride + s * kcs, w + static_cast<int64_t>(row0 + r) * K + k_begin + s * kcs,
                    kcs, &bar[s]);
            }
            for (int t = lane; t < RT * chunks_s; t += 32)
            {
                int const rt = t / chunks_s;
                int const c = s * chunks_s + t % chunks_s;
                int const rr = row0 + 16 * rt;
                int64_t const sf_base = static_cast<int64_t>(rr / 128) * num_k_tiles * 512 + (rr % 32) * 16;
                bulk_g2s(sf_s + (rt * chunks + c) * 256, wsf + sf_base + static_cast<int64_t>(kb_begin / 4 + c) * 512,
                    256, &bar[s]);
            }
        }
    }
    if (qout.flags & 2)
    {
        asm volatile("griddepcontrol.launch_dependents;");
    }
    if ((qout.flags & 1) == 0)
    {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
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
    if (qout.xq != nullptr && tile == 0)
    {
        for (int p = threadIdx.x; p < M * kc_blocks; p += kThreads)
        {
            int const m = p / kc_blocks;
            int const b = p % kc_blocks;
            uint4 const* src = reinterpret_cast<uint4 const*>(xq_s + m * kc + b * 32);
            uint4* dst = reinterpret_cast<uint4*>(qout.xq + static_cast<int64_t>(m) * K + k_begin + b * 32);
            dst[0] = src[0];
            dst[1] = src[1];
            qout.xsf[csa2_sf_offset(m, kb_begin + b, qout.sf_k_tiles)] = xc_s[m * kc_blocks + b];
        }
    }

    int const g = lane >> 2;
    int const j = lane & 3;
    int const col0 = 2 * j;
    int const bps = kcbs / WARPS;
    int const chunks = kc_blocks / 4;
    float acc[RT][4] = {};
    for (int s = 0; s < STAGES; ++s)
    {
        mbar_wait(&bar[s], 0);
#pragma unroll 2
        for (int i = 0; i < bps; ++i)
        {
            int const b = s * kcbs + warp * bps + i;
            int const kb = kb_begin + b;
            uint2 const xb
                = g < M ? *reinterpret_cast<uint2 const*>(xq_s + g * kc + b * 32 + 8 * j) : make_uint2(0, 0);
            float const xs0 = col0 < M ? pow2_from_ue8m0(xc_s[col0 * kc_blocks + b]) : 0.f;
            float const xs1 = col0 + 1 < M ? pow2_from_ue8m0(xc_s[(col0 + 1) * kc_blocks + b]) : 0.f;
#pragma unroll
            for (int rt = 0; rt < RT; ++rt)
            {
                uint8_t const* wa_s = w_s + (16 * rt + g) * row_stride + 8 * j + b * 32;
                uint2 const wa = *reinterpret_cast<uint2 const*>(wa_s);
                uint2 const wb = *reinterpret_cast<uint2 const*>(wa_s + 8 * row_stride);
                int const inner_m = ((row0 + 16 * rt) % 128) / 32;
                uint8_t const* sft = sf_s + (rt * chunks + b / 4) * 256 + inner_m * 4 + kb % 4;
                float const wsa = pow2_from_ue8m0(sft[g * 16]);
                float const wsb = pow2_from_ue8m0(sft[(g + 8) * 16]);
                float d[4];
                mma_e4m3(d, wa.x, wb.x, wa.y, wb.y, xb.x, xb.y);
                acc[rt][0] += d[0] * (wsa * xs0);
                acc[rt][1] += d[1] * (wsa * xs1);
                acc[rt][2] += d[2] * (wsb * xs0);
                acc[rt][3] += d[3] * (wsb * xs1);
            }
        }
    }
    asm volatile("griddepcontrol.launch_dependents;");
    if (red_aliased)
    {
        __syncthreads();
    }
#pragma unroll
    for (int rt = 0; rt < RT; ++rt)
    {
        int const ra = 16 * rt + g;
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
            out[static_cast<int64_t>(m) * ldo + row0 + r] = __float2bfloat16_rn(total);
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
}

size_t portSmem(int warps, int ck, int rt, int m, int k)
{
    int const kc = k / ck;
    int const kc_blocks = kc / 32;
    size_t const rows = 16 * rt;
    size_t const w_bytes = rows * (kc + kRowPad);
    size_t const red_bytes = static_cast<size_t>(warps) * rows * 8 * sizeof(float);
    bool const aliased = red_bytes <= w_bytes;
    return w_bytes + rt * (kc_blocks / 4) * 256 + static_cast<size_t>(m) * kc + ((m * kc_blocks + 15) / 16) * 16
        + static_cast<size_t>(ck - 1) * rows * m * sizeof(float) + (aliased ? 0 : red_bytes);
}

template <int W, int C, int R>
void launchPort(__nv_bfloat16 const* x, int ldx, int M, int N, int K, uint8_t const* w, uint8_t const* wsf,
    __nv_bfloat16* out, int ldo, QoutArgs qout, bool pdl, cudaStream_t stream)
{
    auto kernel = csa2_port_kernel<W, C, R>;
    static bool const configured = [&]()
    {
        configure(kernel);
        return true;
    }();
    (void) configured;
    int const kc = K / C;
    int const kc_blocks = kc / 32;
    TORCH_CHECK(N % (16 * R) == 0 && K % (C * 128) == 0 && kc_blocks % W == 0 && kc_blocks % 4 == 0, "bad port cfg");
    size_t const w_bytes = 16 * R * (kc + kRowPad);
    int const red_aliased = static_cast<size_t>(W) * 16 * R * 8 * 4 <= w_bytes ? 1 : 0;
    launch_ex(kernel, N / (16 * R) * C, W * 32, portSmem(W, C, R, M, K), C, pdl, stream, x, ldx, M, N, K, w, wsf,
        (K / 32 + 3) / 4, out, ldo, red_aliased, qout);
}

#define PORT_CONFIGS(X)                                                                                                \
    X(4, 1, 2) X(4, 2, 2) X(4, 2, 1) X(4, 4, 1) X(4, 4, 2) X(8, 2, 2) X(4, 1, 1) X(8, 1, 2) X(8, 2, 4) X(8, 4, 2)    \
        X(4, 8, 1) X(8, 4, 1) X(4, 2, 4)
} // namespace

void port_mm(torch::Tensor x, torch::Tensor w, torch::Tensor wsf, torch::Tensor out, int64_t warps, int64_t ck,
    int64_t rt, int64_t flags, bool pdl, std::optional<torch::Tensor> xq_out, std::optional<torch::Tensor> xsf_out)
{
    int const M = static_cast<int>(x.size(0));
    int const K = static_cast<int>(x.size(1));
    int const N = static_cast<int>(w.size(0));
    TORCH_CHECK(M >= 1 && M <= 8 && x.stride(1) == 1);
    QoutArgs q{xq_out.has_value() ? reinterpret_cast<uint8_t*>(xq_out->data_ptr()) : nullptr,
        xsf_out.has_value() ? reinterpret_cast<uint8_t*>(xsf_out->data_ptr()) : nullptr, (K / 32 + 3) / 4, nullptr,
        static_cast<int>(flags)};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());
#define PORT_DISPATCH(W, C, R)                                                                                         \
    if (warps == W && ck == C && rt == R)                                                                              \
    {                                                                                                                  \
        launchPort<W, C, R>(reinterpret_cast<__nv_bfloat16 const*>(x.data_ptr()), static_cast<int>(x.stride(0)), M, N, \
            K, reinterpret_cast<uint8_t const*>(w.data_ptr()), reinterpret_cast<uint8_t const*>(wsf.data_ptr()),      \
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(out.stride(0)), q, pdl, stream);       \
        return;                                                                                                        \
    }
    PORT_CONFIGS(PORT_DISPATCH)
#undef PORT_DISPATCH
    TORCH_CHECK(false, "unsupported port config");
}

int64_t port_smem(int64_t warps, int64_t ck, int64_t rt, int64_t m, int64_t k)
{
    return static_cast<int64_t>(portSmem(warps, ck, rt, m, k));
}

// ------------------------------------------------------------------------------------------------------------------
// Stand-ins: wait (unless flags & 4: no wait at all), copy src -> dst (n bf16), spin spin_ns from CTA start of work.
// flags bit 1: trigger dependents at start; otherwise at the end.
// ------------------------------------------------------------------------------------------------------------------
__global__ void standin_kernel(__nv_bfloat16 const* src, __nv_bfloat16* dst, int n, long long spin_ns, int flags)
{
    if (flags & 2)
    {
        asm volatile("griddepcontrol.launch_dependents;");
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    unsigned long long t0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        dst[i] = src[i];
    }
    if (spin_ns > 0)
    {
        unsigned long long t;
        do
        {
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        } while (t - t0 < static_cast<unsigned long long>(spin_ns));
    }
    __syncthreads();
    asm volatile("griddepcontrol.launch_dependents;");
}

void standin(torch::Tensor src, torch::Tensor dst, int64_t spin_ns, int64_t flags, int64_t ctas, bool pdl)
{
    int const n = static_cast<int>(src.numel());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(src.get_device());
    launch_ex(standin_kernel, static_cast<int>(ctas), 128, 0, 1, pdl, stream,
        reinterpret_cast<__nv_bfloat16 const*>(src.data_ptr()), reinterpret_cast<__nv_bfloat16*>(dst.data_ptr()), n,
        static_cast<long long>(spin_ns), static_cast<int>(flags));
}

// Live-overlap PDL producer: trigger at once, spin spin_ns, NaN-poison dst, then write src into dst.
__global__ void poison_producer_kernel(__nv_bfloat16 const* src, __nv_bfloat16* dst, int n, long long spin_ns)
{
    asm volatile("griddepcontrol.launch_dependents;");
    asm volatile("griddepcontrol.wait;" ::: "memory");
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        dst[i] = __ushort_as_bfloat16(0x7FC0);
    }
    do
    {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    } while (t - t0 < static_cast<unsigned long long>(spin_ns));
    __threadfence();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        dst[i] = src[i];
    }
}

void poison_producer(torch::Tensor src, torch::Tensor dst, int64_t spin_ns)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(src.get_device());
    launch_ex(poison_producer_kernel, 8, 256, 0, 1, true, stream,
        reinterpret_cast<__nv_bfloat16 const*>(src.data_ptr()), reinterpret_cast<__nv_bfloat16*>(dst.data_ptr()),
        static_cast<int>(src.numel()), static_cast<long long>(spin_ns));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("wo_a", &wo_a);
    m.def("wo_a_smem", &wo_a_smem);
    m.def("port_mm", &port_mm);
    m.def("port_smem", &port_smem);
    m.def("standin", &standin);
    m.def("poison_producer", &poison_producer);
}
