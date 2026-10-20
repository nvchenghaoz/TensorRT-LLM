# Idea 198 / branch 118: load_inline variants of nvfp4W4a16GemvKernel with a causal_conv1d_update
# epilogue (see step0.py). VAR selects how the conv operands reach the epilogue:
#   0: no conv (production kernel copy)
#   1: conv weight (pre-wait) + slot/state (post-wait) held in fp32 registers across the FMAs
#   2: conv weight + slot + state loaded after the FMAs (before the reduction); no live registers
#   3: cp.async into shared memory: weight pre-wait, slot (register) + state post-wait beside x
#   4: like 1 but packed bf16 registers (2 for weight, 2 for state)
import os

CU = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace {

constexpr int kGemvRows = 2;

struct ConvEpilogue
{
    __nv_bfloat16* state;
    __nv_bfloat16 const* weight;
    int const* indices;
    int64_t state_batch_stride;
    int64_t state_c_stride;
    int64_t state_l_stride;
    int64_t weight_c_stride;
    int64_t weight_l_stride;
    int conv_dim;
    int pad_slot_id;
};

__device__ __forceinline__ float bfLo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bfHi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }

__device__ __forceinline__ void cpAsync4(void* smem, void const* gmem)
{
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(s), "l"(gmem) : "memory");
}

__device__ __forceinline__ void cpAsync16(void* smem, void const* gmem)
{
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(gmem) : "memory");
}

__device__ __forceinline__ void cpAsyncWaitAll()
{
    asm volatile("cp.async.wait_all;" ::: "memory");
}

// causal_conv1d_update_kernel_sl1 math (width 4, no bias, silu): same order and intrinsics.
__device__ __forceinline__ float convSilu(float const (&w)[4], float const (&s)[4])
{
    float ov = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        ov = __fmaf_rn(w[i], s[i], ov);
    }
    return ov * __frcp_rn(1.0f + __expf(-ov));
}

template <int M, int BLOCK, int VAR, int MINB>
__global__ void __launch_bounds__(BLOCK, MINB) gemvKernel(uint8_t const* __restrict__ w,
    uint8_t const* __restrict__ sf, __nv_bfloat16 const* __restrict__ x, __nv_bfloat16* __restrict__ out, float g,
    int n, int k, ConvEpilogue ce)
{
    constexpr bool CONV = VAR != 0;
    int const tid = threadIdx.x;
    int const n0 = blockIdx.x * kGemvRows;
    int const kb = k / 2;
    int const ksf = k / 16;

    int4 wq[kGemvRows];
    uint32_t ws[kGemvRows];
#pragma unroll
    for (int r = 0; r < kGemvRows; ++r)
    {
        uint8_t const* p = w + static_cast<size_t>(n0 + r) * kb + tid * 16;
        asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.s32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(wq[r].x), "=r"(wq[r].y), "=r"(wq[r].z), "=r"(wq[r].w)
                     : "l"(p));
    }
#pragma unroll
    for (int r = 0; r < kGemvRows; ++r)
    {
        ws[r] = __ldg(reinterpret_cast<uint16_t const*>(sf + static_cast<size_t>(n0 + r) * ksf + tid * 2));
    }

    bool const convCta = CONV && n0 < ce.conv_dim;
    // VAR 1/4: thread tid < M * kGemvRows owns (row tid / 2, column n0 + tid % 2).
    bool const convThread = convCta && tid < M * kGemvRows;
    // VAR 3: smem staging. weight: 2 channels x 4 taps (16 B); state: M rows x 2 channels x 3 (12 B per row).
    __shared__ __align__(16) __nv_bfloat16 sW[kGemvRows * 4];
    __shared__ __align__(16) __nv_bfloat16 sS[(M > 0 ? M : 1) * kGemvRows * 3 + 2];
    __shared__ int sSlot[(M > 0 ? M : 1)];

    float cw[4] = {0.f, 0.f, 0.f, 0.f};
    uint32_t cwp[2] = {0u, 0u};
    if constexpr (VAR == 1)
    {
        if (convThread)
        {
            __nv_bfloat16 const* p = ce.weight + static_cast<int64_t>(n0 + tid % kGemvRows) * ce.weight_c_stride;
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                cw[i] = __bfloat162float(__ldg(p + i * ce.weight_l_stride));
            }
        }
    }
    if constexpr (VAR == 4)
    {
        if (convThread)
        {
            // Contiguous [conv_dim, 4] weight: 8 B per channel.
            uint2 const v = __ldg(reinterpret_cast<uint2 const*>(ce.weight + static_cast<int64_t>(n0 + tid % kGemvRows) * 4));
            cwp[0] = v.x;
            cwp[1] = v.y;
        }
    }
    if constexpr (VAR == 3)
    {
        if (convCta && tid == 0)
        {
            // Contiguous [conv_dim, 4] weight: the CTA's two channels are 16 contiguous bytes.
            cpAsync16(sW, ce.weight + static_cast<int64_t>(n0) * 4);
        }
    }
    // VAR 6/7/8: prefetch only (no registers live across the FMAs except the slot); the epilogue
    // loads after the FMAs (6/7: right after them, 8: after the warp reduction). 7 = L2, else L1.
    constexpr bool PF = VAR == 6 || VAR == 7 || VAR == 8 || VAR == 9;
    if constexpr (PF)
    {
        if (convCta && tid == 0)
        {
            void const* p = ce.weight + static_cast<int64_t>(n0) * 4;
            if constexpr (VAR == 7)
                asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
            else
                asm volatile("prefetch.global.L1 [%0];" ::"l"(p));
        }
    }

    asm volatile("griddepcontrol.wait;" ::: "memory");

    uint32_t xa[M][16];
#pragma unroll
    for (int m = 0; m < M; ++m)
    {
        int4 const* p = reinterpret_cast<int4 const*>(x + static_cast<size_t>(m) * k + tid * 32);
#pragma unroll
        for (int q = 0; q < 4; ++q)
        {
            int4 const v = p[q];
            xa[m][q * 4 + 0] = v.x;
            xa[m][q * 4 + 1] = v.y;
            xa[m][q * 4 + 2] = v.z;
            xa[m][q * 4 + 3] = v.w;
        }
    }

    int slot = -1;
    float s0f = 0.f, s1f = 0.f, s2f = 0.f;
    uint32_t sp01 = 0u, sp2 = 0u;
    if constexpr (VAR == 1)
    {
        if (convThread)
        {
            slot = ce.indices[tid / kGemvRows];
            if (slot != ce.pad_slot_id)
            {
                __nv_bfloat16 const* sp = ce.state + static_cast<int64_t>(slot) * ce.state_batch_stride
                    + static_cast<int64_t>(n0 + tid % kGemvRows) * ce.state_c_stride;
                s0f = __bfloat162float(sp[0]);
                s1f = __bfloat162float(sp[ce.state_l_stride]);
                s2f = __bfloat162float(sp[2 * ce.state_l_stride]);
            }
        }
    }
    if constexpr (VAR == 4)
    {
        if (convThread)
        {
            slot = ce.indices[tid / kGemvRows];
            if (slot != ce.pad_slot_id)
            {
                // Contiguous [.., conv_dim, 3] state.
                unsigned short const* sp = reinterpret_cast<unsigned short const*>(ce.state)
                    + static_cast<int64_t>(slot) * ce.state_batch_stride + static_cast<int64_t>(n0 + tid % kGemvRows) * 3;
                sp01 = static_cast<uint32_t>(sp[0]) | (static_cast<uint32_t>(sp[1]) << 16);
                sp2 = static_cast<uint32_t>(sp[2]);
            }
        }
    }
    if constexpr (PF)
    {
        if (convThread)
        {
            slot = ce.indices[tid / kGemvRows];
            if (slot != ce.pad_slot_id)
            {
                void const* p = ce.state + static_cast<int64_t>(slot) * ce.state_batch_stride
                    + static_cast<int64_t>(n0 + tid % kGemvRows) * 3;
                if constexpr (VAR == 7)
                    asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
                else
                    asm volatile("prefetch.global.L1 [%0];" ::"l"(p));
            }
        }
    }
    if constexpr (VAR == 3)
    {
        if (convCta && tid < M)
        {
            int const sl = ce.indices[tid];
            sSlot[tid] = sl;
            if (sl != ce.pad_slot_id)
            {
                // Row tid's two channels: 6 contiguous bf16 = 12 B, 4-B aligned (n0 even).
                __nv_bfloat16 const* sp = ce.state + static_cast<int64_t>(sl) * ce.state_batch_stride
                    + static_cast<int64_t>(n0) * 3;
#pragma unroll
                for (int i = 0; i < 3; ++i)
                {
                    cpAsync4(&sS[tid * 6 + 2 * i], sp + 2 * i);
                }
            }
        }
    }

    float acc[M][kGemvRows];
#pragma unroll
    for (int m = 0; m < M; ++m)
    {
#pragma unroll
        for (int r = 0; r < kGemvRows; ++r)
        {
            acc[m][r] = 0.f;
        }
    }

#pragma unroll
    for (int r = 0; r < kGemvRows; ++r)
    {
        float wf[32];
        uint32_t const* wv = reinterpret_cast<uint32_t const*>(&wq[r]);
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            uint32_t h[4];
            asm volatile(
                "{\n.reg .b8 b0, b1, b2, b3;\n"
                "mov.b32 {b0, b1, b2, b3}, %4;\n"
                "cvt.rn.f16x2.e2m1x2 %0, b0;\n"
                "cvt.rn.f16x2.e2m1x2 %1, b1;\n"
                "cvt.rn.f16x2.e2m1x2 %2, b2;\n"
                "cvt.rn.f16x2.e2m1x2 %3, b3;\n}\n"
                : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3])
                : "r"(wv[i]));
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                float2 const f = __half22float2(reinterpret_cast<__half2 const&>(h[j]));
                wf[i * 8 + j * 2] = f.x;
                wf[i * 8 + j * 2 + 1] = f.y;
            }
        }
        __nv_fp8_e4m3 s8;
        s8.__x = static_cast<__nv_fp8_storage_t>(ws[r] & 0xff);
        float const s0 = static_cast<float>(s8);
        s8.__x = static_cast<__nv_fp8_storage_t>((ws[r] >> 8) & 0xff);
        float const s1 = static_cast<float>(s8);
#pragma unroll
        for (int m = 0; m < M; ++m)
        {
            float p0 = 0.f;
            float p1 = 0.f;
#pragma unroll
            for (int l = 0; l < 8; ++l)
            {
                p0 = fmaf(__uint_as_float(xa[m][l] << 16), wf[2 * l], p0);
                p0 = fmaf(__uint_as_float(xa[m][l] & 0xffff0000u), wf[2 * l + 1], p0);
            }
#pragma unroll
            for (int l = 8; l < 16; ++l)
            {
                p1 = fmaf(__uint_as_float(xa[m][l] << 16), wf[2 * l], p1);
                p1 = fmaf(__uint_as_float(xa[m][l] & 0xffff0000u), wf[2 * l + 1], p1);
            }
            acc[m][r] = fmaf(p0, s0, acc[m][r]);
            acc[m][r] = fmaf(p1, s1, acc[m][r]);
        }
    }

    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

    // VAR 9: after the FMAs, load weight + state, fold the three state taps into a partial sum and
    // write the shifted state words 0/1 before the reduction; only tap 3 (x) remains for the tail.
    float part = 0.f;
    float w3 = 0.f;
    if constexpr (VAR == 9)
    {
        if (convThread && slot != ce.pad_slot_id)
        {
            __nv_bfloat16 const* p = ce.weight + static_cast<int64_t>(n0 + tid % kGemvRows) * ce.weight_c_stride;
            float wv0 = __bfloat162float(__ldg(p));
            float wv1 = __bfloat162float(__ldg(p + ce.weight_l_stride));
            float wv2 = __bfloat162float(__ldg(p + 2 * ce.weight_l_stride));
            w3 = __bfloat162float(__ldg(p + 3 * ce.weight_l_stride));
            __nv_bfloat16* sp = ce.state + static_cast<int64_t>(slot) * ce.state_batch_stride
                + static_cast<int64_t>(n0 + tid % kGemvRows) * ce.state_c_stride;
            __nv_bfloat16 const b0 = sp[0];
            __nv_bfloat16 const b1 = sp[ce.state_l_stride];
            __nv_bfloat16 const b2 = sp[2 * ce.state_l_stride];
            part = __fmaf_rn(wv0, __bfloat162float(b0), part);
            part = __fmaf_rn(wv1, __bfloat162float(b1), part);
            part = __fmaf_rn(wv2, __bfloat162float(b2), part);
            sp[0] = __float2bfloat16(__bfloat162float(b1));
            sp[ce.state_l_stride] = __float2bfloat16(__bfloat162float(b2));
        }
    }

    if constexpr (VAR == 2 || VAR == 6 || VAR == 7)
    {
        if (convThread)
        {
            __nv_bfloat16 const* p = ce.weight + static_cast<int64_t>(n0 + tid % kGemvRows) * ce.weight_c_stride;
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                cw[i] = __bfloat162float(__ldg(p + i * ce.weight_l_stride));
            }
            if constexpr (VAR == 2)
            {
                slot = ce.indices[tid / kGemvRows];
            }
            if (slot != ce.pad_slot_id)
            {
                __nv_bfloat16 const* sp = ce.state + static_cast<int64_t>(slot) * ce.state_batch_stride
                    + static_cast<int64_t>(n0 + tid % kGemvRows) * ce.state_c_stride;
                s0f = __bfloat162float(sp[0]);
                s1f = __bfloat162float(sp[ce.state_l_stride]);
                s2f = __bfloat162float(sp[2 * ce.state_l_stride]);
            }
        }
    }

    constexpr int kWarps = BLOCK / 32;
    __shared__ float red[kWarps][M * kGemvRows];
    int const lane = tid & 31;
    int const warp = tid >> 5;
#pragma unroll
    for (int m = 0; m < M; ++m)
    {
#pragma unroll
        for (int r = 0; r < kGemvRows; ++r)
        {
            float v = acc[m][r];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1)
            {
                v += __shfl_xor_sync(0xffffffffu, v, o);
            }
            if (lane == 0)
            {
                red[warp][m * kGemvRows + r] = v;
            }
        }
    }
    if constexpr (VAR == 3)
    {
        cpAsyncWaitAll();
    }
    if constexpr (VAR == 8)
    {
        if (convThread)
        {
            __nv_bfloat16 const* p = ce.weight + static_cast<int64_t>(n0 + tid % kGemvRows) * ce.weight_c_stride;
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                cw[i] = __bfloat162float(__ldg(p + i * ce.weight_l_stride));
            }
            if (slot != ce.pad_slot_id)
            {
                __nv_bfloat16 const* sp = ce.state + static_cast<int64_t>(slot) * ce.state_batch_stride
                    + static_cast<int64_t>(n0 + tid % kGemvRows) * ce.state_c_stride;
                s0f = __bfloat162float(sp[0]);
                s1f = __bfloat162float(sp[ce.state_l_stride]);
                s2f = __bfloat162float(sp[2 * ce.state_l_stride]);
            }
        }
    }
    __syncthreads();
    if constexpr (!CONV)
    {
        for (int i = tid; i < M * kGemvRows; i += BLOCK)
        {
            float v = 0.f;
#pragma unroll
            for (int j = 0; j < kWarps; ++j)
            {
                v += red[j][i];
            }
            int const m = i / kGemvRows;
            int const r = i % kGemvRows;
            out[static_cast<size_t>(m) * n + n0 + r] = __float2bfloat16(v * g);
        }
    }
    else
    {
        if (tid < M * kGemvRows)
        {
            float v = 0.f;
#pragma unroll
            for (int j = 0; j < kWarps; ++j)
            {
                v += red[j][tid];
            }
            int const m = tid / kGemvRows;
            int const r = tid % kGemvRows;
            __nv_bfloat16 o = __float2bfloat16(v * g);
            if constexpr (VAR == 9)
            {
                if (convCta && slot != ce.pad_slot_id)
                {
                    float ov = __fmaf_rn(w3, __bfloat162float(o), part);
                    ov = ov * __frcp_rn(1.0f + __expf(-ov));
                    __nv_bfloat16* sp = ce.state + static_cast<int64_t>(slot) * ce.state_batch_stride
                        + static_cast<int64_t>(n0 + r) * ce.state_c_stride;
                    sp[2 * ce.state_l_stride] = o;
                    o = __float2bfloat16(ov);
                }
            }
            else if (convCta)
            {
                float wv[4];
                float s[4];
                bool live;
                if constexpr (VAR == 3)
                {
                    live = sSlot[m] != ce.pad_slot_id;
                    if (live)
                    {
                        slot = sSlot[m];
#pragma unroll
                        for (int i = 0; i < 4; ++i)
                        {
                            wv[i] = __bfloat162float(sW[r * 4 + i]);
                        }
#pragma unroll
                        for (int i = 0; i < 3; ++i)
                        {
                            s[i] = __bfloat162float(sS[m * 6 + r * 3 + i]);
                        }
                    }
                }
                else if constexpr (VAR == 4)
                {
                    live = slot != ce.pad_slot_id;
                    wv[0] = bfLo(cwp[0]);
                    wv[1] = bfHi(cwp[0]);
                    wv[2] = bfLo(cwp[1]);
                    wv[3] = bfHi(cwp[1]);
                    s[0] = bfLo(sp01);
                    s[1] = bfHi(sp01);
                    s[2] = bfLo(sp2);
                }
                else
                {
                    live = slot != ce.pad_slot_id;
#pragma unroll
                    for (int i = 0; i < 4; ++i)
                    {
                        wv[i] = cw[i];
                    }
                    s[0] = s0f;
                    s[1] = s1f;
                    s[2] = s2f;
                }
                if (live)
                {
                    s[3] = __bfloat162float(o);
                    o = __float2bfloat16(convSilu(wv, s));
                    __nv_bfloat16* sp = ce.state + static_cast<int64_t>(slot) * ce.state_batch_stride
                        + static_cast<int64_t>(n0 + r) * ce.state_c_stride;
#pragma unroll
                    for (int i = 0; i < 3; ++i)
                    {
                        sp[i * ce.state_l_stride] = __float2bfloat16(s[i + 1]);
                    }
                }
            }
            out[static_cast<size_t>(m) * n + n0 + r] = o;
        }
    }
}

template <typename Kern>
void pdlLaunch(Kern kern, dim3 grid, dim3 block, uint8_t const* w, uint8_t const* sf, __nv_bfloat16 const* x,
    __nv_bfloat16* o, float g, int n, int k, ConvEpilogue ce)
{
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
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, w, sf, x, o, g, n, k, ce));
}

template <int M, int VAR, int MINB>
void* kptr()
{
    return reinterpret_cast<void*>(&gemvKernel<M, 160, VAR, MINB>);
}

// var = VAR + 10 * MINB (MINB 0 = plain __launch_bounds__(160)).
void* kernelFor(int m, int code)
{
    int const var = code % 10;
    int const minb = code / 10;
#define KM(MM, VV) \
    switch (minb) \
    { \
    case 0: return kptr<MM, VV, 1>(); \
    case 3: return kptr<MM, VV, 3>(); \
    case 4: return kptr<MM, VV, 4>(); \
    case 5: return kptr<MM, VV, 5>(); \
    case 6: return kptr<MM, VV, 6>(); \
    case 7: return kptr<MM, VV, 7>(); \
    case 8: return kptr<MM, VV, 8>(); \
    case 9: return kptr<MM, VV, 9>(); \
    case 10: return kptr<MM, VV, 10>(); \
    }
#define KV(VV) \
    switch (m) \
    { \
    case 1: KM(1, VV) break; \
    case 2: KM(2, VV) break; \
    case 3: KM(3, VV) break; \
    }
    switch (var)
    {
    case 0: KV(0) break;
    case 1: KV(1) break;
    case 2: KV(2) break;
    case 3: KV(3) break;
    case 4: KV(4) break;
    case 6: KV(6) break;
    case 7: KV(7) break;
    case 8: KV(8) break;
    case 9: KV(9) break;
    }
#undef KV
#undef KM
    return nullptr;
}

}  // namespace

torch::Tensor run(torch::Tensor x, torch::Tensor w, torch::Tensor sf, double g, torch::Tensor state,
    torch::Tensor cw, torch::Tensor idx, int64_t conv_dim, int64_t pad, int64_t var)
{
    int m = x.size(0), k = x.size(1), n = w.size(0);
    TORCH_CHECK(k == 5120, "k 5120 only");
    auto out = torch::empty({m, n}, x.options());
    ConvEpilogue ce{};
    if (var % 10)
    {
        TORCH_CHECK(cw.is_contiguous() && state.stride(2) == 1 && state.stride(1) == 3, "contiguous conv layout");
        ce.state = reinterpret_cast<__nv_bfloat16*>(state.data_ptr());
        ce.weight = reinterpret_cast<__nv_bfloat16 const*>(cw.data_ptr());
        ce.indices = idx.data_ptr<int>();
        ce.state_batch_stride = state.stride(0);
        ce.state_c_stride = state.stride(1);
        ce.state_l_stride = state.stride(2);
        ce.weight_c_stride = cw.stride(0);
        ce.weight_l_stride = cw.stride(1);
        ce.conv_dim = static_cast<int>(conv_dim);
        ce.pad_slot_id = static_cast<int>(pad);
    }
    void* kp = kernelFor(m, static_cast<int>(var));
    TORCH_CHECK(kp != nullptr, "bad m/var");
    auto const* wp = w.data_ptr<uint8_t>();
    auto const* sp = sf.data_ptr<uint8_t>();
    auto const* xp = reinterpret_cast<__nv_bfloat16 const*>(x.data_ptr());
    auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    using KT = void (*)(uint8_t const*, uint8_t const*, __nv_bfloat16 const*, __nv_bfloat16*, float, int, int, ConvEpilogue);
    pdlLaunch(reinterpret_cast<KT>(kp), dim3(n / kGemvRows), dim3(160), wp, sp, xp, op, static_cast<float>(g), n, k, ce);
    return out;
}

std::vector<int64_t> attrs(int64_t m, int64_t var)
{
    void* kp = kernelFor(static_cast<int>(m), static_cast<int>(var));
    TORCH_CHECK(kp != nullptr, "bad m/var");
    cudaFuncAttributes a;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&a, kp));
    int blocks = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kp, 160, 0));
    return {a.numRegs, a.localSizeBytes, a.sharedSizeBytes, blocks};
}
"""

CPP = ("torch::Tensor run(torch::Tensor x, torch::Tensor w, torch::Tensor sf, double g, torch::Tensor state, "
       "torch::Tensor cw, torch::Tensor idx, int64_t conv_dim, int64_t pad, int64_t var);\n"
       "std::vector<int64_t> attrs(int64_t m, int64_t var);")


def build_ext(log=print):
    from torch.utils.cpp_extension import load_inline
    bd = "/tmp/b118_gemv_conv_kern_build"
    os.makedirs(bd, exist_ok=True)
    log("compiling extension ...")
    ext = load_inline(name="b118_gemv_conv_kern", cpp_sources=CPP, cuda_sources=CU, functions=["run", "attrs"],
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a", "-std=c++20",
                                         "-lineinfo"],
                      build_directory=bd, verbose=os.environ.get("B118_VERBOSE", "0") == "1")
    log("extension ready")
    return ext
