/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cutlass/numeric_conversion.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaFp8Utils.h"
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/cudaCoreGemmNVFP4.h"
#include <cub/cub.cuh>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace cuda_core_gemm_nvfp4
{
// Offset of scale-factor column col_idx of row row_idx in the 128x4 swizzled SF layout
// (same math as get_sf_out_offset_128x4 without batching). col_idx is even here, so the SF
// pair (col_idx, col_idx + 1) is contiguous and can be read with one 2-byte load.
__device__ __forceinline__ int swizzledSfOffset(int row_idx, int col_idx, int num_sf_tiles_k)
{
    int const tile_offset = ((row_idx / 128) * num_sf_tiles_k + col_idx / 4) * 512;
    return tile_offset + (row_idx % 32) * 16 + ((row_idx % 128) / 32) * 4 + col_idx % 4;
}

// ACT_SF_SWIZZLED: scale_a is in the 128x4 swizzled layout written by fp4_quantize
// (isSfSwizzledLayout=True) instead of the linear [m, k / 16] layout. Both read the same bytes,
// so the output is identical.
template <typename InputType, typename OutputType, typename ScaleType, SizeType32 TILE_M, SizeType32 TILE_N,
    SizeType32 BLOCK_SIZE, bool ACT_SF_SWIZZLED>
__device__ void cudaCoreGemmImpl(InputType const* __restrict__ act, InputType const* __restrict__ weight,
    ScaleType const* __restrict__ scale_a, ScaleType const* __restrict__ scale_w, float const alpha,
    OutputType* __restrict__ output, OutputType const* __restrict__ bias, SizeType32 m, SizeType32 n, SizeType32 k)
{
    using VecType = int4;
    static constexpr bool PRE_PDL_WEIGHT_LOADS = TILE_M > 1;

    using ScaleVecType = __nv_fp8x2_e4m3;
    using CvtInputType = typename tensorrt_llm::kernels::cutlass_kernels::TllmToCutlassTypeAdapter<InputType>::type;
    static constexpr SizeType32 step_k = static_cast<SizeType32>(128 / cutlass::sizeof_bits<CvtInputType>::value);
    static constexpr SizeType32 nvfp4_scale_granularity = 16;
    static constexpr SizeType32 step_k_scale = step_k / nvfp4_scale_granularity;
    static constexpr SizeType32 tile_k = step_k * BLOCK_SIZE;
    auto tile_id_m = static_cast<SizeType32>(blockIdx.y * TILE_M);
    auto tile_id_n = static_cast<SizeType32>(blockIdx.x * TILE_N);
    auto tid = static_cast<SizeType32>(threadIdx.x);
    float tile_a[step_k];
    float tile_w[TILE_N * step_k];
    float tile_a_scale[step_k_scale];
    float tile_w_scale[TILE_N * step_k_scale];
    float acc[TILE_M * TILE_N];

    static_assert(step_k % 4 == 0);
    using Converter = cutlass::NumericArrayConverter<float, CvtInputType, 8>;
    using CvtSrcType = typename Converter::source_type;
    using CvtResType = typename Converter::result_type;
    static constexpr SizeType32 k_cvt_count = static_cast<SizeType32>(sizeof(VecType) / sizeof(CvtSrcType));

#pragma unroll
    for (SizeType32 i = 0; i < TILE_M * TILE_N; ++i)
    {
        acc[i] = 0;
    }
    act += tile_id_m * k / 2;
    weight += tile_id_n * k / 2;
    output += tile_id_m * n + tile_id_n;
    OutputType const* __restrict__ bias_tile = (bias != nullptr) ? bias + tile_id_n : nullptr;

    if constexpr (!ACT_SF_SWIZZLED)
    {
        scale_a += tile_id_m * k / nvfp4_scale_granularity;
    }

    int const num_cols_sf = k / nvfp4_scale_granularity;
    int const num_sf_tiles_k = (num_cols_sf + 4 - 1) / 4;

    // Raw weight and weight-SF registers of the current K step. Weights and their scale factors are
    // constant (never written by a preceding kernel), so they may be loaded before the PDL wait.
    VecType tile_w_quantized[TILE_N];
    ScaleVecType tile_w_scale_fp8x2[TILE_N];
    auto load_weight_step = [&](SizeType32 idx_k)
    {
#pragma unroll
        for (SizeType32 j = 0; j < TILE_N; ++j)
        {
            tile_w_quantized[j] = reinterpret_cast<VecType const*>(weight + (j * k + idx_k) / 2)[0];
        }
#pragma unroll
        for (SizeType32 j = 0; j < TILE_N; ++j)
        {
            int const dst_idx = swizzledSfOffset(tile_id_n + j, idx_k / nvfp4_scale_granularity, num_sf_tiles_k);
            tile_w_scale_fp8x2[j] = reinterpret_cast<ScaleVecType const*>(scale_w + dst_idx)[0];
        }
    };

    auto convert_weight_step = [&]()
    {
#pragma unroll
        for (SizeType32 j = 0; j < TILE_N; ++j)
        {
#pragma unroll
            for (SizeType32 cvt_idx = 0; cvt_idx < k_cvt_count; ++cvt_idx)
            {
                reinterpret_cast<CvtResType*>(tile_w)[j * k_cvt_count + cvt_idx]
                    = Converter::convert(reinterpret_cast<CvtSrcType*>(&tile_w_quantized[j])[cvt_idx]);
            }
        }
#pragma unroll
        for (SizeType32 j = 0; j < TILE_N; ++j)
        {
            const char2 tmp = reinterpret_cast<char2 const&>(tile_w_scale_fp8x2[j]);
            tile_w_scale[j * step_k_scale + 0] = static_cast<float>(reinterpret_cast<__nv_fp8_e4m3 const&>(tmp.x));
            tile_w_scale[j * step_k_scale + 1] = static_cast<float>(reinterpret_cast<__nv_fp8_e4m3 const&>(tmp.y));
        }
    };

    auto act_fma_step = [&](SizeType32 idx_k)
    {
#pragma unroll
        for (SizeType32 i = 0; i < TILE_M; ++i)
        {
            auto tile_a_quantized = reinterpret_cast<VecType const*>(act + (i * k + idx_k) / 2)[0];
#pragma unroll
            for (SizeType32 cvt_idx = 0; cvt_idx < k_cvt_count; ++cvt_idx)
            {
                reinterpret_cast<CvtResType*>(tile_a)[cvt_idx]
                    = Converter::convert(reinterpret_cast<CvtSrcType*>(&tile_a_quantized)[cvt_idx]);
            }
            int const a_sf_idx = ACT_SF_SWIZZLED
                ? swizzledSfOffset(tile_id_m + i, idx_k / nvfp4_scale_granularity, num_sf_tiles_k)
                : (i * k + idx_k) / nvfp4_scale_granularity;
            auto tile_a_scale_fp8x2 = reinterpret_cast<ScaleVecType const*>(scale_a + a_sf_idx)[0];
            const char2 tmp = reinterpret_cast<char2 const&>(tile_a_scale_fp8x2);
            tile_a_scale[0] = static_cast<float>(reinterpret_cast<__nv_fp8_e4m3 const&>(tmp.x));
            tile_a_scale[1] = static_cast<float>(reinterpret_cast<__nv_fp8_e4m3 const&>(tmp.y));
#pragma unroll
            for (SizeType32 j = 0; j < TILE_N; ++j)
            {
#pragma unroll
                for (SizeType32 l = 0; l < step_k; ++l)
                {
                    acc[i * TILE_N + j] = fma(alpha * tile_a[l] * tile_a_scale[l / nvfp4_scale_granularity],
                        tile_w[j * step_k + l] * tile_w_scale[j * step_k_scale + l / nvfp4_scale_granularity],
                        acc[i * TILE_N + j]);
                }
            }
        }
    };

    SizeType32 const first_idx_k = tid * step_k;
    // K-loop structure, chosen per TILE_M from cold-L2 CUDA-graph measurements on SM120
    // (gate_up 34816x5120 and down 5120x17408). Both orders run the same FMAs in the same order,
    // so the output is bitwise identical.
    // - TILE_M == 1: plain loop after the PDL wait. Peeling the first step raises the register
    //   count (40 -> 48), lowers occupancy and measured ~2% slower.
    // - TILE_M >= 2: the first K step's weight and weight-SF loads (constant data) are issued
    //   before the PDL wait so they overlap the producer's tail, and that step is peeled so the
    //   remaining loop has no branch (M=2 gate_up ~5% faster than the plain loop).
    if constexpr (!PRE_PDL_WEIGHT_LOADS)
    {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        cudaGridDependencySynchronize();
#endif
        for (SizeType32 idx_k = first_idx_k; idx_k < k; idx_k += tile_k)
        {
            load_weight_step(idx_k);
            convert_weight_step();
            act_fma_step(idx_k);
        }
    }
    else
    {
        // First K step peeled; the remaining loop has no branch.
        if (first_idx_k < k)
        {
            load_weight_step(first_idx_k);
        }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        cudaGridDependencySynchronize();
#endif
        if (first_idx_k < k)
        {
            convert_weight_step();
            act_fma_step(first_idx_k);
        }
        for (SizeType32 idx_k = first_idx_k + tile_k; idx_k < k; idx_k += tile_k)
        {
            load_weight_step(idx_k);
            convert_weight_step();
            act_fma_step(idx_k);
        }
    }

    typedef cub::WarpReduce<float> WarpReduce;

    static constexpr SizeType32 warp_size = 32;
    static constexpr SizeType32 warp_num = BLOCK_SIZE / warp_size;
    SizeType32 warp_id = tid / warp_size, lane_id = tid % warp_size;
    __shared__ float shmem[TILE_M * TILE_N * warp_num];
    __shared__ typename WarpReduce::TempStorage temp_storage[warp_num];
#pragma unroll
    for (SizeType32 mi = 0; mi < TILE_M; ++mi)
    {
#pragma unroll
        for (SizeType32 ni = 0; ni < TILE_N; ++ni)
        {
            float val = WarpReduce(temp_storage[warp_id]).Sum(acc[mi * TILE_N + ni]);
            if (lane_id == 0)
            {
                shmem[mi * TILE_N + ni + warp_id * TILE_M * TILE_N] = val;
            }
        }
    }
    __syncthreads();
    for (SizeType32 ii = tid; ii < TILE_M * TILE_N; ii += BLOCK_SIZE)
    {
        SizeType32 mid = ii / TILE_N, nid = ii % TILE_N;
        float val = 0;
#pragma unroll
        for (SizeType32 jj = 0; jj < warp_num; ++jj)
        {
            val += shmem[jj * TILE_M * TILE_N + ii];
        }
        if (bias_tile != nullptr)
        {
            val += static_cast<float>(bias_tile[nid]);
        }
        output[mid * n + nid] = static_cast<OutputType>(val);
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <typename InputType, typename OutputType, typename ScaleType, SizeType32 TILE_M, SizeType32 TILE_N,
    SizeType32 BLOCK_SIZE, bool ACT_SF_SWIZZLED>
__global__ void cudaCoreGemmFp4(InputType const* __restrict__ act, InputType const* __restrict__ weight,
    ScaleType const* __restrict__ scale_a, ScaleType const* __restrict__ scale_w, float const* alpha_ptr,
    OutputType* __restrict__ output, OutputType const* __restrict__ bias, SizeType32 m, SizeType32 n, SizeType32 k)
{
    float alpha = alpha_ptr[0];
    cudaCoreGemmImpl<InputType, OutputType, ScaleType, TILE_M, TILE_N, BLOCK_SIZE, ACT_SF_SWIZZLED>(
        reinterpret_cast<InputType const*>(act), reinterpret_cast<InputType const*>(weight),
        reinterpret_cast<ScaleType const*>(scale_a), reinterpret_cast<ScaleType const*>(scale_w), alpha,
        reinterpret_cast<OutputType*>(output), reinterpret_cast<OutputType const*>(bias), m, n, k);
}

template <typename InputType, typename OutputType, typename ScaleType, SizeType32 TILE_M, SizeType32 TILE_N,
    SizeType32 BLOCK_SIZE>
void cudaCoreGemmKernel(Params const& params, cudaStream_t stream)
{
    dim3 block(BLOCK_SIZE);
    // N rides grid.x: its tile count is unbounded (a quantized LM head can be hundreds of
    // thousands wide) and only grid.x allows more than 65535 blocks. M is safe on grid.y
    // because it never exceeds cudaCoreGemmTemplateMaxM.
    dim3 grid(params.n / TILE_N, params.m / TILE_M);

    using KernelFn = void (*)(InputType const*, InputType const*, ScaleType const*, ScaleType const*, float const*,
        OutputType*, OutputType const*, SizeType32, SizeType32, SizeType32);
    KernelFn const cudaCoreGemmFp4Kernel = params.act_sf_swizzled
        ? &cudaCoreGemmFp4<InputType, OutputType, ScaleType, TILE_M, TILE_N, BLOCK_SIZE, true>
        : &cudaCoreGemmFp4<InputType, OutputType, ScaleType, TILE_M, TILE_N, BLOCK_SIZE, false>;

    if (tensorrt_llm::common::getEnvEnablePDL())
    {
        TLLM_LOG_DEBUG("Enable PDL in fp8_gemm_plugin");
        cudaLaunchConfig_t kernelConfig = {0};
        kernelConfig.gridDim = grid;
        kernelConfig.blockDim = block;
        kernelConfig.dynamicSmemBytes = 0;
        kernelConfig.stream = stream;

        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute[0].val.programmaticStreamSerializationAllowed = 1;
        kernelConfig.attrs = attribute;
        kernelConfig.numAttrs = 1;

        if (params.scale_a && params.scale_b && params.alpha_ptr)
        {
            TLLM_CUDA_CHECK(cudaLaunchKernelEx(&kernelConfig, cudaCoreGemmFp4Kernel,
                reinterpret_cast<InputType const*>(params.act), reinterpret_cast<InputType const*>(params.weight),
                reinterpret_cast<ScaleType const*>(params.scale_a), reinterpret_cast<ScaleType const*>(params.scale_b),
                params.alpha_ptr, reinterpret_cast<OutputType*>(params.output),
                reinterpret_cast<OutputType const*>(params.bias), params.m, params.n, params.k));
        }
    }
    else
    {
        if (params.scale_a && params.scale_b && params.alpha_ptr)
        {
            cudaCoreGemmFp4Kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<InputType const*>(params.act), reinterpret_cast<InputType const*>(params.weight),
                reinterpret_cast<ScaleType const*>(params.scale_a), reinterpret_cast<ScaleType const*>(params.scale_b),
                params.alpha_ptr, reinterpret_cast<OutputType*>(params.output),
                reinterpret_cast<OutputType const*>(params.bias), params.m, params.n, params.k);
        }
    }
}

template <typename InputType, typename OutputType, typename ScaleType, int TILE_M, int TILE_N, int BLOCK_SIZE>
bool cudaCoreGemmTemplateCaller(Params const& params, cudaStream_t stream)
{
    constexpr int cudaCoreGemmTemplateMaxM = 16;
    if (params.m == TILE_M)
    {
        cudaCoreGemmKernel<InputType, OutputType, ScaleType, TILE_M, TILE_N, BLOCK_SIZE>(params, stream);
        return true;
    }
    if constexpr (TILE_M < cudaCoreGemmTemplateMaxM)
    {
        return cudaCoreGemmTemplateCaller<InputType, OutputType, ScaleType, TILE_M + 1, TILE_N, BLOCK_SIZE>(
            params, stream);
    }
    return false;
}

template <typename InputType, typename OutputType, typename ScaleType = float>
bool cudaCoreGemmLauncher(Params const& params, cudaStream_t stream)
{
    return cudaCoreGemmTemplateCaller<InputType, OutputType, ScaleType, 1, 2, 128>(params, stream);
}

bool cudaCoreGemmDispatcher(Params const& params, cudaStream_t stream)
{
    bool dispatched = true;
    if (params.n % 2 != 0)
    {
        dispatched = false;
    }
    else if (params.inputType == CUDA_R_8U)
    {
        if (params.k % 16 != 0)
        {
            // Expect k % 16 == 0 for nvfp4 scaling granularity
            dispatched = false;
        }
        else if (params.outputType == CUDA_R_16F)
        {
            dispatched = cudaCoreGemmLauncher<__nv_fp4_e2m1, half, __nv_fp8_e4m3>(params, stream);
        }
        else if (params.outputType == CUDA_R_16BF)
        {
            dispatched = cudaCoreGemmLauncher<__nv_fp4_e2m1, __nv_bfloat16, __nv_fp8_e4m3>(params, stream);
        }
        else if (params.outputType == CUDA_R_32F)
        {
            dispatched = cudaCoreGemmLauncher<__nv_fp4_e2m1, float, __nv_fp8_e4m3>(params, stream);
        }
        else
        {
            dispatched = false;
        }
    }
    else
    {
        dispatched = false;
    }

    if (!dispatched)
    {
        TLLM_LOG_WARNING(
            "tensorrt_llm::kernels::cuda_core_gemm_nvfp4::cudaCoreGemmDispatcher [NOT DISPATCHED], inputType=%d, "
            "outputType=%d, "
            "m=%d, "
            "n=%d, k=%d",
            params.inputType, params.outputType, params.m, params.n, params.k);
    }
    return dispatched;
}

} // namespace cuda_core_gemm_nvfp4
} // namespace kernels

TRTLLM_NAMESPACE_END
