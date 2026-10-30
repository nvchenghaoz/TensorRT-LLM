/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Launcher of the FLUX.3 FP8 attention on SM120 (see flux3FmhaSm120.h). Built with the generated SM120 context FMHA
// kernels (contextFusedMultiHeadAttention/CMakeLists.txt) for their include paths and definitions, and with
// --use_fast_math: the outputs require the softmax's exp2f / expf and the normalizer's 1 / sum as the fast-math
// instructions (ex2.approx.ftz, approximate reciprocal).

#include <fmha/kernel_traits.h>
#include <fused_multihead_attention.h>
#include <fused_multihead_attention_kernel.h>
#include <fused_multihead_flash_attention_kernel_noloop_tiled.h>

#include "flux3FmhaG3Kernel.cuh"

#include "flux3FmhaSm120.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaDriverWrapper.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <climits>
#include <cstring>

namespace flux3FmhaSm120
{

namespace g3 = ::fused_multihead_attention::flux3;

// The traits of the fmha_v2 tiled flash kernel
// fmha_v2_flash_attention_e4m3_fp32_64_64_S_q_k_v_128x128_output_bf16_sm120 (setup.py flash_attention_kernel_template,
// Kernel_traits_nl_tiled: kv step 64, head 128, q step 64, 4x1x1 warps; flags 0x1007 = LDGSTS for Q/K/V + granular
// tiling, 0x200 = no loop; padding mask (2); BF16 output).
using Kt = fmha::Kernel_traits_v2_q_k_v<fmha::Ada_qmma_e4m3_fp32_traits, 64, 128, 128, 64, 4, 1, 1, 0x1007u | 0x200u, 2,
    true, fmha::bf16_t, 0, 0, 0>;

using Layout = g3::Layout<Kt>;

constexpr int kHeadDim = 128;

template <bool NVFP4_OUT>
__global__ __launch_bounds__(g3::THREADS, 1) void fp8AttentionSm120Kernel(
    const __grid_constant__ bert::Fused_multihead_attention_params_v2 params, const __grid_constant__ CUtensorMap tmK,
    const __grid_constant__ CUtensorMap tmV, const __grid_constant__ g3::Nvfp4Out nfo)
{
    g3::device_flash_attention_g3<Kt, NVFP4_OUT>(params, &tmK, &tmV, nfo);
}

uint32_t floatBits(float x)
{
    uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    return bits;
}

// The K map: [B, S, H, 128] E4M3 bytes, box {64 B of head h, 1 head, 64 rows, 1 batch}, SWIZZLE_64B (the Smem_tile_k
// layout of one 64-column chunk).
void makeKMap(CUtensorMap* map, void const* ptr, int b, int s, int h, int64_t rowStrideBytes)
{
    cuuint64_t dims[4] = {(cuuint64_t) kHeadDim, (cuuint64_t) h, (cuuint64_t) s, (cuuint64_t) b};
    cuuint64_t strides[3]
        = {(cuuint64_t) kHeadDim, (cuuint64_t) rowStrideBytes, (cuuint64_t) rowStrideBytes * (cuuint64_t) s};
    cuuint32_t box[4] = {64, 1, 64, 1};
    cuuint32_t elemStrides[4] = {1, 1, 1, 1};
    TLLM_CU_CHECK(tensorrt_llm::common::CUDADriverWrapper::getInstance()->cuTensorMapEncodeTiled(map,
        CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, const_cast<void*>(ptr), dims, strides, box, elemStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

// The V map whose box lands key k of a 64-key step at smem row swap23(k). The box is {128 B of head h, 4 keys (lo),
// 2 (key bit 3), 2 (key bit 2), 4 groups of 16 keys}: smem row lo + 4 * b3 + 8 * b2 + 16 * g holds key
// lo + 8 * b3 + 4 * b2 + 16 * g. Heads fold into dim 0 (head stride == 128 B) and batches into the 16-key-group dim
// (batch stride == S rows), so the map needs S % 16 == 0 and, for B > 1, S % 64 == 0 (no step crosses a batch: keys
// past S then read as zero-filled out-of-bounds rows).
void makeVMap(CUtensorMap* map, void const* ptr, int b, int s, int h, int64_t rowStrideBytes)
{
    cuuint64_t const row = (cuuint64_t) rowStrideBytes;
    cuuint64_t dims[5] = {(cuuint64_t) h * (cuuint64_t) kHeadDim, 4, 2, 2, (cuuint64_t) b * (cuuint64_t) (s / 16)};
    cuuint64_t strides[4] = {row, 8 * row, 4 * row, 16 * row};
    cuuint32_t box[5] = {128, 4, 2, 2, 4};
    cuuint32_t elemStrides[5] = {1, 1, 1, 1, 1};
    TLLM_CU_CHECK(tensorrt_llm::common::CUDADriverWrapper::getInstance()->cuTensorMapEncodeTiled(map,
        CU_TENSOR_MAP_DATA_TYPE_UINT8, 5, const_cast<void*>(ptr), dims, strides, box, elemStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

// The fmha_v2 params of a fixed-stride BSHD launch (no cu_seqlens), as the fmha_v2 runner sets them for SEPARATE_Q_K_V
// E4M3 -> BF16 with scale_bmm1 = scale_softmax = scale_bmm2 = 1 and the device scales.
bert::Fused_multihead_attention_params_v2 makeParams(tensorrt_llm::kernels::flux3::Fp8AttentionParams const& args)
{
    bert::Fused_multihead_attention_params_v2 params{};
    int64_t const rowBytes = int64_t(args.h) * kHeadDim; // E4M3
    params.o_ptr = args.o;
    params.o_stride_in_bytes = rowBytes * 2; // BF16
    params.q_ptr = const_cast<void*>(args.q);
    params.k_ptr = const_cast<void*>(args.k);
    params.v_ptr = const_cast<void*>(args.v);
    params.q_stride_in_bytes = rowBytes;
    params.k_stride_in_bytes = rowBytes;
    params.v_stride_in_bytes = rowBytes;
    params.packed_mask_ptr = nullptr;
    params.packed_mask_stride_in_bytes = (int64_t(args.s) + fmha::FLASH_ATTEN_MASK_N_ALIGNMENT - 1)
        / fmha::FLASH_ATTEN_MASK_N_ALIGNMENT * fmha::FLASH_ATTEN_MASK_N_ALIGNMENT / 8;
    params.attention_sinks = nullptr;
    params.softmax_stats_ptr = nullptr;
    params.softmax_stats_stride_in_bytes = int64_t(args.h) * 2 * sizeof(float);
    params.b = args.b;
    params.h = args.h;
    params.s = args.s;
    params.d = kHeadDim;
    params.dv = kHeadDim;
    params.num_grouped_heads = 1;
    params.sliding_window_size = INT_MAX;
    params.log2_chunked_attention_size = 0;
    params.cu_q_seqlens = nullptr;
    params.cu_kv_seqlens = nullptr;
    params.cu_mask_rows = nullptr;
    params.scale_bmm1 = floatBits(1.f);
    params.scale_bmm1_d = reinterpret_cast<uint32_t*>(const_cast<float*>(args.scale_bmm1));
    // The softmax packs P without multiplying by scale_softmax (flux3FmhaSoftmax.cuh): it must be exactly 1.
    params.scale_softmax = floatBits(1.f);
    params.scale_bmm2 = floatBits(1.f);
    params.scale_bmm2_d = reinterpret_cast<uint32_t*>(const_cast<float*>(args.scale_bmm2));
    params.softcapping_scale_bmm1 = 0;
    params.h_kv = args.h;
    params.h_q_per_kv = 1;
    params.has_alibi = false;
    params.alibi_params = fmha::AlibiParams(args.h);
    params.is_s_padded = true;
    return params;
}

template <bool NVFP4_OUT>
void setSmemAttribute()
{
    constexpr int kMaxDevices = 64;
    static bool sDone[kMaxDevices] = {};
    int device = 0;
    TLLM_CUDA_CHECK(cudaGetDevice(&device));
    TLLM_CHECK_WITH_INFO(device >= 0 && device < kMaxDevices, "device %d out of range", device);
    if (!sDone[device])
    {
        TLLM_CUDA_CHECK(cudaFuncSetAttribute(
            fp8AttentionSm120Kernel<NVFP4_OUT>, cudaFuncAttributeMaxDynamicSharedMemorySize, Layout::SMEM_BYTES));
        sDone[device] = true;
    }
}

} // namespace flux3FmhaSm120

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace flux3
{

void launchFp8AttentionSm120(Fp8AttentionParams const& args, cudaStream_t stream)
{
    namespace ns = ::flux3FmhaSm120;
    namespace g3 = ::fused_multihead_attention::flux3;

    TLLM_CHECK_WITH_INFO(args.b > 0 && args.s > 0 && args.h > 0, "empty FLUX.3 attention launch");
    TLLM_CHECK_WITH_INFO(args.s % 16 == 0, "FLUX.3 attention needs S %% 16 == 0, got S = %d", args.s);
    TLLM_CHECK_WITH_INFO(args.b == 1 || args.s % 64 == 0, "FLUX.3 attention needs S %% 64 == 0 when B > 1");
    TLLM_CHECK_WITH_INFO(int64_t(args.b) * (args.s / 16) >= 4, "FLUX.3 attention needs B * S >= 64");
    TLLM_CHECK(args.q != nullptr && args.k != nullptr && args.v != nullptr);
    TLLM_CHECK(args.scale_bmm1 != nullptr && args.scale_bmm2 != nullptr);
    TLLM_CHECK_WITH_INFO(
        (reinterpret_cast<uintptr_t>(args.k) & 15) == 0 && (reinterpret_cast<uintptr_t>(args.v) & 15) == 0,
        "FLUX.3 attention needs 16-B aligned K and V");
    bool const nvfp4Out = args.o == nullptr;
    if (nvfp4Out)
    {
        TLLM_CHECK_WITH_INFO(args.fp4 != nullptr && args.sf != nullptr && args.fp4_global_scale != nullptr,
            "FLUX.3 attention: no BF16 output and no NVFP4 output");
    }

    ::bert::Fused_multihead_attention_params_v2 const params = ns::makeParams(args);
    TLLM_CHECK_WITH_INFO(reinterpret_cast<float const&>(params.scale_softmax) == 1.f,
        "the FLUX.3 attention softmax requires scale_softmax == 1");

    int64_t const rowBytes = int64_t(args.h) * ns::kHeadDim;
    CUtensorMap tmK;
    CUtensorMap tmV;
    ns::makeKMap(&tmK, args.k, args.b, args.s, args.h, rowBytes);
    ns::makeVMap(&tmV, args.v, args.b, args.s, args.h, rowBytes);

    g3::Nvfp4Out nfo{};
    nfo.fp4 = args.fp4;
    nfo.sf = args.sf;
    nfo.global_scale = args.fp4_global_scale;

    // The kernel derives its tile range from blockIdx.x and params.s only, so a smaller grid.x computes the leading
    // q_tiles * 64 rows (rounded up to whole CTAs) over every K/V row.
    int const qRows = args.q_rows > 0 && args.q_rows < args.s ? args.q_rows : args.s;
    int const qTiles = (qRows + 63) / 64;
    dim3 const grid((qTiles + g3::GROUPS - 1) / g3::GROUPS, args.h, args.b);
    constexpr int kSmemBytes = ns::Layout::SMEM_BYTES;
    if (nvfp4Out)
    {
        ns::setSmemAttribute<true>();
        ns::fp8AttentionSm120Kernel<true><<<grid, g3::THREADS, kSmemBytes, stream>>>(params, tmK, tmV, nfo);
    }
    else
    {
        ns::setSmemAttribute<false>();
        ns::fp8AttentionSm120Kernel<false><<<grid, g3::THREADS, kSmemBytes, stream>>>(params, tmK, tmV, nfo);
    }
    TLLM_CUDA_CHECK(cudaGetLastError());
}

} // namespace flux3
} // namespace kernels

TRTLLM_NAMESPACE_END
