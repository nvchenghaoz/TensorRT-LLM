// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/losslessBf16Gemm.h"
TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace
{
// Each warp owns an output channel and reuses decoded weights across a token tile.
// Per-token accumulation follows the ordered K-stride-32 FP32 FMA chain; only
// the final result is rounded to BF16. The format preserves all BF16 weight bits.
template <int kBatchTile>
__global__ void losslessBf16GemmKernel(__nv_bfloat16 const* input, uint8_t const* weight, uint64_t const* metadata,
    uint8_t const* exceptions, __nv_bfloat16* output, int rows, int outFeatures, int inFeatures)
{
    int const row = blockIdx.x * 4 + threadIdx.y, lane = threadIdx.x;
    if (row >= outFeatures)
    {
        return;
    }
    float sums[kBatchTile] = {};
    for (int col = lane; col < inFeatures; col += 128)
    {
        int const block = row * (inFeatures / 128) + col / 128;
        auto const header = metadata[block];
        int const escape = header >> 8;
        auto const* data = weight + int64_t(block) * 176;
        const uint32_t signsMantissas = *reinterpret_cast<uint32_t const*>(data + lane * 4);
        int const offset = 128 + (lane / 2) * 3 + lane % 2;
        const uint32_t codes = (uint32_t(data[offset]) | (uint32_t(data[offset + 1]) << 8)) >> ((lane % 2) * 4);
        float decodedWeights[4];
        int prior = 0;
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            const uint32_t signMantissa = (signsMantissas >> (j * 8)) & 255, code = (codes >> (j * 3)) & 7;
            uint32_t exponent = (header & 255) - code;
            if (escape)
            {
                const uint32_t mask = __ballot_sync(0xffffffff, code == 7);
                if (code == 7)
                {
                    exponent = exceptions[escape - 1 + prior + __popc(mask & ((1u << lane) - 1))];
                }
                prior += __popc(mask);
            }
            const uint32_t bits = ((signMantissa & 128) << 8) | (exponent << 7) | (signMantissa & 127);
            decodedWeights[j] = __uint_as_float(bits << 16);
        }
#pragma unroll
        for (int t = 0; t < kBatchTile; t++)
        {
            int const token = blockIdx.y * kBatchTile + t;
            if (token < rows)
            {
#pragma unroll
                for (int j = 0; j < 4; j++)
                {
                    sums[t] = __fmaf_rn(__bfloat162float(input[int64_t(token) * inFeatures + col + j * 32]),
                        decodedWeights[j], sums[t]);
                }
            }
        }
    }
#pragma unroll
    for (int t = 0; t < kBatchTile; t++)
    {
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
        {
            sums[t] = __fadd_rn(sums[t], __shfl_down_sync(0xffffffff, sums[t], offset));
        }
        int const token = blockIdx.y * kBatchTile + t;
        if (lane == 0 && token < rows)
        {
            output[int64_t(token) * outFeatures + row] = __float2bfloat16_rn(sums[t]);
        }
    }
}

} // namespace

void invokeLosslessBf16Gemm(__nv_bfloat16 const* input, uint8_t const* weight, uint64_t const* metadata,
    uint8_t const* exceptions, __nv_bfloat16* output, int rows, int outFeatures, int inFeatures, cudaStream_t stream)
{
    if (rows == 0)
    {
        return;
    }
    if (rows == 1)
    {
        losslessBf16GemmKernel<1><<<dim3((outFeatures + 3) / 4, rows), dim3(32, 4), 0, stream>>>(
            input, weight, metadata, exceptions, output, rows, outFeatures, inFeatures);
    }
    else if (rows == 2)
    {
        losslessBf16GemmKernel<2><<<dim3((outFeatures + 3) / 4, 1), dim3(32, 4), 0, stream>>>(
            input, weight, metadata, exceptions, output, rows, outFeatures, inFeatures);
    }
    else
    {
        losslessBf16GemmKernel<4><<<dim3((outFeatures + 3) / 4, (rows + 3) / 4), dim3(32, 4), 0, stream>>>(
            input, weight, metadata, exceptions, output, rows, outFeatures, inFeatures);
    }
}
} // namespace kernels

TRTLLM_NAMESPACE_END
