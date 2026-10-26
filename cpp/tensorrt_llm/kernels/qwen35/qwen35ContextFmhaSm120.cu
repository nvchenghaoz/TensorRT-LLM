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

// Launcher of the Qwen3.5-family target's SM120 context FMHA (see qwen35ContextFmhaSm120.h). It replaces the launcher
// of the generated fmha_v2_flash_attention_e4m3_fp32_64_128_S_q_paged_kv_256_causal_output_bf16_sm120_kernel_nl_tiled
// when Launch_params::sm120_target_fmha is set. The kernel is the generated one's device_flash_attention_nl_tiled with
// the same traits plus USE_EXP2_SOFTMAX, so it takes the softmax in base 2 (ex2.approx).
//
// That kernel runs 2 CTAs/SM with one CTA per (64-row q-tile, head, sequence), and all CTAs of a long-prefix prefill
// do about the same work. A 1 x 2048-token prefill with 24 heads is 768 CTAs = 2 x 376 + 16 on 188 SMs: the last 16
// CTAs run as a tail wave. For 1 to 3 sequences, when dropping the last q-tile from the grid saves a whole wave:
//   A) the kernel runs q-tiles 0 .. Q-2 (a params copy with s - 64; bitwise the same rows), or is skipped when the
//      caller reads only each sequence's last row (Launch_params::skip_leading_q_tiles);
//   B) the last q-tile of every (head, sequence) runs as `pieces` CTAs over contiguous KV ranges (one wave);
//   C) the pieces are merged in a fixed order and O is stored with the unchanged epilogue.
// The device code of B and C is in qwen35ContextFmhaLastTileSplit.cuh.

#include <cstdint>
#include <fmha/kernel_traits.h>
#include <fused_multihead_attention.h>
#include <fused_multihead_attention_kernel.h>
#include <fused_multihead_flash_attention_kernel_noloop_tiled.h>

#include "qwen35ContextFmhaLastTileSplit.cuh"

#include "../contextFusedMultiHeadAttention/fused_multihead_attention_common.h"
#include "qwen35ContextFmhaSm120.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <cstring>

namespace qwen35ContextFmhaSm120
{

// The causal kernel traits of the generated kernel (setup.py flash_attention_kernel_template,
// Kernel_traits_nl_tiled_causal; flags 0x1027: LDGSTS for Q/K/V, not head-interleaved, tiled; 0x200: no loop) with
// USE_EXP2_SOFTMAX (0x4000).
using Kt = fmha::Kernel_traits_v2_paged_kv_cache<fmha::Ada_qmma_e4m3_fp32_traits, 128, 256, 0, 64, 4, 1, 1,
    0x1027u | 0x200u | 0x4000u, 3, true, fmha::bf16_t>;

using Ws = fused_multihead_attention::last_tile_split::Workspace<Kt>;

// The generated kernel this launcher replaces.
constexpr char kGeneratedKernelName[]
    = "fmha_v2_flash_attention_e4m3_fp32_64_128_S_q_paged_kv_256_causal_output_bf16_sm120_kernel_nl_tiled";

// The q-tile height of the kernel (the generated launcher's noloop_step).
constexpr int kStepQ = 64;
// At most this many pieces per (head, sequence).
constexpr int kMaxPieces = 16;
constexpr int kMaxPiecesTotal = tensorrt_llm::kernels::qwen35::kContextFmhaSm120MaxPiecesTotal;
constexpr int kMaxBatch = tensorrt_llm::kernels::qwen35::kContextFmhaSm120MaxBatch;

static_assert(Kt::STORE_O_FROM_REGS, "");
static_assert(Kt::USE_EXP2_SOFTMAX, "");
static_assert(Kt::Cta_tile_p::M == kStepQ, "");
static_assert(sizeof(float4) * (Ws::ACC_FLOAT4S_PER_PIECE + Ws::STATS_FLOAT4S_PER_PIECE) * kMaxPiecesTotal
        == tensorrt_llm::kernels::qwen35::kContextFmhaSm120WorkspaceBytes,
    "kContextFmhaSm120WorkspaceBytes does not match the split workspace");

__global__ void contextFmhaE4m3D256CausalPagedKvSm120Kernel(bert::Fused_multihead_attention_params_v2 params)
{
    fused_multihead_attention::device_flash_attention_nl_tiled<Kt>(params);
}

// The generated launcher's causal branch, for this kernel.
void launchKernel(bert::Fused_multihead_attention_params_v2 const& params, cudaStream_t stream)
{
    using Cta_tile_o = typename Kt::Cta_tile_o;
    int const ctasPerORow = (params.d + Cta_tile_o::N - 1) / Cta_tile_o::N;
    int const loopIters = (params.s + kStepQ - 1) / kStepQ;
    dim3 const grid(loopIters * ctasPerORow, params.h, params.b);
    constexpr int kSmemSize = Kt::BYTES_PER_SMEM;
    if (kSmemSize >= 48 * 1024)
    {
        TLLM_CUDA_CHECK(cudaFuncSetAttribute(
            contextFmhaE4m3D256CausalPagedKvSm120Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
    }
    contextFmhaE4m3D256CausalPagedKvSm120Kernel<<<grid, Kt::THREADS, Kt::BYTES_PER_SMEM, stream>>>(params);
}

// CTAs/SM of the piece kernel (the kernel's body with the same traits and shared memory, so the same occupancy), per
// device.
int pieceKernelOccupancy()
{
    constexpr int kMaxDevices = 64;
    static int sOccupancy[kMaxDevices] = {};
    int device = 0;
    TLLM_CUDA_CHECK(cudaGetDevice(&device));
    if (device < 0 || device >= kMaxDevices)
    {
        return 0;
    }
    if (sOccupancy[device] == 0)
    {
        constexpr int kSmemSize = Kt::BYTES_PER_SMEM;
        auto pieceKernel = fused_multihead_attention::last_tile_split::fmha_v2_nl_tiled_last_tile_piece_kernel<Kt>;
        if (kSmemSize >= 48 * 1024)
        {
            TLLM_CUDA_CHECK(cudaFuncSetAttribute(pieceKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
        }
        int occupancy = 0;
        TLLM_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, pieceKernel, Kt::THREADS, kSmemSize));
        sOccupancy[device] = occupancy;
    }
    return sOccupancy[device];
}

} // namespace qwen35ContextFmhaSm120

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// Runs the kernel named kernelName when it is the generated kernel this launcher replaces and returns true; returns
// false (nothing launched) otherwise.
bool runQwen35ContextFmhaSm120(Fused_multihead_attention_params_v2& params, Launch_params const& launch_params,
    char const* kernelName, cudaStream_t stream)
{
    namespace lts = fused_multihead_attention::last_tile_split;
    namespace ns = qwen35ContextFmhaSm120;

    if (kernelName == nullptr || std::strcmp(kernelName, ns::kGeneratedKernelName) != 0 || params.d != 256
        || params.dv != 256 || !launch_params.flash_attention || !launch_params.granular_tiling
        || launch_params.warp_specialization || launch_params.attention_mask_type != ContextAttentionMaskType::CAUSAL
        || launch_params.attention_input_layout != AttentionInputLayout::Q_PAGED_KV)
    {
        return false;
    }
    auto const& bertParams = reinterpret_cast<bert::Fused_multihead_attention_params_v2 const&>(params);

    // The last q-tile split: only for the plain causal prefill of 1 to kMaxBatch unpadded sequences, and only with
    // the split workspace.
    bool const splittable = launch_params.sm120_target_fmha_workspace != nullptr && params.b >= 1
        && params.b <= ns::kMaxBatch && !params.is_s_padded && !params.has_alibi
        && !launch_params.enableAttnLogitSoftcapping && !launch_params.interleaved
        && params.softmax_stats_ptr == nullptr && params.attention_sinks_ptr == nullptr
        && launch_params.sage_block_size_q == 0 && launch_params.sage_block_size_k == 0
        && launch_params.sage_block_size_v == 0 && !launch_params.enableSkipSoftmax;
    int pieces = 0;
    if (splittable)
    {
        int const slots = launch_params.multi_processor_count * ns::pieceKernelOccupancy();
        pieces = lts::choose_num_pieces(
            params.b, params.h, params.s, ns::kStepQ, slots, ns::kMaxPiecesTotal, ns::kMaxPieces, ns::kMaxBatch);
    }
    if (pieces == 0)
    {
        ns::launchKernel(bertParams, stream);
        return true;
    }

    // A) q-tiles 0 .. Q-2, unless the caller reads only each sequence's last row and has checked that every
    // sequence's last row lies in q-tile Q-1: then those rows are left unwritten and B + C below write the last q-tile
    // exactly as they always do.
    if (!launch_params.skip_leading_q_tiles)
    {
        bert::Fused_multihead_attention_params_v2 paramsA = bertParams;
        paramsA.s -= ns::kStepQ;
        ns::launchKernel(paramsA, stream);
    }

    // B) + C) The last q-tile.
    auto* const wsAcc = static_cast<float4*>(launch_params.sm120_target_fmha_workspace);
    auto* const wsStats = wsAcc + static_cast<size_t>(ns::Ws::ACC_FLOAT4S_PER_PIECE) * ns::kMaxPiecesTotal;
    int const qLoopLast = (params.s + ns::kStepQ - 1) / ns::kStepQ - 1;
    cudaError_t const err = lts::launch_last_tile_split<ns::Kt>(bertParams, qLoopLast, pieces, wsAcc, wsStats, stream);
    TLLM_CHECK_WITH_INFO(
        err == cudaSuccess, "SM120 e4m3 context FMHA last q-tile KV split launch failed: %s", cudaGetErrorString(err));
    return true;
}

} // namespace kernels

TRTLLM_NAMESPACE_END
