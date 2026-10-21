/*
 * SPDX-FileCopyrightText: Copyright (c) 2011-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Last q-tile KV split for the sm_120 e4m3 head-256 causal paged-KV tiled context FMHA
// (fmha_v2_flash_attention_e4m3_fp32_64_128_S_q_paged_kv_256_causal_output_bf16_sm120_kernel_nl_tiled).
//
// That kernel runs 2 CTAs/SM (40 KB smem, register-direct O store) with one CTA per (64-row q-tile, head, sequence),
// and all CTAs of a long-prefix prefill do about the same work. A 1 x 2048-token prefill with 24 heads is
// 768 CTAs = 2 x 376 + 16 on 188 SMs: the last 16 CTAs run as a tail wave (~17% of the call). The bridge below:
//   A) runs the generated launcher on q-tiles 0 .. Q-2 (a params copy with s - 64; bitwise the same rows);
//   B) runs the last q-tile of every (head, sequence) as `pieces` CTAs over contiguous KV ranges (one wave);
//   C) merges the pieces in a fixed order and stores O with the unchanged epilogue.
// The device code lives in cpp/kernels/fmha_v2/src/fused_multihead_flash_attention_kernel_noloop_tiled_last_tile_split.h.
// It is only used when dropping the last q-tile from the generated grid saves a whole wave, and for batches of 1 to 3
// sequences (measured: -14.5% per call at 1 x 2048 over 32768 cached tokens, -5% at 2 x 2048, -3% at 3 x 2048
// (job 702: 2304 CTAs = 6.13 waves -> A 2232 CTAs = 5.94 waves + 360 pieces, k = 5); -1% at 4 x 2048, where it is
// not used).

#include <fmha/kernel_traits.h>
#include <fused_multihead_attention.h>
#include <fused_multihead_attention_kernel.h>
#include <fused_multihead_flash_attention_kernel_noloop_tiled_last_tile_split.h>

namespace fmha_last_tile_split_sm120
{

// The causal kernel traits of the generated kernel (setup.py flash_attention_kernel_template,
// Kernel_traits_nl_tiled_causal; flags 0x1027: LDGSTS for Q/K/V, not head-interleaved, tiled).
using Kt_causal = fmha::Kernel_traits_v2_paged_kv_cache<fmha::Ada_qmma_e4m3_fp32_traits, 128, 256, 0, 64, 4, 1, 1,
    0x1027u | 0x200, 3, true, nv_bfloat16>;

using Ws = fused_multihead_attention::last_tile_split::Workspace<Kt_causal>;

// The q-tile height of the generated kernel (the launcher's noloop_step).
constexpr int kStepQ = 64;
// At most this many pieces per (head, sequence), and in total (the workspace).
constexpr int kMaxPieces = 16;
constexpr int kMaxPiecesTotal = 384;
// The largest batch the split is used for (3: the c4 serving trio prefill, 3 x 24 x 5 = 360 pieces <= 384).
constexpr int kMaxBatch = 3;

static_assert(Kt_causal::STORE_O_FROM_REGS, "");
static_assert(Kt_causal::Cta_tile_p::M == kStepQ, "");

// The workspace (25 MB): FP32 accumulators and row max / sum of every piece. Static, so a call needs no allocation.
__device__ float4 g_ws_acc[Ws::ACC_FLOAT4S_PER_PIECE * kMaxPiecesTotal];
__device__ float4 g_ws_stats[Ws::STATS_FLOAT4S_PER_PIECE * kMaxPiecesTotal];

struct Device_state
{
    bool init = false;
    int occupancy = 0;
    float4* acc = nullptr;
    float4* stats = nullptr;
};

} // namespace fmha_last_tile_split_sm120

#include "../fused_multihead_attention_common.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{
// Per-device state: the workspace addresses and the CTAs/SM of the piece kernel (the generated kernel's body with
// the same traits and shared memory, so the same occupancy as launch A). nullptr if the device index is out of range.
fmha_last_tile_split_sm120::Device_state* get_last_tile_split_state()
{
    namespace lts = fused_multihead_attention::last_tile_split;
    namespace ns = fmha_last_tile_split_sm120;
    using Kt = ns::Kt_causal;
    static ns::Device_state s_state[64];
    int device = 0;
    TLLM_CUDA_CHECK(cudaGetDevice(&device));
    if (device < 0 || device >= 64)
    {
        return nullptr;
    }
    ns::Device_state& state = s_state[device];
    if (!state.init)
    {
        constexpr int smem = Kt::BYTES_PER_SMEM;
        auto piece_kernel = lts::fmha_v2_nl_tiled_last_tile_piece_kernel<Kt>;
        if (smem >= 48 * 1024)
        {
            TLLM_CUDA_CHECK(cudaFuncSetAttribute(piece_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        }
        TLLM_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&state.occupancy, piece_kernel, Kt::THREADS, smem));
        void* acc = nullptr;
        void* stats = nullptr;
        TLLM_CUDA_CHECK(cudaGetSymbolAddress(&acc, ns::g_ws_acc));
        TLLM_CUDA_CHECK(cudaGetSymbolAddress(&stats, ns::g_ws_stats));
        state.acc = static_cast<float4*>(acc);
        state.stats = static_cast<float4*>(stats);
        state.init = true;
        TLLM_LOG_INFO("fmha_v2 e4m3 sm120 last q-tile KV split: %d CTAs/SM, workspace %zu bytes", state.occupancy,
            sizeof(ns::g_ws_acc) + sizeof(ns::g_ws_stats));
    }
    return &state;
}
} // namespace

bool run_fmha_v2_e4m3_d256_causal_paged_kv_last_tile_split_sm120(Fused_multihead_attention_params_v2& params,
    Launch_params const& launch_params,
    void (*launcher)(Fused_multihead_attention_params_v2&, Launch_params const&, cudaStream_t), cudaStream_t stream)
{
    namespace lts = fused_multihead_attention::last_tile_split;
    namespace ns = fmha_last_tile_split_sm120;
    using Kt = ns::Kt_causal;

    if (launcher == nullptr || params.b < 1 || params.b > ns::kMaxBatch || params.is_s_padded)
    {
        return false;
    }
    ns::Device_state* const state_ptr = get_last_tile_split_state();
    if (state_ptr == nullptr)
    {
        return false;
    }
    ns::Device_state& state = *state_ptr;

    int const slots = launch_params.multi_processor_count * state.occupancy;
    int const pieces = lts::choose_num_pieces(
        params.b, params.h, params.s, ns::kStepQ, slots, ns::kMaxPiecesTotal, ns::kMaxPieces, ns::kMaxBatch);
    if (pieces == 0)
    {
        return false;
    }

    // A) The generated kernel on q-tiles 0 .. Q-2.
    Fused_multihead_attention_params_v2 params_a = params;
    params_a.s -= ns::kStepQ;
    launcher(params_a, launch_params, stream);

    // B) + C) The last q-tile.
    auto const& bert_params = reinterpret_cast<bert::Fused_multihead_attention_params_v2 const&>(params);
    int const q_loop_last = (params.s + ns::kStepQ - 1) / ns::kStepQ - 1;
    cudaError_t const err
        = lts::launch_last_tile_split<Kt>(bert_params, q_loop_last, pieces, state.acc, state.stats, stream);
    TLLM_CHECK_WITH_INFO(err == cudaSuccess, "fmha_v2 e4m3 sm120 last q-tile KV split launch failed: %s",
        cudaGetErrorString(err));
    return true;
}

// Only the last q-tile of every (head, sequence), as the split's pieces + merge (B + C above, no launch A): the rows
// of the other q-tiles are not written. Used when the caller reads only each sequence's last row (the final decoder
// layer at prefill). The piece count is choose_num_pieces' k for the same (b, h), so when the split above runs (b <= 3
// on a whole-wave saving) the last q-tile rows are bitwise those of the split. Returns false (nothing launched) when
// the workspace cannot hold one piece per (head, sequence).
bool run_fmha_v2_e4m3_d256_causal_paged_kv_last_tile_only_sm120(
    Fused_multihead_attention_params_v2& params, Launch_params const& launch_params, cudaStream_t stream)
{
    namespace lts = fused_multihead_attention::last_tile_split;
    namespace ns = fmha_last_tile_split_sm120;
    using Kt = ns::Kt_causal;

    if (params.b < 1 || params.h < 1 || params.s < 1 || params.is_s_padded)
    {
        return false;
    }
    long long const hb = static_cast<long long>(params.h) * params.b;
    if (hb > ns::kMaxPiecesTotal)
    {
        return false;
    }
    ns::Device_state* const state_ptr = get_last_tile_split_state();
    if (state_ptr == nullptr)
    {
        return false;
    }
    ns::Device_state& state = *state_ptr;
    long long const slots = static_cast<long long>(launch_params.multi_processor_count) * state.occupancy;
    long long k = std::min<long long>(slots / hb, ns::kMaxPieces);
    k = std::min<long long>(k, ns::kMaxPiecesTotal / hb);
    k = std::max<long long>(k, 1);

    auto const& bert_params = reinterpret_cast<bert::Fused_multihead_attention_params_v2 const&>(params);
    // q_loop -1: each sequence's own last q-tile.
    cudaError_t const err
        = lts::launch_last_tile_split<Kt>(bert_params, -1, static_cast<int>(k), state.acc, state.stats, stream);
    TLLM_CHECK_WITH_INFO(err == cudaSuccess, "fmha_v2 e4m3 sm120 last q-tile only launch failed: %s",
        cudaGetErrorString(err));
    return true;
}

} // namespace kernels

TRTLLM_NAMESPACE_END
