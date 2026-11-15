/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Selection policy for the head_dim-512 FP8-KV 64-row-tile paged prefill kernel
// (prefill_d512_tile64.cuh). Shared by the planner (scheduler.cuh, which picks
// cta_tile_q) and the paged launcher (prefill.cuh, which runs CTA_TILE_Q == 64 at
// head_dim 512 on that kernel), so the two cannot disagree.
#ifndef FLASHINFER_ATTENTION_D512_TILE64_CONFIG_CUH_
#define FLASHINFER_ATTENTION_D512_TILE64_CONFIG_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include <vector>

#include "../pos_enc.cuh"

namespace flashinfer {

// Compile-time eligibility of a batch-prefill module for the 64-row d512 kernel:
// 16-bit Q/O, one-byte FP8 K/V (no NVFP4, no per-token-head scales), head_dim 512 for
// both QK and VO, no fused RoPE, FP32 QK accumulation.
template <typename DTypeQ, typename DTypeKV, typename DTypeO, uint32_t HEAD_DIM_QK,
          uint32_t HEAD_DIM_VO, PosEncodingMode POS_ENCODING_MODE, bool USE_FP16_QK_REDUCTION,
          bool USE_TOKEN_HEAD_SF>
constexpr bool FA2D512Tile64Eligible() {
  return (std::is_same_v<DTypeQ, nv_bfloat16> || std::is_same_v<DTypeQ, half>) &&
         (std::is_same_v<DTypeO, nv_bfloat16> || std::is_same_v<DTypeO, half>) &&
         (std::is_same_v<DTypeKV, __nv_fp8_e4m3> || std::is_same_v<DTypeKV, __nv_fp8_e5m2>) &&
         HEAD_DIM_QK == 512 && HEAD_DIM_VO == 512 &&
         POS_ENCODING_MODE == PosEncodingMode::kNone && !USE_FP16_QK_REDUCTION &&
         !USE_TOKEN_HEAD_SF;
}

// Runtime switch: the kernel was tuned on SM12x (99 KB opt-in shared memory per block,
// one CTA per SM). FLASHINFER_FA2_D512_TILE64=0 turns the path off; it is read on every
// plan, so one process can switch between the 32-row and 64-row paths.
inline bool FA2D512Tile64Enabled() {
  static const bool device_ok = [] {
    int dev = 0, major = 0, smem_optin = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess)
      return false;
    if (cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) !=
        cudaSuccess)
      return false;
    return major == 12 && smem_optin >= 100352;
  }();
  if (!device_ok) return false;
  const char* env = std::getenv("FLASHINFER_FA2_D512_TILE64");
  return !(env != nullptr && env[0] == '0');
}

// Planner hook, called after the 32-row plan's split-KV decision. The 64-row kernel computes
// each 32-row half exactly as the 32-row kernel does, so its output is bitwise equal only
// when both plans run every request as one KV chunk from key 0: no split, no sliding
// window, no CUDA-graph padding plan (which always splits). Ragged plans pass page_size 1
// and have no 64-row instantiation; short packed lengths (decode, speculative verify) keep
// their tile.
inline bool FA2PromoteD512Tile64(uint32_t cta_tile_q, bool split_kv, bool enable_cuda_graph,
                                 int32_t window_left, int32_t fixed_split_size,
                                 bool disable_split_kv, uint32_t page_size, uint32_t head_dim_vo,
                                 uint32_t head_dim_qk, const std::vector<int64_t>& packed_qo_len,
                                 bool d512_tile64_kv) {
  if (!d512_tile64_kv || cta_tile_q != 32 || split_kv || enable_cuda_graph || window_left >= 0 ||
      page_size <= 1 || head_dim_vo != 512 || (head_dim_qk != 0 && head_dim_qk != 512)) {
    return false;
  }
  if (fixed_split_size > 0 && !disable_split_kv) {
    return false;  // the split is decided per request later; keep the 32-row plan
  }
  int64_t sum = 0;
  for (const int64_t len : packed_qo_len) sum += len;
  if (packed_qo_len.empty() || sum / static_cast<int64_t>(packed_qo_len.size()) <= 64) {
    return false;
  }
  return FA2D512Tile64Enabled();
}

// Longest-first CTA order for FA2 paged prefill (FLASHINFER_FA2_LPT; on unless set to "0",
// read on every plan and launch). The planner emits a non-CUDA-graph paged plan's work items
// in descending causal work, and the paged launchers swap the grid to
// (num_kv_heads, 1, padded_batch_size) so the hardware dispatches every kv head of item 0,
// then of item 1, and so on: one global longest-first list instead of heads-major ascending
// tiles. Each CTA still reads its (request, q tile, kv chunk) from the plan arrays and its
// kv head from the grid, so every CTA does exactly the same work; only the dispatch order
// moves.
// Item order. 1 (default) auto: 64-row d512 plans run requests longest-first with each
// request's items descending (on SM120 this beats one global descending list once several
// requests share the grid: 16x300 -3% vs +3.5%, 3x1000 -10% vs -8%), and keep the planner's
// order when the grid fits in one wave of its one CTA per SM (1x300: no tail to fix); every
// other plan uses one global descending list. 2 request-major always; 3 no sort (item-major
// grid only); 4 global always. 0 is main's order and grid.
inline int FA2LptMode() {
  const char* env = std::getenv("FLASHINFER_FA2_LPT");
  if (env == nullptr || env[0] == '\0') return 1;
  if (env[0] >= '0' && env[0] <= '4') return env[0] - '0';
  return 1;
}

inline bool FA2LptEnabled() { return FA2LptMode() != 0; }

// Grid of a paged prefill launch. Plans with a block_valid_mask (CUDA-graph plans) and grids
// whose item count exceeds gridDim.z's limit keep the heads-major layout.
inline dim3 FA2PagedPrefillGrid(uint32_t padded_batch_size, uint32_t num_kv_heads,
                                bool has_block_valid_mask) {
  if (!has_block_valid_mask && padded_batch_size <= 65535u && FA2LptEnabled()) {
    return dim3(num_kv_heads, 1, padded_batch_size);
  }
  return dim3(padded_batch_size, 1, num_kv_heads);
}

#ifdef __CUDACC__
// Work item and kv head of this CTA under either grid layout. The item-major layout has
// gridDim.z != num_kv_heads unless padded_batch_size == num_kv_heads, and then both layouts
// cover the same (item, head) pairs, so either reading visits each pair exactly once.
__device__ __forceinline__ void FA2PagedPrefillBlockCoords(uint32_t num_kv_heads, uint32_t& bx,
                                                           uint32_t& kv_head_idx) {
  const bool item_major = gridDim.z != num_kv_heads;
  bx = item_major ? blockIdx.z : blockIdx.x;
  kv_head_idx = item_major ? blockIdx.x : blockIdx.z;
}
#endif

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_D512_TILE64_CONFIG_CUH_
