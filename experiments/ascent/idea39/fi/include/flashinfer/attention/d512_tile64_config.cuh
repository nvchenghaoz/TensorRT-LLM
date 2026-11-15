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
// one CTA per SM). FLASHINFER_FA2_D512_TILE64=0 turns the path off.
inline bool FA2D512Tile64Enabled() {
  static const bool enabled = [] {
    const char* env = std::getenv("FLASHINFER_FA2_D512_TILE64");
    if (env != nullptr && env[0] == '0') return false;
    int dev = 0, major = 0, smem_optin = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess)
      return false;
    if (cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) !=
        cudaSuccess)
      return false;
    return major == 12 && smem_optin >= 100864;
  }();
  return enabled;
}

// Planner hook: promote a 32-row d512 tile to the 64-row kernel. Ragged plans pass
// page_size 1 and have no 64-row d512 instantiation, so they keep 32. Short packed
// lengths (decode, speculative verify) keep their tile.
inline uint32_t FA2MaybeD512Tile64(uint32_t cta_tile_q, int64_t packed_qo_len,
                                   uint32_t head_dim_vo, uint32_t head_dim_qk, uint32_t page_size,
                                   bool d512_tile64_kv) {
  if (!d512_tile64_kv || cta_tile_q != 32 || packed_qo_len <= 64 || page_size <= 1 ||
      head_dim_vo != 512 || (head_dim_qk != 0 && head_dim_qk != 512)) {
    return cta_tile_q;
  }
  return FA2D512Tile64Enabled() ? 64u : cta_tile_q;
}

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_D512_TILE64_CONFIG_CUH_
