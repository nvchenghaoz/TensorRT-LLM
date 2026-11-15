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
// Paged FA2 prefill for head_dim 512 with one-byte FP8 K/V and 16-bit Q, on a 64-row
// packed Q tile (8 tokens x GQA 8 for Gemma4 global layers).
//
// Layout (one CTA, 8 warps, 1 CTA/SM on 99 KB parts):
//   warp w -> Q-row group wq = w / 2 (16 packed rows), half wk = w % 2.
//   Q.K^T: warp (wq, wk) computes S[16 rows x 16 tokens wk] of each 32-token KV tile over
//          all 512 dims. The pair (wq, 0)/(wq, 1) exchanges row maxima and P fragments
//          through shared memory (named barrier per pair).
//   P.V:   warp (wq, wk) accumulates O[16 rows x dims 256*wk .. 256*wk+255] over the
//          tile's 32 tokens, 128 FP32 registers.
//   K and V tiles have separate buffers: V(i) streams in under Q.K^T(i) and K(i+1)
//   under P.V(i).
// FP8 is dequantized in registers: K to the Q dtype (B operand of Q.K^T; the 4 bytes a
// lane gets from ldmatrix are dims 4t..4t+3, so Q is stored with each 16-dim k-step
// permuted to match), V to FP16 through ldmatrix.trans + byte_perm (even/odd output
// dims), with P kept in FP16 for the P.V MMA.
// All indexing of work items (request/qo/kv tile indices, kv chunks, o_indptr, lse and
// the split-KV merge) follows BatchPrefillWithPagedKVCacheDevice with CTA_TILE_Q = 64.
#ifndef FLASHINFER_ATTENTION_PREFILL_D512_TILE64_CUH_
#define FLASHINFER_ATTENTION_PREFILL_D512_TILE64_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <sstream>
#include <type_traits>

#include "../fastdiv.cuh"
#include "../math.cuh"
#include "../page.cuh"
#include "../utils.cuh"
#include "../vec_dtypes.cuh"
#include "cascade.cuh"
#include "d512_tile64_config.cuh"
#include "mask.cuh"

namespace flashinfer {
namespace d512t64 {

constexpr uint32_t kTileQ = 64;
constexpr uint32_t kTileKV = 32;
constexpr uint32_t kHeadDim = 512;
constexpr uint32_t kNumWarps = 8;
constexpr uint32_t kNumThreads = kNumWarps * 32;
// Q k-step 0 (dims 0..15) lives in registers; smem holds the other 31 k-steps as 62
// chunks of 8 16-bit values, chunk-major ([chunk][row] x 16 B).
constexpr uint32_t kQSmemChunks = (kHeadDim - 16) / 8;
// K/V tiles: 32 chunks of 16 one-byte values, chunk-major ([chunk][token] x 16 B).
constexpr uint32_t kKVChunks = kHeadDim / 16;
constexpr uint32_t kQBytes = kQSmemChunks * kTileQ * 16;   // 63488
constexpr uint32_t kKVBytes = kKVChunks * kTileKV * 16;     // 16384
constexpr uint32_t kPBytes = kNumWarps * 32 * 16;           // 4096
constexpr uint32_t kMDBytes = 4 * 2 * 16 * sizeof(float);   // 512
constexpr uint32_t kQOff = 0;
constexpr uint32_t kKOff = kQOff + kQBytes;
constexpr uint32_t kVOff = kKOff + kKVBytes;
constexpr uint32_t kPOff = kVOff + kKVBytes;
constexpr uint32_t kMDOff = kPOff + kPBytes;
constexpr uint32_t kSmemBytes = kMDOff + kMDBytes;  // 100864
static_assert(kSmemBytes <= 101376, "d512 tile64 smem must fit the 99 KB opt-in limit");

__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t* r) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

__device__ __forceinline__ void ldsm_x4_trans(uint32_t addr, uint32_t* r) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

template <typename T>
__device__ __forceinline__ void mma_m16n8k16(float* c, const uint32_t* a, const uint32_t* b) {
  if constexpr (std::is_same_v<T, half>) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  } else {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  }
}

__device__ __forceinline__ void cp_async_16_zfill(uint32_t smem_addr, const void* gptr,
                                                  bool pred) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_addr), "l"(gptr),
               "r"(pred ? 16 : 0));
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// Named barrier over the two warps (64 threads) of one Q-row group.
__device__ __forceinline__ void pair_barrier(uint32_t id) {
  asm volatile("bar.sync %0, 64;\n" ::"r"(id) : "memory");
}

// Two FP8 values (low 16 bits) -> FP16x2 (exact).
template <typename DTypeKV>
__device__ __forceinline__ uint32_t fp8x2_to_f16x2(uint32_t x) {
  const __half2_raw r = __nv_cvt_fp8x2_to_halfraw2(
      static_cast<__nv_fp8x2_storage_t>(x & 0xffffu),
      std::is_same_v<DTypeKV, __nv_fp8_e4m3> ? __NV_E4M3 : __NV_E5M2);
  return uint32_t(r.x) | (uint32_t(r.y) << 16);
}

// Four FP8 values -> two packed 16-bit pairs of type T (exact): out[0] = (v0, v1),
// out[1] = (v2, v3).
template <typename T, typename DTypeKV>
__device__ __forceinline__ void fp8x4_to_16b(uint32_t w, uint32_t* out) {
  if constexpr (std::is_same_v<T, half>) {
    out[0] = fp8x2_to_f16x2<DTypeKV>(w);
    out[1] = fp8x2_to_f16x2<DTypeKV>(w >> 16);
  } else {
    uint2 o;
    fast_dequant_f8f16x4<DTypeKV, nv_bfloat16>(&w, &o);
    out[0] = o.x;
    out[1] = o.y;
  }
}

__device__ __forceinline__ uint32_t pack_half2(float a, float b) {
  const __half2 h = __floats2half2_rn(a, b);
  return *reinterpret_cast<const uint32_t*>(&h);
}

template <typename DTypeO>
__device__ __forceinline__ void store_4(DTypeO* ptr, float a, float b, float c, float d) {
  vec_t<DTypeO, 4> v;
  v[0] = DTypeO(a);
  v[1] = DTypeO(b);
  v[2] = DTypeO(c);
  v[3] = DTypeO(d);
  v.store(ptr);
}

// Global offsets of this thread's 4 rows (tokens ld_tok + 8 i) of K/V tile `tile`.
template <bool SAME_KV_STRIDES, typename DTypeKV, typename IdType>
__device__ __forceinline__ uint32_t kv_tile_offsets(const paged_kv_t<DTypeKV, IdType>& paged_kv,
                                                   uint32_t packed_page_iter_base, uint32_t tile,
                                                   uint32_t ld_tok, uint32_t ld_chunk,
                                                   uint32_t kv_head_idx, uint32_t chunk_size,
                                                   IdType last_indptr, size_t* k_off,
                                                   size_t* v_off) {
  uint32_t valid = 0;
#pragma unroll
  for (uint32_t i = 0; i < 4; ++i) {
    const uint32_t rel = tile * kTileKV + ld_tok + 8 * i;
    uint32_t page_iter, entry_idx;
    paged_kv.page_size.divmod(packed_page_iter_base + rel, page_iter, entry_idx);
    const bool in_range = static_cast<IdType>(page_iter) < last_indptr;
    const size_t page = in_range ? static_cast<size_t>(__ldg(paged_kv.indices + page_iter)) : 0;
    k_off[i] = in_range ? page * paged_kv.stride_page + size_t(kv_head_idx) * paged_kv.stride_h +
                              size_t(entry_idx) * paged_kv.stride_n + ld_chunk * 16
                        : 0;
    if constexpr (!SAME_KV_STRIDES) {
      v_off[i] = in_range ? page * paged_kv.v_stride_page +
                                size_t(kv_head_idx) * paged_kv.v_stride_h +
                                size_t(entry_idx) * paged_kv.v_stride_n + ld_chunk * 16
                          : 0;
    }
    valid |= (rel < chunk_size ? 1u : 0u) << i;
  }
  return valid;
}

template <typename DTypeKV>
__device__ __forceinline__ void kv_tile_issue(uint32_t smem_dst, const DTypeKV* data,
                                              const size_t* off, uint32_t valid) {
#pragma unroll
  for (uint32_t i = 0; i < 4; ++i) {
    cp_async_16_zfill(smem_dst + i * 8 * 16, data + off[i], (valid >> i) & 1u);
  }
}

template <typename Params, typename AttentionVariant, MaskMode MASK_MODE>
struct Supported {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  static constexpr bool value =
      (std::is_same_v<DTypeQ, nv_bfloat16> || std::is_same_v<DTypeQ, half>) &&
      (std::is_same_v<DTypeO, nv_bfloat16> || std::is_same_v<DTypeO, half>) &&
      (std::is_same_v<DTypeKV, __nv_fp8_e4m3> || std::is_same_v<DTypeKV, __nv_fp8_e5m2>) &&
      AttentionVariant::use_softmax && MASK_MODE != MaskMode::kMultiItemScoring;
};

template <bool SAME_KV_STRIDES, MaskMode MASK_MODE, typename AttentionVariant, typename Params>
__global__ __launch_bounds__(kNumThreads, 1) void BatchPrefillPagedD512Tile64Kernel(
    const __grid_constant__ Params params) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;
  extern __shared__ __align__(16) uint8_t smem[];

  const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
  const uint32_t wq = warp >> 1, wk = warp & 1u;
  const uint32_t g = lane >> 2, t = lane & 3u;
  const uint32_t bx = blockIdx.x, kv_head_idx = blockIdx.z;

  if (params.block_valid_mask && !params.block_valid_mask[bx]) {
    return;
  }
  const uint32_t request_idx = params.request_indices[bx];
  const uint32_t qo_tile_idx = params.qo_tile_indices[bx];
  const uint32_t kv_tile_idx = params.kv_tile_indices[bx];
  AttentionVariant variant(params, /*batch_idx=*/request_idx, smem);
  const uint32_t qo_len = variant.qo_len, kv_len = variant.kv_len,
                 window_left = variant.window_left;
  const uint_fastdiv& group_size = params.group_size;
  const uint32_t num_qo_heads = gridDim.z * group_size;
  const bool partition_kv = params.partition_kv;
  const uint32_t kv_chunk_size = *(params.kv_chunk_size_ptr);
  const uint32_t kv_len_safe = kv_len > 0 ? kv_len : 1;

  const uint32_t qo_upper_bound = min(qo_len, ceil_div((qo_tile_idx + 1) * kTileQ, group_size));
  const uint32_t kv_start_idx =
      sub_if_greater_or_zero(kv_len + (qo_tile_idx * kTileQ) / group_size, qo_len + window_left);
  const uint32_t max_chunk_size = partition_kv ? kv_chunk_size : kv_len - kv_start_idx;
  const uint32_t chunk_start =
      partition_kv ? min(kv_tile_idx * max_chunk_size + kv_start_idx, kv_len) : kv_start_idx;
  const uint32_t chunk_end =
      partition_kv ? min((kv_tile_idx + 1) * max_chunk_size + kv_start_idx, kv_len) : kv_len;
  const uint32_t chunk_size = chunk_end - chunk_start;

  const uint32_t num_iterations = ceil_div(
      (MASK_MODE == MaskMode::kCausal
           ? min(chunk_size,
                 sub_if_greater_or_zero(
                     kv_len - qo_len + ceil_div((qo_tile_idx + 1) * kTileQ, group_size),
                     chunk_start))
           : chunk_size),
      kTileKV);
  const uint32_t window_iteration =
      ceil_div(sub_if_greater_or_zero(kv_len + ceil_div((qo_tile_idx + 1) * kTileQ, group_size),
                                      qo_len + window_left + chunk_start),
               kTileKV);
  const uint32_t mask_iteration =
      (MASK_MODE == MaskMode::kCausal
           ? min(chunk_size,
                 sub_if_greater_or_zero(
                     kv_len + ceil_div(qo_tile_idx * kTileQ, group_size) - qo_len, chunk_start))
           : chunk_size) /
      kTileKV;

  const uint32_t smem_base = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  const uint32_t q_s = smem_base + kQOff, k_s = smem_base + kKOff, v_s = smem_base + kVOff;
  uint4* p_smem = reinterpret_cast<uint4*>(smem + kPOff);
  float* md_smem = reinterpret_cast<float*>(smem + kMDOff);

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  // ---------------------------------------------------------------- K/V tile loader
  const paged_kv_t<DTypeKV, IdType>& paged_kv = params.paged_kv;
  const IdType last_indptr = paged_kv.indptr[paged_kv.batch_size];
  const uint32_t packed_page_iter_base =
      paged_kv.indptr[request_idx] * paged_kv.page_size + chunk_start;
  // Thread -> one 16-byte chunk column, tokens ld_tok + 8 i (i < 4). An 8-lane phase
  // writes 8 consecutive tokens of one chunk: 128 contiguous smem bytes.
  const uint32_t ld_chunk = (warp << 2) | (lane >> 3);
  const uint32_t ld_tok = lane & 7u;
  const uint32_t ld_smem = (ld_chunk * kTileKV + ld_tok) * 16;
  const DTypeKV* k_data = paged_kv.k_data;
  const DTypeKV* v_data = paged_kv.v_data;
  size_t k_off[4];
  size_t v_off_store[SAME_KV_STRIDES ? 1 : 4];
  size_t* v_off = SAME_KV_STRIDES ? k_off : v_off_store;
  uint32_t ld_valid = 0;

  if (num_iterations > 0) {
    ld_valid = kv_tile_offsets<SAME_KV_STRIDES>(paged_kv, packed_page_iter_base, 0, ld_tok,
                                                ld_chunk, kv_head_idx, chunk_size, last_indptr,
                                                k_off, v_off_store);
    kv_tile_issue(k_s + ld_smem, k_data, k_off, ld_valid);
  }
  cp_async_commit();
  if (num_iterations > 0) {
    kv_tile_issue(v_s + ld_smem, v_data, v_off, ld_valid);
  }
  cp_async_commit();

  // ---------------------------------------------------------------- Q tile
  const size_t q_stride_n = params.q_stride_n, q_stride_h = params.q_stride_h;
  const DTypeQ* q_base = params.q + size_t(params.q_indptr[request_idx]) * q_stride_n +
                         size_t(kv_head_idx * group_size) * q_stride_h;
  // smem: k-steps 1..31, each split into two permuted chunks: chunk 0 holds the words
  // {0,2,4,6} of the k-step (dims 4t, 4t+1 for word t), chunk 1 the words {1,3,5,7}.
#pragma unroll 2
  for (uint32_t task = tid; task < kTileQ * 31; task += kNumThreads) {
    const uint32_t row = task & (kTileQ - 1), ks = 1 + task / kTileQ;
    uint32_t qi, hi;
    group_size.divmod(qo_tile_idx * kTileQ + row, qi, hi);
    uint4 a = make_uint4(0, 0, 0, 0), b = make_uint4(0, 0, 0, 0);
    if (qi < qo_upper_bound) {
      const uint4* src =
          reinterpret_cast<const uint4*>(q_base + qi * q_stride_n + hi * q_stride_h + ks * 16);
      a = __ldg(src);
      b = __ldg(src + 1);
    }
    *reinterpret_cast<uint4*>(smem + kQOff + ((2 * ks - 2) * kTileQ + row) * 16) =
        make_uint4(a.x, a.z, b.x, b.z);
    *reinterpret_cast<uint4*>(smem + kQOff + ((2 * ks - 1) * kTileQ + row) * 16) =
        make_uint4(a.y, a.w, b.y, b.w);
  }
  // k-step 0 A-fragment (rows g, g + 8 of this warp's 16; dims 4t..4t+3).
  uint32_t qa0[4];
  uint32_t row_q[2], row_h[2];
#pragma unroll
  for (uint32_t j = 0; j < 2; ++j) {
    group_size.divmod(qo_tile_idx * kTileQ + 16 * wq + g + 8 * j, row_q[j], row_h[j]);
    uint2 v = make_uint2(0, 0);
    if (row_q[j] < qo_upper_bound) {
      v = __ldg(reinterpret_cast<const uint2*>(q_base + row_q[j] * q_stride_n +
                                               row_h[j] * q_stride_h + 4 * t));
    }
    qa0[j] = v.x;
    qa0[2 + j] = v.y;
  }

  cp_async_wait<1>();
  __syncthreads();

  // ---------------------------------------------------------------- main loop
  const float sm_scale = variant.sm_scale_log2;
  const uint32_t frag_row = (lane & 7u) + ((lane >> 3) & 1u) * 8;
  const uint32_t q_lane = q_s + ((lane >> 4) * kTileQ + 16 * wq + frag_row) * 16;
  const uint32_t k_lane = k_s + ((lane >> 4) * kTileKV + 16 * wk + frag_row) * 16;
  const uint32_t v_lane = v_s + ((lane >> 4) * kTileKV + frag_row) * 16 + wk * 16 * kTileKV * 16;

  float o[16][2][4];
#pragma unroll
  for (uint32_t c = 0; c < 16; ++c) {
#pragma unroll
    for (uint32_t p = 0; p < 2; ++p) {
#pragma unroll
      for (uint32_t r = 0; r < 4; ++r) o[c][p][r] = 0.f;
    }
  }
  float m[2] = {-math::inf, -math::inf};
  float dsum[2] = {0.f, 0.f};
  float* md_own = md_smem + (wq * 2 + wk) * 16;
  float* md_partner = md_smem + (wq * 2 + (wk ^ 1u)) * 16;
  const uint32_t pair_bar_id = 1 + wq;

#pragma unroll 1
  for (uint32_t iter = 0; iter < num_iterations; ++iter) {
    const bool has_next = iter + 1 < num_iterations;
    if (has_next) {
      ld_valid = kv_tile_offsets<SAME_KV_STRIDES>(paged_kv, packed_page_iter_base, iter + 1,
                                                  ld_tok, ld_chunk, kv_head_idx, chunk_size,
                                                  last_indptr, k_off, v_off_store);
    }

    // S = Q . K^T for this warp's 16 rows x 16 tokens.
    float s[2][4];
#pragma unroll
    for (uint32_t n = 0; n < 2; ++n) {
#pragma unroll
      for (uint32_t r = 0; r < 4; ++r) s[n][r] = 0.f;
    }
#pragma unroll
    for (uint32_t sp = 0; sp < 16; ++sp) {
      uint32_t kr[4];
      ldsm_x4(k_lane + sp * 2 * kTileKV * 16, kr);
#pragma unroll
      for (uint32_t h = 0; h < 2; ++h) {
        const uint32_t ks = 2 * sp + h;
        uint32_t a[4];
        if (ks == 0) {
#pragma unroll
          for (uint32_t r = 0; r < 4; ++r) a[r] = qa0[r];
        } else {
          ldsm_x4(q_lane + (2 * ks - 2) * kTileQ * 16, a);
        }
#pragma unroll
        for (uint32_t n = 0; n < 2; ++n) {
          uint32_t b[2];
          fp8x4_to_16b<DTypeQ, DTypeKV>(kr[2 * h + n], b);
          mma_m16n8k16<DTypeQ>(s[n], a, b);
        }
      }
    }

    // logits transform + mask. Element (n, r): row j = r / 2, token 8 n + 2 t + r % 2.
    const uint32_t kv_idx_base = chunk_start + iter * kTileKV + 16 * wk + 2 * t;
#pragma unroll
    for (uint32_t n = 0; n < 2; ++n) {
#pragma unroll
      for (uint32_t r = 0; r < 4; ++r) {
        const uint32_t j = r >> 1;
        const uint32_t kv_idx = kv_idx_base + 8 * n + (r & 1u);
        if (kv_idx < chunk_end) {
          s[n][r] = variant.LogitsTransform(params, s[n][r], request_idx, row_q[j], kv_idx,
                                            kv_head_idx * group_size + row_h[j], kv_head_idx);
        }
      }
    }
    if (MASK_MODE == MaskMode::kCustom || iter >= mask_iteration || iter < window_iteration) {
#pragma unroll
      for (uint32_t n = 0; n < 2; ++n) {
#pragma unroll
        for (uint32_t r = 0; r < 4; ++r) {
          const uint32_t j = r >> 1;
          const uint32_t kv_idx = kv_idx_base + 8 * n + (r & 1u);
          const bool keep =
              (!(MASK_MODE == MaskMode::kCausal
                     ? (kv_idx + qo_len > kv_len + row_q[j] || kv_idx >= chunk_end)
                     : kv_idx >= chunk_end)) &&
              variant.LogitsMask(params, request_idx, row_q[j], kv_idx,
                                 kv_head_idx * group_size + row_h[j], kv_head_idx);
          s[n][r] = keep ? s[n][r] : -math::inf;
        }
      }
    }

    // Online softmax, consistent across the warp pair.
    float mloc[2];
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      mloc[j] = max(max(s[0][2 * j], s[0][2 * j + 1]), max(s[1][2 * j], s[1][2 * j + 1]));
      mloc[j] = max(mloc[j], math::shfl_xor_sync(mloc[j], 0x1));
      mloc[j] = max(mloc[j], math::shfl_xor_sync(mloc[j], 0x2));
    }
    if (t == 0) {
      md_own[g] = mloc[0];
      md_own[g + 8] = mloc[1];
    }
    pair_barrier(pair_bar_id);
    float m_scaled[2], o_scale[2];
    bool changed = false;
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      const float m_new = max(m[j], max(mloc[j], md_partner[g + 8 * j]));
      m_scaled[j] = max(m_new * sm_scale, -cuda::std::numeric_limits<float>::max());
      o_scale[j] = math::ptx_exp2(m[j] * sm_scale - m_scaled[j]);
      changed |= (m_new != m[j]);
      m[j] = m_new;
      dsum[j] *= o_scale[j];
    }
    if (__any_sync(0xffffffffu, changed)) {
#pragma unroll
      for (uint32_t c = 0; c < 16; ++c) {
#pragma unroll
        for (uint32_t p = 0; p < 2; ++p) {
          o[c][p][0] *= o_scale[0];
          o[c][p][1] *= o_scale[0];
          o[c][p][2] *= o_scale[1];
          o[c][p][3] *= o_scale[1];
        }
      }
    }
    uint32_t pf[4];
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      const float p00 = math::ptx_exp2(s[0][2 * j] * sm_scale - m_scaled[j]);
      const float p01 = math::ptx_exp2(s[0][2 * j + 1] * sm_scale - m_scaled[j]);
      const float p10 = math::ptx_exp2(s[1][2 * j] * sm_scale - m_scaled[j]);
      const float p11 = math::ptx_exp2(s[1][2 * j + 1] * sm_scale - m_scaled[j]);
      dsum[j] += (p00 + p01) + (p10 + p11);
      pf[j] = pack_half2(p00, p01);
      pf[2 + j] = pack_half2(p10, p11);
    }
    p_smem[warp * 32 + lane] = make_uint4(pf[0], pf[1], pf[2], pf[3]);
    pair_barrier(pair_bar_id);
    const uint4 pp = p_smem[(warp ^ 1u) * 32 + lane];

    // V(i) landed and every warp is past Q.K^T(i): K buffer is free.
    cp_async_wait<0>();
    __syncthreads();
    if (has_next) {
      kv_tile_issue(k_s + ld_smem, k_data, k_off, ld_valid);
    }
    cp_async_commit();

    // O += P . V over this warp's 256 output dims.
#pragma unroll
    for (uint32_t kk = 0; kk < 2; ++kk) {
      uint32_t a[4];
      const bool own = (kk == wk);
      a[0] = own ? pf[0] : pp.x;
      a[1] = own ? pf[1] : pp.y;
      a[2] = own ? pf[2] : pp.z;
      a[3] = own ? pf[3] : pp.w;
#pragma unroll
      for (uint32_t cp = 0; cp < 8; ++cp) {
        uint32_t vr[4];
        ldsm_x4_trans(v_lane + kk * 16 * 16 + 2 * cp * kTileKV * 16, vr);
#pragma unroll
        for (uint32_t h = 0; h < 2; ++h) {
          const uint32_t we = __byte_perm(vr[2 * h], vr[2 * h + 1], 0x6420);
          const uint32_t wo = __byte_perm(vr[2 * h], vr[2 * h + 1], 0x7531);
          uint32_t be[2], bo[2];
          fp8x4_to_16b<half, DTypeKV>(we, be);
          fp8x4_to_16b<half, DTypeKV>(wo, bo);
          mma_m16n8k16<half>(o[2 * cp + h][0], a, be);
          mma_m16n8k16<half>(o[2 * cp + h][1], a, bo);
        }
      }
    }

    // K(i+1) landed and every warp is past P.V(i): V buffer is free.
    cp_async_wait<0>();
    __syncthreads();
    if (has_next) {
      kv_tile_issue(v_s + ld_smem, v_data, v_off, ld_valid);
    }
    cp_async_commit();
  }
  cp_async_wait<0>();

  // ---------------------------------------------------------------- epilogue
#pragma unroll
  for (uint32_t j = 0; j < 2; ++j) {
    dsum[j] += math::shfl_xor_sync(dsum[j], 0x1);
    dsum[j] += math::shfl_xor_sync(dsum[j], 0x2);
  }
  if (t == 0) {
    md_own[g] = dsum[0];
    md_own[g + 8] = dsum[1];
  }
  pair_barrier(pair_bar_id);
#pragma unroll
  for (uint32_t j = 0; j < 2; ++j) {
    dsum[j] += md_partner[g + 8 * j];
    if (m[j] != -math::inf) {
      m[j] *= sm_scale;
    }
  }

  const uint32_t num_kv_chunks = ceil_div(min(kv_len_safe, window_left + kTileQ), kv_chunk_size);
  const size_t o_stride_n = size_t(num_qo_heads) * kHeadDim, o_stride_h = kHeadDim;
  DTypeO* o_base = partition_kv
                       ? params.o + (size_t(params.o_indptr[request_idx]) + kv_tile_idx) * o_stride_n +
                             size_t(kv_head_idx * group_size) * o_stride_h
                       : params.o + size_t(params.o_indptr[request_idx]) * o_stride_n +
                             size_t(kv_head_idx * group_size) * o_stride_h;
  const size_t o_row_stride = partition_kv ? size_t(num_kv_chunks) * o_stride_n : o_stride_n;
#pragma unroll
  for (uint32_t j = 0; j < 2; ++j) {
    const float rd = (dsum[j] > 0.f) ? math::ptx_rcp(dsum[j]) : 0.f;
    if (row_q[j] < qo_upper_bound) {
      DTypeO* orow = o_base + row_q[j] * o_row_stride + row_h[j] * o_stride_h + 256 * wk + 4 * t;
#pragma unroll
      for (uint32_t c = 0; c < 16; ++c) {
        store_4<DTypeO>(orow + 16 * c, o[c][0][2 * j] * rd, o[c][1][2 * j] * rd,
                        o[c][0][2 * j + 1] * rd, o[c][1][2 * j + 1] * rd);
      }
    }
  }
  if (params.lse != nullptr && wk == 0 && t == 0) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      if (row_q[j] < qo_upper_bound) {
        const uint32_t qo_head_idx = kv_head_idx * group_size + row_h[j];
        const float val = math::ptx_log2(dsum[j]) + m[j];
        if (partition_kv) {
          params.lse[(size_t(params.o_indptr[request_idx]) + row_q[j] * num_kv_chunks +
                      kv_tile_idx) *
                         num_qo_heads +
                     qo_head_idx] = val;
        } else {
          params.lse[(size_t(params.o_indptr[request_idx]) + row_q[j]) * num_qo_heads +
                     qo_head_idx] = val;
        }
      }
    }
  }

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <bool SAME_KV_STRIDES, MaskMode MASK_MODE, typename AttentionVariant, typename Params>
cudaError_t BatchPrefillPagedD512Tile64Dispatched(Params params, typename Params::DTypeO* tmp_v,
                                                  float* tmp_s, bool enable_pdl,
                                                  cudaStream_t stream) {
  if constexpr (!Supported<Params, AttentionVariant, MASK_MODE>::value) {
    FLASHINFER_ERROR(
        "The head_dim-512 64-row paged prefill kernel supports 16-bit Q/O, FP8 K/V, softmax "
        "variants and non-multi-item masks only; set FLASHINFER_FA2_D512_TILE64=0 to plan "
        "32-row tiles instead.");
    return cudaErrorNotSupported;
  } else {
    const uint32_t padded_batch_size = params.padded_batch_size;
    if (padded_batch_size == 0) {
      return cudaSuccess;
    }
    const uint32_t num_kv_heads = params.paged_kv.num_heads;
    const uint32_t num_qo_heads = params.num_qo_heads;
    dim3 nblks(padded_batch_size, 1, num_kv_heads);
    dim3 nthrs(kNumThreads, 1, 1);
    auto kernel = BatchPrefillPagedD512Tile64Kernel<SAME_KV_STRIDES, MASK_MODE, AttentionVariant,
                                                    Params>;
    int dev_id = 0, max_smem_per_block_optin = 0;
    FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
    FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&max_smem_per_block_optin,
                                                cudaDevAttrMaxSharedMemoryPerBlockOptin, dev_id));
    if (max_smem_per_block_optin < static_cast<int>(kSmemBytes)) {
      std::ostringstream err_msg;
      err_msg << "The head_dim-512 64-row paged prefill kernel needs " << kSmemBytes
              << " bytes of shared memory per block; this GPU allows " << max_smem_per_block_optin;
      FLASHINFER_ERROR(err_msg.str());
    }
    FLASHINFER_CUDA_CALL(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
    cudaLaunchAttribute attribute[1];
    cudaLaunchConfig_t config;
    if (enable_pdl) {
      attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attribute[0].val.programmaticStreamSerializationAllowed = 1;
      config.attrs = attribute;
      config.numAttrs = 1;
      config.gridDim = nblks;
      config.blockDim = nthrs;
      config.dynamicSmemBytes = kSmemBytes;
      config.stream = stream;
    }
    auto launch = [&]() -> cudaError_t {
      if (enable_pdl) {
        return cudaLaunchKernelEx(&config, kernel, params);
      }
      void* args[] = {(void*)&params};
      return cudaLaunchKernel((void*)kernel, nblks, nthrs, args, kSmemBytes, stream);
    };
    if (tmp_v == nullptr) {
      params.partition_kv = false;
      FLASHINFER_CUDA_CALL(launch());
    } else {
      params.partition_kv = true;
      auto o = params.o;
      auto lse = params.lse;
      params.o = tmp_v;
      params.lse = tmp_s;
      FLASHINFER_CUDA_CALL(launch());
      FLASHINFER_CUDA_CALL(VariableLengthMergeStates(tmp_v, tmp_s, params.merge_indptr, o, lse,
                                                     params.max_total_num_rows,
                                                     params.total_num_rows, num_qo_heads,
                                                     kHeadDim, enable_pdl, stream));
    }
    return cudaSuccess;
  }
}

}  // namespace d512t64
}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_PREFILL_D512_TILE64_CUH_
