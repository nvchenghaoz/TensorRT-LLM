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
// Paged FA2 prefill for head_dim 512 with one-byte FP8 K/V and 16-bit Q on a 64-row packed
// Q tile, bitwise equal to the 32-row kernel (BatchPrefillWithPagedKVCacheDevice with
// KernelTraits<CTA_TILE_Q 32, NUM_MMA_Q 2, NUM_MMA_KV 1, NUM_WARPS_Q 1, NUM_WARPS_KV 4>,
// VO split, 64-key tiles, vosplit_softmax_store_p / vosplit_compute_pv).
//
// One CTA = 2 row groups x 4 KV/VO warps. Row group g runs exactly the 32-row CTA at
// virtual Q tile 2 * qo_tile_idx + g: the same Q.K^T chain (warp w: 32 rows x keys
// 16w..16w+15, d-steps 0..31 in order), the same cross-warp softmax (max over w = 0..3,
// BF16 P, dsum shfl 2 then 1, d += sum over w = 0..3) and the same P.V order (o *= o_scale,
// then keys 0..15, 16..31, 32..47, 48..63 into this warp's 128 dims). Each group keeps its
// own iteration count and mask bounds, so it never runs a tile the 32-row kernel skips.
// The two groups share the K/V loads: each 64-key tile streams through a 3-slot ring of
// 8 KB FP8 sub-tiles (K as four 128-dim chunks [64 keys][128 B], then V as four 16-key
// chunks [16 keys][512 B]) with two sub-tiles in flight, so loads overlap the MMAs.
// Only the MMA operand delivery differs from the 32-row kernel; every operand value, k
// position and accumulation order is the same (P.V output columns are held in an even/odd
// dim order, which changes no arithmetic).
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

constexpr uint32_t kTileQ = 64;    // packed rows per CTA (2 groups x 32)
constexpr uint32_t kTileKV = 64;   // keys per tile, as the 32-row kernel
constexpr uint32_t kHeadDim = 512;
constexpr uint32_t kNumThreads = 256;
constexpr uint32_t kSubTiles = 8;  // per KV tile: K chunks 0..3, V chunks 0..3
constexpr uint32_t kNumSlots = 3;
constexpr uint32_t kSlotBytes = 8192;
// Shared memory, all "chunk-major" in 16-byte units ([chunk][row] x 16 B), so 8 consecutive
// rows of one chunk are 128 contiguous bytes for ldmatrix and the loaders.
constexpr uint32_t kQBytes = kTileQ * kHeadDim * 2;  // [64 chunks][64 rows]
constexpr uint32_t kPBytes = kTileQ * kTileKV * 2;   // [8 chunks][64 rows], BF16 P
constexpr uint32_t kMDBytes = 2 * 2 * 4 * 32 * sizeof(float);  // max / sum x group x warp x row
constexpr uint32_t kQOff = 0;
constexpr uint32_t kRingOff = kQOff + kQBytes;
constexpr uint32_t kPOff = kRingOff + kNumSlots * kSlotBytes;
constexpr uint32_t kMDOff = kPOff + kPBytes;
constexpr uint32_t kSmemBytes = kMDOff + kMDBytes;  // 100352
static_assert(kSmemBytes <= 101376, "d512 tile64 smem must fit the 99 KB opt-in limit");

#ifndef D512T64_D_UPDATE
#define D512T64_D_UPDATE 1
#endif
#ifndef D512T64_K_LDS128
#define D512T64_K_LDS128 0
#endif

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

__device__ __forceinline__ uint32_t lds_u32(uint32_t addr) {
  uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(v) : "r"(addr));
  return v;
}

__device__ __forceinline__ uint4 lds_u128(uint32_t addr) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "r"(addr));
  return v;
}

__device__ __forceinline__ void sts_u32(uint32_t addr, uint32_t v) {
  asm volatile("st.shared.u32 [%0], %1;\n" ::"r"(addr), "r"(v));
}

// One m16n8k16 MMA, f32 accumulate (the instruction mma_sync_m16n16k16_row_col_f16f16f32
// issues per n8 half).
template <typename T>
__device__ __forceinline__ void mma_m16n8k16(float* c, const uint32_t* a, const uint32_t* b) {
  if constexpr (std::is_same_v<T, half>) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]),
          "f"(c[2]), "f"(c[3]));
  } else {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]),
          "f"(c[2]), "f"(c[3]));
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

// Named barrier over one row group (4 warps, 128 threads).
__device__ __forceinline__ void group_barrier(uint32_t id) {
  asm volatile("bar.sync %0, 128;\n" ::"r"(id) : "memory");
}

// Four FP8 values -> two packed 16-bit pairs (exact), out[0] = (v0, v1), out[1] = (v2, v3):
// vec_cast<DTypeQ, DTypeKV>::cast<4>, the conversion the 32-row kernel uses.
template <typename T, typename DTypeKV>
__device__ __forceinline__ void fp8x4_to_16b(uint32_t w, uint32_t* out) {
  vec_cast<T, DTypeKV>::template cast<4>(reinterpret_cast<T*>(out),
                                         reinterpret_cast<const DTypeKV*>(&w));
}

template <typename T>
__device__ __forceinline__ uint32_t pack2(float a, float b) {
  if constexpr (std::is_same_v<T, half>) {
    const half2 h = make_half2(T(a), T(b));
    return *reinterpret_cast<const uint32_t*>(&h);
  } else {
    const nv_bfloat162 h = make_bfloat162(T(a), T(b));
    return *reinterpret_cast<const uint32_t*>(&h);
  }
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

// K/V ring loader: thread -> key `key` of each 64-key tile, 16-byte part `part` (an 8-lane
// phase = 8 keys of one part: 128 contiguous smem bytes; a warp instruction covers 8 keys x
// 64 contiguous global bytes).
template <bool SAME_KV_STRIDES, typename DTypeKV, typename IdType>
struct RingLoader {
  const DTypeKV* k_data;
  const DTypeKV* v_data;
  size_t koff, voff;
  uint32_t key, part, warp, ring_s;
  bool valid;

  __device__ __forceinline__ void set_tile(const paged_kv_t<DTypeKV, IdType>& paged_kv,
                                           uint32_t packed_page_iter_base, IdType last_indptr,
                                           uint32_t kv_head_idx, uint32_t chunk_size,
                                           uint32_t tile) {
    const uint32_t rel = tile * kTileKV + key;
    uint32_t page_iter, entry_idx;
    paged_kv.page_size.divmod(packed_page_iter_base + rel, page_iter, entry_idx);
    const bool in_range = static_cast<IdType>(page_iter) < last_indptr;
    const size_t page = in_range ? static_cast<size_t>(__ldg(paged_kv.indices + page_iter)) : 0;
    koff = in_range ? page * paged_kv.stride_page + size_t(kv_head_idx) * paged_kv.stride_h +
                          size_t(entry_idx) * paged_kv.stride_n
                    : 0;
    if constexpr (SAME_KV_STRIDES) {
      voff = koff;
    } else {
      voff = in_range ? page * paged_kv.v_stride_page + size_t(kv_head_idx) * paged_kv.v_stride_h +
                            size_t(entry_idx) * paged_kv.v_stride_n
                      : 0;
    }
    valid = rel < chunk_size;
  }

  // Sub-tile st (0..3: K dims 128 st.. of all 64 keys; 4..7: V keys 16 (st-4).. , all dims).
  __device__ __forceinline__ void issue(uint32_t st, uint32_t slot) const {
    const uint32_t dst_base = ring_s + slot * kSlotBytes;
    if (st < 4) {
      // smem [8 pieces][64 keys] x 16 B
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        const uint32_t piece = part + 4 * j;
        cp_async_16_zfill(dst_base + (piece * kTileKV + key) * 16,
                          k_data + koff + st * 128 + piece * 16, valid);
      }
    } else if ((warp >> 1) == st - 4) {
      // smem [32 pieces][16 keys] x 16 B, loaded by the two warps that own these keys
      const uint32_t kk = key & 15u;
#pragma unroll
      for (uint32_t j = 0; j < 8; ++j) {
        const uint32_t piece = part + 4 * j;
        cp_async_16_zfill(dst_base + (piece * 16 + kk) * 16, v_data + voff + piece * 16, valid);
      }
    }
  }
};

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
  const uint32_t gq = warp >> 2;  // row group
  const uint32_t w = warp & 3u;   // KV/VO warp within the group (warp_kv_idx)
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

  // Work bounds. The CTA loads K/V for its 64 rows; group gq computes exactly the 32-row
  // kernel's bounds at virtual tile q32 (CTA_TILE_Q = 32). The planner only runs this
  // kernel without a sliding window, where kv_start_idx is 0 for both groups.
  const uint32_t q32 = 2 * qo_tile_idx + gq;
  const uint32_t qo_upper_bound64 =
      min(qo_len, ceil_div((qo_tile_idx + 1) * kTileQ, group_size));
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
  const uint32_t g_num_iterations = ceil_div(
      (MASK_MODE == MaskMode::kCausal
           ? min(chunk_size, sub_if_greater_or_zero(
                                 kv_len - qo_len + ceil_div((q32 + 1) * 32, group_size),
                                 chunk_start))
           : chunk_size),
      kTileKV);
  const uint32_t g_window_iteration =
      ceil_div(sub_if_greater_or_zero(kv_len + ceil_div((q32 + 1) * 32, group_size),
                                      qo_len + window_left + chunk_start),
               kTileKV);
  const uint32_t g_mask_iteration =
      (MASK_MODE == MaskMode::kCausal
           ? min(chunk_size, sub_if_greater_or_zero(
                                 kv_len + ceil_div(q32 * 32, group_size) - qo_len, chunk_start))
           : chunk_size) /
      kTileKV;

  const uint32_t smem_base = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  const uint32_t q_s = smem_base + kQOff, ring_s = smem_base + kRingOff;
  const uint32_t p_s = smem_base + kPOff;
  float* md_max = reinterpret_cast<float*>(smem + kMDOff) + gq * 4 * 32;
  float* md_sum = reinterpret_cast<float*>(smem + kMDOff) + 2 * 4 * 32 + gq * 4 * 32;

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  // ------------------------------------------------------------- loaders
  const paged_kv_t<DTypeKV, IdType>& paged_kv = params.paged_kv;
  const IdType last_indptr = paged_kv.indptr[paged_kv.batch_size];
  const uint32_t packed_page_iter_base =
      paged_kv.indptr[request_idx] * paged_kv.page_size + chunk_start;
  RingLoader<SAME_KV_STRIDES, DTypeKV, IdType> loader;
  loader.k_data = paged_kv.k_data;
  loader.v_data = paged_kv.v_data;
  loader.koff = 0;
  loader.voff = 0;
  loader.key = 8 * warp + (lane & 7u);
  loader.part = lane >> 3;
  loader.warp = warp;
  loader.ring_s = ring_s;
  loader.valid = false;

  // Q tile: [64 chunks][64 rows] x 16 B; rows past the request are zero.
  {
    const size_t q_stride_n = params.q_stride_n, q_stride_h = params.q_stride_h;
    const DTypeQ* q_base = params.q + size_t(params.q_indptr[request_idx]) * q_stride_n +
                           size_t(kv_head_idx * group_size) * q_stride_h;
    const uint32_t row = loader.key;  // 8 warps x 8 rows
    uint32_t qi, hi;
    group_size.divmod(qo_tile_idx * kTileQ + row, qi, hi);
    const bool valid = qi < qo_upper_bound64;
    const DTypeQ* src = q_base + (valid ? (qi * q_stride_n + hi * q_stride_h) : 0);
#pragma unroll
    for (uint32_t i = 0; i < 16; ++i) {
      const uint32_t chunk = 4 * i + loader.part;
      cp_async_16_zfill(q_s + (chunk * kTileQ + row) * 16, src + chunk * 8, valid);
    }
  }
  // Prologue: sub-tiles 0 and 1 in flight (Q rides in the first group).
  if (num_iterations > 0) {
    loader.set_tile(paged_kv, packed_page_iter_base, last_indptr, kv_head_idx, chunk_size, 0);
    loader.issue(0, 0);
  }
  cp_async_commit();
  if (num_iterations > 0) {
    loader.issue(1, 1);
  }
  cp_async_commit();
  uint32_t ld_slot = 2;

  // ------------------------------------------------------------- per-thread fragments
  const uint32_t frag_row = (lane & 7u) + ((lane >> 3) & 1u) * 8;
  const uint32_t q_lane[2] = {
      q_s + ((lane >> 4) * kTileQ + 32 * gq + 0 + frag_row) * 16,
      q_s + ((lane >> 4) * kTileQ + 32 * gq + 16 + frag_row) * 16};
  const uint32_t p_lane[2] = {
      p_s + ((lane >> 4) * kTileQ + 32 * gq + 0 + frag_row) * 16,
      p_s + ((lane >> 4) * kTileQ + 32 * gq + 16 + frag_row) * 16};
  // K B-fragment words: key 16 w + 8 nt + g, bytes 2t, 2t+1 (word t/2) and 2t+8, 2t+9.
  const uint32_t k_lane = ((16 * w + g) * 16) + 4 * (t >> 1);
  [[maybe_unused]] const uint32_t k_lane128 = (16 * w + g) * 16;
  const uint32_t k_sel = (t & 1u) ? 0x7632u : 0x5410u;
  // V ldmatrix.trans: keys frag_row, piece 8 w + (lane >> 4) (+ 2 pp).
  const uint32_t v_lane = ((8 * w + (lane >> 4)) * 16 + frag_row) * 16;

  uint32_t row_q[2][2], row_h[2][2];
#pragma unroll
  for (uint32_t mq = 0; mq < 2; ++mq) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      group_size.divmod(q32 * 32 + 16 * mq + g + 8 * j, row_q[mq][j], row_h[mq][j]);
    }
  }

  float o[2][8][2][4];
#pragma unroll
  for (uint32_t mq = 0; mq < 2; ++mq) {
#pragma unroll
    for (uint32_t lp = 0; lp < 8; ++lp) {
#pragma unroll
      for (uint32_t p = 0; p < 2; ++p) {
#pragma unroll
        for (uint32_t r = 0; r < 4; ++r) o[mq][lp][p][r] = 0.f;
      }
    }
  }
  float m[2][2], d[2][2];
#pragma unroll
  for (uint32_t mq = 0; mq < 2; ++mq) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      m[mq][j] = -math::inf;
      d[mq][j] = 0.f;
    }
  }
  float s[2][2][4];
  const float sm_scale = variant.sm_scale_log2;
  const uint32_t bar_id = 1 + gq;
  uint32_t cons_slot = 0;

#pragma unroll 1
  for (uint32_t iter = 0; iter < num_iterations; ++iter) {
    const bool active = iter < g_num_iterations;
#pragma unroll
    for (uint32_t st = 0; st < kSubTiles; ++st) {
      // Sub-tile st landed; every warp is past sub-tile st - 1, so its slot is free for
      // sub-tile st + 2.
      cp_async_wait<1>();
      __syncthreads();
      {
        const uint32_t ltile = iter + ((st + 2) / kSubTiles);
        if (ltile < num_iterations) {
          if ((st + 2) % kSubTiles == 0) {
            loader.set_tile(paged_kv, packed_page_iter_base, last_indptr, kv_head_idx,
                            chunk_size, ltile);
          }
          loader.issue((st + 2) % kSubTiles, ld_slot);
        }
        cp_async_commit();
        ld_slot = (ld_slot == kNumSlots - 1) ? 0 : ld_slot + 1;
      }
      const uint32_t sb = ring_s + cons_slot * kSlotBytes;
      cons_slot = (cons_slot == kNumSlots - 1) ? 0 : cons_slot + 1;
      if (!active) {
        continue;
      }
      if (st < 4) {
        // ---------------- Q.K^T, d-steps 8 st .. 8 st + 7
#pragma unroll
        for (uint32_t sl = 0; sl < 8; ++sl) {
          const uint32_t mma_d = 8 * st + sl;
          uint32_t a[2][4];
#pragma unroll
          for (uint32_t mq = 0; mq < 2; ++mq) {
            ldsm_x4(q_lane[mq] + mma_d * 2 * kTileQ * 16, a[mq]);
          }
          uint32_t b[2][2];
#pragma unroll
          for (uint32_t nt = 0; nt < 2; ++nt) {
#if D512T64_K_LDS128
            // One 16-byte load of the key's piece; lanes keep words t/2 and t/2 + 2.
            const uint4 kw = lds_u128(sb + k_lane128 + (sl * kTileKV + 8 * nt) * 16);
            const uint32_t lo = (t & 2u) ? kw.y : kw.x;
            const uint32_t hi = (t & 2u) ? kw.w : kw.z;
            const uint32_t x = __byte_perm(lo, hi, k_sel);
#else
            const uint32_t addr = sb + k_lane + (sl * kTileKV + 8 * nt) * 16;
            const uint32_t x = __byte_perm(lds_u32(addr), lds_u32(addr + 8), k_sel);
#endif
            fp8x4_to_16b<DTypeQ, DTypeKV>(x, b[nt]);
          }
          if (mma_d == 0) {
#pragma unroll
            for (uint32_t mq = 0; mq < 2; ++mq) {
#pragma unroll
              for (uint32_t nt = 0; nt < 2; ++nt) {
#pragma unroll
                for (uint32_t r = 0; r < 4; ++r) s[mq][nt][r] = 0.f;
              }
            }
          }
#pragma unroll
          for (uint32_t mq = 0; mq < 2; ++mq) {
#pragma unroll
            for (uint32_t nt = 0; nt < 2; ++nt) {
              mma_m16n8k16<DTypeQ>(s[mq][nt], a[mq], b[nt]);
            }
          }
        }
        if (st == 3) {
          // ---------------- logits transform + mask (as logits_transform / logits_mask)
          const uint32_t kv_idx_base = chunk_start + iter * kTileKV + 16 * w + 2 * t;
#pragma unroll
          for (uint32_t mq = 0; mq < 2; ++mq) {
#pragma unroll
            for (uint32_t nt = 0; nt < 2; ++nt) {
#pragma unroll
              for (uint32_t r = 0; r < 4; ++r) {
                const uint32_t j = r >> 1;
                const uint32_t kv_idx = kv_idx_base + 8 * nt + (r & 1u);
                if (kv_idx < chunk_end) {
                  s[mq][nt][r] = variant.LogitsTransform(
                      params, s[mq][nt][r], request_idx, row_q[mq][j], kv_idx,
                      kv_head_idx * group_size + row_h[mq][j], kv_head_idx);
                }
              }
            }
          }
          if (MASK_MODE == MaskMode::kCustom || iter >= g_mask_iteration ||
              iter < g_window_iteration) {
#pragma unroll
            for (uint32_t mq = 0; mq < 2; ++mq) {
#pragma unroll
              for (uint32_t nt = 0; nt < 2; ++nt) {
#pragma unroll
                for (uint32_t r = 0; r < 4; ++r) {
                  const uint32_t j = r >> 1;
                  const uint32_t kv_idx = kv_idx_base + 8 * nt + (r & 1u);
                  const bool keep =
                      (!(MASK_MODE == MaskMode::kCausal
                             ? (kv_idx + qo_len > kv_len + row_q[mq][j] || kv_idx >= chunk_end)
                             : kv_idx >= chunk_end)) &&
                      variant.LogitsMask(params, request_idx, row_q[mq][j], kv_idx,
                                         kv_head_idx * group_size + row_h[mq][j], kv_head_idx);
                  s[mq][nt][r] = keep ? s[mq][nt][r] : -math::inf;
                }
              }
            }
          }
          // ---------------- softmax: vosplit_softmax_store_p for this group
          // (1) per-warp tile row max
#pragma unroll
          for (uint32_t mq = 0; mq < 2; ++mq) {
#pragma unroll
            for (uint32_t j = 0; j < 2; ++j) {
              float mt = -math::inf;
              mt = max(mt, max(max(s[mq][0][j * 2 + 0], s[mq][0][j * 2 + 1]),
                               max(s[mq][1][j * 2 + 0], s[mq][1][j * 2 + 1])));
              mt = max(mt, math::shfl_xor_sync(mt, 0x2));
              mt = max(mt, math::shfl_xor_sync(mt, 0x1));
              if (t == 0) {
                md_max[w * 32 + mq * 16 + j * 8 + g] = mt;
              }
            }
          }
          group_barrier(bar_id);
          // (2) cross-warp max, running m / d, o_scale (vosplit_compute_pv applies it to o)
          float o_scale[2][2];
#pragma unroll
          for (uint32_t mq = 0; mq < 2; ++mq) {
#pragma unroll
            for (uint32_t j = 0; j < 2; ++j) {
              float mt = -math::inf;
#pragma unroll
              for (uint32_t w2 = 0; w2 < 4; ++w2) {
                mt = max(mt, md_max[w2 * 32 + mq * 16 + j * 8 + g]);
              }
              const float m_prev = m[mq][j];
              const float m_new = max(m_prev, mt);
              const float m_new_scaled =
                  max(m_new * sm_scale, -cuda::std::numeric_limits<float>::max());
              o_scale[mq][j] = math::ptx_exp2(m_prev * sm_scale - m_new_scaled);
              m[mq][j] = m_new;
            }
          }
#pragma unroll
          for (uint32_t mq = 0; mq < 2; ++mq) {
#pragma unroll
            for (uint32_t lp = 0; lp < 8; ++lp) {
#pragma unroll
              for (uint32_t p = 0; p < 2; ++p) {
                o[mq][lp][p][0] *= o_scale[mq][0];
                o[mq][lp][p][1] *= o_scale[mq][0];
                o[mq][lp][p][2] *= o_scale[mq][1];
                o[mq][lp][p][3] *= o_scale[mq][1];
              }
            }
          }
          // (3) P = exp2(s * scale - m), BF16 into p_smem, partial row sums
#pragma unroll
          for (uint32_t mq = 0; mq < 2; ++mq) {
#pragma unroll
            for (uint32_t j = 0; j < 2; ++j) {
              const float m_new =
                  max(m[mq][j] * sm_scale, -cuda::std::numeric_limits<float>::max());
              float dsum = 0.f;
              float p0[2], p1[2];
#pragma unroll
              for (uint32_t k = 0; k < 2; ++k) {
                p0[k] = math::ptx_exp2(s[mq][0][j * 2 + k] * sm_scale - m_new);
                p1[k] = math::ptx_exp2(s[mq][1][j * 2 + k] * sm_scale - m_new);
                dsum += p0[k] + p1[k];
              }
              const uint32_t row = 32 * gq + 16 * mq + 8 * j + g;
              sts_u32(p_s + ((2 * w) * kTileQ + row) * 16 + 4 * t,
                      pack2<DTypeQ>(p0[0], p0[1]));
              sts_u32(p_s + ((2 * w + 1) * kTileQ + row) * 16 + 4 * t,
                      pack2<DTypeQ>(p1[0], p1[1]));
              dsum += math::shfl_xor_sync(dsum, 0x2);
              dsum += math::shfl_xor_sync(dsum, 0x1);
              if (t == 0) {
                md_sum[w * 32 + mq * 16 + j * 8 + g] = dsum;
              }
            }
          }
          group_barrier(bar_id);
          // (4) cross-warp row sum
#pragma unroll
          for (uint32_t mq = 0; mq < 2; ++mq) {
#pragma unroll
            for (uint32_t j = 0; j < 2; ++j) {
              float dt = 0.f;
#pragma unroll
              for (uint32_t w2 = 0; w2 < 4; ++w2) {
                dt += md_sum[w2 * 32 + mq * 16 + j * 8 + g];
              }
              // d = d * o_scale + dt, rounded as the 32-row kernel's compiled code does.
#if D512T64_D_UPDATE == 1
              d[mq][j] = __fmaf_rn(d[mq][j], o_scale[mq][j], dt);
#else
              d[mq][j] = __fadd_rn(__fmul_rn(d[mq][j], o_scale[mq][j]), dt);
#endif
            }
          }
        }
      } else {
        // ---------------- O += P . V, keys 16 (st - 4) .. + 15
        const uint32_t c = st - 4;
        uint32_t a[2][4];
#pragma unroll
        for (uint32_t mq = 0; mq < 2; ++mq) {
          ldsm_x4(p_lane[mq] + 2 * c * kTileQ * 16, a[mq]);
        }
#pragma unroll
        for (uint32_t pp = 0; pp < 4; ++pp) {
          uint32_t vr[4];
          ldsm_x4_trans(sb + v_lane + pp * 2 * 16 * 16, vr);
#pragma unroll
          for (uint32_t h = 0; h < 2; ++h) {
            uint32_t be[2], bo[2];
            fp8x4_to_16b<DTypeQ, DTypeKV>(__byte_perm(vr[2 * h], vr[2 * h + 1], 0x6420), be);
            fp8x4_to_16b<DTypeQ, DTypeKV>(__byte_perm(vr[2 * h], vr[2 * h + 1], 0x7531), bo);
#pragma unroll
            for (uint32_t mq = 0; mq < 2; ++mq) {
              mma_m16n8k16<DTypeQ>(o[mq][2 * pp + h][0], a[mq], be);
              mma_m16n8k16<DTypeQ>(o[mq][2 * pp + h][1], a[mq], bo);
            }
          }
        }
      }
    }
  }
  cp_async_wait<0>();

  // ------------------------------------------------------------- epilogue (vosplit_write_o)
#pragma unroll
  for (uint32_t mq = 0; mq < 2; ++mq) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      if (m[mq][j] != -math::inf) {
        m[mq][j] *= sm_scale;
      }
    }
  }
  const uint32_t num_kv_chunks = ceil_div(min(kv_len_safe, window_left + 32), kv_chunk_size);
  const size_t o_stride_n = size_t(num_qo_heads) * kHeadDim, o_stride_h = kHeadDim;
  DTypeO* o_base =
      partition_kv
          ? params.o + (size_t(params.o_indptr[request_idx]) + kv_tile_idx) * o_stride_n +
                size_t(kv_head_idx * group_size) * o_stride_h
          : params.o + size_t(params.o_indptr[request_idx]) * o_stride_n +
                size_t(kv_head_idx * group_size) * o_stride_h;
  const size_t o_row_stride = partition_kv ? size_t(num_kv_chunks) * o_stride_n : o_stride_n;
#pragma unroll
  for (uint32_t mq = 0; mq < 2; ++mq) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      const float rd = (d[mq][j] > 0.f) ? math::ptx_rcp(d[mq][j]) : 0.f;
      if (row_q[mq][j] < qo_len) {
        DTypeO* orow = o_base + row_q[mq][j] * o_row_stride + row_h[mq][j] * o_stride_h +
                       128 * w + 4 * t;
#pragma unroll
        for (uint32_t lp = 0; lp < 8; ++lp) {
          store_4<DTypeO>(orow + 16 * lp, o[mq][lp][0][2 * j] * rd, o[mq][lp][1][2 * j] * rd,
                          o[mq][lp][0][2 * j + 1] * rd, o[mq][lp][1][2 * j + 1] * rd);
        }
      }
    }
  }
  if (params.lse != nullptr && w == 0 && t == 0) {
#pragma unroll
    for (uint32_t mq = 0; mq < 2; ++mq) {
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        if (row_q[mq][j] < qo_len) {
          const uint32_t qo_head_idx = kv_head_idx * group_size + row_h[mq][j];
          const float val = math::ptx_log2(d[mq][j]) + float(m[mq][j]);
          if (partition_kv) {
            params.lse[(size_t(params.o_indptr[request_idx]) + row_q[mq][j] * num_kv_chunks +
                        kv_tile_idx) *
                           num_qo_heads +
                       qo_head_idx] = val;
          } else {
            params.lse[(size_t(params.o_indptr[request_idx]) + row_q[mq][j]) * num_qo_heads +
                       qo_head_idx] = val;
          }
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
    if (tmp_v == nullptr) {
      params.partition_kv = false;
      if (enable_pdl) {
        FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, params));
      } else {
        void* args[] = {(void*)&params};
        FLASHINFER_CUDA_CALL(
            cudaLaunchKernel((void*)kernel, nblks, nthrs, args, kSmemBytes, stream));
      }
    } else {
      params.partition_kv = true;
      auto o = params.o;
      auto lse = params.lse;
      params.o = tmp_v;
      params.lse = tmp_s;
      if (enable_pdl) {
        FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, params));
      } else {
        void* args[] = {(void*)&params};
        FLASHINFER_CUDA_CALL(
            cudaLaunchKernel((void*)kernel, nblks, nthrs, args, kSmemBytes, stream));
      }
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
