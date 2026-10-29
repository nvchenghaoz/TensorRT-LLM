/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
// FLUX.3 (add-only): the NVFP4 block-tail pair (ff.down_proj, K=9216, and attn_out, K=3072, both
// N=3072) as ONE persistent SM120 CUTLASS launch WITH the block's gated residual in the attn_out
// tiles' epilogue:
//
//   mlp_y = BF16(alpha0 * acc_down)                        (group 0 tiles, scratch output)
//   out   = x + gate * (BF16(alpha1 * acc_attn) + mlp_y)   (group 1 tiles, BF16 round after
//                                                           every op, as Inductor's
//                                                           emulate_precision_casts kernel)
//
// gate is a row table [R, N] (BF16) picked per output row by a uint8 row index (or row 0 when
// there is no index): the [1, 2, D] table + bool select of the single blocks and the [1, 1, D]
// row of the mode blocks.
//
// Mainloop and per-tile MMA sequence: the same CUTLASS SM120 OpClassBlockScaledTensorOp ptr-array
// grouped GEMM as flux3_fp4_lpt_pair_sm120.cu (128x128x128 FP4 tiles, 1x1x1 cluster, Cooperative),
// so BF16(alpha * acc) of each tile is bitwise the stock kernel's. The epilogue is register-only
// (operator-style, Sm90TmaWarpSpecializedAdapter, no smem): each quad of lanes transposes its
// m16n8 fragments with 3 shuffles so every lane owns 8 contiguous columns of one row, and all
// global accesses are 16-B row segments (64 B contiguous per quad).
// The scheduler warp L2-prefetches (cp.async.bulk.prefetch.L2) the x and mlp_y rows of every
// attn_out tile it claims, one tile ahead of the epilogue that reads them.
//
// Ordering. All tiles, including each CTA's first, are claimed from one global atomic counter
// (group 0 = down before group 1 = attn_out), so a tile is only ever owned by a running CTA and
// every down tile is claimed before any attn_out tile. Warp w of the attn_out tile (m, n) reads
// exactly the mlp_y elements that warp w of the down tile (m, n) wrote (same fragment layout), so
// the handoff is per warp. The down warp's flag is published at the start of the CTA's NEXT
// epilogue (or in store_tail at kernel end), when its stores have drained: all lanes membar.gl,
// __syncwarp, lane 0 st.release.gpu of the launch epoch to flags[(m, n), w]. The attn_out warp
// spins on ld.acquire.gpu of that flag until it equals the epoch, then reads mlp_y with
// ld.global.cg. Deadlock-free: a CTA processes its tiles in claim order, never waits inside a down
// tile, and publishes its pending down tile before any wait, so every awaited flag belongs to a
// running CTA that publishes it after a finite mainloop.
// The epoch is a device counter in the workspace that the args kernel (stream ordered before the
// GEMM, PDL) increments on every launch; flags hold the epoch of their last write, so no reset is
// needed and CUDA-graph replays are safe. The workspace must be zero-initialized once.
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "tvm_ffi_utils.h"

namespace flux3_lptr {
using ProblemShape = cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;
}  // namespace flux3_lptr

namespace cutlass::gemm::kernel::detail {

// PersistentTileSchedulerSm90Group whose every tile (the first included) is claimed from a global
// atomic counter, with a one-stage scheduler pipeline (a CTA claims at most one tile ahead).
template <class GroupProblemShape, int SchedulerPipelineStageCount>
class Flux3LptResidualScheduler
    : public PersistentTileSchedulerSm90Group<GroupProblemShape, SchedulerPipelineStageCount> {
  using Base = PersistentTileSchedulerSm90Group<GroupProblemShape, SchedulerPipelineStageCount>;

 public:
  static constexpr int LptStages = 1;
  using WorkTileInfo = typename Base::WorkTileInfo;
  using SchedulerResponse = WorkTileInfo;
  using Pipeline = PipelineAsync<LptStages>;
  using PipelineStorage = cutlass::PipelineDetail::PipelineAsyncSharedStorage<LptStages>;
  using PipelineState = cutlass::PipelineDetail::PipelineAsyncPipelineState<LptStages>;
  using BaseParams = typename Base::Params;
  using BaseArguments = typename Base::Arguments;
  using RasterOrder = typename Base::RasterOrder;
  using RasterOrderOptions = typename Base::RasterOrderOptions;

  struct Arguments : BaseArguments {
    uint32_t* tile_counter = nullptr;
    // L2 prefetch of each claimed group-1 tile's epilogue inputs (x and mlp_y rows, bf16 [M, ld]).
    void const* prefetch_x = nullptr;
    void const* prefetch_y = nullptr;
    int64_t prefetch_ld = 0;
    int prefetch_rows = 0;
  };

  struct Params : BaseParams {
    uint32_t* tile_counter = nullptr;
    void const* prefetch_x = nullptr;
    void const* prefetch_y = nullptr;
    int64_t prefetch_ld = 0;
    int prefetch_rows = 0;
  };

  template <class TileShape, class ClusterShape>
  static Params to_underlying_arguments(GroupProblemShape problem_shapes, TileShape tile_shape,
                                        ClusterShape cluster_shape,
                                        KernelHardwareInfo const& hw_info,
                                        Arguments const& arguments, void* workspace = nullptr,
                                        const uint32_t epilogue_subtile = 1,
                                        uint32_t ktile_start_alignment_count = 1u) {
    Params params;
    static_cast<BaseParams&>(params) = Base::to_underlying_arguments(
        problem_shapes, tile_shape, cluster_shape, hw_info,
        static_cast<BaseArguments const&>(arguments), workspace, epilogue_subtile,
        ktile_start_alignment_count);
    params.tile_counter = arguments.tile_counter;
    params.prefetch_x = arguments.prefetch_x;
    params.prefetch_y = arguments.prefetch_y;
    params.prefetch_ld = arguments.prefetch_ld;
    params.prefetch_rows = arguments.prefetch_rows;
    return params;
  }

  Flux3LptResidualScheduler() = default;

  // Constructed by every thread of the CTA at the same point of the kernel (before the warp
  // roles split), after griddepcontrol.wait: thread 0 claims the CTA's first tile.
  CUTLASS_DEVICE explicit Flux3LptResidualScheduler(Params const& params,
                                                    SchedulerResponse* response_ptr)
      : Base(static_cast<BaseParams const&>(params), response_ptr),
        tile_counter_(params.tile_counter),
        prefetch_x_(static_cast<char const*>(params.prefetch_x)),
        prefetch_y_(static_cast<char const*>(params.prefetch_y)),
        prefetch_ld_bytes_(params.prefetch_ld * 2),
        prefetch_rows_(params.prefetch_rows) {
#if defined(__CUDA_ARCH__)
    // Response slot 6 is unused by the one-stage pipeline (slot 0).
    uint32_t* first = reinterpret_cast<uint32_t*>(reinterpret_cast<WorkTileInfo*>(response_ptr) + 6);
    if (threadIdx.x == 0) {
      *first = atomicAdd(tile_counter_, 1u);
    }
    __syncthreads();
    lpt_linear_idx_ = uint64_t(*first);
#endif
  }

  template <class ClusterShape>
  CUTLASS_DEVICE WorkTileInfo initial_work_tile_info(ClusterShape) {
    return this->get_current_work_for_linear_idx(lpt_linear_idx_);
  }

  template <typename TileSchedulerPipeline, typename TileSchedulerPipelineState,
            typename CallbackBeforeCommit = WorkTileInfo (*)(WorkTileInfo)>
  CUTLASS_DEVICE auto advance_to_next_work(
      TileSchedulerPipeline& scheduler_pipeline,
      TileSchedulerPipelineState scheduler_pipe_producer_state, uint32_t advance_count = 1,
      CallbackBeforeCommit callback_before_commit = [](WorkTileInfo info) { return info; }) {
    // Claim only once the response slot is free, so a CTA runs at most LptStages tiles ahead.
    scheduler_pipeline.producer_acquire(scheduler_pipe_producer_state);
    uint32_t claimed = 0;
    if ((threadIdx.x % 32) == 0) {
      claimed = atomicAdd(tile_counter_, 1u);
    }
    claimed = __shfl_sync(0xffffffff, claimed, 0);
    lpt_linear_idx_ = uint64_t(claimed);
    auto work_tile = this->get_current_work_for_linear_idx(lpt_linear_idx_);
    using WorkTileWithCallbackInfo = decltype(callback_before_commit(work_tile));
    WorkTileWithCallbackInfo info = work_tile;
    if (info.is_valid()) {
      info = callback_before_commit(work_tile);
    }
    if (cute::elect_one_sync()) {
      reinterpret_cast<WorkTileWithCallbackInfo*>(
          this->response_ptr_)[scheduler_pipe_producer_state.index()] = info;
      cutlass::arch::fence_view_async_shared();
      scheduler_pipeline.producer_commit(scheduler_pipe_producer_state);
    }
#if defined(__CUDA_ARCH__)
    // This CTA runs the claimed tile after its current one: pull the attn_out tile's epilogue
    // inputs (x and mlp_y, 128 rows x 256 B each) into L2 meanwhile. A hint only (L2 is the
    // coherence point, so a prefetched line is never stale); mlp_y's down tile is waited on in the
    // epilogue as usual.
    if (work_tile.is_valid() && work_tile.L_idx == 1 && prefetch_x_ != nullptr) {
      int const lane = threadIdx.x % 32;
      int64_t const col_bytes = int64_t(work_tile.N_idx) * 128 * 2;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < 4; ++i) {
        int const r = work_tile.M_idx * 128 + lane + 32 * i;
        if (r < prefetch_rows_) {
          int64_t const off = int64_t(r) * prefetch_ld_bytes_ + col_bytes;
          asm volatile("cp.async.bulk.prefetch.L2.global [%0], 256;" ::"l"(prefetch_x_ + off)
                       : "memory");
          asm volatile("cp.async.bulk.prefetch.L2.global [%0], 256;" ::"l"(prefetch_y_ + off)
                       : "memory");
        }
      }
    }
#endif
    return cute::make_tuple(info, true);
  }

 private:
  uint32_t* tile_counter_ = nullptr;
  char const* prefetch_x_ = nullptr;
  char const* prefetch_y_ = nullptr;
  int64_t prefetch_ld_bytes_ = 0;
  int prefetch_rows_ = 0;
  uint64_t lpt_linear_idx_ = 0;
};

template <class TileShape, class ClusterShape, uint32_t SchedulerPipelineStageCount>
struct TileSchedulerSelector<GroupScheduler, arch::Sm120, TileShape, ClusterShape,
                             SchedulerPipelineStageCount, flux3_lptr::ProblemShape> {
  using Scheduler = Flux3LptResidualScheduler<flux3_lptr::ProblemShape,
                                              static_cast<int>(SchedulerPipelineStageCount)>;
};

}  // namespace cutlass::gemm::kernel::detail

namespace flux3_lptr {

using namespace cute;

constexpr int kMmaWarps = 8;  // Cooperative: two 4-warp consumer warpgroups share each tile

__device__ __forceinline__ float add_rn(float a, float b) {
  float d;
  asm("add.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
__device__ __forceinline__ float mul_rn(float a, float b) {
  float d;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ float bf_lo(uint32_t u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bf_hi(uint32_t u) { return __uint_as_float(u & 0xFFFF0000u); }
__device__ __forceinline__ uint32_t ld_acquire_gpu(uint32_t const* p) {
  uint32_t v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release_gpu(uint32_t* p, uint32_t v) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint4 ld_cg_v4(void const* p) {
  uint4 v;
  asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}
__device__ __forceinline__ uint32_t sel4(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         int i) {
  uint32_t const lo = (i & 1) ? a1 : a0;
  uint32_t const hi = (i & 1) ? a3 : a2;
  return (i & 2) ? hi : lo;
}
__device__ __forceinline__ int sel4i(int a0, int a1, int a2, int a3, int i) {
  int const lo = (i & 1) ? a1 : a0;
  int const hi = (i & 1) ? a3 : a2;
  return (i & 2) ? hi : lo;
}
// Quad transpose. In: lane q of each 4-lane quad holds v[j] = its bf16x2 column pair (2q, 2q+1)
// of m16n8 atom j (one row). Out: lane q holds w[i] = column pair i of atom q, i.e. atom q's
// 8 contiguous columns (16 B). Three shuffles; every lane of the warp must call it.
__device__ __forceinline__ uint4 quad_transpose(uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3,
                                                int q) {
  uint32_t const r0 = sel4(v0, v1, v2, v3, q);
  uint32_t const r1 = __shfl_sync(0xffffffffu, sel4(v0, v1, v2, v3, (q - 1) & 3), (q + 1) & 3, 4);
  uint32_t const r2 = __shfl_sync(0xffffffffu, sel4(v0, v1, v2, v3, (q - 2) & 3), (q + 2) & 3, 4);
  uint32_t const r3 = __shfl_sync(0xffffffffu, sel4(v0, v1, v2, v3, (q - 3) & 3), (q + 3) & 3, 4);
  // r_k = w[(q + k) & 3]  =>  w[i] = r_{(i - q) & 3}
  uint4 w;
  w.x = sel4(r0, r1, r2, r3, (0 - q) & 3);
  w.y = sel4(r0, r1, r2, r3, (1 - q) & 3);
  w.z = sel4(r0, r1, r2, r3, (2 - q) & 3);
  w.w = sel4(r0, r1, r2, r3, (3 - q) & 3);
  return w;
}

// bf16(x + bf16(g * bf16(ya + ym))) for one bf16 element (f32 ops, .rn, no FMA contraction).
__device__ __forceinline__ float residual_f32(float ya, float ym, float g, float x) {
  float const o = __bfloat162float(__float2bfloat16_rn(add_rn(ya, ym)));
  float const p = __bfloat162float(__float2bfloat16_rn(mul_rn(g, o)));
  return add_rn(x, p);
}
__device__ __forceinline__ uint32_t residual_x2(uint32_t a, uint32_t m, uint32_t g, uint32_t x) {
  return pack_bf16x2(residual_f32(bf_lo(a), bf_lo(m), bf_lo(g), bf_lo(x)),
                     residual_f32(bf_hi(a), bf_hi(m), bf_hi(g), bf_hi(x)));
}

template <class StrideCD_>
class PairResidualEpilogue {
 public:
  using EpilogueSchedule = cutlass::epilogue::NoSmemWarpSpecialized;
  using DispatchPolicy = EpilogueSchedule;
  using ElementOutput = cutlass::bfloat16_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementScalar = float;
  using ElementC = cutlass::bfloat16_t;
  using StrideC = StrideCD_;
  using InternalStrideC = StrideCD_;
  using ElementD = cutlass::bfloat16_t;
  using StrideD = StrideCD_;
  using InternalStrideD = StrideCD_;
  using GmemTiledCopyC = void;
  using GmemTiledCopyD = void;

  struct SharedStorage {};
  using TensorStorage = SharedStorage;

  struct Arguments {
    __nv_bfloat16* mlp_y = nullptr;        // [M, N] group 0 output (scratch)
    __nv_bfloat16* out = nullptr;          // [M, N] fused residual output
    __nv_bfloat16 const* x = nullptr;      // [M, N] residual input
    __nv_bfloat16 const* gate = nullptr;   // [R, N] gate row table
    uint8_t const* gate_index = nullptr;   // [M] row of the table per output row, or nullptr
    float const* alpha0 = nullptr;         // device scalar (down)
    float const* alpha1 = nullptr;         // device scalar (attn_out)
    uint32_t* flags = nullptr;             // [tiles_m * tiles_n * kMmaWarps]
    uint32_t const* epoch = nullptr;       // this launch's epoch (written by the args kernel)
    int ld = 0;                            // N (row stride of every [*, N] tensor)
    int tiles_n = 0;                       // N / 128
    int diag = 0;  // diagnostics only (FLUX3_LPTR_DIAG): 1 no input loads, 2 no flag wait,
                   // 4 group 1 stores attn_y only (wrong outputs; timing attribution)
  };
  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(ProblemShape const&, Arguments const& args,
                                                  void*) {
    return args;
  }
  template <class ProblemShape, class... Rest>
  static size_t get_workspace_size(ProblemShape const&, Arguments const&, Rest...) {
    return 0;
  }
  template <class ProblemShape, class... Rest>
  static cutlass::Status initialize_workspace(ProblemShape const&, Arguments const&, void*,
                                              cudaStream_t, Rest...) {
    return cutlass::Status::kSuccess;
  }
  template <class ProblemShape>
  static bool can_implement(ProblemShape const&, Arguments const&) {
    return true;
  }

  CUTLASS_HOST_DEVICE
  PairResidualEpilogue(Params const& params_, SharedStorage const& = SharedStorage())
      : params(params_) {}

  CUTLASS_DEVICE bool is_source_needed() { return false; }

  template <class ProblemShapeMNKL, class BlockShapeMNK, class BlockCoordMNKL, class FrgEngine,
            class FrgLayout, class TiledMma, class ResidueMNK>
  CUTLASS_DEVICE void operator()(ProblemShapeMNKL problem_shape_mnkl, BlockShapeMNK,
                                 BlockCoordMNKL blk_coord_mnkl,
                                 cute::Tensor<FrgEngine, FrgLayout> const& accumulators,
                                 TiledMma tiled_mma, ResidueMNK, int thread_idx, char*) {
    constexpr int BM = size<0>(BlockShapeMNK{});
    constexpr int BN = size<1>(BlockShapeMNK{});
    static_assert(size<0>(FrgLayout{}) == 4, "m16n8 C fragment expected");
    constexpr int MMA_M = size<1>(FrgLayout{});
    constexpr int MMA_N = size<2>(FrgLayout{});

    int const m_blk = static_cast<int>(get<0>(blk_coord_mnkl));
    int const n_blk = static_cast<int>(get<1>(blk_coord_mnkl));
    int const group = static_cast<int>(get<3>(blk_coord_mnkl));
    int const M = static_cast<int>(get<0>(problem_shape_mnkl));

    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    auto cD = make_identity_tensor(make_shape(Int<BM>{}, Int<BN>{}));
    auto tCcD = thr_mma.partition_C(cD);  // (4, MMA_M, MMA_N) -> (row, col) in tile

    int const row_base = m_blk * BM;
    int const col_base = n_blk * BN;
    int64_t const ld = params.ld;
    uint32_t const epoch = *params.epoch;
    int const warp = thread_idx >> 5;
    uint32_t* flag = params.flags +
                     (static_cast<int64_t>(m_blk) * params.tiles_n + n_blk) * kMmaWarps + warp;

    // Publish the previous down tile first: a warp never waits while holding an unpublished
    // down tile, so every awaited flag is published in finite time.
    release_pending();

    // Fragment chunk (mi, h, k): one row r = rows(mi, h) and the 4 atoms ni = 4k..4k+3. After a
    // quad transpose, lane q owns atom 4k+q's 8 contiguous columns starting at seg_col (16 B).
    int const q = thread_idx & 3;
    constexpr int KCH = MMA_N / 4;
    static_assert(MMA_N % 4 == 0, "quad transpose needs 4 atoms per chunk");
    auto chunk_row = [&](int mi, int h) { return row_base + get<0>(tCcD(2 * h, mi, 0)); };
    auto seg_col = [&](int mi, int h, int k) {
      return col_base - 2 * q +
             sel4i(get<1>(tCcD(2 * h, mi, 4 * k + 0)), get<1>(tCcD(2 * h, mi, 4 * k + 1)),
                   get<1>(tCcD(2 * h, mi, 4 * k + 2)), get<1>(tCcD(2 * h, mi, 4 * k + 3)), q);
    };
    // BF16(alpha * acc) of chunk (mi, h, k), transposed (bitwise the stock epilogue's values).
    auto chunk_bf16 = [&](float alpha, int mi, int h, int k) {
      return quad_transpose(
          pack_bf16x2(alpha * accumulators(2 * h, mi, 4 * k + 0),
                      alpha * accumulators(2 * h + 1, mi, 4 * k + 0)),
          pack_bf16x2(alpha * accumulators(2 * h, mi, 4 * k + 1),
                      alpha * accumulators(2 * h + 1, mi, 4 * k + 1)),
          pack_bf16x2(alpha * accumulators(2 * h, mi, 4 * k + 2),
                      alpha * accumulators(2 * h + 1, mi, 4 * k + 2)),
          pack_bf16x2(alpha * accumulators(2 * h, mi, 4 * k + 3),
                      alpha * accumulators(2 * h + 1, mi, 4 * k + 3)),
          q);
    };

    if (group == 0) {
      float const alpha = *params.alpha0;
      CUTLASS_PRAGMA_UNROLL
      for (int mi = 0; mi < MMA_M; ++mi) {
        CUTLASS_PRAGMA_UNROLL
        for (int h = 0; h < 2; ++h) {
          CUTLASS_PRAGMA_UNROLL
          for (int k = 0; k < KCH; ++k) {
            uint4 const w = chunk_bf16(alpha, mi, h, k);
            int const r = chunk_row(mi, h);
            if (r < M) {
              *reinterpret_cast<uint4*>(params.mlp_y + r * ld + seg_col(mi, h, k)) = w;
            }
          }
        }
      }
      // Published at the start of this CTA's next epilogue (or at kernel end, store_tail), when
      // the stores have long drained, so the fence does not stall this tile.
      pending_flag_ = flag;
      pending_epoch_ = epoch;
      return;
    }

    float const alpha = *params.alpha1;
    // ya = attn_y as main stores it, BF16(alpha * acc), transposed to 16-B row segments: the f32
    // accumulators die here, so the staged inputs below fit the kernel's 168-register budget.
    uint4 ya[MMA_M][2][KCH];
    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < MMA_M; ++mi) {
      CUTLASS_PRAGMA_UNROLL
      for (int h = 0; h < 2; ++h) {
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < KCH; ++k) {
          ya[mi][h][k] = chunk_bf16(alpha, mi, h, k);
        }
      }
    }
    if (params.diag & 4) {
      CUTLASS_PRAGMA_UNROLL
      for (int mi = 0; mi < MMA_M; ++mi) {
        CUTLASS_PRAGMA_UNROLL
        for (int h = 0; h < 2; ++h) {
          CUTLASS_PRAGMA_UNROLL
          for (int k = 0; k < KCH; ++k) {
            int const r = chunk_row(mi, h);
            if (r < M) {
              *reinterpret_cast<uint4*>(params.out + r * ld + seg_col(mi, h, k)) = ya[mi][h][k];
            }
          }
        }
      }
      return;
    }
    bool const do_loads = (params.diag & 1) == 0;
    // One MMA_M row chunk at a time: x, gate and mlp_y as 16-B row segments.
    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < MMA_M; ++mi) {
      uint4 xv[2][KCH];
      uint4 gv[2][KCH];
      uint4 ym[2][KCH];
      CUTLASS_PRAGMA_UNROLL
      for (int h = 0; h < 2; ++h) {
        int const r = chunk_row(mi, h);
        int const gi =
            (params.gate_index != nullptr && r < M) ? int(params.gate_index[r]) : 0;
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < KCH; ++k) {
          int const c = seg_col(mi, h, k);
          xv[h][k] = make_uint4(0u, 0u, 0u, 0u);
          gv[h][k] = make_uint4(0u, 0u, 0u, 0u);
          if (r < M && do_loads) {
            xv[h][k] = __ldg(reinterpret_cast<uint4 const*>(params.x + r * ld + c));
            gv[h][k] = __ldg(reinterpret_cast<uint4 const*>(params.gate + gi * ld + c));
          }
        }
      }
      if (mi == 0 && (params.diag & 2) == 0) {
        // The down tile (m, n) of this launch: written by the same warp index of its CTA.
        while (ld_acquire_gpu(flag) != epoch) {
          __nanosleep(64);
        }
      }
      CUTLASS_PRAGMA_UNROLL
      for (int h = 0; h < 2; ++h) {
        int const r = chunk_row(mi, h);
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < KCH; ++k) {
          ym[h][k] = make_uint4(0u, 0u, 0u, 0u);
          if (r < M && do_loads) {
            ym[h][k] = ld_cg_v4(params.mlp_y + r * ld + seg_col(mi, h, k));
          }
        }
      }
      CUTLASS_PRAGMA_UNROLL
      for (int h = 0; h < 2; ++h) {
        int const r = chunk_row(mi, h);
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < KCH; ++k) {
          if (r < M) {
            uint4 const a = ya[mi][h][k];
            uint4 const m = ym[h][k];
            uint4 const g = gv[h][k];
            uint4 const xx = xv[h][k];
            uint4 o;
            o.x = residual_x2(a.x, m.x, g.x, xx.x);
            o.y = residual_x2(a.y, m.y, g.y, xx.y);
            o.z = residual_x2(a.z, m.z, g.z, xx.z);
            o.w = residual_x2(a.w, m.w, g.w, xx.w);
            *reinterpret_cast<uint4*>(params.out + r * ld + seg_col(mi, h, k)) = o;
          }
        }
      }
    }
  }

  // Warp-uniform. All lanes fence their mlp_y stores, then lane 0 stores the epoch (release).
  CUTLASS_DEVICE void release_pending() {
    if (pending_flag_ != nullptr) {
      __threadfence();
      __syncwarp();
      if ((threadIdx.x & 31) == 0) {
        st_release_gpu(pending_flag_, pending_epoch_);
      }
      pending_flag_ = nullptr;
    }
  }

 private:
  Params params;
  uint32_t* pending_flag_ = nullptr;
  uint32_t pending_epoch_ = 0;
};

// The operator-style adapter plus a store_tail that publishes the CTA's last down tile.
template <class EpilogueOp>
class PairResidualAdapter
    : public cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<EpilogueOp> {
  using Base = cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<EpilogueOp>;

 public:
  using Base::Base;

  CUTLASS_DEVICE auto store_tail(typename Base::LoadPipeline, typename Base::LoadPipelineState
                                     load_pipe_consumer_state,
                                 typename Base::StorePipeline,
                                 typename Base::StorePipelineState store_pipe_producer_state) {
    this->release_pending();
    return cute::make_tuple(load_pipe_consumer_state, store_pipe_producer_state);
  }
};

using ElementA = cutlass::float_e2m1_t;
using ElementSF = cutlass::float_ue4m3_t;
using ElementMainloop = cutlass::nv_float4_t<ElementA>;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
constexpr int AlignmentAB = 32;  // 128 bits of FP4
using TileShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

using StrideCD = cute::Stride<int64_t, cute::Int<1>, int64_t>;
using CollectiveEpilogue = PairResidualAdapter<PairResidualEpilogue<StrideCD>>;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, ElementMainloop, LayoutA*,
    AlignmentAB, ElementMainloop, LayoutB*, AlignmentAB, float, TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel =
    cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue,
                                         void>;

using UnderlyingShape = typename ProblemShape::UnderlyingProblemShape;
using StrideA = typename GemmKernel::InternalStrideA;
using StrideB = typename GemmKernel::InternalStrideB;
using ScaleConfig = typename GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using LayoutSFA = typename GemmKernel::CollectiveMainloop::InternalLayoutSFA;
using LayoutSFB = typename GemmKernel::CollectiveMainloop::InternalLayoutSFB;
using GemmElementA = typename GemmKernel::ElementA;
using GemmElementB = typename GemmKernel::ElementB;

static_assert(GemmKernel::CollectiveMainloop::TiledMma::SFVecSize == 16, "NVFP4 SF vector 16");
static_assert(size(typename GemmKernel::TiledMma{}) == 32 * kMmaWarps, "8 MMA warps expected");

constexpr int kGroups = 2;

struct PairInputs {
  void const* a[kGroups];
  void const* b[kGroups];
  void const* sfa[kGroups];
  void const* sfb[kGroups];
  int m[kGroups];
  int n[kGroups];
  int k[kGroups];
};

struct ArgsLayout {
  UnderlyingShape* problem_sizes;
  GemmElementA const** a_ptr;
  GemmElementB const** b_ptr;
  ElementSF const** sfa_ptr;
  ElementSF const** sfb_ptr;
  StrideA* stride_a;
  StrideB* stride_b;
  LayoutSFA* layout_sfa;
  LayoutSFB* layout_sfb;
  uint32_t* counter;
  uint32_t* epoch;
};

// Fills the grouped-GEMM argument arrays, resets the tile counter and bumps the epoch. Launched
// with PDL: it waits for the previous kernel on the stream (which may be the previous pair GEMM
// still using these arrays, the counter and the epoch) before writing anything.
__global__ void flux3_lptr_args_kernel(PairInputs in, ArgsLayout out) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;");
#endif
  int i = threadIdx.x;
  if (i >= kGroups) {
    return;
  }
  int const m = in.m[i];
  int const n = in.n[i];
  int const k = in.k[i];
  out.problem_sizes[i] = UnderlyingShape(m, n, k);
  out.stride_a[i] = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  out.stride_b[i] = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  out.a_ptr[i] = static_cast<GemmElementA const*>(in.a[i]);
  out.b_ptr[i] = static_cast<GemmElementB const*>(in.b[i]);
  out.sfa_ptr[i] = static_cast<ElementSF const*>(in.sfa[i]);
  out.sfb_ptr[i] = static_cast<ElementSF const*>(in.sfb[i]);
  out.layout_sfa[i] = ScaleConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
  out.layout_sfb[i] = ScaleConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));
  if (i == 0) {
    *out.counter = 0u;
    *out.epoch = *out.epoch + 1u;
  }
}

constexpr size_t kArgsRegionBytes = 8192;
constexpr size_t kFlagsOffset = size_t(4) << 20;  // flags live in [4 MB, workspace end)

inline size_t align_up(size_t x, size_t a) { return (x + a - 1) / a * a; }

template <class T>
T* carve(uint8_t* base, size_t& offset, size_t count) {
  offset = align_up(offset, 128);
  T* p = reinterpret_cast<T*>(base + offset);
  offset += sizeof(T) * count;
  return p;
}

inline ArgsLayout make_args_layout(uint8_t* base) {
  size_t off = 0;
  ArgsLayout l;
  l.counter = carve<uint32_t>(base, off, 1);
  l.epoch = carve<uint32_t>(base, off, 1);
  l.problem_sizes = carve<UnderlyingShape>(base, off, kGroups);
  l.a_ptr = carve<GemmElementA const*>(base, off, kGroups);
  l.b_ptr = carve<GemmElementB const*>(base, off, kGroups);
  l.sfa_ptr = carve<ElementSF const*>(base, off, kGroups);
  l.sfb_ptr = carve<ElementSF const*>(base, off, kGroups);
  l.stride_a = carve<StrideA>(base, off, kGroups);
  l.stride_b = carve<StrideB>(base, off, kGroups);
  l.layout_sfa = carve<LayoutSFA>(base, off, kGroups);
  l.layout_sfb = carve<LayoutSFB>(base, off, kGroups);
  if (off > kArgsRegionBytes) {
    throw std::runtime_error("flux3_lpt_pair_residual: args region overflow");
  }
  return l;
}

inline int device_sm_count(int device_id) {
  static thread_local int cached_device = -1;
  static thread_local int cached_count = 0;
  if (cached_device != device_id) {
    cached_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);
    cached_device = device_id;
  }
  return cached_count;
}

}  // namespace flux3_lptr

namespace torch_ext {

namespace {

void check_gemm_operand(TensorView a, TensorView b, TensorView sfa, TensorView sfb,
                        TensorView alpha) {
  TVM_FFI_ICHECK_EQ(a.dtype(), dl_uint8) << "a must be packed FP4 (uint8)";
  TVM_FFI_ICHECK_EQ(b.dtype(), dl_uint8) << "b must be packed FP4 (uint8)";
  TVM_FFI_ICHECK_EQ(sfa.dtype(), dl_uint8) << "sfa must be uint8";
  TVM_FFI_ICHECK_EQ(sfb.dtype(), dl_uint8) << "sfb must be uint8";
  TVM_FFI_ICHECK_EQ(alpha.dtype(), dl_float32) << "alpha must be float32";
  TVM_FFI_ICHECK_EQ(alpha.numel(), 1) << "alpha must have one element";
  TVM_FFI_ICHECK_EQ(a.ndim(), 2) << "a must be 2D";
  TVM_FFI_ICHECK_EQ(b.ndim(), 2) << "b must be 2D";
  TVM_FFI_ICHECK_EQ(b.size(1), a.size(1)) << "K mismatch";
  TVM_FFI_ICHECK(b.size(0) % 128 == 0) << "N must be a multiple of 128";
  TVM_FFI_ICHECK((a.size(1) * 2) % 128 == 0) << "K must be a multiple of 128";
  CHECK_DEVICE(a, b);
  CHECK_DEVICE(a, sfa);
  CHECK_DEVICE(a, sfb);
  CHECK_DEVICE(a, alpha);
}

void check_bf16_2d(TensorView t, int64_t rows, int64_t cols, char const* name) {
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(t.dtype()), bfloat16_code) << name << " must be bf16";
  TVM_FFI_ICHECK_EQ(t.ndim(), 2) << name << " must be 2D";
  if (rows >= 0) {
    TVM_FFI_ICHECK_EQ(t.size(0), rows) << name << " rows";
  }
  TVM_FFI_ICHECK_EQ(t.size(1), cols) << name << " cols";
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0)
      << name << " must be 16-byte aligned";
}

}  // namespace

// Group 0 = down (a0, b0, ...; scheduled first), group 1 = attn_out (a1, ...). mlp_y: [M, N] bf16
// scratch for down's output. out = x + gate[gate_index] * (attn_out + down), [M, N] bf16 (fresh; must
// not alias x). gate: [R, N] bf16; gate_index: [>= M] uint8 (read only when has_index != 0).
// All [*, N] tensors contiguous with row stride N. workspace: >= 8 MB, zero-initialized once.
void lpt_pair_residual(TensorView a0, TensorView b0, TensorView sfa0, TensorView sfb0,
                       TensorView alpha0, TensorView mlp_y, TensorView a1, TensorView b1,
                       TensorView sfa1, TensorView sfb1, TensorView alpha1, TensorView x,
                       TensorView gate, TensorView gate_index, int64_t has_index, TensorView out,
                       TensorView workspace) {
  using namespace flux3_lptr;
  check_gemm_operand(a0, b0, sfa0, sfb0, alpha0);
  check_gemm_operand(a1, b1, sfa1, sfb1, alpha1);
  CHECK_DEVICE(a0, a1);
  CHECK_DEVICE(a0, workspace);
  CHECK_DEVICE(a0, x);
  CHECK_DEVICE(a0, out);
  CHECK_DEVICE(a0, mlp_y);
  CHECK_DEVICE(a0, gate);
  int64_t const M = a0.size(0);
  int64_t const N = b0.size(0);
  TVM_FFI_ICHECK_EQ(a1.size(0), M) << "both groups must have the same M";
  TVM_FFI_ICHECK_EQ(b1.size(0), N) << "both groups must have the same N";
  check_bf16_2d(mlp_y, M, N, "mlp_y");
  check_bf16_2d(x, M, N, "x");
  check_bf16_2d(out, M, N, "out");
  check_bf16_2d(gate, -1, N, "gate");
  TVM_FFI_ICHECK(gate.size(0) >= 1) << "gate needs at least one row";
  TVM_FFI_ICHECK(out.data_ptr() != x.data_ptr()) << "out must not alias x";
  if (has_index != 0) {
    CHECK_DEVICE(a0, gate_index);
    TVM_FFI_ICHECK_EQ(gate_index.dtype(), dl_uint8) << "gate_index must be uint8";
    TVM_FFI_ICHECK(gate_index.numel() >= M) << "gate_index needs one entry per row";
  }

  PairInputs in{};
  TensorView const* as[2] = {&a0, &a1};
  TensorView const* bs[2] = {&b0, &b1};
  TensorView const* sfas[2] = {&sfa0, &sfa1};
  TensorView const* sfbs[2] = {&sfb0, &sfb1};
  for (int i = 0; i < kGroups; ++i) {
    in.a[i] = as[i]->data_ptr();
    in.b[i] = bs[i]->data_ptr();
    in.sfa[i] = sfas[i]->data_ptr();
    in.sfb[i] = sfbs[i]->data_ptr();
    in.m[i] = static_cast<int>(as[i]->size(0));
    in.n[i] = static_cast<int>(bs[i]->size(0));
    in.k[i] = static_cast<int>(as[i]->size(1) * 2);
  }

  size_t const ws_bytes = static_cast<size_t>(workspace.numel()) * get_element_size(workspace);
  TVM_FFI_ICHECK(ws_bytes > kFlagsOffset) << "workspace too small";
  auto* ws = static_cast<uint8_t*>(workspace.data_ptr());
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(ws) % 256 == 0) << "workspace must be 256B aligned";
  ArgsLayout args_layout = make_args_layout(ws);
  int64_t const tiles_m = (M + 127) / 128;
  int64_t const tiles_n = N / 128;
  size_t const flag_bytes = static_cast<size_t>(tiles_m * tiles_n * kMmaWarps) * sizeof(uint32_t);
  TVM_FFI_ICHECK(kFlagsOffset + flag_bytes <= ws_bytes)
      << "flux3_lpt_pair_residual: workspace too small for " << tiles_m * tiles_n << " tiles";

  cudaStream_t stream = get_stream(a0.device());
  int const device_id = a0.device().device_id;

  {
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(1);
    config.blockDim = dim3(32);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    config.numAttrs = 1;
    config.attrs = attrs;
    cudaError_t err = cudaLaunchKernelEx(&config, flux3_lptr_args_kernel, in, args_layout);
    TVM_FFI_ICHECK(err == cudaSuccess)
        << "flux3_lpt_pair_residual: args kernel launch failed: " << cudaGetErrorString(err);
  }

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device_id;
  hw_info.sm_count = device_sm_count(device_id);

  typename GemmKernel::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {kGroups, args_layout.problem_sizes, /*problem_sizes_host=*/nullptr},
      {args_layout.a_ptr, args_layout.stride_a, args_layout.b_ptr, args_layout.stride_b,
       args_layout.sfa_ptr, args_layout.layout_sfa, args_layout.sfb_ptr, args_layout.layout_sfb},
      {},
      hw_info};
  arguments.scheduler.tile_counter = args_layout.counter;
  arguments.scheduler.prefetch_x = x.data_ptr();
  arguments.scheduler.prefetch_y = mlp_y.data_ptr();
  arguments.scheduler.prefetch_ld = N;
  arguments.scheduler.prefetch_rows = static_cast<int>(M);
  auto& epi = arguments.epilogue;
  epi.mlp_y = static_cast<__nv_bfloat16*>(mlp_y.data_ptr());
  epi.out = static_cast<__nv_bfloat16*>(out.data_ptr());
  epi.x = static_cast<__nv_bfloat16 const*>(x.data_ptr());
  epi.gate = static_cast<__nv_bfloat16 const*>(gate.data_ptr());
  epi.gate_index = has_index != 0 ? static_cast<uint8_t const*>(gate_index.data_ptr()) : nullptr;
  epi.alpha0 = static_cast<float const*>(alpha0.data_ptr());
  epi.alpha1 = static_cast<float const*>(alpha1.data_ptr());
  epi.flags = reinterpret_cast<uint32_t*>(ws + kFlagsOffset);
  epi.epoch = args_layout.epoch;
  epi.ld = static_cast<int>(N);
  epi.tiles_n = static_cast<int>(tiles_n);
  {
    char const* d = std::getenv("FLUX3_LPTR_DIAG");
    epi.diag = d != nullptr ? std::atoi(d) : 0;
    if (epi.diag & 8) {
      arguments.scheduler.prefetch_x = nullptr;  // 8: no L2 prefetch
    }
  }

  TVM_FFI_ICHECK(GemmKernel::can_implement(arguments)) << "flux3_lpt_pair_residual: cannot implement";
  size_t const need = GemmKernel::get_workspace_size(arguments);
  TVM_FFI_ICHECK(kArgsRegionBytes + need <= kFlagsOffset)
      << "flux3_lpt_pair_residual: GEMM workspace overlaps the flags (" << need << " B)";
  void* gemm_ws = ws + kArgsRegionBytes;
  auto status = GemmKernel::initialize_workspace(arguments, gemm_ws, stream, nullptr);
  TVM_FFI_ICHECK(status == cutlass::Status::kSuccess)
      << "flux3_lpt_pair_residual: initialize_workspace failed";
  typename GemmKernel::Params params = GemmKernel::to_underlying_arguments(arguments, gemm_ws);
  dim3 const grid = GemmKernel::get_grid_shape(params);
  dim3 const block = GemmKernel::get_block_shape();
  int const smem = GemmKernel::SharedStorageSize;
  static bool attr_set = false;
  if (!attr_set) {
    cudaError_t err = cudaFuncSetAttribute(cutlass::device_kernel<GemmKernel>,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    TVM_FFI_ICHECK(err == cudaSuccess) << "cudaFuncSetAttribute: " << cudaGetErrorString(err);
    attr_set = true;
  }
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = smem;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.numAttrs = 1;
  config.attrs = attrs;
  cudaError_t err = cudaLaunchKernelEx(&config, cutlass::device_kernel<GemmKernel>, params);
  TVM_FFI_ICHECK(err == cudaSuccess)
      << "flux3_lpt_pair_residual: GEMM launch failed: " << cudaGetErrorString(err);
}

// Build facts for the microbenchmark log: stages * 1000000 + kernel shared storage bytes.
int64_t lpt_pair_residual_info() {
  using namespace flux3_lptr;
  return static_cast<int64_t>(CollectiveMainloop::DispatchPolicy::Stages) * 1000000 +
         static_cast<int64_t>(GemmKernel::SharedStorageSize);
}

}  // namespace torch_ext

TVM_FFI_DLL_EXPORT_TYPED_FUNC(lpt_pair_residual, torch_ext::lpt_pair_residual);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(lpt_pair_residual_info, torch_ext::lpt_pair_residual_info);
