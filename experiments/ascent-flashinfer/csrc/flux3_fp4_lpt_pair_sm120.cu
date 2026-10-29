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
// N=3072, BF16 out) as ONE persistent SM120 CUTLASS launch.
//
// Kernel: CUTLASS SM120 OpClassBlockScaledTensorOp ptr-array grouped GEMM (GroupProblemShape, 2
// groups, per-group M/N/K, pointers, scale layouts and alpha), TileShape 128x128x128 FP4 elements,
// 1x1x1 cluster, Cooperative mainloop, LinearCombination epilogue (BF16(alpha * acc), C void).
// Each output tile runs the same MMA sequence over K as FlashInfer's stock SM120 DP kernel
// (tactic 17) / TRT-LLM FP4GemmRunner tactic 0, so the outputs are expected bitwise equal to t0.
//
// Tile order: group 0 (put the long-K GEMM there) before group 1 in the linear tile index. The
// stock group scheduler hands CTA c the tiles c, c + grid, c + 2*grid, ... (static round-robin).
// mode=1 replaces that with a dynamic order: after its first (static) tile, a CTA's scheduler warp
// claims the next linear tile from a global atomic counter, so CTAs that finish early take more of
// the short group-1 tiles (longest-first list scheduling). The scheduler pipeline has ONE stage, so
// a CTA claims at most one tile ahead of the tile its MMA warps are working on. The counter lives in
// the caller's workspace and is reset by the args kernel that precedes every GEMM launch (stream
// ordered, CUDA-graph safe; the GEMM waits for the args kernel via griddepcontrol.wait).
// The raster argument picks how a linear tile index maps to (m, n) within a group: CUTLASS's
// grouped default AlongM walks every M tile of one N column first, so at M=33280 a wave streams
// down's 153 MB A operand (over the 128 MB L2) once per N column; AlongN walks the 24 N tiles of
// one M panel first, so each A panel is read from DRAM about once.
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "tvm_ffi_utils.h"

namespace flux3_lpt {
using ProblemShape = cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;
}  // namespace flux3_lpt

namespace cutlass::gemm::kernel::detail {

// PersistentTileSchedulerSm90Group with an optional dynamic (atomic counter) tile order and a
// one-stage scheduler pipeline. Only the kernel's Scheduler warp calls advance_to_next_work; the
// other warps read its responses through the pipeline (fetch_next_work, inherited).
template <class GroupProblemShape, int SchedulerPipelineStageCount>
class Flux3LptGroupScheduler
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
    uint32_t* tile_counter = nullptr;  // nullptr: static round-robin (stock order)
    int on_demand = 0;  // 1: claim a tile only once the mainloop DMA warp asks for it
  };

  struct Params : BaseParams {
    uint32_t* tile_counter = nullptr;
    int on_demand = 0;
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
    params.on_demand = arguments.on_demand;
    return params;
  }

  Flux3LptGroupScheduler() = default;

  CUTLASS_DEVICE explicit Flux3LptGroupScheduler(Params const& params,
                                                 SchedulerResponse* response_ptr)
      : Base(static_cast<BaseParams const&>(params), response_ptr),
        tile_counter_(params.tile_counter),
        on_demand_(params.tile_counter != nullptr && params.on_demand != 0) {
#if defined(__CUDA_ARCH__)
    // Thread 0 (lane 0 of the mainloop DMA warp, the only writer of the request count) zeroes it
    // before its own first request (program order). The count only paces the claims; a stale
    // read can make one claim early but never changes which tiles are computed.
    if (on_demand_ && threadIdx.x == 0) {
      *request_count() = 0u;
    }
    if (params.raster_order_ == RasterOrder::AlongN) {
      lpt_linear_idx_ = uint64_t(blockIdx.x) + uint64_t(blockIdx.y) * uint64_t(gridDim.x);
    } else {
      lpt_linear_idx_ = uint64_t(blockIdx.x) * uint64_t(gridDim.y) + uint64_t(blockIdx.y);
    }
    lpt_grid_size_ = uint64_t(gridDim.x) * uint64_t(gridDim.y) * uint64_t(gridDim.z);
#endif
  }

  template <typename TileSchedulerPipeline, typename TileSchedulerPipelineState,
            typename CallbackBeforeCommit = WorkTileInfo (*)(WorkTileInfo)>
  CUTLASS_DEVICE auto advance_to_next_work(
      TileSchedulerPipeline& scheduler_pipeline,
      TileSchedulerPipelineState scheduler_pipe_producer_state, uint32_t advance_count = 1,
      CallbackBeforeCommit callback_before_commit = [](WorkTileInfo info) { return info; }) {
    // Claim only once the response slot is free, so a CTA runs at most LptStages tiles ahead.
    scheduler_pipeline.producer_acquire(scheduler_pipe_producer_state);
    if (tile_counter_ == nullptr) {
      lpt_linear_idx_ += lpt_grid_size_ * uint64_t(advance_count);
    } else {
      uint32_t claimed = 0;
      ++claims_;
      if ((threadIdx.x % 32) == 0) {
        if (on_demand_) {
          // Wait until the DMA warp has asked for this tile (it asks right after issuing the
          // last loads of its current tile, a few k-stages before the MMA warps need it).
          volatile uint32_t* req = request_count();
          while (*req < claims_) {
            __nanosleep(32);
          }
        }
        claimed = atomicAdd(tile_counter_, 1u);
      }
      claimed = __shfl_sync(0xffffffff, claimed, 0);
      lpt_linear_idx_ = lpt_grid_size_ + uint64_t(claimed);
    }
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
    return cute::make_tuple(info, true);
  }

  // Consumers read the scheduler's responses here. In on-demand mode the mainloop DMA warp
  // (warp 0 of the producer warpgroup) first bumps the request count the scheduler warp waits on.
  template <typename WorkTileWithCallbackInfo, typename TileSchedulerPipeline,
            typename TileSchedulerPipelineState>
  CUTLASS_DEVICE auto fetch_next_work(WorkTileWithCallbackInfo work_tile_with_callback_info,
                                      TileSchedulerPipeline& scheduler_pipeline,
                                      TileSchedulerPipelineState scheduler_pipe_consumer_state) {
    if (on_demand_ && threadIdx.x == 0) {
      volatile uint32_t* req = request_count();
      *req = *req + 1u;
    }
    return Base::fetch_next_work(work_tile_with_callback_info, scheduler_pipeline,
                                 scheduler_pipe_consumer_state);
  }

 private:
  // The kernel reserves 8 response slots in shared memory; this scheduler's pipeline uses only
  // slot 0, so the last slot holds the request count.
  CUTLASS_DEVICE uint32_t* request_count() const {
    return reinterpret_cast<uint32_t*>(reinterpret_cast<WorkTileInfo*>(this->response_ptr_) + 7);
  }

  uint32_t* tile_counter_ = nullptr;
  bool on_demand_ = false;
  uint32_t claims_ = 0;
  uint64_t lpt_linear_idx_ = 0;
  uint64_t lpt_grid_size_ = 0;
};

// Route this TU's SM120 grouped GEMM (and only it: the module holds no other grouped kernel) to
// the scheduler above. More specialized than CUTLASS's GroupScheduler/Sm120 selector.
template <class TileShape, class ClusterShape, uint32_t SchedulerPipelineStageCount>
struct TileSchedulerSelector<GroupScheduler, arch::Sm120, TileShape, ClusterShape,
                             SchedulerPipelineStageCount, flux3_lpt::ProblemShape> {
  using Scheduler = Flux3LptGroupScheduler<flux3_lpt::ProblemShape,
                                           static_cast<int>(SchedulerPipelineStageCount)>;
};

}  // namespace cutlass::gemm::kernel::detail

namespace flux3_lpt {

using namespace cute;

using ElementA = cutlass::float_e2m1_t;
using ElementSF = cutlass::float_ue4m3_t;
using ElementD = cutlass::bfloat16_t;
using ElementMainloop = cutlass::nv_float4_t<ElementA>;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::RowMajor;
constexpr int AlignmentAB = 32;  // 128 bits of FP4
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
using TileShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto, float, float, void, LayoutD*, AlignmentD,
    ElementD, LayoutD*, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

#ifdef FLUX3_LPT_STAGES
using StageCountType = cutlass::gemm::collective::StageCount<FLUX3_LPT_STAGES>;
#else
using StageCountType = cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
    sizeof(typename CollectiveEpilogue::SharedStorage))>;
#endif

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, ElementMainloop, LayoutA*,
    AlignmentAB, ElementMainloop, LayoutB*, AlignmentAB, float, TileShape, ClusterShape,
    StageCountType, cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel =
    cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue,
                                         void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using UnderlyingShape = typename ProblemShape::UnderlyingProblemShape;
using StrideA = typename GemmKernel::InternalStrideA;
using StrideB = typename GemmKernel::InternalStrideB;
using StrideD = typename GemmKernel::InternalStrideD;
using ScaleConfig = typename GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using LayoutSFA = typename GemmKernel::CollectiveMainloop::InternalLayoutSFA;
using LayoutSFB = typename GemmKernel::CollectiveMainloop::InternalLayoutSFB;
using GemmElementA = typename Gemm::ElementA;
using GemmElementB = typename Gemm::ElementB;
using GemmElementD = typename Gemm::EpilogueOutputOp::ElementOutput;

static_assert(GemmKernel::CollectiveMainloop::TiledMma::SFVecSize == 16, "NVFP4 SF vector 16");

constexpr int kGroups = 2;

struct PairInputs {
  void const* a[kGroups];
  void const* b[kGroups];
  void const* sfa[kGroups];
  void const* sfb[kGroups];
  float const* alpha[kGroups];
  void* d[kGroups];
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
  GemmElementD** d_ptr;
  StrideA* stride_a;
  StrideB* stride_b;
  StrideD* stride_d;
  LayoutSFA* layout_sfa;
  LayoutSFB* layout_sfb;
  float* alphas;
  uint32_t* counter;
};

// Fills the grouped-GEMM argument arrays and resets the tile counter. Launched with PDL: it waits
// for the previous kernel on the stream (which may be the previous pair GEMM still using these
// arrays and the counter) before writing anything.
__global__ void flux3_lpt_pair_args_kernel(PairInputs in, ArgsLayout out) {
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
  out.stride_d[i] = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});
  out.a_ptr[i] = static_cast<GemmElementA const*>(in.a[i]);
  out.b_ptr[i] = static_cast<GemmElementB const*>(in.b[i]);
  out.sfa_ptr[i] = static_cast<ElementSF const*>(in.sfa[i]);
  out.sfb_ptr[i] = static_cast<ElementSF const*>(in.sfb[i]);
  out.d_ptr[i] = static_cast<GemmElementD*>(in.d[i]);
  out.layout_sfa[i] = ScaleConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
  out.layout_sfb[i] = ScaleConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));
  out.alphas[i] = *in.alpha[i];
  if (i == 0) {
    *out.counter = 0u;
  }
}

constexpr size_t kArgsRegionBytes = 8192;

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
  l.alphas = carve<float>(base, off, kGroups);
  l.problem_sizes = carve<UnderlyingShape>(base, off, kGroups);
  l.a_ptr = carve<GemmElementA const*>(base, off, kGroups);
  l.b_ptr = carve<GemmElementB const*>(base, off, kGroups);
  l.sfa_ptr = carve<ElementSF const*>(base, off, kGroups);
  l.sfb_ptr = carve<ElementSF const*>(base, off, kGroups);
  l.d_ptr = carve<GemmElementD*>(base, off, kGroups);
  l.stride_a = carve<StrideA>(base, off, kGroups);
  l.stride_b = carve<StrideB>(base, off, kGroups);
  l.stride_d = carve<StrideD>(base, off, kGroups);
  l.layout_sfa = carve<LayoutSFA>(base, off, kGroups);
  l.layout_sfb = carve<LayoutSFB>(base, off, kGroups);
  if (off > kArgsRegionBytes) {
    throw std::runtime_error("flux3_lpt_pair: args region overflow");
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

}  // namespace flux3_lpt

namespace torch_ext {

namespace {

void check_operand(TensorView a, TensorView b, TensorView sfa, TensorView sfb, TensorView alpha,
                   TensorView out) {
  TVM_FFI_ICHECK_EQ(a.dtype(), dl_uint8) << "a must be packed FP4 (uint8)";
  TVM_FFI_ICHECK_EQ(b.dtype(), dl_uint8) << "b must be packed FP4 (uint8)";
  TVM_FFI_ICHECK_EQ(sfa.dtype(), dl_uint8) << "sfa must be uint8";
  TVM_FFI_ICHECK_EQ(sfb.dtype(), dl_uint8) << "sfb must be uint8";
  TVM_FFI_ICHECK_EQ(alpha.dtype(), dl_float32) << "alpha must be float32";
  TVM_FFI_ICHECK_EQ(alpha.numel(), 1) << "alpha must have one element";
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(out.dtype()), bfloat16_code) << "out must be bf16";
  TVM_FFI_ICHECK_EQ(a.ndim(), 2) << "a must be 2D";
  TVM_FFI_ICHECK_EQ(b.ndim(), 2) << "b must be 2D";
  TVM_FFI_ICHECK_EQ(out.ndim(), 2) << "out must be 2D";
  TVM_FFI_ICHECK_EQ(b.size(1), a.size(1)) << "K mismatch";
  TVM_FFI_ICHECK_EQ(out.size(0), a.size(0)) << "out rows must equal M";
  TVM_FFI_ICHECK_EQ(out.size(1), b.size(0)) << "out cols must equal N";
  TVM_FFI_ICHECK(b.size(0) % 8 == 0) << "N must be a multiple of 8";
  TVM_FFI_ICHECK((a.size(1) * 2) % 128 == 0) << "K must be a multiple of 128";
  CHECK_DEVICE(a, b);
  CHECK_DEVICE(a, sfa);
  CHECK_DEVICE(a, sfb);
  CHECK_DEVICE(a, alpha);
  CHECK_DEVICE(a, out);
}

}  // namespace

// Group 0 = (a0, b0, ...) is scheduled first (put the long-K GEMM there), group 1 after it.
// mode: 0 = static round-robin (stock group order), 1 = dynamic longest-first (atomic counter,
// each CTA claims one tile ahead), 2 = dynamic, claimed when the mainloop DMA warp asks for it.
// grid_ctas: 0 = one persistent CTA per SM; >0 overrides the persistent grid (diagnostics).
// raster: 0 = AlongM (CUTLASS's grouped default: M fastest within an N column), 1 = AlongN (N
// fastest within an M panel, so one wave reuses each A panel across all N tiles from L2).
// max_swizzle: CUTLASS max_swizzle_size (1 = none). raster=0, max_swizzle=1 is main's order.
// Only which CTA computes which output tile (and when) changes; each tile's MMA sequence over K
// and its epilogue are the same, so outputs are bitwise identical across raster settings.
void lpt_pair(TensorView a0, TensorView b0, TensorView sfa0, TensorView sfb0, TensorView alpha0,
              TensorView out0, TensorView a1, TensorView b1, TensorView sfa1, TensorView sfb1,
              TensorView alpha1, TensorView out1, TensorView workspace, int64_t mode,
              int64_t grid_ctas, int64_t raster, int64_t max_swizzle) {
  using namespace flux3_lpt;
  check_operand(a0, b0, sfa0, sfb0, alpha0, out0);
  check_operand(a1, b1, sfa1, sfb1, alpha1, out1);
  CHECK_DEVICE(a0, a1);
  CHECK_DEVICE(a0, workspace);
  TVM_FFI_ICHECK(mode >= 0 && mode <= 2)
      << "mode must be 0 (static), 1 (dynamic, one tile ahead) or 2 (dynamic, on demand)";
  TVM_FFI_ICHECK(raster == 0 || raster == 1) << "raster must be 0 (AlongM) or 1 (AlongN)";
  TVM_FFI_ICHECK(max_swizzle == 1 || max_swizzle == 2 || max_swizzle == 4 || max_swizzle == 8)
      << "max_swizzle must be 1, 2, 4 or 8";

  PairInputs in{};
  TensorView const* as[2] = {&a0, &a1};
  TensorView const* bs[2] = {&b0, &b1};
  TensorView const* sfas[2] = {&sfa0, &sfa1};
  TensorView const* sfbs[2] = {&sfb0, &sfb1};
  TensorView const* alphas[2] = {&alpha0, &alpha1};
  TensorView const* outs[2] = {&out0, &out1};
  for (int i = 0; i < kGroups; ++i) {
    in.a[i] = as[i]->data_ptr();
    in.b[i] = bs[i]->data_ptr();
    in.sfa[i] = sfas[i]->data_ptr();
    in.sfb[i] = sfbs[i]->data_ptr();
    in.alpha[i] = static_cast<float const*>(alphas[i]->data_ptr());
    in.d[i] = outs[i]->data_ptr();
    in.m[i] = static_cast<int>(as[i]->size(0));
    in.n[i] = static_cast<int>(bs[i]->size(0));
    in.k[i] = static_cast<int>(as[i]->size(1) * 2);
  }

  size_t const ws_bytes = static_cast<size_t>(workspace.numel()) * get_element_size(workspace);
  TVM_FFI_ICHECK(ws_bytes > kArgsRegionBytes) << "workspace too small";
  auto* ws = static_cast<uint8_t*>(workspace.data_ptr());
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(ws) % 256 == 0) << "workspace must be 256B aligned";
  ArgsLayout args_layout = make_args_layout(ws);

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
    cudaError_t err = cudaLaunchKernelEx(&config, flux3_lpt_pair_args_kernel, in, args_layout);
    TVM_FFI_ICHECK(err == cudaSuccess)
        << "flux3_lpt_pair: args kernel launch failed: " << cudaGetErrorString(err);
  }

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device_id;
  hw_info.sm_count = grid_ctas > 0 ? static_cast<int>(grid_ctas) : device_sm_count(device_id);

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {kGroups, args_layout.problem_sizes, /*problem_sizes_host=*/nullptr},
      {args_layout.a_ptr, args_layout.stride_a, args_layout.b_ptr, args_layout.stride_b,
       args_layout.sfa_ptr, args_layout.layout_sfa, args_layout.sfb_ptr, args_layout.layout_sfb},
      {{}, nullptr, nullptr, args_layout.d_ptr, args_layout.stride_d},
      hw_info};
  arguments.scheduler.tile_counter = mode >= 1 ? args_layout.counter : nullptr;
  arguments.scheduler.on_demand = mode == 2 ? 1 : 0;
  using SchedulerRasterOptions = decltype(arguments.scheduler.raster_order);
  arguments.scheduler.raster_order =
      raster == 1 ? SchedulerRasterOptions::AlongN : SchedulerRasterOptions::AlongM;
  arguments.scheduler.max_swizzle_size = static_cast<int>(max_swizzle);
  auto& fusion_args = arguments.epilogue.thread;
  fusion_args.alpha = 0.0f;
  fusion_args.beta = 0.0f;
  fusion_args.alpha_ptr = args_layout.alphas;
  fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 1};

  Gemm gemm;
  size_t const need = Gemm::get_workspace_size(arguments);
  TVM_FFI_ICHECK(need + kArgsRegionBytes <= ws_bytes)
      << "flux3_lpt_pair: workspace too small, need " << need + kArgsRegionBytes << " got "
      << ws_bytes;
  void* gemm_ws = ws + kArgsRegionBytes;
  auto status = gemm.can_implement(arguments);
  TVM_FFI_ICHECK(status == cutlass::Status::kSuccess)
      << "flux3_lpt_pair: can_implement failed: " << cutlass::cutlassGetStatusString(status);
  status = gemm.initialize(arguments, gemm_ws, stream);
  TVM_FFI_ICHECK(status == cutlass::Status::kSuccess)
      << "flux3_lpt_pair: initialize failed: " << cutlass::cutlassGetStatusString(status);
  status = gemm.run(stream, /*cuda_adapter=*/nullptr, /*launch_with_pdl=*/true);
  TVM_FFI_ICHECK(status == cutlass::Status::kSuccess)
      << "flux3_lpt_pair: run failed: " << cutlass::cutlassGetStatusString(status);
}

// Build facts for the microbenchmark log: stages * 1000000 + kernel shared storage bytes.
int64_t lpt_pair_info() {
  using namespace flux3_lpt;
  return static_cast<int64_t>(CollectiveMainloop::DispatchPolicy::Stages) * 1000000 +
         static_cast<int64_t>(GemmKernel::SharedStorageSize);
}

}  // namespace torch_ext

TVM_FFI_DLL_EXPORT_TYPED_FUNC(lpt_pair, torch_ext::lpt_pair);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(lpt_pair_info, torch_ext::lpt_pair_info);
