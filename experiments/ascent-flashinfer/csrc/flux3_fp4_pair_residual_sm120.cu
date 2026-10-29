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
// FLUX.3 (add-only): the NVFP4 block tail in ONE launch.
//
//   x_new = bf16(x + bf16(gate * bf16(bf16(alpha_a * A_a @ B_a^T) + bf16(alpha_d * A_d @ B_d^T))))
//
// (a) = attn_out (K=3072), (d) = ff.down_proj (K=9216), both N=3072. The rounding points are
// those of main's BF16 pair outputs (attn_y, mlp_y) followed by the Inductor pointwise compiled
// with emulate_precision_casts: bf16(attn_y + mlp_y), bf16(gate * out), bf16(x + prod), each op
// in fp32 (no contraction). gate is a [G, N] BF16 table: G == 1 (mode blocks) or G == 2 with a
// per-row uint8 select (single blocks: select[r] != 0 picks row 1).
//
// Kernel: one persistent CTA per SM, 1 producer warpgroup (warp 0 issues TMA) + 2 consumer
// warpgroups (Cooperative 128x128 tile). Work unit = one 128x128 output tile (row-major tile
// order, static round robin). The producer streams attn_out's k-tiles, then down's k-tiles of the
// same output tile, through ONE smem pipeline (CUTLASS SM120 block-scaled TMA mainloop, two Params
// sets). The consumers run attn_out's K loop (CollectiveMma::mma: the stock DP kernel's MMA
// sequence per tile), keep BF16(alpha_a * acc) packed in registers, run down's K loop into a fresh
// accumulator, then the epilogue reads x, gate and select from global memory and stores x_new.
#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <stdexcept>

#include "cute/tensor.hpp"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "tvm_ffi_utils.h"

namespace flux3_tres {

using namespace cute;

using ElementA = cutlass::float_e2m1_t;
using ElementSF = cutlass::float_ue4m3_t;
using ElementMainloop = cutlass::nv_float4_t<ElementA>;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using TileShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

#ifdef FLUX3_TRES_STAGES
using StageCountType = cutlass::gemm::collective::StageCount<FLUX3_TRES_STAGES>;
#else
using StageCountType = cutlass::gemm::collective::StageCountAutoCarveout<0>;
#endif

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, ElementMainloop, LayoutA, 32,
    ElementMainloop, LayoutB, 32, float, TileShape, ClusterShape, StageCountType,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;

using TiledMma = typename CollectiveMainloop::TiledMma;
using MainloopParams = typename CollectiveMainloop::Params;
using MainloopArguments = typename CollectiveMainloop::Arguments;
using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
using PipelineState = typename CollectiveMainloop::PipelineState;
using StrideA = typename CollectiveMainloop::StrideA;
using StrideB = typename CollectiveMainloop::StrideB;
using ScaleConfig = typename CollectiveMainloop::Sm1xxBlkScaledConfig;
using MmaElementA = typename CollectiveMainloop::ElementA;
using MmaElementB = typename CollectiveMainloop::ElementB;
using MmaElementSF = typename CollectiveMainloop::ElementSF;

static_assert(size(TiledMma{}) == 256, "Cooperative tile: 2 consumer warpgroups");
constexpr int kThreads = 384;
constexpr int kTileM = 128;
constexpr int kTileN = 128;
constexpr int kTileK = 128;

#ifdef FLUX3_TRES_STAGED
// x_new tile staging for the asynchronous bulk store: 128 rows of 128 BF16, rows padded to 136
// elements (272 B) so the accumulator-layout writes hit 32 distinct banks.
constexpr int kOutRowElems = kTileN + 8;
#endif

struct SharedStorage {
  typename CollectiveMainloop::TensorStorage tensors;
  alignas(16) typename CollectiveMainloop::PipelineStorage pipeline;
#ifdef FLUX3_TRES_STAGED
  alignas(128) __nv_bfloat16 out_tile[kTileM * kOutRowElems];
#endif
};

struct KernelParams {
  MainloopParams ml[2];  // 0: attn_out (run first), 1: down
  int m;
  int n;
  int k[2];
  int tiles_n;
  int num_tiles;
  int mode;        // 0: x_new; 1: write BF16 attn_y / mlp_y to dbg_attn / dbg_mlp (tests)
  int prefetch_x;  // 1: the producer warp prefetches the tile's x rows to L2 before down's loads
  int gate_mode;   // 0: gate row 0 for every row; 1: gate row select[r]; 2: gate row r ([M, N])
  float const* alpha[2];
  __nv_bfloat16 const* x;
  __nv_bfloat16 const* gate;
  uint8_t const* select;
  __nv_bfloat16* out;
  __nv_bfloat16* dbg_attn;
  __nv_bfloat16* dbg_mlp;
};

__device__ __forceinline__ float round_bf16(float v) {
  return __bfloat162float(__float2bfloat16_rn(v));
}

// The residual ops as Triton emits them: IEEE fp32 add / mul, round to nearest, no flush to zero
// (the module is built with -use_fast_math, so plain operators would get .ftz).
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

// A 4-byte read-only load the compiler keeps where it is written (volatile): the early variant
// issues it before down's K loop and first uses it after.
__device__ __forceinline__ uint32_t ldg_nc_b32(void const* ptr) {
  uint32_t v;
  asm volatile("ld.global.nc.b32 %0, [%1];" : "=r"(v) : "l"(ptr));
  return v;
}

__device__ __forceinline__ __nv_bfloat162 as_bf162(uint32_t v) {
  __nv_bfloat162 r;
  memcpy(&r, &v, sizeof(r));
  return r;
}

template <bool kEarly>
__global__ void __launch_bounds__(kThreads, 1)
    flux3_pair_residual_kernel(__grid_constant__ KernelParams const p) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
  extern __shared__ char smem_buf[];
  SharedStorage& ss = *reinterpret_cast<SharedStorage*>(smem_buf);

  int const thread_idx = int(threadIdx.x);
  int const warp_group = thread_idx / 128;
  int const warp_in_wg = (thread_idx / 32) % 4;
  int const lane = thread_idx % 32;

  if (warp_group == 0 && warp_in_wg == 0 && lane == 0) {
    CollectiveMainloop::prefetch_tma_descriptors(p.ml[0]);
    CollectiveMainloop::prefetch_tma_descriptors(p.ml[1]);
  }

  typename MainloopPipeline::Params pp;
  if (warp_group == 0 && warp_in_wg == 0) {
    pp.role = MainloopPipeline::ThreadCategory::Producer;
  }
  if (warp_group >= 1) {
    pp.role = MainloopPipeline::ThreadCategory::Consumer;
  }
  pp.is_leader = (thread_idx % 128) == 0;
  pp.num_consumers = size(TiledMma{});
  pp.num_producers = CollectiveMainloop::NumProducerThreadEvents;
  pp.transaction_bytes = p.ml[0].tma_transaction_bytes;
  MainloopPipeline pipeline(ss.pipeline, pp, ClusterShape{});

  __syncthreads();
  // PDL: the previous kernels on the stream wrote our inputs; wait before reading any of them.
  asm volatile("griddepcontrol.wait;" ::: "memory");

  CollectiveMainloop mainloop;
  int const m = p.m;
  int const n = p.n;
  int const kt0 = (p.k[0] + kTileK - 1) / kTileK;
  int const kt1 = (p.k[1] + kTileK - 1) / kTileK;

  if (warp_group == 0) {
    cutlass::arch::warpgroup_reg_dealloc<40>();
    if (warp_in_wg == 0) {
      auto in0 = mainloop.load_init(make_shape(m, n, p.k[0], 1), p.ml[0]);
      auto in1 = mainloop.load_init(make_shape(m, n, p.k[1], 1), p.ml[1]);
      auto k_shape0 = shape<3>(get<0>(in0));
      auto k_shape1 = shape<3>(get<0>(in1));
      PipelineState st = cutlass::make_producer_start_state<MainloopPipeline>();
      for (int t = blockIdx.x; t < p.num_tiles; t += gridDim.x) {
        int const mt = t / p.tiles_n;
        int const nt = t - mt * p.tiles_n;
        auto blk = make_coord(mt, nt, _, 0);
        auto it0 = cute::make_coord_iterator(idx2crd(0, k_shape0), k_shape0);
        mainloop.load(p.ml[0], pipeline, st, in0, blk, it0, kt0, lane, 0u, ss.tensors);
        st.advance(kt0);
        if (p.prefetch_x) {
          // x rows [mt*128, +128), columns [nt*128, +128): 256 B = 2 lines per row.
          int const col0 = nt * kTileN;
          for (int r = lane; r < kTileM; r += 32) {
            int const row = mt * kTileM + r;
            if (row < m) {
              char const* ptr = reinterpret_cast<char const*>(p.x + size_t(row) * n + col0);
              asm volatile("prefetch.global.L2 [%0];" ::"l"(ptr));
              asm volatile("prefetch.global.L2 [%0];" ::"l"(ptr + 128));
            }
          }
        }
        auto it1 = cute::make_coord_iterator(idx2crd(0, k_shape1), k_shape1);
        mainloop.load(p.ml[1], pipeline, st, in1, blk, it1, kt1, lane, 0u, ss.tensors);
        st.advance(kt1);
      }
      mainloop.load_tail(pipeline, st);
    }
  } else {
    cutlass::arch::warpgroup_reg_alloc<232>();
    int const mma_tid = thread_idx - 128;
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_slice(mma_tid);
    auto tCcC =
        thr_mma.partition_C(make_identity_tensor(make_shape(Int<kTileM>{}, Int<kTileN>{})));
    float const alpha0 = *p.alpha[0];
    float const alpha1 = *p.alpha[1];
    PipelineState st;
    for (int t = blockIdx.x; t < p.num_tiles; t += gridDim.x) {
      int const mt = t / p.tiles_n;
      int const nt = t - mt * p.tiles_n;
      auto blk = make_coord(mt, nt, _, 0);
      auto acc = partition_fragment_C(tiled_mma, take<0, 2>(TileShape{}));
      constexpr int kVals = decltype(size(acc))::value;
      static_assert(kVals % 2 == 0, "accumulator pairs");
      constexpr int kPairs = kVals / 2;

      mainloop.mma(pipeline, st, acc, kt0, mma_tid, ss.tensors, p.ml[0], blk);
      st.advance(kt0);
      __nv_bfloat162 attn[kPairs];
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < kPairs; ++j) {
        // BF16(alpha * acc) as the LPT pair's LinearCombination epilogue computes it (same
        // compile flags: a plain fp32 multiply, then round to nearest).
        attn[j] = __floats2bfloat162_rn(alpha0 * acc(2 * j), alpha0 * acc(2 * j + 1));
      }

      int const m0 = mt * kTileM;
      int const n0 = nt * kTileN;
      // The residual operands of this thread's accumulator pairs (rows >= M stay unused).
      uint32_t xv[kPairs];
      uint32_t gv[kPairs];
      auto load_residual = [&]() {
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < kPairs; ++j) {
          auto crd = tCcC(2 * j);
          int const r = m0 + int(get<0>(crd));
          int const c = n0 + int(get<1>(crd));
          xv[j] = 0u;
          gv[j] = 0u;
          if (r < m) {
            size_t g = 0;
            if (p.gate_mode == 1) {
              g = __ldg(p.select + r);
            } else if (p.gate_mode == 2) {
              g = size_t(r);
            }
            xv[j] = ldg_nc_b32(p.x + size_t(r) * n + c);
            gv[j] = ldg_nc_b32(p.gate + g * n + c);
          }
        }
      };
      // kEarly: issue the loads before down's K loop so they land while it runs.
      if constexpr (kEarly) {
        if (p.mode == 0) {
          load_residual();
        }
      }

      mainloop.mma(pipeline, st, acc, kt1, mma_tid, ss.tensors, p.ml[1], blk);
      st.advance(kt1);

      if (p.mode == 2) {
        // Diagnostics: mainloops only (no epilogue traffic); keeps the results live.
        if (acc(0) != acc(0) && __low2float(attn[0]) != __low2float(attn[0])) {
          p.out[0] = __float2bfloat16_rn(acc(0));
        }
        continue;
      }
      if (p.mode == 0) {
        if constexpr (!kEarly) {
          load_residual();
        }
        auto x_new = [&](int j) {
          float const d0 = round_bf16(alpha1 * acc(2 * j));
          float const d1 = round_bf16(alpha1 * acc(2 * j + 1));
          float const s0 = round_bf16(add_rn(__low2float(attn[j]), d0));
          float const s1 = round_bf16(add_rn(__high2float(attn[j]), d1));
          __nv_bfloat162 const gp = as_bf162(gv[j]);
          __nv_bfloat162 const xp = as_bf162(xv[j]);
          float const q0 = round_bf16(mul_rn(__low2float(gp), s0));
          float const q1 = round_bf16(mul_rn(__high2float(gp), s1));
          return __floats2bfloat162_rn(add_rn(__low2float(xp), q0),
                                       add_rn(__high2float(xp), q1));
        };
#ifdef FLUX3_TRES_STAGED
        // The previous tile's bulk stores must have read the staging tile before it is reused.
        if (mma_tid < kTileM) {
          asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
        }
        cutlass::arch::NamedBarrier::sync(
            size(TiledMma{}), cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < kPairs; ++j) {
          auto crd = tCcC(2 * j);
          int const lr = int(get<0>(crd));
          int const lc = int(get<1>(crd));
          *reinterpret_cast<__nv_bfloat162*>(&ss.out_tile[lr * kOutRowElems + lc]) = x_new(j);
        }
        // Generic-proxy smem writes -> visible to the bulk-copy (async proxy) reads.
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        cutlass::arch::NamedBarrier::sync(
            size(TiledMma{}), cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
        if (mma_tid < kTileM && m0 + mma_tid < m) {
          uint32_t const src = static_cast<uint32_t>(
              __cvta_generic_to_shared(&ss.out_tile[mma_tid * kOutRowElems]));
          __nv_bfloat16* dst = p.out + size_t(m0 + mma_tid) * n + n0;
          asm volatile(
              "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(dst), "r"(src),
              "n"(kTileN * 2)
              : "memory");
          asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        }
#else
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < kPairs; ++j) {
          auto crd = tCcC(2 * j);
          int const r = m0 + int(get<0>(crd));
          int const c = n0 + int(get<1>(crd));
          if (r < m) {
            *reinterpret_cast<__nv_bfloat162*>(p.out + size_t(r) * n + c) = x_new(j);
          }
        }
#endif
      } else {
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < kPairs; ++j) {
          auto crd = tCcC(2 * j);
          auto crd1 = tCcC(2 * j + 1);
          int const r = m0 + int(get<0>(crd));
          int const c = n0 + int(get<1>(crd));
          if (int(get<0>(crd1)) != int(get<0>(crd)) || int(get<1>(crd1)) != int(get<1>(crd)) + 1) {
            __trap();  // the pair assumption of the mode-0 epilogue does not hold
          }
          if (r < m) {
            *reinterpret_cast<__nv_bfloat162*>(p.dbg_attn + size_t(r) * n + c) = attn[j];
            *reinterpret_cast<__nv_bfloat162*>(p.dbg_mlp + size_t(r) * n + c) =
                __floats2bfloat162_rn(alpha1 * acc(2 * j), alpha1 * acc(2 * j + 1));
          }
        }
      }
    }
#ifdef FLUX3_TRES_STAGED
    // Drain this CTA's bulk stores before it exits.
    if (mma_tid < kTileM) {
      asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
    }
#endif
  }
#endif
}

inline int device_sm_count(int device_id) {
  static thread_local int cached_device = -1;
  static thread_local int cached_count = 0;
  if (cached_device != device_id) {
    int count = 0;
    cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device_id);
    cached_count = count;
    cached_device = device_id;
  }
  return cached_count;
}

inline MainloopParams make_mainloop(void const* a, void const* b, void const* sfa, void const* sfb,
                                    int m, int n, int k) {
  MainloopArguments args{};
  args.ptr_A = static_cast<MmaElementA const*>(a);
  args.dA = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  args.ptr_B = static_cast<MmaElementB const*>(b);
  args.dB = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  args.ptr_SFA = static_cast<MmaElementSF const*>(sfa);
  args.layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
  args.ptr_SFB = static_cast<MmaElementSF const*>(sfb);
  args.layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));
  if (!CollectiveMainloop::can_implement(make_shape(m, n, k, 1), args)) {
    throw std::runtime_error("flux3_pair_residual: mainloop can_implement failed");
  }
  return CollectiveMainloop::to_underlying_arguments(make_shape(m, n, k, 1), args, nullptr);
}

}  // namespace flux3_tres

namespace torch_ext {

namespace {

void check_gemm(TensorView a, TensorView b, TensorView sfa, TensorView sfb, TensorView alpha) {
  TVM_FFI_ICHECK_EQ(a.dtype(), dl_uint8) << "a must be packed FP4 (uint8)";
  TVM_FFI_ICHECK_EQ(b.dtype(), dl_uint8) << "b must be packed FP4 (uint8)";
  TVM_FFI_ICHECK_EQ(sfa.dtype(), dl_uint8) << "sfa must be uint8";
  TVM_FFI_ICHECK_EQ(sfb.dtype(), dl_uint8) << "sfb must be uint8";
  TVM_FFI_ICHECK_EQ(alpha.dtype(), dl_float32) << "alpha must be float32";
  TVM_FFI_ICHECK_EQ(alpha.numel(), 1) << "alpha must have one element";
  TVM_FFI_ICHECK_EQ(a.ndim(), 2) << "a must be 2D";
  TVM_FFI_ICHECK_EQ(b.ndim(), 2) << "b must be 2D";
  TVM_FFI_ICHECK_EQ(b.size(1), a.size(1)) << "K mismatch";
  TVM_FFI_ICHECK((a.size(1) * 2) % 128 == 0) << "K must be a multiple of 128";
  int64_t const m = a.size(0);
  int64_t const k = a.size(1) * 2;
  int64_t const n = b.size(0);
  int64_t const sf_k = (k / 16 + 3) / 4 * 4;
  TVM_FFI_ICHECK(sfa.numel() >= (m + 127) / 128 * 128 * sf_k) << "sfa too small";
  TVM_FFI_ICHECK(sfb.numel() >= (n + 127) / 128 * 128 * sf_k) << "sfb too small";
  CHECK_DEVICE(a, b);
  CHECK_DEVICE(a, sfa);
  CHECK_DEVICE(a, sfb);
  CHECK_DEVICE(a, alpha);
}

}  // namespace

// x_new = x + gate * (attn_out(a_attn) + down(a_down)) with main's BF16 rounding points, one
// launch. mode 0: writes out; mode 1 (tests): writes BF16 attn_y to dbg_attn and mlp_y to
// dbg_mlp. gate_mode 0: gate is one [N] row; 1: gate is a [K, N] table and select (1-byte [M],
// bool or uint8) holds each row's table index; 2: gate is a per-row [M, N] tensor.
// grid_ctas 0: one CTA per SM. prefetch_x: producer-warp L2 prefetch of the x tile. early: the
// consumers issue the x / gate loads before down's K loop instead of after it.
void pair_residual(TensorView a_attn, TensorView b_attn, TensorView sfa_attn, TensorView sfb_attn,
                   TensorView alpha_attn, TensorView a_down, TensorView b_down, TensorView sfa_down,
                   TensorView sfb_down, TensorView alpha_down, TensorView x, TensorView gate,
                   TensorView select, int64_t gate_mode, TensorView out, TensorView dbg_attn,
                   TensorView dbg_mlp, int64_t mode, int64_t grid_ctas, int64_t prefetch_x,
                   int64_t early) {
  using namespace flux3_tres;
  check_gemm(a_attn, b_attn, sfa_attn, sfb_attn, alpha_attn);
  check_gemm(a_down, b_down, sfa_down, sfb_down, alpha_down);
  int const m = static_cast<int>(a_attn.size(0));
  int const n = static_cast<int>(b_attn.size(0));
  int const k0 = static_cast<int>(a_attn.size(1) * 2);
  int const k1 = static_cast<int>(a_down.size(1) * 2);
  TVM_FFI_ICHECK_EQ(a_down.size(0), m) << "attn_out and down rows differ";
  TVM_FFI_ICHECK_EQ(b_down.size(0), n) << "attn_out and down N differ";
  TVM_FFI_ICHECK(n % kTileN == 0) << "N must be a multiple of 128";
  TVM_FFI_ICHECK(mode >= 0 && mode <= 2) << "mode must be 0, 1 or 2 (diagnostics)";
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(x.dtype()), bfloat16_code) << "x must be bf16";
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(gate.dtype()), bfloat16_code) << "gate must be bf16";
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(out.dtype()), bfloat16_code) << "out must be bf16";
  TVM_FFI_ICHECK_EQ(x.numel(), static_cast<int64_t>(m) * n) << "x must hold M x N";
  TVM_FFI_ICHECK_EQ(out.numel(), static_cast<int64_t>(m) * n) << "out must hold M x N";
  TVM_FFI_ICHECK(gate_mode >= 0 && gate_mode <= 2) << "gate_mode must be 0, 1 or 2";
  if (gate_mode == 0) {
    TVM_FFI_ICHECK_EQ(gate.numel(), n) << "gate_mode 0: gate must hold one row of N";
  } else if (gate_mode == 1) {
    TVM_FFI_ICHECK(gate.numel() >= n && gate.numel() % n == 0 && gate.numel() / n <= 256)
        << "gate_mode 1: gate must be a [K, N] table";
    TVM_FFI_ICHECK_EQ(select.numel(), m) << "select must hold M entries";
    TVM_FFI_ICHECK_EQ(get_element_size(select), 1) << "select must be 1-byte (bool/uint8)";
    CHECK_DEVICE(a_attn, select);
  } else {
    TVM_FFI_ICHECK_EQ(gate.numel(), static_cast<int64_t>(m) * n)
        << "gate_mode 2: gate must hold M x N";
  }
  if (mode == 1) {
    TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(dbg_attn.dtype()), bfloat16_code);
    TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(dbg_mlp.dtype()), bfloat16_code);
    TVM_FFI_ICHECK_EQ(dbg_attn.numel(), static_cast<int64_t>(m) * n);
    TVM_FFI_ICHECK_EQ(dbg_mlp.numel(), static_cast<int64_t>(m) * n);
  }
  CHECK_DEVICE(a_attn, a_down);
  CHECK_DEVICE(a_attn, x);
  CHECK_DEVICE(a_attn, gate);
  CHECK_DEVICE(a_attn, out);

  KernelParams params{};
  params.ml[0] = make_mainloop(a_attn.data_ptr(), b_attn.data_ptr(), sfa_attn.data_ptr(),
                               sfb_attn.data_ptr(), m, n, k0);
  params.ml[1] = make_mainloop(a_down.data_ptr(), b_down.data_ptr(), sfa_down.data_ptr(),
                               sfb_down.data_ptr(), m, n, k1);
  params.m = m;
  params.n = n;
  params.k[0] = k0;
  params.k[1] = k1;
  params.tiles_n = n / kTileN;
  params.num_tiles = (m + kTileM - 1) / kTileM * params.tiles_n;
  params.mode = static_cast<int>(mode);
  params.prefetch_x = prefetch_x ? 1 : 0;
  params.gate_mode = static_cast<int>(gate_mode);
  params.alpha[0] = static_cast<float const*>(alpha_attn.data_ptr());
  params.alpha[1] = static_cast<float const*>(alpha_down.data_ptr());
  params.x = static_cast<__nv_bfloat16 const*>(x.data_ptr());
  params.gate = static_cast<__nv_bfloat16 const*>(gate.data_ptr());
  params.select = gate_mode == 1 ? static_cast<uint8_t const*>(select.data_ptr()) : nullptr;
  params.out = static_cast<__nv_bfloat16*>(out.data_ptr());
  params.dbg_attn = mode == 1 ? static_cast<__nv_bfloat16*>(dbg_attn.data_ptr()) : nullptr;
  params.dbg_mlp = mode == 1 ? static_cast<__nv_bfloat16*>(dbg_mlp.data_ptr()) : nullptr;
  if (params.num_tiles == 0) {
    return;
  }

  int const device_id = a_attn.device().device_id;
  int const ctas = grid_ctas > 0 ? static_cast<int>(grid_ctas) : device_sm_count(device_id);
  int const grid = ctas < params.num_tiles ? ctas : params.num_tiles;
  int const smem = static_cast<int>(sizeof(SharedStorage));

  static thread_local int attr_device = -1;
  if (attr_device != device_id) {
    for (auto fn : {flux3_pair_residual_kernel<true>, flux3_pair_residual_kernel<false>}) {
      cudaError_t e =
          cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
      TVM_FFI_ICHECK(e == cudaSuccess)
          << "flux3_pair_residual: smem attribute failed: " << cudaGetErrorString(e);
    }
    attr_device = device_id;
  }
  auto kernel = early ? flux3_pair_residual_kernel<true> : flux3_pair_residual_kernel<false>;

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(grid);
  config.blockDim = dim3(kThreads);
  config.dynamicSmemBytes = smem;
  config.stream = get_stream(a_attn.device());
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.numAttrs = 1;
  config.attrs = attrs;
  cudaError_t err = cudaLaunchKernelEx(&config, kernel, params);
  TVM_FFI_ICHECK(err == cudaSuccess)
      << "flux3_pair_residual: launch failed: " << cudaGetErrorString(err);
}

// Build facts: stages * 1000000 + shared storage bytes; params bytes separately.
int64_t pair_residual_info() {
  using namespace flux3_tres;
  return static_cast<int64_t>(CollectiveMainloop::DispatchPolicy::Stages) * 1000000 +
         static_cast<int64_t>(sizeof(SharedStorage));
}

int64_t pair_residual_param_bytes() {
  return static_cast<int64_t>(sizeof(flux3_tres::KernelParams));
}

}  // namespace torch_ext

TVM_FFI_DLL_EXPORT_TYPED_FUNC(pair_residual, torch_ext::pair_residual);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(pair_residual_info, torch_ext::pair_residual_info);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(pair_residual_param_bytes, torch_ext::pair_residual_param_bytes);
