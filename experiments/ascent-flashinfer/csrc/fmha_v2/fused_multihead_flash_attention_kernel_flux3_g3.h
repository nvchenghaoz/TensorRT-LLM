/*
 * FLUX.3 G3 shared-K/V variant of device_flash_attention_nl_tiled (idea 245, branch 165).
 *
 * One CTA = 3 groups x 4 warps (384 threads). Each group runs the unmodified 64-row
 * nl_tiled math (same Kernel_traits, same fragments, same MMA order, same softmax) on its own
 * 64-row Q tile (q_loop = 3 * blockIdx.x + group). K/V are loaded ONCE per CTA by one thread
 * with TMA into a STAGES-deep ring of 16 KB stages (K chunk0 4 KB | K chunk1 4 KB | V 8 KB),
 * guarded by full (tx-count) and empty (4 warps x active groups) mbarriers. The ring stages
 * reproduce the exact byte layout of Smem_tile_k (SWIZZLE_64B, 64-B rows packed 2 per 128-B
 * smem row) and Smem_tile_v (SWIZZLE_128B), so the unchanged Smem_tile_{q,k,v}::load()
 * fragment loads read identical values: the output is bitwise equal to the 4-warp kernel.
 * TMA out-of-bounds rows are zero-filled, like the predicated-off LDGSTS (src-size 0).
 *
 * Supported: padding mask (non-causal), no ALiBi, no softmax stats, h == h_kv, d == dv == 128,
 * e4m3, fixed-stride BSHD (is_s_padded, no cu_seqlens). The launcher falls back otherwise.
 *
 * This header is included by the generated kernel file after the stock kernels (global scope),
 * so it relies on the fmha headers that file already includes.
 */
#pragma once

#include <cuda.h>
#include <cudaTypedefs.h>

#include <stdexcept>

namespace fused_multihead_attention {
namespace flux3_g3 {

#ifndef FMHA_FLUX3_G3_STAGES
#define FMHA_FLUX3_G3_STAGES 4
#endif

enum { GROUPS = 3 };
enum { THREADS = GROUPS * 128 };
enum { STAGES = FMHA_FLUX3_G3_STAGES };
enum { K_CHUNK_BYTES = 4096, K_BYTES = 8192, V_BYTES = 8192, STAGE_BYTES = K_BYTES + V_BYTES };
enum { RING_BYTES = STAGES * STAGE_BYTES };

template <typename Kernel_traits>
struct Layout {
  enum { Q_BYTES = Kernel_traits::Smem_tile_q::BYTES_PER_TILE };
  enum { O_BYTES = Kernel_traits::Smem_tile_o::BYTES_PER_TILE };
  enum { Q_OFFSET = RING_BYTES };
  enum {
    DATA_BYTES = (RING_BYTES + GROUPS * Q_BYTES) > (GROUPS * O_BYTES)
                     ? (RING_BYTES + GROUPS * Q_BYTES)
                     : (GROUPS * O_BYTES)
  };
  enum { BAR_OFFSET = (DATA_BYTES + 127) / 128 * 128 };
  enum { SMEM_BYTES = BAR_OFFSET + 2 * STAGES * 8 };
};

inline __device__ uint32_t smem_u32(void const* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

inline __device__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar), "r"(count) : "memory");
}

inline __device__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar), "r"(bytes)
               : "memory");
}

inline __device__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(bar) : "memory");
}

inline __device__ bool mbar_try_wait(uint32_t bar, uint32_t phase) {
  uint32_t ok;
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
      "selp.b32 %0, 1, 0, p;\n\t}\n"
      : "=r"(ok)
      : "r"(bar), "r"(phase)
      : "memory");
  return ok != 0;
}

inline __device__ bool mbar_test_wait(uint32_t bar, uint32_t phase) {
  uint32_t ok;
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
      "selp.b32 %0, 1, 0, p;\n\t}\n"
      : "=r"(ok)
      : "r"(bar), "r"(phase)
      : "memory");
  return ok != 0;
}

inline __device__ void mbar_wait(uint32_t bar, uint32_t phase) {
  while (!mbar_try_wait(bar, phase)) {
  }
}

inline __device__ void tma_load_4d(uint32_t dst, CUtensorMap const* map, uint32_t bar, int c0,
                                   int c1, int c2, int c3) {
  asm volatile(
      // CTA-local destination: the .shared::cluster form makes the compiler add a per-copy
      // "is this my CTA" check with a CALL slow path (8 B of stack) in a non-cluster launch.
      "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes "
      "[%0], [%1, {%2, %3, %4, %5}], [%6];\n" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(bar)
      : "memory");
}

inline __device__ void named_bar_sync(int id, int threads) {
  asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(threads) : "memory");
}

// Issue the TMA loads of KV step j (64 keys) into ring stage j % STAGES.
inline __device__ void issue_stage(int j, uint32_t ring, uint32_t full0, CUtensorMap const* tm_k,
                                   CUtensorMap const* tm_v, int bidh, int bidb) {
  int const slot = j % STAGES;
  uint32_t const bar = full0 + slot * 8;
  uint32_t const dst = ring + slot * STAGE_BYTES;
  mbar_expect_tx(bar, STAGE_BYTES);
  tma_load_4d(dst, tm_k, bar, 0, bidh, j * 64, bidb);
  tma_load_4d(dst + K_CHUNK_BYTES, tm_k, bar, 64, bidh, j * 64, bidb);
  tma_load_4d(dst + K_BYTES, tm_v, bar, 0, bidh, j * 64, bidb);
}

#if defined(FMHA_FLUX3_V_B8T) && FMHA_FLUX3_V_B8T
// v_b8t (idea 39): V B-fragments with one ldmatrix.m16n16.x2.trans.b8 per (ki, 16-d chunk)
// instead of LDSM.MT88.4 (b16 transpose) + 4 PRMT (swizzle_rows).
//
// The V stage is [64 keys][128 d] e4m3, 128-B rows, SWIZZLE_128B (16-B chunk c of row r at
// chunk c ^ (r & 7)). Lanes 0-15 / 16-31 give the row addresses of matrix 0 / 1 (keys 0-15 /
// 16-31 of the k32 step). Lane i addresses key perm(i), perm(4q + j) = 2q + (j & 1) + 8 (j >> 1)
// (+16 for matrix 1): thread (g = lane/4, q = lane%4) then gets, in k-slots 4q..4q+3, the keys
// {2q, 2q+1, 2q+8, 2q+9} -- the k-order of P's A-fragment, exactly as main's repack gives.
// So every QMMA multiplies the same P bytes by the same V bytes per k-slot as main.
//
// Head dim: r0 / r1 hold d = base + g / base + 8 + g (main: base + 2g / base + 2g + 1), so
// acc_o's n-col -> d map differs; v_b8t_unpermute() restores main's layout once per tile.
inline __device__ uint32_t v_b8t_lane_offset(int lane) {
  int const key = 16 * (lane >> 4) + 2 * ((lane >> 2) & 3) + (lane & 1) + 8 * ((lane >> 1) & 1);
  return (uint32_t)(key * 128 + ((key & 7) << 4));
}

template <typename Fragment>
inline __device__ void ldsm_v_b8t(Fragment& b, uint32_t addr) {
  uint32_t r0, r1, r2, r3;
  asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared::cta.b8 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
  // MMA 0 (b.reg 0/1): d = base + g, keys 0-15 / 16-31; MMA 1 (b.reg 2/3): d = base + 8 + g.
  b.reg(0) = r0;
  b.reg(1) = r2;
  b.reg(2) = r1;
  b.reg(3) = r3;
}

// One helper call per (ki, d-chunk ni): v_stage = the stage's V base (1024-B aligned).
template <typename Fragment>
inline __device__ void load_v_b8t_chunk(Fragment& b, uint32_t v_stage, uint32_t lane_off, int ki,
                                        int ni) {
  ldsm_v_b8t(b, v_stage + (uint32_t)(ki * 32 * 128) + (lane_off ^ (uint32_t)(ni << 4)));
}

template <typename Fragment, int N>
inline __device__ void load_v_b8t(Fragment (&b)[N], uint32_t v_stage, uint32_t lane_off, int ki) {
#pragma unroll
  for (int ni = 0; ni < N; ++ni) {
    load_v_b8t_chunk(b[ni], v_stage, lane_off, ki, ni);
  }
}

// Quad-local move of acc_o from the v_b8t d-map to main's. Per 16-d chunk and row half h, thread
// q holds pairs L = d {2q, 2q+1} (regs 2h, 2h+1) and H = d {8+2q, 9+2q} (regs 4+2h, 5+2h); main
// has thread q hold d 4q (reg 2h), 4q+1 (reg 4+2h), 4q+2 (reg 2h+1), 4q+3 (reg 5+2h), i.e. pair
// indices p = 2q, 2q+1 where the b8t thread p % 4 holds pair p. Two shuffle rounds of a pair:
//   round A: send (q odd ? L : H), read lane 2(q&1) + (q < 2)
//   round B: send (q odd ? H : L), read lane 2(q&1) + (q >> 1)
//   pair 2q = (q < 2 ? B : A), pair 2q+1 = (q < 2 ? A : B).
template <typename Acc, int M, int N>
inline __device__ void v_b8t_unpermute(Acc (&acc)[M][N], int lane) {
  int const q = lane & 3;
  bool const odd = (q & 1) != 0;
  bool const low = q < 2;
  int const quad = lane & ~3;
  int const src_a = quad + 2 * (q & 1) + (low ? 1 : 0);
  int const src_b = quad + 2 * (q & 1) + (q >> 1);
#pragma unroll
  for (int mi = 0; mi < M; ++mi) {
#pragma unroll
    for (int ni = 0; ni < N; ++ni) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        uint32_t const lx = acc[mi][ni].reg(2 * h), ly = acc[mi][ni].reg(2 * h + 1);
        uint32_t const hx = acc[mi][ni].reg(4 + 2 * h), hy = acc[mi][ni].reg(5 + 2 * h);
        uint32_t const ax = __shfl_sync(0xffffffffu, odd ? lx : hx, src_a);
        uint32_t const ay = __shfl_sync(0xffffffffu, odd ? ly : hy, src_a);
        uint32_t const bx = __shfl_sync(0xffffffffu, odd ? hx : lx, src_b);
        uint32_t const by = __shfl_sync(0xffffffffu, odd ? hy : ly, src_b);
        acc[mi][ni].reg(2 * h) = low ? bx : ax;      // d 4q
        acc[mi][ni].reg(4 + 2 * h) = low ? by : ay;  // d 4q+1
        acc[mi][ni].reg(2 * h + 1) = low ? ax : bx;  // d 4q+2
        acc[mi][ni].reg(5 + 2 * h) = low ? ay : by;  // d 4q+3
      }
    }
  }
}
#endif  // FMHA_FLUX3_V_B8T

}  // namespace flux3_g3

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits, typename Params>
inline __device__ void device_flash_attention_nl_tiled_flux3_g3(Params const& params,
                                                                CUtensorMap const* tm_k,
                                                                CUtensorMap const* tm_v) {
  using namespace flux3_g3;
  using Traits_p = typename Kernel_traits::Traits_p;
  using Traits_o = typename Kernel_traits::Traits_o;
  using Cta_tile_p = typename Kernel_traits::Cta_tile_p;
  using Cta_tile_o = typename Kernel_traits::Cta_tile_o;
  using Mma_tile_p = typename Traits_p::template Mma_tile<Cta_tile_p>;
  using Mma_tile_o = typename Traits_o::template Mma_tile<Cta_tile_o>;
  using Gmem_tile_q = typename Kernel_traits::Gmem_tile_q;
  using Smem_tile_q = typename Kernel_traits::Smem_tile_q;
  using Smem_tile_k = typename Kernel_traits::Smem_tile_k;
  using Smem_tile_v = typename Kernel_traits::Smem_tile_v;
  using Gmem_tile_o = typename Kernel_traits::Gmem_tile_o;
  using Smem_tile_o = typename Kernel_traits::Smem_tile_o;
  using L = Layout<Kernel_traits>;

  enum { USE_LDGSTS = Kernel_traits::USE_LDGSTS_Q };
  static_assert(USE_LDGSTS, "");
  static_assert(Kernel_traits::THREADS == 128, "G3 runs three copies of the 4-warp tile");
  static_assert(Cta_tile_p::M == 64 && Cta_tile_p::N == 64, "");
  static_assert((int)Kernel_traits::CTA_P_TILE_K == 64 && (int)Kernel_traits::VALID_D == 128, "");
  static_assert((int)Kernel_traits::BUFFERS_PER_TILE_SMEM_Q == 2, "");
  static_assert((int)Smem_tile_k::BYTES_PER_BUFFER == K_CHUNK_BYTES, "");
  static_assert((int)Smem_tile_v::BYTES_PER_BUFFER == V_BYTES, "");
  static_assert((int)Smem_tile_k::ROWS_PER_XOR_PATTERN == 4, "K layout must be SWIZZLE_64B");
  static_assert((int)Smem_tile_v::ROWS_PER_XOR_PATTERN == 8, "V layout must be SWIZZLE_128B");
  static_assert((int)Smem_tile_k::BYTES_PER_ROW == 128 && (int)Smem_tile_v::BYTES_PER_ROW == 128,
                "");
  static_assert((int)Kernel_traits::TOTAL_BMM2_MMAS_K == (int)Mma_tile_o::MMAS_K,
                "one V tile per KV step");
  static_assert(Kernel_traits::MASK_VERSION == 2 && !Kernel_traits::IS_MTP, "padding mask only");
  static_assert(!Kernel_traits::LIMIT_QK_FRAGMENTS && !Kernel_traits::SHARE_SMEM_FOR_K_AND_V, "");
  static_assert(Gmem_tile_o::LOOPS >= 1, "");

  constexpr int BMM1_VALID_MMAS_K = Mma_tile_p::VALID_MMAS_K;
  constexpr int BMM1_TAIL_MMAS_K_BOUND = BMM1_VALID_MMAS_K % Mma_tile_p::MMAS_K
                                             ? BMM1_VALID_MMAS_K % Mma_tile_p::MMAS_K
                                             : Mma_tile_p::MMAS_K;
  constexpr int BMM1_MAIN_MMAS_K_BOUND = BMM1_VALID_MMAS_K - BMM1_TAIL_MMAS_K_BOUND;
  static_assert(BMM1_MAIN_MMAS_K_BOUND == Mma_tile_p::MMAS_K, "two 64-col K chunks");
  constexpr int BMM2_TAIL_MMAS_K_BOUND = Mma_tile_o::MMAS_K;

  enum { CHECK_NEG_INF = Kernel_traits::SLIDING_WINDOW_ATTENTION || Kernel_traits::CUSTOM_MASK };
  using Softmax = fmha::Softmax<Traits_p, Cta_tile_p, Kernel_traits>;
  static_assert(!Softmax::USE_SHARED_MEMORY, "");

  extern __shared__ __align__(1024) char smem_[];

  int const tid = threadIdx.x;
  int const group = tid / 128;
  int const tidx = tid % 128;  // the thread index inside the 4-warp tile
  int const lane = tid % 32;
  int const bidb = blockIdx.z;
  int const bidh = blockIdx.y;
  int const q_loop = blockIdx.x * GROUPS + group;

  Single_cta<Kernel_traits::VERSION> const binfo(params, bidb, bidh, 0, tidx);
  int const q_tiles = (binfo.actual_q_seqlen + Gmem_tile_q::ROWS - 1) / Gmem_tile_q::ROWS;
  int active_groups = q_tiles - (int)blockIdx.x * GROUPS;
  active_groups = active_groups < GROUPS ? active_groups : GROUPS;
  bool const active = group < active_groups;
  int const nsteps = (binfo.actual_kv_seqlen + Cta_tile_p::N - 1) / Cta_tile_p::N;

  uint32_t const ring = smem_u32(smem_);
  uint32_t const full0 = smem_u32(smem_ + L::BAR_OFFSET);
  uint32_t const empty0 = full0 + STAGES * 8;

  // Barrier init + TMA prologue (one thread).
  int issued = 0;
  if (tid == 0) {
    if (ring & 1023u) {
      __trap();  // SWIZZLE_128B/64B stages need a 1024-B aligned ring base
    }
#pragma unroll
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(full0 + s * 8, 1);
      mbar_init(empty0 + s * 8, 4 * active_groups);
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    asm volatile("prefetch.tensormap [%0];\n" ::"l"(reinterpret_cast<uint64_t>(tm_k))
                 : "memory");
    asm volatile("prefetch.tensormap [%0];\n" ::"l"(reinterpret_cast<uint64_t>(tm_v))
                 : "memory");
  }
  __syncthreads();
  if (tid == 0) {
    int const n0 = nsteps < STAGES ? nsteps : STAGES;
    for (; issued < n0; ++issued) {
      issue_stage(issued, ring, full0, tm_k, tm_v, bidh, bidb);
    }
  }

  char* const smem_q_ptr = smem_ + L::Q_OFFSET + group * L::Q_BYTES;

  // The local sequence offset of Q (non-causal, no past KV: q_sequence_start = q_loop * 64).
  int const q_sequence_start =
      q_loop * Gmem_tile_q::ROWS + binfo.actual_kv_seqlen - binfo.actual_q_seqlen;

  fmha::Mask_dispatcher<Traits_p, Cta_tile_p, Kernel_traits::MASK_VERSION, Kernel_traits::IS_MTP>
      mask(params, binfo, tidx);

  Gmem_tile_q gmem_q(params, 0, binfo, tidx, q_loop * Gmem_tile_q::ROWS);
  Smem_tile_q smem_q(smem_q_ptr, tidx);
  Smem_tile_k smem_k(smem_, tidx);
  Smem_tile_v smem_v(smem_ + K_BYTES, tidx);
#if defined(FMHA_FLUX3_V_B8T) && FMHA_FLUX3_V_B8T
  static_assert((int)Mma_tile_o::VALID_MMAS_N == 8 && (int)Mma_tile_o::MMAS_K == 2, "");
  uint32_t const v_ring = smem_u32(smem_ + K_BYTES);
  uint32_t const v_lane_off = v_b8t_lane_offset(lane);
#endif

#if defined(FMHA_FLUX3_MASK_SKIP) && FMHA_FLUX3_MASK_SKIP
  int const kv_mask_loop_start = int(binfo.actual_seqlen / Cta_tile_p::N) * Cta_tile_p::N;
#else
  int const kv_mask_loop_start = int(q_sequence_start / Cta_tile_p::N) * Cta_tile_p::N;
#endif

  typename Smem_tile_q::Fragment frag_q[Mma_tile_p::MMAS_K][Mma_tile_p::MMAS_M];
  typename Smem_tile_k::Fragment frag_k[Mma_tile_p::MMAS_K][Mma_tile_p::MMAS_N];
  typename Smem_tile_v::Fragment frag_v[Mma_tile_o::MMAS_K][Mma_tile_o::VALID_MMAS_N];

  Softmax softmax(params, smem_q_ptr, bidb, tidx);

  fmha::Fragment_accumulator<Traits_o> acc_o[Mma_tile_o::MMAS_M][Mma_tile_o::VALID_MMAS_N];
  using Acc_type_o = typename Traits_o::Accumulator_type;
  fmha::Clear_accumulator<Acc_type_o, Cta_tile_o::WARPS_K>::apply(acc_o);

  fmha::Tile_o_normalizer<Traits_o, Cta_tile_o> acc_o_normalizer(params, binfo);
  float global_max[Softmax::ROWS_PER_THREAD];
  float global_sum[Softmax::ROWS_PER_THREAD];

  if (active) {
    // Resident Q: both 64-column chunks, once (the same bytes the 4-warp kernel loads).
    mask.load(q_sequence_start);
    gmem_q.load(smem_q);
    gmem_q.move_col();
    smem_q.move_to_next_write_buffer();
    gmem_q.load(smem_q);
    fmha::ldgdepbar<USE_LDGSTS>();
    fmha::depbar_<USE_LDGSTS, 0>();
    named_bar_sync(1 + group, 128);

#if defined(FMHA_FLUX3_Q_REGS) && FMHA_FLUX3_Q_REGS
    // q_regs: the Q A-fragments of both 64-column chunks, loaded from smem once (bitwise equal).
    typename Smem_tile_q::Fragment frag_q_all[2][Mma_tile_p::MMAS_K][Mma_tile_p::MMAS_M];
#pragma unroll
    for (int c = 0; c < 2; ++c) {
#pragma unroll
      for (int ki = 0; ki < Mma_tile_p::MMAS_K; ++ki) {
        smem_q.load(frag_q_all[c][ki], ki);
      }
      smem_q.move_to_next_read_buffer();
    }
#endif

    for (int step = 0; step < nsteps; ++step) {
      int const kv_loop = step * Cta_tile_p::N;
      int const slot = step % STAGES;
      uint32_t const phase = (step / STAGES) & 1;

      // Producer: keep the ring full. The load of this step must be issued before we wait.
      if (tid == 0) {
        while (issued < nsteps && issued < step + STAGES) {
          int const j = issued;
          uint32_t const ebar = empty0 + (j % STAGES) * 8;
          uint32_t const eph = ((j / STAGES) - 1) & 1;
          if (j <= step) {
            mbar_wait(ebar, eph);
          } else if (!mbar_test_wait(ebar, eph)) {
            break;
          }
          issue_stage(j, ring, full0, tm_k, tm_v, bidh, bidb);
          ++issued;
        }
      }

      bool const first_step = (step == 0);
      bool const apply_mask = params.has_alibi || (kv_loop >= kv_mask_loop_start);

      fmha::Fragment_accumulator<Traits_p> acc_p[Mma_tile_p::MMAS_M][Mma_tile_p::MMAS_N];
      using Acc_type_p = typename Traits_p::Accumulator_type;
      fmha::Clear_accumulator<Acc_type_p, Cta_tile_p::WARPS_K>::apply(acc_p);

      mask.move_to_offset(kv_loop);

      // Wait for this step's K/V stage.
      mbar_wait(full0 + slot * 8, phase);

      // BMM1, K columns 0..63 (stage chunk 0).
      smem_k.smem_read_buffer_ = slot * STAGE_BYTES;
#if defined(FMHA_FLUX3_Q_REGS) && FMHA_FLUX3_Q_REGS
#pragma unroll
      for (int ki = 0; ki < Mma_tile_p::MMAS_K; ++ki) {
        smem_k.load(frag_k[ki], ki);
        fmha::gemm(acc_p, frag_q_all[0][ki], frag_k[ki]);
      }

      // BMM1, K columns 64..127 (stage chunk 1).
      smem_k.smem_read_buffer_ = slot * STAGE_BYTES + K_CHUNK_BYTES;
#pragma unroll
      for (int ki = 0; ki < Mma_tile_p::MMAS_K; ++ki) {
        smem_k.load(frag_k[ki], ki);
        if (Cta_tile_p::VALID_K % Cta_tile_p::K == 0 || ki < BMM1_TAIL_MMAS_K_BOUND) {
          fmha::gemm(acc_p, frag_q_all[1][ki], frag_k[ki]);
        }
      }
#else
#pragma unroll
      for (int ki = 0; ki < Mma_tile_p::MMAS_K; ++ki) {
        smem_q.load(frag_q[ki], ki);
        smem_k.load(frag_k[ki], ki);
        fmha::gemm(acc_p, frag_q[ki], frag_k[ki]);
      }
      smem_q.move_to_next_read_buffer();

      // BMM1, K columns 64..127 (stage chunk 1).
      smem_k.smem_read_buffer_ = slot * STAGE_BYTES + K_CHUNK_BYTES;
#pragma unroll
      for (int ki = 0; ki < Mma_tile_p::MMAS_K; ++ki) {
        smem_q.load(frag_q[ki], ki);
        smem_k.load(frag_k[ki], ki);
        if (Cta_tile_p::VALID_K % Cta_tile_p::K == 0 || ki < BMM1_TAIL_MMAS_K_BOUND) {
          fmha::gemm(acc_p, frag_q[ki], frag_k[ki]);
        }
      }
      smem_q.move_to_next_read_buffer();
#endif

      softmax.unpack(acc_p);

      if (apply_mask) {
        if (params.has_alibi) {
          softmax.apply_mask_alibi(mask, bidh, params.alibi_params);
        } else {
          softmax.apply_mask(mask);
        }
      }

      if (first_step) {
        softmax.template reduce<fmha::Max_>(global_max);
        softmax.template apply_exp_with_mask<CHECK_NEG_INF>(global_max);
        softmax.template reduce<fmha::Sum_>(global_sum);
      } else {
        float tmp[Softmax::ROWS_PER_THREAD];
#pragma unroll
        for (int i = 0; i < Softmax::ROWS_PER_THREAD; i++) {
          tmp[i] = global_max[i];
        }
        softmax.template reduce<fmha::Max_>(global_max);
        acc_o_normalizer.update(acc_o, global_max, tmp, global_sum);
        softmax.template apply_exp_with_mask<CHECK_NEG_INF>(global_max);
#pragma unroll
        for (int i = 0; i < Softmax::ROWS_PER_THREAD; i++) {
          tmp[i] = global_sum[i];
          global_sum[i] = 0.f;
        }
        softmax.template reduce<fmha::Sum_>(global_sum);
#pragma unroll
        for (int i = 0; i < Softmax::ROWS_PER_THREAD; i++) {
          global_sum[i] += tmp[i];
        }
      }

      fmha::Fragment_a<Traits_p, fmha::Row> frag_p[Kernel_traits::TOTAL_BMM2_MMAS_K]
                                                  [Mma_tile_o::MMAS_M];
      softmax.pack(frag_p);

      // BMM2 on the stage's V tile.
#if defined(FMHA_FLUX3_V_B8T) && FMHA_FLUX3_V_B8T
      uint32_t const v_stage = v_ring + slot * STAGE_BYTES;
#pragma unroll
      for (int ki = 0; ki < BMM2_TAIL_MMAS_K_BOUND; ++ki) {
        load_v_b8t(frag_v[ki], v_stage, v_lane_off, ki);
        fmha::gemm(acc_o, frag_p[ki], frag_v[ki]);
      }
#else
      smem_v.smem_read_buffer_ = slot * STAGE_BYTES;
#pragma unroll
      for (int ki = 0; ki < BMM2_TAIL_MMAS_K_BOUND; ++ki) {
        smem_v.load(frag_v[ki], ki);
        fmha::gemm(acc_o, frag_p[ki], frag_v[ki]);
      }
#endif

      // Release the stage: every warp of every active group arrives once.
      __syncwarp();
      if (lane == 0) {
        mbar_arrive(empty0 + slot * 8);
      }
    }

    acc_o_normalizer.update_sum(global_max, global_sum);
    acc_o_normalizer.final_update(acc_o, global_sum);
#if defined(FMHA_FLUX3_V_B8T) && FMHA_FLUX3_V_B8T
    // Back to main's acc_o d-map before the unchanged BF16 / NVFP4 epilogues.
    v_b8t_unpermute(acc_o, lane);
#endif
  }

  // Every group is done with the ring and the Q tiles; all issued TMA loads were consumed.
  __syncthreads();

  if (active) {
    Gmem_tile_o gmem_o(params, binfo, tidx, q_loop * Gmem_tile_o::ROWS, 0);
    Smem_tile_o smem_o(smem_ + group * L::O_BYTES, tidx);
#if defined(FMHA_FLUX3_NVFP4_OUT) && FMHA_FLUX3_NVFP4_OUT
    // NVFP4 output epilogue (fmha/flux3_nvfp4_out.h), per group: the 4-warp kernel's loop with
    // its __syncthreads() replaced by the group's named barrier. run_nvfp4_out passes the FP4
    // buffer as o_ptr, so this path must never reach gmem_o.store. All indices are group-local
    // (tidx 0..127, smem_o at the group's O region, gmem_o at the group's q_loop).
    if (params.flux3_fp4_ptr != nullptr) {
#pragma unroll
      for (int ii = 0; ii < Gmem_tile_o::LOOPS; ++ii) {
        smem_o.store(acc_o, ii);
        named_bar_sync(1 + group, 128);
        fmha::flux3_nvfp4::store_from_smem(params, gmem_o, smem_o, ii, bidb, bidh, tidx);
        if (ii < Gmem_tile_o::LOOPS - 1) {
          named_bar_sync(1 + group, 128);
        }
      }
      fmha::flux3_nvfp4::zero_padding(params, Gmem_tile_o::ROWS, q_loop, bidb, bidh, tidx, 128);
      return;
    }
#endif  // FMHA_FLUX3_NVFP4_OUT
#pragma unroll
    for (int ii = 0; ii < Gmem_tile_o::LOOPS; ++ii) {
      smem_o.store(acc_o, ii);
      named_bar_sync(1 + group, 128);
      uint4 out[Gmem_tile_o::STGS_PER_LOOP];
      smem_o.load(out);
      if (ii < Gmem_tile_o::LOOPS - 1) {
        named_bar_sync(1 + group, 128);
      }
      gmem_o.store(out, ii);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace flux3_g3 {

inline PFN_cuTensorMapEncodeTiled_v12000 encode_fn() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (fn == nullptr) {
    cudaDriverEntryPointQueryResult status;
    void* ptr = nullptr;
#if CUDA_VERSION >= 12050
    cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &ptr, 12000, cudaEnableDefault,
                                     &status);
#else
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &ptr, cudaEnableDefault, &status);
#endif
    if (status != cudaDriverEntryPointSuccess || ptr == nullptr) {
      throw std::runtime_error("flux3 g3: cuTensorMapEncodeTiled unavailable");
    }
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(ptr);
  }
  return fn;
}

// [B, S, H, D] e4m3 tensor, box {box_inner bytes, 1 head, 64 rows, 1 batch}.
inline void make_map(CUtensorMap* map, void const* ptr, int b, int s, int h, int d,
                     int64_t row_stride_bytes, uint32_t box_inner, CUtensorMapSwizzle swizzle) {
  cuuint64_t dims[4] = {(cuuint64_t)d, (cuuint64_t)h, (cuuint64_t)s, (cuuint64_t)b};
  cuuint64_t strides[3] = {(cuuint64_t)d, (cuuint64_t)row_stride_bytes,
                           (cuuint64_t)row_stride_bytes * (cuuint64_t)s};
  cuuint32_t box[4] = {box_inner, 1, 64, 1};
  cuuint32_t elem_strides[4] = {1, 1, 1, 1};
  CUresult r = encode_fn()(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, const_cast<void*>(ptr), dims,
                           strides, box, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
                           CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) {
    throw std::runtime_error("flux3 g3: cuTensorMapEncodeTiled failed");
  }
}

inline bool supported(bert::Fused_multihead_attention_params_v2 const& params,
                      bert::Fused_multihead_attention_launch_params const& launch_params) {
  // NVFP4-out modules (FMHA_FLUX3_NVFP4_OUT): the G3 epilogue branches on params.flux3_fp4_ptr
  // like the 4-warp kernel and writes FP4 + SF per group (never BF16 into the FP4 o_ptr).
  if (launch_params.attention_mask_type != fmha::Attention_mask_type::PADDING) return false;
  if (params.has_alibi || params.softmax_stats_ptr != nullptr) return false;
  if (params.cu_q_seqlens != nullptr || params.cu_kv_seqlens != nullptr) return false;
  if (!params.is_s_padded) return false;
  if (params.h_kv != params.h || params.d != 128 || params.dv != 128) return false;
  if (params.k_ptr == nullptr || params.v_ptr == nullptr) return false;
  if ((reinterpret_cast<uintptr_t>(params.k_ptr) & 15) || (reinterpret_cast<uintptr_t>(params.v_ptr) & 15))
    return false;
  if ((params.k_stride_in_bytes & 15) || (params.v_stride_in_bytes & 15)) return false;
  if (params.s <= 0 || params.b <= 0) return false;
  return true;
}

}  // namespace flux3_g3
}  // namespace fused_multihead_attention
