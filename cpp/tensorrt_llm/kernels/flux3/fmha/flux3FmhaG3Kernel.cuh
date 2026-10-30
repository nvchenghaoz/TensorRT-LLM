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

#pragma once

// Device code of the FLUX.3 FP8 attention on SM120 (see flux3FmhaSm120.h): the tiled flash kernel
// (device_flash_attention_nl_tiled) of the e4m3 / fp32-accumulator 64x64 tile, head 128, run three times per CTA.
//
// One CTA = 3 groups x 4 warps (384 threads, 1 CTA/SM). Each group runs the 64-row nl_tiled math (same Kernel_traits,
// fragments and MMA order) on its own 64-row Q tile (q_loop = 3 * blockIdx.x + group). Q is loaded once per group and
// its A-fragments stay in registers for the whole KV loop. K/V are loaded once per CTA by one thread with TMA into a
// 4-stage ring of 16 KB stages (K chunk 0 4 KB | K chunk 1 4 KB | V 8 KB), guarded by full (tx-count) and empty
// (4 warps x active groups) mbarriers. The K stages reproduce the byte layout of Smem_tile_k (SWIZZLE_64B, 64-B rows
// packed 2 per 128-B smem row), so Smem_tile_k::load reads the values the LDGSTS kernel reads. TMA out-of-bounds rows
// are zero-filled, like a predicated-off LDGSTS (src-size 0).
//
// V: the stage holds the 64 keys x 128 d E4M3 tile in 128-B rows (SWIZZLE_128B) with key bits 2 and 3 exchanged
// inside each 16-key group (the 5D "swap23" map, make_v_map), and every V B-fragment is one
// ldmatrix.m16n16.x2.trans.b8 from per-lane key-permuted rows: lanes 0-7 then address 8 distinct 16-B chunks (no bank
// conflict), and each QMMA multiplies the same P bytes by the same V bytes per k-slot as the LDSM.MT88 + PRMT path.
// The head-dim order of acc_o differs from that path's; unpermute_acc_o restores it once per tile before the epilogue.
//
// Softmax: flux3FmhaSoftmax.cuh (base 2). The padding mask runs only on the last, partial KV step (every earlier step
// is all-valid); a full last step is not masked.
//
// Supported: padding mask (non-causal), no ALiBi, no softmax stats, h == h_kv, d == dv == 128, e4m3, fixed-stride BSHD
// with S % 16 == 0 (S % 64 == 0 when B > 1). The launcher checks all of it.

#include <fused_multihead_attention.h>
#include <fused_multihead_flash_attention_kernel_noloop_tiled.h>

#include "flux3FmhaNvfp4Out.cuh"
#include "flux3FmhaSoftmax.cuh"

#include <cuda.h>
#include <stdint.h>
#include <type_traits>

namespace fused_multihead_attention
{
namespace flux3
{

// 3 groups of the 4-warp tile per CTA.
static constexpr int GROUPS = 3;
static constexpr int THREADS = GROUPS * 128;
// The K/V ring depth. 4 stages fill the 1 CTA/SM shared memory budget with the 3 O tiles (98 368 B in all).
static constexpr int STAGES = 4;
static constexpr int K_CHUNK_BYTES = 4096;
static constexpr int K_BYTES = 8192;
static constexpr int V_BYTES = 8192;
static constexpr int STAGE_BYTES = K_BYTES + V_BYTES;
static constexpr int RING_BYTES = STAGES * STAGE_BYTES;

// Shared memory: the ring, then the 3 groups' Q tiles; the 3 groups' O tiles overlay both after the KV loop.
template <typename Kernel_traits>
struct Layout
{
    static constexpr int Q_BYTES = Kernel_traits::Smem_tile_q::BYTES_PER_TILE;
    static constexpr int O_BYTES = Kernel_traits::Smem_tile_o::BYTES_PER_TILE;
    static constexpr int Q_OFFSET = RING_BYTES;
    static constexpr int DATA_BYTES
        = (RING_BYTES + GROUPS * Q_BYTES) > (GROUPS * O_BYTES) ? (RING_BYTES + GROUPS * Q_BYTES) : (GROUPS* O_BYTES);
    static constexpr int BAR_OFFSET = (DATA_BYTES + 127) / 128 * 128;
    static constexpr int SMEM_BYTES = BAR_OFFSET + 2 * STAGES * 8;
};

// The block info: Single_cta<2> of the fmha_v2 kernels, plus the fixed-shape case (null cu_seqlens: every sequence
// has params.s tokens), which is the only one the launcher produces (is_s_padded, no cu_seqlens). The runtime checks
// are kept on purpose: together with the has_alibi test and mask.load in device_flash_attention_g3 they keep nvcc's
// KV-loop schedule; the constant-folded form (params.s everywhere, no alibi test) schedules the loop differently and
// measured 0.8% slower at S = 16464 and 32848.
struct Block_info
{
    template <typename Params>
    inline __device__ Block_info(Params const& params, int bidb_, int bidh_)
        : bidb(bidb_)
        , bidh(bidh_)
        , bidn(0)
        , num_heads(params.h)
    {
        if (params.is_s_padded)
        {
            sum_s = params.s * bidb;
            sum_s_kv = params.s * bidb;
        }
        else
        {
            sum_s = params.cu_q_seqlens[bidb];
            sum_s_kv = params.cu_kv_seqlens[bidb];
        }
        if (params.cu_q_seqlens == nullptr)
        {
            actual_q_seqlen = params.s;
            actual_kv_seqlen = params.s;
        }
        else
        {
            actual_q_seqlen = params.cu_q_seqlens[bidb + 1] - params.cu_q_seqlens[bidb];
            actual_kv_seqlen = params.cu_kv_seqlens ? (params.cu_kv_seqlens[bidb + 1] - params.cu_kv_seqlens[bidb])
                                                    : actual_q_seqlen;
        }
        actual_seqlen = actual_kv_seqlen;
        sum_mask_row = params.cu_mask_rows ? params.cu_mask_rows[bidb] : sum_s;
        bidx = sum_s * params.h + bidh;
    }

    // The length of the sequence.
    int actual_q_seqlen;
    int actual_kv_seqlen;
    // The same as actual_kv_seqlen.
    int actual_seqlen;
    // The total number of mask rows.
    int sum_mask_row;
    // The indices of the block (batch, head, linear index).
    int bidb, bidh, bidn, bidx;
    // The total number of q tokens.
    int sum_s;
    // The total number of kv tokens.
    int sum_s_kv;
    int num_heads;
};

inline __device__ uint32_t smem_u32(void const* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

inline __device__ void mbar_init(uint32_t bar, uint32_t count)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar), "r"(count) : "memory");
}

inline __device__ void mbar_expect_tx(uint32_t bar, uint32_t bytes)
{
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar), "r"(bytes) : "memory");
}

inline __device__ void mbar_arrive(uint32_t bar)
{
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(bar) : "memory");
}

inline __device__ bool mbar_try_wait(uint32_t bar, uint32_t phase)
{
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

inline __device__ bool mbar_test_wait(uint32_t bar, uint32_t phase)
{
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

inline __device__ void mbar_wait(uint32_t bar, uint32_t phase)
{
    while (!mbar_try_wait(bar, phase))
    {
    }
}

inline __device__ void tma_load_4d(uint32_t dst, CUtensorMap const* map, uint32_t bar, int c0, int c1, int c2, int c3)
{
    asm volatile(
        // CTA-local destination: the .shared::cluster form makes the compiler add a per-copy "is this my CTA" check
        // with a CALL slow path (8 B of stack) in a non-cluster launch.
        "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3, %4, %5}], [%6];\n" ::"r"(dst),
        "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(bar)
        : "memory");
}

inline __device__ void tma_load_5d(
    uint32_t dst, CUtensorMap const* map, uint32_t bar, int c0, int c1, int c2, int c3, int c4)
{
    asm volatile(
        "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3, %4, %5, %6}], [%7];\n" ::"r"(dst),
        "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4), "r"(bar)
        : "memory");
}

inline __device__ void named_bar_sync(int id, int threads)
{
    asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(threads) : "memory");
}

// Issue the TMA loads of KV step j (64 keys) into ring stage j % STAGES. tm_v is the 5D swap23 map (make_v_map), whose
// box writes key k of the step to smem row swap23(k); v_g0 = bidb * (s / 16) is the batch's first 16-key group.
inline __device__ void issue_stage(int j, uint32_t ring, uint32_t full0, CUtensorMap const* tm_k,
    CUtensorMap const* tm_v, int bidh, int bidb, int v_g0)
{
    int const slot = j % STAGES;
    uint32_t const bar = full0 + slot * 8;
    uint32_t const dst = ring + slot * STAGE_BYTES;
    mbar_expect_tx(bar, STAGE_BYTES);
    tma_load_4d(dst, tm_k, bar, 0, bidh, j * 64, bidb);
    tma_load_4d(dst + K_CHUNK_BYTES, tm_k, bar, 64, bidh, j * 64, bidb);
    tma_load_5d(dst + K_BYTES, tm_v, bar, bidh * 128, 0, 0, 0, v_g0 + j * 4);
}

// The V B-fragments: one ldmatrix.m16n16.x2.trans.b8 per (ki, 16-d chunk) instead of LDSM.MT88.4 (b16 transpose) +
// 4 PRMT. The V stage is [64 keys][128 d] e4m3, 128-B rows, SWIZZLE_128B (16-B chunk c of row r at chunk c ^ (r & 7)).
// Lanes 0-15 / 16-31 give the row addresses of matrix 0 / 1 (keys 0-15 / 16-31 of the k32 step). Lane i addresses key
// perm(i), perm(4q + j) = 2q + (j & 1) + 8 (j >> 1) (+16 for matrix 1): thread (g = lane/4, q = lane%4) then gets, in
// k-slots 4q..4q+3, the keys {2q, 2q+1, 2q+8, 2q+9} -- the k-order of P's A-fragment. With the swap23 map the stage
// holds key k at smem row swap23(k), so lanes 0-7 (keys {0,1,8,9,2,3,10,11}) address rows {0,1,4,5,2,3,6,7}: 8 distinct
// 16-B chunks under SWIZZLE_128B, one wavefront per 8 rows.
//
// Head dim: r0 / r1 hold d = base + g / base + 8 + g (the LDSM path: base + 2g / base + 2g + 1), so acc_o's n-col -> d
// map differs; unpermute_acc_o() restores the LDSM path's layout once per tile.
inline __device__ uint32_t v_lane_offset(int lane)
{
    int const key = 16 * (lane >> 4) + 2 * ((lane >> 2) & 3) + (lane & 1) + 8 * ((lane >> 1) & 1);
    int const row = (key & ~12) | ((key & 4) << 1) | ((key & 8) >> 1);
    return (uint32_t) (row * 128 + ((row & 7) << 4));
}

template <typename Fragment>
inline __device__ void ldsm_v(Fragment& b, uint32_t addr)
{
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

// The V fragments of step ki for every 16-d chunk ni; v_stage = the stage's V base (1024-B aligned).
template <typename Fragment, int N>
inline __device__ void load_v(Fragment (&b)[N], uint32_t v_stage, uint32_t lane_off, int ki)
{
#pragma unroll
    for (int ni = 0; ni < N; ++ni)
    {
        ldsm_v(b[ni], v_stage + (uint32_t) (ki * 32 * 128) + (lane_off ^ (uint32_t) (ni << 4)));
    }
}

// Quad-local move of acc_o from the ldmatrix.b8 d-map to the LDSM path's. Per 16-d chunk and row half h, thread q holds
// pairs L = d {2q, 2q+1} (regs 2h, 2h+1) and H = d {8+2q, 9+2q} (regs 4+2h, 5+2h); the LDSM path has thread q hold
// d 4q (reg 2h), 4q+1 (reg 4+2h), 4q+2 (reg 2h+1), 4q+3 (reg 5+2h), i.e. pair indices p = 2q, 2q+1 where thread p % 4
// holds pair p. Two shuffle rounds of a pair:
//   round A: send (q odd ? L : H), read lane 2(q&1) + (q < 2)
//   round B: send (q odd ? H : L), read lane 2(q&1) + (q >> 1)
//   pair 2q = (q < 2 ? B : A), pair 2q+1 = (q < 2 ? A : B).
template <typename Acc, int M, int N>
inline __device__ void unpermute_acc_o(Acc (&acc)[M][N], int lane)
{
    int const q = lane & 3;
    bool const odd = (q & 1) != 0;
    bool const low = q < 2;
    int const quad = lane & ~3;
    int const src_a = quad + 2 * (q & 1) + (low ? 1 : 0);
    int const src_b = quad + 2 * (q & 1) + (q >> 1);
#pragma unroll
    for (int mi = 0; mi < M; ++mi)
    {
#pragma unroll
        for (int ni = 0; ni < N; ++ni)
        {
#pragma unroll
            for (int h = 0; h < 2; ++h)
            {
                uint32_t const lx = acc[mi][ni].reg(2 * h), ly = acc[mi][ni].reg(2 * h + 1);
                uint32_t const hx = acc[mi][ni].reg(4 + 2 * h), hy = acc[mi][ni].reg(5 + 2 * h);
                uint32_t const ax = __shfl_sync(0xffffffffu, odd ? lx : hx, src_a);
                uint32_t const ay = __shfl_sync(0xffffffffu, odd ? ly : hy, src_a);
                uint32_t const bx = __shfl_sync(0xffffffffu, odd ? hx : lx, src_b);
                uint32_t const by = __shfl_sync(0xffffffffu, odd ? hy : ly, src_b);
                acc[mi][ni].reg(2 * h) = low ? bx : ax;     // d 4q
                acc[mi][ni].reg(4 + 2 * h) = low ? by : ay; // d 4q+1
                acc[mi][ni].reg(2 * h + 1) = low ? ax : bx; // d 4q+2
                acc[mi][ni].reg(5 + 2 * h) = low ? ay : by; // d 4q+3
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// NVFP4_OUT: the group's O tile is written as NVFP4 (flux3FmhaNvfp4Out.cuh, nfo) instead of BF16 into params.o_ptr.
template <typename Kernel_traits, bool NVFP4_OUT, typename Params>
inline __device__ void device_flash_attention_g3(
    Params const& params, CUtensorMap const* tm_k, CUtensorMap const* tm_v, Nvfp4Out const& nfo)
{
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

    constexpr bool USE_LDGSTS = Kernel_traits::USE_LDGSTS_Q;
    static_assert(USE_LDGSTS, "");
    static_assert(std::is_same<Traits_p, fmha::Ada_qmma_e4m3_fp32_traits>::value, "e4m3 / fp32 accumulators");
    static_assert(Kernel_traits::THREADS == 128, "G3 runs three copies of the 4-warp tile");
    static_assert(Cta_tile_p::M == 64 && Cta_tile_p::N == 64, "");
    static_assert((int) Kernel_traits::CTA_P_TILE_K == 64 && (int) Kernel_traits::VALID_D == 128, "");
    static_assert((int) Kernel_traits::BUFFERS_PER_TILE_SMEM_Q == 2, "");
    static_assert((int) Smem_tile_k::BYTES_PER_BUFFER == K_CHUNK_BYTES, "");
    static_assert((int) Smem_tile_v::BYTES_PER_BUFFER == V_BYTES, "");
    static_assert((int) Smem_tile_k::ROWS_PER_XOR_PATTERN == 4, "K layout must be SWIZZLE_64B");
    static_assert((int) Smem_tile_v::ROWS_PER_XOR_PATTERN == 8, "V layout must be SWIZZLE_128B");
    static_assert((int) Smem_tile_k::BYTES_PER_ROW == 128 && (int) Smem_tile_v::BYTES_PER_ROW == 128, "");
    static_assert((int) Kernel_traits::TOTAL_BMM2_MMAS_K == (int) Mma_tile_o::MMAS_K, "one V tile per KV step");
    static_assert(Kernel_traits::MASK_VERSION == 2 && !Kernel_traits::IS_MTP, "padding mask only");
    static_assert(!Kernel_traits::LIMIT_QK_FRAGMENTS && !Kernel_traits::SHARE_SMEM_FOR_K_AND_V, "");
    static_assert(Gmem_tile_o::LOOPS >= 1, "");
    static_assert((int) Mma_tile_o::VALID_MMAS_N == 8 && (int) Mma_tile_o::MMAS_K == 2, "");

    constexpr int BMM1_VALID_MMAS_K = Mma_tile_p::VALID_MMAS_K;
    constexpr int BMM1_TAIL_MMAS_K_BOUND
        = BMM1_VALID_MMAS_K % Mma_tile_p::MMAS_K ? BMM1_VALID_MMAS_K % Mma_tile_p::MMAS_K : Mma_tile_p::MMAS_K;
    constexpr int BMM1_MAIN_MMAS_K_BOUND = BMM1_VALID_MMAS_K - BMM1_TAIL_MMAS_K_BOUND;
    static_assert(BMM1_MAIN_MMAS_K_BOUND == Mma_tile_p::MMAS_K, "two 64-col K chunks");
    constexpr int BMM2_TAIL_MMAS_K_BOUND = Mma_tile_o::MMAS_K;

    constexpr bool CHECK_NEG_INF = Kernel_traits::SLIDING_WINDOW_ATTENTION || Kernel_traits::CUSTOM_MASK;
    using Softmax = flux3::Softmax<Cta_tile_p, Kernel_traits>;
    static_assert(!Softmax::USE_SHARED_MEMORY, "");

    extern __shared__ __align__(1024) char smem_[];

    int const tid = threadIdx.x;
    int const group = tid / 128;
    int const tidx = tid % 128; // the thread index inside the 4-warp tile
    int const lane = tid % 32;
    int const bidb = blockIdx.z;
    int const bidh = blockIdx.y;
    int const q_loop = blockIdx.x * GROUPS + group;

    Block_info const binfo(params, bidb, bidh);
    int const q_tiles = (binfo.actual_q_seqlen + Gmem_tile_q::ROWS - 1) / Gmem_tile_q::ROWS;
    int active_groups = q_tiles - (int) blockIdx.x * GROUPS;
    active_groups = active_groups < GROUPS ? active_groups : GROUPS;
    bool const active = group < active_groups;
    int const nsteps = (binfo.actual_kv_seqlen + Cta_tile_p::N - 1) / Cta_tile_p::N;

    uint32_t const ring = smem_u32(smem_);
    uint32_t const full0 = smem_u32(smem_ + L::BAR_OFFSET);
    uint32_t const empty0 = full0 + STAGES * 8;
    int const v_g0 = bidb * (int) (params.s / 16);

    // Barrier init + TMA prologue (one thread).
    int issued = 0;
    if (tid == 0)
    {
        if (ring & 1023u)
        {
            __trap(); // SWIZZLE_128B/64B stages need a 1024-B aligned ring base
        }
#pragma unroll
        for (int s = 0; s < STAGES; ++s)
        {
            mbar_init(full0 + s * 8, 1);
            mbar_init(empty0 + s * 8, 4 * active_groups);
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
        asm volatile("prefetch.tensormap [%0];\n" ::"l"(reinterpret_cast<uint64_t>(tm_k)) : "memory");
        asm volatile("prefetch.tensormap [%0];\n" ::"l"(reinterpret_cast<uint64_t>(tm_v)) : "memory");
    }
    __syncthreads();
    if (tid == 0)
    {
        int const n0 = nsteps < STAGES ? nsteps : STAGES;
        for (; issued < n0; ++issued)
        {
            issue_stage(issued, ring, full0, tm_k, tm_v, bidh, bidb, v_g0);
        }
    }

    char* const smem_q_ptr = smem_ + L::Q_OFFSET + group * L::Q_BYTES;

    fmha::Mask_dispatcher<Traits_p, Cta_tile_p, Kernel_traits::MASK_VERSION, Kernel_traits::IS_MTP> mask(
        params, binfo, tidx);

    Gmem_tile_q gmem_q(params, 0, binfo, tidx, q_loop * Gmem_tile_q::ROWS);
    Smem_tile_q smem_q(smem_q_ptr, tidx);
    Smem_tile_k smem_k(smem_, tidx);
    uint32_t const v_ring = smem_u32(smem_ + K_BYTES);
    uint32_t const v_lane_off = v_lane_offset(lane);

    // Only the last, partial KV step needs the padding mask.
    int const kv_mask_loop_start = int(binfo.actual_seqlen / Cta_tile_p::N) * Cta_tile_p::N;

    typename Smem_tile_k::Fragment frag_k[Mma_tile_p::MMAS_K][Mma_tile_p::MMAS_N];
    typename Smem_tile_v::Fragment frag_v[Mma_tile_o::MMAS_K][Mma_tile_o::VALID_MMAS_N];

    Softmax softmax(params, smem_q_ptr, bidb, tidx);

    fmha::Fragment_accumulator<Traits_o> acc_o[Mma_tile_o::MMAS_M][Mma_tile_o::VALID_MMAS_N];
    using Acc_type_o = typename Traits_o::Accumulator_type;
    fmha::Clear_accumulator<Acc_type_o, Cta_tile_o::WARPS_K>::apply(acc_o);

    flux3::Tile_o_normalizer<Cta_tile_o> acc_o_normalizer(params, binfo);
    float global_max[Softmax::ROWS_PER_THREAD];
    float global_sum[Softmax::ROWS_PER_THREAD];

    if (active)
    {
        // The local sequence offset of Q (non-causal, no past KV: q_loop * 64). A no-op for the padding mask; kept for
        // the KV loop code (see Block_info).
        mask.load(q_loop * Gmem_tile_q::ROWS + binfo.actual_kv_seqlen - binfo.actual_q_seqlen);
        // Both 64-column chunks of the Q tile, once.
        gmem_q.load(smem_q);
        gmem_q.move_col();
        smem_q.move_to_next_write_buffer();
        gmem_q.load(smem_q);
        fmha::ldgdepbar<USE_LDGSTS>();
        fmha::depbar_<USE_LDGSTS, 0>();
        named_bar_sync(1 + group, 128);

        // The Q A-fragments of both chunks, read from smem once and kept in registers.
        typename Smem_tile_q::Fragment frag_q[2][Mma_tile_p::MMAS_K][Mma_tile_p::MMAS_M];
#pragma unroll
        for (int c = 0; c < 2; ++c)
        {
#pragma unroll
            for (int ki = 0; ki < Mma_tile_p::MMAS_K; ++ki)
            {
                smem_q.load(frag_q[c][ki], ki);
            }
            smem_q.move_to_next_read_buffer();
        }

        for (int step = 0; step < nsteps; ++step)
        {
            int const kv_loop = step * Cta_tile_p::N;
            int const slot = step % STAGES;
            uint32_t const phase = (step / STAGES) & 1;

            // Producer: keep the ring full. The load of this step must be issued before we wait.
            if (tid == 0)
            {
                while (issued < nsteps && issued < step + STAGES)
                {
                    int const j = issued;
                    uint32_t const ebar = empty0 + (j % STAGES) * 8;
                    uint32_t const eph = ((j / STAGES) - 1) & 1;
                    if (j <= step)
                    {
                        mbar_wait(ebar, eph);
                    }
                    else if (!mbar_test_wait(ebar, eph))
                    {
                        break;
                    }
                    issue_stage(j, ring, full0, tm_k, tm_v, bidh, bidb, v_g0);
                    ++issued;
                }
            }

            bool const first_step = (step == 0);
            // has_alibi is always false (the launcher sets it); the test stays for the KV loop code (see Block_info).
            bool const apply_mask = params.has_alibi || (kv_loop >= kv_mask_loop_start);

            fmha::Fragment_accumulator<Traits_p> acc_p[Mma_tile_p::MMAS_M][Mma_tile_p::MMAS_N];
            using Acc_type_p = typename Traits_p::Accumulator_type;
            fmha::Clear_accumulator<Acc_type_p, Cta_tile_p::WARPS_K>::apply(acc_p);

            mask.move_to_offset(kv_loop);

            // Wait for this step's K/V stage.
            mbar_wait(full0 + slot * 8, phase);

            // BMM1, K columns 0..63 (stage chunk 0).
            smem_k.smem_read_buffer_ = slot * STAGE_BYTES;
#pragma unroll
            for (int ki = 0; ki < Mma_tile_p::MMAS_K; ++ki)
            {
                smem_k.load(frag_k[ki], ki);
                fmha::gemm(acc_p, frag_q[0][ki], frag_k[ki]);
            }

            // BMM1, K columns 64..127 (stage chunk 1).
            smem_k.smem_read_buffer_ = slot * STAGE_BYTES + K_CHUNK_BYTES;
#pragma unroll
            for (int ki = 0; ki < Mma_tile_p::MMAS_K; ++ki)
            {
                smem_k.load(frag_k[ki], ki);
                if (Cta_tile_p::VALID_K % Cta_tile_p::K == 0 || ki < BMM1_TAIL_MMAS_K_BOUND)
                {
                    fmha::gemm(acc_p, frag_q[1][ki], frag_k[ki]);
                }
            }

            softmax.unpack(acc_p);

            if (apply_mask)
            {
                if (params.has_alibi)
                {
                    softmax.apply_mask_alibi(mask, bidh, params.alibi_params);
                }
                else
                {
                    softmax.apply_mask(mask);
                }
            }

            if (first_step)
            {
                softmax.template reduce<fmha::Max_>(global_max);
                softmax.template apply_exp_with_mask<CHECK_NEG_INF>(global_max);
                softmax.template reduce<fmha::Sum_>(global_sum);
            }
            else
            {
                float tmp[Softmax::ROWS_PER_THREAD];
#pragma unroll
                for (int i = 0; i < Softmax::ROWS_PER_THREAD; i++)
                {
                    tmp[i] = global_max[i];
                }
                softmax.template reduce<fmha::Max_>(global_max);
                acc_o_normalizer.update(acc_o, global_max, tmp, global_sum);
                softmax.template apply_exp_with_mask<CHECK_NEG_INF>(global_max);
#pragma unroll
                for (int i = 0; i < Softmax::ROWS_PER_THREAD; i++)
                {
                    tmp[i] = global_sum[i];
                    global_sum[i] = 0.f;
                }
                softmax.template reduce<fmha::Sum_>(global_sum);
#pragma unroll
                for (int i = 0; i < Softmax::ROWS_PER_THREAD; i++)
                {
                    global_sum[i] += tmp[i];
                }
            }

            fmha::Fragment_a<Traits_p, fmha::Row> frag_p[Kernel_traits::TOTAL_BMM2_MMAS_K][Mma_tile_o::MMAS_M];
            softmax.pack(frag_p);

            // BMM2 on the stage's V tile.
            uint32_t const v_stage = v_ring + slot * STAGE_BYTES;
#pragma unroll
            for (int ki = 0; ki < BMM2_TAIL_MMAS_K_BOUND; ++ki)
            {
                load_v(frag_v[ki], v_stage, v_lane_off, ki);
                fmha::gemm(acc_o, frag_p[ki], frag_v[ki]);
            }

            // Release the stage: every warp of every active group arrives once.
            __syncwarp();
            if (lane == 0)
            {
                mbar_arrive(empty0 + slot * 8);
            }
        }

        acc_o_normalizer.update_sum(global_max, global_sum);
        acc_o_normalizer.final_update(acc_o, global_sum);
        // Back to the LDSM path's acc_o d-map before the epilogue.
        unpermute_acc_o(acc_o, lane);
    }

    // Every group is done with the ring and the Q tiles; all issued TMA loads were consumed.
    __syncthreads();

    if (active)
    {
        Gmem_tile_o gmem_o(params, binfo, tidx, q_loop * Gmem_tile_o::ROWS, 0);
        Smem_tile_o smem_o(smem_ + group * L::O_BYTES, tidx);
        // Per group: the 4-warp kernel's store loop with its __syncthreads() replaced by the group's named barrier.
        // All indices are group-local (tidx 0..127, smem_o at the group's O region, gmem_o at the group's q_loop).
        // The NVFP4-out kernel keeps the BF16 store behind a runtime check of nfo.fp4 (the launcher always sets it):
        // with both epilogues present nvcc keeps the KV-loop schedule; the NVFP4-only kernel schedules the loop
        // differently and measured 0.3-0.4% slower.
        if (NVFP4_OUT && nfo.fp4 != nullptr)
        {
#pragma unroll
            for (int ii = 0; ii < Gmem_tile_o::LOOPS; ++ii)
            {
                smem_o.store(acc_o, ii);
                named_bar_sync(1 + group, 128);
                nvfp4_out::store_from_smem(params, nfo, gmem_o, smem_o, ii, bidb, bidh, tidx);
                if (ii < Gmem_tile_o::LOOPS - 1)
                {
                    named_bar_sync(1 + group, 128);
                }
            }
            nvfp4_out::zero_padding(params, nfo, Gmem_tile_o::ROWS, q_loop, bidb, bidh, tidx, 128);
        }
        else
        {
#pragma unroll
            for (int ii = 0; ii < Gmem_tile_o::LOOPS; ++ii)
            {
                smem_o.store(acc_o, ii);
                named_bar_sync(1 + group, 128);
                uint4 out[Gmem_tile_o::STGS_PER_LOOP];
                smem_o.load(out);
                if (ii < Gmem_tile_o::LOOPS - 1)
                {
                    named_bar_sync(1 + group, 128);
                }
                gmem_o.store(out, ii);
            }
        }
    }
}

} // namespace flux3
} // namespace fused_multihead_attention
