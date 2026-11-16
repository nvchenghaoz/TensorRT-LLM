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

// Sequential-K BF16 GEMM for SM120: Y[M, N] = X[M, K] @ W[N, K]^T, both operands K-major.
//
// Numerics: every output's FP32 accumulator walks the k16 steps of mma.sync.m16n8k16 in ascending
// K order and is rounded once to BF16 (RN). That is bitwise equal to cuBLASLt's non-split-K
// nvjet_sm120 / cutlass_80 s16816 BF16 kernels, whose bits depend only on that K order, not on the
// CTA tile, BLOCK_K or operand orientation.
//
// Structure: one CTA per (token tile, weight tile), TM tokens x TN weights; 4*WPS warps, WPS per SM
// sub-partition. SMSP s = warp & 3 owns a (TM/2) x (TN/2) quadrant (MT m16 token tiles x NTS n8
// weight tiles) and its WPS warps split the quadrant's n8 tiles, so every SMSP issues the same MMA
// count per k16 step. Operand k-tiles arrive by TMA (64B or 128B swizzle) into an ST-deep ring
// guarded by full/empty mbarriers; thread 0 refills a stage as soon as every warp has released it.
// Fragments are double-buffered across k16 steps and across stages. The epilogue stages the BF16
// tile through shared memory and writes 16-byte coalesced rows.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace sm120_seqk_detail
{

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define SM120_SEQK_DEVICE_OK 1
#else
#define SM120_SEQK_DEVICE_OK 0
#endif

// Device helpers exist only for SM90+ device passes (TMA, mbarrier tx counts); other passes of a
// multi-arch build compile a trapping kernel stub.
#if SM120_SEQK_DEVICE_OK
__device__ __forceinline__ uint32_t smem_u32(void const* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void ldsm_x2(uint32_t addr, uint32_t& r0, uint32_t& r1)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(addr));
}

__device__ __forceinline__ void mma_bf16(float* c, uint32_t const* a, uint32_t const* b)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

#if SM120_SEQK_DEVICE_OK
__device__ __forceinline__ void mbar_init(uint32_t a, uint32_t cnt)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(a), "r"(cnt));
}

__device__ __forceinline__ void mbar_expect_tx(uint32_t a, uint32_t bytes)
{
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(a), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbar_arrive(uint32_t a)
{
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(a) : "memory");
}

__device__ __forceinline__ void mbar_wait(uint32_t a, uint32_t phase)
{
    uint32_t done;
    do
    {
        asm volatile(
            "{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
            " selp.u32 %0, 1, 0, p;\n}\n"
            : "=r"(done)
            : "r"(a), "r"(phase)
            : "memory");
    } while (!done);
}

__device__ __forceinline__ void tma_load_2d(uint32_t dst, CUtensorMap const* map, uint32_t mbar, int c0, int c1)
{
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], "
        "[%2];\n" ::"r"(dst),
        "l"(reinterpret_cast<uint64_t>(map)), "r"(mbar), "r"(c0), "r"(c1)
        : "memory");
}
#endif

// Byte offset of 16-byte chunk c of row r in a [rows][BK] BF16 tile, matching TMA's 128B (BK=64) or
// 64B (BK=32) swizzle relative to a 1 KB-aligned tile base.
template <int BK>
__device__ __forceinline__ uint32_t swz(int r, int c)
{
    if constexpr (BK == 64)
    {
        return static_cast<uint32_t>((r * 8 + (c ^ (r & 7))) * 16);
    }
    else
    {
        static_assert(BK == 32, "BK must be 32 or 64");
        return static_cast<uint32_t>((r * 4 + (c ^ ((r >> 1) & 3))) * 16);
    }
}
#endif // SM120_SEQK_DEVICE_OK

template <int MT_, int NTS_, int BK_, int ST_, int WPS_>
struct SeqKCfg
{
    static constexpr int MT = MT_;
    static constexpr int NTS = NTS_;
    static constexpr int BK = BK_;
    static constexpr int ST = ST_;
    static constexpr int WPS = WPS_;
    static constexpr int kWarps = 4 * WPS;
    static constexpr int kThreads = 32 * kWarps;
    static constexpr int TM = 2 * MT * 16;
    static constexpr int TN = 2 * NTS * 8;
    static constexpr int kStageA = TM * BK * 2;
    static constexpr int kStageB = TN * BK * 2;
    static constexpr int kStage = kStageA + kStageB;
    static constexpr int kMain = ST * kStage;
    static constexpr int kEpiPitch = TN + 8; // elements: 16B-aligned rows, conflict-free fragment stores
    static constexpr int kEpi = TM * kEpiPitch * 2;
    static constexpr int kBarOff = ((kMain > kEpi ? kMain : kEpi) + 127) / 128 * 128;
    static constexpr int kSmem = kBarOff + 2 * ST * 8 + 1024; // + slack to align the base to 1 KB

    static constexpr int nw(int j)
    {
        return NTS / WPS + (j < NTS % WPS ? 1 : 0);
    }

    static constexpr int off(int j)
    {
        return j == 0 ? 0 : off(j - 1) + nw(j - 1);
    }

    static_assert(kStageA % 1024 == 0 && kStageB % 1024 == 0, "swizzle atoms need 1 KB-aligned tiles");
    static_assert((BK / 16) % 2 == 0, "fragment double buffer needs an even k16 count per stage");
    static_assert(TM <= 256 && TN <= 256, "TMA box rows must not exceed 256");
};

#if SM120_SEQK_DEVICE_OK
template <int MT, int NW, int BK>
__device__ __forceinline__ void load_frags(uint32_t sA, uint32_t sB, int kk, int wm0, int wn0, int lane,
    uint32_t (&a)[MT][4], uint32_t (&b)[NW][2])
{
    int const a_row = lane & 15;
    int const a_chk = lane >> 4;
    int const b_row = (lane & 7) + ((lane >> 4) << 3);
    int const b_chk = (lane >> 3) & 1;
#pragma unroll
    for (int mi = 0; mi < MT; ++mi)
    {
        int r = wm0 + mi * 16 + a_row;
        ldsm_x4(sA + swz<BK>(r, kk * 2 + a_chk), a[mi][0], a[mi][1], a[mi][2], a[mi][3]);
    }
#pragma unroll
    for (int nj = 0; nj < NW / 2; ++nj)
    {
        int r = wn0 + nj * 16 + b_row;
        ldsm_x4(sB + swz<BK>(r, kk * 2 + b_chk), b[2 * nj][0], b[2 * nj][1], b[2 * nj + 1][0], b[2 * nj + 1][1]);
    }
    if constexpr (NW % 2 == 1)
    {
        int r = wn0 + (NW - 1) * 8 + (lane & 7);
        ldsm_x2(sB + swz<BK>(r, kk * 2 + b_chk), b[NW - 1][0], b[NW - 1][1]);
    }
}

template <class C, int NW>
__device__ __forceinline__ void seqk_consumer(char* smem, uint32_t sbase, uint32_t bars, CUtensorMap const* tmX,
    CUtensorMap const* tmW, __nv_bfloat16* __restrict__ Y, int M, int N, int K, int64_t ldy, int m0, int n0, int wm0,
    int wn0, int tid, int lane)
{
#if SM120_SEQK_DEVICE_OK
    constexpr int MT = C::MT;
    constexpr int BK = C::BK;
    constexpr int ST = C::ST;
    constexpr int KS = BK / 16;
    float acc[MT][NW][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NW; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e)
                acc[i][j][e] = 0.f;
    uint32_t a[2][MT][4];
    uint32_t b[2][NW][2];
    int const ktiles = K / BK;
    mbar_wait(bars, 0);
    load_frags<MT, NW, BK>(sbase, sbase + C::kStageA, 0, wm0, wn0, lane, a[0], b[0]);
    for (int kt = 0; kt < ktiles; ++kt)
    {
        int const s = kt % ST;
        uint32_t const sA = sbase + s * C::kStage;
        // Thread 0 refills the stage every warp released after k-tile kt-1 with k-tile kt-1+ST.
        if (tid == 0 && kt >= 1 && kt - 1 + ST < ktiles)
        {
            int const sp = (kt - 1) % ST;
            mbar_wait(bars + (ST + sp) * 8, ((kt - 1) / ST) & 1);
            uint32_t const full = bars + sp * 8;
            mbar_expect_tx(full, C::kStage);
            uint32_t const dst = sbase + sp * C::kStage;
            tma_load_2d(dst, tmX, full, (kt - 1 + ST) * BK, m0);
            tma_load_2d(dst + C::kStageA, tmW, full, (kt - 1 + ST) * BK, n0);
        }
#pragma unroll
        for (int kk = 0; kk < KS; ++kk)
        {
            if (kk + 1 < KS)
            {
                load_frags<MT, NW, BK>(
                    sA, sA + C::kStageA, kk + 1, wm0, wn0, lane, a[(kk + 1) & 1], b[(kk + 1) & 1]);
            }
            else if (kt + 1 < ktiles)
            {
                int const s2 = (kt + 1) % ST;
                mbar_wait(bars + s2 * 8, ((kt + 1) / ST) & 1);
                uint32_t const sA2 = sbase + s2 * C::kStage;
                load_frags<MT, NW, BK>(sA2, sA2 + C::kStageA, 0, wm0, wn0, lane, a[0], b[0]);
            }
            // k16 steps strictly ascending per accumulator: this order fixes the output bits.
#pragma unroll
            for (int mi = 0; mi < MT; ++mi)
#pragma unroll
                for (int nj = 0; nj < NW; ++nj)
                    mma_bf16(acc[mi][nj], a[kk & 1][mi], b[kk & 1][nj]);
        }
        __syncwarp();
        if (lane == 0)
        {
            mbar_arrive(bars + (ST + s) * 8);
        }
    }

    // Epilogue: one RN cast per output, staged through smem, then 16-byte coalesced stores.
    asm volatile("bar.sync 1, %0;\n" ::"n"(C::kThreads) : "memory");
    __nv_bfloat16* Cs = reinterpret_cast<__nv_bfloat16*>(smem);
    int const qr = lane >> 2;
    int const qc = (lane & 3) * 2;
#pragma unroll
    for (int mi = 0; mi < MT; ++mi)
    {
#pragma unroll
        for (int nj = 0; nj < NW; ++nj)
        {
            int r = wm0 + mi * 16 + qr;
            int c = wn0 + nj * 8 + qc;
            *reinterpret_cast<__nv_bfloat162*>(Cs + r * C::kEpiPitch + c)
                = __floats2bfloat162_rn(acc[mi][nj][0], acc[mi][nj][1]);
            *reinterpret_cast<__nv_bfloat162*>(Cs + (r + 8) * C::kEpiPitch + c)
                = __floats2bfloat162_rn(acc[mi][nj][2], acc[mi][nj][3]);
        }
    }
    asm volatile("bar.sync 1, %0;\n" ::"n"(C::kThreads) : "memory");
    constexpr int kChunks = C::TN / 8;
    for (int i = tid; i < C::TM * kChunks; i += C::kThreads)
    {
        int r = i / kChunks, c = i % kChunks;
        int gr = m0 + r, gc = n0 + c * 8;
        if (gr < M && gc < N)
        {
            *reinterpret_cast<uint4*>(Y + static_cast<int64_t>(gr) * ldy + gc)
                = *reinterpret_cast<uint4 const*>(Cs + r * C::kEpiPitch + c * 8);
        }
    }
#endif
}
#endif // SM120_SEQK_DEVICE_OK

template <class C>
__global__ void __launch_bounds__(C::kThreads, 1) sm120SeqKGemmKernel(const __grid_constant__ CUtensorMap tmX,
    const __grid_constant__ CUtensorMap tmW, __nv_bfloat16* __restrict__ Y, int M, int N, int K, int64_t ldy, int ntm)
{
#if SM120_SEQK_DEVICE_OK
    extern __shared__ __align__(1024) char smem_raw[];
    char* smem = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~static_cast<uintptr_t>(1023));
    uint32_t const sbase = smem_u32(smem);
    uint32_t const bars = sbase + C::kBarOff; // full[ST], then empty[ST]
    int const tid = threadIdx.x;
    int const warp = tid >> 5;
    int const lane = tid & 31;
    // Raster: the token tiles of one weight tile are consecutive CTAs, so its re-reads hit L2.
    int const bid = blockIdx.x;
    int const tt = bid % ntm;
    int const wt = bid / ntm;
    int const m0 = tt * C::TM;
    int const n0 = wt * C::TN;
    if (tid == 0)
    {
#pragma unroll
        for (int s = 0; s < C::ST; ++s)
        {
            mbar_init(bars + s * 8, 1);
            mbar_init(bars + (C::ST + s) * 8, C::kWarps);
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
        int const ktiles = K / C::BK;
#pragma unroll
        for (int s = 0; s < C::ST; ++s)
        {
            if (s < ktiles)
            {
                uint32_t const full = bars + s * 8;
                mbar_expect_tx(full, C::kStage);
                uint32_t const dst = sbase + s * C::kStage;
                tma_load_2d(dst, &tmX, full, s * C::BK, m0);
                tma_load_2d(dst + C::kStageA, &tmW, full, s * C::BK, n0);
            }
        }
    }
    __syncthreads();
    int const s4 = warp & 3;
    int const j = warp >> 2;
    int const wm0 = (s4 & 1) * (C::MT * 16);
    int const wn_base = (s4 >> 1) * (C::NTS * 8);
#define SM120_SEQK_RUN(J)                                                                                              \
    seqk_consumer<C, C::nw(J)>(smem, sbase, bars, &tmX, &tmW, Y, M, N, K, ldy, m0, n0, wm0, wn_base + C::off(J) * 8,    \
        tid, lane)
    if (j == 0)
    {
        SM120_SEQK_RUN(0);
    }
    if constexpr (C::WPS > 1)
    {
        if (j == 1)
        {
            SM120_SEQK_RUN(1);
        }
    }
    if constexpr (C::WPS > 2)
    {
        if (j == 2)
        {
            SM120_SEQK_RUN(2);
        }
    }
#undef SM120_SEQK_RUN
#else
    __trap();
#endif
}

} // namespace sm120_seqk_detail
