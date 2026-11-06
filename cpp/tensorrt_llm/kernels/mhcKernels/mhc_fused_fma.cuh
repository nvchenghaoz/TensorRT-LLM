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

// CUDA-core FMA fused post_mapping + pre_mapping kernels for the mhc
// fused_hc pipeline. Provides the small-M variants:
//   - fused_pmap_gemm_fma_ksplit   (Path E, 2-kernel: this + mhcBigFuseKernel)
//   - fused_pmap_gemm_fma_allinone (Path F, 1-kernel: pmap + GEMM + bigFuse)
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace fused_fma_kernels
{

// ===================================================================
// Fused K-split pmap+GEMM FMA: new_r is NEVER materialized to HBM.  Each CTA
// scans one HIDDEN-slice inline: for each h in its slice it loads residual[j,h]
// for ALL HC j-indices plus x[h], computes FULL new_r[j,h] = pm[j]*x[h] + Σ_k
// comb[k,j]*r[k,h] in registers, then multiplies by W_T and accumulates partial
// acc+sqr.  Writes Yp[k_split, token, n] and Rp[k_split, token].  A reducer
// kernel sums over splits.  Eliminates the 32-KB/token new_r round-trip at tiny M.
//
// CORRECTNESS: We split the HIDDEN dimension (not HC).  For each fixed h,
// new_r[*, h] depends only on x[h] and residual[*, h]; the HC summation is
// fully local to h, so a HIDDEN-split is algebraically exact.  The final GEMM
// acc[n] = Σ_{j,h} new_r[j,h] * W_T[n, j*HIDDEN+h] is then partitioned into
// disjoint h-ranges across splits and reduced by mhcBigFuseKernel.
//
// Grid: (M, N/N_PER_BLOCK, NUM_K_SPLITS).  Block: 256 threads.
// NUM_K_SPLITS must evenly divide HIDDEN (typically ∈ {2, 4, 8}).
// BF16_VEC_OVERRIDE lets the caller force a smaller per-thread vector (e.g. 4)
// so that BLOCK_SIZE*BF16_VEC == HIDDEN_PER_SPLIT and every thread is active.
// Pass 0 to auto-pick: 8 for ks=2, 4 for ks=4, 2 for ks=8.
// ===================================================================
// Optional WRITE_RESIDUAL path: when true, the CTA with (blockIdx.y == 0) emits
// the post-mapped residual (bf16 [M, HC_MULT, hidden]) into residual_out, using
// the same disjoint h-slice it already owns.  When false the residual_out arg
// may be a nullptr; the parameter stays in the signature for simplicity.
template <int N_PER_BLOCK, int NUM_K_SPLITS, int BF16_VEC_OVERRIDE = 0, bool WRITE_RESIDUAL = false>
__launch_bounds__(256) __global__ void fused_pmap_gemm_fma_ksplit(__nv_bfloat16 const* __restrict__ residual_in,
    __nv_bfloat16 const* __restrict__ x_in, float const* __restrict__ post_mix, float const* __restrict__ comb_mix,
    float const* __restrict__ W_T, float* __restrict__ Yp, float* __restrict__ Rp, int hidden_size, int N, int K,
    __nv_bfloat16* __restrict__ residual_out = nullptr)
{
    static_assert(NUM_K_SPLITS == 1 || NUM_K_SPLITS == 2 || NUM_K_SPLITS == 4 || NUM_K_SPLITS == 8,
        "ksplit ∈ {1, 2, 4, 8} supported");
    constexpr int HC_MULT = 4;
    constexpr int BLOCK_SIZE = 256;
    constexpr int WARP_SIZE = 32;
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    // Default BF16_VEC by ks so that BLOCK_SIZE*BF16_VEC = HIDDEN/ks at HIDDEN=4096.
    // ks=1 takes the full HIDDEN in one slice; reuses the BF16_VEC=8 (uint4) load path.
    constexpr int BF16_VEC = (BF16_VEC_OVERRIDE != 0) ? BF16_VEC_OVERRIDE
                                                      : ((NUM_K_SPLITS == 1 || NUM_K_SPLITS == 2) ? 8
                                                              : (NUM_K_SPLITS == 4)               ? 4
                                                                                                  : 2);
    static_assert(BF16_VEC == 2 || BF16_VEC == 4 || BF16_VEC == 8, "BF16_VEC must be 2, 4, or 8");

    int const token = blockIdx.x;
    int const n_start = blockIdx.y * N_PER_BLOCK;
    int const k_split = blockIdx.z;
    int const tid = threadIdx.x;
    int const warp_id = tid / WARP_SIZE;
    int const lane = tid % WARP_SIZE;
    bool const do_sqr = (n_start == 0);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaGridDependencySynchronize();
#endif

    int const hidden_per_split = hidden_size / NUM_K_SPLITS;
    int const h_start = k_split * hidden_per_split;
    int const h_end = h_start + hidden_per_split;

    __shared__ float s_post[HC_MULT];
    __shared__ float s_comb[HC_MULT][HC_MULT];

    if (tid < HC_MULT)
        s_post[tid] = post_mix[token * HC_MULT + tid];
    if (tid < HC_MULT * HC_MULT)
    {
        int const r = tid / HC_MULT;
        int const c = tid % HC_MULT;
        s_comb[r][c] = comb_mix[token * HC_MULT * HC_MULT + r * HC_MULT + c];
    }
    __syncthreads();

    float pm[HC_MULT];
#pragma unroll
    for (int j = 0; j < HC_MULT; j++)
        pm[j] = s_post[j];

    float acc[N_PER_BLOCK];
#pragma unroll
    for (int n = 0; n < N_PER_BLOCK; n++)
        acc[n] = 0.0f;
    float sqr = 0.0f;

    long long const tok_res = static_cast<long long>(token) * HC_MULT * hidden_size;
    long long const tok_x = static_cast<long long>(token) * hidden_size;

    // Iterate h over this CTA's HIDDEN slice.  Each thread owns BF16_VEC
    // contiguous positions per step; stride BLOCK_SIZE*BF16_VEC.
    for (int h = h_start + tid * BF16_VEC; h < h_end; h += BLOCK_SIZE * BF16_VEC)
    {
        // Load x[h..h+BF16_VEC).  Vector width depends on BF16_VEC.
        float xf[BF16_VEC];
        if constexpr (BF16_VEC == 8)
        {
            uint4 x_raw = *reinterpret_cast<uint4 const*>(&x_in[tok_x + h]);
            __nv_bfloat162 const* xp = reinterpret_cast<__nv_bfloat162 const*>(&x_raw);
#pragma unroll
            for (int v = 0; v < 4; v++)
            {
                float2 f = __bfloat1622float2(xp[v]);
                xf[2 * v + 0] = f.x;
                xf[2 * v + 1] = f.y;
            }
        }
        else if constexpr (BF16_VEC == 4)
        {
            uint2 x_raw = *reinterpret_cast<uint2 const*>(&x_in[tok_x + h]);
            __nv_bfloat162 const* xp = reinterpret_cast<__nv_bfloat162 const*>(&x_raw);
#pragma unroll
            for (int v = 0; v < 2; v++)
            {
                float2 f = __bfloat1622float2(xp[v]);
                xf[2 * v + 0] = f.x;
                xf[2 * v + 1] = f.y;
            }
        }
        else
        { // BF16_VEC == 2
            unsigned x_raw = *reinterpret_cast<unsigned const*>(&x_in[tok_x + h]);
            __nv_bfloat162 xp = *reinterpret_cast<__nv_bfloat162 const*>(&x_raw);
            float2 f = __bfloat1622float2(xp);
            xf[0] = f.x;
            xf[1] = f.y;
        }

        // Seed new_r[j, v] = pm[j] * x[v] for all j
        float new_r[HC_MULT][BF16_VEC];
#pragma unroll
        for (int j = 0; j < HC_MULT; j++)
#pragma unroll
            for (int v = 0; v < BF16_VEC; v++)
                new_r[j][v] = pm[j] * xf[v];

// Add Σ_k comb[k,j] * r[k, v] — FULL HC sum (all k ∈ [0, HC_MULT))
#pragma unroll
        for (int k = 0; k < HC_MULT; k++)
        {
            float rf_k[BF16_VEC];
            __nv_bfloat16 const* r_ptr = &residual_in[tok_res + k * hidden_size + h];
            if constexpr (BF16_VEC == 8)
            {
                uint4 r_raw = *reinterpret_cast<uint4 const*>(r_ptr);
                __nv_bfloat162 const* rp = reinterpret_cast<__nv_bfloat162 const*>(&r_raw);
#pragma unroll
                for (int v = 0; v < 4; v++)
                {
                    float2 f = __bfloat1622float2(rp[v]);
                    rf_k[2 * v + 0] = f.x;
                    rf_k[2 * v + 1] = f.y;
                }
            }
            else if constexpr (BF16_VEC == 4)
            {
                uint2 r_raw = *reinterpret_cast<uint2 const*>(r_ptr);
                __nv_bfloat162 const* rp = reinterpret_cast<__nv_bfloat162 const*>(&r_raw);
#pragma unroll
                for (int v = 0; v < 2; v++)
                {
                    float2 f = __bfloat1622float2(rp[v]);
                    rf_k[2 * v + 0] = f.x;
                    rf_k[2 * v + 1] = f.y;
                }
            }
            else
            { // BF16_VEC == 2
                unsigned r_raw = *reinterpret_cast<unsigned const*>(r_ptr);
                __nv_bfloat162 rp = *reinterpret_cast<__nv_bfloat162 const*>(&r_raw);
                float2 f = __bfloat1622float2(rp);
                rf_k[0] = f.x;
                rf_k[1] = f.y;
            }
#pragma unroll
            for (int j = 0; j < HC_MULT; j++)
#pragma unroll
                for (int v = 0; v < BF16_VEC; v++)
                    new_r[j][v] = fmaf(s_comb[k][j], rf_k[v], new_r[j][v]);
        }

        if (do_sqr)
        {
#pragma unroll
            for (int j = 0; j < HC_MULT; j++)
#pragma unroll
                for (int v = 0; v < BF16_VEC; v++)
                    sqr = fmaf(new_r[j][v], new_r[j][v], sqr);
        }

        // Optional: emit the post-mapped residual to HBM (bf16) for downstream
        // bigfuse.  Guard on WRITE_RESIDUAL (compile-time) AND do_sqr so only
        // one n-tile per (token, k_split) writes — other n-tiles compute the
        // same new_r but we don't want duplicate writes.
        if constexpr (WRITE_RESIDUAL)
        {
            if (do_sqr)
            {
#pragma unroll
                for (int j = 0; j < HC_MULT; j++)
                {
                    __nv_bfloat16* o_ptr = residual_out + tok_res + j * hidden_size + h;
                    if constexpr (BF16_VEC == 8)
                    {
                        uint4 packed;
                        __nv_bfloat162* pairs = reinterpret_cast<__nv_bfloat162*>(&packed);
#pragma unroll
                        for (int v = 0; v < 4; v++)
                            pairs[v] = __float22bfloat162_rn(make_float2(new_r[j][2 * v], new_r[j][2 * v + 1]));
                        *reinterpret_cast<uint4*>(o_ptr) = packed;
                    }
                    else if constexpr (BF16_VEC == 4)
                    {
                        uint2 packed;
                        __nv_bfloat162* pairs = reinterpret_cast<__nv_bfloat162*>(&packed);
#pragma unroll
                        for (int v = 0; v < 2; v++)
                            pairs[v] = __float22bfloat162_rn(make_float2(new_r[j][2 * v], new_r[j][2 * v + 1]));
                        *reinterpret_cast<uint2*>(o_ptr) = packed;
                    }
                    else
                    { // BF16_VEC == 2
                        __nv_bfloat162 packed = __float22bfloat162_rn(make_float2(new_r[j][0], new_r[j][1]));
                        *reinterpret_cast<unsigned*>(o_ptr) = *reinterpret_cast<unsigned*>(&packed);
                    }
                }
            }
        }

// GEMM partial: for each n, contract over ALL HC j and this h-slice.
// W_T layout: W_T[n, K=HC_MULT*hidden_size], inner dim packed as (j, h).
#pragma unroll
        for (int n = 0; n < N_PER_BLOCK; n++)
        {
#pragma unroll
            for (int j = 0; j < HC_MULT; j++)
            {
                long long const w_base = static_cast<long long>(n_start + n) * K + j * hidden_size + h;
                float w[BF16_VEC];
                if constexpr (BF16_VEC == 8)
                {
                    float w0, w1, w2, w3, w4, w5, w6, w7;
                    asm volatile("ld.global.L1::evict_last.v4.f32 {%0, %1, %2, %3}, [%4];"
                                 : "=f"(w0), "=f"(w1), "=f"(w2), "=f"(w3)
                                 : "l"(W_T + w_base));
                    asm volatile("ld.global.L1::evict_last.v4.f32 {%0, %1, %2, %3}, [%4];"
                                 : "=f"(w4), "=f"(w5), "=f"(w6), "=f"(w7)
                                 : "l"(W_T + w_base + 4));
                    w[0] = w0;
                    w[1] = w1;
                    w[2] = w2;
                    w[3] = w3;
                    w[4] = w4;
                    w[5] = w5;
                    w[6] = w6;
                    w[7] = w7;
                }
                else if constexpr (BF16_VEC == 4)
                {
                    float w0, w1, w2, w3;
                    asm volatile("ld.global.L1::evict_last.v4.f32 {%0, %1, %2, %3}, [%4];"
                                 : "=f"(w0), "=f"(w1), "=f"(w2), "=f"(w3)
                                 : "l"(W_T + w_base));
                    w[0] = w0;
                    w[1] = w1;
                    w[2] = w2;
                    w[3] = w3;
                }
                else
                { // BF16_VEC == 2
                    float w0, w1;
                    asm volatile("ld.global.L1::evict_last.v2.f32 {%0, %1}, [%2];"
                                 : "=f"(w0), "=f"(w1)
                                 : "l"(W_T + w_base));
                    w[0] = w0;
                    w[1] = w1;
                }
#pragma unroll
                for (int v = 0; v < BF16_VEC; v++)
                    acc[n] = fmaf(new_r[j][v], w[v], acc[n]);
            }
        }
    }

    constexpr int COLS_PER_GROUP = NUM_WARPS;
    constexpr int SQRSUM_SLOT = COLS_PER_GROUP;
    constexpr int N_GROUPS = (N_PER_BLOCK + COLS_PER_GROUP - 1) / COLS_PER_GROUP;
    constexpr unsigned FULL_WARP_MASK = 0xffffffff;

    __shared__ float s_warp[NUM_WARPS][COLS_PER_GROUP + 1];

#pragma unroll
    for (int g = 0; g < N_GROUPS; g++)
    {
        constexpr int LAST_COLS = N_PER_BLOCK - (N_GROUPS - 1) * COLS_PER_GROUP;
        int const n_cols = (g == N_GROUPS - 1) ? LAST_COLS : COLS_PER_GROUP;
#pragma unroll
        for (int n = 0; n < COLS_PER_GROUP; n++)
        {
            if (n < n_cols)
            {
#pragma unroll
                for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
                    acc[g * COLS_PER_GROUP + n] += __shfl_xor_sync(FULL_WARP_MASK, acc[g * COLS_PER_GROUP + n], off);
            }
        }
        if (g == 0 && do_sqr)
        {
#pragma unroll
            for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
                sqr += __shfl_xor_sync(FULL_WARP_MASK, sqr, off);
        }
        if (lane == 0)
        {
#pragma unroll
            for (int n = 0; n < COLS_PER_GROUP; n++)
                if (n < n_cols)
                    s_warp[warp_id][n] = acc[g * COLS_PER_GROUP + n];
            if (g == 0 && do_sqr)
                s_warp[warp_id][SQRSUM_SLOT] = sqr;
        }
        __syncthreads();
        if (lane == 0 && warp_id < n_cols)
        {
            float val = 0.0f;
#pragma unroll
            for (int w = 0; w < NUM_WARPS; w++)
                val += s_warp[w][warp_id];
            // Yp layout: [NUM_K_SPLITS, M, N]  (M = gridDim.x)
            int const n_global = n_start + g * COLS_PER_GROUP + warp_id;
            Yp[(static_cast<long long>(k_split) * static_cast<long long>(gridDim.x) + token) * N + n_global] = val;
        }
        if (g == 0 && do_sqr && lane == 0 && warp_id == 0)
        {
            float sq = 0.0f;
#pragma unroll
            for (int w = 0; w < NUM_WARPS; w++)
                sq += s_warp[w][SQRSUM_SLOT];
            // Rp layout: [NUM_K_SPLITS, M]
            Rp[static_cast<long long>(k_split) * static_cast<long long>(gridDim.x) + token] = sq;
        }
        __syncthreads();
    }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// ===================================================================
// fused_pmap_gemm_fma_allinone<TN, KS, TM, FULL_N>
// -------------------------------------------------------------------
// 1-kernel FMA path: pmap + GEMM + bigFuse, all in a single launch.
// Parallelism: TM tokens per CTA × TN N-columns × KS K-slices.
// Grid:  (ceil(M / TM), FULL_N / TN, KS)
// Block: 256 threads.
//
// Uses atomicAdd into y_acc[M, FULL_N] and r_acc[M] for cross-CTA reduction
// (KS > 1 and/or TN < FULL_N).  Uses per-token-batch atomic counter to elect
// the last-home CTA, which __threadfences, reloads the now-complete y_acc /
// r_acc / residual_cur rows, and runs bigFuse inline to produce
// post_mix_out, comb_mix_out, layer_input_out.
//
// Caller MUST zero y_acc[M, FULL_N], r_acc[M], done_counter[ceil(M / TM)]
// before launch.  FULL_N must equal HC_MULT * (2 + HC_MULT) = 24.
// ===================================================================
template <int TN, int KS, int TM = 1, int FULL_N = 24, int BF16_VEC_OVERRIDE = 0, bool kFuseNorm = false,
    bool kLaggedPre = false>
__launch_bounds__(256) __global__ void fused_pmap_gemm_fma_allinone(__nv_bfloat16 const* __restrict__ residual_in,
    __nv_bfloat16 const* __restrict__ x_in, float const* __restrict__ post_mix_prev,
    float const* __restrict__ comb_mix_prev, float const* __restrict__ W_T, float const* __restrict__ hc_scale,
    float const* __restrict__ hc_base, __nv_bfloat16* __restrict__ residual_out, float* __restrict__ post_mix_out,
    float* __restrict__ comb_mix_out, __nv_bfloat16* __restrict__ layer_input_out, float* __restrict__ y_acc,
    float* __restrict__ r_acc, int* __restrict__ done_counter, int M, int K, int hidden_size, float rms_eps,
    float hc_pre_eps, float hc_sinkhorn_eps, float hc_post_mult_value, int sinkhorn_repeat,
    // When kFuseNorm: apply next-layer RMSNorm on layer_input inline:
    //   layer_input[t,h] = bf16(li * rsqrt(mean(li²)+norm_eps) * norm_weight[h]).
    __nv_bfloat16 const* __restrict__ norm_weight = nullptr, float norm_eps = 0.f,
    // kLaggedPre (V4.1), see mhcBigFuseKernel: collapse with pre_mix_ext, write the own gate to pre_mix_out.
    float const* __restrict__ pre_mix_ext = nullptr, float* __restrict__ pre_mix_out = nullptr)
{
    constexpr int HC_MULT = 4;
    constexpr int HC_MULT2 = HC_MULT * HC_MULT;
    constexpr int HC_MULT3 = HC_MULT * (2 + HC_MULT);
    static_assert(TM >= 1, "TM>=1");
    static_assert(FULL_N % TN == 0, "TN must divide FULL_N");
    static_assert(HC_MULT3 == FULL_N, "FULL_N must be HC_MULT*(2+HC_MULT)=24");
    static_assert(KS == 1 || KS == 2 || KS == 4 || KS == 8, "KS in {1,2,4,8}");
    constexpr int BLOCK_SIZE = 256;
    constexpr int WARP_SIZE = 32;
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    constexpr int BF16_VEC = (BF16_VEC_OVERRIDE != 0) ? BF16_VEC_OVERRIDE
                                                      : ((KS == 1 || KS == 2) ? 8
                                                              : (KS == 4)     ? 4
                                                                              : 2);
    static_assert(BF16_VEC == 2 || BF16_VEC == 4 || BF16_VEC == 8, "BF16_VEC in {2,4,8}");
    constexpr int CTAS_PER_BATCH = (FULL_N / TN) * KS;

    int const base_tok = blockIdx.x * TM;
    int const n_start = blockIdx.y * TN;
    int const k_split = blockIdx.z;
    int const tid = threadIdx.x;
    int const warp_id = tid / WARP_SIZE;
    int const lane = tid % WARP_SIZE;
    bool const is_n0 = (n_start == 0);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaGridDependencySynchronize();
#endif

    if (base_tok >= M)
    {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
        cudaTriggerProgrammaticLaunchCompletion();
#endif
        return;
    }

    int const hidden_per_split = hidden_size / KS;
    int const h_lo = k_split * hidden_per_split;
    int const h_hi = h_lo + hidden_per_split;

    bool valid[TM];
#pragma unroll
    for (int t = 0; t < TM; t++)
        valid[t] = (base_tok + t) < M;

    __shared__ float s_pm[TM][HC_MULT];
    __shared__ float s_cm[TM][HC_MULT][HC_MULT];

    {
        constexpr int N_PM = TM * HC_MULT;
        for (int i = tid; i < N_PM; i += BLOCK_SIZE)
        {
            int const t = i / HC_MULT;
            int const j = i % HC_MULT;
            float v = 0.0f;
            if (valid[t])
                v = post_mix_prev[(base_tok + t) * HC_MULT + j];
            s_pm[t][j] = v;
        }
    }
    {
        constexpr int N_CM = TM * HC_MULT * HC_MULT;
        for (int i = tid; i < N_CM; i += BLOCK_SIZE)
        {
            int const t = i / HC_MULT2;
            int const r = (i / HC_MULT) % HC_MULT;
            int const c = i % HC_MULT;
            float v = 0.0f;
            if (valid[t])
                v = comb_mix_prev[(base_tok + t) * HC_MULT2 + r * HC_MULT + c];
            s_cm[t][r][c] = v;
        }
    }
    __syncthreads();

    float acc[TM][TN];
    float sqr[TM];
#pragma unroll
    for (int t = 0; t < TM; t++)
    {
        sqr[t] = 0.0f;
#pragma unroll
        for (int n = 0; n < TN; n++)
            acc[t][n] = 0.0f;
    }

    // Phase 1: pmap + GEMM + residual_out write + sqrsum accumulate.
    // Iterate h over this CTA's HIDDEN slice; each thread owns BF16_VEC.
    for (int h = h_lo + tid * BF16_VEC; h < h_hi; h += BLOCK_SIZE * BF16_VEC)
    {
#pragma unroll
        for (int t = 0; t < TM; t++)
        {
            if (!valid[t])
                continue;
            long long const tok_x = static_cast<long long>(base_tok + t) * hidden_size;
            long long const tok_r = static_cast<long long>(base_tok + t) * HC_MULT * hidden_size;

            float xf[BF16_VEC];
            if constexpr (BF16_VEC == 8)
            {
                uint4 x_raw = *reinterpret_cast<uint4 const*>(&x_in[tok_x + h]);
                __nv_bfloat162 const* xp = reinterpret_cast<__nv_bfloat162 const*>(&x_raw);
#pragma unroll
                for (int v = 0; v < 4; v++)
                {
                    float2 f = __bfloat1622float2(xp[v]);
                    xf[2 * v + 0] = f.x;
                    xf[2 * v + 1] = f.y;
                }
            }
            else if constexpr (BF16_VEC == 4)
            {
                uint2 x_raw = *reinterpret_cast<uint2 const*>(&x_in[tok_x + h]);
                __nv_bfloat162 const* xp = reinterpret_cast<__nv_bfloat162 const*>(&x_raw);
#pragma unroll
                for (int v = 0; v < 2; v++)
                {
                    float2 f = __bfloat1622float2(xp[v]);
                    xf[2 * v + 0] = f.x;
                    xf[2 * v + 1] = f.y;
                }
            }
            else
            {
                unsigned x_raw = *reinterpret_cast<unsigned const*>(&x_in[tok_x + h]);
                __nv_bfloat162 xp = *reinterpret_cast<__nv_bfloat162 const*>(&x_raw);
                float2 f = __bfloat1622float2(xp);
                xf[0] = f.x;
                xf[1] = f.y;
            }

            float rf[HC_MULT][BF16_VEC];
#pragma unroll
            for (int k = 0; k < HC_MULT; k++)
            {
                __nv_bfloat16 const* r_ptr = &residual_in[tok_r + k * hidden_size + h];
                if constexpr (BF16_VEC == 8)
                {
                    uint4 r_raw = *reinterpret_cast<uint4 const*>(r_ptr);
                    __nv_bfloat162 const* rp = reinterpret_cast<__nv_bfloat162 const*>(&r_raw);
#pragma unroll
                    for (int v = 0; v < 4; v++)
                    {
                        float2 f = __bfloat1622float2(rp[v]);
                        rf[k][2 * v + 0] = f.x;
                        rf[k][2 * v + 1] = f.y;
                    }
                }
                else if constexpr (BF16_VEC == 4)
                {
                    uint2 r_raw = *reinterpret_cast<uint2 const*>(r_ptr);
                    __nv_bfloat162 const* rp = reinterpret_cast<__nv_bfloat162 const*>(&r_raw);
#pragma unroll
                    for (int v = 0; v < 2; v++)
                    {
                        float2 f = __bfloat1622float2(rp[v]);
                        rf[k][2 * v + 0] = f.x;
                        rf[k][2 * v + 1] = f.y;
                    }
                }
                else
                {
                    unsigned r_raw = *reinterpret_cast<unsigned const*>(r_ptr);
                    __nv_bfloat162 rp = *reinterpret_cast<__nv_bfloat162 const*>(&r_raw);
                    float2 f = __bfloat1622float2(rp);
                    rf[k][0] = f.x;
                    rf[k][1] = f.y;
                }
            }

            float pm[HC_MULT];
            float cm[HC_MULT][HC_MULT];
#pragma unroll
            for (int j = 0; j < HC_MULT; j++)
                pm[j] = s_pm[t][j];
#pragma unroll
            for (int k = 0; k < HC_MULT; k++)
#pragma unroll
                for (int j = 0; j < HC_MULT; j++)
                    cm[k][j] = s_cm[t][k][j];

            float new_r[HC_MULT][BF16_VEC];
#pragma unroll
            for (int j = 0; j < HC_MULT; j++)
#pragma unroll
                for (int v = 0; v < BF16_VEC; v++)
                    new_r[j][v] = pm[j] * xf[v];
#pragma unroll
            for (int k = 0; k < HC_MULT; k++)
#pragma unroll
                for (int j = 0; j < HC_MULT; j++)
#pragma unroll
                    for (int v = 0; v < BF16_VEC; v++)
                        new_r[j][v] = fmaf(cm[k][j], rf[k][v], new_r[j][v]);

            if (is_n0)
            {
#pragma unroll
                for (int j = 0; j < HC_MULT; j++)
                {
                    __nv_bfloat16* o_ptr = residual_out + tok_r + j * hidden_size + h;
                    if constexpr (BF16_VEC == 8)
                    {
                        uint4 packed;
                        __nv_bfloat162* pairs = reinterpret_cast<__nv_bfloat162*>(&packed);
#pragma unroll
                        for (int v = 0; v < 4; v++)
                            pairs[v] = __float22bfloat162_rn(make_float2(new_r[j][2 * v], new_r[j][2 * v + 1]));
                        *reinterpret_cast<uint4*>(o_ptr) = packed;
                    }
                    else if constexpr (BF16_VEC == 4)
                    {
                        uint2 packed;
                        __nv_bfloat162* pairs = reinterpret_cast<__nv_bfloat162*>(&packed);
#pragma unroll
                        for (int v = 0; v < 2; v++)
                            pairs[v] = __float22bfloat162_rn(make_float2(new_r[j][2 * v], new_r[j][2 * v + 1]));
                        *reinterpret_cast<uint2*>(o_ptr) = packed;
                    }
                    else
                    {
                        __nv_bfloat162 packed = __float22bfloat162_rn(make_float2(new_r[j][0], new_r[j][1]));
                        *reinterpret_cast<unsigned*>(o_ptr) = *reinterpret_cast<unsigned*>(&packed);
                    }
                }

#pragma unroll
                for (int j = 0; j < HC_MULT; j++)
#pragma unroll
                    for (int v = 0; v < BF16_VEC; v++)
                        sqr[t] = fmaf(new_r[j][v], new_r[j][v], sqr[t]);
            }

#pragma unroll
            for (int n = 0; n < TN; n++)
            {
#pragma unroll
                for (int j = 0; j < HC_MULT; j++)
                {
                    long long const w_base = static_cast<long long>(n_start + n) * K + j * hidden_size + h;
                    float w[BF16_VEC];
                    if constexpr (BF16_VEC == 8)
                    {
                        float w0, w1, w2, w3, w4, w5, w6, w7;
                        asm volatile("ld.global.L1::evict_last.v4.f32 {%0, %1, %2, %3}, [%4];"
                                     : "=f"(w0), "=f"(w1), "=f"(w2), "=f"(w3)
                                     : "l"(W_T + w_base));
                        asm volatile("ld.global.L1::evict_last.v4.f32 {%0, %1, %2, %3}, [%4];"
                                     : "=f"(w4), "=f"(w5), "=f"(w6), "=f"(w7)
                                     : "l"(W_T + w_base + 4));
                        w[0] = w0;
                        w[1] = w1;
                        w[2] = w2;
                        w[3] = w3;
                        w[4] = w4;
                        w[5] = w5;
                        w[6] = w6;
                        w[7] = w7;
                    }
                    else if constexpr (BF16_VEC == 4)
                    {
                        float w0, w1, w2, w3;
                        asm volatile("ld.global.L1::evict_last.v4.f32 {%0, %1, %2, %3}, [%4];"
                                     : "=f"(w0), "=f"(w1), "=f"(w2), "=f"(w3)
                                     : "l"(W_T + w_base));
                        w[0] = w0;
                        w[1] = w1;
                        w[2] = w2;
                        w[3] = w3;
                    }
                    else
                    {
                        float w0, w1;
                        asm volatile("ld.global.L1::evict_last.v2.f32 {%0, %1}, [%2];"
                                     : "=f"(w0), "=f"(w1)
                                     : "l"(W_T + w_base));
                        w[0] = w0;
                        w[1] = w1;
                    }
#pragma unroll
                    for (int v = 0; v < BF16_VEC; v++)
                        acc[t][n] = fmaf(new_r[j][v], w[v], acc[t][n]);
                }
            }
        }
    }

    // Phase 2: warp-reduce + SMEM cross-warp + atomicAdd into y_acc / r_acc.
    constexpr int COLS_PER_GROUP = NUM_WARPS;
    constexpr int SQRSUM_SLOT = COLS_PER_GROUP;
    constexpr int N_GROUPS = (TN + COLS_PER_GROUP - 1) / COLS_PER_GROUP;
    constexpr unsigned FULL_WARP_MASK = 0xffffffff;

    __shared__ float s_warp[NUM_WARPS][COLS_PER_GROUP + 1];

#pragma unroll
    for (int t = 0; t < TM; t++)
    {
        if (!valid[t])
            continue;
#pragma unroll
        for (int g = 0; g < N_GROUPS; g++)
        {
            constexpr int LAST_COLS = TN - (N_GROUPS - 1) * COLS_PER_GROUP;
            int const n_cols = (g == N_GROUPS - 1) ? LAST_COLS : COLS_PER_GROUP;
#pragma unroll
            for (int n = 0; n < COLS_PER_GROUP; n++)
            {
                if (n < n_cols)
                {
#pragma unroll
                    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
                        acc[t][g * COLS_PER_GROUP + n]
                            += __shfl_xor_sync(FULL_WARP_MASK, acc[t][g * COLS_PER_GROUP + n], off);
                }
            }
            if (g == 0 && is_n0)
            {
#pragma unroll
                for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
                    sqr[t] += __shfl_xor_sync(FULL_WARP_MASK, sqr[t], off);
            }
            if (lane == 0)
            {
#pragma unroll
                for (int n = 0; n < COLS_PER_GROUP; n++)
                    if (n < n_cols)
                        s_warp[warp_id][n] = acc[t][g * COLS_PER_GROUP + n];
                if (g == 0 && is_n0)
                    s_warp[warp_id][SQRSUM_SLOT] = sqr[t];
            }
            __syncthreads();
            if (lane == 0 && warp_id < n_cols)
            {
                float val = 0.0f;
#pragma unroll
                for (int ww = 0; ww < NUM_WARPS; ww++)
                    val += s_warp[ww][warp_id];
                long long y_idx
                    = static_cast<long long>(base_tok + t) * FULL_N + n_start + g * COLS_PER_GROUP + warp_id;
                if constexpr (KS == 1)
                {
                    y_acc[y_idx] = val;
                }
                else
                {
                    atomicAdd(y_acc + y_idx, val);
                }
            }
            if (g == 0 && is_n0 && lane == 0 && warp_id == 0)
            {
                float sq = 0.0f;
#pragma unroll
                for (int ww = 0; ww < NUM_WARPS; ww++)
                    sq += s_warp[ww][SQRSUM_SLOT];
                if constexpr (KS == 1)
                {
                    r_acc[base_tok + t] = sq;
                }
                else
                {
                    atomicAdd(r_acc + (base_tok + t), sq);
                }
            }
            __syncthreads();
        }
    }

    // Phase 3: elect last-home CTA per (ceil(M/TM)) token-batch via counter.
    __shared__ int s_is_last;
    if constexpr (CTAS_PER_BATCH == 1)
    {
        s_is_last = 1;
    }
    else
    {
        __threadfence();
        __syncthreads();
        if (tid == 0)
        {
            int const prev = atomicAdd(&done_counter[blockIdx.x], 1);
            s_is_last = (prev == CTAS_PER_BATCH - 1) ? 1 : 0;
        }
        __syncthreads();
    }
    if (!s_is_last)
    {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
        cudaTriggerProgrammaticLaunchCompletion();
#endif
        return;
    }

    // Phase 4: inline bigFuse for the TM tokens in this batch.
    // Layout: FULL_N = HC_MULT*(2+HC_MULT) = 24
    //   y_acc[0..HC_MULT)               → s_pre_mix (sigmoid gate)
    //   y_acc[HC_MULT..2*HC_MULT)       → post_mix_out (sigmoid*hc_post_mult)
    //   y_acc[2*HC_MULT..FULL_N)        → comb_mix_out (Sinkhorn)
    __shared__ float s_pre_mix[TM][HC_MULT];

#pragma unroll
    for (int t = 0; t < TM; t++)
    {
        if (!valid[t])
        {
            __syncthreads();
            continue;
        }
        int const tok = base_tok + t;

        if (warp_id == 0 && lane < HC_MULT)
        {
            float cm_vals[HC_MULT];
            float const r_val = r_acc[tok];
            float const* y_row = y_acc + static_cast<long long>(tok) * FULL_N;

            float const rstd = rsqrtf(r_val / static_cast<float>(K) + rms_eps);
            float const s0 = hc_scale[0], s1 = hc_scale[1], s2 = hc_scale[2];

            float v = y_row[lane] * rstd * s0 + hc_base[lane];
            s_pre_mix[t][lane] = 1.0f / (1.0f + expf(-v)) + hc_pre_eps;
            if constexpr (kLaggedPre)
                pre_mix_out[tok * HC_MULT + lane] = s_pre_mix[t][lane];

            v = y_row[HC_MULT + lane] * rstd * s1 + hc_base[HC_MULT + lane];
            post_mix_out[tok * HC_MULT + lane] = 1.0f / (1.0f + expf(-v)) * hc_post_mult_value;

#pragma unroll
            for (int k = 0; k < HC_MULT; k++)
                cm_vals[k]
                    = y_row[2 * HC_MULT + lane * HC_MULT + k] * rstd * s2 + hc_base[2 * HC_MULT + lane * HC_MULT + k];

            constexpr unsigned LANE_MASK = (1u << HC_MULT) - 1;
            float const rowMax = fmaxf(fmaxf(cm_vals[0], cm_vals[1]), fmaxf(cm_vals[2], cm_vals[3]));
#pragma unroll
            for (int k = 0; k < HC_MULT; k++)
                cm_vals[k] = expf(cm_vals[k] - rowMax);
            // Reciprocal-multiply: 1 fdiv + 4 fmul vs 4 fdivs (faster on B200).
            float inv_rs = 1.0f / (cm_vals[0] + cm_vals[1] + cm_vals[2] + cm_vals[3]);
#pragma unroll
            for (int k = 0; k < HC_MULT; k++)
                cm_vals[k] = cm_vals[k] * inv_rs + hc_sinkhorn_eps;
#pragma unroll
            for (int k = 0; k < HC_MULT; k++)
            {
                float cs = cm_vals[k];
                cs += __shfl_xor_sync(LANE_MASK, cs, 1);
                cs += __shfl_xor_sync(LANE_MASK, cs, 2);
                cm_vals[k] *= 1.0f / (cs + hc_sinkhorn_eps);
            }
            for (int it = 1; it < sinkhorn_repeat; it++)
            {
                inv_rs = 1.0f / (cm_vals[0] + cm_vals[1] + cm_vals[2] + cm_vals[3] + hc_sinkhorn_eps);
#pragma unroll
                for (int k = 0; k < HC_MULT; k++)
                    cm_vals[k] *= inv_rs;
#pragma unroll
                for (int k = 0; k < HC_MULT; k++)
                {
                    float cs = cm_vals[k];
                    cs += __shfl_xor_sync(LANE_MASK, cs, 1);
                    cs += __shfl_xor_sync(LANE_MASK, cs, 2);
                    cm_vals[k] *= 1.0f / (cs + hc_sinkhorn_eps);
                }
            }
            float* cm_out_ptr = comb_mix_out + tok * HC_MULT2;
#pragma unroll
            for (int k = 0; k < HC_MULT; k++)
                cm_out_ptr[lane * HC_MULT + k] = cm_vals[k];
        }
        __syncthreads();

        // layer_input: all eight warps process hidden in parallel.
        //
        // When kFuseNorm is true, accumulate per-thread sum_sq while writing
        // un-normalized layer_input bf16; after a __syncthreads, all threads
        // run pass 2: re-LDG from L2 (hot from pass 1's STGs), multiply by
        // rsqrt * norm_weight, STG normalized bf16. Saves one HBM read+write
        // pair vs the separate flashinfer.rmsnorm kernel.
        constexpr int kFmaBigFuseWarps = BLOCK_SIZE / WARP_SIZE;
        __shared__ float s_sumsq_li[kFmaBigFuseWarps];
        __shared__ float s_rsqrt_li;
        constexpr int BF16_VEC_LI = 8;
        __nv_bfloat16* obase = layer_input_out + static_cast<long long>(tok) * hidden_size;
        {
            float pm[HC_MULT];
#pragma unroll
            for (int j = 0; j < HC_MULT; j++)
            {
                if constexpr (kLaggedPre)
                    pm[j] = pre_mix_ext[tok * HC_MULT + j];
                else
                    pm[j] = s_pre_mix[t][j];
            }

            __nv_bfloat16 const* rbase = residual_out + static_cast<long long>(tok) * HC_MULT * hidden_size;
            int const p2_tid = tid;
            constexpr int p2_threads = BLOCK_SIZE;
            float sum_sq_local = 0.f;

            for (int h = p2_tid * BF16_VEC_LI; h < hidden_size; h += p2_threads * BF16_VEC_LI)
            {
                float acc_li[BF16_VEC_LI] = {};
#pragma unroll
                for (int j = 0; j < HC_MULT; j++)
                {
                    uint4 raw = *reinterpret_cast<uint4 const*>(&rbase[j * hidden_size + h]);
                    __nv_bfloat162 const* pairs = reinterpret_cast<__nv_bfloat162 const*>(&raw);
#pragma unroll
                    for (int v = 0; v < BF16_VEC_LI / 2; v++)
                    {
                        float2 f = __bfloat1622float2(pairs[v]);
                        acc_li[2 * v + 0] += pm[j] * f.x;
                        acc_li[2 * v + 1] += pm[j] * f.y;
                    }
                }
                uint4 out_raw;
                __nv_bfloat162* opairs = reinterpret_cast<__nv_bfloat162*>(&out_raw);
#pragma unroll
                for (int v = 0; v < BF16_VEC_LI / 2; v++)
                    opairs[v] = __float22bfloat162_rn(make_float2(acc_li[2 * v], acc_li[2 * v + 1]));
                *reinterpret_cast<uint4*>(&obase[h]) = out_raw;
                if constexpr (kFuseNorm)
                {
#pragma unroll
                    for (int v = 0; v < BF16_VEC_LI / 2; v++)
                    {
                        float2 b = __bfloat1622float2(opairs[v]);
                        sum_sq_local += b.x * b.x + b.y * b.y;
                    }
                }
            }

            if constexpr (kFuseNorm)
            {
                sum_sq_local += __shfl_xor_sync(0xffffffff, sum_sq_local, 16);
                sum_sq_local += __shfl_xor_sync(0xffffffff, sum_sq_local, 8);
                sum_sq_local += __shfl_xor_sync(0xffffffff, sum_sq_local, 4);
                sum_sq_local += __shfl_xor_sync(0xffffffff, sum_sq_local, 2);
                sum_sq_local += __shfl_xor_sync(0xffffffff, sum_sq_local, 1);
                if ((tid & 31) == 0)
                    s_sumsq_li[warp_id] = sum_sq_local;
            }
        }
        if constexpr (kFuseNorm)
        {
            __syncthreads();
            if (warp_id == 0 && (tid & 31) == 0)
            {
                float total = 0.f;
#pragma unroll
                for (int w = 0; w < kFmaBigFuseWarps; w++)
                    total += s_sumsq_li[w];
                s_rsqrt_li = rsqrtf(total / static_cast<float>(hidden_size) + norm_eps);
            }
            __syncthreads();
            {
                int const p2_tid = tid;
                constexpr int p2_threads = BLOCK_SIZE;
                float const rsqrt_val = s_rsqrt_li;
                for (int h = p2_tid * BF16_VEC_LI; h < hidden_size; h += p2_threads * BF16_VEC_LI)
                {
                    uint4 li_raw = *reinterpret_cast<uint4 const*>(&obase[h]);
                    uint4 nw_raw = *reinterpret_cast<uint4 const*>(&norm_weight[h]);
                    __nv_bfloat162 const* li_pairs = reinterpret_cast<__nv_bfloat162 const*>(&li_raw);
                    __nv_bfloat162 const* nw_pairs = reinterpret_cast<__nv_bfloat162 const*>(&nw_raw);
                    uint4 out_raw;
                    __nv_bfloat162* opairs = reinterpret_cast<__nv_bfloat162*>(&out_raw);
#pragma unroll
                    for (int v = 0; v < BF16_VEC_LI / 2; v++)
                    {
                        float2 lif = __bfloat1622float2(li_pairs[v]);
                        float2 nwf = __bfloat1622float2(nw_pairs[v]);
                        opairs[v]
                            = __float22bfloat162_rn(make_float2(lif.x * rsqrt_val * nwf.x, lif.y * rsqrt_val * nwf.y));
                    }
                    *reinterpret_cast<uint4*>(&obase[h]) = out_raw;
                }
            }
        }
        __syncthreads();
    }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// ===================================================================
// fused_pmap_gemm_fma_mtile<N_PER_BLOCK, TM>
// -------------------------------------------------------------------
// M-tiled variant of fused_pmap_gemm_fma_ksplit<N_PER_BLOCK, 1, 0, true>
// (NUM_K_SPLITS = 1, BF16_VEC = 8, WRITE_RESIDUAL = true) for prefill-size M.
// Each CTA handles TM consecutive tokens, so every W_T register load is reused
// TM times instead of once (the ksplit kernel streams the whole 24 x 4*hidden
// fp32 W_T through L2 once per token).
//
// BITWISE EQUAL to the ksplit kernel: for every token, every thread owns the
// same h positions (tid*8 + s*2048), and every accumulator sees the same fmaf
// chain:
//   new_r[j][v] = pm[j]*x[v], then fmaf(comb[k][j], r[k][v], .) for k = 0..3;
//   sqr  += fmaf over (step, j, v);
//   acc[n] += fmaf over (step, j, v).
// The loop is reorganised j-outer / n-inner, which leaves each per-(n) and
// per-(j,v) chain unchanged.  The warp xor-shuffle tree and the sequential
// 8-warp sum into Yp/Rp are the ksplit kernel's, run once per token.
// Grid: (ceil(M / TM), N / N_PER_BLOCK, 1).  Block: 256 threads.
// Yp layout [1, M, N], Rp [1, M] (same as ksplit with ks = 1).
// ===================================================================
template <int N_PER_BLOCK, int TM>
__launch_bounds__(256, (TM * N_PER_BLOCK <= 48) ? 2 : 1) __global__
    void fused_pmap_gemm_fma_mtile(__nv_bfloat16 const* __restrict__ residual_in,
        __nv_bfloat16 const* __restrict__ x_in, float const* __restrict__ post_mix,
        float const* __restrict__ comb_mix, float const* __restrict__ W_T, float* __restrict__ Yp,
        float* __restrict__ Rp, int M, int hidden_size, int N, int K, __nv_bfloat16* __restrict__ residual_out)
{
    constexpr int HC_MULT = 4;
    constexpr int BLOCK_SIZE = 256;
    constexpr int WARP_SIZE = 32;
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    constexpr int BF16_VEC = 8;

    int const tok0 = blockIdx.x * TM;
    int const n_start = blockIdx.y * N_PER_BLOCK;
    int const tid = threadIdx.x;
    int const warp_id = tid / WARP_SIZE;
    int const lane = tid % WARP_SIZE;
    bool const do_sqr = (n_start == 0);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaGridDependencySynchronize();
#endif

    __shared__ float s_post[TM][HC_MULT];
    __shared__ float s_comb[TM][HC_MULT][HC_MULT];

    if (tid < TM * HC_MULT)
    {
        int const t = tid / HC_MULT;
        int const c = tid % HC_MULT;
        int const tok = min(tok0 + t, M - 1);
        s_post[t][c] = post_mix[tok * HC_MULT + c];
    }
    if (tid < TM * HC_MULT * HC_MULT)
    {
        int const t = tid / (HC_MULT * HC_MULT);
        int const rc = tid % (HC_MULT * HC_MULT);
        int const tok = min(tok0 + t, M - 1);
        s_comb[t][rc / HC_MULT][rc % HC_MULT] = comb_mix[tok * HC_MULT * HC_MULT + rc];
    }
    __syncthreads();

    // Tokens past M (last CTA) recompute token M-1 and never write.
    long long tok_res[TM];
    long long tok_x[TM];
    bool tok_ok[TM];
#pragma unroll
    for (int t = 0; t < TM; t++)
    {
        int const tok = min(tok0 + t, M - 1);
        tok_ok[t] = (tok0 + t) < M;
        tok_res[t] = static_cast<long long>(tok) * HC_MULT * hidden_size;
        tok_x[t] = static_cast<long long>(tok) * hidden_size;
    }

    float acc[TM][N_PER_BLOCK];
    float sqr[TM];
#pragma unroll
    for (int t = 0; t < TM; t++)
    {
        sqr[t] = 0.0f;
#pragma unroll
        for (int n = 0; n < N_PER_BLOCK; n++)
            acc[t][n] = 0.0f;
    }

    for (int h = tid * BF16_VEC; h < hidden_size; h += BLOCK_SIZE * BF16_VEC)
    {
        uint4 x_raw[TM];
        uint4 r_raw[TM][HC_MULT];
#pragma unroll
        for (int t = 0; t < TM; t++)
        {
            x_raw[t] = *reinterpret_cast<uint4 const*>(&x_in[tok_x[t] + h]);
#pragma unroll
            for (int k = 0; k < HC_MULT; k++)
                r_raw[t][k] = *reinterpret_cast<uint4 const*>(&residual_in[tok_res[t] + k * hidden_size + h]);
        }

#pragma unroll
        for (int j = 0; j < HC_MULT; j++)
        {
            float new_r[TM][BF16_VEC];
#pragma unroll
            for (int t = 0; t < TM; t++)
            {
                float const pmj = s_post[t][j];
                __nv_bfloat162 const* xp = reinterpret_cast<__nv_bfloat162 const*>(&x_raw[t]);
#pragma unroll
                for (int v = 0; v < 4; v++)
                {
                    float2 f = __bfloat1622float2(xp[v]);
                    new_r[t][2 * v + 0] = pmj * f.x;
                    new_r[t][2 * v + 1] = pmj * f.y;
                }
#pragma unroll
                for (int k = 0; k < HC_MULT; k++)
                {
                    float const ckj = s_comb[t][k][j];
                    __nv_bfloat162 const* rp = reinterpret_cast<__nv_bfloat162 const*>(&r_raw[t][k]);
#pragma unroll
                    for (int v = 0; v < 4; v++)
                    {
                        float2 f = __bfloat1622float2(rp[v]);
                        new_r[t][2 * v + 0] = fmaf(ckj, f.x, new_r[t][2 * v + 0]);
                        new_r[t][2 * v + 1] = fmaf(ckj, f.y, new_r[t][2 * v + 1]);
                    }
                }
            }

            if (do_sqr)
            {
#pragma unroll
                for (int t = 0; t < TM; t++)
                {
#pragma unroll
                    for (int v = 0; v < BF16_VEC; v++)
                        sqr[t] = fmaf(new_r[t][v], new_r[t][v], sqr[t]);
                    if (tok_ok[t])
                    {
                        uint4 packed;
                        __nv_bfloat162* pairs = reinterpret_cast<__nv_bfloat162*>(&packed);
#pragma unroll
                        for (int v = 0; v < 4; v++)
                            pairs[v] = __float22bfloat162_rn(make_float2(new_r[t][2 * v], new_r[t][2 * v + 1]));
                        *reinterpret_cast<uint4*>(residual_out + tok_res[t] + j * hidden_size + h) = packed;
                    }
                }
            }

#pragma unroll
            for (int n = 0; n < N_PER_BLOCK; n++)
            {
                long long const w_base = static_cast<long long>(n_start + n) * K + j * hidden_size + h;
                float w[BF16_VEC];
                asm volatile("ld.global.L1::evict_last.v4.f32 {%0, %1, %2, %3}, [%4];"
                             : "=f"(w[0]), "=f"(w[1]), "=f"(w[2]), "=f"(w[3])
                             : "l"(W_T + w_base));
                asm volatile("ld.global.L1::evict_last.v4.f32 {%0, %1, %2, %3}, [%4];"
                             : "=f"(w[4]), "=f"(w[5]), "=f"(w[6]), "=f"(w[7])
                             : "l"(W_T + w_base + 4));
#pragma unroll
                for (int t = 0; t < TM; t++)
#pragma unroll
                    for (int v = 0; v < BF16_VEC; v++)
                        acc[t][n] = fmaf(new_r[t][v], w[v], acc[t][n]);
            }
        }
    }

    constexpr int COLS_PER_GROUP = NUM_WARPS;
    constexpr int SQRSUM_SLOT = COLS_PER_GROUP;
    constexpr int N_GROUPS = (N_PER_BLOCK + COLS_PER_GROUP - 1) / COLS_PER_GROUP;
    constexpr unsigned FULL_WARP_MASK = 0xffffffff;

    __shared__ float s_warp[NUM_WARPS][COLS_PER_GROUP + 1];

#pragma unroll
    for (int t = 0; t < TM; t++)
    {
        int const token = tok0 + t;
#pragma unroll
        for (int g = 0; g < N_GROUPS; g++)
        {
            constexpr int LAST_COLS = N_PER_BLOCK - (N_GROUPS - 1) * COLS_PER_GROUP;
            int const n_cols = (g == N_GROUPS - 1) ? LAST_COLS : COLS_PER_GROUP;
#pragma unroll
            for (int n = 0; n < COLS_PER_GROUP; n++)
            {
                if (n < n_cols)
                {
#pragma unroll
                    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
                        acc[t][g * COLS_PER_GROUP + n]
                            += __shfl_xor_sync(FULL_WARP_MASK, acc[t][g * COLS_PER_GROUP + n], off);
                }
            }
            if (g == 0 && do_sqr)
            {
#pragma unroll
                for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
                    sqr[t] += __shfl_xor_sync(FULL_WARP_MASK, sqr[t], off);
            }
            if (lane == 0)
            {
#pragma unroll
                for (int n = 0; n < COLS_PER_GROUP; n++)
                    if (n < n_cols)
                        s_warp[warp_id][n] = acc[t][g * COLS_PER_GROUP + n];
                if (g == 0 && do_sqr)
                    s_warp[warp_id][SQRSUM_SLOT] = sqr[t];
            }
            __syncthreads();
            if (tok_ok[t] && lane == 0 && warp_id < n_cols)
            {
                float val = 0.0f;
#pragma unroll
                for (int w = 0; w < NUM_WARPS; w++)
                    val += s_warp[w][warp_id];
                int const n_global = n_start + g * COLS_PER_GROUP + warp_id;
                Yp[static_cast<long long>(token) * N + n_global] = val;
            }
            if (tok_ok[t] && g == 0 && do_sqr && lane == 0 && warp_id == 0)
            {
                float sq = 0.0f;
#pragma unroll
                for (int w = 0; w < NUM_WARPS; w++)
                    sq += s_warp[w][SQRSUM_SLOT];
                Rp[token] = sq;
            }
            __syncthreads();
        }
    }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// ===================================================================
// fused_pmap_gemm_fma_mtile_v2<N_PER_BLOCK, TM, RAW_SMEM, MIN_BLOCKS>
// -------------------------------------------------------------------
// Spill-free rebuild of fused_pmap_gemm_fma_mtile<N_PER_BLOCK, TM>.  BITWISE
// EQUAL to it (and so to the ks = 1 ksplit kernel): every thread owns the same
// h positions, and every fmaf chain (new_r, sqr, acc) and the shuffle / 8-warp
// epilogue are the same operations in the same order.  Only register
// allocation and data movement change:
//   - MIN_BLOCKS is explicit (1 lifts the 128-register cap that made the
//     (12,4)/(8,4)/(24,2) builds spill).
//   - token offsets are int32 (the launcher checks M * 4 * hidden < 2^31).
//   - the W_T loads stay volatile asm, as in v1 (non-volatile let ptxas hoist
//     them and spill more, job 1426).
//   - RAW_SMEM stages the thread-private bf16 inputs (x and the 4 residual
//     streams of each token) in shared memory with cp.async instead of
//     keeping 20 * TM registers live across the j loop.  Each thread reads
//     only the slots it wrote, so no barrier is needed.  The next h-step's
//     copies are issued once j = 3 has consumed the current ones.
//     Dynamic smem: TM * 5 * 256 * 16 bytes.
// Grid: (ceil(M / TM), N / N_PER_BLOCK, 1).  Block: 256 threads.
// ===================================================================
// Thread-private cp.async of one h-step's bf16 inputs (4 residual streams + x per token)
// into slots [(t * 5 + k) * 256 + tid] of the dynamic smem buffer.
template <int TM>
__device__ __forceinline__ void mtile_v2_issue_raw(uint4* s_raw, __nv_bfloat16 const* __restrict__ residual_in,
    __nv_bfloat16 const* __restrict__ x_in, int const (&tok_res)[TM], int hidden_size, int hh, int tid)
{
#pragma unroll
    for (int t = 0; t < TM; t++)
    {
        // x offset = tok * hidden = tok_res / 4 (tok_res = tok * 4 * hidden >= 0): no separate
        // tok_x registers live across the h loop.
        int const tok_x = tok_res[t] >> 2;
#pragma unroll
        for (int k = 0; k < 5; k++)
        {
            __nv_bfloat16 const* src = (k < 4) ? &residual_in[tok_res[t] + k * hidden_size + hh] : &x_in[tok_x + hh];
            unsigned const dst = static_cast<unsigned>(__cvta_generic_to_shared(&s_raw[(t * 5 + k) * 256 + tid]));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src) : "memory");
        }
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N_PER_BLOCK, int TM, bool RAW_SMEM, int MIN_BLOCKS, bool J_ROLLED = false>
__launch_bounds__(256, MIN_BLOCKS) __global__
    void fused_pmap_gemm_fma_mtile_v2(__nv_bfloat16 const* __restrict__ residual_in,
        __nv_bfloat16 const* __restrict__ x_in, float const* __restrict__ post_mix,
        float const* __restrict__ comb_mix, float const* __restrict__ W_T, float* __restrict__ Yp,
        float* __restrict__ Rp, int M, int hidden_size, int N, int K, __nv_bfloat16* __restrict__ residual_out)
{
    constexpr int HC_MULT = 4;
    constexpr int BLOCK_SIZE = 256;
    constexpr int WARP_SIZE = 32;
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    constexpr int BF16_VEC = 8;
    constexpr int RAW_SLOTS = HC_MULT + 1; // slot k < 4: residual stream k; slot 4: x

    int const tok0 = blockIdx.x * TM;
    int const n_start = blockIdx.y * N_PER_BLOCK;
    int const tid = threadIdx.x;
    int const warp_id = tid / WARP_SIZE;
    int const lane = tid % WARP_SIZE;
    bool const do_sqr = (n_start == 0);

    extern __shared__ uint4 s_raw_dyn[];

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaGridDependencySynchronize();
#endif

    __shared__ float s_post[TM][HC_MULT];
    __shared__ float s_comb[TM][HC_MULT][HC_MULT];

    if (tid < TM * HC_MULT)
    {
        int const t = tid / HC_MULT;
        int const c = tid % HC_MULT;
        int const tok = min(tok0 + t, M - 1);
        s_post[t][c] = post_mix[tok * HC_MULT + c];
    }
    if (tid < TM * HC_MULT * HC_MULT)
    {
        int const t = tid / (HC_MULT * HC_MULT);
        int const rc = tid % (HC_MULT * HC_MULT);
        int const tok = min(tok0 + t, M - 1);
        s_comb[t][rc / HC_MULT][rc % HC_MULT] = comb_mix[tok * HC_MULT * HC_MULT + rc];
    }

    // Tokens past M (last CTA) recompute token M-1 and never write.
    int tok_res[TM];
    int tok_x[TM];
    bool tok_ok[TM];
#pragma unroll
    for (int t = 0; t < TM; t++)
    {
        int const tok = min(tok0 + t, M - 1);
        tok_ok[t] = (tok0 + t) < M;
        tok_res[t] = tok * HC_MULT * hidden_size;
        tok_x[t] = tok * hidden_size;
    }

    if constexpr (RAW_SMEM)
    {
        if (tid * BF16_VEC < hidden_size)
            mtile_v2_issue_raw<TM>(s_raw_dyn, residual_in, x_in, tok_res, hidden_size, tid * BF16_VEC, tid);
    }
    __syncthreads();

    float acc[TM][N_PER_BLOCK];
    float sqr[TM];
#pragma unroll
    for (int t = 0; t < TM; t++)
    {
        sqr[t] = 0.0f;
#pragma unroll
        for (int n = 0; n < N_PER_BLOCK; n++)
            acc[t][n] = 0.0f;
    }

    for (int h = tid * BF16_VEC; h < hidden_size; h += BLOCK_SIZE * BF16_VEC)
    {
        uint4 x_raw[RAW_SMEM ? 1 : TM];
        uint4 r_raw[RAW_SMEM ? 1 : TM][HC_MULT];
        if constexpr (RAW_SMEM)
        {
            asm volatile("cp.async.wait_all;\n" ::: "memory");
        }
        else
        {
#pragma unroll
            for (int t = 0; t < TM; t++)
            {
                x_raw[t] = *reinterpret_cast<uint4 const*>(&x_in[tok_x[t] + h]);
#pragma unroll
                for (int k = 0; k < HC_MULT; k++)
                    r_raw[t][k] = *reinterpret_cast<uint4 const*>(&residual_in[tok_res[t] + k * hidden_size + h]);
            }
        }

        // J_ROLLED keeps the j loop rolled: nothing register-resident is indexed by j
        // (s_post/s_comb are smem, W_T/residual_out are addresses), so this only stops ptxas
        // from overlapping j+1's new_r with j's GEMM, which costs 8 * TM extra live registers.
#pragma unroll(J_ROLLED ? 1 : HC_MULT)
        for (int j = 0; j < HC_MULT; j++)
        {
            float new_r[TM][BF16_VEC];
#pragma unroll
            for (int t = 0; t < TM; t++)
            {
                float const pmj = s_post[t][j];
                uint4 xr;
                if constexpr (RAW_SMEM)
                    xr = s_raw_dyn[(t * RAW_SLOTS + HC_MULT) * BLOCK_SIZE + tid];
                else
                    xr = x_raw[t];
                __nv_bfloat162 const* xp = reinterpret_cast<__nv_bfloat162 const*>(&xr);
#pragma unroll
                for (int v = 0; v < 4; v++)
                {
                    float2 f = __bfloat1622float2(xp[v]);
                    new_r[t][2 * v + 0] = pmj * f.x;
                    new_r[t][2 * v + 1] = pmj * f.y;
                }
#pragma unroll
                for (int k = 0; k < HC_MULT; k++)
                {
                    float const ckj = s_comb[t][k][j];
                    uint4 rr;
                    if constexpr (RAW_SMEM)
                        rr = s_raw_dyn[(t * RAW_SLOTS + k) * BLOCK_SIZE + tid];
                    else
                        rr = r_raw[t][k];
                    __nv_bfloat162 const* rp = reinterpret_cast<__nv_bfloat162 const*>(&rr);
#pragma unroll
                    for (int v = 0; v < 4; v++)
                    {
                        float2 f = __bfloat1622float2(rp[v]);
                        new_r[t][2 * v + 0] = fmaf(ckj, f.x, new_r[t][2 * v + 0]);
                        new_r[t][2 * v + 1] = fmaf(ckj, f.y, new_r[t][2 * v + 1]);
                    }
                }
            }

            if constexpr (RAW_SMEM)
            {
                // All of this step's slots are consumed: prefetch the next step.
                if (j == HC_MULT - 1 && h + BLOCK_SIZE * BF16_VEC < hidden_size)
                    mtile_v2_issue_raw<TM>(
                        s_raw_dyn, residual_in, x_in, tok_res, hidden_size, h + BLOCK_SIZE * BF16_VEC, tid);
            }

            if (do_sqr)
            {
#pragma unroll
                for (int t = 0; t < TM; t++)
                {
#pragma unroll
                    for (int v = 0; v < BF16_VEC; v++)
                        sqr[t] = fmaf(new_r[t][v], new_r[t][v], sqr[t]);
                    if (tok_ok[t])
                    {
                        uint4 packed;
                        __nv_bfloat162* pairs = reinterpret_cast<__nv_bfloat162*>(&packed);
#pragma unroll
                        for (int v = 0; v < 4; v++)
                            pairs[v] = __float22bfloat162_rn(make_float2(new_r[t][2 * v], new_r[t][2 * v + 1]));
                        *reinterpret_cast<uint4*>(residual_out + tok_res[t] + j * hidden_size + h) = packed;
                    }
                }
            }

#pragma unroll
            for (int n = 0; n < N_PER_BLOCK; n++)
            {
                float const* wp = W_T + (n_start + n) * K + j * hidden_size + h;
                float w[BF16_VEC];
                // volatile, as in v1: job 1426 showed non-volatile W_T loads let ptxas hoist
                // them across n/j and spill (STACK 536-1200 B at 255 regs).
                asm volatile("ld.global.L1::evict_last.v4.f32 {%0, %1, %2, %3}, [%4];"
                             : "=f"(w[0]), "=f"(w[1]), "=f"(w[2]), "=f"(w[3])
                             : "l"(wp));
                asm volatile("ld.global.L1::evict_last.v4.f32 {%0, %1, %2, %3}, [%4];"
                             : "=f"(w[4]), "=f"(w[5]), "=f"(w[6]), "=f"(w[7])
                             : "l"(wp + 4));
#pragma unroll
                for (int t = 0; t < TM; t++)
#pragma unroll
                    for (int v = 0; v < BF16_VEC; v++)
                        acc[t][n] = fmaf(new_r[t][v], w[v], acc[t][n]);
            }
        }
    }

    constexpr int COLS_PER_GROUP = NUM_WARPS;
    constexpr int SQRSUM_SLOT = COLS_PER_GROUP;
    constexpr int N_GROUPS = (N_PER_BLOCK + COLS_PER_GROUP - 1) / COLS_PER_GROUP;
    constexpr unsigned FULL_WARP_MASK = 0xffffffff;

    __shared__ float s_warp[NUM_WARPS][COLS_PER_GROUP + 1];

#pragma unroll
    for (int t = 0; t < TM; t++)
    {
        int const token = tok0 + t;
#pragma unroll
        for (int g = 0; g < N_GROUPS; g++)
        {
            constexpr int LAST_COLS = N_PER_BLOCK - (N_GROUPS - 1) * COLS_PER_GROUP;
            int const n_cols = (g == N_GROUPS - 1) ? LAST_COLS : COLS_PER_GROUP;
#pragma unroll
            for (int n = 0; n < COLS_PER_GROUP; n++)
            {
                if (n < n_cols)
                {
#pragma unroll
                    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
                        acc[t][g * COLS_PER_GROUP + n]
                            += __shfl_xor_sync(FULL_WARP_MASK, acc[t][g * COLS_PER_GROUP + n], off);
                }
            }
            if (g == 0 && do_sqr)
            {
#pragma unroll
                for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
                    sqr[t] += __shfl_xor_sync(FULL_WARP_MASK, sqr[t], off);
            }
            if (lane == 0)
            {
#pragma unroll
                for (int n = 0; n < COLS_PER_GROUP; n++)
                    if (n < n_cols)
                        s_warp[warp_id][n] = acc[t][g * COLS_PER_GROUP + n];
                if (g == 0 && do_sqr)
                    s_warp[warp_id][SQRSUM_SLOT] = sqr[t];
            }
            __syncthreads();
            if (tok_ok[t] && lane == 0 && warp_id < n_cols)
            {
                float val = 0.0f;
#pragma unroll
                for (int w = 0; w < NUM_WARPS; w++)
                    val += s_warp[w][warp_id];
                int const n_global = n_start + g * COLS_PER_GROUP + warp_id;
                Yp[static_cast<long long>(token) * N + n_global] = val;
            }
            if (tok_ok[t] && g == 0 && do_sqr && lane == 0 && warp_id == 0)
            {
                float sq = 0.0f;
#pragma unroll
                for (int w = 0; w < NUM_WARPS; w++)
                    sq += s_warp[w][SQRSUM_SLOT];
                Rp[token] = sq;
            }
            __syncthreads();
        }
    }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// ===================================================================
// fused_pmap_gemm_fma_mtile_v3<TM, NG, MIN_BLOCKS>
// -------------------------------------------------------------------
// TM tokens x all 24 N columns per CTA (v1/v2 take at most 4 tokens), so each
// CTA streams W_T [24, 4*hidden] fp32 from L2 once per TM tokens instead of
// once per 4.  BITWISE EQUAL to fused_pmap_gemm_fma_mtile (v1) and to the
// ks = 1 ksplit kernel.
//
// ORDER MAP v1 -> v3.  In v1, thread tid ("vtid") of warp w = tid / 32 owns
// h = tid*8 + s*2048 (s = 0, 1, ...; h < hidden), and for every (token t, n):
//   new_r[j][v] = pm[t][j] * x[h+v], then fmaf(comb[t][k][j], r[t][k][h+v], .), k = 0..3
//   acc[t][n]  = fmaf chain over (s, j, v) in that lexicographic order, from 0
//   sqr[t]     = fmaf(new_r, new_r, .) chain over (s, j, v), from 0
//   warp value = xor-shuffle tree (off = 16, 8, 4, 2, 1) over the 32 lanes
//   Yp / Rp    = ((((0 + warp0) + warp1) + ...) + warp7)
// v3 keeps every one of those chains and trees; only WHO runs them changes:
//   - The CTA walks v1's 8 warps one at a time ("virtual warp" vw = 0..7) and,
//     inside vw, v1's s steps in order.  Lane l of EVERY physical warp plays
//     vtid = 32*vw + l, so the per-lane h positions (and the shuffle partners)
//     are v1's.
//   - pmap phase: physical warp p computes new_r for tokens p, p+8, ... (TM/8
//     of them) at vtid's h, all 4 j, with v1's expressions, writes the bf16
//     residual_out exactly as v1, runs the sqr chain, and stages new_r (fp32,
//     unrounded) in shared memory.  Each (t, vtid) new_r is computed once.
//   - GEMM phase: physical warp p = (token group a = p / NG, n group g = p % NG)
//     reads new_r of its TM/(8/NG) tokens from shared memory and its 24/NG
//     columns of W_T (volatile ld.global, prefetched one j ahead), and extends
//     acc[t][n] with v1's fmaf(new_r[v], w[v], acc) for v = 0..7 at each (s, j):
//     the same per-(t, n, vtid) chain, in the same order.
//   - At the end of vw, acc and sqr go through v1's xor-shuffle tree inside the
//     physical warp (lanes = vtids of vw) and lane 0 stores the warp value in
//     s_part[vw] / s_sqp[vw] (virtual warps with no valid h store 0, as v1's
//     idle warps do).  The final pass adds s_part[0..7] in order from 0.0f.
// TM need not be a multiple of 8: pmap slot i of warp p exists only when t = p + 8 i < TM
// (compile-time true when TM % 8 == 0, so those builds are unchanged); each token still runs
// v1's chains once, so the result stays bitwise equal.
// Constraints: N == 24, M * 4 * hidden < 2^31 (int32 offsets), TM % (8 / NG) == 0.
// Grid: (ceil(M / TM), 1, 1).  Block: 256 threads.
// Dynamic smem: 4 * TM * 32 lanes * 8 floats = TM * 4 KB (new_r of one vw step).
// ===================================================================
__device__ __forceinline__ uint4 mtile_v3_ld_stream(void const* p)
{
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                 : "l"(p));
    return r;
}

template <int TM, int NG, int MIN_BLOCKS>
__launch_bounds__(256, MIN_BLOCKS) __global__
    void fused_pmap_gemm_fma_mtile_v3(__nv_bfloat16 const* __restrict__ residual_in,
        __nv_bfloat16 const* __restrict__ x_in, float const* __restrict__ post_mix,
        float const* __restrict__ comb_mix, float const* __restrict__ W_T, float* __restrict__ Yp,
        float* __restrict__ Rp, int M, int hidden_size, int N, int K, __nv_bfloat16* __restrict__ residual_out)
{
    constexpr int HC_MULT = 4;
    constexpr int BLOCK_SIZE = 256;
    constexpr int WARP_SIZE = 32;
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    constexpr int BF16_VEC = 8;
    constexpr int N_TOTAL = 24;
    constexpr int H_STEP = BLOCK_SIZE * BF16_VEC; // v1's h stride per s step (2048)
    constexpr int VW_SPAN = WARP_SIZE * BF16_VEC; // h span of one v1 warp (256)
    static_assert(NUM_WARPS % NG == 0 && N_TOTAL % NG == 0, "NG must divide 8 and 24");
    constexpr int NT_R = (TM + NUM_WARPS - 1) / NUM_WARPS; // pmap token slots per thread
    constexpr bool TM_FULL = (TM % NUM_WARPS) == 0;        // every slot of every warp is a token
    constexpr int A_GROUPS = NUM_WARPS / NG;
    static_assert(TM % A_GROUPS == 0, "TM must divide into token groups");
    constexpr int TMT = TM / A_GROUPS;   // GEMM tokens per thread
    constexpr int TNT = N_TOTAL / NG;    // GEMM columns per thread
    constexpr unsigned FULL_WARP_MASK = 0xffffffff;

    int const tok0 = blockIdx.x * TM;
    int const tid = threadIdx.x;
    int const warp_id = tid / WARP_SIZE;
    int const lane = tid % WARP_SIZE;
    int const a_grp = warp_id / NG;
    int const n_grp = warp_id % NG;

    extern __shared__ float4 s_nr[]; // [HC_MULT][TM][2 halves][32 lanes]
    __shared__ float s_part[NUM_WARPS][TM][N_TOTAL];
    __shared__ float s_sqp[NUM_WARPS][TM];
    __shared__ float s_post[TM][HC_MULT];
    __shared__ float s_comb[TM][HC_MULT][HC_MULT];

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaGridDependencySynchronize();
#endif

    for (int i = tid; i < TM * HC_MULT; i += BLOCK_SIZE)
    {
        int const t = i / HC_MULT;
        int const tok = min(tok0 + t, M - 1);
        s_post[t][i % HC_MULT] = post_mix[tok * HC_MULT + i % HC_MULT];
    }
    for (int i = tid; i < TM * HC_MULT * HC_MULT; i += BLOCK_SIZE)
    {
        int const t = i / (HC_MULT * HC_MULT);
        int const rc = i % (HC_MULT * HC_MULT);
        int const tok = min(tok0 + t, M - 1);
        s_comb[t][rc / HC_MULT][rc % HC_MULT] = comb_mix[tok * HC_MULT * HC_MULT + rc];
    }
    // Virtual warps with no valid h keep 0 partials (v1's idle warps write 0).
    for (int i = tid; i < NUM_WARPS * TM * N_TOTAL; i += BLOCK_SIZE)
        (&s_part[0][0][0])[i] = 0.0f;
    for (int i = tid; i < NUM_WARPS * TM; i += BLOCK_SIZE)
        (&s_sqp[0][0])[i] = 0.0f;

    // pmap tokens of this thread: warp_id + 8 * i.  Tokens past M recompute M-1, never write.
    int tok_res[NT_R];
    bool tok_ok[NT_R];
#pragma unroll
    for (int i = 0; i < NT_R; i++)
    {
        int const t = warp_id + NUM_WARPS * i;
        tok_ok[i] = (tok0 + t) < M;
        tok_res[i] = min(tok0 + t, M - 1) * HC_MULT * hidden_size;
    }
    // Pmap slot i of this warp holds a CTA token (t = warp_id + 8 i < TM).
    auto slot_in = [&](int i) { return TM_FULL || (warp_id + NUM_WARPS * i) < TM; };

    uint4 raw[NT_R][HC_MULT + 1]; // slot k < 4: residual stream k; slot 4: x
    auto load_raw = [&](int hh)
    {
#pragma unroll
        for (int i = 0; i < NT_R; i++)
        {
            if (slot_in(i))
            {
#pragma unroll
                for (int k = 0; k < HC_MULT; k++)
                    raw[i][k] = mtile_v3_ld_stream(&residual_in[tok_res[i] + k * hidden_size + hh]);
                raw[i][HC_MULT] = mtile_v3_ld_stream(&x_in[(tok_res[i] >> 2) + hh]);
            }
        }
    };

    float const* const w_row = W_T + n_grp * TNT * K;
    float wcur[TNT][BF16_VEC];
    auto load_w = [&](float (&wd)[TNT][BF16_VEC], int hh, int jj)
    {
#pragma unroll
        for (int n = 0; n < TNT; n++)
        {
            float const* wp = w_row + n * K + jj * hidden_size + hh;
            asm volatile("ld.global.L1::evict_last.v4.f32 {%0, %1, %2, %3}, [%4];"
                         : "=f"(wd[n][0]), "=f"(wd[n][1]), "=f"(wd[n][2]), "=f"(wd[n][3])
                         : "l"(wp));
            asm volatile("ld.global.L1::evict_last.v4.f32 {%0, %1, %2, %3}, [%4];"
                         : "=f"(wd[n][4]), "=f"(wd[n][5]), "=f"(wd[n][6]), "=f"(wd[n][7])
                         : "l"(wp + 4));
        }
    };

    float acc[TMT][TNT];
    float sqr[NT_R];
#pragma unroll
    for (int i = 0; i < TMT; i++)
#pragma unroll
        for (int n = 0; n < TNT; n++)
            acc[i][n] = 0.0f;
#pragma unroll
    for (int i = 0; i < NT_R; i++)
        sqr[i] = 0.0f;

    int vw = 0;
    int s = 0;
    int h = lane * BF16_VEC;
    bool valid = h < hidden_size;
    bool const any_step = hidden_size > 0;
    if (valid)
    {
        load_raw(h);
        load_w(wcur, h, 0);
    }
    __syncthreads();

    while (any_step)
    {
        // ---- pmap phase: new_r of tokens warp_id + 8 i at (vw, s), all 4 j ----
        if (valid)
        {
#pragma unroll
            for (int i = 0; i < NT_R; i++)
            {
                if (!slot_in(i))
                    continue;
                int const t = warp_id + NUM_WARPS * i;
#pragma unroll
                for (int j = 0; j < HC_MULT; j++)
                {
                    float new_r[BF16_VEC];
                    float const pmj = s_post[t][j];
                    __nv_bfloat162 const* xp = reinterpret_cast<__nv_bfloat162 const*>(&raw[i][HC_MULT]);
#pragma unroll
                    for (int v = 0; v < 4; v++)
                    {
                        float2 f = __bfloat1622float2(xp[v]);
                        new_r[2 * v + 0] = pmj * f.x;
                        new_r[2 * v + 1] = pmj * f.y;
                    }
#pragma unroll
                    for (int k = 0; k < HC_MULT; k++)
                    {
                        float const ckj = s_comb[t][k][j];
                        __nv_bfloat162 const* rp = reinterpret_cast<__nv_bfloat162 const*>(&raw[i][k]);
#pragma unroll
                        for (int v = 0; v < 4; v++)
                        {
                            float2 f = __bfloat1622float2(rp[v]);
                            new_r[2 * v + 0] = fmaf(ckj, f.x, new_r[2 * v + 0]);
                            new_r[2 * v + 1] = fmaf(ckj, f.y, new_r[2 * v + 1]);
                        }
                    }
#pragma unroll
                    for (int v = 0; v < BF16_VEC; v++)
                        sqr[i] = fmaf(new_r[v], new_r[v], sqr[i]);
                    if (tok_ok[i])
                    {
                        uint4 packed;
                        __nv_bfloat162* pairs = reinterpret_cast<__nv_bfloat162*>(&packed);
#pragma unroll
                        for (int v = 0; v < 4; v++)
                            pairs[v] = __float22bfloat162_rn(make_float2(new_r[2 * v], new_r[2 * v + 1]));
                        *reinterpret_cast<uint4*>(residual_out + tok_res[i] + j * hidden_size + h) = packed;
                    }
                    s_nr[((j * TM + t) * 2 + 0) * WARP_SIZE + lane]
                        = make_float4(new_r[0], new_r[1], new_r[2], new_r[3]);
                    s_nr[((j * TM + t) * 2 + 1) * WARP_SIZE + lane]
                        = make_float4(new_r[4], new_r[5], new_r[6], new_r[7]);
                }
            }
        }

        // ---- next step (uniform across the CTA): v1's next s, else the next virtual warp ----
        int nvw = vw;
        int ns = s + 1;
        if (vw * VW_SPAN + ns * H_STEP >= hidden_size)
        {
            nvw = vw + 1;
            ns = 0;
        }
        bool const has_next = nvw < NUM_WARPS && nvw * VW_SPAN < hidden_size;
        int const nh = nvw * VW_SPAN + lane * BF16_VEC + ns * H_STEP;
        bool const nvalid = has_next && nh < hidden_size;
        if (nvalid)
            load_raw(nh); // raw[] is consumed: prefetch the next step's inputs
        __syncthreads();  // s_nr visible

        // ---- GEMM phase: acc[t][n] += new_r * W_T over j = 0..3, v = 0..7 ----
#pragma unroll 1
        for (int j = 0; j < HC_MULT; j++)
        {
            float wnxt[TNT][BF16_VEC];
            if (j + 1 < HC_MULT)
            {
                if (valid)
                    load_w(wnxt, h, j + 1);
            }
            else if (nvalid)
            {
                load_w(wnxt, nh, 0);
            }
            if (valid)
            {
#pragma unroll
                for (int i = 0; i < TMT; i++)
                {
                    int const t = a_grp * TMT + i;
                    float4 const lo = s_nr[((j * TM + t) * 2 + 0) * WARP_SIZE + lane];
                    float4 const hi = s_nr[((j * TM + t) * 2 + 1) * WARP_SIZE + lane];
                    float const nr[BF16_VEC] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
#pragma unroll
                    for (int v = 0; v < BF16_VEC; v++)
#pragma unroll
                        for (int n = 0; n < TNT; n++)
                            acc[i][n] = fmaf(nr[v], wcur[n][v], acc[i][n]);
                }
            }
#pragma unroll
            for (int n = 0; n < TNT; n++)
#pragma unroll
                for (int v = 0; v < BF16_VEC; v++)
                    wcur[n][v] = wnxt[n][v];
        }

        // ---- end of virtual warp vw: v1's xor-shuffle trees, warp value to smem ----
        if (nvw != vw || !has_next)
        {
#pragma unroll
            for (int i = 0; i < TMT; i++)
            {
#pragma unroll
                for (int n = 0; n < TNT; n++)
                {
#pragma unroll
                    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
                        acc[i][n] += __shfl_xor_sync(FULL_WARP_MASK, acc[i][n], off);
                    if (lane == 0)
                        s_part[vw][a_grp * TMT + i][n_grp * TNT + n] = acc[i][n];
                    acc[i][n] = 0.0f;
                }
            }
#pragma unroll
            for (int i = 0; i < NT_R; i++)
            {
                if (!slot_in(i))
                    continue;
#pragma unroll
                for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
                    sqr[i] += __shfl_xor_sync(FULL_WARP_MASK, sqr[i], off);
                if (lane == 0)
                    s_sqp[vw][warp_id + NUM_WARPS * i] = sqr[i];
                sqr[i] = 0.0f;
            }
        }
        __syncthreads(); // GEMM reads of s_nr done before the next pmap phase overwrites it
        if (!has_next)
            break;
        vw = nvw;
        s = ns;
        h = nh;
        valid = nvalid;
    }

    // ---- v1's sequential 8-warp sums ----
    for (int i = tid; i < TM * N_TOTAL; i += BLOCK_SIZE)
    {
        int const t = i / N_TOTAL;
        int const n = i % N_TOTAL;
        if (tok0 + t < M)
        {
            float val = 0.0f;
#pragma unroll
            for (int w = 0; w < NUM_WARPS; w++)
                val += s_part[w][t][n];
            Yp[(tok0 + t) * N + n] = val;
        }
    }
    if (tid < TM && tok0 + tid < M)
    {
        float sq = 0.0f;
#pragma unroll
        for (int w = 0; w < NUM_WARPS; w++)
            sq += s_sqp[w][tid];
        Rp[tok0 + tid] = sq;
    }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

} // namespace fused_fma_kernels
