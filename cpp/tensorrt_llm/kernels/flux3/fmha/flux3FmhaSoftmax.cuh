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

// The softmax and the O normalizer of the FLUX.3 FP8 attention (flux3FmhaG3Kernel.cuh): the e4m3 / fp32-accumulator
// fmha::Softmax and fmha::Tile_o_normalizer with a base-2 softmax. Only what differs from the stock classes is
// redefined here; everything else (masking, the sum reduction, update_sum, final_update) is inherited unchanged.
//
// - unpack keeps the raw FP32 QK accumulators (scale_bmm1 > 0, so the row max of the raw values is the row max of the
//   scaled ones); the scale s = scale_bmm1 is folded into the exponent:
//     p = exp2(fma(x, s * log2e, log2(256) - max * s * log2e)),   alpha = exp2((prev_max - max) * s * log2e).
//   Not bitwise the stock expf path (reordered); no attention sinks, no LSE.
// - the row max reduces as a balanced fmaxf tree (depth log2(32) instead of a 31-long chain; fmaxf is exact and order
//   free, so the max is bitwise the chain's). The row sum keeps the stock linear order.
// - P converts to E4M3 without the multiply by scale_softmax, which must therefore be exactly 1.0f (the launcher
//   checks it).
// - the O rescale is skipped by a warp whose rows all keep their running max (alpha == 1.0f; x * 1.0f == x).
//
// The launcher compiles them with --use_fast_math, which the outputs require (exp2f is ex2.approx.ftz, 1 / x is the
// approximate reciprocal).

#include <fused_multihead_attention.h>
#include <fused_multihead_flash_attention_kernel_noloop_tiled.h>

#include <cfloat>

namespace fused_multihead_attention
{
namespace flux3
{

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, typename Kernel_traits>
struct Softmax : public fmha::Softmax<fmha::Ada_qmma_e4m3_fp32_traits, Cta_tile, Kernel_traits>
{
    // The traits.
    using Traits = fmha::Ada_qmma_e4m3_fp32_traits;
    // The base class: the stock e4m3 / fp32 softmax.
    using Base = fmha::Softmax<Traits, Cta_tile, Kernel_traits>;

    // The MMAs.
    static constexpr int MMAS_M = Base::MMAS_M;
    static constexpr int MMAS_N = Base::MMAS_N;

    // The accumulators.
    using Accumulator = fmha::Fragment_accumulator<Traits>;

    static_assert(Cta_tile::WARPS_N == 1, "the row reductions are warp-local (reduce_Nx1)");
    static_assert(((MMAS_N * 4) & (MMAS_N * 4 - 1)) == 0, "the max tree needs a power-of-two row slice");

    // Ctor.
    template <typename Params>
    inline __device__ Softmax(Params const& params, void* smem, int bidb, int tidx)
        : Base(params, smem, bidb, tidx)
    {
    }

    // Convert from accumulators to floats, unscaled (the scale is applied in apply_exp_with_mask).
    inline __device__ void unpack(Accumulator const (&acc)[MMAS_M][MMAS_N])
    {
        float const scale = 1.f;
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {
                this->elt_[2 * mi + 0][4 * ni + 0] = acc[mi][ni].elt(0) * scale;
                this->elt_[2 * mi + 0][4 * ni + 1] = acc[mi][ni].elt(1) * scale;
                this->elt_[2 * mi + 0][4 * ni + 2] = acc[mi][ni].elt(4) * scale;
                this->elt_[2 * mi + 0][4 * ni + 3] = acc[mi][ni].elt(5) * scale;
                this->elt_[2 * mi + 1][4 * ni + 0] = acc[mi][ni].elt(2) * scale;
                this->elt_[2 * mi + 1][4 * ni + 1] = acc[mi][ni].elt(3) * scale;
                this->elt_[2 * mi + 1][4 * ni + 2] = acc[mi][ni].elt(6) * scale;
                this->elt_[2 * mi + 1][4 * ni + 3] = acc[mi][ni].elt(7) * scale;
            }
        }
    }

    // exp(s * x - (s * max - ln Q)) == exp2(fma(x, s * log2e, log2 Q - max * s * log2e)), Q = the P quant scale.
    template <bool APPLY_MASK = false>
    inline __device__ void apply_exp_with_mask(float const (&max)[MMAS_M * 2])
    {
        float const s_log2e = reinterpret_cast<float const&>(this->params_scale_bmm1_) * 1.4426950408889634f;
        float const log2_quant = log2f(Traits::SOFTMAX_FP_QUANT_SCALE);
#pragma unroll
        for (int mi = 0; mi < MMAS_M * 2; ++mi)
        {
            float const offset = APPLY_MASK && max[mi] == -FLT_MAX ? 0.f : fmaf(-max[mi], s_log2e, log2_quant);
#pragma unroll
            for (int ni = 0; ni < MMAS_N * 4; ++ni)
            {
                this->elt_[mi][ni] = exp2f(fmaf(this->elt_[mi][ni], s_log2e, offset));
            }
        }
    }

    // The warp-wide row reduction: the max as a balanced tree inside the thread, the sum as the stock chain.
    template <typename Functor>
    inline __device__ void reduce(float (&dst)[MMAS_M * 2])
    {
        if constexpr (Functor::IS_SUM)
        {
            this->template reduce_Nx1<Functor>(dst);
        }
        else
        {
#pragma unroll
            for (int mi = 0; mi < MMAS_M * 2; ++mi)
            {
                float t[MMAS_N * 4];
#pragma unroll
                for (int ni = 0; ni < MMAS_N * 4; ++ni)
                {
                    t[ni] = this->elt_[mi][ni];
                }
#pragma unroll
                for (int stride = MMAS_N * 2; stride >= 1; stride /= 2)
                {
#pragma unroll
                    for (int ni = 0; ni < stride; ++ni)
                    {
                        t[ni] = Functor::apply(t[ni], t[ni + stride]);
                    }
                }
                dst[mi] = t[0];
            }

// Then across each group of 4 threads, as reduce_Nx1.
#pragma unroll
            for (int mi = 0; mi < MMAS_M * 2; ++mi)
            {
                dst[mi] = Functor::apply(dst[mi], __shfl_xor_sync(uint32_t(-1), dst[mi], 1));
                dst[mi] = Functor::apply(dst[mi], __shfl_xor_sync(uint32_t(-1), dst[mi], 2));
            }
        }
    }

    // Pack P to E4M3 A-fragments for BMM2, without the multiply by scale_softmax (== 1.0f).
    template <typename Fragment_a, int K, int M>
    inline __device__ void pack(Fragment_a (&dst)[K][M]) const
    {
        float const scale = 1.f;

// The canonical layout in K should be R0: [0,1,2,3] R2: [16,17,18,19]
// Note below that this is not possible with the register layout of the accumulator.
#pragma unroll
        for (int mi = 0; mi < M; ++mi)
        {
#pragma unroll
            for (int ki = 0; ki < K; ++ki)
            {
                // 1st row - 8 elements per row.
                float tmp_00 = this->elt_[2 * mi + 0][8 * ki + 0] * scale; // + 0
                float tmp_01 = this->elt_[2 * mi + 0][8 * ki + 1] * scale; // + 1
                float tmp_02 = this->elt_[2 * mi + 0][8 * ki + 2] * scale; // + 8
                float tmp_03 = this->elt_[2 * mi + 0][8 * ki + 3] * scale; // + 9
                float tmp_04 = this->elt_[2 * mi + 0][8 * ki + 4] * scale; // +16
                float tmp_05 = this->elt_[2 * mi + 0][8 * ki + 5] * scale; // +17
                float tmp_06 = this->elt_[2 * mi + 0][8 * ki + 6] * scale; // +24
                float tmp_07 = this->elt_[2 * mi + 0][8 * ki + 7] * scale; // +25

                // 2nd row - 4 elements per row.
                float tmp_10 = this->elt_[2 * mi + 1][8 * ki + 0] * scale; // + 0
                float tmp_11 = this->elt_[2 * mi + 1][8 * ki + 1] * scale; // + 1
                float tmp_12 = this->elt_[2 * mi + 1][8 * ki + 2] * scale; // + 8
                float tmp_13 = this->elt_[2 * mi + 1][8 * ki + 3] * scale; // + 9
                float tmp_14 = this->elt_[2 * mi + 1][8 * ki + 4] * scale; // +16
                float tmp_15 = this->elt_[2 * mi + 1][8 * ki + 5] * scale; // +17
                float tmp_16 = this->elt_[2 * mi + 1][8 * ki + 6] * scale; // +24
                float tmp_17 = this->elt_[2 * mi + 1][8 * ki + 7] * scale; // +25

                // Pack to 4 registers.
                dst[ki][mi].reg(0) = fmha::float4_to_fp8x4<Traits::A_type>(tmp_00, tmp_01, tmp_02, tmp_03);
                dst[ki][mi].reg(1) = fmha::float4_to_fp8x4<Traits::A_type>(tmp_10, tmp_11, tmp_12, tmp_13);
                dst[ki][mi].reg(2) = fmha::float4_to_fp8x4<Traits::A_type>(tmp_04, tmp_05, tmp_06, tmp_07);
                dst[ki][mi].reg(3) = fmha::float4_to_fp8x4<Traits::A_type>(tmp_14, tmp_15, tmp_16, tmp_17);
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile>
struct Tile_o_normalizer : public fmha::Tile_o_normalizer<fmha::Ada_qmma_e4m3_fp32_traits, Cta_tile>
{
    // The traits.
    using Traits = fmha::Ada_qmma_e4m3_fp32_traits;
    // The base class: the stock e4m3 / fp32 normalizer (its update_sum and final_update are used unchanged).
    using Base = fmha::Tile_o_normalizer<Traits, Cta_tile>;

    // The row maxima are raw (unscaled) QK accumulators, so the rescale factor is
    // exp(s * (prev - cur)) = exp2((prev - cur) * s * log2e).
    template <typename Params, typename Block_info>
    inline __device__ Tile_o_normalizer(Params const& params, Block_info const& binfo)
        : Base(params, binfo)
    {
        uint32_t const bits = params.scale_bmm1_d ? *params.scale_bmm1_d : params.scale_bmm1;
        s_log2e_ = reinterpret_cast<float const&>(bits) * 1.4426950408889634f;
    }

    // Update o.
    inline __device__ void update(typename Base::Fragment_accu (&acc_o)[Base::MMAS_M][Base::MMAS_N],
        float (&curr_max)[Base::ROWS_PER_THREAD], float const (&prev_max)[Base::ROWS_PER_THREAD],
        float (&sum)[Base::ROWS_PER_THREAD])
    {
#pragma unroll
        for (int mi = 0; mi < Base::MMAS_M; ++mi)
        {
            float alpha[2];
#pragma unroll
            for (int ii = 0; ii < 2; ++ii)
            {
                int jj = 2 * mi + ii;
                curr_max[jj] = fmax(prev_max[jj], curr_max[jj]);
                alpha[ii] = exp2f((prev_max[jj] - curr_max[jj]) * s_log2e_);
                sum[jj] *= alpha[ii];
            }
            // x * 1.0f == x bitwise, so a warp whose rows all keep their max skips the O rescale. update() runs
            // converged (after the max reduce's shuffles).
            if (!__all_sync(0xffffffffu, alpha[0] == 1.f && alpha[1] == 1.f))
            {
#pragma unroll
                for (int ni = 0; ni < Base::MMAS_N; ++ni)
                {
#pragma unroll
                    for (int ii = 0; ii < Base::REGS_PER_THREAD; ++ii)
                    {
                        acc_o[mi][ni].elt(ii) = alpha[(ii & 2) / 2] * acc_o[mi][ni].elt(ii);
                    }
                }
            }
        }
    }

    // scale_bmm1 * log2(e).
    float s_log2e_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace flux3
} // namespace fused_multihead_attention
