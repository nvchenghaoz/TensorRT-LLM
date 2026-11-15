/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/common/config.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{
namespace cublas_lut
{

struct HashTuple
{
    size_t operator()(std::tuple<int32_t, int32_t, int32_t> const& x) const
    {
        return std::get<0>(x) ^ std::get<1>(x) ^ std::get<2>(x);
    }
};

// {mp2, k, n}: {algo, m_tile, m_stages, m_numsK, m_reduction, m_swizzle, m_custom, m_cga}
using AlgoListType = std::unordered_map<std::tuple<int32_t, int32_t, int32_t>, std::array<int, 8>, HashTuple>;

// {k, n}: entries {m_max, {algo, m_tile, m_stages, m_numsK, m_reduction, m_swizzle, m_custom, m_cga}} sorted by
// m_max. An m uses the first entry with m <= m_max; an algo id < 0 keeps the cuBLASLt heuristic for that range.
using RangeAlgoEntry = std::pair<int32_t, std::array<int, 8>>;
using RangeAlgoListType = std::map<std::pair<int32_t, int32_t>, std::vector<RangeAlgoEntry>>;

inline const AlgoListType spark_bf16_algo_list = {
    // llama 8b instruct fp16 decode
    // [-algo67 -m_tile6 -m_stages35 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom130 -m_mma0 -m_cga2 -m_scheduling1]
    {{8, 4096, 4096}, {67, 6, 35, 1, 0, 0, 130, 2}},
    // [-algo67 -m_tile393 -m_stages35 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom142 -m_mma0 -m_cga2 -m_scheduling1]
    {{8, 4096, 6144}, {67, 393, 35, 1, 0, 0, 142, 2}},
    // [-algo67 -m_tile393 -m_stages35 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom142 -m_mma0 -m_cga2 -m_scheduling1]
    {{8, 4096, 128256}, {67, 393, 35, 1, 0, 0, 142, 2}},

    // gpt-oss mxfp4-fp16 decode
    // [-algo67 -m_tile393 -m_stages35 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom142 -m_mma0 -m_cga2 -m_scheduling1]
    {{8, 2880, 201088}, {67, 393, 35, 1, 0, 0, 142, 2}},
    // [-algo14 -m_tile0 -m_stages35 -m_numsK10 -m_reduction2 -m_swizzle0 -m_custom0 -m_mma0 -m_cga0 -m_scheduling1]
    {{8, 2880, 32}, {14, 0, 0, 10, 2, 0, 0, 0}},
    // [-algo21 -m_tile11 -m_stages13 -m_numsK9 -m_reduction1 -m_swizzle0 -m_custom0 -m_mma0 -m_cga0 -m_scheduling1]
    //-k2880
    {{2048, 2880, 32}, {21, 11, 13, 9, 1, 0, 0, 0}},
    // [-algo21 -m_tile11 -m_stages19 -m_numsK11 -m_reduction1 -m_swizzle0 -m_custom0 -m_mma0 -m_cga0 -m_scheduling1]
    //-m_workmem1024 -k2880
    {{4096, 2880, 32}, {21, 11, 19, 11, 1, 0, 0, 0}},
    // [-algo23 -m_tile11 -m_stages8 -m_numsK2 -m_reduction1 -m_swizzle0 -m_custom0 -m_mma0 -m_cga0 -m_scheduling1]
    //-m_workmem1024 -k2880
    {{8, 2880, 5120}, {23, 11, 8, 2, 1, 0, 0, 0}},
    // [-algo21 -m_tile20 -m_stages15 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom0 -m_mma0 -m_cga0 -m_scheduling1]
    {{2048, 2880, 5120}, {21, 20, 15, 1, 0, 0, 0, 0}},
    // [-algo21 -m_tile20 -m_stages15 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom0 -m_mma0 -m_cga0 -m_scheduling1]
    {{4096, 2880, 5120}, {21, 20, 15, 1, 0, 0, 0, 0}},
    // [-algo23 -m_tile11 -m_stages14 -m_numsK24 -m_reduction1 -m_swizzle0 -m_custom0 -m_mma0 -m_cga0 -m_scheduling1]
    {{8, 4096, 2880}, {23, 11, 14, 24, 1, 0, 0, 0}},
    // [-algo21 -m_tile20 -m_stages15 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom0 -m_mma0 -m_cga0 -m_scheduling1]
    {{2048, 4096, 2880}, {21, 20, 15, 1, 0, 0, 0, 0}},
    // [-algo21 -m_tile20 -m_stages15 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom0 -m_mma0 -m_cga0 -m_scheduling1]
    {{4096, 4096, 2880}, {21, 20, 15, 1, 0, 0, 0, 0}},

    // nemotron super v3
    // [-algo67 -m_tile6 -m_stages37 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom16 -m_mma0 -m_cga0 -m_scheduling1]
    {{8, 1024, 4096}, {67, 6, 37, 1, 0, 0, 16, 0}},
    // [-algo67 -m_tile6 -m_stages37 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom93 -m_mma0 -m_cga0 -m_scheduling1]
    {{8, 4096, 512}, {67, 6, 37, 1, 0, 0, 93, 0}},
    // [-algo67 -m_tile10 -m_stages37 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom142 -m_mma0 -m_cga0 -m_scheduling1]
    {{8, 4096, 1024}, {67, 10, 37, 1, 0, 0, 142, 0}},
    // [-algo13 -m_tile0 -m_stages0 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom83 -m_mma0 -m_cga0 -m_scheduling1]
    {{8, 4096, 4096}, {13, 0, 0, 1, 0, 0, 83, 0}},
    // [-algo13 -m_tile0 -m_stages0 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom83 -m_mma0 -m_cga0 -m_scheduling1]
    {{8, 4096, 4608}, {13, 0, 0, 1, 0, 0, 83, 0}},
    // [-algo13 -m_tile0 -m_stages0 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom83 -m_mma0 -m_cga0 -m_scheduling1]
    {{8, 4096, 5376}, {13, 0, 0, 1, 0, 0, 83, 0}},
    // [-algo13 -m_tile0 -m_stages0 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom75 -m_mma0 -m_cga0 -m_scheduling1]
    {{8, 4096, 18560}, {13, 0, 0, 1, 0, 0, 75, 0}},
    // [-algo13 -m_tile0 -m_stages0 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom83 -m_mma0 -m_cga0 -m_scheduling1]
    {{8, 4096, 131072}, {13, 0, 0, 1, 0, 0, 83, 0}},
    // [-algo67 -m_tile3 -m_stages37 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom76 -m_mma0 -m_cga0 -m_scheduling1]
    {{8, 5376, 4096}, {67, 3, 37, 1, 0, 0, 76, 0}},
    // [-algo67 -m_tile3 -m_stages38 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom0 -m_mma0 -m_cga0 -m_scheduling1]
    {{8, 8192, 4096}, {67, 3, 38, 1, 0, 0, 0, 0}},
};

// bf16*bf16->fp32->bf16
inline const AlgoListType bf16_algo_list = {
    // Deepseek v3/R1 router gemm
    // [-algo66 -m_tile10 -m_stages35 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom3 -m_mma0 -m_cga2 -m_scheduling1]
    {{8, 7168, 256}, {66, 10, 35, 1, 0, 0, 3, 2}},
    {{512, 7168, 256}, {66, 48, 35, 1, 0, 0, 0, 2}},
    {{1024, 7168, 256}, {66, 13, 35, 1, 0, 0, 1, 3}},
};

// fp8*fp8->fp32->fp16
inline const AlgoListType fp8_algo_list = {
    // Llama-3.1-70B
    // [-algo66 -m_tile393 -m_stages36 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom5 -m_mma0 -m_cga2 -m_scheduling1]
    {{8, 8192, 8192}, {66, 393, 36, 1, 0, 0, 5, 2}},
    // [-algo66 -m_tile10 -m_stages36 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom1 -m_mma0 -m_cga2 -m_scheduling1]
    {{8, 8192, 57344}, {66, 10, 36, 1, 0, 0, 1, 2}},
    // Llama-3.3-70B TP4 (this is the default algo on B200. Here we aim to use the same algo on GB200.)
    // [-algo66 -m_tile393 -m_stages36 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom1 -m_mma0 -m_cga4 -m_scheduling1]
    {{8, 8192, 14336}, {66, 393, 36, 1, 0, 1, 1, 4}},
};

// fp8*fp8 on SM 10.3, keyed {m, k, n} with the exact m; looked up before fp8_algo_list
inline const AlgoListType sm103_fp8_exact_m_algo_list = {
    {{224, 4096, 2560}, {66, 321, 36, 1, 0, 0, 3, 3}},
    {{280, 4096, 2560}, {66, 324, 36, 1, 0, 0, 1, 3}},
    {{336, 4096, 2560}, {66, 328, 36, 1, 0, 0, 3, 3}},
    {{392, 4096, 2560}, {66, 331, 36, 1, 0, 0, 3, 3}},
    {{448, 4096, 2560}, {66, 333, 36, 1, 0, 0, 1, 3}},
    {{504, 4096, 2560}, {66, 19, 36, 1, 0, 0, 1, 3}},
    {{560, 4096, 2560}, {66, 20, 36, 1, 0, 0, 1, 4}},
    {{616, 4096, 2560}, {66, 412, 36, 1, 0, 0, 1, 4}},
    {{672, 4096, 2560}, {66, 30, 36, 1, 0, 0, 0, 6}},
    {{728, 4096, 2560}, {66, 118, 36, 1, 0, 0, 1, 3}},
    {{784, 4096, 2560}, {66, 29, 36, 1, 0, 0, 3, 3}},
    {{840, 4096, 2560}, {66, 119, 36, 1, 0, 0, 1, 3}},
    {{896, 4096, 2560}, {66, 119, 36, 1, 0, 0, 1, 3}},
};

// bf16*bf16->fp32->bf16 on SM 12.0 (RTX PRO 6000), keyed {k, n} with M ranges; looked up before bf16_algo_list.
// Gemma-4-31B attention projections (sliding/global qkv incl. the k_eq_v V-dedup width, sliding/global o_proj),
// cuBLASLt 13.8. Tuned on cold-L2 CUDA-event sweeps over every config the cuBLASLt heuristic returns at any M
// (experiments/ascent/sm120_bf16_lut/sweep3.py): exact entries at the token counts the serving benchmark forms
// (multiples of 300/1000/2000/4000/8000 up to 8192) and 64-row intervals above 1280. No in-place (atomic) split-K.
// modeling_gemma4.py routes only the token counts covered here; everything else keeps F.linear.
inline const RangeAlgoListType sm120_bf16_range_algo_list = {
    // sl_qkv: N=16384 K=5376
    {{5376, 16384},
        {
            {299, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {300, {67, 467, 35, 1, 0, 0, 30, 0}}, // a67/192x176/64xAUTO/sk1/r0/sw0/c30/g0
            {899, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {900, {67, 24, 35, 1, 0, 0, 31, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c31/g2
            {999, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {1000, {67, 24, 35, 1, 0, 0, 31, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c31/g2
            {1199, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {1200, {67, 409, 35, 1, 0, 0, 30, 0}}, // a67/128x176/64xAUTO/sk1/r0/sw0/c30/g0
            {1279, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {1344, {67, 20, 35, 1, 0, 0, 23, 0}}, // a67/128x128/64xAUTO/sk1/r0/sw0/c23/g0
            {1408, {67, 24, 35, 1, 0, 0, 31, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c31/g2
            {1536, {67, 202, 35, 1, 0, 0, 15, 0}}, // a67/176x192/64xAUTO/sk1/r0/sw0/c15/g0
            {1600, {67, 29, 35, 1, 0, 0, 23, 0}}, // a67/128x160/64xAUTO/sk1/r0/sw0/c23/g0
            {1728, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {1792, {67, 24, 35, 1, 0, 0, 31, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c31/g2
            {1920, {67, 202, 35, 1, 0, 0, 15, 0}}, // a67/176x192/64xAUTO/sk1/r0/sw0/c15/g0
            {2048, {67, 412, 35, 1, 0, 0, 31, 0}}, // a67/128x208/64xAUTO/sk1/r0/sw0/c31/g0
            {2100, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {2240, {67, 24, 35, 1, 0, 0, 31, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c31/g2
            {2304, {67, 202, 35, 1, 0, 0, 15, 0}}, // a67/176x192/64xAUTO/sk1/r0/sw0/c15/g0
            {2496, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {2624, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {2688, {67, 202, 35, 1, 0, 0, 15, 0}}, // a67/176x192/64xAUTO/sk1/r0/sw0/c15/g0
            {2700, {67, 412, 35, 1, 0, 0, 31, 0}}, // a67/128x208/64xAUTO/sk1/r0/sw0/c31/g0
            {2880, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {2999, {67, 24, 35, 1, 0, 0, 31, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c31/g2
            {3072, {67, 202, 35, 1, 0, 0, 15, 0}}, // a67/176x192/64xAUTO/sk1/r0/sw0/c15/g0
            {3264, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {3392, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {3456, {67, 202, 35, 1, 0, 0, 15, 0}}, // a67/176x192/64xAUTO/sk1/r0/sw0/c15/g0
            {3599, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {3648, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {3776, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {3840, {67, 202, 35, 1, 0, 0, 15, 0}}, // a67/176x192/64xAUTO/sk1/r0/sw0/c15/g0
            {3900, {67, 412, 35, 1, 0, 0, 31, 0}}, // a67/128x208/64xAUTO/sk1/r0/sw0/c31/g0
            {4160, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {4224, {67, 202, 35, 1, 0, 0, 15, 0}}, // a67/176x192/64xAUTO/sk1/r0/sw0/c15/g0
            {4416, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {4499, {67, 24, 35, 1, 0, 0, 31, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c31/g2
            {4608, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {4928, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {4992, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {5312, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {5376, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {5696, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {5760, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {5888, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {5952, {67, 24, 35, 1, 0, 0, 31, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c31/g2
            {6080, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {6144, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {6299, {67, 24, 35, 1, 0, 0, 145, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c145/g2
            {6528, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {6784, {67, 24, 35, 1, 0, 0, 145, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c145/g2
            {7104, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {8192, {67, 24, 35, 1, 0, 0, 145, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c145/g2
        }},
    // gl_qkv: N=20480 K=5376
    {{5376, 20480},
        {
            {599, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {600, {67, 412, 35, 1, 0, 0, 31, 0}}, // a67/128x208/64xAUTO/sk1/r0/sw0/c31/g0
            {899, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {900, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {999, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {1000, {67, 463, 35, 1, 0, 0, 30, 0}}, // a67/192x144/64xAUTO/sk1/r0/sw0/c30/g0
            {1199, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {1200, {67, 467, 35, 1, 0, 0, 30, 0}}, // a67/192x176/64xAUTO/sk1/r0/sw0/c30/g0
            {1279, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {1344, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {1408, {67, 24, 35, 1, 0, 0, 31, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c31/g2
            {1536, {67, 202, 35, 1, 0, 0, 15, 0}}, // a67/176x192/64xAUTO/sk1/r0/sw0/c15/g0
            {1664, {67, 412, 35, 1, 0, 0, 31, 0}}, // a67/128x208/64xAUTO/sk1/r0/sw0/c31/g0
            {1799, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {1920, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {2000, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {2099, {67, 24, 35, 1, 0, 0, 31, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c31/g2
            {2112, {67, 202, 35, 1, 0, 0, 15, 0}}, // a67/176x192/64xAUTO/sk1/r0/sw0/c15/g0
            {2368, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {2432, {67, 467, 35, 1, 0, 0, 30, 0}}, // a67/192x176/64xAUTO/sk1/r0/sw0/c30/g0
            {2699, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {2700, {67, 463, 35, 1, 0, 0, 30, 0}}, // a67/192x144/64xAUTO/sk1/r0/sw0/c30/g0
            {2880, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {2999, {67, 24, 35, 1, 0, 0, 31, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c31/g2
            {3072, {67, 202, 35, 1, 0, 0, 15, 0}}, // a67/176x192/64xAUTO/sk1/r0/sw0/c15/g0
            {3136, {67, 24, 35, 1, 0, 0, 145, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c145/g2
            {3264, {67, 24, 35, 1, 0, 0, 31, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c31/g2
            {3328, {67, 467, 35, 1, 0, 0, 30, 0}}, // a67/192x176/64xAUTO/sk1/r0/sw0/c30/g0
            {3599, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {3648, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {3840, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {3899, {67, 24, 35, 1, 0, 0, 31, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c31/g2
            {4032, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {4096, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {4160, {67, 24, 35, 1, 0, 0, 31, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c31/g2
            {4224, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {4416, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {4480, {67, 24, 35, 1, 0, 0, 31, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c31/g2
            {4499, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {4608, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {4672, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {4736, {67, 24, 35, 1, 0, 0, 31, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c31/g2
            {4799, {67, 24, 35, 1, 0, 0, 145, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c145/g2
            {5568, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {5696, {67, 24, 35, 1, 0, 0, 145, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c145/g2
            {5760, {67, 202, 35, 1, 0, 0, 15, 0}}, // a67/176x192/64xAUTO/sk1/r0/sw0/c15/g0
            {5952, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {6016, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {6080, {67, 24, 35, 1, 0, 0, 31, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c31/g2
            {6144, {67, 202, 35, 1, 0, 0, 15, 0}}, // a67/176x192/64xAUTO/sk1/r0/sw0/c15/g0
            {6336, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {6464, {67, 24, 35, 1, 0, 0, 145, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c145/g2
            {6720, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {7199, {67, 24, 35, 1, 0, 0, 145, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c145/g2
            {7296, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {7488, {67, 24, 35, 1, 0, 0, 145, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c145/g2
            {7680, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {7872, {67, 24, 35, 1, 0, 0, 145, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c145/g2
            {8064, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {8099, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {8100, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {8192, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
        }},
    // sl_o: N=5376 K=8192
    {{8192, 5376},
        {
            {299, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {300, {67, 29, 35, 2, 2, 0, 23, 0}}, // a67/128x160/64xAUTO/sk2/r2/sw0/c23/g0
            {599, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {600, {67, 29, 35, 1, 0, 0, 22, 0}}, // a67/128x160/64xAUTO/sk1/r0/sw0/c22/g0
            {899, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {900, {67, 24, 35, 1, 0, 0, 144, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c144/g2
            {999, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {1000, {67, 24, 35, 1, 0, 0, 144, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c144/g2
            {1728, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {1856, {67, 463, 35, 1, 0, 0, 30, 0}}, // a67/192x144/64xAUTO/sk1/r0/sw0/c30/g0
            {2099, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {2112, {67, 24, 35, 1, 0, 0, 31, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c31/g2
            {2176, {67, 24, 35, 1, 0, 0, 30, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c30/g2
            {2240, {67, 467, 35, 1, 0, 0, 30, 0}}, // a67/192x176/64xAUTO/sk1/r0/sw0/c30/g0
            {2304, {67, 202, 35, 1, 0, 0, 15, 0}}, // a67/176x192/64xAUTO/sk1/r0/sw0/c15/g0
            {2496, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {2560, {67, 152, 35, 1, 0, 0, 22, 0}}, // a67/96x256/64xAUTO/sk1/r0/sw0/c22/g0
            {2624, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {2700, {67, 412, 35, 1, 0, 0, 31, 0}}, // a67/128x208/64xAUTO/sk1/r0/sw0/c31/g0
            {2880, {67, 463, 35, 1, 0, 0, 30, 0}}, // a67/192x144/64xAUTO/sk1/r0/sw0/c30/g0
            {2944, {67, 409, 35, 1, 0, 0, 30, 0}}, // a67/128x176/64xAUTO/sk1/r0/sw0/c30/g0
            {2999, {67, 24, 35, 1, 0, 0, 145, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c145/g2
            {3136, {67, 24, 35, 1, 0, 0, 31, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c31/g2
            {3264, {67, 24, 35, 1, 0, 0, 145, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c145/g2
            {3299, {67, 24, 35, 1, 0, 0, 31, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c31/g2
            {3300, {67, 24, 35, 1, 0, 0, 145, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c145/g2
            {3328, {67, 24, 35, 1, 0, 0, 31, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c31/g2
            {3392, {67, 467, 35, 1, 0, 0, 30, 0}}, // a67/192x176/64xAUTO/sk1/r0/sw0/c30/g0
            {3584, {67, 29, 35, 1, 0, 0, 23, 0}}, // a67/128x160/64xAUTO/sk1/r0/sw0/c23/g0
            {3840, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {3904, {67, 497, 35, 1, 0, 0, 30, 0}}, // a67/256x112/64xAUTO/sk1/r0/sw0/c30/g0
            {3968, {67, 20, 35, 1, 0, 0, 23, 0}}, // a67/128x128/64xAUTO/sk1/r0/sw0/c23/g0
            {3999, {67, 495, 35, 1, 0, 0, 30, 0}}, // a67/256x96/64xAUTO/sk1/r0/sw0/c30/g0
            {4000, {67, 29, 35, 1, 0, 0, 23, 0}}, // a67/128x160/64xAUTO/sk1/r0/sw0/c23/g0
            {4032, {67, 495, 35, 1, 0, 0, 30, 0}}, // a67/256x96/64xAUTO/sk1/r0/sw0/c30/g0
            {4160, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {4480, {67, 24, 35, 1, 0, 0, 145, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c145/g2
            {4499, {67, 412, 35, 1, 0, 0, 31, 0}}, // a67/128x208/64xAUTO/sk1/r0/sw0/c31/g0
            {4500, {67, 467, 35, 1, 0, 0, 30, 0}}, // a67/192x176/64xAUTO/sk1/r0/sw0/c30/g0
            {4544, {67, 412, 35, 1, 0, 0, 31, 0}}, // a67/128x208/64xAUTO/sk1/r0/sw0/c31/g0
            {4608, {67, 202, 35, 1, 0, 0, 15, 0}}, // a67/176x192/64xAUTO/sk1/r0/sw0/c15/g0
            {4736, {67, 463, 35, 1, 0, 0, 30, 0}}, // a67/192x144/64xAUTO/sk1/r0/sw0/c30/g0
            {4799, {67, 497, 35, 1, 0, 0, 30, 0}}, // a67/256x112/64xAUTO/sk1/r0/sw0/c30/g0
            {4800, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {4928, {67, 497, 35, 1, 0, 0, 30, 0}}, // a67/256x112/64xAUTO/sk1/r0/sw0/c30/g0
            {4992, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {5056, {67, 495, 35, 1, 0, 0, 30, 0}}, // a67/256x96/64xAUTO/sk1/r0/sw0/c30/g0
            {5120, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {5184, {67, 412, 35, 1, 0, 0, 31, 0}}, // a67/128x208/64xAUTO/sk1/r0/sw0/c31/g0
            {5312, {67, 24, 35, 1, 0, 0, 145, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c145/g2
            {5568, {67, 24, 35, 1, 0, 0, 144, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c144/g2
            {5696, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {5760, {67, 463, 35, 1, 0, 0, 30, 0}}, // a67/192x144/64xAUTO/sk1/r0/sw0/c30/g0
            {5888, {67, 497, 35, 1, 0, 0, 30, 0}}, // a67/256x112/64xAUTO/sk1/r0/sw0/c30/g0
            {5952, {67, 495, 35, 1, 0, 0, 30, 0}}, // a67/256x96/64xAUTO/sk1/r0/sw0/c30/g0
            {6336, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {6528, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {6592, {67, 24, 35, 1, 0, 0, 144, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c144/g2
            {6720, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {6848, {67, 24, 35, 1, 0, 0, 144, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c144/g2
            {7040, {67, 467, 35, 1, 0, 0, 30, 0}}, // a67/192x176/64xAUTO/sk1/r0/sw0/c30/g0
            {7680, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {7999, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {8192, {67, 467, 35, 1, 0, 0, 30, 0}}, // a67/192x176/64xAUTO/sk1/r0/sw0/c30/g0
        }},
    // gl_o: N=5376 K=16384
    {{16384, 5376},
        {
            {899, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {900, {67, 24, 35, 1, 0, 0, 144, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c144/g2
            {999, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {1000, {67, 24, 35, 1, 0, 0, 144, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c144/g2
            {1279, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {1280, {67, 28, 35, 1, 0, 0, 22, 0}}, // a67/96x128/64xAUTO/sk1/r0/sw0/c22/g0
            {1344, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {1408, {67, 30, 35, 1, 0, 0, 22, 0}}, // a67/160x128/64xAUTO/sk1/r0/sw0/c22/g0
            {1499, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {1500, {67, 20, 35, 1, 0, 0, 23, 0}}, // a67/128x128/64xAUTO/sk1/r0/sw0/c23/g0
            {1728, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {1856, {67, 463, 35, 1, 0, 0, 30, 0}}, // a67/192x144/64xAUTO/sk1/r0/sw0/c30/g0
            {1999, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {2000, {67, 24, 35, 1, 0, 0, 145, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c145/g2
            {2099, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {2112, {67, 24, 35, 1, 0, 0, 145, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c145/g2
            {2176, {67, 24, 35, 1, 0, 0, 31, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c31/g2
            {2240, {67, 467, 35, 1, 0, 0, 30, 0}}, // a67/192x176/64xAUTO/sk1/r0/sw0/c30/g0
            {2304, {67, 202, 35, 1, 0, 0, 15, 0}}, // a67/176x192/64xAUTO/sk1/r0/sw0/c15/g0
            {2496, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {2624, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {2700, {67, 412, 35, 1, 0, 0, 31, 0}}, // a67/128x208/64xAUTO/sk1/r0/sw0/c31/g0
            {2880, {67, 463, 35, 1, 0, 0, 30, 0}}, // a67/192x144/64xAUTO/sk1/r0/sw0/c30/g0
            {2944, {67, 409, 35, 1, 0, 0, 30, 0}}, // a67/128x176/64xAUTO/sk1/r0/sw0/c30/g0
            {3136, {67, 23, 35, 1, 0, 0, 22, 0}}, // a67/128x256/64xAUTO/sk1/r0/sw0/c22/g0
            {3264, {67, 24, 35, 1, 0, 0, 145, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c145/g2
            {3299, {67, 23, 35, 1, 0, 0, 22, 0}}, // a67/128x256/64xAUTO/sk1/r0/sw0/c22/g0
            {3300, {67, 24, 35, 1, 0, 0, 145, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c145/g2
            {3328, {67, 23, 35, 1, 0, 0, 22, 0}}, // a67/128x256/64xAUTO/sk1/r0/sw0/c22/g0
            {3392, {67, 467, 35, 1, 0, 0, 30, 0}}, // a67/192x176/64xAUTO/sk1/r0/sw0/c30/g0
            {3456, {67, 29, 35, 1, 0, 0, 22, 0}}, // a67/128x160/64xAUTO/sk1/r0/sw0/c22/g0
            {3520, {67, 467, 35, 1, 0, 0, 30, 0}}, // a67/192x176/64xAUTO/sk1/r0/sw0/c30/g0
            {3840, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {3904, {67, 497, 35, 1, 0, 0, 30, 0}}, // a67/256x112/64xAUTO/sk1/r0/sw0/c30/g0
            {3999, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {4000, {67, 23, 35, 1, 0, 0, 22, 0}}, // a67/128x256/64xAUTO/sk1/r0/sw0/c22/g0
            {4032, {67, 495, 35, 1, 0, 0, 30, 0}}, // a67/256x96/64xAUTO/sk1/r0/sw0/c30/g0
            {4096, {67, 23, 35, 1, 0, 0, 22, 0}}, // a67/128x256/64xAUTO/sk1/r0/sw0/c22/g0
            {4199, {67, 495, 35, 1, 0, 0, 30, 0}}, // a67/256x96/64xAUTO/sk1/r0/sw0/c30/g0
            {4200, {67, 24, 35, 1, 0, 0, 31, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c31/g2
            {4224, {67, 23, 35, 1, 0, 0, 22, 0}}, // a67/128x256/64xAUTO/sk1/r0/sw0/c22/g0
            {4480, {67, 24, 35, 1, 0, 0, 31, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c31/g2
            {4544, {67, 467, 35, 1, 0, 0, 30, 0}}, // a67/192x176/64xAUTO/sk1/r0/sw0/c30/g0
            {4608, {67, 202, 35, 1, 0, 0, 15, 0}}, // a67/176x192/64xAUTO/sk1/r0/sw0/c15/g0
            {4736, {67, 463, 35, 1, 0, 0, 30, 0}}, // a67/192x144/64xAUTO/sk1/r0/sw0/c30/g0
            {4992, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {4999, {67, 495, 35, 1, 0, 0, 30, 0}}, // a67/256x96/64xAUTO/sk1/r0/sw0/c30/g0
            {5000, {67, 152, 35, 1, 0, 0, 22, 0}}, // a67/96x256/64xAUTO/sk1/r0/sw0/c22/g0
            {5056, {67, 495, 35, 1, 0, 0, 30, 0}}, // a67/256x96/64xAUTO/sk1/r0/sw0/c30/g0
            {5120, {67, 152, 35, 1, 0, 0, 22, 0}}, // a67/96x256/64xAUTO/sk1/r0/sw0/c22/g0
            {5312, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {5376, {67, 23, 35, 1, 0, 0, 22, 0}}, // a67/128x256/64xAUTO/sk1/r0/sw0/c22/g0
            {5440, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {5504, {67, 24, 35, 1, 0, 0, 31, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c31/g2
            {5568, {67, 23, 35, 1, 0, 0, 22, 0}}, // a67/128x256/64xAUTO/sk1/r0/sw0/c22/g0
            {5696, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {5760, {67, 463, 35, 1, 0, 0, 30, 0}}, // a67/192x144/64xAUTO/sk1/r0/sw0/c30/g0
            {5824, {67, 414, 35, 1, 0, 0, 22, 0}}, // a67/128x224/64xAUTO/sk1/r0/sw0/c22/g0
            {5888, {67, 497, 35, 1, 0, 0, 30, 0}}, // a67/256x112/64xAUTO/sk1/r0/sw0/c30/g0
            {6336, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {6400, {67, 412, 35, 1, 0, 0, 30, 0}}, // a67/128x208/64xAUTO/sk1/r0/sw0/c30/g0
            {6464, {67, 23, 35, 1, 0, 0, 22, 0}}, // a67/128x256/64xAUTO/sk1/r0/sw0/c22/g0
            {6592, {67, 463, 35, 1, 0, 0, 30, 0}}, // a67/192x144/64xAUTO/sk1/r0/sw0/c30/g0
            {6720, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {6848, {67, 24, 35, 1, 0, 0, 31, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c31/g2
            {7040, {67, 467, 35, 1, 0, 0, 30, 0}}, // a67/192x176/64xAUTO/sk1/r0/sw0/c30/g0
            {7680, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {7936, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {7999, {67, 23, 35, 1, 0, 0, 22, 0}}, // a67/128x256/64xAUTO/sk1/r0/sw0/c22/g0
            {8192, {67, 467, 35, 1, 0, 0, 30, 0}}, // a67/192x176/64xAUTO/sk1/r0/sw0/c30/g0
        }},
    // gl_qkv_dedup: N=18432 K=5376
    {{5376, 18432},
        {
            {599, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {600, {67, 24, 35, 1, 0, 0, 145, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c145/g2
            {899, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {900, {67, 202, 35, 1, 0, 0, 15, 0}}, // a67/176x192/64xAUTO/sk1/r0/sw0/c15/g0
            {999, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {1000, {67, 412, 35, 1, 0, 0, 31, 0}}, // a67/128x208/64xAUTO/sk1/r0/sw0/c31/g0
            {1199, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {1200, {67, 24, 35, 1, 0, 0, 31, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c31/g2
            {1279, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {1280, {67, 24, 35, 1, 0, 0, 31, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c31/g2
            {1344, {67, 202, 35, 1, 0, 0, 15, 0}}, // a67/176x192/64xAUTO/sk1/r0/sw0/c15/g0
            {1408, {67, 497, 35, 1, 0, 0, 30, 0}}, // a67/256x112/64xAUTO/sk1/r0/sw0/c30/g0
            {1472, {67, 29, 35, 1, 0, 0, 23, 0}}, // a67/128x160/64xAUTO/sk1/r0/sw0/c23/g0
            {1600, {67, 24, 35, 1, 0, 0, 31, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c31/g2
            {1728, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {1800, {67, 412, 35, 1, 0, 0, 31, 0}}, // a67/128x208/64xAUTO/sk1/r0/sw0/c31/g0
            {1984, {67, 24, 35, 1, 0, 0, 31, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c31/g2
            {1999, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {2000, {67, 414, 35, 1, 0, 0, 22, 0}}, // a67/128x224/64xAUTO/sk1/r0/sw0/c22/g0
            {2112, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {2240, {67, 24, 35, 1, 0, 0, 31, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c31/g2
            {2368, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {2496, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {2624, {67, 24, 35, 1, 0, 0, 31, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c31/g2
            {2688, {67, 202, 35, 1, 0, 0, 15, 0}}, // a67/176x192/64xAUTO/sk1/r0/sw0/c15/g0
            {2700, {67, 412, 35, 1, 0, 0, 31, 0}}, // a67/128x208/64xAUTO/sk1/r0/sw0/c31/g0
            {2880, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {2999, {67, 24, 35, 1, 0, 0, 31, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c31/g2
            {3072, {67, 202, 35, 1, 0, 0, 15, 0}}, // a67/176x192/64xAUTO/sk1/r0/sw0/c15/g0
            {3136, {67, 24, 35, 1, 0, 0, 31, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c31/g2
            {3264, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {3328, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {3392, {67, 24, 35, 1, 0, 0, 31, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c31/g2
            {3599, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {3648, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {3712, {67, 412, 35, 1, 0, 0, 31, 0}}, // a67/128x208/64xAUTO/sk1/r0/sw0/c31/g0
            {3900, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {3999, {67, 24, 35, 1, 0, 0, 31, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c31/g2
            {4032, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {4288, {67, 24, 35, 1, 0, 0, 31, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c31/g2
            {4416, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {4672, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {4800, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {4928, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {4999, {67, 24, 35, 1, 0, 0, 31, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c31/g2
            {5184, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {5312, {67, 24, 35, 1, 0, 0, 31, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c31/g2
            {5568, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {5632, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {5696, {67, 24, 35, 1, 0, 0, 145, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c145/g2
            {5952, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {6080, {-1, 0, 0, 0, 0, 0, 0, 0}}, // cuBLASLt heuristic
            {6336, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {6400, {67, 24, 35, 1, 0, 0, 145, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c145/g2
            {6899, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {6900, {67, 24, 35, 1, 0, 0, 145, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c145/g2
            {7104, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {7360, {67, 24, 35, 1, 0, 0, 145, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c145/g2
            {7488, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {7744, {67, 24, 35, 1, 0, 0, 145, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c145/g2
            {7872, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
            {7999, {67, 24, 35, 1, 0, 0, 145, 2}}, // a67/256x128/64xAUTO/sk1/r0/sw0/c145/g2
            {8192, {67, 209, 35, 1, 0, 0, 31, 0}}, // a67/192x192/64xAUTO/sk1/r0/sw0/c31/g0
        }},
};

} // namespace cublas_lut
} // namespace torch_ext

TRTLLM_NAMESPACE_END
