/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef MARLIN_NAMESPACE_NAME
#define MARLIN_NAMESPACE_NAME marlin
#endif

#define MARLIN_DECLARE_SINGLE_EXPERT_KERNEL
#include "marlin_nvfp4.h"
#include "marlin_nvfp4_template.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace marlin
{

using namespace marlin_nvfp4_dispatch;

__global__ void MarlinDefault(MARLIN_KERNEL_PARAMS){};

using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);

// Single-expert shared-memory size (no block-meta overhead).
int get_kernel_cache_size(thread_config_t const& th_config, int thread_m_blocks, int prob_n, int prob_k, int num_bits,
    int group_size, int stages)
{
    int pack_factor = 32 / num_bits;
    int tb_k = th_config.thread_k;
    int tb_n = th_config.thread_n;
    int tb_m = thread_m_blocks * 16;
    int sh_a_size = stages * (tb_m * tb_k) * 2;
    int sh_b_size = stages * (tb_k * tb_n / pack_factor) * 4;
    int sh_red_size = tb_m * (tb_n + 8) * 2;
    int sh_bias_size = tb_n * 2;
    int tmp_size = (sh_b_size > sh_red_size ? sh_red_size : sh_b_size) + sh_bias_size;
    tmp_size = std::max(std::max(sh_b_size, sh_red_size), tmp_size);
    int sh_s_size = get_scales_cache_size(th_config, prob_n, prob_k, num_bits, group_size, stages);
    return tmp_size + sh_a_size + sh_s_size;
}

bool is_valid_config(thread_config_t const& th_config, int thread_m_blocks, int prob_n, int prob_k, int num_bits,
    int group_size, int stages, int max_shared_mem)
{
    if (!is_config_feasible(th_config, prob_n, prob_k))
        return false;
    return get_kernel_cache_size(th_config, thread_m_blocks, prob_n, prob_k, num_bits, group_size, stages)
        <= max_shared_mem;
}

MarlinFuncPtr get_marlin_kernel(int thread_m_blocks, int thread_n_blocks, int thread_k_blocks, bool m_block_size_8,
    int group_blocks, int threads, int stages)
{
#define MARLIN_KERNEL_MATCH(T, M, N, K, M8)                                                                            \
    (threads == (T) && thread_m_blocks == (M) && thread_n_blocks == (N) && thread_k_blocks == (K)                      \
        && m_block_size_8 == (M8) && stages == 4 && group_blocks == 1)
#define MARLIN_KERNEL_IF(T, M, N, K, M8)                                                                               \
    if (MARLIN_KERNEL_MATCH(T, M, N, K, M8))                                                                           \
        return Marlin<nv_bfloat16, T, M, N, K, M8, 4, 1>;

    MARLIN_KERNEL_IF(256, 1, 8, 8, true)
    MARLIN_KERNEL_IF(128, 1, 8, 4, true)
    MARLIN_KERNEL_IF(128, 1, 4, 8, true)
    MARLIN_KERNEL_IF(256, 1, 8, 8, false)
    MARLIN_KERNEL_IF(128, 1, 8, 4, false)
    MARLIN_KERNEL_IF(128, 1, 4, 8, false)
    MARLIN_KERNEL_IF(256, 2, 16, 4, false)
    MARLIN_KERNEL_IF(128, 2, 8, 4, false)
    MARLIN_KERNEL_IF(128, 2, 4, 8, false)
    MARLIN_KERNEL_IF(256, 3, 16, 4, false)
    MARLIN_KERNEL_IF(128, 3, 8, 4, false)
    MARLIN_KERNEL_IF(128, 3, 4, 8, false)
    MARLIN_KERNEL_IF(256, 4, 16, 4, false)
    MARLIN_KERNEL_IF(128, 4, 8, 4, false)
    MARLIN_KERNEL_IF(128, 4, 4, 8, false)

#undef MARLIN_KERNEL_MATCH
#undef MARLIN_KERNEL_IF
    return MarlinDefault;
}

// Deeper-pipeline single-m-block 16-row kernels (decode rows 9-16), see get_m16_launch.
MarlinFuncPtr get_m16_kernel(thread_config_t const& th, int stages)
{
#define MARLIN_M16_IF(T, N, K, S)                                                                                      \
    if (th.num_threads == (T) && th.thread_n == (N) * 16 && th.thread_k == (K) * 16 && stages == (S))                  \
        return Marlin<nv_bfloat16, T, 1, N, K, false, S, 1>;
    MARLIN_M16_IF(128, 4, 8, 4)
    MARLIN_M16_IF(128, 4, 8, 6)
    MARLIN_M16_IF(128, 4, 8, 8)
    MARLIN_M16_IF(128, 8, 4, 4)
    MARLIN_M16_IF(128, 8, 4, 6)
    MARLIN_M16_IF(128, 8, 4, 8)
    MARLIN_M16_IF(256, 8, 8, 4)
    MARLIN_M16_IF(256, 8, 8, 6)
#undef MARLIN_M16_IF
    return MarlinDefault;
}

struct m16_launch_t
{
    thread_config_t th;
    int blocks;
    int stages;
};

struct m16_table_entry_t
{
    int prob_n;
    int prob_k;
    m16_launch_t launch;
};

// Decode rows 9-16 (one 16-row m block) launch shapes per (N, K), chosen by a cold-L2
// CUDA-graph sweep on SM120 (RTX PRO 6000, 188 SMs). Exact (N, K) match only.
m16_table_entry_t const kM16Table[] = {
    {5120, 6144, {{128, 64, 128}, 80, 8}},
    {16384, 5120, {{64, 128, 128}, 128, 6}},
    {14336, 5120, {{64, 128, 128}, 112, 6}},
};

// TRTLLM_MARLIN_M16_CFG="thread_k,thread_n,threads,blocks,stages" overrides every rows 9-16 call
// (sweeps); TRTLLM_MARLIN_M16_TABLE=0 disables the table. Both are read per call (uncached).
bool get_m16_launch(int prob_n, int prob_k, int sms, m16_launch_t& out)
{
    char const* cfg = std::getenv("TRTLLM_MARLIN_M16_CFG");
    if (cfg != nullptr && cfg[0] != '\0')
    {
        m16_launch_t o{{-1, -1, -1}, 0, 0};
        if (std::sscanf(cfg, "%d,%d,%d,%d,%d", &o.th.thread_k, &o.th.thread_n, &o.th.num_threads, &o.blocks,
                &o.stages)
            == 5)
        {
            out = o;
            return true;
        }
        return false;
    }
    char const* table = std::getenv("TRTLLM_MARLIN_M16_TABLE");
    if (table != nullptr && table[0] == '0' && table[1] == '\0')
        return false;
    if (sms != 188)
        return false;
    for (auto const& e : kM16Table)
    {
        if (e.prob_n == prob_n && e.prob_k == prob_k)
        {
            out = e.launch;
            return true;
        }
    }
    return false;
}

exec_config_t determine_exec_config(int prob_m, int prob_n, int prob_k, int thread_m_blocks, bool m_block_size_8,
    int num_bits, int group_size, int stages, int max_shared_mem, int sms)
{
    exec_config_t exec_cfg{1, {-1, -1, -1}};
    thread_config_t const* cfgs = thread_m_blocks > 1 ? kLargeBatchConfigs : kSmallBatchConfigs;
    int cfg_count = thread_m_blocks > 1 ? kLargeBatchConfigCount : kSmallBatchConfigCount;

    for (int i = 0; i < cfg_count; i++)
    {
        thread_config_t th = cfgs[i];
        if (!is_valid_config(th, thread_m_blocks, prob_n, prob_k, num_bits, group_size, stages, max_shared_mem - 512))
            continue;
        int group_blocks = group_size == -1 ? -1 : group_size / 16;
        auto kernel = get_marlin_kernel(
            thread_m_blocks, th.thread_n / 16, th.thread_k / 16, m_block_size_8, group_blocks, th.num_threads, stages);
        if (kernel == MarlinDefault)
            continue;
        return {1, th};
    }
    return exec_cfg;
}

// Full-K launch (kernel template flag full_k, see kMarlinNvfp4FullKGridShift):
// TRTLLM_MARLIN_FULLK, read per call; "0" restores the split-K launch.
bool marlin_full_k_enabled()
{
    char const* v = std::getenv("TRTLLM_MARLIN_FULLK");
    return !(v != nullptr && v[0] == '0' && v[1] == '\0');
}

// Dynamic shared memory the full_k kernel addresses: the kernel's own layout for
// BF16 A, 8-row m block, no act_order / zero points, plus the stripe stash.
int full_k_smem_bytes(int threads, int thread_n_blocks, int thread_k_blocks, int group_blocks, int stages)
{
    int const b_sh_stage = ((thread_n_blocks * 16) * 16 / 8 / 4) * thread_k_blocks;
    int const sh_red = (2 * thread_n_blocks + 1) * 16;
    int const sh_b = stages * b_sh_stage;
    int const sh_bias = thread_n_blocks * 16 / 8;
    int const red_b = std::max(std::max(sh_red, sh_b), std::min(sh_red, sh_b) + sh_bias);
    int const s_tb_groups = group_blocks < thread_k_blocks ? thread_k_blocks / group_blocks : 1;
    int const s_sh_stage = s_tb_groups * thread_n_blocks;
    int const a_sh_stage = (16 * thread_k_blocks / 8) * 8;
    int const int4s = red_b + stages * s_sh_stage + stages * a_sh_stage;
    return int4s * 16 + kMarlinNvfp4FullKMaxStripes * 16 * threads * static_cast<int>(sizeof(float));
}

// Full-K launch of the in_proj decode shapes (n = 16384 / 14336, k = 5120):
// TRTLLM_MARLIN_INPROJ_FULLK, read per call; "0" restores the split-K launch.
bool marlin_inproj_full_k_enabled()
{
    char const* v = std::getenv("TRTLLM_MARLIN_INPROJ_FULLK");
    return !(v != nullptr && v[0] == '0' && v[1] == '\0');
}

// Largest number of split-K stripes the default `grid`-CTA launch gives one column.
int full_k_max_stripes(int k_tiles, int n_tiles, int grid)
{
    int const iters = div_ceil(k_tiles * n_tiles, grid);
    int most = 0;
    for (int c = 0; c < n_tiles; c++)
    {
        most = std::max(most, ((c + 1) * k_tiles - 1) / iters - (c * k_tiles) / iters + 1);
    }
    return most;
}

void marlin_mm_nvfp4(void const* A, void const* B, void* C, void* C_tmp, void const* b_s, void const* g_s, int prob_m,
    int prob_n, int prob_k, int* locks, int num_groups, int group_size, bool use_fp32_reduce, int dev,
    cudaStream_t stream, bool enable_pdl)
{
    constexpr int num_bits = 4;

    int group_blocks = group_size == -1 ? -1 : group_size / 16;

    int4 const* A_ptr = (int4 const*) A;
    int4 const* B_ptr = (int4 const*) B;
    int4* C_ptr = (int4*) C;
    int4* C_tmp_ptr = (int4*) C_tmp;
    int4 const* b_s_ptr = (int4 const*) b_s;
    uint16_t const* g_s_ptr = (uint16_t const*) g_s;

    int max_shared_mem = 0;
    cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);

    int stages = 4;
    int sms = -1;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);

    int max_par_val = 16;
    if (prob_n <= 4096)
        max_par_val = 16 * 8;
    int max_shared_mem_new = max_shared_mem;
    int rest_m = prob_m;
    int max_thread_m_blocks = 4;
    int lda = prob_k;

    while (rest_m)
    {
        int par_count = std::min(rest_m / (max_thread_m_blocks * 16), max_par_val);
        int prob_m_split = par_count > 0 ? (par_count * (max_thread_m_blocks * 16)) : rest_m;

        int thread_m_blocks = std::min(div_ceil(prob_m_split, 16), max_thread_m_blocks);
        bool m_block_size_8 = prob_m_split <= 8;

        exec_config_t exec_cfg = determine_exec_config(prob_m_split, prob_n, prob_k, thread_m_blocks, m_block_size_8,
            num_bits, group_size, stages, max_shared_mem, sms);
        thread_config_t thread_tfg = exec_cfg.tb_cfg;

        if (thread_tfg.thread_k == -1 && max_thread_m_blocks > 1)
        {
            max_thread_m_blocks--;
            continue;
        }

        if (thread_tfg.thread_k == -1)
        {
            break;
        }

        // Small wave optimization
        if (thread_tfg.thread_n != -1)
        {
            if (prob_n / thread_tfg.thread_n * div_ceil(prob_m_split, thread_m_blocks * 16) * 4 <= sms)
            {
                if (is_valid_config({128, 64, 128}, thread_m_blocks, prob_n, prob_k, num_bits, group_size, stages,
                        max_shared_mem_new))
                {
                    thread_tfg = {128, 64, 128};
                    exec_cfg = {1, thread_tfg};
                }
            }
        }

        int num_threads = thread_tfg.num_threads;
        int thread_k = thread_tfg.thread_k;
        int thread_n = thread_tfg.thread_n;
        int blocks = sms * exec_cfg.blocks_per_sm;
        if (exec_cfg.blocks_per_sm > 1)
            max_shared_mem_new = max_shared_mem / exec_cfg.blocks_per_sm - 1024;

        int thread_k_blocks = thread_k / 16;
        int thread_n_blocks = thread_n / 16;

        auto kernel = get_marlin_kernel(
            thread_m_blocks, thread_n_blocks, thread_k_blocks, m_block_size_8, group_blocks, num_threads, stages);

        if (kernel == MarlinDefault)
        {
            break;
        }

        // GDN out_proj / attention o_proj (n=5120, k=6144) at decode rows <= 8: one CTA
        // per 64-wide column with an 8-stage pipeline instead of `blocks` split-K CTAs
        // with 4 stages. The kernel replays the split-K launch's reduction chain, so the
        // output is bitwise equal to it.
        int grid = blocks;
        int full_k_flags = 0;
        if (m_block_size_8 && thread_m_blocks == 1 && num_threads == 128 && thread_n == 64 && thread_k == 128
            && group_blocks == 1 && prob_n == 5120 && prob_k == 6144 && exec_cfg.blocks_per_sm == 1
            && !use_fp32_reduce && blocks <= kMarlinNvfp4FullKGridMask && prob_n / thread_n <= blocks
            && max_shared_mem_new < (1 << kMarlinNvfp4FullKGridShift)
            && full_k_max_stripes(prob_k / thread_k, prob_n / thread_n, blocks) <= kMarlinNvfp4FullKMaxStripes
            && full_k_smem_bytes(num_threads, thread_n_blocks, thread_k_blocks, group_blocks, 8)
                <= max_shared_mem_new - 512
            && marlin_full_k_enabled())
        {
            kernel = Marlin<nv_bfloat16, 128, 1, 4, 8, true, 8, 1, true>;
            grid = prob_n / thread_n;
            full_k_flags = blocks << kMarlinNvfp4FullKGridShift;
        }

        // GDN in_proj_qkvz (n=16384) / attention qkv_proj (n=14336), k=5120, at decode
        // rows <= 8: one CTA per 128-wide column (128 threads, {64k, 128n}, 6 stages)
        // streaming the whole K (iters == k_tiles, slice_count 1, no lock use) instead
        // of `blocks` split-K CTAs whose partial sums chain through the BF16 output.
        // Reordered: each column's fp32 sum is rounded to BF16 once.
        thread_config_t const in_proj_cfg = {64, 128, 128};
        if (full_k_flags == 0 && prob_m <= 8 && m_block_size_8 && thread_m_blocks == 1 && group_blocks == 1
            && exec_cfg.blocks_per_sm == 1 && !use_fp32_reduce && prob_k == 5120
            && (prob_n == 16384 || prob_n == 14336) && prob_n / in_proj_cfg.thread_n <= blocks
            && is_valid_config(
                in_proj_cfg, thread_m_blocks, prob_n, prob_k, num_bits, group_size, 6, max_shared_mem_new - 512)
            && marlin_inproj_full_k_enabled())
        {
            kernel = Marlin<nv_bfloat16, 128, 1, 8, 4, true, 6, 1>;
            num_threads = in_proj_cfg.num_threads;
            grid = prob_n / in_proj_cfg.thread_n;
        }

        // Decode rows 9-16 (one 16-row m block): per-(N, K) thread config, grid and
        // cp.async stages (kM16Table). With grid == n_tiles each CTA streams one full-K
        // column (no split-K BF16 chain through C, no stripe wrap). Reordered.
        m16_launch_t m16;
        if (full_k_flags == 0 && !m_block_size_8 && thread_m_blocks == 1 && prob_m_split > 8 && group_blocks == 1
            && exec_cfg.blocks_per_sm == 1 && !use_fp32_reduce && get_m16_launch(prob_n, prob_k, sms, m16)
            && m16.blocks >= 1 && m16.blocks <= sms
            && is_valid_config(
                m16.th, thread_m_blocks, prob_n, prob_k, num_bits, group_size, m16.stages, max_shared_mem_new - 512)
            && get_m16_kernel(m16.th, m16.stages) != MarlinDefault)
        {
            kernel = get_m16_kernel(m16.th, m16.stages);
            num_threads = m16.th.num_threads;
            grid = m16.blocks;
        }

        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem_new);

        // The kernel starts with griddepcontrol.wait, so a PDL launch orders all of
        // its memory accesses (A, scales, locks, C) after the predecessor grid; that
        // also serializes the launches of this loop, which share `locks`. Only a PDL
        // launch (flag in the last argument) triggers dependents early.
        int4 const* b_bias_null = nullptr;
        float const* a_scales_null = nullptr;
        int4 const* zp_null = nullptr;
        int const* g_idx_null = nullptr;
        // clang-format off
        tensorrt_llm::common::launchWithPdl("marlin_nvfp4", enable_pdl, kernel, dim3(grid), dim3(num_threads),
            static_cast<size_t>(max_shared_mem_new), stream,
            A_ptr, B_ptr, C_ptr, C_tmp_ptr,
            b_bias_null,    // b_bias
            a_scales_null,  // a_scales
            b_s_ptr, g_s_ptr,
            zp_null,        // zp
            g_idx_null,     // g_idx
            num_groups, prob_m_split, prob_n, prob_k, lda, locks,
            false,  // has_bias
            false,  // use_atomic_add
            use_fp32_reduce, max_shared_mem_new | full_k_flags | (enable_pdl ? kMarlinNvfp4PdlLaunchFlag : 0));
        // clang-format on

        A_ptr += prob_m_split * (lda / 8);
        C_ptr += prob_m_split * (prob_n / 8);
        rest_m -= prob_m_split;
    }
}

// Explicit template instantiations for BF16 + NVFP4 Marlin kernels.
// clang-format off
template __global__ void Marlin<nv_bfloat16, 256, 1, 8, 8, true,  4, 1>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<nv_bfloat16, 128, 1, 8, 4, true,  4, 1>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<nv_bfloat16, 128, 1, 4, 8, true,  4, 1>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<nv_bfloat16, 256, 1, 8, 8, false, 4, 1>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<nv_bfloat16, 128, 1, 8, 4, false, 4, 1>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<nv_bfloat16, 128, 1, 4, 8, false, 4, 1>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<nv_bfloat16, 256, 2, 16, 4, false, 4, 1>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<nv_bfloat16, 128, 2, 8, 4, false, 4, 1>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<nv_bfloat16, 128, 2, 4, 8, false, 4, 1>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<nv_bfloat16, 256, 3, 16, 4, false, 4, 1>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<nv_bfloat16, 128, 3, 8, 4, false, 4, 1>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<nv_bfloat16, 128, 3, 4, 8, false, 4, 1>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<nv_bfloat16, 256, 4, 16, 4, false, 4, 1>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<nv_bfloat16, 128, 4, 8, 4, false, 4, 1>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<nv_bfloat16, 128, 4, 4, 8, false, 4, 1>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<nv_bfloat16, 128, 1, 4, 8, true,  8, 1, true>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<nv_bfloat16, 128, 1, 8, 4, true,  6, 1>( MARLIN_KERNEL_PARAMS );
// Decode rows 9-16 deeper-pipeline variants (get_m16_kernel).
template __global__ void Marlin<nv_bfloat16, 128, 1, 4, 8, false, 6, 1>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<nv_bfloat16, 128, 1, 4, 8, false, 8, 1>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<nv_bfloat16, 128, 1, 8, 4, false, 6, 1>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<nv_bfloat16, 128, 1, 8, 4, false, 8, 1>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<nv_bfloat16, 256, 1, 8, 8, false, 6, 1>( MARLIN_KERNEL_PARAMS );

// clang-format on

// FP4 E2M1 -> BF16 activation dequant: out[i] = fp4_to_bf16(act[i]) *
// block_scale[i/16] * global_scale. Block scales are FP8 E4M3 (swizzled).
__global__ void dequant_fp4_act_kernel(uint8_t const* __restrict__ act_fp4, // [M, K/2] packed FP4
    uint8_t const* __restrict__ act_sf,                                     // FP8 E4M3 block scales (swizzled)
    float const* __restrict__ alpha,                                        // global scale
    nv_bfloat16* __restrict__ out,                                          // [M, K] BF16 output
    int M, int K)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = M * (K / 2);
    if (idx >= total_pairs)
        return;

    float global_s = *alpha;
    int row = idx / (K / 2);
    int col_pair = idx % (K / 2);

    uint8_t packed = act_fp4[idx];

    // Unpack two FP4 E2M1 values (low nibble first)
    auto fp4_to_float = [](uint8_t nibble) -> float
    {
        // FP4 E2M1: 1 sign + 2 exponent + 1 mantissa
        uint8_t sign = (nibble >> 3) & 1;
        uint8_t exp = (nibble >> 1) & 0x3;
        uint8_t mant = nibble & 1;
        float val;
        if (exp == 0)
        {
            // subnormal: (-1)^s * 0.mantissa * 2^(1-bias) = (-1)^s * mant * 0.5
            val = mant * 0.5f;
        }
        else
        {
            // normal: (-1)^s * 1.mantissa * 2^(exp-bias), bias=1
            val = (1.0f + mant * 0.5f) * (float) (1 << (exp - 1));
        }
        return sign ? -val : val;
    };

    float v0 = fp4_to_float(packed & 0x0F);
    float v1 = fp4_to_float((packed >> 4) & 0x0F);

    // Block scale: one FP8 E4M3 per 16 FP4 elements = per 8 bytes
    // The scale layout is swizzled 128x4 — for now use linear indexing
    // as a reasonable approximation. TODO: handle swizzled layout properly.
    int elem0 = col_pair * 2;
    int scale_idx = row * (K / 16) + elem0 / 16;
    uint8_t sf_byte = act_sf[scale_idx];
    // FP8 E4M3 -> float: reinterpret as __nv_fp8_e4m3
    __nv_fp8_e4m3 sf_fp8 = *reinterpret_cast<__nv_fp8_e4m3 const*>(&sf_byte);
    float sf = float(sf_fp8);

    float scale = sf * global_s;
    int out_idx = row * K + col_pair * 2;
    out[out_idx] = __float2bfloat16(v0 * scale);
    out[out_idx + 1] = __float2bfloat16(v1 * scale);
}

} // namespace marlin

namespace marlin_nvfp4
{

void dequantFp4Activations(
    void const* act_fp4, void const* act_sf, float const* alpha, void* act_bf16, int m, int k, cudaStream_t stream)
{

    int total_pairs = m * (k / 2);
    int threads = 256;
    int blocks = (total_pairs + threads - 1) / threads;
    ::marlin::dequant_fp4_act_kernel<<<blocks, threads, 0, stream>>>(
        (uint8_t const*) act_fp4, (uint8_t const*) act_sf, alpha, (nv_bfloat16*) act_bf16, m, k);
}

void marlinNvfp4Gemm(void const* act_bf16, void const* weight, void* output, void* C_tmp, void const* weight_sf,
    void const* global_scale_bf16, int m, int n, int k, int* workspace, int num_groups, int group_size,
    bool use_fp32_reduce, cudaStream_t stream, bool enable_pdl)
{
    int const sm = tensorrt_llm::common::getSMVersion();
    TLLM_CHECK_WITH_INFO(isMarlinNvfp4DenseSmSupported(sm),
        "Marlin NVFP4 GEMM is only supported on SM89, SM90-99, and SM120/121; current SM = %d", sm);

    int dev;
    cudaGetDevice(&dev);

    // griddepcontrol needs SM90+; on SM89 launch without the PDL attribute.
    bool const pdl = enable_pdl && sm >= 90;
    ::marlin::marlin_mm_nvfp4(act_bf16, weight, output, C_tmp, weight_sf, global_scale_bf16, m, n, k, workspace,
        num_groups, group_size, use_fp32_reduce, dev, stream, pdl);
}

} // namespace marlin_nvfp4
