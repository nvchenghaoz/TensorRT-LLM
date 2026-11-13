/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Bitwise-FA2 forward for SM120 at higher occupancy.
 *
 * This compiles PyTorch's own FlashAttention-2 forward (the sources torch was built from,
 * third_party/flash-attention/csrc/flash_attn/src, with torch's compile definitions:
 * FLASHATTENTION_DISABLE_ALIBI/SOFTCAP, UNFUSE_FMA, no fast-math) and only changes the
 * CTA tile along the query rows:
 *   bulk: Flash_fwd_kernel_traits<128, 256, 64, 8>  (8 warps, 96 KB smem, 1 CTA/SM)
 *   tail: Flash_fwd_kernel_traits<128,  64, 64, 4>  (rows left after whole bulk waves)
 * torch's SM120 dispatch picks <128, 128, 64, 4> (4 warps, 1 CTA/SM). FA2's TiledMma lays
 * warps along M only, so each query row runs the same instructions in the same order
 * (same kBlockN=64 key blocks in the same reverse order, same softmax and epilogue):
 * the output is bitwise equal to torch SDPA's flash kernel.
 */
#include "namespace_config.h"
#include "flash.h"
#include "flash_fwd_launch_template.h"

#include <algorithm>

namespace FLASH_NAMESPACE {

using TraitsBulk = Flash_fwd_kernel_traits<128, 256, 64, 8, false, false, cutlass::bfloat16_t>;
using TraitsTail = Flash_fwd_kernel_traits<128, 64, 64, 4, false, false, cutlass::bfloat16_t>;

struct RowMap {
    int g;     // heads [0, g) use (cntA, loA); heads [g, h) use (cntB, loB)
    int cntA;  // row blocks per head in group A
    int loA;   // first row block of group A
    int cntB;
    int loB;
};

// Same template arguments torch's run_flash_fwd picks for this call (non-causal, no dropout,
// d == 128, seqlen not a multiple of the block): Is_even_MN=false, Is_even_K=true, rest false.
// CTA i maps to (head, m_block) head-major with m fastest, the order of torch's grid.
template <typename KT>
__global__ void flash_fwd_mapped_kernel(KERNEL_PARAM_MODIFIER const Flash_fwd_params params, const RowMap map) {
#if defined(ARCH_SUPPORTS_FLASH)
    const int i = blockIdx.x;
    const int nA = map.g * map.cntA;
    int bidh, m_block;
    if (i < nA) {
        bidh = i / map.cntA;
        m_block = map.loA + (i - bidh * map.cntA);
    } else {
        const int j = i - nA;
        const int hh = j / map.cntB;
        bidh = map.g + hh;
        m_block = map.loB + (j - hh * map.cntB);
    }
    compute_attn_1rowblock<KT, false, false, false, false, false, true, false, false>(params, blockIdx.y, bidh, m_block);
#endif
}

template <typename KT>
static void launch_mapped(const Flash_fwd_params &params, cudaStream_t stream, const RowMap &map, int nctas) {
    constexpr size_t smem_size = KT::kSmemSize;
    auto kernel = &flash_fwd_mapped_kernel<KT>;
    if (smem_size >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    dim3 grid(nctas, params.b, 1);
    kernel<<<grid, KT::kNThreads, smem_size, stream>>>(params, map);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static int bulk_slots(int dev) {
    static int cached_dev = -1;
    static int slots = 0;
    if (cached_dev != dev) {
        int num_sms = 0, occ = 0;
        C10_CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev));
        constexpr size_t smem_size = TraitsBulk::kSmemSize;
        auto kernel = &flash_fwd_mapped_kernel<TraitsBulk>;
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kernel, TraitsBulk::kNThreads, smem_size));
        slots = std::max(1, occ * num_sms);
        cached_dev = dev;
    }
    return slots;
}

// Bulk row blocks run in whole waves of the 256-row tile; the rows left over run as 64-row
// tiles in a second launch on the same stream, unless the leftover is more than
// max_rem_frac of a wave (then one launch of the 256-row tile covers everything).
// Returns the number of launches.
int run_fa2_sm120(Flash_fwd_params &params, cudaStream_t stream, float max_rem_frac) {
    int dev = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    const int H = params.h, S = params.seqlen_q;
    const int nb = (S + TraitsBulk::kBlockM - 1) / TraitsBulk::kBlockM;
    const int total = H * nb;
    const int slots = bulk_slots(dev);
    const int bulk = params.b == 1 ? (total / slots) * slots : total;
    const int rem = total - bulk;
    if (bulk == 0 || rem == 0 || float(rem) > max_rem_frac * float(slots)) {
        launch_mapped<TraitsBulk>(params, stream, RowMap{H, nb, 0, 0, 0}, total);
        return 1;
    }
    const int base = bulk / H, extra = bulk % H;
    launch_mapped<TraitsBulk>(params, stream, RowMap{extra, base + 1, 0, base, 0}, bulk);
    constexpr int r = TraitsBulk::kBlockM / TraitsTail::kBlockM;
    const int nt = (S + TraitsTail::kBlockM - 1) / TraitsTail::kBlockM;
    const int cA = std::max(0, nt - r * (base + 1));
    const int cB = std::max(0, nt - r * base);
    const int tail = extra * cA + (H - extra) * cB;
    if (tail > 0) {
        launch_mapped<TraitsTail>(params, stream, RowMap{extra, cA, r * (base + 1), cB, r * base}, tail);
    }
    return 2;
}

}  // namespace FLASH_NAMESPACE
