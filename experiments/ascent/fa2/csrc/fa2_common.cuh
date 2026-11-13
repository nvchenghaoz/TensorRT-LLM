// Rebuild of PyTorch's FA2 forward (pytorch_flash::flash_fwd_kernel) from the
// exact sources torch was built from (/opt/pytorch/pytorch/third_party/flash-attention),
// with the same compile definitions (FLASHATTENTION_DISABLE_ALIBI/SOFTCAP, UNFUSE_FMA,
// no fast-math). Only the Kernel_traits tile shape (kBlockM, kNWarps, Q-in-regs,
// Share-Q-K-smem) varies; kHeadDim=128, kBlockN=64 and the per-row math are fixed.
#pragma once
#include "namespace_config.h"
#include "flash.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template <int V> struct VariantTraits;
//                         kHeadDim kBlockM kBlockN kNWarps Q_in_regs Share_Q_K
template <> struct VariantTraits<0> { using T = Flash_fwd_kernel_traits<128, 128, 64, 4, false, false, cutlass::bfloat16_t>; };
template <> struct VariantTraits<1> { using T = Flash_fwd_kernel_traits<128, 256, 64, 8, false, false, cutlass::bfloat16_t>; };
template <> struct VariantTraits<2> { using T = Flash_fwd_kernel_traits<128, 64, 64, 4, false, false, cutlass::bfloat16_t>; };
template <> struct VariantTraits<3> { using T = Flash_fwd_kernel_traits<128, 192, 64, 6, false, false, cutlass::bfloat16_t>; };
template <> struct VariantTraits<4> { using T = Flash_fwd_kernel_traits<128, 128, 64, 4, true, true, cutlass::bfloat16_t>; };
template <> struct VariantTraits<5> { using T = Flash_fwd_kernel_traits<128, 128, 64, 8, false, false, cutlass::bfloat16_t>; };
template <> struct VariantTraits<6> { using T = Flash_fwd_kernel_traits<128, 64, 64, 4, true, false, cutlass::bfloat16_t>; };
template <> struct VariantTraits<7> { using T = Flash_fwd_kernel_traits<128, 128, 64, 4, true, false, cutlass::bfloat16_t>; };

struct VariantInfo { int block_m; int nthreads; int smem; int regs; int max_blocks_per_sm; };

// Same template arguments torch's run_flash_fwd picks for a non-causal, no-dropout,
// d=128 call whose seqlen is not a multiple of the block sizes:
// <Is_dropout=0, Is_causal=0, Is_local=0, Has_alibi=0, Is_even_MN=0, Is_even_K=1, Is_softcap=0, Return_softmax=0>
template <int V>
void run_variant(Flash_fwd_params &params, cudaStream_t stream) {
    using KT = typename VariantTraits<V>::T;
    constexpr size_t smem_size = KT::kSmemSize;
    const int num_m_block = (params.seqlen_q + KT::kBlockM - 1) / KT::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);
    auto kernel = &flash_fwd_kernel<KT, false, false, false, false, false, true, false, false>;
    if (smem_size >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    kernel<<<grid, KT::kNThreads, smem_size, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int V>
VariantInfo variant_info() {
    using KT = typename VariantTraits<V>::T;
    constexpr size_t smem_size = KT::kSmemSize;
    auto kernel = &flash_fwd_kernel<KT, false, false, false, false, false, true, false, false>;
    if (smem_size >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    cudaFuncAttributes attr;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));
    int nb = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kernel, KT::kNThreads, smem_size));
    return VariantInfo{KT::kBlockM, KT::kNThreads, (int)smem_size, attr.numRegs, nb};
}

}  // namespace FLASH_NAMESPACE

#define ASCENT_FA2_INSTANTIATE(V)                                                                   \
    namespace FLASH_NAMESPACE {                                                                     \
    void ascent_run_v##V(Flash_fwd_params &p, cudaStream_t s) { run_variant<V>(p, s); }             \
    VariantInfo ascent_info_v##V() { return variant_info<V>(); }                                    \
    }
