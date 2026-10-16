// Kernels that must compile to the same code on main and on this branch (STORE_O_FROM_REGS is false for them):
// BF16 / FP16 HMMA tiled no-loop flash kernels (head 256, paged KV, causal) and the non-tiled e4m3 no-loop kernel.
#include <fused_multihead_flash_attention_kernel_noloop.h>
#include <fused_multihead_flash_attention_kernel_noloop_tiled.h>

using Kt_bf16_tiled = fmha::Kernel_traits_v2_paged_kv_cache<fmha::Ampere_hmma_bf16_traits, 128, 256, 0, 64, 4, 1, 1,
    0x1227u, 3, true>;
using Kt_fp16_tiled = fmha::Kernel_traits_v2_paged_kv_cache<fmha::Ampere_hmma_fp16_traits, 128, 256, 0, 64, 4, 1, 1,
    0x1227u, 3, true>;
using Kt_e4m3_nl = fmha::Kernel_traits_v2_paged_kv_cache<fmha::Ada_qmma_e4m3_fp32_traits, 32, 256, 0, 64, 4, 1, 1,
    0x0227u, 3, true, nv_bfloat16>;

extern "C" __global__ void k_bf16_tiled(bert::Fused_multihead_attention_params_v2 params)
{
    fused_multihead_attention::device_flash_attention_nl_tiled<Kt_bf16_tiled>(params);
}

extern "C" __global__ void k_fp16_tiled(bert::Fused_multihead_attention_params_v2 params)
{
    fused_multihead_attention::device_flash_attention_nl_tiled<Kt_fp16_tiled>(params);
}

extern "C" __global__ void k_e4m3_nl(bert::Fused_multihead_attention_params_v2 params)
{
    fused_multihead_attention::device_flash_attention_nl<Kt_e4m3_nl>(params);
}

// The launch smem sizes (must be unchanged too).
__device__ int smem_sizes[3] = {Kt_bf16_tiled::BYTES_PER_SMEM, Kt_fp16_tiled::BYTES_PER_SMEM, Kt_e4m3_nl::BYTES_PER_SMEM};
