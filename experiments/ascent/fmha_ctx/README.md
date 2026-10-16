# fmha_ctx: SM120 e4m3 tiled context FMHA harness (branch 16, idea 45)

Standalone harness for
`fmha_v2_flash_attention_e4m3_fp32_64_128_S_q_paged_kv_256_causal_output_bf16_sm120_kernel_nl_tiled`.
It instantiates the kernel template from a given `cpp/kernels/fmha_v2/src` tree with nvcc (same
template arguments and compile definitions as the generated TU), so the base tree and the candidate
tree can be timed and compared bitwise without loading the model.
