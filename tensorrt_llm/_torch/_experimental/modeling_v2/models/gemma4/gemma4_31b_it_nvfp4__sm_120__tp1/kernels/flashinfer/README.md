# FlashInfer kernel sources of this target

Copies of FlashInfer v0.7.3 (flashinfer-ai/flashinfer `775207deb34055667df89ed618fb4b06fc84de74`,
Apache-2.0) source files this target changes, plus the new files it adds, laid out as in the
FlashInfer tree. They are compiled at runtime by FlashInfer's JIT against the installed
`flashinfer-python==0.7.3` (`ops/_flashinfer_jit.py`): `include/` is laid over the installed
include tree, and the prefill module is generated from `csrc/batch_prefill.cu` and
`csrc/batch_prefill_paged_kernel_inst.jinja` under its own URI. The installed FlashInfer is not
modified; with another FlashInfer release the target runs FlashInfer's own kernels (same output).

Every change keeps the output of the plans it touches bitwise equal to v0.7.3's:

| File | Change |
|---|---|
| `include/flashinfer/attention/prefill_d512_tile64.cuh` (new) | FA2 paged prefill kernel for head_dim 512 with FP8 KV: 64-row Q tiles (8 tokens x GQA 8), 8 warps, BF16 mma.sync with in-register FP8->BF16 K/V dequant, double-buffered paged K/V loads, coalesced quarter-warp cp.async fills with an XOR row swizzle. Each 32-row half computes exactly what the 32-row kernel does. |
| `include/flashinfer/attention/d512_tile64_config.cuh` (new) | The planner's promotion of an unsplit, full-attention FP8-KV d512 plan with average packed length > 64 to the 64-row tile (SM12x with 100 KB+ shared memory), and the longest-first item order and heads-minor grid of non-CUDA-graph paged plans. |
| `include/flashinfer/attention/fa2_merge_rows.cuh` (new) | Split-KV merge with one lane group per (position, head): the same float operations, in the same order, as the persistent merge. |
| `include/flashinfer/attention/scheduler.cuh` | Calls the promotion and emits work items longest-first. |
| `include/flashinfer/attention/prefill.cuh` | Dispatches the 64-row kernel and the heads-minor grid; merges with `fa2_merge_rows`. |
| `include/flashinfer/vec_dtypes.cuh` | FP8 e4m3 -> BF16 `vec_cast` through the hardware `cvt.rn.bf16x2.e4m3x2` on SM120 (exact). |
| `csrc/batch_prefill_paged_kernel_inst.jinja`, `csrc/batch_prefill.cu` | Instantiate and plan CTA_TILE_Q 64 for FP8-KV head_dim 512. |
| `include/flashinfer/gemm/nvfp4_gelu_gemm_sm120.cuh`, `csrc/nvfp4_gelu_gemm_sm120.cu` (new) | NVFP4 gate_up GEMM (SM120 256x128x128 block-scaled mainloop) with a gelu_tanh(gate) * up + NVFP4-quantize epilogue, byte-identical to the unfused GEMM + activation-quantize chain. |

Diff against v0.7.3:

     csrc/batch_prefill.cu                              |  10 +-
     csrc/batch_prefill_paged_kernel_inst.jinja         |   3 +-
     csrc/nvfp4_gelu_gemm_sm120.cu                      |  83 +++
     .../flashinfer/attention/d512_tile64_config.cuh    | 135 ++++
     include/flashinfer/attention/fa2_merge_rows.cuh    | 182 +++++
     include/flashinfer/attention/prefill.cuh           |  39 +-
     .../flashinfer/attention/prefill_d512_tile64.cuh   | 805 +++++++++++++++++++++
     include/flashinfer/attention/scheduler.cuh         | 125 +++-
     include/flashinfer/gemm/nvfp4_gelu_gemm_sm120.cuh  | 378 ++++++++++
     include/flashinfer/vec_dtypes.cuh                  |  32 +
     10 files changed, 1779 insertions(+), 13 deletions(-)
