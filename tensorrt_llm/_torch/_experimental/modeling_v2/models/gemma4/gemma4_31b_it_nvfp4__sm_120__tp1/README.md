# `gemma4_31b_it_nvfp4__sm_120__tp1`

Gemma-4-31B-IT (family `gemma4`, architecture `Gemma4ForConditionalGeneration`), the dense
Gemma4 text tower on one RTX PRO 6000 Blackwell, tuned for prefill: prompts of 300 to 8000
tokens at concurrency 1 to 64 (time to first token). It claims the ModelOpt NVFP4 checkpoint
(NVFP4 MLP; every layer's attention and the tied embedding in bf16) with an fp8 KV cache
(the checkpoint's `kv_cache_quant_algo`), FlashInfer attention metadata (the sm_120 default) and
no speculative decoding. The vision half of the composite checkpoint is not part of it: it serves
text, and a request carrying images fails with an error that names `modeling_v2='off'`.

Per layer: the bf16 qkv projection routed by row count (a sequential-K GEMM at about 300 rows,
pinned cuBLASLt algorithms where measured, F.linear elsewhere), one fused q/k/v norm + RoPE +
FP8 kernel that also stores K/V straight into the paged pool on eager steps, FlashInfer FA2 over
the FP8 pool, o_proj, one kernel for the post-attention norm + residual add + pre-FFN norm +
NVFP4 quantize, the gate_up GEMM with a GELU-tanh + NVFP4 epilogue (or an M-chunked chain at
1664+ rows), down_proj, and one kernel for the post-FFN norm + residual add + layer scalar + the
next layer's input norm. The embedding lookup runs fused with the first input norm; the last
layer's attention, the final norm and the lm_head run on the rows the sampler reads only. Prefill
CUDA graphs (breakable, attention eager) replay only steps at a captured token count
(`prefill_cuda_graph_exact_sizes`).

Its own CUDA code is in `csrc/` and builds into `libth_modeling_v2_gemma4.so`: the SM120
sequential-K BF16 GEMM. Its Triton kernels are in `ops/`. Its changes to FlashInfer kernels are
in `kernels/flashinfer/` (copies of the FlashInfer v0.7.3 sources it patches, see the README
there): the head-512 64-row FA2 prefill tile, longest-first CTA order, the row-group split-KV
merge, the hardware FP8->BF16 conversion and the NVFP4 gate_up GEMM with the GELU epilogue. They
are compiled at runtime by FlashInfer's JIT against the installed `flashinfer-python`
(`ops/_flashinfer_jit.py`), which is not modified; the context rows run on the target's own FA2
prefill wrapper planned from the engine metadata's host indptrs.

Where it differs from the built-in `Gemma4ForCausalLM` at the same settings:

| | built-in | this target |
|---|---|---|
| q/k/v norms, RoPE, FP8 cast, KV append | separate kernels | one kernel (stores K/V into the pool on eager steps) |
| full-attention qkv projection | `[q, k, v]` with V a copy of K | `[q, k]`, V from the raw K block |
| attention projections | F.linear | sequential-K GEMM / pinned cuBLASLt / F.linear by rows |
| MLP | gate_up GEMM, fused GELU + quantize, down GEMM | GELU epilogue in the gate_up GEMM; dense tuning buckets and tactic pins |
| norms around attention and MLP | flashinfer rmsnorm + adds | two fused kernels per layer |
| last layer / final norm / lm_head | every row | the rows the sampler reads |

Every fused kernel and every bf16 GEMM route is bitwise equal to the built-in chain it replaces.
The NVFP4 GEMMs pin their tactic per M range where the tuner's pick varies between processes,
so their output is reproducible across servers; the accuracy gate certifies the whole. The
target and the entries it owns are certified on sm_120.
