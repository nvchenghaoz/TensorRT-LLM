<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DeepSeek-V4.1-Flash-1017 / sm_120 / tp4, moe_tp2 x moe_ep2

DeepSeek-V4.1-Flash-1017, as DeepSeek ships it (MXFP4 routed experts, FP8 everything else, FP8
Engram tables in pinned host memory), on four RTX PRO 6000 (SM120): attention TP4, routed experts in
two expert-parallel groups each sharded two ways. This is the model the optimization run served at
its final main (landings 1-90): c1/c2/c4 TPOT about 4.94 / 6.06 / 7.3 ms on the 32k-cached-prefix,
2k-unique, 256-output workload, from 26.5 / 29.9 / 37.0 ms on the base.

## Serving

```yaml
modeling_v2: require
tensor_parallel_size: 4
moe_tensor_parallel_size: 2
moe_expert_parallel_size: 2
allreduce_strategy: LOWPRECISION
enable_tokenization_cache: true
enable_attention_dp: false
enable_chunked_prefill: true
max_batch_size: 8
max_num_tokens: 8192
max_seq_len: 36864
cuda_graph_config: {batch_sizes: [1, 2, 4, 8], enable_padding: true}
kv_cache_config: {enable_block_reuse: true, free_gpu_memory_fraction: 0.5, tokens_per_block: 128}
trust_remote_code: true
```

No environment variables are needed. Importing the target sets the shared-kernel opt-ins it was
measured with (`__init__.py`, see `NATIVE_OPS.md`); an explicit setting in the environment wins.

FlashInfer: the run's FlashInfer changes are full copies of FlashInfer v0.7.2 (`d84335d8`) files,
kept verbatim under `flashinfer_overlay/` at FlashInfer's own paths. Install FlashInfer v0.7.2 with
that directory copied over its tree (the optimization run's FlashInfer commit `a42be7c3` is that,
plus the run's environment switches, which this copy hard-codes at their served values).

## Layout

| Path | From the run's tree | What it holds |
|---|---|---|
| `modeling.py` | `_torch/models/modeling_deepseekv41.py` | V4.1 layers, Engram, the target class |
| `_v4.py` | `_torch/models/modeling_deepseekv4.py` | the DeepSeek-V4 base: MoE, fused router, shared expert, loaders |
| `weights.py` | (new) | the loading entry point; the loaders stay next to their modules |
| `tail_moe_graph.py` | `_torch/models/_dsv41_tail_moe_graph.py` | CUDA graphs for the tail layers' prefill MoE |
| `attention/csa2_module.py` | `attention/backends/sparse/csa2/module.py` | the CSA2 attention module the model builds |
| `attention/low_m_gemm.py`, `b12x_wo.py`, `csa2_weights.py` | `csa2/low_m_gemm.py`, `b12x_wo.py`, `weights.py` | SM120 decode projections (QKV, WO) |
| `attention/prefill_wo_quant.py`, `sparse_output.py` | `deepseek_v4/prefill_wo_quant.py`, `deepseek_v4/module.py` | SM120 prefill output projection |
| `ops/mhc/*` | `_torch/modules/mhc/*` | mHC kernels: lagged split, final collapse, prefill all-reduce pipeline |
| `ops/fp8_shared_decode.py`, `ops/small_n_gemv.py` | `_torch/modules/` | shared-expert decode chain, small-N GEMV |
| `native.py`, `fake_ops.py`, `csrc/` | `cpp/` | target C++ ops (`NATIVE_OPS.md`) |
| `flashinfer_overlay/` | `experiments/ascent-flashinfer/` | FlashInfer file copies |

The built-in `modeling_deepseekv4.py`, `modeling_deepseekv41.py` and `modules/mhc/` are upstream
again: `modeling_v2: off` builds the stock model.

## Environment

During the optimization run every change carried an environment kill switch or tuning knob whose
default was the served behaviour. They are gone: each path is hard-coded at the value the run served
with, and the alternatives those switches selected (and their debug, verify and sleep hooks) are
removed. What still reads the environment:

- the six shared-kernel opt-ins listed in `NATIVE_OPS.md`, which `__init__.py` sets with
  `os.environ.setdefault` (they keep other models on the upstream kernels);
- names TensorRT-LLM itself defines: `TRTLLM_ENABLE_PDL`, `FORCE_DETERMINISTIC`,
  `TRTLLM_MHC_ENABLE_FUSED_HC`, `TRTLLM_V41_MHC_DEFER`, `TRTLLM_DISAGG_ROLE`;
- `TLLM_NCCL_PCIE_DEFAULTS=0` (`distributed/nccl_tuning.py`), the opt-out of the PCIe NCCL defaults,
  and `TLLM_PREFIX_TOKEN_CACHE_SHARED_DIR`, which `trtllm-serve` uses to hand the shared prefix token
  cache directory to its frontend processes;
- in `flashinfer_overlay/`, only the variables FlashInfer v0.7.2 itself reads.
