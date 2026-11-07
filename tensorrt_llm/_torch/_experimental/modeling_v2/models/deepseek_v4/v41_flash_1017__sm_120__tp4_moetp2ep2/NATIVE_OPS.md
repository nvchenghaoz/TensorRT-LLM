<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Native ops of `v41_flash_1017__sm_120__tp4_moetp2ep2`

Built from `csrc/` into `libth_modeling_v2_deepseek_v4.so` (`ARCHS 120f`), namespace
`torch.ops.mv2_deepseek_v4_v41_flash_1017_sm_120_tp4_moetp2ep2`. Python calls them as
`native_ops.<name>` with `from .native import native_ops`; fakes are in `fake_ops.py`. Names and
schemas are unchanged from the optimization run (`torch.ops.trtllm.<name>` there), so a call site
only swaps `torch.ops.trtllm` for `native_ops`, and `hasattr(torch.ops.trtllm, ...)` probes become
unnecessary (the target loads only when the library carries all of them).

Run call sites below are at the run's final main (local ref `ascent/dsv41-1017-final`, a1718d4c).

| Op | Schema | Source | Run call sites |
|---|---|---|---|
| `dsv4_fused_router` | `(Tensor x, Tensor weight, Tensor bias, Tensor(a!) counter, float route_scale) -> (Tensor indices, Tensor weights, Tensor logits)` | `csrc/dsv4FusedRouter*.{h,cu,cpp}` (was `kernels/moe/routing/dsv4FusedRouter`, `thop/moe/routing/dsv4FusedRouterOp.cpp`) | `modeling_deepseekv4.py:1684, 1735` |
| `dsv4_fused_router_route` | `(..., float route_scale, int slot_start, int num_local, Tensor(b!) offsets, Tensor(c!) token_map, Tensor(d!) token_weights, Tensor(e!) dst_rows, Tensor(f!) scale_dst_rows, Tensor(g!) q_out, Tensor(h!) scale_out, Tensor(i!) zero_buf) -> (indices, weights, logits)` | same | `modeling_deepseekv4.py:1704, 1709` |
| `dsv4_fused_router_k1fold` | `(Tensor(j!) x, Tensor x_prev, Tensor residual, Tensor post, Tensor comb, Tensor pre, Tensor norm_weight, float norm_eps, int reducer_threads, Tensor weight, Tensor bias, Tensor(a!) counter, float route_scale) -> (indices, weights, logits)` | same | `modeling_deepseekv4.py:1642, 1738` |
| `dsv4_fused_router_route_k1fold` | k1fold inputs + the route outputs of `dsv4_fused_router_route` | same | `modeling_deepseekv4.py:1643, 1712` |
| `csa2_mxfp8_low_m_gemm` | `(Tensor x, Tensor w, Tensor wsf, int warps, int cluster_k, int row_tiles, int n_split, bool pdl, Tensor? xq=None, Tensor? xsf=None, Tensor(a!)? xq_out=None, Tensor(b!)? xsf_out=None, Tensor(c!)? dbg_ts=None, int stages=1, int debug_flags=0, Tensor? pf_w=None, Tensor? pf_sf=None, int pf_frac=256) -> Tensor[]` | `csrc/csa2LowMGemm*.{h,cu,cpp}` (split out of `kernels/minimaxM3/mxfp8LowMGemm.cu` and its op file) | `attention/backends/sparse/csa2/low_m_gemm.py:97, 116, 148, 151, 211`; `csa2/b12x_wo.py:427, 501` |
| `csa2_qnorm_low_m_gemm` | `(Tensor x, Tensor w, Tensor wsf, Tensor norm_w, float eps, int warps, int cluster_k, int row_tiles, int n_split, bool pdl, int debug_flags=0, Tensor(a!)? xq_out=None, Tensor(b!)? xsf_out=None) -> Tensor[]` | same | `csa2/low_m_gemm.py:172, 199, 233` |
| `csa2_lowm_wo_a` | `(Tensor o, Tensor positions, Tensor cos_sin, Tensor w, Tensor wsc, int groups, int head_dim, int nope_dim, int warps, int cluster_k, int row_tiles, bool pdl, Tensor(a!)? xq_dbg=None, Tensor(b!)? xs_dbg=None, int debug_flags=0) -> Tensor` | `csrc/csa2LowMWo*.{h,cu,cpp}` (was `kernels/minimaxM3/csa2LowMWo`, `thop/minimaxM3/csa2LowMWoOp.cpp`) | `csa2/b12x_wo.py:427, 474` |

The run registered fakes for these in `custom_ops/cpp_custom_ops.py:613-656`,
`csa2/low_m_gemm.py:237-258` and `csa2/b12x_wo.py:532-540`; drop those, `fake_ops.py` replaces them.

## Shared native changes that stay in `torch.ops.trtllm`

These extend stock ops and kernels that other code also uses, so they stay in place (separate
commit). Additive paths need nothing; the behaviour-changing ones are opt-in through environment
variables that default to upstream behaviour. **The target must set these before its first
forward, in every rank process** (for example `os.environ.setdefault(...)` in the target's
`__init__.py`), or it runs the upstream paths and loses the run's speed:

| Variable | Value | Effect (run default) |
|---|---|---|
| `TLLM_AR_SM120_CTAS_PER_TOKEN` | `4` | one-shot plain all-reduce split over 4 CTAs per token without clusters |
| `TLLM_AR_SM120_PDL` | `1` | keep the PDL launch attribute on SM120 all-reduce fusion kernels |
| `TLLM_PCIE_CUSTOM_AR_MAX_BYTES` | `524288` | custom one/two-shot all-reduce over PCIe P2P up to 512 KiB, NCCL above |
| `TRTLLM_LP_AR_PUSH` | `1` | posted-write (PUSH) low-precision two-shot for messages >= 2 MiB |
| `TRTLLM_SM120_FP8_BLOCKSCALE_TILE_HEURISTIC` | `1` | wave-cost tile choice for SM120 FP8 block-scale GEMM/BMM at M > 64 |
| `TRTLLM_GATE_FORWARD_FAST_BODY` | `1` | faster, bitwise-equal top-k body of `gate_forward` |

`TRTLLM_LP_AR_PUSH=1` needs the larger low-precision barrier buffer: the run's
`distributed/allreduce_helper.py` (`CustomAllReduceHelper.lowprecision_barrier_bytes`, used at its
line 123) sizes `RANKS * (1 + LP_TWOSHOT_MAX_CTAS + LP_PUSH_CTAS_PER_PEER * (1 + RANKS))` flags; that
Python change must come with it. The low-precision kernels run only with
`allreduce_strategy: LOWPRECISION`, which the run's serving config sets (with
`moe_tensor_parallel_size: 2`, `moe_expert_parallel_size: 2`, `enable_tokenization_cache: true`).

Additive (no switch): `dsv3_router_gemm_op` tensor-core path for 384 experts x 5120 hidden (M <= 16);
`mhc_fused_hc` with the `fused_half_fma` backend and `tile_m > 1` (the M-tiled prefill kernels;
upstream passes `tile_m = 1` there); SM120 FP8 block-scale builder TileN 32/64; the CUTLASS MoE runner
accepting BF16/FP16 input with MXFP4 weights and `use_mxfp8_act_scaling` (MXFP8 quantization inside the
row expansion); `allreduce_strategy: LOWPRECISION` now compares its 2 MiB threshold with the whole
message instead of the hidden size and falls back to the AUTO choice below it.
