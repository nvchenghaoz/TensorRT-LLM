"""FLUX.3 d128 FP8 attention: an add-only FlashInfer fmha_v2 JIT variant.

FlashInfer's stock ``trtllm_fmha_v2`` module runs
``fmha_v2_flash_attention_e4m3_fp32_64_64_S_q_k_v_128x128_output_bf16_sm120_nl_tiled``
for E4M3 Q/K/V with head_dim 128 and BF16 output. This module compiles only that
kernel (renamed, so it cannot alias the stock symbols) plus the binding, with
macro-gated exact levers in ``fused_multihead_flash_attention_kernel_noloop_tiled.h``:

- ``mask_skip``  (FMHA_FLUX3_MASK_SKIP): the padding mask runs only on the last partial
  KV step instead of on every step from the causal diagonal on (bitwise equal).
- ``resident_q`` (FMHA_FLUX3_RESIDENT_Q): Q is loaded into its two smem buffers once
  instead of being re-fetched every KV step (bitwise equal).
- ``min_blocks``: ``__launch_bounds__(128, min_blocks)`` on the kernel.

The stock module is untouched (macros default off), so both can be loaded in one
process. The module is built on first use; callers build it at warmup.

This file must not import tensorrt_llm at module level: microbenchmarks load it by path.
"""

from __future__ import annotations

import functools
import os
import re
from typing import Optional, Sequence

import torch

STOCK_KERNEL = "fmha_v2_flash_attention_e4m3_fp32_64_64_S_q_k_v_128x128_output_bf16_sm120"

_LEVER_FLAGS = {
    "mask_skip": ("ms", "-DFMHA_FLUX3_MASK_SKIP=1"),
    "resident_q": ("rq", "-DFMHA_FLUX3_RESIDENT_Q=1"),
    # REORDERED levers (not bitwise vs stock):
    # base2 (FMHA_FLUX3_BASE2, softmax.h / fragment.h): raw QK accumulators, one
    #   exp2(fma(x, s*log2e, log2(256) - max*s*log2e)) per element and
    #   alpha = exp2((prev - cur)*s*log2e). No LSE / attention-sink support.
    "base2": ("b2", "-DFMHA_FLUX3_BASE2=1"),
    # kv128: 64 q rows x 128-key step (the stock d128 E4M3 tile steps 64 keys). Padding mask
    #   only: the causal / sliding / custom-mask kernels are not compiled in.
    "kv128": ("k128", None),
    # unit_alpha (FMHA_FLUX3_SKIP_UNIT_ALPHA, fragment.h, base2 only): a warp whose 2x16 rows
    #   all keep their running max (alpha == 1.0f) skips the O rescale (bitwise vs base2).
    "unit_alpha": ("ua", "-DFMHA_FLUX3_SKIP_UNIT_ALPHA=1"),
}

# The lever set main runs.
DEFAULT_LEVERS: tuple[str, ...] = ("mask_skip", "resident_q")
# The lever set for non-causal, no-LSE calls: base-2 softmax on main's ms_rq 64x64 tile.
# Job 727 (interleaved, S=4176/4096): ms_rq 427.7/408.4 us, +base2 415.2/397.0 (-2.9%/pass),
# base2+kv128 (branch 30's 64x128 rqms_b2) 415.5/397.1 (same speed, rel L2 1.0017x ms_rq's;
# base2 at 64x64 is 1.0000x), kv128 alone 437.9/415.6 (slower).
ROUTED_LEVERS: tuple[str, ...] = ("base2", "mask_skip", "resident_q", *(("unit_alpha",) if os.environ.get("TRTLLM_FMHA_V2_FLUX3_UA", "1") != "0" else ()))  # noqa: E501
DEFAULT_MIN_BLOCKS = 0


def _tag(levers: Sequence[str], min_blocks: int) -> str:
    parts = [_LEVER_FLAGS[name][0] for name in sorted(set(levers))]
    tag = "_".join(parts) if parts else "base"
    if min_blocks:
        tag += f"_mb{min_blocks}"
    return tag


def gen_flux3_d128_spec(levers: Sequence[str] = DEFAULT_LEVERS, min_blocks: int = 0):
    """JitSpec of the d128 E4M3->BF16 variant with the given levers."""
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.attention.fmha_v2.generate_kernels import enumerate_kernels
    from flashinfer.jit.core import current_compilation_context, gen_jit_spec
    from flashinfer.jit.utils import write_if_different

    tag = _tag(levers, min_blocks)
    uri = f"trtllm_fmha_v2_flux3_d128_{tag}"
    gen_dir = jit_env.FLASHINFER_JIT_DIR / uri
    gen_dir.mkdir(parents=True, exist_ok=True)
    fmha_v2_src_dir = jit_env.FLASHINFER_CSRC_DIR / "fmha_v2"

    enumerate_kernels(fmha_v2_src_dir, gen_dir)
    stock_path = gen_dir / "generated" / f"{STOCK_KERNEL}.cu"
    text = stock_path.read_text()
    kernel = STOCK_KERNEL.replace("fmha_v2_flash_attention_", f"fmha_v2_flux3_{tag}_")
    text = text.replace(STOCK_KERNEL, kernel)
    if min_blocks:
        text, count = re.subn(
            r"__global__\s*\nvoid (\w+_nl_tiled)\(",
            rf"__global__ __launch_bounds__(128, {int(min_blocks)})\nvoid \1(",
            text,
        )
        if count == 0:
            raise RuntimeError("could not place __launch_bounds__ in the generated kernel")
    if "kv128" in levers:
        # Kernel_traits<instruction_traits, kv_loop_step, head_size, head_size_v, noloop_step, ...>
        text, count = re.subn(
            r"(fmha::\w+_traits,\s*\n\s*)64(,\s*\n\s*128,\s*\n\s*128,\s*\n\s*64,)",
            r"\g<1>128\g<2>",
            text,
        )
        if count == 0:
            raise RuntimeError("could not set kv_loop_step=128 in the generated kernel")
        for mask in ("causal_mask", "sliding_or_chunked_causal_mask", "custom_mask"):
            text = text.replace(f"#if 1 // {mask}\n", f"#if 0 // {mask}\n")
    kernel_path = gen_dir / "generated" / f"{kernel}.cu"
    write_if_different(kernel_path, text)

    nvcc_flags = current_compilation_context.get_nvcc_flags_list(supported_major_versions=[12])
    nvcc_flags.append(f"-I{fmha_v2_src_dir}")
    nvcc_flags.append("-Wno-deprecated-gpu-targets")
    nvcc_flags.append(f"-DFMHA_FLUX3_D128_LAUNCHER=run_{kernel}_nl_tiled")
    for name in sorted(set(levers)):
        if _LEVER_FLAGS[name][1]:
            nvcc_flags.append(_LEVER_FLAGS[name][1])
    binding = jit_env.FLASHINFER_CSRC_DIR / "trtllm_fmha_v2_binding.cu"
    return gen_jit_spec(uri, [kernel_path, binding], extra_cuda_cflags=nvcc_flags)


@functools.cache
def get_flux3_d128_module(levers: tuple[str, ...] = DEFAULT_LEVERS, min_blocks: int = 0):
    return gen_flux3_d128_spec(levers, min_blocks).build_and_load()


def flux3_d128_kernel_name(levers: Sequence[str] = DEFAULT_LEVERS, min_blocks: int = 0) -> str:
    tag = _tag(levers, min_blocks)
    return STOCK_KERNEL.replace("fmha_v2_flash_attention_", f"fmha_v2_flux3_{tag}_") + "_nl_tiled"


def fmha_v2_flux3_d128(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    num_heads: int,
    head_dim: int,
    seq_len: int,
    *,
    scale_bmm1_d: Optional[torch.Tensor],
    scale_bmm2_d: Optional[torch.Tensor],
    causal: bool = False,
    lse: Optional[torch.Tensor] = None,
    module=None,
) -> torch.Tensor:
    """Same argument convention as flashinfer.prefill.fmha_v2_prefill_sm120 (E4M3, d128,
    BF16 out, contiguous BSHD, device scales, scale_softmax = scale_bmm1 = scale_bmm2 = 1)."""
    if head_dim != 128 or query.dtype != torch.float8_e4m3fn or out.dtype != torch.bfloat16:
        raise ValueError("fmha_v2_flux3_d128 supports E4M3 head_dim 128 with BF16 output only.")
    if module is None:
        module = get_flux3_d128_module()
    module.run(
        query,
        key,
        value,
        out,
        lse,
        scale_bmm1_d,
        scale_bmm2_d,
        num_heads,
        head_dim,
        seq_len,
        1.0,
        1.0,
        1.0,
        causal,
        True,
        True,
    )
    return out
