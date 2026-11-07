# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-launch exact replacement of ``mHC.collapse`` for V4.1's final ``_finalize_hc_state`` (decode).

``mHC.collapse(x, pre_mix)`` (hyper_connection.py) runs as 12 ATen launches for mult = 4:

    acc = x[..., 0, :].float() * pre_mix[..., 0, :]
    acc += x[..., i, :].float() * pre_mix[..., i, :]      for i = 1 .. mult - 1
    return acc.to(bf16)

Each product is rounded to fp32 on its own and the adds run in order i = 0 .. mult - 1 with no
FMA; the cast to bf16 rounds to nearest even. :func:`mhc_final_collapse` computes exactly that in
one Triton program per (token, BLOCK-column tile), compiled with ``enable_fp_fusion=False`` so
no mul+add contracts into an FMA. It is bitwise equal to ``mHC.collapse``.

Only V4.1's ``_finalize_hc_state`` uses it, under :func:`final_collapse_supported` (decode shape:
1 <= M <= 8, mult == 4, contiguous bf16 residual ``[M, 4, hidden]``, fp32 pre_mix ``[M, 4, 1]``
of any strides); every other shape keeps ``mHC.collapse``. ``TRTLLM_FUSED_FINAL_HC=0`` turns it off.
Plain launch (no PDL): its producer, ``mhc_post_mapping`` / the last sublayer, launches without PDL.
"""


import torch
import triton
import triton.language as tl
from ...switches import env_get  # run switches frozen at served values

FUSED_FINAL_HC_ENV_ON = env_get("TRTLLM_FUSED_FINAL_HC", "1") != "0"
_MAX_TOKENS = 8
_BLOCK = 1024
_NUM_WARPS = 4

# Compile keys (see _compile_key) already JIT-compiled eagerly, outside any stream capture. A first
# call inside a capture falls back to mHC.collapse rather than JIT-compiling there (branch 144 job
# 1342: cudaErrorStreamCaptureInvalidated). The CUDA-graph runner's eager warmup at each batch size
# runs first, so in serving the capture finds the kernel compiled.
_WARM_KEYS = set()


@triton.jit
def _hc_final_collapse_kernel(
    x_ptr,
    pre_ptr,
    out_ptr,
    PRE_S0: tl.constexpr,
    PRE_S1: tl.constexpr,
    HIDDEN: tl.constexpr,
    MULT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tok = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < HIDDEN
    xbase = x_ptr + tok * (MULT * HIDDEN)
    pbase = pre_ptr + tok * PRE_S0
    # acc = x0 * p0; acc = acc + xi * pi (i = 1 .. MULT - 1), each product rounded on its own.
    acc = tl.load(xbase + offs, mask=mask, other=0.0).to(tl.float32) * tl.load(pbase)
    for i in tl.static_range(1, MULT):
        xi = tl.load(xbase + i * HIDDEN + offs, mask=mask, other=0.0).to(tl.float32)
        acc = acc + xi * tl.load(pbase + i * PRE_S1)
    tl.store(out_ptr + tok * HIDDEN + offs, acc.to(tl.bfloat16), mask=mask)


def final_collapse_supported(x: torch.Tensor, pre_mix: torch.Tensor) -> bool:
    """Host gate: decode shape and layouts the kernel covers; anything else keeps mHC.collapse."""
    if not FUSED_FINAL_HC_ENV_ON:
        return False
    if x.dim() != 3 or not x.is_cuda:
        return False
    M, mult, hidden = x.shape
    if not (1 <= M <= _MAX_TOKENS) or mult != 4:
        return False
    if x.dtype != torch.bfloat16 or not x.is_contiguous():
        return False
    if pre_mix.dtype != torch.float32 or tuple(pre_mix.shape) != (M, mult, 1):
        return False
    if pre_mix.device != x.device:
        return False
    if torch.compiler.is_compiling():
        return False
    if _compile_key(x, pre_mix) not in _WARM_KEYS and torch.cuda.is_current_stream_capturing():
        return False
    return True


def _compile_key(x: torch.Tensor, pre_mix: torch.Tensor):
    # Everything Triton specializes this kernel on: the constexprs and the pointer alignments.
    return (
        x.shape[-1],
        pre_mix.stride(0),
        pre_mix.stride(1),
        x.data_ptr() % 16 == 0,
        pre_mix.data_ptr() % 16 == 0,
    )


def mhc_final_collapse(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    """``[M, mult, hidden]`` bf16 x ``[M, mult, 1]`` fp32 -> ``[M, hidden]`` bf16, bitwise == mHC.collapse."""
    M, mult, hidden = x.shape
    out = torch.empty((M, hidden), dtype=x.dtype, device=x.device)
    if M == 0:
        return out
    _hc_final_collapse_kernel[(M, triton.cdiv(hidden, _BLOCK))](
        x,
        pre_mix,
        out,
        PRE_S0=pre_mix.stride(0),
        PRE_S1=pre_mix.stride(1),
        HIDDEN=hidden,
        MULT=mult,
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
        enable_fp_fusion=False,
    )
    _WARM_KEYS.add(_compile_key(x, pre_mix))
    return out
