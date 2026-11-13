"""Multi-tensor per-tensor E4M3 quantization of Q, K and V for the SM120 FP8 attention path.

Bitwise equal to three calls of ``torch.ops.trtllm.quantize_e4m3_per_tensor(t.contiguous())``
(the C++ op's PER_TENSOR path, cpp/tensorrt_llm/common/cudaFp8Utils.cu):

- amax  = max |float(x)| over the tensor (max is order independent, so any partition is exact);
- scale = bf16(max(amax / 448.0f, 1 / (448 * 512)))       (T_S = bf16 for bf16 input);
- out   = cvt_e4m3_satfinite_rn(float(x) * (1.0f / float(scale)));
- descale returned as float(scale).

Two launches for all three tensors instead of 3 x (memset + amax + quantize) plus the
``.contiguous()`` copy of a strided V view: K1 writes per-program amax partials (no memset, no
atomics), K2 reduces its tensor's partials, derives the scale and quantizes. Rows are read with
their own row stride, so a V that is a column slice of the packed qkv buffer is never copied.
"""

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl

FP8_E4M3_MAX = 448.0
# The C++ op's constexpr min_scaling_factor = 1.0f / (FP8_E4M3_MAX * 512.f), rounded to fp32.
_MIN_SCALING_FACTOR = float(np.float32(1.0) / np.float32(448.0 * 512.0))

_BLOCK_R = 16
_NUM_WARPS = 8
# Persistent programs per SM, summed over the three tensors. With the quantize pass walking
# tiles in reverse of the amax pass (its first tiles are the amax pass's last, still in L2),
# 3 programs/SM, 16x512 tiles, 8 warps measured 1.334 ms vs main's 1.802 ms (job 563, SM120).
_PROGRAMS_PER_SM = 3


@triton.jit
def _select(t, a, b, c):
    if t == 0:
        r = a
    elif t == 1:
        r = b
    else:
        r = c
    return r


@triton.jit
def _fp8_prep_amax_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    q_rs,
    k_rs,
    v_rs,
    partial_ptr,
    n_rows,
    width,
    n_col_blocks,
    n_tiles,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    t = tl.program_id(1)
    nprog = tl.num_programs(0)
    base = _select(t, q_ptr, k_ptr, v_ptr)
    rs = _select(t, q_rs, k_rs, v_rs).to(tl.int64)
    r_off = tl.arange(0, BLOCK_R)
    c_off = tl.arange(0, BLOCK_C)
    acc = tl.zeros([BLOCK_R, BLOCK_C], dtype=tl.float32)
    for tile in range(pid, n_tiles, nprog):
        rb = tile // n_col_blocks
        cb = tile - rb * n_col_blocks
        rows = rb * BLOCK_R + r_off
        cols = cb * BLOCK_C + c_off
        mask = (rows < n_rows)[:, None] & (cols < width)[None, :]
        x = tl.load(base + rows.to(tl.int64)[:, None] * rs + cols[None, :], mask=mask, other=0.0)
        acc = tl.maximum(acc, tl.abs(x.to(tl.float32)))
    amax = tl.max(tl.max(acc, axis=1), axis=0)
    tl.store(partial_ptr + t * nprog + pid, amax)


@triton.jit
def _fp8_prep_quantize_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    q_rs,
    k_rs,
    v_rs,
    qo_ptr,
    ko_ptr,
    vo_ptr,
    partial_ptr,
    descale_ptr,
    n_amax_programs,
    n_rows,
    width,
    n_col_blocks,
    n_tiles,
    MIN_SCALE: tl.constexpr,
    P_POW2: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    REVERSE: tl.constexpr,
):
    pid = tl.program_id(0)
    t = tl.program_id(1)
    nprog = tl.num_programs(0)
    p_off = tl.arange(0, P_POW2)
    partials = tl.load(
        partial_ptr + t * n_amax_programs + p_off, mask=p_off < n_amax_programs, other=0.0
    )
    amax = tl.max(partials, axis=0)
    # (T_S) std::max(max / FP8_E4M3_MAX, min_scaling_factor), T_S = bf16; IEEE division.
    scale = tl.maximum(amax / 448.0, MIN_SCALE).to(tl.bfloat16).to(tl.float32)
    factor = 1.0 / scale
    if pid == 0:
        tl.store(descale_ptr + t, scale)
    base = _select(t, q_ptr, k_ptr, v_ptr)
    out = _select(t, qo_ptr, ko_ptr, vo_ptr)
    rs = _select(t, q_rs, k_rs, v_rs).to(tl.int64)
    r_off = tl.arange(0, BLOCK_R)
    c_off = tl.arange(0, BLOCK_C)
    for i in range(pid, n_tiles, nprog):
        if REVERSE:
            tile = n_tiles - 1 - i
        else:
            tile = i
        rb = tile // n_col_blocks
        cb = tile - rb * n_col_blocks
        rows = rb * BLOCK_R + r_off
        cols = cb * BLOCK_C + c_off
        mask = (rows < n_rows)[:, None] & (cols < width)[None, :]
        rows64 = rows.to(tl.int64)[:, None]
        x = tl.load(base + rows64 * rs + cols[None, :], mask=mask, other=0.0)
        y = (x.to(tl.float32) * factor).to(tl.float8e4nv)
        tl.store(out + rows64 * width + cols[None, :], y, mask=mask)


def _as_rows(t: torch.Tensor, width: int) -> torch.Tensor | None:
    """A [rows, width] view with unit column stride, or None if the layout needs a copy."""
    try:
        t2 = t.view(-1, width)
    except RuntimeError:
        return None
    if t2.stride(1) != 1:
        return None
    return t2


def _block_c(width: int) -> int:
    block_c = 512
    while block_c > 16 and width % block_c != 0:
        block_c //= 2
    return block_c


_SM_COUNT: dict[int, int] = {}


def _sm_count(device: torch.device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _SM_COUNT:
        _SM_COUNT[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _SM_COUNT[idx]


def fp8_qkv_prep_supported(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    if not (q.shape == k.shape == v.shape and q.dim() >= 2 and q.numel() > 0):
        return False
    if not (q.dtype == k.dtype == v.dtype == torch.bfloat16):
        return False
    if not (q.is_cuda and k.device == q.device and v.device == q.device):
        return False
    width = q.shape[-1] * q.shape[-2] if q.dim() >= 3 else q.shape[-1]
    return all(_as_rows(t, width) is not None for t in (q, k, v))


def quantize_e4m3_qkv_per_tensor(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    programs_per_sm: int = _PROGRAMS_PER_SM,
    reverse: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """E4M3 Q, K, V (contiguous, input shapes) and their float32 descales as one [3] tensor.

    Bitwise equal to ``quantize_e4m3_per_tensor`` on each (made contiguous) tensor. Requires
    ``fp8_qkv_prep_supported(q, k, v)``.
    """
    width = q.shape[-1] * q.shape[-2] if q.dim() >= 3 else q.shape[-1]
    q2, k2, v2 = (_as_rows(t, width) for t in (q, k, v))
    n_rows = q2.shape[0]
    block_c = _block_c(width)
    n_col_blocks = triton.cdiv(width, block_c)
    n_tiles = triton.cdiv(n_rows, _BLOCK_R) * n_col_blocks
    programs = max(1, min(n_tiles, programs_per_sm * _sm_count(q.device) // 3))
    p_pow2 = triton.next_power_of_2(programs)

    q_fp8 = torch.empty(q.shape, dtype=torch.float8_e4m3fn, device=q.device)
    k_fp8 = torch.empty(k.shape, dtype=torch.float8_e4m3fn, device=q.device)
    v_fp8 = torch.empty(v.shape, dtype=torch.float8_e4m3fn, device=q.device)
    partials = torch.empty(3 * programs, dtype=torch.float32, device=q.device)
    descales = torch.empty(3, dtype=torch.float32, device=q.device)

    grid = (programs, 3)
    _fp8_prep_amax_kernel[grid](
        q2,
        k2,
        v2,
        q2.stride(0),
        k2.stride(0),
        v2.stride(0),
        partials,
        n_rows,
        width,
        n_col_blocks,
        n_tiles,
        BLOCK_R=_BLOCK_R,
        BLOCK_C=block_c,
        num_warps=_NUM_WARPS,
    )
    _fp8_prep_quantize_kernel[grid](
        q2,
        k2,
        v2,
        q2.stride(0),
        k2.stride(0),
        v2.stride(0),
        q_fp8,
        k_fp8,
        v_fp8,
        partials,
        descales,
        programs,
        n_rows,
        width,
        n_col_blocks,
        n_tiles,
        MIN_SCALE=_MIN_SCALING_FACTOR,
        P_POW2=p_pow2,
        BLOCK_R=_BLOCK_R,
        BLOCK_C=block_c,
        REVERSE=reverse,
        num_warps=_NUM_WARPS,
    )
    return q_fp8, k_fp8, v_fp8, descales
