# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-head Q/K RMSNorm + interleaved RoPE pass for the SM120 FLUX.3 NVFP4 QKV projection.

Applies exactly what the fused CuTe QKV epilogue (flux3_sm120_nvfp4_kernel.py, QKV branch)
applies to the BF16 GEMM output, with the same arithmetic:
- lane l of the warp owning a (row, head) holds columns l, 32+l, 64+l, 96+l and sums their
  squares in that order, then a butterfly shuffle reduction (16, 8, 4, 2, 1);
- inv_rms = rsqrt(sum / 128 + 1e-6); scaled = (x * inv_rms) * w;
- RoPE pair p = (2p, 2p+1) with cos = cos_sin[pos, p], sin = cos_sin[pos, 64 + p]:
  even' = even * cos - odd * sin, odd' = odd * cos + even * sin, each rounded to BF16.
Each lane rotates the columns it holds and gets its pair partner's scaled value by an xor-1
shuffle (pairs are independent, so this changes no result). One CTA per row: warps 0-3 own the
q heads, 4-7 the k heads (heads/4 each), 8-11 copy v when the source is not the output itself.
All of a warp's loads are issued before its stores, so the pass can run in place. The row
count is a runtime value: one compile per layout serves every text length.
"""

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import torch
from cutlass._mlir.dialects import nvvm
from cutlass.cute.runtime import make_ptr

_COMPILED = {}


@cute.kernel
def _qknorm_rope_kernel(
    src: cute.Pointer,
    k_off: cutlass.Int64,
    v_off: cutlass.Int64,
    dst: cute.Pointer,
    q_norm: cute.Pointer,
    k_norm: cute.Pointer,
    cos_sin: cute.Pointer,
    positions: cute.Pointer,
    heads: cutlass.Constexpr,
    src_ld: cutlass.Constexpr,
    copy_v: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    warp = tidx // 32
    lane = tidx % 32
    row64 = cutlass.Int64(row)
    width = heads * 128
    per_warp = heads // 4
    span = per_warp * 128
    if warp < 8:
        plane = warp // 4
        head0 = (warp % 4) * per_warp
        norm = q_norm if plane == 0 else k_norm
        offset = k_off if plane == 1 else cutlass.Int64(0)
        source = cute.make_tensor(
            src + offset + row64 * src_ld + head0 * 128, cute.make_layout(span)
        )
        output = cute.make_tensor(
            dst + row64 * (3 * width) + plane * width + head0 * 128, cute.make_layout(span)
        )
        position = (positions + row).load()
        # Load everything first (in-place safe): the sum-of-squares columns l + 32g and the
        # RoPE pair columns 2p, 2p+1 for pairs p = l and l + 32.
        values = []
        pairs = []
        for h in cutlass.range_constexpr(per_warp):
            for group in cutlass.range_constexpr(4):
                values.append(source[h * 128 + group * 32 + lane])
            for pair_group in cutlass.range_constexpr(2):
                even_col = 2 * (pair_group * 32 + lane)
                pairs.append(source[h * 128 + even_col])
                pairs.append(source[h * 128 + even_col + 1])
        for h in cutlass.range_constexpr(per_warp):
            sum_squares = cutlass.Float32(0.0)
            for group in cutlass.range_constexpr(4):
                value = values[h * 4 + group].to(cutlass.Float32)
                sum_squares += value * value
            for delta in (16, 8, 4, 2, 1):
                sum_squares += cute.arch.shuffle_sync_op(
                    sum_squares, delta, kind=nvvm.ShflKind.bfly
                )
            inv_rms = cute.rsqrt(sum_squares / cutlass.Float32(128) + cutlass.Float32(1.0e-6))
            for pair_group in cutlass.range_constexpr(2):
                pair = pair_group * 32 + lane
                even_col = 2 * pair
                even = (
                    pairs[h * 4 + 2 * pair_group].to(cutlass.Float32)
                    * inv_rms
                    * (norm + even_col).load().to(cutlass.Float32)
                )
                odd = (
                    pairs[h * 4 + 2 * pair_group + 1].to(cutlass.Float32)
                    * inv_rms
                    * (norm + even_col + 1).load().to(cutlass.Float32)
                )
                cosine = (cos_sin + position * 128 + pair).load()
                sine = (cos_sin + position * 128 + 64 + pair).load()
                output[h * 128 + even_col] = cutlass.BFloat16(even * cosine - odd * sine)
                output[h * 128 + even_col + 1] = cutlass.BFloat16(odd * cosine + even * sine)
    else:
        if cutlass.const_expr(copy_v):
            head0 = (warp - 8) * per_warp
            source = cute.make_tensor(
                src + v_off + row64 * src_ld + head0 * 128, cute.make_layout(span)
            )
            output = cute.make_tensor(
                dst + row64 * (3 * width) + 2 * width + head0 * 128, cute.make_layout(span)
            )
            copied = []
            for i in cutlass.range_constexpr(per_warp * 4):
                copied.append(source[i * 32 + lane])
            for i in cutlass.range_constexpr(per_warp * 4):
                output[i * 32 + lane] = copied[i]


@cute.jit
def _launch(
    src: cute.Pointer,
    k_off: cutlass.Int64,
    v_off: cutlass.Int64,
    dst: cute.Pointer,
    q_norm: cute.Pointer,
    k_norm: cute.Pointer,
    cos_sin: cute.Pointer,
    positions: cute.Pointer,
    rows: cutlass.Int32,
    heads: cutlass.Constexpr,
    src_ld: cutlass.Constexpr,
    copy_v: cutlass.Constexpr,
    stream,
):
    warps = 12 if copy_v else 8
    _qknorm_rope_kernel(
        src, k_off, v_off, dst, q_norm, k_norm, cos_sin, positions, heads, src_ld, copy_v
    ).launch(grid=[rows, 1, 1], block=[warps * 32, 1, 1], stream=stream)


def qknorm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Write norm+RoPE(q), norm+RoPE(k) and v into ``out`` [M, 3*H*128] (q|k|v packed).

    q, k, v are BF16 [M, H*128] row-major tensors sharing one row stride. When q is the first
    column plane of ``out`` (q, k, v its three planes) the pass runs in place and v is not
    touched. Heads must be a multiple of 4; cos_sin is FP32 [*, 128] indexed by positions.
    """
    rows, width = out.shape[0], out.shape[1] // 3
    heads = width // 128
    if heads % 4 or width % 128:
        raise ValueError(f"unsupported QKV width {out.shape[1]}")
    src_ld = q.stride(0)
    if k.stride(0) != src_ld or v.stride(0) != src_ld:
        raise ValueError("q, k and v must share a row stride")
    copy_v = v.data_ptr() != out.data_ptr() + 2 * width * out.element_size()
    k_off = (k.data_ptr() - q.data_ptr()) // q.element_size()
    v_off = (v.data_ptr() - q.data_ptr()) // q.element_size()
    pointers = [
        make_ptr(dtype, tensor.data_ptr(), assumed_align=16)
        for dtype, tensor in (
            (cutlass.BFloat16, q),
            (cutlass.BFloat16, out),
            (cutlass.BFloat16, q_norm),
            (cutlass.BFloat16, k_norm),
            (cutlass.Float32, cos_sin),
            (cutlass.Int64, positions),
        )
    ]
    src, dst, qn, kn, cs, pos = pointers
    key = (out.device.index, heads, src_ld, copy_v)
    stream = cutlass_torch.current_stream()
    compiled = _COMPILED.get(key)
    if compiled is None:
        compiled = cute.compile(
            _launch,
            src,
            cutlass.Int64(k_off),
            cutlass.Int64(v_off),
            dst,
            qn,
            kn,
            cs,
            pos,
            cutlass.Int32(rows),
            heads,
            src_ld,
            copy_v,
            stream,
        )
        _COMPILED[key] = compiled
    compiled(
        src,
        cutlass.Int64(k_off),
        cutlass.Int64(v_off),
        dst,
        qn,
        kn,
        cs,
        pos,
        cutlass.Int32(rows),
        stream,
    )
    return out
