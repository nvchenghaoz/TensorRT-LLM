# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SIMT projections that decode exact BF16 weights in registers."""

import cutlass
import cutlass.cute as cute
import cutlass.utils
from cuda.bindings import driver as cuda
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def _bits_float(bits: cutlass.Uint32, *, loc=None, ip=None) -> cutlass.Float32:
    return cutlass.Float32(llvm.bitcast(T.f32(), bits.ir_value(), loc=loc, ip=ip))


@dsl_user_op
def _fma(
    a: cutlass.Float32, b: cutlass.Float32, c: cutlass.Float32, *, loc=None, ip=None
) -> cutlass.Float32:
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [a.ir_value(), b.ir_value(), c.ir_value()],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            loc=loc,
            ip=ip,
        )
    )


@cute.kernel
def _attention_kernel(
    x: cute.Tensor,
    data: cute.Tensor,
    metadata: cute.Tensor,
    exceptions: cute.Tensor,
    output: cute.Tensor,
    tile: cutlass.Constexpr,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    row, batch_tile, _ = cute.arch.block_idx()
    sums = cute.make_rmem_tensor((tile,), cutlass.Float32)
    sums.fill(0.0)
    decoded = cute.make_rmem_tensor((8,), cutlass.Float32)
    activation = cute.make_rmem_tensor((8,), cutlass.BFloat16)
    words = cute.recast_tensor(data, cutlass.Uint64)
    for col in range(tid * 8, x.shape[1], 256 * 8):
        block = cutlass.Int64(row) * (x.shape[1] // 128) + col // 128
        header = cutlass.Uint64(metadata[block])
        position = col % 128
        signs_mantissas = words[block, position // 8]
        offset = 128 + position // 8 * 3
        codes = (
            cutlass.Uint32(data[block, offset])
            | (cutlass.Uint32(data[block, offset + 1]) << 8)
            | (cutlass.Uint32(data[block, offset + 2]) << 16)
        )
        count = cutlass.Int32(cute.arch.popc(codes & (codes >> 1) & (codes >> 2) & 0x249249))
        prefix = count
        for shift in cutlass.range_constexpr(2):
            other = cute.arch.shuffle_sync_up(
                prefix, 1 << shift, mask=cute.arch.activemask(), mask_and_clamp=28 << 8
            )
            if tid % 4 >= (1 << shift):
                prefix += other
        group = position // 32
        prior = prefix - count
        if group != 0:
            prior += cutlass.Int32((header >> (32 + group * 8)) & 255)
        exception_offset = cutlass.Int32((header >> 8) & 0xFFFFFFFF)
        for j in cutlass.range_constexpr(8):
            sm = cutlass.Uint32((signs_mantissas >> (j * 8)) & 255)
            code = (codes >> (j * 3)) & 7
            exponent = cutlass.Uint32(header & 255) - code
            if code == 7:
                exponent = cutlass.Uint32(exceptions[exception_offset + prior])
                prior += 1
            bits = ((sm & 128) << 8) | (exponent << 7) | (sm & 127)
            decoded[j] = _bits_float(bits << 16)
        for t in cutlass.range_constexpr(tile):
            token = batch_tile * tile + t
            if token < x.shape[0]:
                source = cute.make_tensor(
                    x.iterator + cute.assume(cutlass.Int64(token) * x.shape[1] + col, divby=8),
                    cute.make_layout((8,)),
                )
                cute.autovec_copy(source, activation)
                for j in cutlass.range_constexpr(8):
                    sums[t] = _fma(cutlass.Float32(activation[j]), decoded[j], sums[t])

    partial = cutlass.utils.SmemAllocator().allocate_tensor(
        cutlass.Float32, cute.make_layout((tile, 8)), 16
    )
    for t in cutlass.range_constexpr(tile):
        value = sums[t]
        # Preserve the native CUB reduction's increasing offsets, followed by
        # the ordered sum of the eight warp results. Only lane zero is consumed.
        for shift in cutlass.range_constexpr(5):
            value = value + cute.arch.shuffle_sync_down(value, 1 << shift)
        if tid % 32 == 0:
            partial[t, tid // 32] = value
    cute.arch.sync_threads()
    if tid == 0:
        for t in cutlass.range_constexpr(tile):
            value = cutlass.Float32(0)
            for warp in cutlass.range_constexpr(8):
                value = value + partial[t, warp]
            token = batch_tile * tile + t
            if token < x.shape[0]:
                output[cutlass.Int64(token), row] = cutlass.BFloat16(value)


@cute.kernel
def _head_kernel(
    x: cute.Tensor,
    data: cute.Tensor,
    metadata: cute.Tensor,
    exceptions: cute.Tensor,
    output: cute.Tensor,
    tile: cutlass.Constexpr,
) -> None:
    lane, warp, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    row = bx * 4 + warp
    sums = cute.make_rmem_tensor((tile,), cutlass.Float32)
    sums.fill(0.0)
    decoded = cute.make_rmem_tensor((4,), cutlass.Float32)
    if row < output.shape[1]:
        for group in range(x.shape[1] // 128):
            block = cutlass.Int64(row) * (x.shape[1] // 128) + group
            header = cutlass.Uint64(metadata[block])
            escape = cutlass.Int32(header >> 8)
            offset = 128 + (lane // 2) * 3 + lane % 2
            codes = (
                cutlass.Uint32(data[block, offset]) | (cutlass.Uint32(data[block, offset + 1]) << 8)
            ) >> ((lane % 2) * 4)
            prior = cutlass.Int32(0)
            for j in cutlass.range_constexpr(4):
                sm = cutlass.Uint32(data[block, lane * 4 + j])
                code = (codes >> (j * 3)) & 7
                exponent = cutlass.Uint32(header & 255) - code
                if escape != 0:
                    mask = cutlass.Uint32(cute.arch.vote_ballot_sync(code == 7))
                    if code == 7:
                        rank = cutlass.Int32(
                            cute.arch.popc(mask & ((cutlass.Uint32(1) << lane) - 1))
                        )
                        exponent = cutlass.Uint32(exceptions[escape - 1 + prior + rank])
                    prior += cutlass.Int32(cute.arch.popc(mask))
                bits = ((sm & 128) << 8) | (exponent << 7) | (sm & 127)
                decoded[j] = _bits_float(bits << 16)
            for t in cutlass.range_constexpr(tile):
                token = by * tile + t
                if token < x.shape[0]:
                    for j in cutlass.range_constexpr(4):
                        a = cutlass.Float32(x[cutlass.Int64(token), group * 128 + lane + j * 32])
                        sums[t] = _fma(a, decoded[j], sums[t])
        for t in cutlass.range_constexpr(tile):
            value = sums[t]
            for i in cutlass.range_constexpr(5):
                value = value + cute.arch.shuffle_sync_down(value, 16 >> i)
            token = by * tile + t
            if lane == 0 and token < x.shape[0]:
                output[cutlass.Int64(token), row] = cutlass.BFloat16(value)


class LosslessBf16Gemm:
    def __init__(self, reassociated: bool) -> None:
        self.reassociated = reassociated

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        data: cute.Tensor,
        metadata: cute.Tensor,
        exceptions: cute.Tensor,
        output: cute.Tensor,
        stream: cuda.CUstream,
    ) -> None:
        tile = 1 if x.shape[0] == 1 else 2 if x.shape[0] == 2 else 4
        if cutlass.const_expr(self.reassociated):
            _attention_kernel(x, data, metadata, exceptions, output, tile).launch(
                grid=(output.shape[1], cute.ceil_div(x.shape[0], tile), 1),
                block=(256, 1, 1),
                stream=stream,
            )
        else:
            _head_kernel(x, data, metadata, exceptions, output, tile).launch(
                grid=(cute.ceil_div(output.shape[1], 4), cute.ceil_div(x.shape[0], tile), 1),
                block=(32, 4, 1),
                stream=stream,
            )
