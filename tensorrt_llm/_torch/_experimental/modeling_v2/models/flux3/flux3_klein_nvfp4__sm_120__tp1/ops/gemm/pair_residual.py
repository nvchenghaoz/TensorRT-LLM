# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 block tail with the gated residual in one launch
(``torch.ops.trtllm.flux3_nvfp4_pair_residual``):

    out = x + gate[gate_index] * (attn_out(attn) + down(hidden))   (BF16 round after every op)

One persistent SM120 CUTLASS grouped launch: down's tiles (K = 9216) first, then
attn_out's (K = 3072), every tile claimed from one atomic counter. A down tile stores
BF16(alpha * acc) to a scratch mlp_y and publishes a per-warp flag; the attn_out tile
(m, n) waits on it and writes the residual. Tile instance and raster by row count:
M >= 8192 the 256x128 tiles walked AlongN, below that the 128x128 tiles at the
scheduler's default raster (all bitwise equal).

The kernel keeps a launch epoch and per-tile flags in a persistent workspace that must be
zero-filled once before the first call and never inside a CUDA graph capture
(`allocate_workspace`, at model load).
"""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

from ..quantization._nvfp4_reference import SF_VEC, dequantize_swizzled, swizzled_sf_numel
from . import _cpp_fakes  # noqa: F401 — fake for torch.compile tracing

#: Bytes of the persistent workspace (kPairResidualWorkspaceBytes): 8 KB of grouped-GEMM
#: arguments, the CUTLASS workspace, then 8 flags per 128x128 tile from 4 MB.
WORKSPACE_BYTES = 8 * 1024 * 1024

#: From this row count the 256x128 instance, walked AlongN: t2i single 16464 1033.6 ->
#: 1003.2 us and mode 16384 1012.6 -> 985.0 us for AlongN; the i2i single 32848 beats the
#: 128 instance at either raster. Below it (1024^2) the 128 instance at the default raster.
LARGE_M_MIN = 8192


def allocate_workspace(device: torch.device) -> torch.Tensor:
    """The zero-filled workspace for one device; allocate it at load, outside capture."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("the pair-residual workspace must be allocated outside graph capture")
    return torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device=device)


def launch_config(rows: int) -> tuple[bool, bool]:
    """(256x128 tile instance, AlongN raster) of a launch of ``rows`` output rows; the
    alternative to AlongN is the scheduler's default raster (AlongM)."""
    large = rows >= LARGE_M_MIN
    return large, large


def _aligned(t: torch.Tensor) -> torch.Tensor:
    """The kernel reads 16-B row segments: copy a view that is not 16-B aligned."""
    return t if t.data_ptr() % 16 == 0 else t.clone()


class Nvfp4PairResidual(OpWrapper):
    """``x + gate[gate_index] * (bf16(attn_alpha * attn @ Wo^T) + bf16(down_alpha * hidden @
    Wd^T))`` as bf16 shaped like ``x``, each sum and product rounded to BF16.

    ``down_act`` ``[M, 9216/2]`` / ``down_act_sf``: the SwiGLU epilogue's NVFP4 output;
    ``down_weight`` / ``down_weight_scale``: down_proj as loaded; ``down_alpha`` fp32 [1];
    ``attn_act`` ``[M, 3072/2]`` (any leading shape) / ``attn_act_sf``: attn_out's NVFP4
    activation as the FMHA writes it; ``attn_weight`` / ``attn_weight_scale`` /
    ``attn_alpha``: attn_out; ``x`` ``[..., N]`` bf16 with M rows (the residual stream);
    ``gate`` ``[1, R, N]`` bf16 row table; ``gate_index`` ``[1, M, 1]`` bool (row 1 where
    true) / uint8 (row index), or None (row 0); ``workspace``: `allocate_workspace`'s buffer.
    """

    ARCHS = frozenset({Arch.SM_120})

    # down 3072 x 9216 and attn_out 3072 x 3072 at the served M and gate forms.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="2048^2 mode blocks: [1, 1, D] gate row, 256x128 AlongN",
            spec=dict(m=16384, gate="row"),
        ),
        Cell(
            why="2048^2 t2i single blocks: [1, 2, D] table + bool select",
            spec=dict(m=16464, gate="bool"),
        ),
        Cell(
            why="2048^2 i2i single blocks: [1, 3, D] table + u8 index",
            spec=dict(m=32848, gate="index"),
        ),
        Cell(
            why="1024^2 mode blocks: 128x128 tiles, default raster", spec=dict(m=4096, gate="row")
        ),
        Cell(
            why="1024^2 t2i single blocks: partial last tile, bool select",
            spec=dict(m=4176, gate="bool"),
        ),
    )

    note = """
    Metadata consumed: none. Every NVFP4 block's tail (single block 27 on its first 16384
    rows).

    Preconditions: the workspace zero-filled once before the first call (outside capture)
    and used by this op alone, stream-ordered (the launch epoch lives in it); `out` never
    aliases `x` (the op allocates it).

    Numerics: BF16(alpha * acc) per GEMM is the stock SM120 CUTLASS NVFP4 GEMM's; the
    residual is bf16(x + bf16(gate * bf16(attn_y + mlp_y))) with fp32 .rn ops, Inductor's
    emulate_precision_casts arithmetic for that expression. Both instances and every raster
    give the same bytes. Against `reference` (fp32 GEMMs) a BF16 GEMM value can move by one
    ulp and carry that through the residual roundings.

    Launch: an args kernel then the GEMM, both with the PDL attribute.
    """

    def __call__(
        self,
        down_act: torch.Tensor,
        down_weight: torch.Tensor,
        down_act_sf: torch.Tensor,
        down_weight_scale: torch.Tensor,
        down_alpha: torch.Tensor,
        attn_act: torch.Tensor,
        attn_weight: torch.Tensor,
        attn_act_sf: torch.Tensor,
        attn_weight_scale: torch.Tensor,
        attn_alpha: torch.Tensor,
        x: torch.Tensor,
        gate: torch.Tensor,
        gate_index: torch.Tensor | None,
        workspace: torch.Tensor,
    ) -> torch.Tensor:
        width = down_weight.shape[0]
        attn_2d = attn_act.reshape(-1, attn_act.shape[-1])
        rows = attn_2d.shape[0]
        t256, along_n = launch_config(rows)
        index = (
            None if gate_index is None else gate_index.reshape(-1).contiguous().view(torch.uint8)
        )
        out = torch.ops.trtllm.flux3_nvfp4_pair_residual(
            down_act,
            down_weight,
            down_act_sf.reshape(-1),
            down_weight_scale.reshape(-1),
            down_alpha,
            attn_2d,
            attn_weight,
            attn_act_sf.reshape(-1),
            attn_weight_scale.reshape(-1),
            attn_alpha,
            _aligned(x.reshape(rows, width).contiguous()),
            _aligned(gate.reshape(-1, width).contiguous()),
            index,
            workspace,
            t256,
            along_n,
        )
        return out.reshape(x.shape)

    def reference(
        self,
        down_act: torch.Tensor,
        down_weight: torch.Tensor,
        down_act_sf: torch.Tensor,
        down_weight_scale: torch.Tensor,
        down_alpha: torch.Tensor,
        attn_act: torch.Tensor,
        attn_weight: torch.Tensor,
        attn_act_sf: torch.Tensor,
        attn_weight_scale: torch.Tensor,
        attn_alpha: torch.Tensor,
        x: torch.Tensor,
        gate: torch.Tensor,
        gate_index: torch.Tensor | None,
        workspace: torch.Tensor,
    ) -> torch.Tensor:
        """Each GEMM in fp32 on the dequantized operands (TF32 off) times its alpha, rounded
        to BF16; then the residual with a BF16 rounding after the sum, the gate product and
        the final add. ``workspace`` is ignored."""
        width = down_weight.shape[0]
        attn_2d = attn_act.reshape(-1, attn_act.shape[-1])
        rows = attn_2d.shape[0]

        def gemm(act, act_sf, weight, weight_scale, alpha):
            a = dequantize_swizzled(act, act_sf)
            w = dequantize_swizzled(weight, weight_scale)
            with true_fp32_matmul():
                return (alpha.float().reshape(()) * (a @ w.t())).to(torch.bfloat16).float()

        mlp_y = gemm(down_act, down_act_sf, down_weight, down_weight_scale, down_alpha)
        attn_y = gemm(attn_2d, attn_act_sf, attn_weight, attn_weight_scale, attn_alpha)
        table = gate.reshape(-1, width).float()
        if gate_index is None:
            g = table[0].expand(rows, width)
        else:
            g = table[gate_index.reshape(-1).to(torch.long)]
        y = (attn_y + mlp_y).to(torch.bfloat16).float()
        y = (g * y).to(torch.bfloat16).float()
        out = (x.reshape(rows, width).float() + y).to(torch.bfloat16)
        return out.reshape(x.shape)

    def is_valid(
        self,
        down_act: torch.Tensor,
        down_weight: torch.Tensor,
        down_act_sf: torch.Tensor,
        down_weight_scale: torch.Tensor,
        down_alpha: torch.Tensor,
        attn_act: torch.Tensor,
        attn_weight: torch.Tensor,
        attn_act_sf: torch.Tensor,
        attn_weight_scale: torch.Tensor,
        attn_alpha: torch.Tensor,
        x: torch.Tensor,
        gate: torch.Tensor,
        gate_index: torch.Tensor | None,
        workspace: torch.Tensor,
    ) -> None:
        width = down_weight.shape[0]
        # The epilogue reads table row gate_index[r] without a bound check.
        if gate_index is not None:
            rows_in_table = gate.numel() // width
            assert int(gate_index.to(torch.int32).max()) < rows_in_table, (
                "gate_index points past the gate table"
            )
        # A launch on a workspace that was never zero-filled can match a stale flag.
        assert workspace.numel() >= WORKSPACE_BYTES, "workspace must be allocate_workspace's"
        rows = down_act.shape[0]
        k = down_act.shape[1] * 2
        assert down_act_sf.numel() >= swizzled_sf_numel(rows, k // SF_VEC), (
            "down_act_sf is shorter than the padded swizzled rectangle of the activation"
        )


nvfp4_pair_residual = Nvfp4PairResidual()
