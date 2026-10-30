# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 block head: LayerNorm + adaLN modulate + NVFP4 quantize in one kernel.

``(1 + scale) * LayerNorm(x) + shift`` (bf16 rounding points of the block's Inductor
kernel), quantized to NVFP4 with gate_up's activation global scale; the BF16 modulated
activation is never stored (QKV and gate_up read only the NVFP4 data). Two kernels:

* rows > 4096, or a uint8 row index: the warp-per-row CUDA kernel
  (``torch.ops.trtllm.flux3_adaln_nvfp4_warp``);
* rows <= 4096 with a [1, 1, D] row or a [1, 2, D] table + bool select (the 1024^2 mode
  blocks): the Triton kernel (`_adaln_nvfp4_triton`), where the two tied in-model
  (15.12 vs 15.17 us/call at M = 4096).
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ..gemm import _cpp_fakes  # noqa: F401 — fake for torch.compile tracing
from ..quantization._nvfp4_reference import quantize, swizzle_sf
from ._adaln_nvfp4_triton import D, adaln_nvfp4_triton

#: Up to this many rows a bool / no-select call takes the Triton kernel.
TRITON_MAX_ROWS = 4096
#: FLUX.3 pre_norm's epsilon.
_EPS = 1e-6


def _uses_warp(rows: int, select: torch.Tensor | None) -> bool:
    return rows > TRITON_MAX_ROWS or (select is not None and select.dtype == torch.uint8)


class AdalnNvfp4(OpWrapper):
    """``nvfp4((1 + scale[sel]) * LayerNorm(x) + shift[sel], global_scale)`` as ``(fp4 [M,
    D/2] uint8, swizzled sf [ceil128(M) * D/16] uint8)``, padding scale rows zero.

    ``x`` ``[1, S, D]`` contiguous bf16, D = 3072; ``select``: None (``scale`` / ``shift`` are
    the [1, 1, D] rows), ``[1, S, 1]`` bool (row 1 of [1, 2, D] tables where true) or uint8
    (row index into [1, K, D] tables); ``scale`` / ``shift``: bf16 with a unit last stride
    and 16-B aligned rows (any storage offset); ``global_scale`` fp32 [1]: gate_up's
    activation global scale; ``eps``: pre_norm's epsilon. The LayerNorm has no affine.
    """

    ARCHS = frozenset({Arch.SM_120})

    CELLS: tuple[Cell, ...] = (
        Cell(why="2048^2 mode blocks: [1, 1, D] row, warp kernel", spec=dict(m=16384, form="row")),
        Cell(
            why="2048^2 t2i single blocks: [1, 2, D] table + bool", spec=dict(m=16464, form="bool")
        ),
        Cell(
            why="2048^2 i2i single blocks: [1, 3, D] table + u8 index",
            spec=dict(m=32848, form="index"),
        ),
        Cell(why="1024^2 mode blocks: [1, 1, D] row, Triton kernel", spec=dict(m=4096, form="row")),
        Cell(
            why="1024^2 t2i single blocks: warp kernel, partial tile",
            spec=dict(m=4176, form="bool"),
        ),
        Cell(
            why="1024^2 i2i single blocks: 2 x 4096 + 80 rows, u8 index",
            spec=dict(m=8272, form="index"),
        ),
    )

    note = """
    Metadata consumed: none. The head of every NVFP4 block.

    Preconditions: D = 3072; pre_norm without weight and bias (both kernels drop the
    affine); global_scale the activation scale QKV and gate_up were prepared with.

    Numerics: the Triton kernel is bitwise Inductor's LN + modulate kernel for this block
    followed by FlashInfer's fast-math NVFP4 quantize. The warp kernel replays the same
    Welford tree with one leaf grouping differing: its modulated value is 1 bf16 ulp off on
    ~1e-6 of the elements, which moves an NVFP4 code or a block scale only where that value
    is the block max or sits on a code boundary; it was bitwise the Triton kernel's fp4 and
    scales in every tested case. Against `reference` (torch layer_norm, exact-division
    quantize) codes and scales can move by one step at rounding boundaries.
    """

    def __call__(
        self,
        x: torch.Tensor,
        select: torch.Tensor | None,
        scale: torch.Tensor,
        shift: torch.Tensor,
        global_scale: torch.Tensor,
        eps: float = _EPS,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = x.numel() // x.shape[-1]
        if _uses_warp(rows, select):
            return torch.ops.trtllm.flux3_adaln_nvfp4_warp(
                x,
                None if select is None else select.reshape(-1),
                scale,
                shift,
                global_scale,
                eps,
            )
        return adaln_nvfp4_triton(x, select, scale, shift, global_scale, eps)

    def reference(
        self,
        x: torch.Tensor,
        select: torch.Tensor | None,
        scale: torch.Tensor,
        shift: torch.Tensor,
        global_scale: torch.Tensor,
        eps: float = _EPS,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """fp32 LayerNorm (torch's, no affine) rounded to bf16, ``bf16(1 + scale)``,
        ``bf16(that * norm)``, ``bf16(that + shift)``; then the exact-division NVFP4 recipe
        with ``global_scale``, scales swizzled over ``ceil128(M)`` rows."""
        rows = x.numel() // D
        xf = x.reshape(rows, D).float()
        norm = torch.nn.functional.layer_norm(xf, (D,), eps=eps).to(torch.bfloat16).float()
        sc_t = scale.reshape(-1, D)
        sh_t = shift.reshape(-1, D)
        if select is None:
            sc, sh = sc_t[0].expand(rows, D), sh_t[0].expand(rows, D)
        else:
            idx = select.reshape(-1).to(torch.long)
            sc, sh = sc_t[idx], sh_t[idx]
        one_plus = (sc.float() + 1.0).to(torch.bfloat16).float()
        prod = (one_plus * norm).to(torch.bfloat16).float()
        x_mod = (prod + sh.float()).to(torch.bfloat16).float()
        codes, sf = quantize(x_mod, float(global_scale.float().reshape(()).item()))
        return codes, swizzle_sf(sf)

    def is_valid(
        self,
        x: torch.Tensor,
        select: torch.Tensor | None,
        scale: torch.Tensor,
        shift: torch.Tensor,
        global_scale: torch.Tensor,
        eps: float = _EPS,
    ) -> None:
        # The kernels read the selected table row without a bound check, and the Triton
        # kernel reads a uint8 select as a bool.
        if select is not None:
            rows_in_table = scale.shape[1]
            assert int(select.to(torch.int32).max()) < rows_in_table, "select points past the table"
            assert select.dtype == torch.bool or _uses_warp(x.numel() // D, select)
        assert global_scale.numel() == 1, "global_scale must hold exactly one element"


adaln_nvfp4 = AdalnNvfp4()
