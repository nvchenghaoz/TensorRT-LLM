# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Q/K RMSNorm + RoPE of the packed QKV GEMM output straight to per-tensor E4M3 Q/K/V with a
dynamic amax: the attention prologue of ``image.0``, the FP8-rowwise mode block.

Two launches of one CuTe DSL kernel (``_qknorm_rope_fp8``): an amax pass that rounds every
normed and rotated value to BF16 and folds it into per-CTA partials, then a pass that reduces the
partials, writes the scales and quantizes. No BF16 q/k is stored.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._qknorm_rope_fp8 import _MIN_SCALING_FACTOR, qknorm_rope_fp8_op

_E4M3_MAX = 448.0


def _cell(why: str, rows: int) -> Cell:
    return Cell(why=why, spec=dict(rows=rows, heads=24, layout="packed_gemm_columns"))


class QknormRopeFp8(OpWrapper):
    """``(qkv8 [3, M, W] float8_e4m3fn, scales [8] float32)`` from the BF16 GEMM output's q, k, v.

    Per row and head: ``x * rsqrt(mean(x^2) + 1e-6) * w`` (``q_norm`` / ``k_norm``, 128 BF16
    weights), RoPE on the (even, odd) pairs with ``cos_sin[positions[row]]`` (``[:64]`` cos,
    ``[64:]`` sin), rounded to BF16. Per plane (q, k, v): ``scale = bf16(max(amax / 448,
    1 / (448 * 512)))``, ``factor = 1 / scale`` (both ``div.rn``), the plane ``e4m3(x * factor)``
    (satfinite). ``scales = [q_scale, k_scale, v_scale, (q_scale * k_scale) * sm_scale,
    q_factor, k_factor, v_factor, 0]`` -- the layout the FP8 attention entries read.

    ``q``, ``k``, ``v``: BF16 ``[M, W]`` (W = 128 H, H % 8 == 0) views sharing one row stride
    (the target passes the three column slices of the ``[M, 3W]`` GEMM output); ``cos_sin``:
    FP32 ``[P, 128]``; ``positions``: int64 ``[M]`` rows of ``cos_sin``; ``sm_scale``: the
    softmax scale.

    The source rows are dropped from L2 as they are consumed (when 128-B aligned), so ``q``,
    ``k`` and ``v`` are undefined after the call: the caller must not read them again.

    Fusion boundary: the norms, RoPE, the amax and the quantize. The GEMM and the attention are
    the caller's.
    """

    ARCHS = frozenset({Arch.SM_120})

    CELLS: tuple[Cell, ...] = (
        _cell("image.0 at 2048^2: M = 16384 rows of the [M, 9216] GEMM output", 16384),
        _cell("image.0 at 1024^2: M = 4096 (fewer rows than 2 waves of the grid)", 4096),
    )

    note = """
    Metadata consumed: none.

    Numerics: the V plane, its scale and factor are bitwise `reference`'s (amax of the BF16
    input, `div.rn`, `mul.rn`, `cvt.rn.satfinite`). For Q and K the kernel sums the squares in
    its lane order (four 32-column groups per lane, then a butterfly) where `reference` uses
    torch's sum, so `inv_rms` can differ in its last FP32 bits; a rotated value then rounds to
    the other BF16 neighbour only next to a BF16 midpoint, and its E4M3 code moves only if that
    also crosses an E4M3 midpoint. RoPE is `fma(even, cos, -(odd * sin))` and
    `fma(odd, cos, even * sin)` in both.

    Preconditions: rows 16-B aligned; positions in range of cos_sin (not checked by the
    kernel, see is_valid).
    """

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_norm: torch.Tensor,
        k_norm: torch.Tensor,
        cos_sin: torch.Tensor,
        positions: torch.Tensor,
        sm_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return qknorm_rope_fp8_op(q, k, v, q_norm, k_norm, cos_sin, positions, sm_scale)

    def reference(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_norm: torch.Tensor,
        k_norm: torch.Tensor,
        cos_sin: torch.Tensor,
        positions: torch.Tensor,
        sm_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """FP32 torch: the norm and RoPE as above (the two fused multiply-adds evaluated in FP64,
        exact before the one FP32 rounding), BF16-rounded, then the per-plane scale and E4M3."""
        rows, width = q.shape
        heads = width // 128
        cs = cos_sin.float()[positions]  # [M, 128]
        cosine = cs[:, None, :64].double()
        sine = cs[:, None, 64:].double()

        def norm_rope(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            x = x.float().reshape(rows, heads, 128)
            inv_rms = torch.rsqrt(x.pow(2).sum(-1, keepdim=True) / 128.0 + 1.0e-6)
            w = w.float()
            even = x[..., 0::2] * inv_rms * w[0::2]
            odd = x[..., 1::2] * inv_rms * w[1::2]
            rot_even = (even.double() * cosine - (odd * sine.float()).double()).float()
            rot_odd = (odd.double() * cosine + (even * sine.float()).double()).float()
            return torch.stack((rot_even, rot_odd), -1).reshape(rows, width).to(torch.bfloat16)

        planes = (norm_rope(q, q_norm), norm_rope(k, k_norm), v.to(torch.bfloat16))
        qkv8 = torch.empty((3, rows, width), dtype=torch.float8_e4m3fn, device=q.device)
        scales = torch.zeros(8, dtype=torch.float32, device=q.device)
        for p, x in enumerate(planes):
            amax = x.float().abs().nan_to_num(0.0).amax()
            min_scale = torch.tensor(_MIN_SCALING_FACTOR, device=q.device)
            scale = torch.maximum(amax / _E4M3_MAX, min_scale)
            scale = scale.to(torch.bfloat16).float()
            factor = 1.0 / scale
            qkv8[p] = (x.float() * factor).clamp(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn)
            scales[p] = scale
            scales[4 + p] = factor
        scales[3] = (scales[0] * scales[1]) * torch.tensor(sm_scale, dtype=torch.float32)
        return qkv8, scales

    def is_valid(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_norm: torch.Tensor,
        k_norm: torch.Tensor,
        cos_sin: torch.Tensor,
        positions: torch.Tensor,
        sm_scale: float,
    ) -> None:
        # The kernel reads these through raw pointers of fixed element types: a different dtype,
        # a strided tensor or an out-of-range position reads the wrong bytes without an error.
        assert positions.dtype == torch.int64 and positions.is_contiguous(), (
            "positions must be contiguous int64"
        )
        assert positions.numel() >= q.shape[0], "positions must hold one entry per row"
        assert cos_sin.dtype == torch.float32 and cos_sin.is_contiguous(), (
            "cos_sin must be contiguous float32"
        )
        assert cos_sin.dim() == 2 and cos_sin.shape[1] == 128, "cos_sin must be [P, 128]"
        assert int(positions[: q.shape[0]].max()) < cos_sin.shape[0], "a position is past cos_sin"
        for name, w in (("q_norm", q_norm), ("k_norm", k_norm)):
            assert w.dtype == torch.bfloat16 and w.is_contiguous() and w.numel() == 128, (
                f"{name} must be 128 contiguous bfloat16 weights"
            )
        assert q.dtype == k.dtype == v.dtype == torch.bfloat16, "q, k and v must be bfloat16"


qknorm_rope_fp8 = QknormRopeFp8()
