# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM120 decode output projection (wo_a + wo_b) on FlashInfer's experimental b12x GEMMs.

Main's SM120 chain for the grouped 128x128-block FP8 ``o_a_proj`` is
``mla_rope_inplace`` (BF16 inverse RoPE) -> 1x128 UE8M0 per-token quant ->
SM120 block-scaled BMM -> ``copy_`` into ``o_lora`` -> ``o_b_proj``
(``mxfp8_quantize`` + b12x MXFP8 GEMM, then the row-parallel all-reduce).
For decode row counts this module runs

1. one Triton launch with the same BF16 inverse RoPE and the same 1x128
   UE8M0 quantization, writing the scale byte into the four 1x32 MXFP8 slots
   b12x reads;
2. one grouped b12x MXFP8 dense GEMM over all local groups (wo_a), reading the
   resmoothed E4M3 bytes and UE8M0 128x128 block scales main uses;
3. ``o_b_proj`` as one b12x GEMM that quantizes its group-major BF16 input to
   MXFP8 inside the CTA (byte-equal to ``mxfp8_quantize``), then
   ``o_b_proj.all_reduce`` exactly as ``Linear.forward`` calls it. When the
   Linear has anything this cannot mirror (bias, LoRA, fused GEMM+all-reduce,
   NCCL window output, locality-domain shards) it runs ``o_b_proj(o_lora)``.

Every stage is byte-identical to main's (checked bitwise at M=1..8).
"""

import os
from typing import Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger

_ENV = "TRTLLM_B12X_WO_PROJECTION"
_ENV_WO_B = "TRTLLM_B12X_WO_B_FUSED"
# wo_b fused-quant producer layout (FlashInfer b12x fq_pipeline wide=True: 16 lanes
# per row, 2 rows per pass, packed sA stores). Bitwise equal to the pinned
# producer; used only at the decode row counts where it is faster in main's chain.
_ENV_WO_B_WIDE_ROWS = "TRTLLM_B12X_WO_B_WIDE_ROWS"
_WO_B_WIDE_ROWS_DEFAULT = "2,3,4"
MAX_DECODE_ROWS = 8


def _env_on(name: str) -> bool:
    return os.environ.get(name, "1").strip().lower() not in ("0", "false", "off", "")


def _wo_b_wide_rows() -> frozenset:
    raw = os.environ.get(_ENV_WO_B_WIDE_ROWS, _WO_B_WIDE_ROWS_DEFAULT).strip().lower()
    if raw in ("", "0", "off", "none", "false"):
        return frozenset()
    return frozenset(int(x) for x in raw.split(",") if x.strip())


def b12x_wo_projection_enabled() -> bool:
    return _env_on(_ENV)


@triton.jit
def _inv_rope_quant128_kernel(
    o,
    positions,
    cos_sin,
    values,
    scale_mma,
    o_stride_t,
    cs_stride_pos,
    v_stride_t,
    v_stride_d,
    v_stride_g,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    fp8_max,
    fp8_min,
    rope_out,
    GROUP_WIDTH: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    HALF_ROPE: tl.constexpr,
    NB: tl.constexpr,
    ROPE_VARIANT: tl.constexpr,
    STORE_ROPE: tl.constexpr,
):
    token = tl.program_id(0)
    group = tl.program_id(1)
    cb = tl.program_id(2)
    kc = cb * NB + tl.arange(0, NB)
    d = kc[:, None] * 128 + tl.arange(0, 128)[None, :]
    base = o + token * o_stride_t + group * GROUP_WIDTH
    src = tl.load(base + d).to(tl.float32)
    head_d = d % HEAD_DIM
    is_rope = head_d >= NOPE_DIM
    partner = tl.load(base + (d ^ 1), mask=is_rope, other=0.0).to(tl.float32)
    pos = tl.load(positions + token)
    ci = tl.maximum((head_d - NOPE_DIM) >> 1, 0)
    cbase = cos_sin + pos.to(tl.int64) * cs_stride_pos
    c = tl.load(cbase + ci, mask=is_rope, other=1.0)
    s = tl.load(cbase + HALF_ROPE + ci, mask=is_rope, other=0.0)
    # mla_rope_inplace (interleaved, inverse): coef = (cos, -sin),
    # x' = coef.x * x - coef.y * y, y' = coef.x * y + coef.y * x, rounded to BF16.
    ns = -s
    if ROPE_VARIANT == 0:
        r_even = tl.fma(c, src, -(ns * partner))
        r_odd = tl.fma(c, src, ns * partner)
    else:
        # Alternative contraction order (diagnostics only).
        r_even = tl.fma(-ns, partner, c * src)
        r_odd = tl.fma(ns, partner, c * src)
    rot = tl.where((head_d & 1) == 0, r_even, r_odd)
    rot = rot.to(tl.bfloat16).to(tl.float32)
    v = tl.where(is_rope, rot, src)
    if STORE_ROPE:
        tl.store(rope_out + token * o_stride_t + group * GROUP_WIDTH + d, v.to(tl.bfloat16))
    # fp8_utils._per_token_quant_and_transform_kernel (SCALE_UE8M0) arithmetic.
    amax = tl.maximum(tl.max(tl.abs(v), axis=1), 1e-10)
    sc = amax / fp8_max
    sc = tl.exp2(tl.ceil(tl.log2(tl.abs(sc))))
    q = tl.clamp(v / sc[:, None], fp8_min, fp8_max).to(values.dtype.element_ty)
    tl.store(values + token * v_stride_t + d * v_stride_d + group * v_stride_g, q)
    byte = ((sc.to(tl.int32, bitcast=True) >> 23) & 0xFF).to(tl.uint8)
    j = tl.arange(0, 4)
    row32 = token % 32
    row4 = (token // 32) % 4
    tile_m = token // 128
    addr = (
        row32 * s0 + row4 * s1 + tile_m * s2 + j[None, :] * s3 + kc[:, None] * s4 + group * s5
    )
    tl.store(scale_mma + addr, tl.broadcast_to(byte[:, None], (NB, 4)))


def inv_rope_quant128(
    o3: torch.Tensor,
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
    x_q,
    *,
    groups: int,
    group_width: int,
    nope_dim: int,
    rope_dim: int,
    nb: int = 4,
    rope_variant: int = 0,
    rope_out: Optional[torch.Tensor] = None,
) -> None:
    """BF16 inverse RoPE + 1x128 UE8M0 quant of ``o3`` [M, heads, head_dim] into ``x_q``.

    ``o3`` is not modified. ``x_q`` is a b12x ``MXFP8Rows`` with values
    ``[M, group_width, groups]`` and the dense-GEMM MMA scale view.
    """
    m = o3.shape[0]
    head_dim = o3.shape[-1]
    if (
        o3.stride(-1) != 1
        or o3.stride(-2) != head_dim
        or group_width % (128 * nb)
        or cos_sin.stride(-1) != 1
    ):
        raise ValueError("inv_rope_quant128 requires contiguous heads and cos/sin rows")
    sm = x_q.scale_mma.view(torch.uint8)
    vals = x_q.values
    _inv_rope_quant128_kernel[(m, groups, group_width // 128 // nb)](
        o3,
        positions,
        cos_sin,
        vals,
        sm,
        o3.stride(0),
        cos_sin.stride(0),
        vals.stride(0),
        vals.stride(1),
        vals.stride(2) if vals.ndim == 3 else 0,
        *sm.stride(),
        448.0,
        -448.0,
        rope_out if rope_out is not None else o3,
        GROUP_WIDTH=group_width,
        HEAD_DIM=head_dim,
        NOPE_DIM=nope_dim,
        HALF_ROPE=rope_dim // 2,
        NB=nb,
        ROPE_VARIANT=rope_variant,
        STORE_ROPE=rope_out is not None,
        num_warps=4,
    )


def _direct_wo_b_supported(linear) -> bool:
    """Whether ``o_b_proj``'s forward is exactly MXFP8 GEMM (+ plain all-reduce)."""
    from tensorrt_llm._torch.modules.linear import MXFP8LinearMethod, TensorParallelMode

    method = getattr(linear, "quant_method", None)
    if not isinstance(method, MXFP8LinearMethod) or not method.use_cutlass:
        return False
    if linear.tp_mode != TensorParallelMode.ROW or linear.bias is not None:
        return False
    if getattr(linear, "lora", None) is not None:
        return False
    if getattr(linear, "partition_plan", None) is not None and linear.partition_plan.enabled:
        return False
    if getattr(linear, "use_fused_gemm_allreduce", False):
        return False
    if linear.reduce_output and linear.all_reduce is None:
        return False
    weight, scale = linear.weight, linear.weight_scale
    n, k = weight.shape
    return (
        weight.dtype == torch.float8_e4m3fn
        and n % 128 == 0
        and k % 128 == 0
        and scale.numel() == n * k // 32
    )


class B12xWOProjection:
    """Packed b12x views of a CSA2 module's ``o_a_proj`` and ``o_b_proj`` weights.

    E4M3 values are zero-copy views of the live parameters. Only the wo_a
    UE8M0 scales are expanded (G * rank * group_width / 32 bytes); the wo_b
    scale view reuses ``o_b_proj.weight_scale``'s 128x4-swizzled bytes.
    """

    def __init__(self, module) -> None:
        from b12x.gemm._shared import wo_mxfp8 as W

        self._W = W
        weight = module.o_a_proj
        groups, rank, group_width = weight.shape
        block = module._o_a_proj_block_scale  # fp32 powers of two [G, rank/128, gw/128]
        exps = torch.log2(block.float())
        if not bool((exps == exps.round()).all()):
            raise ValueError("b12x WO projection needs UE8M0 (power-of-two) wo_a block scales")
        scale_u8 = (exps.round() + 127).to(torch.uint8).contiguous()
        self.wo_a = W.pack_fp8_block_scaled_weight_mxfp8(
            weight.view(groups * rank, group_width),
            scale_u8,
            m=rank,
            k=group_width,
            num_groups=groups,
        )
        if self.wo_a.values.data_ptr() != weight.data_ptr():
            raise RuntimeError("b12x WO projection must view the live o_a_proj storage")
        self.groups = groups
        self.rank = rank
        self.group_width = group_width
        self.heads = module.num_heads_tp
        self.head_dim = module.qk_head_dim
        self.nope_dim = module.qk_nope_head_dim
        self.rope_dim = module.qk_rope_head_dim
        self.cos_sin = module.inverse_rotary_emb.rotary_cos_sin
        self.alpha = W._cached_alpha_one(weight.device)
        self.o_b_proj = module.o_b_proj
        self.wo_b = None
        if _env_on(_ENV_WO_B) and _direct_wo_b_supported(self.o_b_proj):
            wb, wsf = self.o_b_proj.weight, self.o_b_proj.weight_scale
            n, k = wb.shape
            sfb = (
                wsf.view(torch.uint8)
                .view(torch.float8_e8m0fnu)
                .view(1, n // 128, k // 128, 32, 4, 4)
                .permute(3, 4, 1, 5, 2, 0)
            )
            if sfb.data_ptr() != wsf.data_ptr() or k != groups * rank:
                raise RuntimeError("b12x wo_b must view the live o_b_proj weight scales")
            self.wo_b = W.MXFP8Rows(values=wb, scale_rows=sfb, scale_mma=sfb)
            self.hidden = n
        self._wo_b_wide = _wo_b_wide_rows()
        self._warm_rows = set()

    def _wo_b(self, tmp: torch.Tensor, out: torch.Tensor, m: int) -> None:
        # (b_first, wide, prefetch): b_first stays off (it races, see FlashInfer
        # dense_gemm fused_quant_a_b_first); the pinned producer elsewhere.
        pipe = (False, True, 0) if m in self._wo_b_wide else (False, False, 0)
        self._W.wo_b_dense_gemm_fused_quant_mxfp8(
            tmp, self.wo_b, out=out, expected_m=m, sfb_k_replicated=False, fq_pipeline=pipe
        )

    def accepts(self, output: torch.Tensor) -> bool:
        m = output.shape[0]
        if not (1 <= m <= MAX_DECODE_ROWS) or output.dtype != torch.bfloat16:
            return False
        if not output.is_contiguous():
            return False
        # Never JIT-compile a CuTe DSL GEMM inside CUDA-graph capture: rows not
        # warmed eagerly keep main's chain (exact either way).
        if torch.cuda.is_current_stream_capturing() and m not in self._warm_rows:
            return False
        return True

    def o_lora(self, output: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """wo_a only: tmp [M, rank, groups] (physical [groups, M, rank]) BF16."""
        W = self._W
        m = output.shape[0]
        o3 = output.view(m, self.heads, self.head_dim)
        values, scale_rows, scale_phys = W.empty_mxfp8_rows_bases(
            m,
            self.group_width,
            num_groups=self.groups,
            device=output.device,
            initialize_scales=False,
        )
        x_q = W.mxfp8_rows_from_bases(
            values, scale_rows, scale_phys, m, self.group_width, num_groups=self.groups
        )
        inv_rope_quant128(
            o3,
            positions.reshape(-1),
            self.cos_sin,
            x_q,
            groups=self.groups,
            group_width=self.group_width,
            nope_dim=self.nope_dim,
            rope_dim=self.rope_dim,
        )
        tmp = W.empty_dense_gemm_mnl_view(
            m, self.rank, self.groups, device=output.device, dtype=torch.bfloat16
        )
        W.wo_a_dense_gemm_mxfp8(
            x_q,
            self.wo_a,
            out=tmp,
            alpha=self.alpha,
            expected_m=m,
            sfb_k_replicated=True,
        )
        return tmp

    def _use_direct_wo_b(self) -> bool:
        if self.wo_b is None:
            return False
        lin = self.o_b_proj
        # Linear writes its GEMM output into the NCCL window when that is live.
        return not (
            lin.reduce_output
            and lin.quant_method.supports_nccl_symmetric_memory_window_output
            and lin.all_reduce.uses_nccl_symmetric_memory_window()
        )

    def __call__(self, output: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Return o_b_proj(o_a_proj(inverse_rope(output))) as [M, hidden]."""
        m = output.shape[0]
        tmp = self.o_lora(output, positions)
        if self._use_direct_wo_b():
            out = torch.empty((m, self.hidden, 1), dtype=torch.bfloat16, device=output.device)
            self._wo_b(tmp, out, m)
            result = out[:, :, 0]
            if self.o_b_proj.reduce_output:
                result = self.o_b_proj.all_reduce(result, all_reduce_params=None)
        else:
            # tmp is physical [groups, M, rank]; o_b_proj wants [M, groups * rank].
            o_lora = tmp.permute(0, 2, 1).reshape(m, self.groups * self.rank).contiguous()
            result = self.o_b_proj(o_lora)
        if not torch.cuda.is_current_stream_capturing():
            self._warm_rows.add(m)
        return result

    def prewarm(self, device: torch.device) -> None:
        """Compile both GEMMs for every decode row count outside graph capture."""
        W = self._W
        for m in range(1, MAX_DECODE_ROWS + 1):
            dummy = torch.zeros((m, self.heads * self.head_dim), dtype=torch.bfloat16, device=device)
            positions = torch.zeros((m,), dtype=torch.int32, device=device)
            tmp = self.o_lora(dummy, positions)
            if self.wo_b is not None:
                out = torch.empty((m, self.hidden, 1), dtype=torch.bfloat16, device=device)
                self._wo_b(tmp, out, m)
            self._warm_rows.add(m)
        torch.cuda.current_stream(device).synchronize()


def build_b12x_wo_projection(module, prewarm: bool = True) -> Optional[B12xWOProjection]:
    """Build the packed projection for an SM120 grouped-128 FP8 CSA2 module, or None."""
    if not b12x_wo_projection_enabled() or get_sm_version() != 120:
        return None
    if not getattr(module, "_o_a_fp8_block128", False):
        return None
    if getattr(module, "_o_a_proj_block_scale", None) is None:
        return None
    if module.inverse_rotary_emb.is_neox or module.o_a_proj.dtype != torch.float8_e4m3fn:
        return None
    try:
        projection = B12xWOProjection(module)
        if prewarm:
            projection.prewarm(module.o_a_proj.device)
        return projection
    except Exception as error:  # pragma: no cover - optional backend
        logger.warning_once(
            f"b12x WO projection unavailable ({error!r}); using the SM120 BMM chain.",
            key="csa2_b12x_wo_unavailable",
        )
        return None
