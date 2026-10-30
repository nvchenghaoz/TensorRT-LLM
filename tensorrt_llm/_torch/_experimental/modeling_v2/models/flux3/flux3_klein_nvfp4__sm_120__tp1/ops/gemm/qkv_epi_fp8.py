# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 QKV GEMM straight to E4M3 attention operands (``torch.ops.trtllm.flux3_qkv_epi_fp8``).

One batched (L = 3) SM120 CUTLASS NVFP4 GEMM over the q, k and v planes of the fused QKV
weight whose epilogue does, for q/k, BF16(alpha_p * acc) -> per-(row, head) RMSNorm ->
RoPE -> times the static Q/K factor -> E4M3, and for v stores BF16(alpha_v * acc) plus one
max|v| partial per tile; then a V-only quantize reduces the partials, writes the eight
attention scales and quantizes V. The BF16 q/k planes are never written.

RoPE comes from a compact per-axis table (`build_rope_compact`, built once per request from
the [S, 128] fp32 cos|sin table) so a tile reads a few hundred hot entries instead of 128
rows x 512 B.
"""

from __future__ import annotations

from functools import lru_cache

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

_HEAD_DIM = 128
_ROPE_AXES = 4
_ROPE_SLOTS = 8  # lane slot s = q + 4 wn of the epilogue (its pair p0)
#: fp32(1 / (448 * 512)), the per-tensor attention scale floor.
_MIN_SCALE_BITS = 0x36924925
#: The static Q/K bound's constants: 11.313743 (sqrt(128) as the kernel carries it, not the
#: round-to-nearest fp32), 1 + 2^-6, 1e-6 and 2^-7.
_SQRT128_BITS = 0x41350517
_MARGIN_BITS = 0x3F820000


def _f32(bits: int) -> float:
    return torch.tensor([bits], dtype=torch.int32).view(torch.float32).item()


@lru_cache(maxsize=None)
def _rope_columns(device: torch.device) -> torch.Tensor:
    """[4, 32] cos|sin columns of each axis entry: slot s, k -> cos(16a+s), cos(16a+s+8),
    sin(16a+s), sin(16a+s+8) (k = 0..3)."""
    cols = [
        [(k >> 1) * 64 + 16 * a + s + 8 * (k & 1) for s in range(_ROPE_SLOTS) for k in range(4)]
        for a in range(_ROPE_AXES)
    ]
    return torch.tensor(cols, dtype=torch.long, device=device)


def build_rope_compact(cos_sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(rope_tab fp32 [4S, 32], rope_idx int32 [S, 4]) for the fp32 [S, 128] cos|sin table
    (cos of pair i at column i, sin at 64 + i; row = token).

    Device ops of fixed shape only (no host sync; graph-capturable). rope_tab[4r + a] is row
    r's axis-a slice (pairs 16a .. 16a+15) in the epilogue's lane order; rope_idx[r][a] =
    4 rep + a for the first row rep whose slice key (its first cos and sin words) matches,
    kept only where all 32 words of the two slices are bit-equal, else 4r + a (the row's own
    entry). So rope_tab[rope_idx[r][a]] == row r's slice bit for bit for every row and axis.
    """
    if not (
        cos_sin.dtype == torch.float32
        and cos_sin.dim() == 2
        and cos_sin.shape[1] == 2 * 64
        and cos_sin.is_contiguous()
        and cos_sin.is_cuda
        and 0 < 4 * cos_sin.shape[0] < 2**31
    ):
        raise ValueError("build_rope_compact takes a contiguous CUDA fp32 [S, 128] cos|sin table")
    rows = cos_sin.shape[0]
    dev = cos_sin.device
    perm = _rope_columns(dev).reshape(-1)
    tab = cos_sin.view(torch.int32)[:, perm].contiguous()  # [S, 4 * 32]
    slab = tab.view(rows, _ROPE_AXES, 32)
    key = (slab[:, :, 0].long() << 32) | (slab[:, :, 2].long() & 0xFFFFFFFF)  # [S, 4]
    sorted_key, order = torch.sort(key.t().contiguous(), dim=1, stable=True)  # [4, S]
    start = torch.ones_like(sorted_key, dtype=torch.bool)
    start[:, 1:] = sorted_key[:, 1:] != sorted_key[:, :-1]
    pos = torch.arange(rows, device=dev).expand(_ROPE_AXES, rows)
    seg = torch.where(start, pos, torch.zeros_like(pos)).cummax(dim=1).values
    rep = torch.empty_like(order).scatter_(1, order, order.gather(1, seg)).t()  # [S, 4]
    axis = torch.arange(_ROPE_AXES, device=dev)
    ent = rep * _ROPE_AXES + axis
    own = torch.arange(rows, device=dev).unsqueeze(1) * _ROPE_AXES + axis
    entries = tab.view(rows * _ROPE_AXES, 32)
    match = (entries[ent.reshape(-1)].view(rows, _ROPE_AXES, 32) == slab).all(dim=-1)
    idx = torch.where(match, ent, own).to(torch.int32).contiguous()
    return entries.view(torch.float32), idx


def rope_from_compact(
    rope_tab: torch.Tensor, rope_idx: torch.Tensor, rows: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """(cos, sin) fp32 [rows, 64] per RoPE pair, read back from the compact table."""
    entries = rope_tab.view(-1, _ROPE_SLOTS, 4)[rope_idx[:rows].long()]  # [rows, 4 axes, 8, 4]
    # Pair 16a + s + 8j: cos at k = j, sin at k = 2 + j.
    cos = torch.stack([entries[..., 0], entries[..., 1]], dim=2).reshape(rows, 64)
    sin = torch.stack([entries[..., 2], entries[..., 3]], dim=2).reshape(rows, 64)
    return cos, sin


def _static_scale(norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(scale, factor) fp32 of the static Q/K bound, the kernel's fp32 ops in order."""
    m = norm.float().abs().amax()
    b = (_f32(_SQRT128_BITS) * m) * _f32(_MARGIN_BITS)
    min_scale = torch.tensor(_f32(_MIN_SCALE_BITS), device=norm.device)
    s = torch.maximum(b / 448.0, min_scale).to(torch.bfloat16).float()
    return s, 1.0 / s


def _e4m3(x: torch.Tensor) -> torch.Tensor:
    """fp32 -> E4M3 bytes, round-to-nearest-even, saturating at +-448."""
    return x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)


class QkvEpiFp8(OpWrapper):
    """``(qkv8 [3, M, W] float8_e4m3fn, scales [8] fp32, gemm [M, 3W] bf16)`` of the NVFP4
    QKV projection, Q/K RMSNorm + RoPE applied, every plane scaled to E4M3.

    ``act_fp4`` ``[M, K/2]`` / ``act_sf`` swizzled: the block's shared NVFP4 activation
    (adaLN head output, gate_up's global scale); ``weight`` ``[3W, K/2]`` / ``weight_scale``:
    the q, k, v weights concatenated by rows with their swizzled scales back to back
    (`_prepare.prepare_qkv`); ``qkv_alpha`` fp32 ``[3]``: per-plane ``input_gs * weight_gs``;
    ``q_norm`` / ``k_norm`` bf16 ``[128]``; ``rope_tab`` / ``rope_idx``: `build_rope_compact`
    of the request's cos|sin table; ``sm_scale``: the attention softmax scale; ``keep_v``:
    False when nothing reads ``gemm``'s v plane afterwards (its L2 lines are dropped).

    scales: ``[s_q, s_k, s_v, s_q * s_k * sm_scale, 1/s_q, 1/s_k, 1/s_v, 0]``; s_q / s_k from
    the static bound ``bf16(max(11.313743 * max|w| * (1 + 2^-6) / 448, 1/(448*512)))``, s_v
    from ``max|v|``. ``gemm.view(3, M, W)[2]`` is the BF16 v plane; its q/k planes are
    undefined.
    """

    ARCHS = frozenset({Arch.SM_120})

    # QKV 9216 x 3072 (three planes of 3072 = 24 heads x 128) at the served M.
    CELLS: tuple[Cell, ...] = (
        Cell(why="2048^2 image / image_cond mode blocks: 16384 image rows", spec=dict(m=16384)),
        Cell(why="2048^2 t2i single blocks: 16384 image + 80 text rows", spec=dict(m=16464)),
        Cell(why="2048^2 i2i single blocks: 2 x 16384 + 80 rows", spec=dict(m=32848)),
        Cell(why="1024^2 mode blocks: 4096 rows", spec=dict(m=4096)),
        Cell(
            why="1024^2 t2i single blocks: 4096 + 80, a partial last 128-row tile",
            spec=dict(m=4176),
        ),
    )

    note = """
    Metadata consumed: none. Called for every NVFP4 block (image.1-4, image_cond.0-4,
    single.0-27) at every served M.

    Preconditions: weight rows q | k | v with the per-plane alphas in that order; head_dim
    128 and plane width a multiple of 128; the activation quantized with the global scale
    the alphas assume (gate_up's: `_prepare.prepare_qkv` refuses a QKV input scale that
    differs from gate_up's); rope_idx from `build_rope_compact` (every entry is verified on
    device, never trusted).

    Numerics: BF16(alpha_p * acc) is bitwise the stock SM120 CUTLASS NVFP4 GEMM (same k64
    MMA chain per element). Q/K: the RMS sum of squares is a fixed tree (cols l, l+32, l+64,
    l+96 chained with fma.rn, then the halves 16, 8, 4, 2, 1 added), inv = rsqrt.approx of
    ss * 2^-7 + 1e-6, RoPE as two fma.rn per pair, each rotated value BF16-rounded before
    the E4M3 scale. scales[0..7] and V's E4M3 bytes are exact functions of the norms and of
    the BF16 v plane. Against `reference` (fp32 GEMM, torch rsqrt) the BF16 GEMM value and
    so an E4M3 code can move by one step near a rounding boundary.

    PDL: the GEMM is launched with the PDL attribute; the V quantize is not.
    """

    def __call__(
        self,
        act_fp4: torch.Tensor,
        weight: torch.Tensor,
        act_sf: torch.Tensor,
        weight_scale: torch.Tensor,
        qkv_alpha: torch.Tensor,
        q_norm: torch.Tensor,
        k_norm: torch.Tensor,
        rope_tab: torch.Tensor,
        rope_idx: torch.Tensor,
        sm_scale: float,
        keep_v: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.ops.trtllm.flux3_qkv_epi_fp8(
            act_fp4,
            weight,
            act_sf,
            weight_scale,
            qkv_alpha,
            q_norm,
            k_norm,
            rope_tab,
            rope_idx,
            sm_scale,
            keep_v,
        )

    def reference(
        self,
        act_fp4: torch.Tensor,
        weight: torch.Tensor,
        act_sf: torch.Tensor,
        weight_scale: torch.Tensor,
        qkv_alpha: torch.Tensor,
        q_norm: torch.Tensor,
        k_norm: torch.Tensor,
        rope_tab: torch.Tensor,
        rope_idx: torch.Tensor,
        sm_scale: float,
        keep_v: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per plane: fp32 GEMM of the dequantized operands (TF32 off) times its alpha,
        rounded to BF16. q/k: fp32 RMSNorm (torch's sum and rsqrt), the norm weight,
        interleaved RoPE (pairs 2i, 2i + 1), BF16 round, times the static factor, E4M3.
        v: E4M3 of BF16(v) / s_v. The gemm output's q/k planes are zero here (undefined in
        the op). ``keep_v`` changes nothing and is ignored."""
        rows, half = act_fp4.shape
        width = weight.shape[0] // 3
        a = dequantize_swizzled(act_fp4, act_sf)
        plane_sf = weight_scale.reshape(-1).view(torch.uint8).reshape(3, -1)
        planes = []
        for p in range(3):
            w = dequantize_swizzled(weight[p * width : (p + 1) * width], plane_sf[p])
            with true_fp32_matmul():
                planes.append((qkv_alpha[p].float() * (a @ w.t())).to(torch.bfloat16))
            del w
        del a
        cos, sin = rope_from_compact(rope_tab, rope_idx, rows)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)  # broadcast over heads
        qkv8 = torch.empty((3, rows, width), dtype=torch.float8_e4m3fn, device=act_fp4.device)
        scales = torch.zeros(8, dtype=torch.float32, device=act_fp4.device)
        for p, norm in ((0, q_norm), (1, k_norm)):
            x = planes[p].float().view(rows, width // _HEAD_DIM, _HEAD_DIM)
            inv = torch.rsqrt(x.pow(2).sum(-1, keepdim=True) * 2.0**-7 + 1e-6)
            y = (inv * x) * norm.float()
            even, odd = y[..., 0::2], y[..., 1::2]
            rot = torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1)
            s, f = _static_scale(norm)
            qkv8[p] = _e4m3(rot.reshape(rows, width).to(torch.bfloat16).float() * f)
            scales[p], scales[4 + p] = s, f
        v = planes[2].float()
        min_scale = torch.tensor(_f32(_MIN_SCALE_BITS), device=v.device)
        s_v = torch.maximum(v.abs().amax() / 448.0, min_scale).to(torch.bfloat16).float()
        qkv8[2] = _e4m3(v * (1.0 / s_v))
        scales[2], scales[6] = s_v, 1.0 / s_v
        scales[3] = (scales[0] * scales[1]) * sm_scale
        gemm = torch.zeros((rows, 3 * width), dtype=torch.bfloat16, device=act_fp4.device)
        gemm.view(3, rows, width)[2] = planes[2]
        return qkv8, scales, gemm

    def is_valid(
        self,
        act_fp4: torch.Tensor,
        weight: torch.Tensor,
        act_sf: torch.Tensor,
        weight_scale: torch.Tensor,
        qkv_alpha: torch.Tensor,
        q_norm: torch.Tensor,
        k_norm: torch.Tensor,
        rope_tab: torch.Tensor,
        rope_idx: torch.Tensor,
        sm_scale: float,
        keep_v: bool = True,
    ) -> None:
        # The epilogue indexes rope_tab with rope_idx without a bound check.
        assert rope_tab.shape[0] == _ROPE_AXES * rope_idx.shape[0], (
            "rope_tab must be build_rope_compact's [4 S, 32] table for rope_idx's S rows"
        )
        assert int(rope_idx.min()) >= 0 and int(rope_idx.max()) < rope_tab.shape[0], (
            "rope_idx points outside rope_tab"
        )
        k = act_fp4.shape[1] * 2
        assert act_sf.numel() >= swizzled_sf_numel(act_fp4.shape[0], k // SF_VEC), (
            "act_sf is shorter than the padded swizzled rectangle of the activation"
        )


qkv_epi_fp8 = QkvEpiFp8()
