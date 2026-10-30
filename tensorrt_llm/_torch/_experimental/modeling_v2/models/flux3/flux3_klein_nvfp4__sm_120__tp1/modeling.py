# SPDX-FileCopyrightText: Copyright (c) 2026 Black Forest Labs, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: FLUX.3-klein NVFP4 / sm_120 / tp1 — a VisualGen pipeline.

The package is BFL's flux-3-image-klein, ``variants/nvfp4``: a flow-map joint image/text
DiT (hidden 3072, 24 heads of 128, MLP 9216) with five per-modality mode blocks for each
of ``image``, ``image_cond`` (i2i references) and text, then 28 joint single blocks over
``image | image_cond | text``; a Qwen3-VL text encoder and a KL image VAE. One prompt, no
CFG, no PAG.

**Precision layout.** The package's NVFP4 blocks (image.1-4, image_cond.0-4, single
0-23) and single_blocks.24-27 (BF16 in the package, quantized to NVFP4 W4A4 at load with a
unit activation scale) run the NVFP4 block route below. content_mode_blocks.image.0
(BF16 in the package) runs FP8 rowwise (per-channel weights quantized at load, per-token
activations). The text blocks, the embedders, the modulations and the final layer stay
BF16 (``F.linear`` under Inductor, cuBLAS).

**NVFP4 block** (one launch per step unless noted):
  1. LayerNorm + adaLN modulate + NVFP4 quantize (``adaln_nvfp4_quant``; the bf16 x_mod
     is not stored: QKV and gate_up read the NVFP4 activation alone);
  2. the QKV GEMM whose epilogue does per-head RMSNorm, RoPE (compact per-request
     table) and a static Q/K E4M3 scale, then V's E4M3 quantize (``qkv_rope_fp8``);
  3. per-tensor FP8 attention (G3 fmha) whose epilogue writes attn_out's NVFP4
     activation, with gate_up (SwiGLU + NVFP4 in the epilogue, v2w-permuted rows)
     forked onto an equal-priority side stream right after the FMHA is enqueued;
  4. down || attn_out with ``x + gate * (attn_out + down)`` in attn_out's epilogue
     (``lpt_pair_residual``).
The last single block computes only the final layer's rows: QKV and the FMHA K/V still
see every joint row, the FMHA runs on the image-row prefix, and gate_up and the tail run
at M = image rows.

**FP8R block** (image.0): fused LN + modulate + per-token E4M3; the FP8 rowwise QKV GEMM;
QK RMSNorm + RoPE + per-tensor E4M3 (dynamic Q/K amax); BF16-out G3 attention; gate_up
FP8R GEMM -> SwiGLU + E4M3; down || attn_out on two streams; the residual in Inductor.

**Step batching.** A flow-map text stream depends only on the prompt and each step's
target time, so the five text blocks run once per request over all N steps (M = N*T,
BF16 SDPA attention per step); the image stream's vec rows (time + target embeddings,
early/single modulations, final adaLN) run once per request at M = N. Each step's
transformer forward (captured as a CUDA graph per resolution and mode) runs emb_in, the
image (and i2i reference) mode blocks, the single blocks and the final layer.

**Numerics relative to the built-in pipeline**: FP8 E4M3 attention on every image/joint
block (static RMSNorm-bound Q/K scale on the NVFP4 blocks), NVFP4 W4A4 for
single_blocks.24-27, FP8 rowwise for image.0, a bf16 VAE decoder trunk with e4m3
ResnetBlock convs (``vae.py``), and step-batched (reordered) text and vec GEMMs.
Everything else is bitwise the built-in ``Flux3Pipeline``'s path. The accuracy gate certifies the whole.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tensorrt_llm._torch.modules.layer_norm import LayerNorm
from tensorrt_llm._torch.modules.swiglu import swiglu
from tensorrt_llm._torch.visual_gen.cuda_graph_runner import CUDAGraphRunner, CUDAGraphRunnerConfig

# Called through the module attribute so it can be patched: call it as
# ``pipeline_flux3._load_image_pixels``, never bind the function.
from tensorrt_llm._torch.visual_gen.models.flux3 import pipeline_flux3
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.packed import Layout, pack
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.scheduler import Flux3FlowMatchScheduler
from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import I2I_COND_L, I2I_COND_T
from tensorrt_llm._torch.visual_gen.output import CudaPhaseTimer, PipelineOutput
from tensorrt_llm._torch.visual_gen.pipeline import (
    BasePipeline,
    ExtraParamSchema,
    RefSlotSpec,
    RoleSpec,
)
from tensorrt_llm.logger import logger
from tensorrt_llm.media import png_decode

from . import weights as _weights
from .ops.attention.fp8_attention_bf16_out import fp8_attention_bf16_out
from .ops.attention.fp8_attention_nvfp4_out import fp8_attention_nvfp4_out
from .ops.attention.fp8_attention_nvfp4_out_q_rows import fp8_attention_nvfp4_out_q_rows
from .ops.attention.qknorm_rope_fp8 import qknorm_rope_fp8
from .ops.gemm.fp8r_gemm import fp8r_gemm
from .ops.gemm.gate_up_swiglu_nvfp4 import allocate_outputs as gate_up_outputs
from .ops.gemm.gate_up_swiglu_nvfp4 import gate_up_swiglu_nvfp4
from .ops.gemm.pair_residual import allocate_workspace as allocate_pair_workspace
from .ops.gemm.pair_residual import nvfp4_pair_residual
from .ops.gemm.qkv_epi_fp8 import build_rope_compact, qkv_epi_fp8
from .ops.norm.adaln_nvfp4 import adaln_nvfp4
from .ops.norm.fp8r_adaln_quant import fp8r_adaln_quant
from .ops.quantization.fp8r_quantize import fp8r_quantize
from .ops.quantization.swiglu_fp8r_quant import swiglu_fp8r_quant
from .text_encoder import Flux3KleinTextEncoder
from .vae import Flux3KleinVAE

_SM = (12, 0)

# Each constant below is a property of this target's path, not a knob.

#: txt_mode_blocks run QKV as one GEMM on cat([wq, wk, wv]) at exactly this many rows
#: (16 steps x T = 80). Bitwise the three GEMMs at every M = 16*T, T = 80..960; measured
#: faster only here (-30.5 us/block) and slower at M = 2560.
_TEXT_QKV_FUSED_ROWS = 1280
#: The last single block's kept row count must be a multiple of this (the NVFP4-out FMHA's
#: row blocks and the tail's GEMM M tiles); otherwise it runs on every row.
_LAST_BLOCK_ROWS_MULTIPLE = 256
#: Time embedding width.
_TIME_DIM = 256

# Step-invariant transformer.forward kwargs: built eagerly once per request (fresh tensors,
# never written in place), so a graph replay skips their copy after the first step.
_GRAPH_REUSABLE_INPUTS = frozenset(
    {
        f"_flux3_{stream}_{table}"
        for stream in ("image", "image_cond", "joint")
        for table in ("rope_tab", "rope_idx")
    }
    | {
        "_flux3_image_cos_sin",
        "_flux3_joint_select",
        "_flux3_joint_index",
        "_flux3_image_positions",
        "_flux3_image_cond_vec",
        "x_image_cond",
    }
)


def _rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    """FLUX.3's 2x2 rotation matrices of one axis, fp64 -> fp32."""
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos.to(torch.float64), omega)
    cos, sin = torch.cos(out), torch.sin(out)
    out = torch.stack([cos, -sin, sin, cos], dim=-1)
    out = out.reshape(*out.shape[:-1], 2, 2)
    return out.float()


def _pos_embed(ids: Tensor, axes_dim: tuple[int, ...], theta: int) -> Tensor:
    """[1, L, 4] ids -> pe [1, 1, L, head_dim / 2, 2, 2]."""
    emb = torch.cat([_rope(ids[..., i], axes_dim[i], theta) for i in range(len(axes_dim))], dim=-3)
    return emb.unsqueeze(1)


def _pe_to_cos_sin(pe: Tensor) -> tuple[Tensor, Tensor]:
    """pe [1, 1, L, head_dim / 2, 2, 2] -> (cos, sin) each [1, L, 1, head_dim] fp32."""
    cos = pe[..., 0, 0].repeat_interleave(2, dim=-1)
    sin = pe[..., 1, 0].repeat_interleave(2, dim=-1)
    return cos.transpose(1, 2).contiguous(), sin.transpose(1, 2).contiguous()


def _cos_sin_table(cos: Tensor, sin: Tensor) -> Tensor:
    """The fused QK-norm + RoPE kernels' FP32 [rows, 128] table: the cos half, then the sin
    half (every other column of the repeat-interleaved cos / sin)."""
    table = torch.cat((cos.reshape(-1, 128)[:, ::2], sin.reshape(-1, 128)[:, ::2]), dim=-1)
    return table.float().contiguous()


def _timestep_embedding(t: Tensor, dim: int, out_dtype: torch.dtype) -> Tensor:
    t = 1000.0 * t
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half
    )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding.to(out_dtype)


def _apply_rotary_emb(x: Tensor, freqs_cos: Tensor, freqs_sin: Tensor) -> Tensor:
    freqs_cos = freqs_cos.to(x.dtype)
    freqs_sin = freqs_sin.to(x.dtype)
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


def _bf16(*shape: int) -> Tensor:
    return torch.empty(*shape, dtype=torch.bfloat16)


@lru_cache(maxsize=8)
def _gate_up_stream(device_index: int, priority: int) -> torch.cuda.Stream:
    """gate_up's side stream, at the priority of the stream that forks it."""
    return torch.cuda.Stream(device_index, priority=priority)


@lru_cache(maxsize=8)
def _tail_pair_stream(device_index: int) -> torch.cuda.Stream:
    """The FP8R block's attn_out stream (default priority)."""
    return torch.cuda.Stream(device_index)


@torch.library.custom_op(
    "trtllm::flux3_klein_attention_nvfp4_out_gate_up", mutates_args=(), device_types="cuda"
)
def _attention_nvfp4_out_gate_up(
    qkv8: Tensor,
    qkv8_scales: Tensor,
    heads: int,
    q_rows: int,
    out_act_global: Tensor,
    up_act: Tensor,
    up_act_sf: Tensor,
    up_weight: Tensor,
    up_weight_sf: Tensor,
    up_alpha: Tensor,
    down_act_global: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """FP8 attention writing attn_out's NVFP4 activation, with the block's gate_up beside it.

    Order (fmha_first): gate_up's outputs are allocated on the current stream, the side
    stream waits on the current one, the FMHA is enqueued, then gate_up on the side stream,
    then the current stream waits on the side stream. gate_up's CTAs fill the FMHA's last
    wave. Returns (fp4 [1, S, W/2], sf, packed gate_up [M, I/2], packed sf); q_rows > 0
    runs the FMHA on the Q-row prefix [0, q_rows) (only those fp4 rows and SF row blocks are
    defined) and gate_up on the first q_rows rows of its activation.
    """
    device = qkv8.device
    rows = qkv8.shape[1]
    up_rows = q_rows if q_rows > 0 else rows
    if q_rows > 0:
        # The leading q_rows / 128 row blocks of the flat 128x4 scale layout.
        up_cols = (up_act.shape[-1] * 2 // 16 + 3) // 4 * 4
        up_act, up_act_sf = up_act[:q_rows], up_act_sf.reshape(-1)[: q_rows * up_cols]
    up_act = up_act.contiguous()
    packed, packed_sf = gate_up_outputs(up_rows, up_weight.shape[0], device)
    current = torch.cuda.current_stream(device)
    side = _gate_up_stream(device.index, current.priority)
    side.wait_stream(current)
    if q_rows > 0:
        fp4, sf = fp8_attention_nvfp4_out_q_rows(
            qkv8, qkv8_scales, 1, heads, out_act_global, min(q_rows, rows)
        )
    else:
        fp4, sf = fp8_attention_nvfp4_out(qkv8, qkv8_scales, 1, heads, out_act_global)
    with torch.cuda.stream(side):
        packed, packed_sf = gate_up_swiglu_nvfp4(
            up_act, up_weight, up_act_sf, up_weight_sf, up_alpha, down_act_global, packed, packed_sf
        )
    current.wait_stream(side)
    return fp4, sf, packed, packed_sf


@_attention_nvfp4_out_gate_up.register_fake
def _(
    qkv8,
    qkv8_scales,
    heads,
    q_rows,
    out_act_global,
    up_act,
    up_act_sf,
    up_weight,
    up_weight_sf,
    up_alpha,
    down_act_global,
):
    _, rows, width = qkv8.shape
    up_rows = q_rows if q_rows > 0 else rows
    padded = (rows + 127) // 128 * 128
    up_padded = (up_rows + 127) // 128 * 128
    hidden = up_weight.shape[0] // 2
    return (
        qkv8.new_empty((1, rows, width // 2), dtype=torch.uint8),
        qkv8.new_empty((padded * ((width // 16 + 3) // 4 * 4),), dtype=torch.uint8),
        qkv8.new_empty((up_rows, hidden // 2), dtype=torch.uint8),
        qkv8.new_empty((up_padded * ((hidden // 16 + 3) // 4 * 4),), dtype=torch.uint8),
    )


@torch.library.custom_op("trtllm::flux3_klein_fp8r_out_pair", mutates_args=(), device_types="cuda")
def _fp8r_out_pair(
    down_data: Tensor,
    down_scales: Tensor,
    down_weight: Tensor,
    down_weight_scale: Tensor,
    attn: Tensor,
    attn_weight: Tensor,
    attn_weight_scale: Tensor,
) -> tuple[Tensor, Tensor]:
    """(attn_out(attn), down(hidden)) of the FP8R block: down on the current stream, attn_out
    (per-token quantize + GEMM) on the side stream."""
    device = attn.device
    current = torch.cuda.current_stream(device)
    side = _tail_pair_stream(device.index)
    side.wait_stream(current)
    down = fp8r_gemm(down_data, down_weight, down_scales, down_weight_scale)
    with torch.cuda.stream(side):
        data, scales = fp8r_quantize(attn)
        attn_y = fp8r_gemm(
            data.contiguous(),
            attn_weight,
            scales.float().reshape(-1).contiguous(),
            attn_weight_scale,
        )
    current.wait_stream(side)
    # attn_y was allocated on the side stream; keep its block until the current stream is done.
    attn_y.record_stream(current)
    return attn_y, down


@_fp8r_out_pair.register_fake
def _(down_data, down_scales, down_weight, down_weight_scale, attn, attn_weight, attn_weight_scale):
    return (
        attn.new_empty((attn.shape[0], attn_weight.shape[0]), dtype=torch.bfloat16),
        attn.new_empty((down_data.shape[0], down_weight.shape[0]), dtype=torch.bfloat16),
    )


class _Linear(nn.Module):
    """A bias-free BF16 projection (``F.linear``, cuBLAS)."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.register_buffer("weight", _bf16(out_features, in_features))

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight)


class _MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.in_layer = _Linear(in_dim, hidden)
        self.out_layer = _Linear(hidden, hidden)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(F.silu(self.in_layer(x)))


class _Modulation(nn.Module):
    """vec -> (shift, scale, gate) rows."""

    def __init__(self, hidden: int):
        super().__init__()
        self.lin = _Linear(hidden, 3 * hidden)

    def forward(self, vec: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        out = self.lin(F.silu(vec))
        if out.ndim == 2:
            out = out[:, None, :]
        shift, scale, gate = out.chunk(3, dim=-1)
        return shift, scale, gate


class _FinalLayer(nn.Module):
    """final_layer.image: LayerNorm + adaLN modulate + linear (compiled)."""

    def __init__(self, hidden: int, out_channels: int, eps: float):
        super().__init__()
        self.norm_final = LayerNorm(
            hidden_size=hidden, eps=eps, has_weights=False, has_bias=False, dtype=torch.bfloat16
        )
        self.linear = _Linear(hidden, out_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), _Linear(hidden, 2 * hidden))

    def forward(self, x: Tensor, mod: Tensor) -> Tensor:
        shift, scale = mod.chunk(2, dim=-1)
        x = (1 + scale) * self.norm_final(x) + shift
        return self.linear(x)


class _TextBlock(nn.Module):
    """txt_mode_blocks.<i>, BF16; run once per request over every step (``steps`` equal
    segments of ``length`` rows), compiled. Attention is BF16 SDPA per segment."""

    def __init__(self, hidden: int, heads: int, mlp_hidden: int, eps: float):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        self.eps = eps
        for name in ("wq", "wk", "wv"):
            self.register_buffer(name, _bf16(hidden, hidden))
        # cat([wq, wk, wv]) for the _TEXT_QKV_FUSED_ROWS GEMM (weights.py).
        self.register_buffer("wqkv", _bf16(3 * hidden, hidden))
        self.register_buffer("w_out", _bf16(hidden, hidden))
        self.register_buffer("w_gate_up", _bf16(2 * mlp_hidden, hidden))  # [gate | up]
        self.register_buffer("w_down", _bf16(hidden, mlp_hidden))
        self.register_buffer("norm_q", _bf16(self.head_dim))
        self.register_buffer("norm_k", _bf16(self.head_dim))
        self.pre_norm = LayerNorm(
            hidden_size=hidden, eps=eps, has_weights=False, has_bias=False, dtype=torch.bfloat16
        )

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        shift: Tensor,
        scale: Tensor,
        gate: Tensor,
        steps: int,
        length: int,
    ) -> Tensor:
        x_mod = (1 + scale) * self.pre_norm(x) + shift
        b = x_mod.shape[0]
        width = self.heads * self.head_dim
        if x_mod.numel() == _TEXT_QKV_FUSED_ROWS * x_mod.shape[-1]:
            q, k, v = F.linear(x_mod, self.wqkv, None).split([width, width, width], dim=-1)
        else:
            q, k, v = F.linear(x_mod, self.wq), F.linear(x_mod, self.wk), F.linear(x_mod, self.wv)
        q = q.view(b, -1, self.heads, self.head_dim)
        k = k.view(b, -1, self.heads, self.head_dim)
        v = v.view(b, -1, self.heads, self.head_dim)
        q, k = q.float(), k.float()
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps) * self.norm_q.float()
        k = k * torch.rsqrt(k.square().mean(-1, keepdim=True) + self.eps) * self.norm_k.float()
        q = _apply_rotary_emb(q, cos, sin).to(v.dtype)
        k = _apply_rotary_emb(k, cos, sin).to(v.dtype)
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
        # Equal segments are independent batch rows: [steps, length, H, D].
        qs, ks, vs = (
            t.view(steps, length, t.shape[-1]).view(steps, -1, self.heads, self.head_dim)
            for t in (q, k, v)
        )
        qs, ks, vs = qs.contiguous(), ks.contiguous(), vs.contiguous()
        out = F.scaled_dot_product_attention(
            qs.transpose(1, 2),
            ks.transpose(1, 2),
            vs.transpose(1, 2),
            is_causal=False,
            scale=1.0 / math.sqrt(self.head_dim),
        )
        attn = out.transpose(1, 2).flatten(2).reshape_as(q)
        mlp_shape = x_mod.shape
        mlp_input = x_mod.reshape(-1, mlp_shape[-1])
        mlp_out = F.linear(swiglu(F.linear(mlp_input, self.w_gate_up)), self.w_down)
        mlp_out = mlp_out.reshape(*mlp_shape[:-1], -1)
        output = F.linear(attn, self.w_out) + mlp_out
        return x + gate * output


def _nvfp4_linear_buffers(
    module: nn.Module, name: str, rows: int, k: int, alphas: int = 1, act_global: bool = True
):
    """Declare one prepared NVFP4 projection: E2M1 weight [rows, k/2], 128x4-swizzled E4M3
    scales (flat), alpha(s) = input global scale x weight global scale and, where a kernel
    quantizes this projection's input, that activation's global scale (1 / input global
    scale). weights.py fills them."""
    module.register_buffer(f"{name}_weight", torch.empty(rows, k // 2, dtype=torch.uint8))
    sf = (rows + 127) // 128 * 128 * ((k // 16 + 3) // 4 * 4)
    module.register_buffer(f"{name}_weight_sf", torch.empty(sf, dtype=torch.uint8))
    module.register_buffer(f"{name}_alpha", torch.empty(alphas, dtype=torch.float32))
    if act_global:
        module.register_buffer(f"{name}_act_global", torch.empty(1, dtype=torch.float32))


class _NVFP4Block(nn.Module):
    """An NVFP4 block's weights and body (see the module docstring), compiled.

    ``select`` is None for a mode block (shift/scale/gate are [1, 1, D] rows) and the joint
    row picker for a single block ([1, S, 1] bool for two chunks, uint8 chunk number for
    three; shift/scale/gate are [1, K, D] tables).
    """

    def __init__(self, hidden: int, heads: int, mlp_hidden: int, eps: float):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        self.eps = eps
        self.sm_scale = 1.0 / math.sqrt(self.head_dim)
        # qkv: [q ; k ; v] rows, one alpha per plane.
        _nvfp4_linear_buffers(self, "qkv", 3 * hidden, hidden, alphas=3, act_global=False)
        # gate_up: [up, gate] row pairs, then the v2w row permutation (its SwiGLU epilogue).
        _nvfp4_linear_buffers(self, "gate_up", 2 * mlp_hidden, hidden)
        _nvfp4_linear_buffers(self, "down", hidden, mlp_hidden)
        _nvfp4_linear_buffers(self, "out", hidden, hidden)
        self.register_buffer("norm_q", _bf16(self.head_dim))
        self.register_buffer("norm_k", _bf16(self.head_dim))
        # nvfp4_pair_residual's zero-filled workspace, shared by every block; the pipeline
        # sets it after load (allocated outside graph capture).
        self.pair_workspace: Tensor | None = None

    def _qkv_attention(self, x, shift, scale, select, rope_tab, rope_idx, q_rows):
        # LayerNorm (unit affine) + modulate + NVFP4; the bf16 x_mod is not stored: QKV and
        # gate_up read only this activation (one global scale for both, checked at load).
        act, act_sf = adaln_nvfp4(x, select, scale, shift, self.gate_up_act_global, self.eps)
        # keep_v=False: nothing reads the BF16 v plane (no PAG); its L2 lines are dropped.
        qkv8, qkv8_scales, _ = qkv_epi_fp8(
            act,
            self.qkv_weight,
            act_sf,
            self.qkv_weight_sf,
            self.qkv_alpha,
            self.norm_q,
            self.norm_k,
            rope_tab,
            rope_idx,
            self.sm_scale,
            False,
        )
        return _attention_nvfp4_out_gate_up(
            qkv8,
            qkv8_scales,
            self.heads,
            q_rows,
            self.out_act_global,
            act,
            act_sf,
            self.gate_up_weight,
            self.gate_up_weight_sf,
            self.gate_up_alpha,
            self.down_act_global,
        )

    def _tail(self, fp4, sf, packed, packed_sf, x, gate, select):
        """x + gate[select] * (attn_out(fp4) + down(packed)), one launch."""
        out = nvfp4_pair_residual(
            packed,
            self.down_weight,
            packed_sf,
            self.down_weight_sf,
            self.down_alpha,
            fp4.reshape(-1, fp4.shape[-1]),
            self.out_weight,
            sf,
            self.out_weight_sf,
            self.out_alpha,
            x,
            gate,
            select,
            self.pair_workspace,
        )
        return out.reshape(x.shape)

    def _block(self, x, shift, scale, gate, select, rope_tab, rope_idx) -> Tensor:
        fp4, sf, packed, packed_sf = self._qkv_attention(
            x, shift, scale, select, rope_tab, rope_idx, 0
        )
        return self._tail(fp4, sf, packed, packed_sf, x, gate, select)


# One subclass per call form, each with its own forward code object (so its own compile
# cache).


class _NVFP4ModeBlock(_NVFP4Block):
    """content_mode_blocks.image.1-4 and image_cond.0-4: [1, 1, D] modulation rows."""

    def forward(
        self,
        x: Tensor,
        shift: Tensor,
        scale: Tensor,
        gate: Tensor,
        rope_tab: Tensor,
        rope_idx: Tensor,
    ) -> Tensor:
        return self._block(x, shift, scale, gate, None, rope_tab, rope_idx)


class _NVFP4SingleBlock(_NVFP4Block):
    """single_blocks.0-26: [1, K, D] modulation tables and the joint row picker."""

    def forward(
        self,
        x: Tensor,
        shift: Tensor,
        scale: Tensor,
        gate: Tensor,
        select: Tensor,
        rope_tab: Tensor,
        rope_idx: Tensor,
    ) -> Tensor:
        return self._block(x, shift, scale, gate, select, rope_tab, rope_idx)


class _LastNVFP4Block(_NVFP4Block):
    """single_blocks[-1]: only rows [0, keep_rows) of the joint stream are read after it (the
    image rows, first). The head, QKV and the FMHA's K/V cover every row; the FMHA's Q rows,
    gate_up and the tail cover the first keep_rows. Bitwise those rows of the full block.
    Its own forward code object, so its compile cache stays apart from the other blocks'."""

    def forward(
        self,
        x: Tensor,
        shift: Tensor,
        scale: Tensor,
        gate: Tensor,
        select: Tensor,
        rope_tab: Tensor,
        rope_idx: Tensor,
        keep_rows: int,
    ) -> Tensor:
        if keep_rows <= 0:
            return self._block(x, shift, scale, gate, select, rope_tab, rope_idx)
        n = keep_rows
        fp4, sf, packed, packed_sf = self._qkv_attention(
            x, shift, scale, select, rope_tab, rope_idx, n
        )
        # The attention output's first n rows: fp4 rows and the leading n / 128 SF row blocks.
        out_cols = (fp4.shape[-1] * 2 // 16 + 3) // 4 * 4
        return self._tail(
            fp4[:, :n], sf[: n * out_cols], packed, packed_sf, x[:, :n], gate, select[:, :n]
        )


class _FP8RBlock(nn.Module):
    """content_mode_blocks.image.0: FP8 rowwise (see the module docstring), compiled."""

    def __init__(self, hidden: int, heads: int, mlp_hidden: int, eps: float):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        self.eps = eps
        self.sm_scale = 1.0 / math.sqrt(self.head_dim)
        e4m3 = torch.float8_e4m3fn
        for name, rows, k in (
            ("qkv", 3 * hidden, hidden),  # [q ; k ; v]
            ("gate_up", 2 * mlp_hidden, hidden),  # [gate | up]
            ("down", hidden, mlp_hidden),
            ("out", hidden, hidden),
        ):
            self.register_buffer(f"{name}_weight", torch.empty(rows, k, dtype=e4m3))
            self.register_buffer(f"{name}_weight_scale", torch.empty(rows, dtype=torch.float32))
        self.register_buffer("norm_q", _bf16(self.head_dim))
        self.register_buffer("norm_k", _bf16(self.head_dim))

    def forward(
        self,
        x: Tensor,
        shift: Tensor,
        scale: Tensor,
        gate: Tensor,
        cos_sin: Tensor,
        positions: Tensor,
    ) -> Tensor:
        # LayerNorm (unit affine) + modulate + per-token E4M3; the bf16 x_mod is not stored:
        # QKV and gate_up read only its E4M3 rows.
        data, scales = fp8r_adaln_quant(x, shift, scale, self.eps)
        data = data.contiguous()
        scales = scales.float().reshape(-1).contiguous()
        batch = x.shape[0]
        gemm = fp8r_gemm(data, self.qkv_weight, scales, self.qkv_weight_scale)
        width = gemm.shape[1] // 3
        # Dynamic Q/K amax; the GEMM output's columns are undefined afterwards (L2 discard).
        qkv8, qkv8_scales = qknorm_rope_fp8(
            gemm[:, :width],
            gemm[:, width : 2 * width],
            gemm[:, 2 * width :],
            self.norm_q,
            self.norm_k,
            cos_sin,
            positions,
            self.sm_scale,
        )
        attn = fp8_attention_bf16_out(qkv8, qkv8_scales, batch, self.heads).flatten(2)
        hidden, hidden_scales = swiglu_fp8r_quant(
            fp8r_gemm(data, self.gate_up_weight, scales, self.gate_up_weight_scale)
        )
        attn_y, mlp_y = _fp8r_out_pair(
            hidden,
            hidden_scales,
            self.down_weight,
            self.down_weight_scale,
            attn.reshape(-1, attn.shape[-1]).contiguous(),
            self.out_weight,
            self.out_weight_scale,
        )
        output = attn_y.reshape(*attn.shape[:-1], -1) + mlp_y.reshape(*x.shape[:-1], -1)
        return x + gate * output


class Flux3KleinTransformer(nn.Module):
    """The DiT. ``forward`` is one denoise step (graph-captured); ``prepare_request`` and
    ``prepare_text_steps`` / ``prepare_image_steps`` run once per request, eagerly."""

    def __init__(self, pretrained_config):
        super().__init__()
        p = pretrained_config
        hidden, heads = p.hidden_size, p.num_heads
        mlp_hidden = int(hidden * p.mlp_ratio)
        eps = getattr(p, "eps", 1e-6)
        self.hidden = hidden
        self.heads = heads
        self.axes_dim = tuple(p.axes_dim)
        self.theta = p.theta
        self.context_in_dim = p.context_in_dim
        self.in_channels = dict(p.in_channels)
        image_ch = self.in_channels["image"]

        self.emb_in = nn.ModuleDict({m: _Linear(c, hidden) for m, c in self.in_channels.items()})
        self.txt_in = _Linear(p.context_in_dim, hidden)
        self.time_in = _MLPEmbedder(_TIME_DIM, hidden)
        self.ref_time_in = _MLPEmbedder(_TIME_DIM, hidden)
        modes = (*sorted(self.in_channels), "txt")
        self.early_stream_modulations = nn.ModuleDict({m: _Modulation(hidden) for m in modes})
        self.single_stream_modulations = nn.ModuleDict({m: _Modulation(hidden) for m in modes})

        block = (hidden, heads, mlp_hidden, eps)
        # image.0 is the FP8R block; image.1-4 and every image_cond block NVFP4.
        self.image_blocks = nn.ModuleList(
            [_FP8RBlock(*block)] + [_NVFP4ModeBlock(*block) for _ in range(p.depth - 1)]
        )
        self.cond_blocks = nn.ModuleList([_NVFP4ModeBlock(*block) for _ in range(p.depth)])
        self.text_blocks = nn.ModuleList([_TextBlock(*block) for _ in range(p.depth)])
        self.single_blocks = nn.ModuleList(
            [_NVFP4SingleBlock(*block) for _ in range(p.depth_single_blocks - 1)]
            + [_LastNVFP4Block(*block)]
        )
        self.final_layer = _FinalLayer(hidden, image_ch, eps)
        self._logged_keep_rows: set = set()

    def _stream_tables(self, prepared: dict, prefix: str, cos: Tensor, sin: Tensor) -> None:
        """The NVFP4 blocks' compact RoPE table (rope_tab, rope_idx); image.0 (FP8R) also
        reads the full fp32 cos|sin table."""
        table = _cos_sin_table(cos, sin)
        if prefix == "_flux3_image_":
            prepared[prefix + "cos_sin"] = table
        prepared[prefix + "rope_tab"], prepared[prefix + "rope_idx"] = build_rope_compact(table)

    def prepare_request(self, inputs: dict, layout: Layout) -> dict:
        """The per-request RoPE tables, joint row picker and positions (forward kwargs), plus
        the text cos / sin the step-batched text stream reads."""
        prepared, positions = {}, {}
        lengths = dict(layout)
        for name, stream_lengths in layout:
            if not any(stream_lengths):
                continue
            pe = _pos_embed(inputs["x_" + name + "_ids"].unsqueeze(0), self.axes_dim, self.theta)
            positions[name] = pe
            cos, sin = _pe_to_cos_sin(pe)
            if name == "text":
                prepared["text_cos"], prepared["text_sin"] = cos, sin
            else:
                self._stream_tables(prepared, f"_flux3_{name}_", cos, sin)
        # Joint order: image | image_cond | text (the layout's sorted stream names).
        chunks = [name for name, stream_lengths in layout if any(stream_lengths)]
        pe = torch.cat([positions[name] for name in chunks], dim=2)
        self._stream_tables(prepared, "_flux3_joint_", *_pe_to_cos_sin(pe))
        device = pe.device
        if len(chunks) == 2:
            # Bool [1, S, 1]: True on the second chunk's rows.
            first, total = lengths[chunks[0]][0], sum(lengths[c][0] for c in chunks)
            prepared["_flux3_joint_select"] = (torch.arange(total, device=device) >= first).view(
                1, total, 1
            )
        else:
            # u8 [1, S, 1]: the chunk number of every row.
            starts, total = [], 0
            for name in chunks:
                starts.append(total)
                total += lengths[name][0]
            ar = torch.arange(total, device=device)
            index = torch.zeros(total, dtype=torch.uint8, device=device)
            for boundary in starts[1:]:
                index += (ar >= boundary).to(torch.uint8)
            prepared["_flux3_joint_index"] = index.view(1, total, 1)
        prepared["_flux3_image_positions"] = torch.arange(
            lengths["image"][0], device=device, dtype=torch.int64
        )
        return prepared

    def prepare_text_steps(
        self,
        x_text: Tensor,
        text_timesteps: Tensor,
        ref: Tensor,
        text_cos: Tensor,
        text_sin: Tensor,
    ) -> dict:
        """The text stream for every step at once: txt_mode_blocks at M = N*T.

        A flow-map text stream depends only on the prompt and step i's target time (``ref``,
        [N, D]), so step i's text tokens and single_* rows are f(ctx, targets[i]). Returns
        [1, N*T, D] tokens and [1, N, D] single_* rows.
        """
        steps, length = ref.shape[0], x_text.shape[0]
        vec = self.time_in(_timestep_embedding(text_timesteps[:1], _TIME_DIM, x_text.dtype))
        vec = (vec + ref).unsqueeze(0)  # [1, N, D]
        early = self.early_stream_modulations["txt"](vec)
        single = self.single_stream_modulations["txt"](vec)

        def per_token(rows: Tensor) -> Tensor:
            return (
                rows[0]
                .unsqueeze(1)
                .expand(steps, length, rows.shape[-1])
                .reshape(1, steps * length, rows.shape[-1])
            )

        # txt_in does not depend on the step: once at M = T, repeated.
        tokens = self.txt_in(x_text).unsqueeze(0).repeat(1, steps, 1)
        shift, scale, gate = (per_token(rows) for rows in early)
        cos, sin = text_cos.repeat(1, steps, 1, 1), text_sin.repeat(1, steps, 1, 1)
        for block in self.text_blocks:
            tokens = block(tokens, cos, sin, shift, scale, gate, steps, length)
        return {"tokens": tokens, "single": single}

    def prepare_image_steps(self, image_timesteps: Tensor, ref: Tensor, dtype) -> dict:
        """The image stream's vec rows for every step at once (M = N): early / single
        modulations and the final layer's adaLN row, laid out [N, 1, 1, *]."""
        steps = image_timesteps.shape[0]
        vec = self.time_in(_timestep_embedding(image_timesteps, _TIME_DIM, dtype))
        vec = (vec + ref).unsqueeze(0)  # [1, N, D]
        rows = {}
        for stage, modules in (
            ("early", self.early_stream_modulations),
            ("single", self.single_stream_modulations),
        ):
            module = modules["image"]
            out = module.lin(F.silu(vec))
            rows[stage] = out.reshape(steps, 1, 1, out.shape[-1])
        mod = self.final_layer.adaLN_modulation(vec)
        rows["final"] = mod.reshape(steps, 1, 1, mod.shape[-1])
        return rows

    def ref_rows(self, targets: Tensor) -> Tensor:
        """ref_time_in(targets) [N, D]: each step's target-time embedding."""
        return self.ref_time_in(_timestep_embedding(targets, _TIME_DIM, targets.dtype))

    def forward(
        self,
        *,
        layout: Layout,
        x_image: Tensor,
        _flux3_image_step_early: Tensor,
        _flux3_image_step_single: Tensor,
        _flux3_image_step_final: Tensor,
        _flux3_text_step_tokens: Tensor,
        _flux3_text_step_single_shift: Tensor,
        _flux3_text_step_single_scale: Tensor,
        _flux3_text_step_single_gate: Tensor,
        _flux3_image_cos_sin: Tensor,
        _flux3_image_rope_tab: Tensor,
        _flux3_image_rope_idx: Tensor,
        _flux3_image_positions: Tensor,
        _flux3_joint_rope_tab: Tensor,
        _flux3_joint_rope_idx: Tensor,
        _flux3_joint_select: Tensor | None = None,
        _flux3_joint_index: Tensor | None = None,
        x_image_cond: Tensor | None = None,
        _flux3_image_cond_vec: Tensor | None = None,
        _flux3_image_cond_rope_tab: Tensor | None = None,
        _flux3_image_cond_rope_idx: Tensor | None = None,
        timesteps_ref: Tensor | None = None,
    ) -> Tensor:
        """Step i's velocity [n, C] for the image rows."""
        lengths = dict(layout)
        n = lengths["image"][0]

        # Image stream: only emb_in runs per step; its rows come from prepare_image_steps.
        image = self.emb_in["image"](x_image).unsqueeze(0)
        shift, scale, gate = _flux3_image_step_early.chunk(3, dim=-1)
        # image.0 is the FP8R block (_FP8RBlock), image.1-4 NVFP4.
        image = self.image_blocks[0](
            image, shift, scale, gate, _flux3_image_cos_sin, _flux3_image_positions
        )
        for block in self.image_blocks[1:]:
            image = block(image, shift, scale, gate, _flux3_image_rope_tab, _flux3_image_rope_idx)
        streams = [(image, _flux3_image_step_single.chunk(3, dim=-1))]

        if x_image_cond is not None:
            # i2i reference stream, per step: its vec carries the step's target time.
            target = timesteps_ref.to(torch.bfloat16)
            ref = self.ref_rows(target)
            vec = _flux3_image_cond_vec + torch.stack([ref[0]]).unsqueeze(0)  # [1, 1, D]
            early = self.early_stream_modulations["image_cond"](vec)
            single = self.single_stream_modulations["image_cond"](vec)
            cond = self.emb_in["image_cond"](x_image_cond).unsqueeze(0)
            for block in self.cond_blocks:
                cond = block(cond, *early, _flux3_image_cond_rope_tab, _flux3_image_cond_rope_idx)
            streams.append((cond, single))

        streams.append(
            (
                _flux3_text_step_tokens,
                (
                    _flux3_text_step_single_shift,
                    _flux3_text_step_single_scale,
                    _flux3_text_step_single_gate,
                ),
            )
        )

        # Joint stream: [1, K, D] modulation tables (one broadcast row per chunk) and the
        # per-row chunk picker.
        combined = torch.cat([tokens for tokens, _ in streams], dim=1)
        shift, scale, gate = (
            torch.cat([rows[field] for _, rows in streams], dim=1) for field in range(3)
        )
        select = _flux3_joint_select if _flux3_joint_select is not None else _flux3_joint_index
        rows = combined.shape[1]
        keep_rows = n if (0 < n < rows and n % _LAST_BLOCK_ROWS_MULTIPLE == 0) else 0
        last = len(self.single_blocks) - 1
        for index, block in enumerate(self.single_blocks):
            args = (
                combined,
                shift,
                scale,
                gate,
                select,
                _flux3_joint_rope_tab,
                _flux3_joint_rope_idx,
            )
            combined = block(*args, keep_rows) if index == last else block(*args)
        if keep_rows:
            if combined.shape[1] != keep_rows:
                raise RuntimeError(
                    f"last single block returned {combined.shape[1]} rows, expected {keep_rows}"
                )
            key = (rows, keep_rows)
            if key not in self._logged_keep_rows:
                self._logged_keep_rows.add(key)
                logger.info(
                    f"FLUX.3 single_blocks.{last}: kept the image rows [0, {keep_rows}) of "
                    f"S={rows}; the tail runs at M={keep_rows}"
                )
        part = combined[:, :n].contiguous()
        # No single-part cat copy. Restride the B=1 view so its batch stride is n*D, not the
        # joint length's: the compiled final layer's guards then ignore the text length.
        tokens = part[0].unsqueeze(0)
        return self.final_layer(tokens, _flux3_image_step_final)[0]


def _sample_image_latents(
    height: int, width: int, seed: int, device: torch.device, image_rope_scale: float
) -> tuple[Tensor, Tensor]:
    """Seeded CPU Philox noise (bitwise contract) [n, 128] bf16 and its position ids [n, 4] fp32,
    both on ``device``; the noise goes through pinned memory and the ids are built there, so
    the host does not wait behind queued GPU work."""
    if min(height, width) < 16 or height % 16 or width % 16:
        raise ValueError("FLUX.3 image dimensions must be positive multiples of 16")
    h, w = height // 16, width // 16
    rng = torch.Generator().manual_seed(seed)
    x = torch.randn((h * w, 128), generator=rng)
    x = x.pin_memory().to(device, non_blocking=True)
    x_ids = torch.cartesian_prod(
        torch.arange(1, device=device),
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        torch.arange(1, device=device),
    ).unsqueeze(0)
    x_ids = x_ids.to(dtype=torch.float32)
    x_ids[..., 1:3] = x_ids[..., 1:3] * image_rope_scale
    return x.unsqueeze(0).to(dtype=torch.bfloat16)[0], x_ids[0]


def _pack_image_condition(latents: list[Tensor], image_rope_scale: float) -> tuple[Tensor, Tensor]:
    """i2i reference latents [128, h, w] -> tokens [c, 128] bf16 and ids [c, 4] fp32."""
    parts, id_parts = [], []
    for i_ref, latent in enumerate(latents):
        _, h, w = latent.shape
        dev = latent.device
        ids = torch.cartesian_prod(
            torch.full((1,), I2I_COND_T, device=dev, dtype=torch.long),
            torch.arange(h, device=dev),
            torch.arange(w, device=dev),
            torch.full((1,), I2I_COND_L - i_ref, device=dev, dtype=torch.long),
        )
        parts.append(latent.permute(1, 2, 0).reshape(h * w, -1))
        id_parts.append(ids)
    x_cond = torch.cat(parts, dim=0).unsqueeze(0)
    x_cond_ids = torch.cat(id_parts, dim=0).unsqueeze(0).to(dtype=torch.float32)
    x_cond_ids[..., 1:3] = x_cond_ids[..., 1:3] * image_rope_scale
    return x_cond[0].to(dtype=torch.bfloat16), x_cond_ids[0]


def _sigmas_to_device(sigmas: Tensor, device, dtype: torch.dtype) -> Tensor:
    """CPU sigmas -> device ``dtype`` through pinned memory (the cast runs on the CPU, as
    the combined ``.to()`` does)."""
    return sigmas.to(dtype=dtype).pin_memory().to(device, non_blocking=True)


class ModelingV2Flux3KleinNvfp4Sm120Tp1(BasePipeline):
    """FLUX.3-klein NVFP4 on one sm120 GPU: t2i and i2i, flow map, one prompt per request.

    Not registered in the pipeline registry: ``models/flux3/routing.py`` returns this class
    and ``AutoPipeline.from_config`` builds it. The denoise loop runs through
    ``BasePipeline._profile_denoise_steps`` and references decode through
    ``pipeline_flux3._load_image_pixels``, both called through the module attribute so they
    can be patched.
    """

    supports_image_edit = True

    def __init__(self, pipeline_config):
        if torch.cuda.get_device_capability() != _SM:
            raise RuntimeError(
                f"target certified on sm_{_SM[0]}{_SM[1]}, running on "
                f"sm_{''.join(map(str, torch.cuda.get_device_capability()))}"
            )
        super().__init__(pipeline_config)

    @property
    def _package(self):
        return self.pipeline_config.extra_attrs["flux3_checkpoint"]

    def _init_transformer(self) -> None:
        self.transformer = Flux3KleinTransformer(self.pipeline_config.primary_pretrained_config)

    def _setup_cuda_graphs(self) -> None:
        if not self.pipeline_config.cuda_graph.enable:
            return
        runner = CUDAGraphRunner(CUDAGraphRunnerConfig(use_cuda_graph=True))
        # Separate graphs for t2i and i2i (and per resolution, through the shapes).
        runner.register_extra_key_fn("flux3_layout", lambda *args, **kwargs: kwargs.get("layout"))
        runner.register_reusable_inputs(_GRAPH_REUSABLE_INPUTS)
        logger.info("CUDA graph runner: wrapping transformer.forward (with torch.compile)")
        self.transformer.forward = runner.wrap(self.transformer.forward)
        self._cuda_graph_runners["transformer"] = runner

    def load_transformer_weights(self, checkpoint_dir: str) -> dict:
        from safetensors.torch import load_file

        return load_file(str(self._package.weights))

    def load_weights(self, weights: dict) -> None:
        logger.info("Loading FLUX.3-klein NVFP4 transformer weights...")
        _weights.load(self.transformer, weights.get("transformer", weights))
        self.transformer.eval()

    def post_load_weights(self) -> None:
        # weights.load derived every prepared tensor; the tail kernel's workspace is zeroed
        # once here, outside any graph capture, and shared by every NVFP4 block.
        workspace = allocate_pair_workspace(self.device)
        for block in self.transformer.modules():
            if isinstance(block, _NVFP4Block):
                block.pair_workspace = workspace

    def load_standard_components(
        self,
        checkpoint_dir: str,
        device: torch.device,
        skip_components: list | None = None,
        **kwargs,
    ) -> None:
        from safetensors.torch import load_file

        package = self._package
        skip_components = skip_components or []
        if "text_encoder" not in skip_components:
            config = package.config["text_encoder"]
            self.text_encoder = (
                Flux3KleinTextEncoder(
                    model_path=str(package.asset(config["path"])),
                    tokenizer_path=str(package.asset(config["tokenizer"])),
                    output_layer=config["layers"],
                    max_length=config["max_tokens"],
                    pad_multiple=config["pad_multiple"],
                    dtype=getattr(torch, config["dtype"]),
                )
                .to(device)
                .eval()
            )
        if "vae" not in skip_components and "image_vae" not in skip_components:
            config = package.config["vae"]
            self.vae = Flux3KleinVAE.load(
                load_file(str(package.asset(config["path"]))),
                device,
                getattr(torch, config["dtype"]),
            )

    def offload_pipeline_components(self) -> dict[str, nn.Module]:
        return {}  # routing declines CPU offload

    def torch_compile(self) -> None:
        """Per-block compile (and the final layer), keeping the BF16 intermediate rounding."""
        transformer = self.transformer
        for name in ("image_blocks", "cond_blocks", "text_blocks", "single_blocks"):
            blocks = getattr(transformer, name)
            logger.info(f"torch.compile: transformer.{name} ({len(blocks)} blocks)")
            for i in range(len(blocks)):
                blocks[i] = torch.compile(
                    blocks[i],
                    dynamic=None,
                    fullgraph=False,
                    options={"emulate_precision_casts": True},
                )
        logger.info("torch.compile: transformer.final_layer")
        transformer.final_layer = torch.compile(
            transformer.final_layer,
            dynamic=None,
            fullgraph=False,
            options={"emulate_precision_casts": True},
        )

    @property
    def default_warmup_resolutions(self) -> list[tuple[int, int]]:
        return [(1024, 1024), (2048, 2048)]

    @property
    def default_warmup_num_frames(self) -> list[int]:
        return [1]

    def warmup_cache_key(self, height: int, width: int, num_frames: int) -> tuple:
        return (height, width)

    def validate_resolution(self, height: int, width: int, num_frames: int) -> None:
        super().validate_resolution(height, width, num_frames)
        if not Flux3KleinVAE.supports_output_size(height, width):
            raise ValueError(
                f"{height}x{width}: the FLUX.3-klein sm_120 target serves output sizes its VAE "
                "decodes (1024x1024 and 2048x2048); use modeling_v2='off' for others"
            )

    @property
    def default_generation_params(self) -> dict:
        sampler = self._package.config["sampler"]
        return dict(
            height=None,
            width=None,
            num_inference_steps=sampler["steps"],
            guidance_scale=sampler["guidance"],
            seed=0,
        )

    @property
    def ref_slot_specs(self) -> dict[str, RefSlotSpec]:
        return {
            "image_reference": RefSlotSpec(
                modality="image", roles=[RoleSpec(role="reference", min=0, max=None)]
            )
        }

    @property
    def extra_param_specs(self) -> dict:
        return {
            "mode": ExtraParamSchema(
                type="str",
                default=None,
                description="t2i or i2i; inferred from image references when omitted.",
            ),
        }

    def warmup(self) -> None:
        super().warmup()
        # 1024^2 references and outputs compile the VAE here when the warmup plan skipped them.
        if self.vae is not None and (1024, 1024) not in self._warmed_up_shapes:
            self.vae.warmup([(1024, 1024)])

    def _run_warmup(self, height: int, width: int, num_frames: int, steps: int) -> None:
        if self.text_encoder is not None:
            self.text_encoder.warmup()
        ctx = self._encode_prompt("warmup")
        modes = self.pipeline_config.compilation.modes or ["t2i", "i2i"]
        if "i2i" in modes:
            # The reference PNG decoder's kernels, compiled here rather than in a request.
            png_decode.warmup(self.device)
        default_steps = self._package.config["sampler"]["steps"]
        for mode in modes:
            images = [torch.zeros(1, 3, height, width, device=self.device)] if mode == "i2i" else []
            self._decode(self.generate(ctx, images, height, width, steps, seed=0), height, width)
            # The text stream runs once per request at M = steps * T, so the sampler's default
            # step count is its own compiled shape: warm it too.
            if mode == "t2i" and steps is not None and steps != default_steps:
                latent = self.generate(ctx, images, height, width, default_steps, seed=0)
                self._decode(latent, height, width)

    def _encode_prompt(self, prompt: str) -> Tensor:
        return self.text_encoder.encode([prompt])[0]

    @torch.inference_mode()
    def generate(
        self,
        ctx: Tensor,
        images: list[Tensor],
        height: int,
        width: int,
        num_inference_steps: int | None,
        seed: int,
        timer: CudaPhaseTimer | None = None,
    ) -> Tensor:
        """Context [1, T, C] (+ reference pixels) -> denoised image tokens [1, n, 128]."""
        package = self._package
        image_rope_scale = package.config["dit"]["image_rope_scale"]
        sampler = package.config["sampler"]
        steps = num_inference_steps if num_inference_steps is not None else sampler["steps"]
        transformer = self.transformer

        cond = None
        if images:
            # Queue the reference encodes first: the CPU randn + pinned copy of the noise then
            # overlaps the GPU encode (the seeded CPU generator does not depend on the order).
            vae_dtype = next(self.vae.parameters()).dtype
            latents = [self.vae.encode(image.to(vae_dtype))[0] for image in images]
            noise, noise_ids = _sample_image_latents(
                height, width, seed, self.device, image_rope_scale
            )
            cond = _pack_image_condition(latents, image_rope_scale)
        else:
            noise, noise_ids = _sample_image_latents(
                height, width, seed, self.device, image_rope_scale
            )

        text = ctx[0]
        text_ids = torch.cartesian_prod(
            torch.arange(1, device=text.device),
            torch.arange(1, device=text.device),
            torch.arange(1, device=text.device),
            torch.arange(text.shape[0], device=text.device),
        )
        sample = {"text": (text, text_ids), "image": (noise, noise_ids)}
        if cond is not None:
            sample["image_cond"] = cond
        packed = pack([sample], {"text": transformer.context_in_dim, **transformer.in_channels})
        inputs = packed.kwargs()
        layout = inputs["layout"]
        text_length = dict(layout)["text"][0]
        if text_length <= 0:
            raise ValueError("an empty prompt: the text stream needs at least one token")

        prepared = transformer.prepare_request(inputs, layout)
        scheduler = Flux3FlowMatchScheduler(sampler["alpha"])
        scheduler.set_timesteps(steps)
        device = inputs["x_text_timesteps"].device
        if timer is not None:
            timer.mark_denoise_start()
        # Once per request: each step's target-time row, the text stream for every step and
        # the image stream's vec rows (pinned, non-blocking host transfers of the schedule).
        targets = _sigmas_to_device(
            scheduler.sigmas[1 : len(scheduler.timesteps) + 1], device, torch.bfloat16
        )
        image_timesteps = _sigmas_to_device(scheduler.timesteps, device, torch.bfloat16)
        ref = transformer.ref_rows(targets)
        text_steps = transformer.prepare_text_steps(
            inputs["x_text"],
            inputs["x_text_timesteps"],
            ref,
            prepared.pop("text_cos"),
            prepared.pop("text_sin"),
        )
        image_steps = transformer.prepare_image_steps(image_timesteps, ref, noise.dtype)
        step_kwargs = {}
        if cond is not None:
            # The reference stream's time row (t = 0); its target row is added per step.
            ts = inputs["x_image_cond_timesteps"]
            step_kwargs["x_image_cond"] = inputs["x_image_cond"]
            step_kwargs["_flux3_image_cond_vec"] = transformer.time_in(
                _timestep_embedding(torch.stack([ts[0]]), _TIME_DIM, ts.dtype)
            ).unsqueeze(0)
            step_kwargs["timesteps_ref"] = inputs["x_text_timesteps"].new_empty(1)

        latent = inputs["x_image"]
        length = text_length
        for step_index, timestep in self._profile_denoise_steps(scheduler.timesteps):
            if cond is not None:
                step_kwargs["timesteps_ref"].fill_(scheduler.sigmas[step_index + 1].item())
            tokens = text_steps["tokens"]
            velocity = transformer(
                layout=layout,
                x_image=latent,
                _flux3_image_step_early=image_steps["early"][step_index],
                _flux3_image_step_single=image_steps["single"][step_index],
                _flux3_image_step_final=image_steps["final"][step_index],
                _flux3_text_step_tokens=tokens[:, step_index * length : (step_index + 1) * length],
                _flux3_text_step_single_shift=text_steps["single"][0][
                    :, step_index : step_index + 1
                ],
                _flux3_text_step_single_scale=text_steps["single"][1][
                    :, step_index : step_index + 1
                ],
                _flux3_text_step_single_gate=text_steps["single"][2][
                    :, step_index : step_index + 1
                ],
                **prepared,
                **step_kwargs,
            )
            latent = scheduler.step(velocity, timestep, latent, return_dict=False)[0]
        return latent.unsqueeze(0)

    @torch.inference_mode()
    def _decode(self, latent: Tensor, height: int, width: int) -> Tensor | None:
        h, w = height // 16, width // 16
        # Raster-ordered tokens: the grid is a view of a contiguous fp32 copy ([1, C, h, w]
        # with channel stride 1, the layout the compiled decoder is specialized on).
        data = latent[0].to(torch.float32, memory_format=torch.contiguous_format)
        grid = data.reshape(1, h, w, data.shape[1]).permute(0, 3, 1, 2)
        return self.decode_latents(
            grid, lambda g: self.vae.decode(g.to(next(self.vae.parameters()).dtype))
        )

    @staticmethod
    def _to_uint8_image(x: Tensor) -> Tensor:
        """VAE pixels in [-1, 1] -> uint8 [B, H, W, C]."""
        x = (x / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1)
        return (x * 255).round().to(torch.uint8)

    def forward(self, mode: str, **kwargs):
        raise NotImplementedError("serve through infer(); warmup calls generate()")

    @torch.inference_mode()
    def infer(self, req) -> PipelineOutput:
        p = req.params
        extra = p.extra_params or {}
        mode = extra.get("mode") or ("i2i" if p.image_reference else "t2i")
        if mode not in ("t2i", "i2i"):
            raise ValueError(f"Unsupported FLUX.3 image mode: {mode}")
        if isinstance(req.prompt, (list, tuple)) and len(req.prompt) != 1:
            raise ValueError("FLUX.3 serves one prompt per request")
        if mode == "i2i" and not p.image_reference:
            raise ValueError("i2i requires at least one image reference")
        if mode == "t2i" and p.image_reference:
            raise ValueError("Use mode=i2i with image references")
        if p.guidance_scale is not None:
            raise ValueError("FLUX.3 flow-map checkpoints sample without CFG; omit guidance")
        if p.negative_prompt:
            raise ValueError("FLUX.3 flow-map checkpoints do not support negative prompts")
        if extra.get("pag_single_layers"):
            raise ValueError("FLUX.3 flow-map checkpoints do not support PAG")
        prompt = req.prompt[0] if isinstance(req.prompt, (list, tuple)) else req.prompt
        timer = CudaPhaseTimer()
        timer.mark_pre_start()
        ctx = self._encode_prompt(prompt)
        images = [
            pipeline_flux3._load_image_pixels(ref.content, device=self.device)
            for ref in p.image_reference or []
        ]
        width, height = self._package.config["sampler"]["size"]
        if images:
            h, w = images[0].shape[-2:]
            scale = math.sqrt(width * height / (w * h))
            width, height = max(16, round(w * scale / 16) * 16), max(16, round(h * scale / 16) * 16)
        height, width = p.height or height, p.width or width
        self.validate_resolution(height, width, 1)
        latent = self.generate(ctx, images, height, width, p.num_inference_steps, p.seed, timer)
        timer.mark_post_start()
        result = PipelineOutput(image=self._to_uint8_image(self._decode(latent, height, width)))
        timer.mark_end()
        return timer.fill(result)
