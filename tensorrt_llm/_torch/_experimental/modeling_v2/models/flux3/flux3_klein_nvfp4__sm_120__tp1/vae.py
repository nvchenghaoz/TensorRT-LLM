# SPDX-FileCopyrightText: Copyright (c) 2026 Black Forest Labs, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FLUX.3 klein image VAE as this target serves it (SM120).

The packaged VAE is a KL-VAE encoder / decoder (ch 128, ch_mult 1-2-4-4, 2 ResnetBlocks a
level, z 32) with a 2x2 pixel-unshuffle and an eval BatchNorm latent normalizer. The
module tree and the persistent state_dict keys are the package's; only the served path
is implemented.

Encoder (i2i references), fp32 as packaged, compiled whole (dynamic=False) on
channels_last weights and input: cuDNN NHWC convs (TF32, torch's default), GroupNorm
from per-channel statistics (fp32 accumulation, exact group combine), and a mid-block
attention with bf16 operands, fp32 scores and the persistent row softmax.

Decoder, compiled whole (dynamic=False, emulate_precision_casts) on channels_last
weights: a bf16 trunk with conv_out (and the output pixels) in fp32;
  - every ResnetBlock conv1 / conv2 (14 blocks, 28 convs) an e4m3 cuDNN conv whose epilogue adds
    the bias and the residual (per-Cout weight scales, per-tensor activation scale from
    the GroupNorm + swish prologue); a 1x1 shortcut runs bias-free and its bias joins
    conv2's as one fp32 epilogue bias;
  - the mid-block attention as the split e4m3 kernel pair;
  - each upsample (nearest-2x + 3x3 conv) as one stride-2 cuDNN dgrad with an fp32 bias
    epilogue;
  - the last block's residual add, norm_out, swish and conv_out as one Triton kernel.

``Flux3KleinVAE.load`` builds it from the packaged fp32 state_dict; ``warmup`` compiles
and builds every graph for the given output sizes.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from tensorrt_llm.logger import logger

from .ops.vae._split_attn_e4m3 import supported as split_attn_supported
from .ops.vae._subpixel_up_conv import subpixel_transpose_weight
from .ops.vae.add_norm_swish_conv_out_3x3 import add_norm_swish_conv_out_3x3
from .ops.vae.fp8_conv3x3 import fp8_conv3x3
from .ops.vae.row_softmax_bf16 import row_softmax_bf16
from .ops.vae.row_softmax_bf16 import supported as row_softmax_supported
from .ops.vae.split_attn_e4m3 import split_attn_e4m3
from .ops.vae.subpixel_up_conv import subpixel_up_conv

# The packaged VAE's architecture.
IN_CHANNELS = 3
CH = 128
OUT_CH = 3
CH_MULT = (1, 2, 4, 4)
NUM_RES_BLOCKS = 2
Z_CHANNELS = 32
#: Latent channels after the 2x2 pixel-unshuffle (the transformer's image channels).
LATENT_CHANNELS = 4 * Z_CHANNELS

FP8 = torch.float8_e4m3fn
QMAX = 448.0
# |min_t swish(t)| = 0.2784645... (at t = -1.27846), rounded UP so the bound holds.
SWISH_NEG_BOUND = 0.2784646

_CL = torch.channels_last
_DECODER_COMPILE_OPTIONS = {"layout_optimization": False, "emulate_precision_casts": True}
_ENCODER_COMPILE_OPTIONS = {"layout_optimization": False}


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


# --------------------------------------------------------------------------- GroupNorm
# The three functions below trace into the compiled graphs; each keeps its exact op
# sequence (Inductor's kernels, and so the output bits, follow the traced ops).


def group_norm_channel_stats(
    x: Tensor, num_groups: int, weight: Tensor, bias: Tensor, eps: float
) -> Tensor:
    """GroupNorm from per-channel statistics (F.group_norm's math, reordered).

    Each channel's population mean / variance over (H, W) is reduced first in fp32 (a
    coalesced reduction on channels_last activations); a group's statistics are then
    combined exactly from its C/G equal-count channels:
    mean_g = mean_c(mean_c), var_g = mean_c(var_c) + mean_c((mean_c - mean_g)^2).
    The normalization is computed in fp32; the output has x's dtype.
    """
    n, c = x.shape[:2]
    g = num_groups
    xf = x.float()
    red = tuple(range(2, x.dim()))
    var_c, mean_c = torch.var_mean(x.to(torch.float32), dim=red, correction=0)
    mean_c = mean_c.view(n, g, c // g)
    var_c = var_c.view(n, g, c // g)
    mean_g = mean_c.mean(-1, keepdim=True)
    var_g = var_c.mean(-1, keepdim=True) + (mean_c - mean_g).square().mean(-1, keepdim=True)
    rstd_g = torch.rsqrt(var_g + eps)
    bshape = (n, c) + (1,) * (x.dim() - 2)
    mean = mean_g.expand(n, g, c // g).reshape(bshape).float()
    rstd = rstd_g.expand(n, g, c // g).reshape(bshape).float()
    y = (xf - mean) * rstd
    y = y * weight.float().view((1, c) + (1,) * (x.dim() - 2))
    y = y + bias.float().view((1, c) + (1,) * (x.dim() - 2))
    return y.to(x.dtype)


def gn_swish_fp8(x: Tensor, norm: nn.GroupNorm) -> tuple[Tensor, Tensor]:
    """swish(GroupNorm(x)) quantized to e4m3 with a per-tensor scale: (q, s_x [1] fp32).

    Statistics as group_norm_channel_stats, plus per-channel min / max of x. With
    a_c = gamma_c * rstd_g and b_c = beta_c - mean_g * a_c, channel c's GroupNorm output
    lies in the affine image of [min_c, max_c], so amax = max(max_c swish(max(hi_c, 0)),
    |min swish|) bounds |swish(GN(x))| (swish is increasing above -1.278 and >= -0.27846)
    without a pass over the activation. s_x = amax / 448 and
    q = sat_e4m3(swish(a_c * x + b_c) / s_x), all fp32, IEEE division, clamped before the
    (non-saturating) cast.
    """
    n, c = x.shape[:2]
    g = norm.num_groups
    red = tuple(range(2, x.dim()))
    var_c, mean_c = torch.var_mean(x.to(torch.float32), dim=red, correction=0)
    mn_c = torch.amin(x, dim=red).float()
    mx_c = torch.amax(x, dim=red).float()
    mean_c = mean_c.view(n, g, c // g)
    var_c = var_c.view(n, g, c // g)
    mean_g = mean_c.mean(-1, keepdim=True)
    var_g = var_c.mean(-1, keepdim=True) + (mean_c - mean_g).square().mean(-1, keepdim=True)
    rstd_g = torch.rsqrt(var_g + norm.eps)
    mean = mean_g.expand(n, g, c // g).reshape(n, c).float()
    rstd = rstd_g.expand(n, g, c // g).reshape(n, c).float()
    a = norm.weight.float().view(1, c) * rstd
    b = norm.bias.float().view(1, c) - mean * a
    hi = torch.maximum(a * mn_c + b, a * mx_c + b).clamp_min(0.0)
    top = (hi * torch.sigmoid(hi)).amax()
    s_x = (torch.clamp_min(top, SWISH_NEG_BOUND) / QMAX).reshape(1)
    bshape = (n, c) + (1,) * (x.dim() - 2)
    y = x.float() * a.view(bshape) + b.view(bshape)
    y = y * torch.sigmoid(y)
    q = torch.div(y, s_x.view((1,) * x.dim())).clamp(-QMAX, QMAX).to(FP8)
    return q, s_x


def group_norm_channel_affine(
    x: Tensor, num_groups: int, weight: Tensor, bias: Tensor, eps: float
) -> tuple[Tensor, Tensor]:
    """fp32 [N, C] (scale, shift) with GroupNorm(x) == x * scale + shift per channel
    (statistics as group_norm_channel_stats)."""
    n, c = x.shape[:2]
    g = num_groups
    red = tuple(range(2, x.dim()))
    var_c, mean_c = torch.var_mean(x.to(torch.float32), dim=red, correction=0)
    mean_c = mean_c.view(n, g, c // g)
    var_c = var_c.view(n, g, c // g)
    mean_g = mean_c.mean(-1, keepdim=True)
    var_g = var_c.mean(-1, keepdim=True) + (mean_c - mean_g).square().mean(-1, keepdim=True)
    rstd_g = torch.rsqrt(var_g + eps)
    mean = mean_g.expand(n, g, c // g).reshape(n, c).float()
    rstd = rstd_g.expand(n, g, c // g).reshape(n, c).float()
    scale = rstd * weight.float().view(1, c)
    shift = bias.float().view(1, c) - mean * scale
    return scale.contiguous(), shift.contiguous()


class GroupNorm(nn.GroupNorm):
    """nn.GroupNorm (same parameters) computed by group_norm_channel_stats."""

    def __init__(self, num_channels: int):
        super().__init__(num_groups=32, num_channels=num_channels, eps=1e-6, affine=True)

    def forward(self, x: Tensor) -> Tensor:
        return group_norm_channel_stats(x, self.num_groups, self.weight, self.bias, self.eps)

    def forward_silu(self, x: Tensor) -> Tensor:
        return swish(self.forward(x))


# --------------------------------------------------------------------------- blocks
class _AttnBlock(nn.Module):
    """Single-head self-attention over the h * w tokens (package parameter names)."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.norm = GroupNorm(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        h_ = self.norm(x)
        return x + self.proj_out(self._attention(self.q(h_), self.k(h_), self.v(h_)))


class EncoderAttnBlock(_AttnBlock):
    @staticmethod
    def _attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """fp32 q / k / v rounded to bf16; S = q^T k as a bf16 GEMM with fp32 output (the
        scores are never rounded); softmax(S * C^-0.5) rounded to bf16; O = v P^T as a
        bf16 GEMM with fp32 accumulation, back in q's dtype."""
        b, c, h, w = q.shape
        dt = q.dtype
        q = q.reshape(b, c, h * w).to(torch.bfloat16)
        k = k.reshape(b, c, h * w).to(torch.bfloat16)
        v = v.reshape(b, c, h * w).to(torch.bfloat16)
        s = torch.bmm(q.transpose(1, 2), k, out_dtype=torch.float32)
        if row_softmax_supported(s):
            p = row_softmax_bf16(s, c**-0.5)
        else:
            # A reference image whose mid-block token count is not a multiple of 8192.
            p = torch.softmax(s * (c**-0.5), dim=-1).to(torch.bfloat16)
        o = torch.bmm(v, p.transpose(1, 2))
        return o.reshape(b, c, h, w).to(dt)


class DecoderAttnBlock(_AttnBlock):
    @staticmethod
    def _attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """bf16 q / k / v (b == 1); V quantized per channel to e4m3 (s_v = amax / 448),
        then the split e4m3 attention; written [T, c] = channels_last [b, c, h, w]."""
        b, c, h, w = q.shape
        dt = q.dtype
        q = q.reshape(b, c, h * w).to(torch.bfloat16)
        k = k.reshape(b, c, h * w).to(torch.bfloat16)
        v = v.reshape(b, c, h * w).to(torch.bfloat16)
        vf = v[0].float()  # [c, T]
        s_v = (vf.abs().amax(dim=-1, keepdim=True) / 448.0).clamp(min=1e-30)  # [c, 1]
        vq = (vf / s_v).clamp(-448.0, 448.0).to(FP8).contiguous()  # [c, T] row-major
        o_t = split_attn_e4m3(q[0], k[0], vq, s_v.reshape(c), c**-0.5)  # [T, c]
        return o_t.t().reshape(b, c, h, w).to(dt)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = GroupNorm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = GroupNorm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1.forward_silu(x)
        h = self.conv1(h)
        h = self.norm2.forward_silu(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class Fp8ResnetBlock(ResnetBlock):
    """conv1 / conv2 as e4m3 cuDNN convs fed by gn_swish_fp8 (after ``quantize_convs``).

    Buffers (non-persistent, set by quantize_convs): ``w1q`` / ``w2q`` e4m3 KRSC weights,
    ``s1`` / ``s2`` fp32 per-Cout scales, and for a 1x1-shortcut block ``bias2_sc`` =
    conv2.bias + nin_shortcut.bias in fp32 (conv2's epilogue bias).
    """

    @torch.no_grad()
    def quantize_convs(self) -> None:
        """Quantize conv1 / conv2 (bf16 trunk weights) to e4m3 with per-Cout scales."""
        for conv, wname, sname in ((self.conv1, "w1q", "s1"), (self.conv2, "w2q", "s2")):
            wq, s_w = quantize_conv3x3_weight(conv.weight)
            self.register_buffer(wname, wq, persistent=False)
            self.register_buffer(sname, s_w, persistent=False)
        if self.in_channels != self.out_channels:
            bias = self.conv2.bias.float() + self.nin_shortcut.bias.float()
            self.register_buffer("bias2_sc", bias, persistent=False)

    def _conv2_input(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """(e4m3 conv2 input, its scale)."""
        hq, sx = gn_swish_fp8(x, self.norm1)
        h = fp8_conv3x3(hq, sx, self.w1q, self.s1, self.conv1.bias, None)
        return gn_swish_fp8(h, self.norm2)

    def forward(self, x: Tensor) -> Tensor:
        hq, sx = self._conv2_input(x)
        if self.in_channels != self.out_channels:
            # The 1x1 shortcut runs bias-free; its bias is part of conv2's fp32 epilogue
            # bias, and its output is conv2's residual.
            sc = nn.functional.conv2d(x, self.nin_shortcut.weight, None)
            return fp8_conv3x3(hq, sx, self.w2q, self.s2, self.bias2_sc, sc)
        return fp8_conv3x3(hq, sx, self.w2q, self.s2, self.conv2.bias, x)

    def forward_parts(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """(shortcut, conv2 output without bias or residual, conv2 bias) of a
        same-channel block: forward(x) == shortcut + (conv2 output + bias)."""
        hq, sx = self._conv2_input(x)
        return x, fp8_conv3x3(hq, sx, self.w2q, self.s2, None, None), self.conv2.bias


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        x = nn.functional.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)


class Upsample(nn.Module):
    """nearest-2x upsample + 3x3 conv as one stride-2 dgrad with the transposed sub-pixel
    weight (``transpose_weight``, a non-persistent buffer derived from conv.weight at
    load by ``Flux3KleinVAE.load``)."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.register_buffer(
            "transpose_weight", torch.zeros(in_channels, in_channels, 4, 4), persistent=False
        )

    def forward(self, x: Tensor) -> Tensor:
        return subpixel_up_conv(x, self.transpose_weight, self.conv.bias)


# --------------------------------------------------------------------------- encoder
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant_conv = nn.Conv2d(2 * Z_CHANNELS, 2 * Z_CHANNELS, 1)
        self.num_resolutions = len(CH_MULT)
        self.conv_in = nn.Conv2d(IN_CHANNELS, CH, kernel_size=3, stride=1, padding=1)
        in_ch_mult = (1,) + CH_MULT
        self.down = nn.ModuleList()
        block_in = CH
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            block_in = CH * in_ch_mult[i_level]
            block_out = CH * CH_MULT[i_level]
            for _ in range(NUM_RES_BLOCKS):
                block.append(ResnetBlock(block_in, block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
            self.down.append(down)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in)
        self.mid.attn_1 = EncoderAttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in)
        self.norm_out = GroupNorm(block_in)
        self.conv_out = nn.Conv2d(block_in, 2 * Z_CHANNELS, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(NUM_RES_BLOCKS):
                hs.append(self.down[i_level].block[i_block](hs[-1]))
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = self.norm_out.forward_silu(h)
        h = self.conv_out(h)
        h = self.quant_conv(h)
        return h


# --------------------------------------------------------------------------- decoder
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.post_quant_conv = nn.Conv2d(Z_CHANNELS, Z_CHANNELS, 1)
        self.num_resolutions = len(CH_MULT)
        block_in = CH * CH_MULT[-1]
        self.conv_in = nn.Conv2d(Z_CHANNELS, block_in, kernel_size=3, stride=1, padding=1)
        self.mid = nn.Module()
        self.mid.block_1 = Fp8ResnetBlock(block_in, block_in)
        self.mid.attn_1 = DecoderAttnBlock(block_in)
        self.mid.block_2 = Fp8ResnetBlock(block_in, block_in)
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = CH * CH_MULT[i_level]
            for _ in range(NUM_RES_BLOCKS + 1):
                block.append(Fp8ResnetBlock(block_in, block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            if i_level != 0:
                up.upsample = Upsample(block_in)
            self.up.insert(0, up)
        self.norm_out = GroupNorm(block_in)
        self.conv_out = nn.Conv2d(block_in, OUT_CH, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        z = self.post_quant_conv(z.to(self.post_quant_conv.weight.dtype))
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(NUM_RES_BLOCKS + 1):
                if i_level == 0 and i_block == NUM_RES_BLOCKS:
                    break  # the last block runs inside the fused tail
                h = self.up[i_level].block[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        # Fused tail: conv_out(swish(norm_out(x + (r + r_bias)))) of the last block's parts;
        # norm_out's per-channel affine comes from the statistics of the sum, which the
        # kernel re-forms as it loads its input (never stored).
        x, r, r_bias = self.up[0].block[NUM_RES_BLOCKS].forward_parts(h)
        h = x + (r + r_bias.view(1, -1, 1, 1))
        norm, conv = self.norm_out, self.conv_out
        scale, shift = group_norm_channel_affine(
            h, norm.num_groups, norm.weight, norm.bias, norm.eps
        )
        return add_norm_swish_conv_out_3x3(x, r, r_bias, scale, shift, conv.weight, conv.bias)


# --------------------------------------------------------------------------- weights
def quantize_conv3x3_weight(weight: Tensor) -> tuple[Tensor, Tensor]:
    """[Cout, Cin, 3, 3] -> (e4m3 KRSC [Cout, 3, 3, Cin] contiguous, s_w [Cout] fp32).

    s_w[o] = amax |W[o]| / 448 (1 for an all-zero row); saturating cast (clamp to +-448
    first: ``.to(float8_e4m3fn)`` does not saturate); IEEE division.
    """
    wf = weight.detach().float()
    amax = wf.abs().amax(dim=(1, 2, 3))
    s_w = torch.where(amax > 0, amax / QMAX, torch.ones_like(amax))
    q = torch.div(wf, s_w.view(-1, 1, 1, 1)).clamp(-QMAX, QMAX).to(FP8)
    return q.permute(0, 2, 3, 1).contiguous(), s_w.contiguous()


# --------------------------------------------------------------------------- VAE
class Flux3KleinVAE(nn.Module):
    """FLUX.3 klein image VAE: encode (i2i references) and decode, served path only."""

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.bn_eps = 1e-4
        self.ps = [2, 2]
        self.bn = nn.BatchNorm2d(
            math.prod(self.ps) * Z_CHANNELS,
            eps=self.bn_eps,
            momentum=0.1,
            affine=False,
            track_running_stats=True,
        )

    # ------------------------------------------------------------------ load
    @classmethod
    @torch.no_grad()
    def load(
        cls, state_dict: dict[str, Tensor], device: torch.device, dtype: torch.dtype
    ) -> "Flux3KleinVAE":
        """Build the served VAE from the package's state_dict.

        ``dtype`` is the package's VAE dtype; the target is built for float32 (the
        encoder runs as packaged, the decoder trunk is cast to bf16 from it). Steps, in
        the order that fixes the bits:
          1. load on the CPU; derive each upsample's transposed sub-pixel weight there
             from the fp32 conv weight (fp64 tap sums, one rounding to fp32);
          2. move to ``device`` in fp32; cast every decoder child except conv_out to bf16
             (the derived weights round once more, to bf16);
          3. decoder and encoder weights to channels_last; quantize the decoder
             ResnetBlock convs to e4m3 from the bf16 channels_last weights;
          4. torch.compile decoder and encoder in place (compiled lazily, per shape).
        """
        if dtype != torch.float32:
            raise ValueError(
                f"the FLUX.3 klein sm_120 VAE is built from a float32 package VAE, got {dtype}"
            )
        with torch.device("cpu"):
            vae = cls()
        vae.load_state_dict(state_dict)
        for m in vae.decoder.modules():
            if isinstance(m, Upsample):
                m.transpose_weight = subpixel_transpose_weight(m.conv.weight.cpu())
        vae = vae.to(device=device, dtype=dtype).eval()

        decoder = vae.decoder
        for name, child in decoder.named_children():
            if name != "conv_out":
                child.to(torch.bfloat16)
        decoder.to(memory_format=_CL)
        n = 0
        for m in decoder.modules():
            if isinstance(m, Fp8ResnetBlock):
                m.quantize_convs()
                n += 2
        logger.info(f"FLUX.3 klein VAE decoder: bf16 trunk, {n} e4m3 ResnetBlock convs")
        vae.encoder.to(memory_format=_CL)

        decoder.compile(dynamic=False, options=_DECODER_COMPILE_OPTIONS)
        vae.encoder.compile(dynamic=False, options=_ENCODER_COMPILE_OPTIONS)
        logger.info("torch.compile: FLUX.3 klein VAE decoder and encoder (dynamic=False)")
        return vae

    # ------------------------------------------------------------------ shapes
    @staticmethod
    def supports_output_size(height: int, width: int) -> bool:
        """Output sizes decode() serves: multiples of 16 whose mid-block token count
        (height / 8 * width / 8) the split attention kernels take (1024^2 and 2048^2)."""
        return (
            height % 16 == 0
            and width % 16 == 0
            and split_attn_supported(CH * CH_MULT[-1], (height // 8) * (width // 8))
        )

    # ------------------------------------------------------------------ latents
    def normalize(self, z: Tensor) -> Tensor:
        self.bn.eval()
        return self.bn(z)

    def inv_normalize(self, z: Tensor) -> Tensor:
        self.bn.eval()
        s = torch.sqrt(self.bn.running_var.view(1, -1, 1, 1) + self.bn_eps)
        m = self.bn.running_mean.view(1, -1, 1, 1)
        return z * s + m

    def encode(self, x: Tensor) -> Tensor:
        """fp32 pixels [1, 3, H, W] in [-1, 1] -> normalized latents [1, 128, H/16, W/16].

        The compiled encoder always sees a canonical channels_last input (one graph per
        size, whatever the caller's layout) and its moments come back contiguous.
        """
        moments = self.encoder(x.clone(memory_format=_CL))
        moments = moments.contiguous()
        mean = torch.chunk(moments, 2, dim=1)[0]
        z = rearrange(
            mean, "... c (i pi) (j pj)  -> ... (c pi pj) i j", pi=self.ps[0], pj=self.ps[1]
        )
        return self.normalize(z)

    def decode(self, z: Tensor) -> Tensor:
        """Normalized latents [1, 128, H/16, W/16] (fp32) -> fp32 pixels [1, 3, H, W].

        The compiled decoder is specialized on the latent's strides as well as its shape;
        warmup() uses the layout the pipeline's token-to-grid view produces.
        """
        height, width = 16 * z.shape[-2], 16 * z.shape[-1]
        if z.shape[0] != 1 or not self.supports_output_size(height, width):
            raise ValueError(
                f"FLUX.3 klein sm_120 VAE decodes one {height}x{width} image only when "
                "supports_output_size(height, width) (the split mid-block attention's "
                f"shapes); got latents {tuple(z.shape)}"
            )
        z = self.inv_normalize(z)
        z = rearrange(z, "... (c pi pj) i j -> ... c (i pi) (j pj)", pi=self.ps[0], pj=self.ps[1])
        return self.decoder(z)

    # ------------------------------------------------------------------ warmup
    @torch.inference_mode()
    def warmup(self, shapes: list[tuple[int, int]]) -> None:
        """Compile the encoder and decoder and build every cuDNN graph for each
        (height, width), so no request pays for it. Runs under inference_mode like the
        pipeline's calls (the compiled graphs are guarded on it). The decode input has
        the pipeline's latent-grid layout: a [1, H/16, W/16, 128] buffer viewed as
        [1, 128, H/16, W/16]."""
        device = self.bn.running_mean.device
        for height, width in shapes:
            self.encode(torch.zeros(1, IN_CHANNELS, height, width, device=device))
            grid = torch.zeros(1, height // 16, width // 16, LATENT_CHANNELS, device=device)
            self.decode(grid.permute(0, 3, 1, 2))
            torch.cuda.synchronize(device)
            logger.info(f"FLUX.3 klein VAE: warmed {height}x{width} encode + decode")
