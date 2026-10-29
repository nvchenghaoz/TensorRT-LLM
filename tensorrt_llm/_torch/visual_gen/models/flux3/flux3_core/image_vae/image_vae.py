# SPDX-FileCopyrightText: Copyright (c) 2026 Black Forest Labs, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from . import decoder_tail_triton as _tail
from . import fp8_conv as _fp8c
from . import subpixel_conv as _spc
from .decoder_tail_triton import (
    add_norm_swish_conv_out_3x3,
    can_use_conv_out_kernel,
    can_use_fused_tail,
    conv_out_3x3,
    group_norm_channel_affine,
)
from .group_norm_triton import can_use_fused as can_use_fused_gn
from .group_norm_triton import fused_group_norm


@dataclass
class Flux3ImageVAEConfig:
    resolution: int = 256
    in_channels: int = 3
    ch: int = 128
    out_ch: int = 3
    ch_mult: list[int] = field(default_factory=lambda: [1, 2, 4, 4])
    num_res_blocks: int = 2
    z_channels: int = 32


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


def group_norm_channel_stats(
    x: Tensor,
    num_groups: int,
    weight: Tensor,
    bias: Tensor,
    eps: float,
    acc_dtype: torch.dtype | None = None,
) -> Tensor:
    """GroupNorm from per-channel statistics (same math as F.group_norm, reordered).

    Each channel's population mean/variance over (H, W) is reduced first; a group's
    statistics are then combined exactly from its C/G equal-count channels:
    mean_g = mean_c(mean_c), var_g = mean_c(var_c) + mean_c((mean_c - mean_g)^2).
    On channels_last activations the per-channel reduction is coalesced (C is the
    contiguous dim), unlike the (N, G, C/G*H*W) group view. The normalization is
    computed in fp32 and the output has x's dtype. Statistics accumulate in fp32 for
    low-precision inputs (the served bf16 trunk) and in fp64 for fp32 inputs (the fp32
    reference decoder), so neither is less accurate than aten's fp32 group statistics.
    acc_dtype overrides that choice (the compiled fp32 encoder uses fp32, as aten's
    GroupNorm does: SM120 fp64 runs at ~1/64 of the fp32 rate).
    """
    n, c = x.shape[:2]
    g = num_groups
    xf = x.float()
    if acc_dtype is not None:
        acc = acc_dtype
    else:
        acc = torch.float64 if x.dtype == torch.float32 else torch.float32
    red = tuple(range(2, x.dim()))
    var_c, mean_c = torch.var_mean(x.to(acc), dim=red, correction=0)
    mean_c = mean_c.view(n, g, c // g)
    var_c = var_c.view(n, g, c // g)
    mean_g = mean_c.mean(-1, keepdim=True)
    var_g = var_c.mean(-1, keepdim=True) + (mean_c - mean_g).square().mean(-1, keepdim=True)
    rstd_g = torch.rsqrt(var_g + eps)
    bshape = (n, c) + (1,) * (x.dim() - 2)
    mean = mean_g.expand(n, g, c // g).reshape(bshape).float()
    rstd = rstd_g.expand(n, g, c // g).reshape(bshape).float()
    y = (xf - mean) * rstd
    if weight is not None:
        y = y * weight.float().view((1, c) + (1,) * (x.dim() - 2))
    if bias is not None:
        y = y + bias.float().view((1, c) + (1,) * (x.dim() - 2))
    return y.to(x.dtype)


class Flux3GroupNorm(nn.GroupNorm):
    """nn.GroupNorm with a split-group fused Triton path (same parameter names).

    ``forward_silu(x)`` computes ``swish(GroupNorm(x))`` in one apply pass.

    Under torch.compile the Triton kernels are opaque to Inductor, so by default the
    traced graph uses aten GroupNorm, which Inductor lowers into split reductions fused
    with the neighbouring residual adds, upsampling and swish. With
    ``channel_stats_under_compile`` the traced graph instead uses
    group_norm_channel_stats (per-channel var_mean + exact group combine), which
    Inductor reduces efficiently on channels_last activations.
    """

    # Per instance: True keeps the Triton kernels inside a compiled graph (as extern calls).
    triton_under_compile = False
    # Per instance: True traces group_norm_channel_stats instead of aten GroupNorm.
    channel_stats_under_compile = False
    # Per instance: accumulation dtype of group_norm_channel_stats (None: its default).
    channel_stats_acc_dtype = None

    def _fused(self, x: Tensor) -> bool:
        if torch.compiler.is_compiling() and not self.triton_under_compile:
            return False
        return can_use_fused_gn(x, self)

    def _unfused(self, x: Tensor) -> Tensor:
        if self.channel_stats_under_compile and torch.compiler.is_compiling():
            return group_norm_channel_stats(
                x, self.num_groups, self.weight, self.bias, self.eps, self.channel_stats_acc_dtype
            )
        return super().forward(x)

    def forward(self, x: Tensor) -> Tensor:
        if self._fused(x):
            return fused_group_norm(x, self.num_groups, self.weight, self.bias, self.eps, False)
        return self._unfused(x)

    def forward_silu(self, x: Tensor) -> Tensor:
        if self._fused(x):
            return fused_group_norm(x, self.num_groups, self.weight, self.bias, self.eps, True)
        return swish(self._unfused(x))


class Flux3ImageAttnBlock(nn.Module):
    # Set per instance by compile_decoder(fp8_convs=True) when the FP8 convs are on:
    # P.V runs as an e4m3 _scaled_mm (output class precision; see _attention_tc).
    _fp8_pv = False

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Flux3GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        if q.is_cuda:
            return self._attention_tc(q, k, v, self._fp8_pv)
        q = rearrange(q, "b c h w -> b 1 (h w) c")
        k = rearrange(k, "b c h w -> b 1 (h w) c")
        v = rearrange(v, "b c h w -> b 1 (h w) c")
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    @staticmethod
    def _attention_tc(q: Tensor, k: Tensor, v: Tensor, fp8_pv: bool = False) -> Tensor:
        """Single-head attention over h*w tokens on bf16 tensor cores.

        SDPA on the rearranged (non-contiguous, d=512) q/k/v takes the MATH path,
        an fp32 SIMT sgemm plus a 16384^2 fp32 softmax (8.6 ms at 1024^2). Here:
        S = q^T k as a bf16 GEMM with fp32 output (scores are never rounded to bf16),
        fp32 softmax with the d^-0.5 scale, P rounded to bf16, O = v P^T as a bf16
        GEMM with fp32 accumulation, written directly in the b c (h w) layout.
        q/k/v are rounded to bf16 (a no-op on the bf16 trunk); the output keeps the
        input dtype.

        fp8_pv=True (served FP8 decoder, b == 1; output class precision): the softmax
        writes the UNNORMALIZED E = exp(z - rowmax z) as e4m3 (E * 448, E <= 1) and keeps
        l = sum E in fp32 from the unrounded E; V is quantized per channel (amax / 448).
        O^T = P V^T then runs as one rowwise-scaled e4m3 _scaled_mm (fp32 accumulation,
        scale_a = 1 / (448 l) [T, 1], scale_b = s_v [1, c], one bf16 rounding in the
        epilogue), written [T, c] row-major = channels_last [b, c, h, w].
        """
        b, c, h, w = q.shape
        dt = q.dtype
        q = q.reshape(b, c, h * w).to(torch.bfloat16)
        k = k.reshape(b, c, h * w).to(torch.bfloat16)
        v = v.reshape(b, c, h * w).to(torch.bfloat16)
        s = torch.bmm(q.transpose(1, 2), k, out_dtype=torch.float32)
        if fp8_pv and b == 1:
            fp8 = torch.float8_e4m3fn
            z = s[0] * (c**-0.5)
            e = torch.exp(z - z.amax(dim=-1, keepdim=True))
            scale_a = 1.0 / (448.0 * e.sum(dim=-1, keepdim=True))  # [T, 1] fp32
            pq = (e * 448.0).to(fp8)
            vf = v[0].float()  # [c, T]
            s_v = (vf.abs().amax(dim=-1, keepdim=True) / 448.0).clamp(min=1e-30)  # [c, 1]
            vq = (vf / s_v).clamp(-448.0, 448.0).to(fp8).contiguous()  # [c, T] row-major
            o_t = torch._scaled_mm(
                pq, vq.t(), scale_a, s_v.reshape(1, c), out_dtype=torch.bfloat16
            )  # [T, c]
            return o_t.t().reshape(b, c, h, w).to(dt)
        p = torch.softmax(s * (c**-0.5), dim=-1).to(torch.bfloat16)
        o = torch.bmm(v, p.transpose(1, 2))
        return o.reshape(b, c, h, w).to(dt)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class Flux3ImageResnetBlock(nn.Module):
    # Set per instance by fp8_conv.prepare_fp8_convs (compile_decoder(fp8_convs=True)):
    # conv1/conv2 then run as e4m3 cuDNN convs fed by gn_swish_fp8 (output class precision).
    _fp8 = False

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = Flux3GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Flux3GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def _fp8_conv2_input(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """FP8 path up to conv2's quantized input: (e4m3 h, s_x)."""
        hq, sx = _fp8c.gn_swish_fp8(x, self.norm1)
        c1 = self.conv1
        h = _fp8c.fp8_conv3x3(hq, sx, c1.fp8_wq, c1.fp8_sw, c1.bias, None)
        return _fp8c.gn_swish_fp8(h, self.norm2)

    def forward(self, x: Tensor) -> Tensor:
        if self._fp8:
            hq, sx = self._fp8_conv2_input(x)
            c2 = self.conv2
            if self.in_channels != self.out_channels and c2.bias is not None:
                # The 1x1 shortcut runs bias-free and its bias joins conv2's bias as one
                # fp32 epilogue bias: no separate bias-add pass over the shortcut output.
                nin = self.nin_shortcut
                sc = nn.functional.conv2d(x, nin.weight, None)
                bias = c2.bias.float()
                if nin.bias is not None:
                    bias = bias + nin.bias.float()
                return _fp8c.fp8_conv3x3(hq, sx, c2.fp8_wq, c2.fp8_sw, bias, sc)
            sc = self.nin_shortcut(x) if self.in_channels != self.out_channels else x
            return _fp8c.fp8_conv3x3(hq, sx, c2.fp8_wq, c2.fp8_sw, c2.bias, sc)

        h = self.norm1.forward_silu(x)
        h = self.conv1(h)

        h = self.norm2.forward_silu(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h

    def forward_parts(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """(shortcut, conv2 output WITHOUT its bias, conv2 bias).

        forward(x) == shortcut + (conv2_nobias + bias). Used by the decoder's fused tail,
        which applies the bias and residual add as it loads the conv2 output (under
        torch.compile the bias add is a separate pointwise op either way).
        On the FP8 path conv2 runs scale-only (no bias, no residual in its epilogue).
        """
        if self._fp8:
            hq, sx = self._fp8_conv2_input(x)
            sc = self.nin_shortcut(x) if self.in_channels != self.out_channels else x
            c2 = self.conv2
            return sc, _fp8c.fp8_conv3x3(hq, sx, c2.fp8_wq, c2.fp8_sw, None, None), c2.bias

        h = self.norm1.forward_silu(x)
        h = self.conv1(h)

        h = self.norm2.forward_silu(h)
        c2 = self.conv2
        h = nn.functional.conv2d(h, c2.weight, None, c2.stride, c2.padding, c2.dilation, c2.groups)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x, h, c2.bias


class Flux3ImageDownsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        pad = (0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


def _subpixel_upsample_weights(weight: Tensor, bias: Tensor) -> tuple[Tensor, Tensor]:
    """Fold nearest-2x upsampling into a 3x3 conv: four 2x2 phase kernels.

    Output pixel (2i+a, 2j+b) of conv3x3(nearest2x(x)) reads low-res rows {i-1, i}
    (a=0, taps [w0, w1+w2]) or {i, i+1} (a=1, taps [w0+w1, w2]); columns likewise.
    Returns W4 [4C_out, C_in, 2, 2] and b4 [4C_out] in phase-major (a, b, c) order.
    Sums are taken in fp64 and rounded once to the weight's dtype.
    """
    w = weight.detach().double()
    r = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]], [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
        dtype=torch.float64,
        device=w.device,
    )  # r[a, p, k]: 2x2 tap p of phase a sums 3x3 taps k
    w4 = torch.einsum("apk,oikl,bql->aboipq", r, w, r)
    w4 = w4.reshape(4 * w.shape[0], w.shape[1], 2, 2).to(weight.dtype)
    b4 = bias.detach().repeat(4).to(bias.dtype)
    return w4.contiguous(), b4.contiguous()


# conv_transpose2d(stride 2, padding 1) tap k of a 4-tap axis reads 2x2 phase a, tap p.
# Output o = 2m + a reads input i with k = o + 1 - 2i: a=0 -> k=1 (i=m, p=1) and
# k=3 (i=m-1, p=0); a=1 -> k=2 (i=m, p=0) and k=0 (i=m+1, p=1).
_TRANSPOSE_TAPS = ((1, 1), (0, 1), (1, 0), (0, 0))


def _subpixel_transpose_weight(w4: Tensor) -> Tensor:
    """Sub-pixel W4 [4C_out, C_in, 2, 2] -> conv_transpose2d weight Wt [C_in, C_out, 4, 4].

    A pure permutation of W4's values (no new rounding): nearest-2x + conv3x3(pad 1)
    == conv_transpose2d(x, Wt, b, stride=2, padding=1), and the phase interleave is
    done by the transposed conv's strided output indexing.
    """
    co, ci = w4.shape[0] // 4, w4.shape[1]
    w6 = w4.detach().reshape(2, 2, co, ci, 2, 2)  # [a, b, o, i, p, q]
    wt = w4.new_empty(ci, co, 4, 4)
    for ky, (a, p) in enumerate(_TRANSPOSE_TAPS):
        for kx, (b, q) in enumerate(_TRANSPOSE_TAPS):
            wt[:, :, ky, kx] = w6[a, b, :, :, p, q].transpose(0, 1)
    return wt.contiguous()


class Flux3ImageUpsample(nn.Module):
    """nearest-2x upsample + 3x3 conv, computed as one sub-pixel 2x2 conv.

    The 2x2 conv with 4C phase-major outputs runs on the low-res input (4/9 of the
    FLOPs of the 3x3 conv on the upsampled input); the phases are then interleaved.
    The combined weights are non-persistent buffers derived from conv.weight/conv.bias
    at load (state_dict keys are unchanged) and follow .to(dtype/device).
    """

    # Per instance: False keeps the sub-pixel conv + interleave on every input (see
    # transpose_path). On the transpose path the decoder may pass the bias-free output
    # to the next block and fold this conv's bias into its 1x1 nin_shortcut
    # (_resnet_block_with_input_bias).
    use_transpose = True
    # Per instance: True runs a bf16 channels_last input off the transpose path as one
    # cuDNN FE graph (flux3_vae::subpixel_up_conv: stride-2 dgrad + fp32 bias epilogue),
    # writing conv + bias straight into the interleaved output (no interleave pass, no
    # separate x + bias pass, no bias fold). fp32 / NCHW inputs keep the path below.
    use_fused_bias_op = False

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.register_buffer(
            "subpixel_weight", torch.zeros(4 * in_channels, in_channels, 2, 2), persistent=False
        )
        self.register_buffer("subpixel_bias", torch.zeros(4 * in_channels), persistent=False)
        self.register_buffer(
            "transpose_weight", torch.zeros(in_channels, in_channels, 4, 4), persistent=False
        )
        self._subpixel_ready = False
        self.register_load_state_dict_post_hook(Flux3ImageUpsample._after_load)

    @staticmethod
    def _after_load(module: "Flux3ImageUpsample", incompatible_keys) -> None:
        module.refresh_subpixel()

    @torch.no_grad()
    def refresh_subpixel(self) -> None:
        """Recompute the 2x2 phase weights from conv.weight/conv.bias."""
        w4, b4 = _subpixel_upsample_weights(self.conv.weight, self.conv.bias)
        self.subpixel_weight = w4.to(self.subpixel_weight.device)
        self.subpixel_bias = b4.to(self.subpixel_bias.device)
        self.transpose_weight = _subpixel_transpose_weight(w4).to(self.transpose_weight.device)
        self._subpixel_ready = True

    def transpose_path(self, x: Tensor) -> bool:
        """Low-precision channels_last input (the compiled bf16 decode): one stride-2 4x4
        conv_transpose2d (cuDNN dgrad writes the interleaved NHWC output directly).
        fp32 and NCHW inputs keep the sub-pixel conv + interleave below."""
        return (
            self.use_transpose
            and x.dim() == 4
            and x.dtype in (torch.bfloat16, torch.float16)
            and x.is_contiguous(memory_format=torch.channels_last)
        )

    def forward_nobias(self, x: Tensor) -> Tensor:
        """transpose_path only: the upsample WITHOUT its bias (conv.bias is added by
        the caller), [n, C, 2h, 2w] channels_last."""
        if not self._subpixel_ready:
            self.refresh_subpixel()
        wt = self.transpose_weight
        if wt.dtype != x.dtype:
            wt = wt.to(x.dtype)
        return nn.functional.conv_transpose2d(x, wt, None, stride=2, padding=1)

    def fused_op_path(self, x: Tensor) -> bool:
        """bf16 channels_last input on a use_fused_bias_op instance: flux3_vae::subpixel_up_conv
        (conv + bias in one cuDNN FE graph). Takes precedence over transpose_path, which
        stays the fallback when the op cannot run."""
        return self.use_fused_bias_op and _spc.can_use(x, self.transpose_weight, self.conv.bias)

    def forward(self, x: Tensor) -> Tensor:
        if not self._subpixel_ready:
            self.refresh_subpixel()
        if self.fused_op_path(x):
            return torch.ops.flux3_vae.subpixel_up_conv(x, self.transpose_weight, self.conv.bias)
        if self.transpose_path(x):
            b = self.conv.bias
            return self.forward_nobias(x) + b.to(x.dtype).view(1, -1, 1, 1)
        w4, b4 = self.subpixel_weight, self.subpixel_bias
        if w4.dtype != x.dtype:
            w4, b4 = w4.to(x.dtype), b4.to(x.dtype)
        n, c, h, w = x.shape
        o = nn.functional.conv2d(x, w4, b4, padding=1)  # [n, 4C, h+1, w+1]
        if o.dim() == 4 and o.is_contiguous(memory_format=torch.channels_last):
            # channels_last (compiled decoder): build the interleaved output in
            # [n, h, 2, w, 2, c] order so it is realized channel-fastest (NHWC), and
            # return it as an NCHW-shaped channels_last view. Same values as below.
            t = o.permute(0, 2, 3, 1).view(n, h + 1, w + 1, 2, 2, c)
            rows = [
                torch.stack([t[:, a : a + h, b : b + w, a, b, :] for b in range(2)], dim=3)
                for a in range(2)
            ]  # each [n, h, w, 2, c]
            y = torch.stack(rows, dim=2)  # [n, h, 2, w, 2, c]
            return y.reshape(n, 2 * h, 2 * w, c).permute(0, 3, 1, 2)
        o = o.view(n, 2, 2, c, h + 1, w + 1)
        rows = [
            torch.stack([o[:, a, b, :, a : a + h, b : b + w] for b in range(2)], dim=-1)
            for a in range(2)
        ]  # each [n, c, h, w, 2]
        y = torch.stack(rows, dim=3)  # [n, c, h, 2, w, 2]
        return y.reshape(n, c, 2 * h, 2 * w)


class Flux3ImageEncoder(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
    ):
        super().__init__()
        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * z_channels, 1)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(Flux3ImageResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Flux3ImageDownsample(block_in)
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = Flux3ImageResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = Flux3ImageAttnBlock(block_in)
        self.mid.block_2 = Flux3ImageResnetBlock(in_channels=block_in, out_channels=block_in)

        self.norm_out = Flux3GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
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


def _can_fold_input_bias(block: nn.Module, upsample: Flux3ImageUpsample, x: Tensor) -> bool:
    """block's 1x1 nin_shortcut can absorb the upsample bias (see _resnet_block_with_input_bias)."""
    nin = getattr(block, "nin_shortcut", None)
    if not isinstance(block, Flux3ImageResnetBlock) or nin is None:
        return False
    b_up = upsample.conv.bias
    return (
        nin.kernel_size == (1, 1)
        and nin.stride == (1, 1)
        and nin.padding == (0, 0)
        and nin.groups == 1
        and nin.bias is not None
        and b_up is not None
        and nin.weight.dtype == x.dtype
        and nin.bias.dtype == x.dtype
        and b_up.dtype == x.dtype
        and upsample.transpose_weight.dtype == x.dtype
    )


def _resnet_block_with_input_bias(block: Flux3ImageResnetBlock, x_nb: Tensor, b_in: Tensor) -> Tensor:
    """block(x_nb + b_in) for a block with a 1x1 nin_shortcut, without materializing
    x_nb + b_in: the norm1 path adds b_in as it loads x_nb (fused under torch.compile),
    and nin(x_nb + b_in) = nin(x_nb) + W_nin @ b_in, so the shortcut conv reads x_nb with
    bias' = b_nin + W_nin[:, :, 0, 0] @ b_in (summed in fp64, kept in fp32 and added to
    the conv output as its bias add is, before the one rounding to the trunk dtype).

    Accuracy caveat: the shortcut conv output is rounded (by cuDNN) BEFORE bias' is
    added. Where W_nin @ b_in largely cancels nin(x_nb), that rounding acts on a much
    larger value than in nin(x_nb + b_in). On the FLUX.3 VAE (bf16 trunk vs the fp32
    decode) the fold into up[1].block[0] kept main's accuracy; the fold into
    up[0].block[0] cost ~0.3 dB PSNR, so that upsample keeps the sub-pixel path."""
    x = x_nb + b_in.view(1, -1, 1, 1)
    if block._fp8:
        # FP8 GN->conv path (prepare_fp8_convs): norm1 still sees x_nb + b_in (fused into
        # gn_swish_fp8); the bias-free shortcut s enters conv2's epilogue as the residual
        # and b_fold joins conv2's bias as one fp32 epilogue bias (no separate add pass).
        hq, sx = block._fp8_conv2_input(x)
        nin = block.nin_shortcut
        w = nin.weight
        b_fold = (nin.bias.double() + torch.mv(w[:, :, 0, 0].double(), b_in.double())).float()
        s = nn.functional.conv2d(x_nb, w, None)
        c2 = block.conv2
        bias = b_fold if c2.bias is None else c2.bias.float() + b_fold
        return _fp8c.fp8_conv3x3(hq, sx, c2.fp8_wq, c2.fp8_sw, bias, s)
    h = block.norm1.forward_silu(x)
    h = block.conv1(h)
    h = block.norm2.forward_silu(h)
    h = block.conv2(h)
    nin = block.nin_shortcut
    w = nin.weight
    b_fold = (nin.bias.double() + torch.mv(w[:, :, 0, 0].double(), b_in.double())).float()
    s = nn.functional.conv2d(x_nb, w, None)
    s = (s.float() + b_fold.view(1, -1, 1, 1)).to(x_nb.dtype)
    return s + h


class Flux3ImageDecoder(nn.Module):
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
    ):
        super().__init__()
        self.post_quant_conv = nn.Conv2d(z_channels, z_channels, 1)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)

        block_in = ch * ch_mult[self.num_resolutions - 1]
        self.z_shape = (
            1,
            z_channels,
            resolution // 2 ** (self.num_resolutions - 1),
            resolution // 2 ** (self.num_resolutions - 1),
        )

        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = Flux3ImageResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = Flux3ImageAttnBlock(block_in)
        self.mid.block_2 = Flux3ImageResnetBlock(in_channels=block_in, out_channels=block_in)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(Flux3ImageResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Flux3ImageUpsample(block_in)
                # The upsample into the full-resolution level keeps the sub-pixel path:
                # its output feeds up[0].block[0].nin_shortcut, and there the bias fold
                # cost ~0.3 dB PSNR vs the fp32 decode (bf16 trunk), while without the
                # fold Inductor materializes x + bias in a separate 0.7 ms pass, which
                # cancels the saved interleave.
                up.upsample.use_transpose = i_level != 1
                # That upsample instead adds its bias in the conv's fp32 epilogue (one
                # cuDNN FE graph writing the interleaved output, bf16 channels_last only).
                # up[2] / up[3] (512 ch at 256^2 / 128^2) too, unless FLUX3_UP_FE_DGRAD_EXTRA=0:
                # there the op replaces aten dgrad + its strided_dgrad helpers (and up[2]'s
                # bias fold into up[1].block[0]); aten stays the fallback (use_transpose).
                up.upsample.use_fused_bias_op = i_level == 1 or (
                    _spc.UP_FE_DGRAD_EXTRA and i_level in _spc.UP_FE_DGRAD_EXTRA_LEVELS
                )
            self.up.insert(0, up)

        self.norm_out = Flux3GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        z = self.post_quant_conv(z)

        upscale_dtype = next(self.up.parameters()).dtype

        h = self.conv_in(z)

        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        h = h.to(upscale_dtype)
        # Fused tail: the last ResnetBlock's residual add, norm_out, swish and conv_out
        # run as one kernel (the sum x + r is never stored).
        last = self.up[0].block[self.num_res_blocks]
        fuse_tail = _tail.FUSE_NORM_PROLOGUE and len(self.up[0].attn) == 0
        up_bias = None  # pending upsample bias, folded into the next block's shortcut
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                if fuse_tail and i_level == 0 and i_block == self.num_res_blocks:
                    break
                if up_bias is not None:
                    h = _resnet_block_with_input_bias(self.up[i_level].block[i_block], h, up_bias)
                    up_bias = None
                else:
                    h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                upsample = self.up[i_level].upsample
                nxt = self.up[i_level - 1].block[0]
                if (
                    not upsample.fused_op_path(h)
                    and upsample.transpose_path(h)
                    and _can_fold_input_bias(nxt, upsample, h)
                ):
                    # The next block's 1x1 shortcut is an extern conv that would force
                    # x + bias to be materialized; pass it the bias-free upsample output
                    # and fold the bias into its own bias instead.
                    h, up_bias = upsample.forward_nobias(h), upsample.conv.bias
                else:
                    h = upsample(h)

        norm, conv = self.norm_out, self.conv_out
        if fuse_tail:
            x, r, r_bias = last.forward_parts(h)
            if r_bias is not None:
                h = x + (r + r_bias.view(1, -1, 1, 1))
            else:
                h = x + r
            if (
                can_use_fused_tail(h, norm, conv)
                and r.dtype == x.dtype
                and r.shape == x.shape
                and r.stride() == x.stride()
                and (r_bias is None or r_bias.dtype == x.dtype)
            ):
                # conv_out(swish(norm_out(x + r + r_bias))): per-channel GroupNorm affine
                # (scale, shift) from norm_out's channel statistics; the kernel forms the
                # sum, the affine and swish in fp32 as it loads the conv input, so the
                # sum and the normalized activation are never stored.
                scale, shift = group_norm_channel_affine(
                    h, norm.num_groups, norm.weight, norm.bias, norm.eps
                )
                return add_norm_swish_conv_out_3x3(
                    x, r, r_bias, scale, shift, conv.weight, conv.bias
                )
        h = norm.forward_silu(h)
        if can_use_conv_out_kernel(h, conv):
            # Same as conv(h) (whose trunk-dtype pre-hook casts h to the weight dtype),
            # computed by one fp32-accumulating Triton kernel with the bias fused in.
            return conv_out_3x3(h.to(conv.weight.dtype), conv.weight, conv.bias)
        h = conv(h)
        return h


class Flux3ImageVAE(nn.Module):
    """FLUX.3 packed image VAE.

    Wraps a standard KL-VAE encoder/decoder with an additional 2x2
    pixel-unshuffle packing step and a fixed (eval-only) BatchNorm2d latent
    normalizer -- this packing+BN step is what distinguishes FLUX.3's image
    latents from a stock SD-VAE's.
    """

    def __init__(self, config: Flux3ImageVAEConfig = Flux3ImageVAEConfig()):
        super().__init__()
        self.config = config
        self.encoder = Flux3ImageEncoder(
            resolution=config.resolution,
            in_channels=config.in_channels,
            ch=config.ch,
            ch_mult=config.ch_mult,
            num_res_blocks=config.num_res_blocks,
            z_channels=config.z_channels,
        )
        self.decoder = Flux3ImageDecoder(
            resolution=config.resolution,
            in_channels=config.in_channels,
            ch=config.ch,
            out_ch=config.out_ch,
            ch_mult=config.ch_mult,
            num_res_blocks=config.num_res_blocks,
            z_channels=config.z_channels,
        )

        self.bn_eps = 1e-4
        self.bn_momentum = 0.1
        self.ps = [2, 2]
        self.bn = nn.BatchNorm2d(
            math.prod(self.ps) * config.z_channels,
            eps=self.bn_eps,
            momentum=self.bn_momentum,
            affine=False,
            track_running_stats=True,
        )

    def normalize(self, z: Tensor) -> Tensor:
        self.bn.eval()
        return self.bn(z)

    def inv_normalize(self, z: Tensor) -> Tensor:
        self.bn.eval()
        s = torch.sqrt(self.bn.running_var.view(1, -1, 1, 1) + self.bn_eps)
        m = self.bn.running_mean.view(1, -1, 1, 1)
        return z * s + m

    # Set by compile_encoder(channels_last=True): encode() hands the compiled encoder a
    # canonical channels_last input (one graph for every input layout) and gets its
    # moments back in the contiguous layout the eager encoder returns.
    _encoder_channels_last = False

    def encode(self, x: Tensor) -> Tensor:
        if self._encoder_channels_last and x.dim() == 4:
            # clone(memory_format=...) always yields the canonical channels_last strides,
            # so the zeros warmup input and a decoded HWC reference share one graph.
            moments = self.encoder(x.clone(memory_format=torch.channels_last))
            moments = moments.contiguous()
        else:
            moments = self.encoder(x)
        mean = torch.chunk(moments, 2, dim=1)[0]

        z = rearrange(
            mean,
            "... c (i pi) (j pj)  -> ... (c pi pj) i j",
            pi=self.ps[0],
            pj=self.ps[1],
        )
        z = self.normalize(z)
        return z

    def compile_decoder(
        self,
        triton_group_norm: bool = False,
        channels_last: bool = True,
        fp8_convs: bool = False,
    ) -> None:
        """torch.compile the decoder in place (state_dict keys are unchanged).

        Inductor lowers each GroupNorm into split reductions and fuses the
        GN apply, swish and residual adds into pointwise kernels around the
        extern cuDNN convolutions. Shapes are static: the served resolution is
        compiled during pipeline warmup. Compilation is lazy (first decode).
        triton_group_norm=True keeps Flux3GroupNorm's Triton kernels in the graph.

        channels_last=True stores the decoder's conv weights channels_last, so cuDNN
        runs its NHWC kernels on NHWC activations end to end without nchwToNhwc /
        nhwcToNchw reformats around every conv, and traces GroupNorm as per-channel
        var_mean + exact group combine (group_norm_channel_stats), which Inductor
        reduces coalesced on NHWC data. aten GroupNorm's (N, G, C/G*H*W) view is a
        strided reduction on channels_last data (bf16 1024^2 decode on SM120: GN
        reductions 5.8 -> 21.5 ms). Inductor's own layout_optimization stays off.

        fp8_convs=True (pipeline_config vae_decode_fp8_convs; output class precision)
        quantizes every ResnetBlock conv1/conv2 to e4m3 (per-Cout scale) and runs them as
        cuDNN FP8 convs fed by gn_swish_fp8 (fp8_conv.prepare_fp8_convs). It needs a bf16
        trunk and channels_last; otherwise it is a logged no-op.
        """
        for m in self.decoder.modules():
            if isinstance(m, Flux3GroupNorm):
                m.triton_under_compile = triton_group_norm
                m.channel_stats_under_compile = channels_last
            elif isinstance(m, Flux3ImageUpsample) and not m._subpixel_ready:
                m.refresh_subpixel()  # weights were not loaded through load_state_dict
        if channels_last:
            self.decoder.to(memory_format=torch.channels_last)
        n_fp8 = _fp8c.prepare_fp8_convs(self.decoder) if fp8_convs else 0
        for m in self.decoder.modules():
            if isinstance(m, Flux3ImageAttnBlock):
                m._fp8_pv = n_fp8 > 0
        dtype = next(self.decoder.parameters()).dtype
        options = {"layout_optimization": False}
        if dtype != torch.float32:
            # Keep low-precision intermediate rounding identical to eager casts.
            options["emulate_precision_casts"] = True
        self.decoder.compile(dynamic=False, options=options)

    def compile_encoder(self, channels_last: bool = True, fp32_channel_stats: bool = True) -> None:
        """torch.compile the encoder in place (state_dict keys are unchanged).

        Mirrors compile_decoder: channels_last=True stores the encoder's conv weights
        channels_last and encode() feeds it a channels_last input, so cuDNN runs NHWC
        kernels end to end (no nchwToNhwc reformat per conv), GroupNorm is traced as
        group_norm_channel_stats, and Inductor fuses conv bias adds, GN apply + swish,
        residual adds, the downsample pad and the mid-block attention's scale + softmax
        + bf16 cast into its own kernels. The math stays fp32 (TF32 cuDNN convs, as
        eager). fp32_channel_stats=True accumulates the GN statistics in fp32 (aten
        GroupNorm's accumulation) instead of fp64, which SM120 runs at ~1/64 rate.
        Shapes are static: the served resolution compiles at warmup (first encode).
        """
        for m in self.encoder.modules():
            if isinstance(m, Flux3GroupNorm):
                m.triton_under_compile = False
                m.channel_stats_under_compile = channels_last
                m.channel_stats_acc_dtype = (
                    torch.float32 if channels_last and fp32_channel_stats else None
                )
        if channels_last:
            self.encoder.to(memory_format=torch.channels_last)
        self._encoder_channels_last = channels_last
        options = {"layout_optimization": False}
        if next(self.encoder.parameters()).dtype != torch.float32:
            options["emulate_precision_casts"] = True
        self.encoder.compile(dynamic=False, options=options)

    def decode(self, z: Tensor) -> Tensor:
        z = self.inv_normalize(z)
        z = rearrange(
            z,
            "... (c pi pj) i j -> ... c (i pi) (j pj)",
            pi=self.ps[0],
            pj=self.ps[1],
        )
        return self.decoder(z)
