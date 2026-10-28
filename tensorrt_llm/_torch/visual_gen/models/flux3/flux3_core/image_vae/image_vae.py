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
    x: Tensor, num_groups: int, weight: Tensor, bias: Tensor, eps: float
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
    """
    n, c = x.shape[:2]
    g = num_groups
    xf = x.float()
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

    def _fused(self, x: Tensor) -> bool:
        if torch.compiler.is_compiling() and not self.triton_under_compile:
            return False
        return can_use_fused_gn(x, self)

    def _unfused(self, x: Tensor) -> Tensor:
        if self.channel_stats_under_compile and torch.compiler.is_compiling():
            return group_norm_channel_stats(x, self.num_groups, self.weight, self.bias, self.eps)
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
            return self._attention_tc(q, k, v)
        q = rearrange(q, "b c h w -> b 1 (h w) c")
        k = rearrange(k, "b c h w -> b 1 (h w) c")
        v = rearrange(v, "b c h w -> b 1 (h w) c")
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    @staticmethod
    def _attention_tc(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Single-head attention over h*w tokens on bf16 tensor cores.

        SDPA on the rearranged (non-contiguous, d=512) q/k/v takes the MATH path,
        an fp32 SIMT sgemm plus a 16384^2 fp32 softmax (8.6 ms at 1024^2). Here:
        S = q^T k as a bf16 GEMM with fp32 output (scores are never rounded to bf16),
        fp32 softmax with the d^-0.5 scale, P rounded to bf16, O = v P^T as a bf16
        GEMM with fp32 accumulation, written directly in the b c (h w) layout.
        q/k/v are rounded to bf16 (a no-op on the bf16 trunk); the output keeps the
        input dtype.
        """
        b, c, h, w = q.shape
        dt = q.dtype
        q = q.reshape(b, c, h * w).to(torch.bfloat16)
        k = k.reshape(b, c, h * w).to(torch.bfloat16)
        v = v.reshape(b, c, h * w).to(torch.bfloat16)
        s = torch.bmm(q.transpose(1, 2), k, out_dtype=torch.float32)
        p = torch.softmax(s * (c**-0.5), dim=-1).to(torch.bfloat16)
        o = torch.bmm(v, p.transpose(1, 2))
        return o.reshape(b, c, h, w).to(dt)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class Flux3ImageResnetBlock(nn.Module):
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

    def forward(self, x: Tensor) -> Tensor:
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
        """
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


class Flux3ImageUpsample(nn.Module):
    """nearest-2x upsample + 3x3 conv, computed as one sub-pixel 2x2 conv.

    The 2x2 conv with 4C phase-major outputs runs on the low-res input (4/9 of the
    FLOPs of the 3x3 conv on the upsampled input); the phases are then interleaved.
    The combined weights are non-persistent buffers derived from conv.weight/conv.bias
    at load (state_dict keys are unchanged) and follow .to(dtype/device).
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.register_buffer(
            "subpixel_weight", torch.zeros(4 * in_channels, in_channels, 2, 2), persistent=False
        )
        self.register_buffer("subpixel_bias", torch.zeros(4 * in_channels), persistent=False)
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
        self._subpixel_ready = True

    def forward(self, x: Tensor) -> Tensor:
        if not self._subpixel_ready:
            self.refresh_subpixel()
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
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                if fuse_tail and i_level == 0 and i_block == self.num_res_blocks:
                    break
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

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

    def encode(self, x: Tensor) -> Tensor:
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

    def compile_decoder(self, triton_group_norm: bool = False, channels_last: bool = True) -> None:
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
        """
        for m in self.decoder.modules():
            if isinstance(m, Flux3GroupNorm):
                m.triton_under_compile = triton_group_norm
                m.channel_stats_under_compile = channels_last
            elif isinstance(m, Flux3ImageUpsample) and not m._subpixel_ready:
                m.refresh_subpixel()  # weights were not loaded through load_state_dict
        if channels_last:
            self.decoder.to(memory_format=torch.channels_last)
        dtype = next(self.decoder.parameters()).dtype
        options = {"layout_optimization": False}
        if dtype != torch.float32:
            # Keep low-precision intermediate rounding identical to eager casts.
            options["emulate_precision_casts"] = True
        self.decoder.compile(dynamic=False, options=options)

    def decode(self, z: Tensor) -> Tensor:
        z = self.inv_normalize(z)
        z = rearrange(
            z,
            "... (c pi pj) i j -> ... c (i pi) (j pj)",
            pi=self.ps[0],
            pj=self.ps[1],
        )
        return self.decoder(z)
