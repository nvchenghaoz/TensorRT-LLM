# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Nearest-2x upsample + 3x3 conv + bias as one cuDNN FE graph (stride-2 dgrad).

    y = bf16( conv_transpose2d(x, Wt, stride 2, padding 1) [fp32 acc] + bias [fp32] )

Wt [C_in, C_out, 4, 4] is a permutation of the four 2x2 sub-pixel phase kernels of the
3x3 conv (``subpixel_transpose_weight``); cuDNN's conv_dgrad writes the phase-interleaved
NHWC output directly and the bias is added in the fp32 epilogue before the one rounding.

Four per-phase 2x2 fprops into strided views of y are not an alternative on SM120 (cuDNN
9.26 / FE 1.27): the fast engines accept the strided output descriptor but write a packed
tile at the view's base pointer; only the generic engines honor it, at 6-12 ms per phase.

Registered as ``torch.ops.trtllm.flux3_klein_subpixel_up_conv``.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor

from . import _cudnn

_CL = torch.channels_last

# (C_in, C_out, H) -> exact name of the plan each key runs (SM120, cuDNN 9.26 / FE 1.27).
PLAN_NAME: Dict[tuple, str] = {
    # 1024x1024 output, measured fastest (ms per call, vs the unfused upsample + conv):
    (512, 512, 128): "eng1_k24=4",  # up[3] 0.395 (0.464)
    (512, 512, 256): "eng1_k24=1",  # up[2] 1.451 (1.598); also 2048x2048's up[3]
    (256, 256, 512): "eng1_k24=1",  # up[1] 1.556 (2.269)
    # 2048x2048 output: bitwise the heuristic's first plan (eng1_k24=10) on real and
    # random operands, and faster:
    (512, 512, 512): "eng1_k24=1",  # up[2] 7.632 -> 6.180 ms
    (256, 256, 1024): "eng1_k24=1",  # up[1] 8.185 -> 6.636 ms
}

_graphs: Dict[tuple, _cudnn.FixedPlanGraph] = {}


def _build(key: tuple, device: torch.device) -> _cudnn.FixedPlanGraph:
    cudnn = _cudnn.require_cudnn()
    n, cin, cout, h, w = key
    bf16, f32 = cudnn.data_type.BFLOAT16, cudnn.data_type.FLOAT
    g = cudnn.pygraph(
        io_data_type=bf16,
        intermediate_data_type=f32,
        compute_data_type=f32,
        handle=_cudnn.handle(device),
    )
    t = {}
    # dgrad: "loss" = x [N, K=C_in, H, W]; filter [K=C_in, C=C_out, 4, 4] (KRSC storage)
    t["X"] = g.tensor(name="X", dim=[n, cin, h, w], stride=_cudnn.nhwc(cin, h, w), data_type=bf16)
    t["W"] = g.tensor(
        name="W", dim=[cin, cout, 4, 4], stride=_cudnn.nhwc(cout, 4, 4), data_type=bf16
    )
    y = g.conv_dgrad(
        loss=t["X"],
        filter=t["W"],
        padding=[1, 1],
        stride=[2, 2],
        dilation=[1, 1],
        compute_data_type=f32,
        name="dgrad",
    )
    y.set_dim([n, cout, 2 * h, 2 * w]).set_stride(_cudnn.nhwc(cout, 2 * h, 2 * w))
    y.set_data_type(f32)
    t["B"] = g.tensor(
        name="bias", dim=[1, cout, 1, 1], stride=[cout, 1, cout, cout], data_type=bf16
    )
    y = g.add(a=y, b=t["B"], compute_data_type=f32, name="bias_add")
    y.set_output(True).set_data_type(bf16)
    t["Y"] = y
    names = _cudnn.build_plans(g)

    want = PLAN_NAME.get((cin, cout, h))
    plan = _cudnn.select_plan(
        "flux3_klein_subpixel_up_conv",
        key,
        names,
        pinned=want,
        match=lambda nm: nm == want,
        unpinned=lambda nms: 0,
    )
    return _cudnn.FixedPlanGraph(g, t, device, plan, names[plan])


def _graph(key: tuple, device: torch.device) -> _cudnn.FixedPlanGraph:
    gr = _graphs.get(key)
    if gr is None:
        gr = _graphs.setdefault(key, _build(key, device))
    return gr


@torch.library.custom_op("trtllm::flux3_klein_subpixel_up_conv", mutates_args=())
def subpixel_up_conv_op(x: Tensor, wt: Tensor, bias: Tensor) -> Tensor:
    """nearest-2x + 3x3 conv (+ bias) of bf16 channels_last x [N, C_in, H, W] from the
    transposed sub-pixel weight wt [C_in, C_out, 4, 4]; bias [C_out] is added in fp32
    before the one rounding -> bf16 [N, C_out, 2H, 2W] channels_last."""
    x = x.contiguous(memory_format=_CL)
    wt = wt.to(torch.bfloat16).contiguous(memory_format=_CL)
    bias = bias.to(torch.bfloat16).contiguous()
    n, cin, h, w = x.shape
    cout = wt.shape[1]
    gr = _graph((n, cin, cout, h, w), x.device)
    y = torch.empty(
        (n, cout, 2 * h, 2 * w), dtype=torch.bfloat16, device=x.device, memory_format=_CL
    )
    gr.run({gr.t["X"]: x, gr.t["W"]: wt, gr.t["B"]: bias.view(1, cout, 1, 1), gr.t["Y"]: y})
    return y


@subpixel_up_conv_op.register_fake
def _(x, wt, bias):
    n, _, h, w = x.shape
    cout = wt.shape[1]
    return torch.empty_strided(
        (n, cout, 2 * h, 2 * w),
        (4 * h * w * cout, 1, 2 * w * cout, cout),
        dtype=torch.bfloat16,
        device=x.device,
    )


# conv_transpose2d(stride 2, padding 1) tap k of a 4-tap axis reads 2x2 phase a, tap p.
# Output o = 2m + a reads input i with k = o + 1 - 2i: a=0 -> k=1 (i=m, p=1) and
# k=3 (i=m-1, p=0); a=1 -> k=2 (i=m, p=0) and k=0 (i=m+1, p=1).
_TRANSPOSE_TAPS = ((1, 1), (0, 1), (1, 0), (0, 0))


def subpixel_phase_weights(weight: Tensor) -> Tensor:
    """Fold nearest-2x upsampling into a 3x3 conv: four 2x2 phase kernels.

    Output pixel (2i+a, 2j+b) of conv3x3(nearest2x(x)) reads low-res rows {i-1, i}
    (a=0, taps [w0, w1+w2]) or {i, i+1} (a=1, taps [w0+w1, w2]); columns likewise.
    Returns W4 [4C_out, C_in, 2, 2] in phase-major (a, b, c) order; the tap sums are
    taken in fp64 and rounded once to the weight's dtype.
    """
    w = weight.detach().double()
    r = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]], [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
        dtype=torch.float64,
        device=w.device,
    )  # r[a, p, k]: 2x2 tap p of phase a sums 3x3 taps k
    w4 = torch.einsum("apk,oikl,bql->aboipq", r, w, r)
    return w4.reshape(4 * w.shape[0], w.shape[1], 2, 2).to(weight.dtype).contiguous()


def subpixel_transpose_weight(weight: Tensor) -> Tensor:
    """3x3 conv weight [C_out, C_in, 3, 3] -> the dgrad weight Wt [C_in, C_out, 4, 4].

    Wt is a permutation of the phase kernels W4 (no rounding beyond W4's one):
    nearest-2x + conv3x3(pad 1) == conv_transpose2d(x, Wt, b, stride=2, padding=1).
    """
    w4 = subpixel_phase_weights(weight)
    co, ci = w4.shape[0] // 4, w4.shape[1]
    w6 = w4.reshape(2, 2, co, ci, 2, 2)  # [a, b, o, i, p, q]
    wt = w4.new_empty(ci, co, 4, 4)
    for ky, (a, p) in enumerate(_TRANSPOSE_TAPS):
        for kx, (b, q) in enumerate(_TRANSPOSE_TAPS):
            wt[:, :, ky, kx] = w6[a, b, :, :, p, q].transpose(0, 1)
    return wt.contiguous()
