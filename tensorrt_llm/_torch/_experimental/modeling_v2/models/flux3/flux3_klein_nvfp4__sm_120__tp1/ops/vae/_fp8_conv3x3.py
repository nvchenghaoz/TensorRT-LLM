# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E4M3 3x3 conv with a fused scale / bias / residual epilogue as one cuDNN FE graph.

    y = bf16( conv(xq, wq) [fp32 acc] * s_x * s_w[o] (+ bias[o]) (+ residual) )

One graph per key (N, Cin, Cout, H, W, has_bias, has_res): conv_fprop (e4m3 NHWC x e4m3
KRSC, fp32 accumulate) -> mul s_x -> mul s_w (-> add bias) (-> add residual) -> bf16 NHWC.
``has_bias`` is 0 (none), 1 (bf16 bias operand) or 2 (fp32 bias operand).

Registered as ``torch.ops.trtllm.flux3_klein_fp8_conv3x3`` so the compiled decoder keeps
it as one opaque call.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from . import _cudnn

BIAS_NONE, BIAS_BF16, BIAS_FP32 = 0, 1, 2
_CL = torch.channels_last

# (Cin, Cout, H) -> knob k24 of the engine-7 plan each key runs (SM120, cuDNN 9.26 / FE
# 1.27; interleaved per-call medians against the heuristic's first engine-7 plan). Every
# graph variant of a shape (bias / residual / fp32 bias / scale-only) runs its shape's pin.
PLAN_K24: Dict[Tuple[int, int, int], int] = {
    # 1024x1024 output: the measured-fastest k24 per shape.
    (512, 512, 128): 21,
    (512, 512, 256): 0,  # also 2048x2048's mid block and up[3]
    (512, 256, 512): 14,
    (256, 256, 512): 13,
    (256, 128, 1024): 13,
    (128, 128, 1024): 13,
    # 2048x2048 output: each pin's output is bitwise the first engine-7 plan's (same
    # accumulation order, different tiling), measured on real and random operands.
    (512, 512, 512): 14,  # bias 1.725 -> 1.702 ms, bias+res 2.267 -> 1.809 ms
    (512, 256, 1024): 14,  # bias 3.532 -> 3.486 ms
    (256, 256, 1024): 14,  # bias 2.301 -> 1.844 ms, bias+res 2.414 -> 1.969 ms
    (256, 128, 2048): 13,  # bias 4.701 -> 3.746 ms
    (128, 128, 2048): 13,  # bias 2.378 -> 2.071 ms, bias+res 2.758 -> 2.322 ms
}

_graphs: Dict[tuple, _cudnn.FixedPlanGraph] = {}


def _is_eng7(name: str) -> bool:
    return _cudnn.plan_tokens(name)[0] == "eng7"


def _k24(name: str) -> Optional[int]:
    try:
        return int(_cudnn.plan_tokens(name)[1]["k24"])
    except (KeyError, ValueError):
        return None


def _build(key: tuple, device: torch.device) -> _cudnn.FixedPlanGraph:
    cudnn = _cudnn.require_cudnn()
    n, cin, cout, h, w, has_bias, has_res = key
    bf16, f32, e4m3 = cudnn.data_type.BFLOAT16, cudnn.data_type.FLOAT, cudnn.data_type.FP8_E4M3
    g = cudnn.pygraph(
        io_data_type=bf16,
        intermediate_data_type=f32,
        compute_data_type=f32,
        handle=_cudnn.handle(device),
    )
    t = {}
    t["X"] = g.tensor(name="X", dim=[n, cin, h, w], stride=_cudnn.nhwc(cin, h, w), data_type=e4m3)
    t["W"] = g.tensor(
        name="W", dim=[cout, cin, 3, 3], stride=_cudnn.nhwc(cin, 3, 3), data_type=e4m3
    )
    t["SX"] = g.tensor(name="SX", dim=[1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=f32)
    t["SW"] = g.tensor(name="SW", dim=[1, cout, 1, 1], stride=[cout, 1, cout, cout], data_type=f32)
    y = g.conv_fprop(
        image=t["X"],
        weight=t["W"],
        padding=[1, 1],
        stride=[1, 1],
        dilation=[1, 1],
        compute_data_type=f32,
        name="conv",
    )
    y.set_data_type(f32)
    y = g.mul(a=y, b=t["SX"], compute_data_type=f32, name="sx")
    y = g.mul(a=y, b=t["SW"], compute_data_type=f32, name="sw")
    if has_bias:
        t["B"] = g.tensor(
            name="bias",
            dim=[1, cout, 1, 1],
            stride=[cout, 1, cout, cout],
            data_type=f32 if has_bias == BIAS_FP32 else bf16,
        )
        y = g.add(a=y, b=t["B"], compute_data_type=f32, name="bias_add")
    if has_res:
        t["R"] = g.tensor(
            name="R", dim=[n, cout, h, w], stride=_cudnn.nhwc(cout, h, w), data_type=bf16
        )
        y = g.add(a=y, b=t["R"], compute_data_type=f32, name="res_add")
    y.set_output(True).set_data_type(bf16)
    t["Y"] = y
    names = _cudnn.build_plans(g)

    want = PLAN_K24.get((cin, cout, h))
    plan = _cudnn.select_plan(
        "flux3_klein_fp8_conv3x3",
        key,
        names,
        pinned=None if want is None else f"eng7_k24={want}",
        match=lambda nm: _is_eng7(nm) and _k24(nm) == want,
        unpinned=lambda nms: next((i for i, nm in enumerate(nms) if _is_eng7(nm)), 0),
    )
    return _cudnn.FixedPlanGraph(g, t, device, plan, names[plan])


def _graph(key: tuple, device: torch.device) -> _cudnn.FixedPlanGraph:
    gr = _graphs.get(key)
    if gr is None:
        gr = _graphs.setdefault(key, _build(key, device))
    return gr


@torch.library.custom_op("trtllm::flux3_klein_fp8_conv3x3", mutates_args=())
def fp8_conv3x3_op(
    xq: Tensor,
    s_x: Tensor,
    wq: Tensor,
    s_w: Tensor,
    bias: Optional[Tensor] = None,
    residual: Optional[Tensor] = None,
) -> Tensor:
    """3x3 / stride 1 / pad 1 conv of e4m3 xq [N, Cin, H, W] with e4m3 KRSC weights wq
    [Cout, 3, 3, Cin]; epilogue * s_x[1] * s_w[Cout] (+ bias[Cout]) (+ residual) -> bf16
    [N, Cout, H, W] channels_last. An fp32 bias is added as fp32; any other bias dtype is
    rounded to bf16 first."""
    xq = xq.contiguous(memory_format=_CL)
    wq = wq.contiguous()
    s_x = s_x.float().contiguous()
    s_w = s_w.float().contiguous()
    n, cin, h, w = xq.shape
    cout = wq.shape[0]
    has_bias = BIAS_NONE
    if bias is not None:
        if bias.dtype == torch.float32:
            bias, has_bias = bias.contiguous(), BIAS_FP32
        else:
            bias, has_bias = bias.to(torch.bfloat16).contiguous(), BIAS_BF16
    if residual is not None:
        residual = residual.to(torch.bfloat16).contiguous(memory_format=_CL)
    key = (n, cin, cout, h, w, has_bias, residual is not None)
    gr = _graph(key, xq.device)
    out = torch.empty((n, cout, h, w), dtype=torch.bfloat16, device=xq.device, memory_format=_CL)
    vp = {
        gr.t["X"]: xq,
        gr.t["W"]: wq.permute(0, 3, 1, 2),  # KRSC storage viewed as [K, C, R, S]
        gr.t["SX"]: s_x.view(1, 1, 1, 1),
        gr.t["SW"]: s_w.view(1, cout, 1, 1),
        gr.t["Y"]: out,
    }
    if has_bias:
        vp[gr.t["B"]] = bias.view(1, cout, 1, 1)
    if residual is not None:
        vp[gr.t["R"]] = residual
    gr.run(vp)
    return out


@fp8_conv3x3_op.register_fake
def _(xq, s_x, wq, s_w, bias=None, residual=None):
    n, _, h, w = xq.shape
    return torch.empty(
        (n, wq.shape[0], h, w), dtype=torch.bfloat16, device=xq.device, memory_format=_CL
    )
