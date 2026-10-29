# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sub-pixel upsample (nearest-2x + 3x3 conv) with its bias fused, as ONE cuDNN FE graph.

Math (same as Flux3ImageUpsample's sub-pixel conv + interleave; epilogue reordered):
    y = bf16( conv_transpose2d(x, Wt, stride 2, padding 1) [fp32 acc] + bias [fp32] )
where Wt [C_in, C_out, 4, 4] is Flux3ImageUpsample.transpose_weight, a pure permutation of
the sub-pixel phase weights W4 (no new rounding). The stride-2 transposed conv (cuDNN
conv_dgrad) writes the phase-interleaved channels_last y directly, and the bias is added in
the fp32 epilogue before the one bf16 rounding. So neither main's [n, 4C, h+1, w+1]
sub-pixel output and interleave pass, nor a separate x + bias pass (what aten
conv_transpose2d + bias costs under Inductor), is ever stored. Main rounds the conv to bf16
and then adds the bias in the interleave (two roundings).

Why not four per-phase 2x2 fprops writing strided views of y: on SM120 (cuDNN 9.26, FE
1.27) the fast engines for that graph (eng7 / eng1 / eng52, ~0.40 ms per phase) accept the
strided output descriptor but write a packed tile at the view's base pointer (silently
wrong); only the generic engines (eng10/22/23/32) honor the strides, at 6-12 ms per phase.

Exposed to torch.compile as the opaque custom op ``torch.ops.flux3_vae.subpixel_up_conv``.
One FE graph per (N, C_in, C_out, H, W); it is built at the first (warmup) call. The plan is
fixed by exact name per shape (PLAN_NAME, measured), else the heuristic's first plan;
FLUX3_SUBPIXEL_AUTOTUNE=1 times every plan instead (diagnostics only).
"""

import os
import threading
import time
from typing import Any, Dict, Optional

import torch
from torch import Tensor

from tensorrt_llm.logger import logger

from . import fp8_conv as _fp8c

cudnn = _fp8c.cudnn
_CL = torch.channels_last

# (C_in, C_out, H) -> exact plan name of the fixed plan (measured on SM120, idea 227 STEP 0).
PLAN_NAME: Dict[tuple, str] = {
    (256, 256, 512): "eng1_k24=1",  # up[1]: 1.556 ms (main fprop + interleave 2.269 ms)
}

_lock = threading.Lock()
_graphs: Dict[tuple, "_UpGraph"] = {}
_logged = set()


def available() -> bool:
    return cudnn is not None


def _log_once(tag: str, msg: str) -> None:
    if tag not in _logged:
        _logged.add(tag)
        logger.info(msg)


class _UpGraph:
    """conv_dgrad(x, Wt; stride 2, pad 1) -> + bias (fp32) -> bf16 NHWC y."""

    def __init__(self, key: tuple, device: torch.device):
        n, cin, cout, h, w = key
        self.key = key
        self.handle = _fp8c._get_handle(device)
        bf16, f32 = cudnn.data_type.BFLOAT16, cudnn.data_type.FLOAT
        g = cudnn.pygraph(
            io_data_type=bf16,
            intermediate_data_type=f32,
            compute_data_type=f32,
            handle=self.handle,
        )
        t = {}
        # dgrad: "loss" = x [N, K=C_in, H, W]; filter [K=C_in, C=C_out, 4, 4] (KRSC storage)
        t["X"] = g.tensor(name="X", dim=[n, cin, h, w], stride=_fp8c._nhwc(cin, h, w), data_type=bf16)
        t["W"] = g.tensor(
            name="W", dim=[cin, cout, 4, 4], stride=_fp8c._nhwc(cout, 4, 4), data_type=bf16
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
        y.set_dim([n, cout, 2 * h, 2 * w]).set_stride(_fp8c._nhwc(cout, 2 * h, 2 * w))
        y.set_data_type(f32)
        t["B"] = g.tensor(name="bias", dim=[1, cout, 1, 1], stride=[cout, 1, cout, cout], data_type=bf16)
        y = g.add(a=y, b=t["B"], compute_data_type=f32, name="bias_add")
        y.set_output(True).set_data_type(bf16)
        t["Y"] = y
        t0 = time.perf_counter()
        g.validate()
        g.build_operation_graph()
        g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        g.check_support()
        g.build_plans(cudnn.build_plan_policy.ALL)
        self.build_s = time.perf_counter() - t0
        self.g, self.t = g, t
        self.names, self.ws = [], []
        for i in range(g.get_execution_plan_count()):
            try:
                self.names.append(g.get_plan_name_at_index(i))
            except Exception:  # noqa: BLE001
                self.names.append("?")
            try:
                self.ws.append(int(g.get_workspace_size_plan_at_index(i)))
            except Exception:  # noqa: BLE001
                self.ws.append(int(g.get_workspace_size()))
        self.plan = self._fixed_plan(cin, cout, h)
        self.select = "fixed"
        self.plan_times_ms: Optional[Dict[str, float]] = None

    def _fixed_plan(self, cin: int, cout: int, h: int) -> int:
        want = os.environ.get("FLUX3_SUBPIXEL_PLAN") or PLAN_NAME.get((cin, cout, h))
        if want is not None and want in self.names:
            return self.names.index(want)
        return 0

    def run(self, vp: dict, plan: Optional[int] = None) -> None:
        i = self.plan if plan is None else plan
        cudnn.set_stream(handle=self.handle, stream=torch.cuda.current_stream().cuda_stream)
        ws = torch.empty(max(self.ws[i], 1), dtype=torch.uint8, device="cuda")
        self.g.execute_plan_at_index(vp, ws, i, handle=self.handle)

    def autotune(self, vp: dict, warm: int = 2, iters: int = 5) -> Dict[str, float]:
        times = {}
        for i, nm in enumerate(self.names):
            if nm.startswith("eng0") or nm.startswith("eng8"):
                continue  # reference engines, 10x slower
            try:
                for _ in range(warm):
                    self.run(vp, i)
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                for _ in range(iters):
                    self.run(vp, i)
                e.record()
                torch.cuda.synchronize()
                times[f"{i}:{nm}"] = s.elapsed_time(e) / iters
            except Exception as ex:  # noqa: BLE001
                times[f"{i}:{nm}"] = float("nan")
                logger.warning(f"subpixel_up_conv plan {i} {nm} failed: {ex!r}"[:300])
        self.plan_times_ms = times
        return times

    def describe(self) -> dict:
        return {
            "key": list(self.key),
            "plan": self.plan,
            "plan_name": self.names[self.plan] if self.names else "?",
            "select": self.select,
            "n_plans": len(self.names),
            "build_s": round(self.build_s, 3),
            "workspace": self.ws[self.plan] if self.ws else 0,
        }


def get_graph(key: tuple, device: torch.device) -> _UpGraph:
    gr = _graphs.get(key)
    if gr is None:
        gr = _UpGraph(key, device)
        with _lock:
            _graphs[key] = gr
        _log_once(f"plan{key}", f"flux3_vae::subpixel_up_conv built {gr.describe()}")
    return gr


def graph_table() -> list:
    out = []
    for gr in _graphs.values():
        d = gr.describe()
        if gr.plan_times_ms is not None:
            d["plan_times_ms"] = gr.plan_times_ms
        out.append(d)
    return out


def can_use(x: Tensor, wt: Tensor, bias: Optional[Tensor]) -> bool:
    """bf16 channels_last CUDA input, [C_in, C_out, 4, 4] transposed weight, a bias."""
    return (
        cudnn is not None
        and bias is not None
        and x.is_cuda
        and x.dim() == 4
        and x.dtype == torch.bfloat16
        and x.is_contiguous(memory_format=_CL)
        and wt.dim() == 4
        and wt.shape[0] == x.shape[1]
        and tuple(wt.shape[2:]) == (4, 4)
        and bias.numel() == wt.shape[1]
        and x.shape[1] % 8 == 0
        and wt.shape[1] % 8 == 0
    )


@torch.library.custom_op("flux3_vae::subpixel_up_conv", mutates_args=())
def subpixel_up_conv(x: Tensor, wt: Tensor, bias: Tensor) -> Tensor:
    """nearest-2x + 3x3 conv (+bias) of bf16 channels_last x [N, C_in, H, W] from the
    transposed sub-pixel weight wt [C_in, C_out, 4, 4]; bias [C_out] is added in fp32 before
    the one rounding -> bf16 [N, C_out, 2H, 2W] channels_last."""
    x = x.contiguous(memory_format=_CL)
    wt = wt.to(torch.bfloat16).contiguous(memory_format=_CL)
    bias = bias.to(torch.bfloat16).contiguous()
    n, cin, h, w = x.shape
    cout = wt.shape[1]
    gr = get_graph((n, cin, cout, h, w), x.device)
    y = torch.empty((n, cout, 2 * h, 2 * w), dtype=torch.bfloat16, device=x.device, memory_format=_CL)
    vp = {gr.t["X"]: x, gr.t["W"]: wt, gr.t["B"]: bias.view(1, cout, 1, 1), gr.t["Y"]: y}
    if os.environ.get("FLUX3_SUBPIXEL_AUTOTUNE") == "1" and gr.select == "fixed":
        times = gr.autotune(vp)
        best = min((k for k in times if times[k] == times[k]), key=lambda k: times[k], default=None)
        if best is not None:
            gr.plan = int(best.split(":")[0])
            gr.select = "autotune"
        _log_once(f"tune{gr.key}", f"flux3_vae::subpixel_up_conv autotuned {gr.describe()} {times}")
    gr.run(vp)
    return y


@subpixel_up_conv.register_fake
def _(x, wt, bias):
    n, _, h, w = x.shape
    cout = wt.shape[1]
    return torch.empty_strided(
        (n, cout, 2 * h, 2 * w),
        (4 * h * w * cout, 1, 2 * w * cout, cout),
        dtype=torch.bfloat16,
        device=x.device,
    )
