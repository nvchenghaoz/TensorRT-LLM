# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP8 (E4M3) 3x3 conv for the FLUX.3 image VAE decoder's GroupNorm->conv sites.

Math (output class: precision):
    y = bf16( conv(xq, wq) [fp32 acc] * s_x * s_w[o] (+ bias[o]) (+ residual) )
with xq = sat_e4m3(h / s_x) (per-tensor activation scale, produced by the GN+swish
prologue, not in this file) and wq = sat_e4m3(W / s_w) (per-output-channel weight
scale, quantized once at load by prepare_fp8_convs).

The conv runs as one cuDNN frontend graph per key (N, Cin, Cout, H, W, has_bias,
has_res): conv_fprop (e4m3 NHWC x e4m3 KRSC, fp32 accumulate) -> mul s_x -> mul s_w
(-> add bias) (-> add residual) -> bf16 NHWC.  Graphs are built and cached at the
first (warmup) call; the plan is FIXED per key (eng7, knob k24 measured best in fact
#1431 / job 1519), with FLUX3_FP8_CONV_AUTOTUNE=1 timing every plan instead.

Exposed to torch.compile as the opaque custom op ``torch.ops.flux3_vae.fp8_conv3x3``.
Nothing in the tree calls this yet (wiring is a follow-up).
"""

import os
import threading
import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from tensorrt_llm.logger import logger

try:
    import cudnn  # cuDNN frontend python bindings
except Exception:  # noqa: BLE001
    cudnn = None

FP8 = torch.float8_e4m3fn
QMAX = 448.0
_CL = torch.channels_last

# (Cin, Cout, H) -> eng7 knob k24 value of the fixed plan (fact #1431, job 1519, SM120).
# sbr / scale-only keys reuse their shape's sb choice.
PLAN_K24: Dict[Tuple[int, int, int], int] = {
    (512, 512, 128): 21,  # S1
    (512, 512, 256): 0,  # S2
    (512, 256, 512): 14,  # S3
    (256, 256, 512): 13,  # S4
    (256, 128, 1024): 13,  # S5
    (128, 128, 1024): 13,  # S6
}

_lock = threading.Lock()
_handles: Dict[int, Any] = {}
_graphs: Dict[tuple, "_Fp8ConvGraph"] = {}
_logged = set()


def _log_once(tag: str, msg: str) -> None:
    if tag not in _logged:
        _logged.add(tag)
        logger.info(msg)


def available() -> bool:
    return cudnn is not None


def _get_handle(device: torch.device) -> Any:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    with _lock:
        h = _handles.get(idx)
        if h is None:
            with torch.cuda.device(idx):
                h = cudnn.create_handle()
            _handles[idx] = h
        return h


def _nhwc(c: int, h: int, w: int) -> list:
    return [c * h * w, 1, w * c, c]


def _plan_k24(name: str) -> Optional[int]:
    for tok in name.split("_"):
        if tok.startswith("k24="):
            try:
                return int(tok[4:])
            except ValueError:
                return None
    return None


class _Fp8ConvGraph:
    """One cuDNN FE graph + chosen plan for a key."""

    def __init__(self, key: tuple, device: torch.device):
        n, cin, cout, h, w, has_bias, has_res = key
        self.key = key
        self.handle = _get_handle(device)
        bf16, f32, e4m3 = cudnn.data_type.BFLOAT16, cudnn.data_type.FLOAT, cudnn.data_type.FP8_E4M3
        g = cudnn.pygraph(
            io_data_type=bf16,
            intermediate_data_type=f32,
            compute_data_type=f32,
            handle=self.handle,
        )
        t = {}
        t["X"] = g.tensor(name="X", dim=[n, cin, h, w], stride=_nhwc(cin, h, w), data_type=e4m3)
        t["W"] = g.tensor(
            name="W", dim=[cout, cin, 3, 3], stride=_nhwc(cin, 3, 3), data_type=e4m3
        )
        t["SX"] = g.tensor(name="SX", dim=[1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=f32)
        t["SW"] = g.tensor(
            name="SW", dim=[1, cout, 1, 1], stride=[cout, 1, cout, cout], data_type=f32
        )
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
                name="bias", dim=[1, cout, 1, 1], stride=[cout, 1, cout, cout], data_type=bf16
            )
            y = g.add(a=y, b=t["B"], compute_data_type=f32, name="bias_add")
        if has_res:
            t["R"] = g.tensor(name="R", dim=[n, cout, h, w], stride=_nhwc(cout, h, w), data_type=bf16)
            y = g.add(a=y, b=t["R"], compute_data_type=f32, name="res_add")
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
        n_plans = g.get_execution_plan_count()
        self.names = []
        self.ws = []
        for i in range(n_plans):
            try:
                self.names.append(g.get_plan_name_at_index(i))
            except Exception:  # noqa: BLE001
                self.names.append("?")
            try:
                self.ws.append(int(g.get_workspace_size_plan_at_index(i)))
            except Exception:  # noqa: BLE001
                self.ws.append(int(g.get_workspace_size()))
        self.plan = self._fixed_plan(cin, cout, h)
        self.plan_times_ms: Optional[Dict[str, float]] = None
        self.select = "fixed"

    def _fixed_plan(self, cin: int, cout: int, h: int) -> int:
        want = PLAN_K24.get((cin, cout, h))
        eng7 = [i for i, nm in enumerate(self.names) if nm.startswith("eng7")]
        if want is not None:
            for i in eng7:
                if _plan_k24(self.names[i]) == want:
                    return i
        return eng7[0] if eng7 else 0

    def variant_pack(self, xq, s_x, wq, s_w, bias, res, out) -> dict:
        n, cin, cout, h, w, has_bias, has_res = self.key
        vp = {
            self.t["X"]: xq,
            self.t["W"]: wq.permute(0, 3, 1, 2),  # KRSC storage viewed as [K,C,R,S]
            self.t["SX"]: s_x.view(1, 1, 1, 1),
            self.t["SW"]: s_w.view(1, cout, 1, 1),
            self.t["Y"]: out,
        }
        if has_bias:
            vp[self.t["B"]] = bias.view(1, cout, 1, 1)
        if has_res:
            vp[self.t["R"]] = res
        return vp

    def run(self, vp: dict, plan: Optional[int] = None) -> None:
        i = self.plan if plan is None else plan
        cudnn.set_stream(handle=self.handle, stream=torch.cuda.current_stream().cuda_stream)
        ws = torch.empty(max(self.ws[i], 1), dtype=torch.uint8, device="cuda")
        self.g.execute_plan_at_index(vp, ws, i, handle=self.handle)

    def autotune(self, vp: dict, warm: int = 3, iters: int = 10) -> Dict[str, float]:
        """Time every plan on these operands (writes only into vp's output)."""
        times = {}
        for i, nm in enumerate(self.names):
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
                logger.warning(f"fp8_conv3x3 plan {i} {nm} failed: {ex!r}"[:300])
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


def get_graph(key: tuple, device: torch.device) -> _Fp8ConvGraph:
    gr = _graphs.get(key)
    if gr is None:
        with _lock:
            gr = _graphs.get(key)
        if gr is None:
            gr = _Fp8ConvGraph(key, device)
            with _lock:
                _graphs[key] = gr
            _log_once(f"plan{key}", f"flux3_vae::fp8_conv3x3 built {gr.describe()}")
    return gr


def graph_table() -> list:
    """Per-key chosen plan (and plan times when autotuned), for logging."""
    out = []
    for gr in _graphs.values():
        d = gr.describe()
        if gr.plan_times_ms is not None:
            d["plan_times_ms"] = gr.plan_times_ms
        out.append(d)
    return out


def _out_like(xq: Tensor, cout: int) -> Tensor:
    # allocate channels_last directly (empty(...).contiguous(CL) would copy the whole output)
    n, _, h, w = xq.shape
    return torch.empty(
        (n, cout, h, w), dtype=torch.bfloat16, device=xq.device, memory_format=_CL
    )


@torch.library.custom_op("flux3_vae::fp8_conv3x3", mutates_args=())
def fp8_conv3x3(
    xq: Tensor,
    s_x: Tensor,
    wq: Tensor,
    s_w: Tensor,
    bias: Optional[Tensor] = None,
    residual: Optional[Tensor] = None,
) -> Tensor:
    """3x3/stride 1/pad 1 conv of e4m3 xq [N,Cin,H,W] with e4m3 KRSC weights wq
    [Cout,3,3,Cin], epilogue *s_x[1] *s_w[Cout] (+bias[Cout]) (+residual) -> bf16
    [N,Cout,H,W] channels_last."""
    xq = xq.contiguous(memory_format=_CL)
    wq = wq.contiguous()
    s_x = s_x.float().contiguous()
    s_w = s_w.float().contiguous()
    n, cin, h, w = xq.shape
    cout = wq.shape[0]
    if bias is not None:
        bias = bias.to(torch.bfloat16).contiguous()
    if residual is not None:
        residual = residual.to(torch.bfloat16).contiguous(memory_format=_CL)
    key = (n, cin, cout, h, w, bias is not None, residual is not None)
    gr = get_graph(key, xq.device)
    out = _out_like(xq, cout)
    vp = gr.variant_pack(xq, s_x, wq, s_w, bias, residual, out)
    if os.environ.get("FLUX3_FP8_CONV_AUTOTUNE") == "1" and gr.select == "fixed":
        times = gr.autotune(vp)
        best = min(
            (k for k in times if times[k] == times[k]), key=lambda k: times[k], default=None
        )
        if best is not None:
            gr.plan = int(best.split(":")[0])
            gr.select = "autotune"
        _log_once(f"tune{key}", f"flux3_vae::fp8_conv3x3 autotuned {gr.describe()} {times}")
    gr.run(vp)
    return out


@fp8_conv3x3.register_fake
def _(xq, s_x, wq, s_w, bias=None, residual=None):
    n, _, h, w = xq.shape
    return torch.empty(
        (n, wq.shape[0], h, w), dtype=torch.bfloat16, device=xq.device, memory_format=_CL
    )


# --------------------------------------------------------------------------- GN+swish -> e4m3
# |min_t swish(t)| = 0.2784645... (at t = -1.27846); rounded UP so the bound holds.
SWISH_NEG_BOUND = 0.2784646


def gn_swish_fp8(x: Tensor, norm: nn.GroupNorm) -> Tuple[Tensor, Tensor]:
    """swish(GroupNorm(x)) quantized to e4m3 with a per-tensor scale: (q, s_x[1] fp32).

    Pure torch ops (traced and fused by Inductor; no custom op, no host sync).
    Per-channel var_mean (fp32 accumulation for low-precision x, as
    group_norm_channel_stats) plus per-channel amin/amax of x; groups combined exactly
    as group_norm_channel_stats does. With a_c = gamma_c*rstd_g, b_c = beta_c - mean_g*a_c
    the GN output of channel c lies in [lo_c, hi_c] (the affine of min_c/max_c), so
    amax = max(max_{n,c} swish(max(hi_c, 0)), |min swish|) bounds |swish(GN(x))|
    (swish is increasing above -1.278 and >= -0.27846). s_x = amax / 448 and
    q = sat_e4m3(swish(a_c*x + b_c) / s_x), all in fp32, IEEE division, clamp before
    the (non-saturating) cast.
    """
    n, c = x.shape[:2]
    g = norm.num_groups
    acc = torch.float64 if x.dtype == torch.float32 else torch.float32
    red = tuple(range(2, x.dim()))
    var_c, mean_c = torch.var_mean(x.to(acc), dim=red, correction=0)
    mn_c = torch.amin(x, dim=red).float()
    mx_c = torch.amax(x, dim=red).float()
    mean_c = mean_c.view(n, g, c // g)
    var_c = var_c.view(n, g, c // g)
    mean_g = mean_c.mean(-1, keepdim=True)
    var_g = var_c.mean(-1, keepdim=True) + (mean_c - mean_g).square().mean(-1, keepdim=True)
    rstd_g = torch.rsqrt(var_g + norm.eps)
    mean = mean_g.expand(n, g, c // g).reshape(n, c).float()
    rstd = rstd_g.expand(n, g, c // g).reshape(n, c).float()
    a = rstd if norm.weight is None else norm.weight.float().view(1, c) * rstd
    b = -mean * a if norm.bias is None else norm.bias.float().view(1, c) - mean * a
    hi = torch.maximum(a * mn_c + b, a * mx_c + b).clamp_min(0.0)
    top = (hi * torch.sigmoid(hi)).amax()
    s_x = (torch.clamp_min(top, SWISH_NEG_BOUND) / QMAX).reshape(1)
    bshape = (n, c) + (1,) * (x.dim() - 2)
    y = x.float() * a.view(bshape) + b.view(bshape)
    y = y * torch.sigmoid(y)
    q = torch.div(y, s_x.view((1,) * x.dim())).clamp(-QMAX, QMAX).to(FP8)
    return q, s_x


# --------------------------------------------------------------------------- weight prep
def quantize_conv3x3_weight(weight: Tensor) -> Tuple[Tensor, Tensor]:
    """[Cout,Cin,3,3] -> (e4m3 KRSC [Cout,3,3,Cin] contiguous, s_w [Cout] fp32).

    s_w[o] = amax|W[o]| / 448 (0 -> 1 guard, all-zero rows stay zero); saturating cast
    (``.to(float8_e4m3fn)`` does not saturate, so clamp to +-448 first; IEEE div).
    """
    wf = weight.detach().float()
    amax = wf.abs().amax(dim=(1, 2, 3))
    s_w = torch.where(amax > 0, amax / QMAX, torch.ones_like(amax))
    q = torch.div(wf, s_w.view(-1, 1, 1, 1)).clamp(-QMAX, QMAX).to(FP8)
    wq = q.permute(0, 2, 3, 1).contiguous()
    return wq, s_w.contiguous()


def _resnet_blocks(decoder: nn.Module):
    for name, m in decoder.named_modules():
        if (
            hasattr(m, "norm1")
            and hasattr(m, "conv1")
            and hasattr(m, "norm2")
            and hasattr(m, "conv2")
            and isinstance(m.conv1, nn.Conv2d)
            and isinstance(m.conv2, nn.Conv2d)
        ):
            yield name, m


def _in_scope(conv: nn.Conv2d) -> bool:
    return (
        conv.kernel_size == (3, 3)
        and conv.stride == (1, 1)
        and conv.padding == (1, 1)
        and conv.dilation == (1, 1)
        and conv.groups == 1
        and conv.in_channels % 16 == 0
        and conv.out_channels % 16 == 0
    )


def prepare_fp8_convs(decoder: nn.Module) -> int:
    """Quantize the conv1/conv2 of every ResnetBlock in ``decoder`` once (load time).

    Stores ``conv.fp8_wq`` (e4m3 KRSC) and ``conv.fp8_sw`` (fp32 [Cout]) as PLAIN
    attributes, so state_dict keys, .to(dtype) and .to(memory_format) leave them alone,
    and sets ``block._fp8 = True``. Does nothing (logs once) on an fp32 decoder, with
    channels_last off, or without the cuDNN frontend. Returns the number of convs prepared.
    """
    if cudnn is None:
        _log_once("nocudnn", "prepare_fp8_convs: cuDNN frontend not importable; FP8 convs off")
        return 0
    blocks = list(_resnet_blocks(decoder))
    if not blocks:
        return 0
    w0 = blocks[0][1].conv1.weight
    if w0.dtype != torch.bfloat16:
        _log_once("dtype", f"prepare_fp8_convs: decoder trunk is {w0.dtype}, not bf16; FP8 convs off")
        return 0
    if not w0.is_contiguous(memory_format=_CL):
        _log_once("cl", "prepare_fp8_convs: decoder not channels_last; FP8 convs off")
        return 0
    n = 0
    with torch.no_grad():
        for _, blk in blocks:
            convs = [blk.conv1, blk.conv2]
            if not all(_in_scope(c) for c in convs):
                continue
            for c in convs:
                wq, s_w = quantize_conv3x3_weight(c.weight)
                object.__setattr__(c, "fp8_wq", wq)
                object.__setattr__(c, "fp8_sw", s_w)
                n += 1
            blk._fp8 = True
    _log_once("prep", f"prepare_fp8_convs: {n} convs quantized to e4m3 (per-Cout scale)")
    return n
