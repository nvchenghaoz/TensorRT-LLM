# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SM120/121 decode chain for W4A8 MXFP4 x MXFP8 MoE: FlashInfer's SM12x CuTe-DSL ops.

The CUTLASS grouped GEMM has one SM120 FP8xFP4 tile (128x128x128B), so a decode batch of
1-8 tokens pads every active expert to 128 rows, and the chain spends ~7 launches per layer
on permute / strides / activation / finalize. FlashInfer 0.7.2 ships the three-op chain
``mxfp8_q0_route_triton`` -> ``cute_dsl_sm12x_fc1_act_q1_mxfp8_mxfp4`` (8x128 swap-AB tile,
SwiGLU + MXFP8 quant in the epilogue) -> ``cute_dsl_sm12x_fc2_finalize_mxfp8_mxfp4`` (routing
weight + scatter into the BF16 output).  Its unified runner refuses expert parallelism and
allocates per call, so this module calls the ops directly, the way
``SM12xMxfp8Mxfp4Runner.forward`` does, with rank-local expert ids (``-1`` for experts on
another rank: the route kernels give such a pair no row).

The FP4 weight bytes are shared with the CUTLASS layout; only the ue8m0 scale factors are
re-laid out (TRT-LLM 128x4 swizzle -> FlashInfer MN-major int32 packs), once at load.
"""

from __future__ import annotations

import threading

import torch

from tensorrt_llm.logger import logger

#: Calls with at most this many tokens take the FlashInfer chain; larger ones (prefill)
#: stay on CUTLASS. 32 tokens x top-6 = 192 pairs keeps the route on its decode kernel
#: (<= 256 pairs) and the GEMMs on the small-M tiles.
FI_DECODE_MAX_TOKENS = 32

#: Prefill calls (FI_DECODE_MAX_TOKENS < M <= FI_PREFILL_MAX_TOKENS) take the chain too, as the
#: two-kernel fc1_act_q1 -> fc2_finalize (never the persistent decode fc12 kernel) at the tiles of
#: prefill_tiles(), with a per-call route workspace, only while a CUDA graph is being captured (the
#: host cost of the chain's ~10 enqueues is then paid once at capture); eager prefill stays on CUTLASS.
FI_PREFILL_MAX_TOKENS = 8192


def fi_prefill_takes(m: int) -> bool:
    """Whether a BF16 call of m > FI_DECODE_MAX_TOKENS tokens takes the FlashInfer chain."""
    if not FI_DECODE_MAX_TOKENS < m <= FI_PREFILL_MAX_TOKENS:
        return False
    return torch.cuda.is_current_stream_capturing()


#: (max tokens, fc1 tile bm, fc2 tile bm) per prefill bucket; tiles are (bm, 128, BK). None = the
#: FlashInfer heuristic (select_tile on all pairs: 8x128 / 32x128 up to 32 pairs per local expert,
#: else 64x128 / 128x128). Measured on one EP2xTP2 rank (192 experts x 1152), idea 237 stage 1
#: (job 1758): the heuristic is within 1% of the best tile pair at M <= 512 and M > 2048; at
#: 512 < M <= 2048 it picks fc2 128x128, 4-9% slower than 64x128 + 32x128.
_PREFILL_TILE_TABLE = ((512, None, None), (2048, 64, 32), (1 << 30, None, None))


def prefill_tiles(m: int):
    """(fc1 tile bm or None, fc2 tile bm or None) of an m-token prefill call."""
    for max_m, bm1, bm2 in _PREFILL_TILE_TABLE:
        if m <= max_m:
            return bm1, bm2
    return None, None


def prefill_warm_tokens() -> list:
    """Token counts the prefill warmup runs: every Triton int specialization of the route's two
    paths (<= 256 pairs: route_assign_decode; above: count/prefix/assign) and one call per tile
    bucket, so no request JITs."""
    lo, hi = FI_DECODE_MAX_TOKENS + 1, FI_PREFILL_MAX_TOKENS
    ms = set(range(lo, min(hi, lo + 23) + 1))
    prev = FI_DECODE_MAX_TOKENS
    for max_m, _, _ in _PREFILL_TILE_TABLE:
        top = min(max_m, hi)
        if top > prev:
            ms.update({prev + 1, top, max(prev + 1, top - 1)})
        prev = top
        if top >= hi:
            break
    return sorted(m for m in ms if lo <= m <= hi)


#: Decode runs fc1_act_q1 + fc2_finalize as ONE persistent kernel (FlashInfer SM12x
#: kernel_moe_mxfp8_mxfp4_fc12_fused.py: one atomic work counter, all fc1 units before any fc2
#: tile, per-expert release/acquire counters) when the FlashInfer build has it and the route
#: zero-fills the outputs itself; otherwise the two-kernel chain.

_LOCK = threading.Lock()
_OPS = None  # (route, make_ws, fc1, out_sf_shape, fc2, FiActivation) once imported
_FUSED_OP = None  # the fused fc1+fc2 entry point, or False when unavailable
#: Layer shapes (hidden, inter, E, pdl, limit) whose fused fc1+fc2 build failed at warmup:
#: those layers keep the two-kernel chain. The fused kernel takes every size from the
#: tensors (EP4 96 x 2304 and EP2xTP2 192 x 1152 both build), so this gate is never
#: stricter than _fi_decode_layer_reason (hidden % 128, inter % 128).
_FC12_BAD: set = set()
_WARMED: set = set()
_WORKSPACES: dict = {}


def _fused_op():
    global _FUSED_OP
    if _FUSED_OP is None:
        _FUSED_OP = False
        try:
            from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.kernel_moe_mxfp8_mxfp4_fc12_fused import (
                cute_dsl_sm12x_fc12_fused_mxfp8_mxfp4,
            )

            _FUSED_OP = cute_dsl_sm12x_fc12_fused_mxfp8_mxfp4
        except ImportError as e:
            logger.info(f"SM12x fused fc1+fc2 decode kernel unavailable: {e!r}")
    return _FUSED_OP


def _load_ops():
    global _OPS
    if _OPS is None:
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_mxfp8_mxfp4_fc1_act_q1 import (
            cute_dsl_sm12x_fc1_act_q1_mxfp8_mxfp4,
            out_sf_shape,
        )
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_mxfp8_mxfp4_fc2_finalize import (
            cute_dsl_sm12x_fc2_finalize_mxfp8_mxfp4,
        )
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_mxfp8_q0_route_triton import (
            make_mxfp8_q0_route_workspace,
            mxfp8_q0_route_triton,
        )
        from flashinfer.tllm_enums import ActivationType as FiActivationType

        _OPS = (
            mxfp8_q0_route_triton,
            make_mxfp8_q0_route_workspace,
            cute_dsl_sm12x_fc1_act_q1_mxfp8_mxfp4,
            out_sf_shape,
            cute_dsl_sm12x_fc2_finalize_mxfp8_mxfp4,
            FiActivationType,
        )
    return _OPS


_ROUTE_FUSES_PROLOGUE = None


def _route_fuses_prologue() -> bool:
    """Whether FlashInfer's route takes ``expert_offset``/``zero_buf`` (global-id remap and
    output zero-fill inside its route-assign kernel); older builds take local ids."""
    global _ROUTE_FUSES_PROLOGUE
    if _ROUTE_FUSES_PROLOGUE is None:
        import inspect

        params = inspect.signature(_load_ops()[0]).parameters
        _ROUTE_FUSES_PROLOGUE = "expert_offset" in params and "zero_buf" in params
    return _ROUTE_FUSES_PROLOGUE


def fi_decode_available(device: torch.device | None = None) -> tuple[bool, str]:
    """The SM12x runner's ``_check_support`` conditions that do not depend on the layer."""
    if not torch.cuda.is_available():
        return False, "no CUDA"
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) not in ((12, 0), (12, 1)):
        return False, f"SM{major}{minor} is not SM120/121"
    try:
        from flashinfer.cute_dsl import is_cute_dsl_available
        from flashinfer.jit.cpp_ext import get_cuda_version

        if get_cuda_version().major < 13:
            return False, "FlashInfer SM12x MXFP8xMXFP4 needs CUDA >= 13"
        if not is_cute_dsl_available():
            return False, "CuTe DSL is not available"
        _load_ops()
    except Exception as e:  # noqa: BLE001 - any import problem means "not available"
        return False, f"FlashInfer SM12x ops unavailable: {e!r}"
    return True, ""


def swizzled_sf_to_fi(sf_swizzled: torch.Tensor, num_experts: int, rows: int, k: int) -> torch.Tensor:
    """TRT-LLM swizzled ue8m0 SF of a (rows, k) MXFP4 weight -> FlashInfer (E, k/128, 4*rows) uint8."""
    raw = torch.ops.trtllm.block_scale_interleave_reverse(
        sf_swizzled.view(torch.uint8).reshape(num_experts, rows, k // 32)
    )
    out = raw.view(torch.int32).transpose(1, 2).contiguous().view(torch.uint8)
    del raw
    return out


def _align(n: int, a: int) -> int:
    return (n + a - 1) // a * a


def _workspace(x: torch.Tensor, local_ids: torch.Tensor, num_experts: int):
    """Route workspace shared by every layer: layers run one after another on one stream."""
    key = (x.device, x.shape[0], x.shape[1], local_ids.shape[1], num_experts)
    ws = _WORKSPACES.get(key)
    if ws is None:
        ws = _load_ops()[1](x, local_ids, num_experts)
        _WORKSPACES[key] = ws
    return ws


class Sm12xW4a8DecodeState:
    """Per-layer FlashInfer views of the expert weights plus the call."""

    def __init__(
        self,
        *,
        w3_w1_weight: torch.Tensor,
        w3_w1_weight_scale: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_weight_scale: torch.Tensor,
        num_local_experts: int,
        slot_start: int,
        swiglu_limit: float | None,
        enable_pdl: bool,
    ):
        e = num_local_experts
        n1 = w3_w1_weight.shape[1]  # 2 * intermediate
        k1 = w3_w1_weight.view(torch.uint8).reshape(e, n1, -1).shape[2] * 2  # hidden
        n2 = w2_weight.shape[1]  # hidden
        k2 = w2_weight.view(torch.uint8).reshape(e, n2, -1).shape[2] * 2  # intermediate
        self.num_experts, self.hidden, self.inter = e, k1, k2
        self.slot_start = slot_start
        self.swiglu_limit = swiglu_limit
        self.enable_pdl = enable_pdl
        # FP4 data: same storage as the CUTLASS weights.
        self.w1 = w3_w1_weight.view(torch.uint8).reshape(e, n1, k1 // 2)
        self.w2 = w2_weight.view(torch.uint8).reshape(e, n2, k2 // 2)
        self.w1_sf = swizzled_sf_to_fi(w3_w1_weight_scale, e, n1, k1)
        self.w2_sf = swizzled_sf_to_fi(w2_weight_scale, e, n2, k2)
        # (x, ids, weights, route workspace, q1/sf1/out buffer) of a route the router kernel already wrote
        # (DeepseekV4Gate.fused_forward -> torch.ops.trtllm.dsv4_fused_router_route); consumed by the next run().
        self._prerouted = None

    def _buf_layout(self, m: int, pairs: int):
        """(sf1 shape, q1 bytes, sf1 bytes, sf1 offset, out offset, total bytes) of run()'s one q1/sf1/out buffer."""
        out_sf_shape = _load_ops()[3]
        sf1_shape = out_sf_shape(pairs, self.inter, self.num_experts)
        n_q1 = pairs * self.inter
        n_sf1 = sf1_shape[0] * sf1_shape[1] * 4
        off_sf1 = _align(n_q1, 256)
        off_out = _align(off_sf1 + n_sf1, 256)
        return sf1_shape, n_q1, n_sf1, off_sf1, off_out, off_out + m * self.hidden * 2

    def preroute_buffers(self, x: torch.Tensor, top_k: int):
        """(route workspace, q1/sf1/out buffer) that run(x, ...) would use, for a router kernel that writes the
        route (route_assign_decode with expert_offset/zero_buf + MXFP8 q0) itself; None when run() routes another
        way (older FlashInfer route) or this shape's shared workspace does not exist yet (warmup creates it
        for 1..FI_DECODE_MAX_TOKENS; it is never allocated here, so never inside a capture)."""
        if not _route_fuses_prologue() or x.dim() != 2 or not 0 < x.shape[0] <= FI_DECODE_MAX_TOKENS:
            return None
        ws = _WORKSPACES.get((x.device, x.shape[0], x.shape[1], top_k, self.num_experts))
        if ws is None:
            return None
        nbytes = self._buf_layout(x.shape[0], x.shape[0] * top_k)[-1]
        return ws, torch.empty(nbytes, dtype=torch.uint8, device=x.device)

    def set_prerouted(self, x, ids, weights, ws, buf) -> None:
        """The route of (x, ids, weights) is in ws and buf is zero-filled: the next run() of exactly these
        tensors skips route()."""
        self._prerouted = (x, ids, weights, ws, buf)

    def _take_prerouted(self, x, ids, weights):
        """Pop the stash; return (ws, buf) only for exactly the tensors it was made for (the identity check
        of the gate's fused_routing stash), so a stale stash is never trusted."""
        pre = self._prerouted
        if pre is None:
            return None
        self._prerouted = None
        px, pids, pw, ws, buf = pre
        if px is not x or pids is not ids or pw is not weights:
            return None
        return ws, buf

    def _shape_key(self):
        return (self.hidden, self.inter, self.num_experts, self.enable_pdl, self.swiglu_limit)

    def _fc12_op(self):
        """The fused fc1+fc2 op for this layer, or False for the two-kernel chain."""
        if getattr(self, "_force_chain", False) or self._shape_key() in _FC12_BAD:
            return False
        return _fused_op()

    @property
    def extra_bytes(self) -> int:
        return self.w1_sf.numel() + self.w2_sf.numel()

    def local_ids(self, token_selected_experts: torch.Tensor) -> torch.Tensor:
        ids = token_selected_experts.to(torch.int32) - self.slot_start
        return torch.where(
            (ids >= 0) & (ids < self.num_experts), ids, torch.full_like(ids, -1)
        )

    def run(self, x: torch.Tensor, token_selected_experts: torch.Tensor,
            token_final_scales: torch.Tensor) -> torch.Tensor:
        route, _, fc1, out_sf_shape, fc2, fi_act = _load_ops()
        m = x.shape[0]
        if m > FI_DECODE_MAX_TOKENS:
            return self._run_prefill(x, token_selected_experts, token_final_scales)
        fused = _route_fuses_prologue()
        pre = self._take_prerouted(x, token_selected_experts, token_final_scales)
        # The gate already returns int32 ids and FP32 weights: no kernels here.
        ids = token_selected_experts.to(torch.int32)
        weights = token_final_scales.to(torch.float32)
        if not fused:
            ids = self.local_ids(ids)
            pre = None
        pairs = ids.numel()
        # q1, sf1 and the output all have to start at zero: FC2 reads rows of q1 that FC1
        # leaves unwritten (stale bytes there are NaN often enough to poison real rows), the
        # padded rows of sf1 likewise, and FC2 scatter-adds into the output. One buffer,
        # zero-filled by the route-assign kernel itself (no memset node) when the route
        # supports it.
        sf1_shape, n_q1, n_sf1, off_sf1, off_out, nbytes = self._buf_layout(m, pairs)
        if pre is not None and pre[1].numel() != nbytes:
            pre = None
        if pre is not None:
            ws, buf = pre
        else:
            ws = _workspace(x, ids, self.num_experts)
            if fused:
                buf = torch.empty(nbytes, dtype=torch.uint8, device=x.device)
            else:
                buf = torch.zeros(nbytes, dtype=torch.uint8, device=x.device)
        q1 = buf[:n_q1].view(torch.float8_e4m3fn).view(pairs, self.inter)
        sf1 = buf[off_sf1:off_sf1 + n_sf1].view(torch.int32).view(sf1_shape)
        out = buf[off_out:].view(torch.bfloat16).view(m, self.hidden)
        if pre is not None:
            # The router kernel's last CTA (dsv4_fused_router_route) already wrote exactly what the route
            # below writes (route_assign_decode with expert_offset = slot_start and zero_buf = buf, then the
            # MXFP8 q0 of x) into this shape's shared workspace: no route kernels.
            offsets, token_map, route_weights, a_q, a_sf = (
                ws.offsets, ws.token_map, ws.token_weights, ws.q_out, ws.scale_out)
        elif fused:
            # Global ids in, rank-local mapping ([slot_start, slot_start + E) -> 0..E-1,
            # else unrouted) and the zero-fill of buf both happen in the route-assign kernel.
            offsets, token_map, route_weights, a_q, a_sf = route(
                x, ids, weights, self.num_experts, workspace=ws, enable_pdl=self.enable_pdl,
                expert_offset=self.slot_start, zero_buf=buf,
            )
        else:
            offsets, token_map, route_weights, a_q, a_sf = route(
                x, ids, weights, self.num_experts, workspace=ws, enable_pdl=self.enable_pdl
            )
        fused_fc12 = self._fc12_op() if fused else False
        if fused_fc12:
            # q1/sf1 bitwise those of fc1_act_q1; out differs only by the scatter atomics'
            # order, as main's own fc2_finalize does from replay to replay.
            fused_fc12(a_q, a_sf, self.w1, self.w1_sf, offsets, self.w2, self.w2_sf,
                       token_map, route_weights, m, out_q=q1, out_sf=sf1, out=out,
                       swiglu_limit=self.swiglu_limit, enable_pdl=self.enable_pdl)
            return out
        fc1(a_q, a_sf, self.w1, self.w1_sf, offsets, tune=False, enable_pdl=self.enable_pdl,
            out_q=q1, out_sf=sf1, activation=fi_act.Swiglu, swiglu_limit=self.swiglu_limit)
        fc2(q1, sf1, self.w2, self.w2_sf, offsets, token_map, route_weights, m, tune=False,
            enable_pdl=self.enable_pdl, out=out)
        return out

    def _run_prefill(self, x: torch.Tensor, token_selected_experts: torch.Tensor,
                     token_final_scales: torch.Tensor) -> torch.Tensor:
        """A prefill call (m > FI_DECODE_MAX_TOKENS): route -> fc1_act_q1 -> fc2_finalize at the
        bucket's tiles. The route workspace is allocated per call (the caching allocator reuses it;
        a per-shape cache would grow with every prefill size), never shared across layers."""
        route, make_ws, fc1, _, fc2, fi_act = _load_ops()
        self._prerouted = None  # a decode stash never matches a prefill call
        m = x.shape[0]
        fused = _route_fuses_prologue()
        ids = token_selected_experts.to(torch.int32)
        weights = token_final_scales.to(torch.float32)
        if not fused:
            ids = self.local_ids(ids)
        pairs = ids.numel()
        sf1_shape, n_q1, n_sf1, off_sf1, off_out, nbytes = self._buf_layout(m, pairs)
        ws = make_ws(x, ids, self.num_experts)
        if fused:
            buf = torch.empty(nbytes, dtype=torch.uint8, device=x.device)
            offsets, token_map, route_weights, a_q, a_sf = route(
                x, ids, weights, self.num_experts, workspace=ws, enable_pdl=self.enable_pdl,
                expert_offset=self.slot_start, zero_buf=buf,
            )
        else:
            buf = torch.zeros(nbytes, dtype=torch.uint8, device=x.device)
            offsets, token_map, route_weights, a_q, a_sf = route(
                x, ids, weights, self.num_experts, workspace=ws, enable_pdl=self.enable_pdl
            )
        q1 = buf[:n_q1].view(torch.float8_e4m3fn).view(pairs, self.inter)
        sf1 = buf[off_sf1:off_sf1 + n_sf1].view(torch.int32).view(sf1_shape)
        out = buf[off_out:].view(torch.bfloat16).view(m, self.hidden)
        bm1, bm2 = prefill_tiles(m)
        kw1, kw2 = {}, {}
        if bm1 is not None:
            from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_mxfp8_mxfp4_fc1_act_q1 as _fc1m

            tile = (bm1, 128, _fc1m.BK)
            tactics = _fc1m.epi_tactics(tile)
            kw1 = {"tile": tile,
                   "epi": _fc1m.DEFAULT_EPI if _fc1m.DEFAULT_EPI in tactics else tactics[0]}
        if bm2 is not None:
            from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_mxfp8_mxfp4_fc2_finalize as _fc2m

            kw2 = {"tile": (bm2, 128, _fc2m.BK)}
        fc1(a_q, a_sf, self.w1, self.w1_sf, offsets, tune=False, enable_pdl=self.enable_pdl,
            out_q=q1, out_sf=sf1, activation=fi_act.Swiglu, swiglu_limit=self.swiglu_limit, **kw1)
        fc2(q1, sf1, self.w2, self.w2_sf, offsets, token_map, route_weights, m, tune=False,
            enable_pdl=self.enable_pdl, out=out, **kw2)
        return out

    def warmup_prefill(self, top_k: int) -> None:
        """JIT every prefill tile and Triton route specialization (prefill_warm_tokens), outside any
        capture, so neither a graph capture nor an eager request compiles anything."""
        ms = prefill_warm_tokens()
        if not ms:
            return
        key = ("prefill", self.hidden, self.inter, self.num_experts, top_k, self.enable_pdl,
               self.swiglu_limit, self.slot_start, tuple(ms), _PREFILL_TILE_TABLE)
        with _LOCK:
            if key in _WARMED:
                return
            dev = self.w1.device
            for m in ms:
                x = torch.zeros(m, self.hidden, dtype=torch.bfloat16, device=dev)
                ids = (torch.arange(m * top_k, device=dev, dtype=torch.int32)
                       % self.num_experts + self.slot_start).view(m, top_k)
                w = torch.full((m, top_k), 1.0 / top_k, dtype=torch.float32, device=dev)
                self._run_prefill(x, ids, w)
                del x, ids, w
            torch.cuda.synchronize(dev)
            _WARMED.add(key)
            logger.info(
                f"SM12x W4A8 MXFP4xMXFP8 FlashInfer PREFILL chain warmed up for {len(ms)} token counts in "
                f"{ms[0]}..{ms[-1]} (tiles={_PREFILL_TILE_TABLE}, hidden={self.hidden}, "
                f"inter={self.inter}, experts={self.num_experts})"
            )

    def warmup(self, top_k: int) -> None:
        """JIT every kernel and Triton specialization decode can hit, outside any graph capture.

        The CuTe-DSL kernels are keyed by tile and SM count only, so one compile serves every
        layer and every decode batch; the Triton route kernels specialize on scalar
        arguments, so each batch size up to the threshold is run once.
        """
        key = (self.hidden, self.inter, self.num_experts, top_k, self.enable_pdl,
               self.swiglu_limit, self.slot_start, bool(_route_fuses_prologue() and _fused_op()))
        with _LOCK:
            if key in _WARMED:
                return
            dev = self.w1.device
            cases = []
            for m in range(1, FI_DECODE_MAX_TOKENS + 1):
                x = torch.zeros(m, self.hidden, dtype=torch.bfloat16, device=dev)
                ids = (torch.arange(m * top_k, device=dev, dtype=torch.int32)
                       % self.num_experts + self.slot_start).view(m, top_k)
                w = torch.full((m, top_k), 1.0 / top_k, dtype=torch.float32, device=dev)
                cases.append((x, ids, w))
            # Both variants, outside any capture: the two-kernel chain (taken whenever the
            # fused kernel is off or fails to build for this shape) and the fused kernel.
            self._force_chain = True
            try:
                for x, ids, w in cases:
                    self.run(x, ids, w)
            finally:
                self._force_chain = False
            torch.cuda.synchronize(dev)
            if _route_fuses_prologue() and self._fc12_op():
                try:
                    for x, ids, w in cases:
                        self.run(x, ids, w)
                    torch.cuda.synchronize(dev)
                except Exception as e:  # build/compile refusal: keep the chain for this shape
                    _FC12_BAD.add(self._shape_key())
                    logger.warning(
                        f"SM12x fused fc1+fc2 decode kernel off for hidden={self.hidden} "
                        f"inter={self.inter} experts={self.num_experts}: {e!r}")
            _WARMED.add(key)
            logger.info(
                f"SM12x W4A8 MXFP4xMXFP8 FlashInfer decode chain warmed up for 1..{FI_DECODE_MAX_TOKENS} "
                f"tokens (hidden={self.hidden}, inter={self.inter}, experts={self.num_experts})"
            )
