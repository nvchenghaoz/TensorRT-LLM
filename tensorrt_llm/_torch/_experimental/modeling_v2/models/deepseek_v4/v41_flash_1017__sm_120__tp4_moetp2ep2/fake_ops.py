# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fake (meta) implementations of this target's native ops, built from its csrc/.

The loader imports this module right after the target is found in the family
library, so a fake for an op the library does not define fails the load.
"""

import torch

from tensorrt_llm._torch._experimental.modeling_v2._native import namespace_of

_NS = namespace_of(__name__)


def _router_outputs(x, weight):
    m = x.shape[0]
    return (
        x.new_empty((m, 6), dtype=torch.int32),
        x.new_empty((m, 6), dtype=torch.float32),
        x.new_empty((m, weight.shape[0]), dtype=torch.float32),
    )


@torch.library.register_fake(f"{_NS}::dsv4_fused_router")
def _(x, weight, bias, counter, route_scale):
    return _router_outputs(x, weight)


@torch.library.register_fake(f"{_NS}::dsv4_fused_router_route")
def _(x, weight, bias, counter, route_scale, slot_start, num_local, offsets, token_map, token_weights, dst_rows,
      scale_dst_rows, q_out, scale_out, zero_buf):
    return _router_outputs(x, weight)


@torch.library.register_fake(f"{_NS}::dsv4_fused_router_k1fold")
def _(x, x_prev, residual, post, comb, pre, norm_weight, norm_eps, reducer_threads, weight, bias, counter,
      route_scale):
    return _router_outputs(x, weight)


@torch.library.register_fake(f"{_NS}::dsv4_fused_router_route_k1fold")
def _(x, x_prev, residual, post, comb, pre, norm_weight, norm_eps, reducer_threads, weight, bias, counter,
      route_scale, slot_start, num_local, offsets, token_map, token_weights, dst_rows, scale_dst_rows, q_out,
      scale_out, zero_buf):
    return _router_outputs(x, weight)


@torch.library.register_fake(f"{_NS}::csa2_mxfp8_low_m_gemm")
def _(x, w, wsf, warps, cluster_k, row_tiles, n_split, pdl, xq=None, xsf=None, xq_out=None, xsf_out=None,
      dbg_ts=None, stages=1, debug_flags=0, pf_w=None, pf_sf=None, pf_frac=256):
    n = w.shape[0]
    if n_split == 0:
        return [x.new_empty((x.shape[0], n))]
    return [x.new_empty((x.shape[0], n_split)), x.new_empty((x.shape[0], n - n_split))]


@torch.library.register_fake(f"{_NS}::csa2_qnorm_low_m_gemm")
def _(x, w, wsf, norm_w, eps, warps, cluster_k, row_tiles, n_split, pdl, debug_flags=0, xq_out=None,
      xsf_out=None):
    n = w.shape[0]
    qr = x.new_empty((x.shape[0], x.shape[1]))
    if n_split == 0:
        return [qr, x.new_empty((x.shape[0], n))]
    return [qr, x.new_empty((x.shape[0], n_split)), x.new_empty((x.shape[0], n - n_split))]


@torch.library.register_fake(f"{_NS}::csa2_lowm_wo_a")
def _(o, positions, cos_sin, w, wsc, groups, head_dim, nope_dim, warps, cluster_k, row_tiles, pdl, xq_dbg=None,
      xs_dbg=None, debug_flags=0):
    rank = w.numel() // o.shape[1]
    return o.new_empty((o.shape[0], groups * rank))
