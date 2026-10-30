# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""cuDNN frontend plumbing shared by the VAE decoder's two cuDNN graph ops.

One cuDNN handle per device; graphs are built once per shape key at the first (warmup)
call and cached for the process. Each key runs one FIXED plan, picked by name from the
plans the heuristics list (``heur_mode.A`` then ``FALLBACK``, every plan built). A plan
name is ``<engine>_<knob>=<value>_...``, e.g. ``eng7_k24=13``; plans are matched on the
exact engine token and knob values, never on a name prefix (``eng7`` is not ``eng71``).
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, Optional

import torch

from tensorrt_llm.logger import logger

try:
    import cudnn  # cuDNN frontend python bindings
except ImportError:  # pragma: no cover - the target requires them; ops raise on use
    cudnn = None

_lock = threading.Lock()
_handles: Dict[int, Any] = {}


def require_cudnn() -> Any:
    if cudnn is None:
        raise RuntimeError(
            "the FLUX.3 klein sm_120 VAE decoder needs the cuDNN frontend python package "
            "(nvidia-cudnn-frontend)"
        )
    return cudnn


def handle(device: torch.device) -> Any:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    with _lock:
        h = _handles.get(idx)
        if h is None:
            with torch.cuda.device(idx):
                h = require_cudnn().create_handle()
            _handles[idx] = h
        return h


def nhwc(c: int, h: int, w: int) -> list:
    """Strides of a dense NHWC tensor described as [N, C, H, W]."""
    return [c * h * w, 1, w * c, c]


def plan_tokens(name: str) -> tuple[str, Dict[str, str]]:
    """``"eng7_k24=13_k2=1"`` -> ``("eng7", {"k24": "13", "k2": "1"})``."""
    engine, *knobs = name.split("_")
    return engine, dict(k.split("=", 1) for k in knobs if "=" in k)


class FixedPlanGraph:
    """A built cuDNN FE graph, its tensors by name, and the index of the one plan it runs."""

    def __init__(self, g: Any, tensors: dict, device: torch.device, plan: int, plan_name: str):
        self.g = g
        self.t = tensors
        self.handle = handle(device)
        self.plan = plan
        try:
            self.workspace = int(g.get_workspace_size_plan_at_index(plan))
        except Exception:  # noqa: BLE001 - older FE: one workspace size for the graph
            self.workspace = int(g.get_workspace_size())
        self.plan_name = plan_name

    def run(self, variant_pack: dict) -> None:
        cudnn.set_stream(handle=self.handle, stream=torch.cuda.current_stream().cuda_stream)
        ws = torch.empty(max(self.workspace, 1), dtype=torch.uint8, device="cuda")
        self.g.execute_plan_at_index(variant_pack, ws, self.plan, handle=self.handle)


def build_plans(g: Any) -> list[str]:
    """Validate, list (heuristics A + fallback) and build every plan; return their names."""
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    g.check_support()
    g.build_plans(cudnn.build_plan_policy.ALL)
    names = []
    for i in range(g.get_execution_plan_count()):
        try:
            names.append(g.get_plan_name_at_index(i))
        except Exception:  # noqa: BLE001 - an unnamed plan can never match a pin
            names.append("?")
    return names


def select_plan(
    op: str,
    key: tuple,
    names: list[str],
    pinned: Optional[str],
    match: Callable[[str], bool],
    unpinned: Callable[[list[str]], int],
) -> int:
    """Index of the plan to run for ``key``.

    A pinned key runs the first plan ``match`` accepts. If the installed cuDNN lists no
    such plan the call FAILS: falling back to another plan would change the conv's
    accumulation order (and its output bits) without anyone seeing it. A key without a
    pin (an output size outside the measured ones) runs ``unpinned(names)``, logged once.
    """
    if pinned is not None:
        for i, nm in enumerate(names):
            if match(nm):
                return i
        raise RuntimeError(
            f"{op}: the pinned cuDNN plan {pinned} for key {key} is not among the "
            f"{len(names)} plans this cuDNN lists ({names}). The pins were measured on "
            "cuDNN 9.26 / FE 1.27 on SM120; re-measure them for this cuDNN rather than "
            "running an unmeasured plan."
        )
    i = unpinned(names)
    logger.info(f"{op}: no pinned plan for key {key}; running {names[i] if names else '?'}")
    return i
