# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3: block i+1's AdaLN modulation table computed in block i's all-reduce drain.

Every transformer block starts with ``mod = adaln_proj.table(temb)``: silu(temb) -> bf16 ->
Linear(K=2688, N=96768, bias) at M = 1 or 2 rows. The 520 MB replicated weight makes it a
DRAM-bound GEMV (~0.38 ms on SM120) on the critical path before norm1, and its input (temb) is the
same for every block of one forward. With the token-chunked TP overlap, block i ends with the
compute stream idle while the last feed-forward all-reduces drain, so block i computes block
i+1's table there and block i+1 takes it instead of recomputing.

Bitwise by construction: the prefetch and the block's own (miss) call run the same eager
``F.linear(x_act, weight, bias)`` on the same ``x_act``, the output of the block's compiled
``silu(temb).to(bf16)`` (computed identically by every block's compiled code from the same temb).

The handoff is one slot per model (``H3AdalnTableSlot``), keyed on the consumer's weight storage
(which identifies the block), the step controller's generation (bumped by
``DenoiseStepController.set_step``/``reset``), and temb's storage, shape and version. Every op call
empties the slot, so a table never outlives the next block call: a table whose consumer did not run
(the last front block of a step-cached forward) is dropped at the next forward's block 0.
"""

from __future__ import annotations

import weakref
from typing import Callable, Optional

import torch
import torch.nn.functional as F

_SLOTS: list["H3AdalnTableSlot"] = []


def _version(t: torch.Tensor) -> int:
    try:
        return int(t._version)
    except RuntimeError:  # inference tensors carry no version counter (and are never mutated here)
        return -1


class H3AdalnTableSlot:
    """Per-model handoff of one precomputed AdaLN modulation table (see the module docstring)."""

    def __init__(self, model: Optional[torch.nn.Module] = None) -> None:
        self.slot_id = len(_SLOTS)
        _SLOTS.append(self)
        self._model = weakref.ref(model) if model is not None else None
        self._hooked_controller = None
        self._key = None
        self._layer: Optional[int] = None
        self._table: Optional[torch.Tensor] = None
        self.enabled = True
        self.hits = 0
        self.misses = 0
        self.prefetches = 0
        self.dropped = 0  # prefetched tables nobody took (cached forwards)
        self.total = {"hits": 0, "misses": 0, "prefetches": 0, "dropped": 0}

    # ----------------------------------------------------------------- generation / logging
    def _controller(self):
        model = self._model() if self._model is not None else None
        ctrl = getattr(model, "step_precision_controller", None) if model is not None else None
        if ctrl is not None and ctrl is not self._hooked_controller:
            hooks = getattr(ctrl, "reset_hooks", None)
            if hooks is not None:
                hooks.append(self.log_and_clear)
            self._hooked_controller = ctrl
        return ctrl

    def generation(self) -> int:
        return int(getattr(self._controller(), "generation", 0))

    def _make_key(self, weight: torch.Tensor, temb: torch.Tensor) -> tuple:
        return (
            weight.data_ptr(),
            tuple(weight.shape),
            self.generation(),
            temb.data_ptr(),
            tuple(temb.shape),
            _version(temb),
        )

    def counts(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "prefetches": self.prefetches,
            "dropped": self.dropped,
        }

    def log_and_clear(self) -> None:
        """Log this request's counts once (called on controller reset) and drop any table."""
        if self._table is not None:
            self.dropped += 1
        self._table = self._key = self._layer = None
        if self.hits or self.misses or self.prefetches:
            import torch.distributed as dist

            from tensorrt_llm.logger import logger

            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            logger.info(f"[adaln_prefetch] rank={rank} request: {self.counts()}")
            for k, v in self.counts().items():
                self.total[k] += v
            self.hits = self.misses = self.prefetches = self.dropped = 0

    # ----------------------------------------------------------------- producer / consumers
    def fill(
        self,
        layer_idx: int,
        linear: torch.nn.Module,
        temb: torch.Tensor,
        x_act: torch.Tensor,
    ) -> None:
        """Compute ``layer_idx``'s table into the slot (block i's drain, compute stream)."""
        if not self.enabled:
            return
        if self._table is not None:
            self.dropped += 1
        weight, bias = linear.weight, linear.bias
        self._table = F.linear(x_act, weight, bias)
        self._key = self._make_key(weight, temb)
        self._layer = layer_idx
        self.prefetches += 1

    def peek(self, layer_idx: int) -> Optional[torch.Tensor]:
        """The table prefetched for ``layer_idx`` in the current generation, else None.

        No side effects: no pop, no hit/miss count, no key or generation change. The op stays the
        only popper. The returned tensor is READ-ONLY: the op later hands this same tensor to block
        ``layer_idx`` as its modulation table, so writing it would change that block's output.
        Returns None when nothing was prefetched (block 0, the last block's drain, the serial path,
        tp_size 1, prefetch disabled), after the op popped it, or when the generation moved on.
        """
        if self._table is None or self._layer != layer_idx or self._key is None:
            return None
        if self._key[2] != self.generation():
            return None
        return self._table

    def take(
        self,
        temb: torch.Tensor,
        x_act: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Pop the slot: the prefetched table on a key match (hit), else compute it (miss)."""
        table, key = self._table, self._key
        self._table = self._key = self._layer = None
        if table is not None:
            if key == self._make_key(weight, temb):
                self.hits += 1
                return table
            self.dropped += 1
        self.misses += 1
        return F.linear(x_act, weight, bias)


def make_prefetch_hook(
    slot: H3AdalnTableSlot, layer_idx: int, linear: torch.nn.Module
) -> Callable[[torch.Tensor, torch.Tensor], None]:
    """A drain hook ``(temb, x_act) -> None`` filling ``layer_idx``'s table (plain module ref)."""

    def hook(temb: torch.Tensor, x_act: torch.Tensor) -> None:
        slot.fill(layer_idx, linear, temb, x_act)

    return hook


@torch.library.custom_op("trtllm::minimax_h3_adaln_table", mutates_args=(), device_types="cuda")
def h3_adaln_table(
    temb: torch.Tensor,
    x_act: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    slot_id: int,
) -> torch.Tensor:
    """``F.linear(x_act, weight, bias)``, taken from the model's prefetch slot when it holds it.

    ``x_act = silu(temb).to(weight.dtype)`` from the caller's compiled code; ``temb`` only keys the
    slot. Opaque to Inductor so the GEMM is the same eager cuBLAS call on hit and miss.
    """
    return _SLOTS[slot_id].take(temb, x_act, weight, bias)


@h3_adaln_table.register_fake
def _(temb, x_act, weight, bias, slot_id):
    return x_act.new_empty((x_act.shape[0], weight.shape[0]))
