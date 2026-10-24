"""Run a row-sliced call on the routes its unsliced call would take.

Some ops route by row count (M): the FP8-origin linears pick a GEMV, Marlin
W4A16 or the W4A4 copy by M, the gate_up SwiGLU-epilogue GEMM is used from 41
rows with an M-dependent tile, the post-attention norm fuses its NVFP4
quantize by M, and the AutoTuner picks a GEMM tactic per M bucket. "M decides
precision, not just tactic": a sliced call (e.g. the final decoder layer run
only on the rows LogitsProcessor reads) must take the route and tactic of the
unsliced M to keep every kept row bitwise equal.

``pin_rows(sliced, full)`` makes ``pinned_rows(sliced)`` return ``full`` and
makes the AutoTuner look up the tactic of ``full`` for a dynamic dimension
equal to ``sliced``. No other row count is affected, and nothing changes
outside the context.
"""

import contextlib
from typing import Optional, Tuple

import torch

_PIN: Optional[Tuple[int, int]] = None


def pin_active() -> bool:
    return _PIN is not None


def pinned_rows(rows: int) -> int:
    """The row count to route ``rows`` rows by (identity when unpinned)."""
    pin = _PIN
    if pin is not None and rows == pin[0]:
        return pin[1]
    return rows


def pin_input_shapes(input_shapes, dynamic_tensor_specs):
    """AutoTuner cache-lookup shapes with pinned dynamic dimensions."""
    pin = _PIN
    if pin is None or not dynamic_tensor_specs:
        return input_shapes
    shapes = None
    for spec in dynamic_tensor_specs:
        i, d = spec.input_idx, spec.dim_idx
        if i >= len(input_shapes) or d >= len(input_shapes[i]):
            continue
        if input_shapes[i][d] != pin[0]:
            continue
        if shapes is None:
            shapes = [list(s) for s in input_shapes]
        shapes[i][d] = pin[1]
    if shapes is None:
        return input_shapes
    return tuple(torch.Size(s) for s in shapes)


@contextlib.contextmanager
def pin_rows(sliced: int, full: int):
    global _PIN
    prev = _PIN
    _PIN = (int(sliced), int(full))
    try:
        yield
    finally:
        _PIN = prev
