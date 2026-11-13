# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in wall-clock stamps along the VisualGen response path.

Set ``TRTLLM_VISUAL_GEN_RESPONSE_TIMING=1`` to log one ``[vg-timing]`` line per
hop (route start, client send, worker generation, worker handoff, client
receive/rebuild/store, route await, save start/end, encode breakdown). Every
line carries ``t=<time.time()>`` so hops in different threads of one process
(or different processes on one host) can be subtracted. Off by default; when
off every call is a single boolean check.
"""

import os
import time

from tensorrt_llm.logger import logger

ENABLED = os.environ.get("TRTLLM_VISUAL_GEN_RESPONSE_TIMING", "0") == "1"


def stamp(event: str, t: float = None, **fields) -> None:
    """Log ``event`` at wall time ``t`` (default: now) with ``fields``."""
    if not ENABLED:
        return
    if t is None:
        t = time.time()
    body = " ".join(f"{k}={v}" for k, v in fields.items())
    logger.info(f"[vg-timing] event={event} t={t:.6f} {body}")


def cuda_allocation_bytes(tensor) -> int:
    """Size of the driver allocation (caching-allocator segment) holding ``tensor``.

    Uses ``cuMemGetAddressRange``; returns -1 when it cannot be read.
    """
    try:
        import ctypes

        lib = ctypes.CDLL("libcuda.so.1")
        base = ctypes.c_uint64()
        size = ctypes.c_size_t()
        rc = lib.cuMemGetAddressRange_v2(
            ctypes.byref(base), ctypes.byref(size), ctypes.c_uint64(tensor.data_ptr())
        )
        return int(size.value) if rc == 0 else -1
    except Exception:
        return -1
