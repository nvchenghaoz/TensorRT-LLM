# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded pinned staging for asynchronous host-to-device metadata uploads."""

import numpy as np
import torch

from tensorrt_llm._utils import maybe_pin_memory, prefer_pinned

_STAGING_RING = 2


def copy_host_to_device(store: dict, key, destination: torch.Tensor, value: torch.Tensor) -> None:
    """Asynchronously copy a host tensor into ``destination`` through pinned staging.

    Pageable or strided sources make ``copy_(non_blocking=True)`` a synchronous
    staged copy inside the driver (a strided source is first materialized as a
    pageable temporary), so the value is first written into a contiguous pinned
    buffer. Each key owns a small ring of buffers with a CUDA event recorded
    after every copy: a buffer is only rewritten once the copy that last read
    it has completed, so the host may run ahead of the device (overlap
    scheduling, chunked prefill) without corrupting a copy still in flight.
    """
    if value.numel() == 0:
        return
    if (
        not destination.is_cuda
        or not prefer_pinned()
        or (value.is_pinned() and value.is_contiguous())
    ):
        # Host destinations (tests) and already pinned sources need no staging.
        destination.copy_(value, non_blocking=destination.is_cuda)
        return
    slot = (key, value.dtype)
    ring = store.get(slot)
    if ring is None:
        ring = store[slot] = {
            "buffers": [None] * _STAGING_RING,
            "events": [None] * _STAGING_RING,
            "next": 0,
        }
    index = ring["next"]
    ring["next"] = (index + 1) % _STAGING_RING
    event = ring["events"][index]
    if event is not None:
        event.synchronize()
    buffer = ring["buffers"][index]
    if buffer is None or buffer.numel() < value.numel():
        buffer = maybe_pin_memory(
            torch.empty(
                max(value.numel(), 2 * buffer.numel() if buffer is not None else 0),
                dtype=value.dtype,
                device="cpu",
            )
        )
        ring["buffers"][index] = buffer
    view = buffer[: value.numel()].view(value.shape)
    view.copy_(value)
    destination.copy_(view, non_blocking=True)
    if event is None:
        event = ring["events"][index] = torch.cuda.Event()
    event.record(torch.cuda.current_stream(destination.device))


def copy_packed_host_to_device(store: dict, key, destination: torch.Tensor, nbytes: int, fill) -> None:
    """Asynchronously upload ``nbytes`` bytes that ``fill`` writes on the host.

    Same ring and event discipline as :func:`copy_host_to_device` (slot
    ``(key, torch.uint8)``), but ``fill(array)`` writes the payload straight
    into the pinned staging buffer through a NumPy ``uint8`` view, so no
    pageable temporary is built and no ``torch`` CPU copy runs. A large torch
    CPU ``copy_`` is split across the intra-op thread pool, whose fork/join
    costs milliseconds on a loaded many-core host; NumPy copies are serial.
    ``destination`` is a contiguous ``uint8`` tensor of ``nbytes`` elements.
    """
    if nbytes == 0:
        return
    if not destination.is_cuda or not prefer_pinned():
        host = np.empty(nbytes, dtype=np.uint8)
        fill(host)
        destination.copy_(torch.from_numpy(host), non_blocking=destination.is_cuda)
        return
    slot = (key, torch.uint8)
    ring = store.get(slot)
    if ring is None:
        ring = store[slot] = {
            "buffers": [None] * _STAGING_RING,
            "events": [None] * _STAGING_RING,
            "next": 0,
        }
    index = ring["next"]
    ring["next"] = (index + 1) % _STAGING_RING
    event = ring["events"][index]
    if event is not None:
        event.synchronize()
    buffer = ring["buffers"][index]
    if buffer is None or buffer.numel() < nbytes:
        buffer = maybe_pin_memory(
            torch.empty(
                max(nbytes, 2 * buffer.numel() if buffer is not None else 0),
                dtype=torch.uint8,
                device="cpu",
            )
        )
        ring["buffers"][index] = buffer
    view = buffer[:nbytes]
    fill(view.numpy())
    destination.copy_(view, non_blocking=True)
    if event is None:
        event = ring["events"][index] = torch.cuda.Event()
    event.record(torch.cuda.current_stream(destination.device))
