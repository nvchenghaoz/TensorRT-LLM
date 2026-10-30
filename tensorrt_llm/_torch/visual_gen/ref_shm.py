# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One POSIX shared-memory file per reference payload, for the coordinator -> rank0 hop.

The producer (coordinator) writes the payload into a fresh tmpfs file
``/dev/shm/trtllm_vgref_<pid>_<uuid4>`` with ``os.write`` (pages are allocated in the
kernel, no user page faults and no torch_shm_manager round trip) and sends
``{"posix_shm": path, "size": n}``. The consumer (rank0) opens it, unlinks it at once (a
handle is consumed exactly once) and reads it back with ``os.read`` into a fresh ``bytes``.

Measured on 2048^2 PNG references (6.7 / 9.1 MB, two processes): the torch shared-tensor
hop takes 8.2 / 9.4 ms, this one 3.4 / 4.6 ms.

``put`` returns None (the caller keeps the torch path) when the payload is empty or any
OSError occurs (no /dev/shm, ENOSPC, ...). Files the producer created and nobody consumed are unlinked at
producer exit (atexit). There is no sweep by PID of other processes' files: /dev/shm can
be shared across PID namespaces, so a PID that looks dead may belong to a live producer.
"""

import atexit
import os
import threading
import uuid
from typing import Any, Dict, Optional

SHM_DIR = "/dev/shm"
PREFIX = "trtllm_vgref_"
HANDLE_KEY = "posix_shm"

_lock = threading.Lock()
_owned: set = set()
_initialized = False
_PRUNE_AT = 64


def is_handle(handle: Any) -> bool:
    return isinstance(handle, dict) and HANDLE_KEY in handle


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _sweep_owned() -> None:
    with _lock:
        names = list(_owned)
        _owned.clear()
    for name in names:
        _unlink(name)


def _init_once() -> None:
    global _initialized
    with _lock:
        if _initialized:
            return
        _initialized = True
    atexit.register(_sweep_owned)


def _prune_owned() -> None:
    # Consumers unlink in another process; forget names that are gone so the set stays small.
    with _lock:
        if len(_owned) < _PRUNE_AT:
            return
        gone = [name for name in _owned if not os.path.exists(name)]
        _owned.difference_update(gone)


def _valid_path(path: Any) -> bool:
    return (
        isinstance(path, str)
        and os.path.dirname(path) == SHM_DIR
        and os.path.basename(path).startswith(PREFIX)
    )


def put(content) -> Optional[Dict[str, Any]]:
    """Write content into a new shm file and return its handle, or None (use the torch path)."""
    try:
        mv = memoryview(content)
        if mv.ndim != 1 or mv.itemsize != 1:
            mv = mv.cast("B")
    except (TypeError, ValueError):
        return None
    n = mv.nbytes
    if n == 0:
        return None
    _init_once()
    _prune_owned()
    path = os.path.join(SHM_DIR, f"{PREFIX}{os.getpid()}_{uuid.uuid4().hex}")
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_CLOEXEC, 0o600)
    except OSError:
        return None
    with _lock:
        _owned.add(path)
    ok = False
    try:
        off = 0
        while off < n:
            written = os.write(fd, mv[off:])
            if written <= 0:
                raise OSError("short write to " + path)
            off += written
        ok = True
    except OSError:
        pass
    finally:
        os.close(fd)
        if not ok:
            discard_path(path)
    return {HANDLE_KEY: path, "size": n} if ok else None


def discard_path(path: str) -> None:
    _unlink(path)
    with _lock:
        _owned.discard(path)


def take(handle: Dict[str, Any]) -> bytes:
    """Read a handle's payload back as bytes; the file is unlinked whatever happens."""
    path = handle[HANDLE_KEY]
    if not _valid_path(path):
        raise ValueError(f"not a reference shm path: {path!r}")
    n = int(handle["size"])
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            os.unlink(path)
            size = os.fstat(fd).st_size
            if size != n:
                raise ValueError(f"{path}: size {size}, expected {n}")
            data = os.read(fd, n)
            if len(data) != n:
                parts = [data]
                got = len(data)
                while got < n:
                    chunk = os.read(fd, n - got)
                    if not chunk:
                        raise EOFError(f"{path}: {got} of {n} bytes")
                    parts.append(chunk)
                    got += len(chunk)
                data = b"".join(parts)
            return data
        finally:
            os.close(fd)
    finally:
        discard_path(path)
