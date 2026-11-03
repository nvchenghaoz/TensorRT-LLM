# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NCCL environment defaults for PCIe-only (no NVLink) GPU groups.

NCCL's default P2P level stops at PXB (GPUs under one PCIe switch). On PCIe
boxes whose GPUs sit behind different switches of the same host bridge, the
ring edges between switches then fall back to SHM (host-staged copies) even
though CUDA peer access works. Allowing P2P through the host bridge
(NCCL_P2P_LEVEL=PHB) keeps every edge on GPU-direct P2P; it never crosses a
CPU socket (that is SYS). This changes only the transport, not the reduction
arithmetic.

NCCL reads these knobs once per process at the first communicator init, so
this must run before any NCCL communicator is created; a value the user set
is never overridden.
"""

import os
from typing import Dict, Optional, Sequence

from tensorrt_llm.logger import logger

_PCIE_DEFAULTS = {"NCCL_P2P_LEVEL": "PHB"}
# User knobs that already decide P2P transport: leave everything alone.
_USER_P2P_KNOBS = ("NCCL_P2P_LEVEL", "NCCL_P2P_DISABLE")

_applied: Optional[Dict[str, str]] = None


def _has_nvlink(device_ids: Sequence[int]) -> bool:
    import torch
    try:
        import pynvml
    except ImportError:
        return True  # cannot tell: assume NVLink, change nothing
    try:
        pynvml.nvmlInit()
    except Exception:
        return True
    try:
        for dev in device_ids:
            uuid = str(torch.cuda.get_device_properties(dev).uuid)
            if not uuid.startswith("GPU-"):
                uuid = "GPU-" + uuid
            handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)
            for link in range(pynvml.NVML_NVLINK_MAX_LINKS):
                try:
                    if pynvml.nvmlDeviceGetNvLinkState(handle, link):
                        return True
                except pynvml.NVMLError:
                    continue
        return False
    except Exception:
        return True


def apply_pcie_nccl_defaults(
        device_ids: Optional[Sequence[int]] = None) -> Dict[str, str]:
    """Set NCCL env defaults when the local GPUs are PCIe-only with peer access.

    Returns the variables this call set (empty when nothing applied).
    """
    global _applied
    if _applied is not None:
        return _applied
    _applied = {}
    if os.environ.get("TLLM_NCCL_PCIE_DEFAULTS", "1") == "0":
        return _applied
    if any(k in os.environ for k in _USER_P2P_KNOBS):
        return _applied
    try:
        import torch
        if not torch.cuda.is_available():
            return _applied
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        device_ids = list(device_ids)
        if len(device_ids) < 2:
            return _applied
        for a in device_ids:
            for b in device_ids:
                if a != b and not torch.cuda.can_device_access_peer(a, b):
                    return _applied
        if _has_nvlink(device_ids):
            return _applied
    except Exception as e:  # never fail startup over a tuning default
        logger.debug(f"NCCL PCIe defaults skipped: {e}")
        return _applied
    for k, v in _PCIE_DEFAULTS.items():
        if k not in os.environ:
            os.environ[k] = v
            _applied[k] = v
    if _applied:
        logger.info(f"PCIe-only GPU group without NVLink: NCCL defaults {_applied}")
    return _applied
