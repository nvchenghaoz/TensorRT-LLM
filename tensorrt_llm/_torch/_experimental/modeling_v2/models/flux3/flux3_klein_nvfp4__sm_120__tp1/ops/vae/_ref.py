# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers of the VAE entries' plain-torch references."""

from __future__ import annotations

import contextlib

import torch


@contextlib.contextmanager
def true_fp32_conv():
    """Make cuDNN's fp32 convolutions actually fp32 (no TF32) for the duration.

    Like ``catalog._op.true_fp32_matmul`` for convolutions: ``allow_tf32`` and, where
    torch has it, ``conv.fp32_precision`` both have to move.
    """
    cudnn = torch.backends.cudnn
    prev_allow = cudnn.allow_tf32
    conv = getattr(cudnn, "conv", None)
    prev_prec = getattr(conv, "fp32_precision", None)
    cudnn.allow_tf32 = False
    if prev_prec is not None:
        conv.fp32_precision = "ieee"
    try:
        yield
    finally:
        cudnn.allow_tf32 = prev_allow
        if prev_prec is not None:
            conv.fp32_precision = prev_prec
