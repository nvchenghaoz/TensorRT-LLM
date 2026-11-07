# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash-1017 / sm_120 / tp4 with moe_tp2 x moe_ep2 (RTX PRO 6000, four GPUs per instance).

Importing the target opts this process into the shared-kernel behaviour the optimization run served
with (see NATIVE_OPS.md). Every rank resolves and imports the target while it loads the model, before
its first forward, and each variable is read lazily on the kernel's first use, so these take effect
in every rank. An explicit setting in the environment wins.
"""

import os

_SERVED_SHARED_KERNEL_ENV = {
    # one-shot plain all-reduce split over 4 CTAs per token (no clusters on SM120)
    "TLLM_AR_SM120_CTAS_PER_TOKEN": "4",
    # keep the PDL launch attribute on the SM120 all-reduce fusion kernels
    "TLLM_AR_SM120_PDL": "1",
    # custom one/two-shot all-reduce over PCIe P2P up to 512 KiB, NCCL above
    "TLLM_PCIE_CUSTOM_AR_MAX_BYTES": "524288",
    # posted-write (PUSH) low-precision two-shot all-reduce for messages >= 2 MiB
    "TRTLLM_LP_AR_PUSH": "1",
    # wave-cost tile choice for SM120 FP8 block-scale GEMM/BMM at M > 64
    "TRTLLM_SM120_FP8_BLOCKSCALE_TILE_HEURISTIC": "1",
    # faster, bitwise-equal top-k body of gate_forward
    "TRTLLM_GATE_FORWARD_FAST_BODY": "1",
}
for _name, _value in _SERVED_SHARED_KERNEL_ENV.items():
    os.environ.setdefault(_name, _value)
