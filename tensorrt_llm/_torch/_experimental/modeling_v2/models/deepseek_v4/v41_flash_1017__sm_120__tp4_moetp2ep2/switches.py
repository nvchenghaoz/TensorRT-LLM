# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The run's environment switches, frozen at the values it served with.

During optimization every change carried an environment kill switch whose default was the shipped
behavior, and the served configuration set none of them. This target keeps the code that reads them
unchanged but answers every run-added name with its in-code default, so the process environment
cannot select an unvalidated path. Names TensorRT-LLM itself defines (``TRTLLM_ENABLE_PDL``,
``FORCE_DETERMINISTIC``, ...) still read the environment.
"""

from __future__ import annotations

import os
from typing import Optional

RUN_SWITCHES = frozenset(
    (
        "TRTLLM_B12X_WO_B_FUSED",
        "TRTLLM_B12X_WO_B_WIDE_ROWS",
        "TRTLLM_B12X_WO_PROJECTION",
        "TRTLLM_CSA2_ATTN_PDL",
        "TRTLLM_CSA2_CAKE_BODY_TRIGGER_ENTRY",
        "TRTLLM_CSA2_CAKE_DECODE",
        "TRTLLM_CSA2_CAKE_F32_PARTIALS",
        "TRTLLM_CSA2_CAKE_F32_VARIANT",
        "TRTLLM_CSA2_FI_DIRECT_EXECUTE",
        "TRTLLM_CSA2_FI_SPARSE_PREFILL",
        "TRTLLM_CSA2_FLASHINFER_FUSED_GLUE",
        "TRTLLM_CSA2_LOWM_WO",
        "TRTLLM_CSA2_QB_FUSED",
        "TRTLLM_CSA2_QB_L2PF",
        "TRTLLM_CSA2_SPLIT_SINK_DECODE",
        "TRTLLM_DSV41_PREFILL_AR_PIPELINE",
        "TRTLLM_DSV41_PREFILL_AR_PIPELINE_CHUNK_ROWS",
        "TRTLLM_DSV41_PREFILL_AR_PIPELINE_MAX_ROWS",
        "TRTLLM_DSV41_PREFILL_AR_PIPELINE_MIN_ROWS",
        "TRTLLM_DSV4_ATTN_AR_EARLY_TRIGGER",
        "TRTLLM_DSV4_EMBED_GATHER",
        "TRTLLM_DSV4_FUSED_ROUTER",
        "TRTLLM_DSV4_FUSED_ROUTER_ROUTE",
        "TRTLLM_DSV4_FUSED_ROUTER_ROUTE_MAX_M",
        "TRTLLM_DSV4_HC_SEED_FAST",
        "TRTLLM_DSV4_MOE_AR_EARLY_TRIGGER",
        "TRTLLM_DSV4_PREFILL_WO_QUANT",
        "TRTLLM_DSV4_PREFILL_WO_QUANT_BLOCK_M",
        "TRTLLM_DSV4_PREFILL_WO_QUANT_FULL_TILE",
        "TRTLLM_DSV4_PREFILL_WO_QUANT_WARPS",
        "TRTLLM_DSV4_ROUTER_K1_FOLD",
        "TRTLLM_DSV4_ROUTER_K1_FOLD_MAX_M",
        "TRTLLM_DSV4_SHARED_DECODE_SPLIT",
        "TRTLLM_DSV4_SHARED_OVERLAP",
        "TRTLLM_DSV4_SHARED_OVERLAP_MIN_TOKENS",
        "TRTLLM_DSV4_SHARED_ROW_SPLIT",
        "TRTLLM_DSV4_SHARED_ROW_SPLIT_MIN_TOKENS",
        "TRTLLM_DSV4_TAIL_MOE_GRAPH",
        "TRTLLM_DSV4_TAIL_MOE_GRAPH_VERIFY",
        "TRTLLM_ENGRAM_KV_SHARD",
        "TRTLLM_FLASHINFER_MXFP8_B12X",
        "TRTLLM_FLASHINFER_MXFP8_MEMO",
        "TRTLLM_FUSED_FINAL_HC",
        "TRTLLM_FUSED_GREEDY_ARGMAX",
        "TRTLLM_FUSED_GREEDY_ARGMAX_BLOCK",
        "TRTLLM_FUSED_GREEDY_ARGMAX_WARPS",
        "TRTLLM_LOW_M_MXFP8",
        "TRTLLM_MHC_K1_FAST",
        "TRTLLM_MHC_K1_TRIGGER_FIRST",
        "TRTLLM_MHC_K1_TRIGGER_FIRST_MIN_TOKENS",
        "TRTLLM_MHC_SPLIT_LAGGED",
        "TRTLLM_MHC_SPLIT_TEST_KEEP_K2",
        "TRTLLM_MHC_SPLIT_TEST_SLEEP_CYCLES",
        "TRTLLM_MOE_MXFP8_QUANT_FOLD_MAX_TOKENS",
        "TRTLLM_MOE_MXFP8_QUANT_FOLD_MIN_TOKENS",
        "TRTLLM_SHARED_DECODE_DN_BN",
        "TRTLLM_SHARED_DECODE_DN_STAGES",
        "TRTLLM_SHARED_DECODE_DN_WARPS",
        "TRTLLM_SHARED_DECODE_GU_BK",
        "TRTLLM_SHARED_DECODE_GU_BN",
        "TRTLLM_SHARED_DECODE_GU_STAGES",
        "TRTLLM_SHARED_DECODE_GU_WARPS",
        "TRTLLM_SHARED_DECODE_PDL",
        "TRTLLM_SHARED_DECODE_V2",
        "TRTLLM_SHARED_DECODE_V2_DN_BN",
        "TRTLLM_SHARED_DECODE_V2_DN_STAGES",
        "TRTLLM_SHARED_DECODE_V2_DN_WARPS",
        "TRTLLM_SHARED_DECODE_V2_GU_BN",
        "TRTLLM_SHARED_DECODE_V2_GU_STAGES",
        "TRTLLM_SHARED_DECODE_V2_GU_SUB",
        "TRTLLM_SHARED_DECODE_V2_GU_WARPS",
        "TRTLLM_SM12X_FI_",
        "TRTLLM_SM12X_FI_FC12_FUSED",
        "TRTLLM_SM12X_W4A8_FI_DECODE",
        "TRTLLM_SM12X_W4A8_FI_DECODE_MAX_TOKENS",
        "TRTLLM_SM12X_W4A8_FI_FUSED_PROLOGUE",
        "TRTLLM_SM12X_W4A8_FI_PREFILL",
        "TRTLLM_SM12X_W4A8_FI_PREFILL_MAX_TOKENS",
        "TRTLLM_SMALL_N_BF16_GEMV",
    )
)


def env_get(name: str, default: Optional[str] = None) -> Optional[str]:
    """``os.environ.get`` for upstream names; the served (default) value for run-added switches."""
    if name in RUN_SWITCHES:
        return default
    return os.environ.get(name, default)
