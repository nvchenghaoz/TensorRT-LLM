# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""fused_gdn_conv_post_conv matches causal_conv1d_fn followed by fused_gdn_post_conv bit for bit."""

import pytest
import torch

from tensorrt_llm._torch.modules.mamba.causal_conv1d import causal_conv1d_fn
from tensorrt_llm._torch.modules.mamba.fuse_elementwise_ops import (
    fused_gdn_conv_post_conv,
    fused_gdn_post_conv,
)

NUM_K_HEADS, NUM_V_HEADS, HEAD_DIM, WIDTH = 4, 8, 128, 4
CONV_DIM = (2 * NUM_K_HEADS + NUM_V_HEADS) * HEAD_DIM
# in_proj emits [Q | K | V | Z]; prefill reads the [Q | K | V] columns as a row-strided view.
PROJ_DIM = CONV_DIM + NUM_V_HEADS * HEAD_DIM


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("seq_lens", [[4096], [1920, 128, 2048], [37, 1, 2, 300], [3, 1, 2, 5]])
def test_fused_prefill_conv_matches_conv_then_post_conv(seq_lens):
    torch.manual_seed(0)
    num_tokens = sum(seq_lens)
    weight = torch.randn(CONV_DIM, WIDTH, device="cuda").to(torch.bfloat16)
    mixed_qkv = torch.randn(num_tokens, PROJ_DIM, device="cuda", dtype=torch.bfloat16)[:, :CONV_DIM]
    ba = torch.randn(num_tokens, 2 * NUM_V_HEADS, device="cuda", dtype=torch.bfloat16)
    b, a = ba[:, :NUM_V_HEADS], ba[:, NUM_V_HEADS:]
    A_log = torch.rand(NUM_V_HEADS, device="cuda").log()
    dt_bias = torch.randn(NUM_V_HEADS, device="cuda")
    cu_seqlens = torch.tensor(
        [0] + torch.tensor(seq_lens).cumsum(0).tolist(), dtype=torch.int32, device="cuda"
    )
    slots = torch.arange(1, 1 + len(seq_lens), dtype=torch.int32, device="cuda")
    has_initial_state = torch.tensor([i % 2 == 0 for i in range(len(seq_lens))], device="cuda")
    conv_states = torch.randn(8, CONV_DIM, WIDTH - 1, device="cuda", dtype=torch.bfloat16)
    heads = (NUM_K_HEADS, HEAD_DIM, NUM_V_HEADS, HEAD_DIM)

    expected_states = conv_states.clone()
    conv_out = causal_conv1d_fn(
        mixed_qkv.t(),
        weight,
        None,
        activation="silu",
        conv_states=expected_states,
        has_initial_state=has_initial_state,
        cache_indices=slots,
        query_start_loc=cu_seqlens,
    )
    expected = fused_gdn_post_conv(conv_out, None, a, b, A_log, dt_bias, *heads)

    actual = fused_gdn_conv_post_conv(
        mixed_qkv,
        weight,
        None,
        conv_states,
        cu_seqlens,
        slots,
        has_initial_state,
        a,
        b,
        A_log,
        dt_bias,
        *heads,
    )

    for got, ref in zip(actual, expected):
        torch.testing.assert_close(got, ref, rtol=0, atol=0)
    torch.testing.assert_close(conv_states, expected_states, rtol=0, atol=0)
