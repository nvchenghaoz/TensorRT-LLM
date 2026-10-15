# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GDN prefill convolves a channel-last view of the projection; check it against the channel-major route."""

from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm._torch.modules.mamba.gdn_mixer as gdn_mixer
from tensorrt_llm._torch.modules.mamba.causal_conv1d import causal_conv1d_fn
from tensorrt_llm._torch.modules.mamba.fuse_elementwise_ops import (
    extract_transpose_prefill_slice,
    fused_gdn_post_conv,
)
from tensorrt_llm._utils import is_flashinfer_gdn_prefill_supported_arch

NUM_K_HEADS, NUM_V_HEADS, HEAD_DIM, CONV_WIDTH = 4, 8, 128, 4
KEY_DIM = NUM_K_HEADS * HEAD_DIM
CONV_DIM = 2 * KEY_DIM + NUM_V_HEADS * HEAD_DIM
# in_proj emits [Q | K | V | Z]; the mixer passes the [Q | K | V] columns as a row-strided view.
PROJ_DIM = CONV_DIM + NUM_V_HEADS * HEAD_DIM
NUM_SLOTS = 8


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_flashinfer_gdn_prefill_supported_arch(),
    reason="needs a GPU supported by the FlashInfer GDN prefill kernel",
)
@pytest.mark.parametrize("seq_lens", [[4096], [37, 1, 2, 300]])
def test_prefill_conv_matches_channel_major(monkeypatch, seq_lens):
    torch.manual_seed(0)
    mixer = SimpleNamespace(
        conv1d=SimpleNamespace(
            weight=torch.randn(CONV_DIM, CONV_WIDTH, device="cuda").to(torch.bfloat16),
            bias=None,
        ),
        activation="silu",
        # This test covers the unfused path: channel-last conv, then fused_gdn_post_conv.
        _fused_prefill_conv=False,
        A_log=torch.rand(NUM_V_HEADS, device="cuda").log(),
        dt_bias=torch.randn(NUM_V_HEADS, device="cuda"),
        num_k_heads_per_tp=NUM_K_HEADS,
        num_v_heads_per_tp=NUM_V_HEADS,
        head_k_dim=HEAD_DIM,
        head_v_dim=HEAD_DIM,
        key_dim_per_tp=KEY_DIM,
    )
    num_tokens = sum(seq_lens)
    mixed_qkv = torch.randn(num_tokens, PROJ_DIM, device="cuda", dtype=torch.bfloat16)[:, :CONV_DIM]
    a = torch.randn(num_tokens, NUM_V_HEADS, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(num_tokens, NUM_V_HEADS, device="cuda", dtype=torch.bfloat16)
    conv_states = torch.randn(
        NUM_SLOTS, CONV_DIM, CONV_WIDTH - 1, device="cuda", dtype=torch.bfloat16
    )
    ssm_states = torch.zeros(
        NUM_SLOTS, NUM_V_HEADS, HEAD_DIM, HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )
    query_start_loc = torch.tensor(
        [0] + torch.tensor(seq_lens).cumsum(0).tolist(), dtype=torch.int32, device="cuda"
    )
    slots = torch.arange(1, 1 + len(seq_lens), dtype=torch.int32, device="cuda")
    has_initial_states = torch.tensor([i % 2 == 0 for i in range(len(seq_lens))], device="cuda")

    expected_states = conv_states.clone()
    conv_out = causal_conv1d_fn(
        extract_transpose_prefill_slice(mixed_qkv, num_tokens, 0, CONV_DIM),
        mixer.conv1d.weight,
        None,
        activation="silu",
        conv_states=expected_states,
        has_initial_state=has_initial_states,
        cache_indices=slots,
        query_start_loc=query_start_loc,
    )
    expected = fused_gdn_post_conv(
        conv_out,
        None,
        a,
        b,
        mixer.A_log,
        mixer.dt_bias,
        NUM_K_HEADS,
        HEAD_DIM,
        NUM_V_HEADS,
        HEAD_DIM,
    )

    captured = []

    def capture_post_conv(*args, **kwargs):
        captured.append(fused_gdn_post_conv(*args, **kwargs))
        return captured[-1]

    monkeypatch.setattr(gdn_mixer, "fused_gdn_post_conv", capture_post_conv)
    gdn_mixer.Qwen3NextGatedDeltaNet.forward_extend(
        mixer,
        conv_states,
        ssm_states,
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        batch_size=len(seq_lens),
        has_initial_states=has_initial_states,
        cache_indices=slots,
        query_start_loc=query_start_loc,
        query_start_loc_long=query_start_loc.long(),
        num_prefill_tokens=num_tokens,
        num_decode_tokens=0,
        state_indices_p=slots,
        state_indices_d=slots[:0],
        num_prefill=len(seq_lens),
        num_decodes=0,
    )

    torch.testing.assert_close(conv_states, expected_states, rtol=0, atol=0)
    q, k, v, g, beta = captured[0]
    # Q and K are L2-normalized and may move by one bf16 ulp; V and the gates match exactly.
    for actual, ref in ((q, expected[0]), (k, expected[1])):
        torch.testing.assert_close(actual, ref, rtol=2**-7, atol=0)
    for actual, ref in zip((v, g, beta), expected[2:]):
        torch.testing.assert_close(actual, ref, rtol=0, atol=0)
