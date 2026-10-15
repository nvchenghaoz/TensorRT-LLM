# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A mixed prefill + decode GDN step must match a prefill-only step plus a decode-only step."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modules.mamba.gdn_mixer import Qwen3NextGatedDeltaNet
from tensorrt_llm._utils import is_flashinfer_gdn_prefill_supported_arch

NUM_K_HEADS, NUM_V_HEADS, HEAD_DIM, CONV_WIDTH = 4, 8, 128, 4
KEY_DIM = NUM_K_HEADS * HEAD_DIM
CONV_DIM = 2 * KEY_DIM + NUM_V_HEADS * HEAD_DIM
NUM_SLOTS = 16


def _mixer():
    g = torch.Generator(device="cuda").manual_seed(1)
    return SimpleNamespace(
        conv1d=SimpleNamespace(
            weight=torch.randn(CONV_DIM, CONV_WIDTH, generator=g, device="cuda").to(torch.bfloat16),
            bias=None,
        ),
        activation="silu",
        _fused_prefill_conv=True,
        A_log=torch.rand(NUM_V_HEADS, generator=g, device="cuda").log(),
        dt_bias=torch.randn(NUM_V_HEADS, generator=g, device="cuda"),
        num_k_heads_per_tp=NUM_K_HEADS,
        num_v_heads_per_tp=NUM_V_HEADS,
        head_k_dim=HEAD_DIM,
        head_v_dim=HEAD_DIM,
        key_dim_per_tp=KEY_DIM,
    )


def _batch_kwargs(prefill_lens, num_decodes, slots, has_initial_states):
    num_prefill = len(prefill_lens)
    seq_lens = list(prefill_lens) + [1] * num_decodes
    query_start_loc = torch.tensor(
        [0] + torch.tensor(seq_lens).cumsum(0).tolist(), dtype=torch.int32, device="cuda"
    )
    num_prefill_tokens = sum(prefill_lens)
    slots = torch.tensor(slots, dtype=torch.int32, device="cuda")
    return dict(
        batch_size=num_prefill + num_decodes,
        has_initial_states=torch.tensor(has_initial_states, device="cuda"),
        cache_indices=slots,
        query_start_loc=query_start_loc,
        query_start_loc_long=query_start_loc.long(),
        num_prefill_tokens=num_prefill_tokens,
        num_decode_tokens=num_decodes,
        state_indices_p=slots[:num_prefill],
        state_indices_d=slots[num_prefill:],
        num_prefill=num_prefill,
        num_decodes=num_decodes,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_flashinfer_gdn_prefill_supported_arch(),
    reason="needs a GPU supported by the FlashInfer GDN prefill kernel",
)
@pytest.mark.parametrize("prefill_lens,num_decodes", [([37], 3), ([300, 5, 64], 8)])
def test_mixed_step_matches_separate_prefill_and_decode_steps(prefill_lens, num_decodes):
    torch.manual_seed(0)
    mixer = _mixer()
    num_prefill = len(prefill_lens)
    num_tokens = sum(prefill_lens) + num_decodes
    mixed_qkv = torch.randn(num_tokens, CONV_DIM, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(num_tokens, NUM_V_HEADS, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(num_tokens, NUM_V_HEADS, device="cuda", dtype=torch.bfloat16)
    conv_states = torch.randn(
        NUM_SLOTS, CONV_DIM, CONV_WIDTH - 1, device="cuda", dtype=torch.bfloat16
    )
    ssm_states = (
        torch.randn(NUM_SLOTS, NUM_V_HEADS, HEAD_DIM, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        * 0.1
    )
    slots = list(range(2, 2 + num_prefill + num_decodes))
    has_initial_states = [i % 2 == 1 for i in range(num_prefill)] + [True] * num_decodes
    num_prefill_tokens = sum(prefill_lens)

    mixed_conv, mixed_ssm = conv_states.clone(), ssm_states.clone()
    mixed_out = Qwen3NextGatedDeltaNet.forward_extend(
        mixer,
        mixed_conv,
        mixed_ssm,
        # The decode conv updates its input in place.
        mixed_qkv=mixed_qkv.clone(),
        a=a,
        b=b,
        **_batch_kwargs(prefill_lens, num_decodes, slots, has_initial_states),
    )

    ref_conv, ref_ssm = conv_states.clone(), ssm_states.clone()
    prefill_out = Qwen3NextGatedDeltaNet.forward_extend(
        mixer,
        ref_conv,
        ref_ssm,
        mixed_qkv=mixed_qkv[:num_prefill_tokens],
        a=a[:num_prefill_tokens],
        b=b[:num_prefill_tokens],
        **_batch_kwargs(prefill_lens, 0, slots[:num_prefill], has_initial_states[:num_prefill]),
    )
    decode_kwargs = _batch_kwargs([], num_decodes, slots[num_prefill:], [True] * num_decodes)
    decode_out = Qwen3NextGatedDeltaNet.forward_decode(
        mixer,
        ref_conv,
        ref_ssm,
        decode_kwargs["query_start_loc_long"],
        mixed_qkv=mixed_qkv[num_prefill_tokens:].clone(),
        a=a[num_prefill_tokens:],
        b=b[num_prefill_tokens:],
        cache_indices=decode_kwargs["cache_indices"],
        num_decodes=num_decodes,
    )

    torch.testing.assert_close(mixed_out[:, :num_prefill_tokens], prefill_out, rtol=0, atol=0)
    torch.testing.assert_close(
        mixed_out[:, num_prefill_tokens:].reshape(decode_out.shape), decode_out, rtol=0, atol=0
    )
    torch.testing.assert_close(mixed_conv, ref_conv, rtol=0, atol=0)
    torch.testing.assert_close(mixed_ssm, ref_ssm, rtol=0, atol=0)
