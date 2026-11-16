# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The target's FlashInfer JIT pieces (`ops/_flashinfer_jit.py`).

* CPU: `supported_prefill_args` admits exactly the FA2 prefill configurations the target
  builds its own module for -- bf16 Q and output over an e4m3 pool, int32 indices, head 256
  or 512 (QK = VO), no RoPE, no soft cap, no fp16 QK reduction, no per-token scales -- with or
  without the sliding window.
* GPU (sm_120): `Fa2PrefillWrapper`, planned and run as the target's context attention does,
  is bitwise FlashInfer's own `BatchPrefillWithPagedKVCacheWrapper` on the same paged FP8 KV:
  the full-attention layers (head 512, 32 q / 4 kv heads, causal) and the sliding-window ones
  (head 256, 16 kv heads, window_left 1023), at 300 / 1000 / 2000-row prompts.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.gemma4.gemma4_31b_it_nvfp4__sm_120__tp1.ops import (  # noqa: E501
    _flashinfer_jit as jit,
)
from tensorrt_llm._utils import get_sm_version

_SM120 = pytest.mark.skipif(
    get_sm_version() != 120, reason="the Gemma-4-31B sm_120 tp1 kernels are built for sm_120"
)

bf16, e4m3 = torch.bfloat16, torch.float8_e4m3fn

# ------------------------------------------------------------- supported_prefill_args ---


def _args(head=512, window=False, **override):
    """`get_batch_prefill_module("fa2", *args)` arguments as the wrapper's plan passes them:
    (dtype_q, dtype_kv, dtype_o, dtype_idx, head_qk, head_vo, pos_encoding_mode,
    use_sliding_window, use_logits_soft_cap, use_fp16_qk_reduction)."""
    fields = dict(
        dtype_q=bf16,
        dtype_kv=e4m3,
        dtype_o=bf16,
        dtype_idx=torch.int32,
        head_qk=head,
        head_vo=head,
        pos_enc=0,
        window=window,
        soft_cap=False,
        fp16_qk=False,
    )
    fields.update(override)
    return tuple(fields.values())


@pytest.mark.cpu_only
@pytest.mark.parametrize("head", [256, 512])
@pytest.mark.parametrize("window", [False, True])
def test_the_served_configurations_are_admitted(head, window):
    assert jit.supported_prefill_args(_args(head, window))
    # A trailing per-token-scale flag, when the planner passes one, must be off.
    assert jit.supported_prefill_args(_args(head, window) + (False,))
    assert not jit.supported_prefill_args(_args(head, window) + (True,))


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "override",
    [
        dict(dtype_q=torch.float16),
        dict(dtype_q=e4m3),
        dict(dtype_kv=bf16),
        dict(dtype_kv=torch.float8_e5m2),
        dict(dtype_o=torch.float16),
        dict(dtype_idx=torch.int64),
        dict(head_qk=128, head_vo=128),
        dict(head_qk=192, head_vo=192),
        dict(head_qk=512, head_vo=256),
        dict(pos_enc=1),
        dict(soft_cap=True),
        dict(fp16_qk=True),
    ],
    ids=lambda o: ",".join(f"{k}={v}" for k, v in o.items()),
)
def test_every_other_configuration_is_left_to_flashinfer(override):
    assert not jit.supported_prefill_args(_args(**override))


# ---------------------------------------------------------------- Fa2PrefillWrapper ---

_PAGE = 32
_WORKSPACE_BYTES = 256 << 20


def _paged_case(seq_lens, num_kv_heads, head_dim, seed):
    """A fresh-prompt step: each request's K/V fills its own (shuffled) pages of an FP8 NHD
    pool, its query rows are the whole prompt."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    pages = [-(-s // _PAGE) for s in seq_lens]
    total = sum(pages)
    kv_cache = torch.randn(
        total, 2, _PAGE, num_kv_heads, head_dim, generator=g, device="cuda"
    ).to(e4m3)
    kv_indptr = torch.tensor([0, *torch.tensor(pages).cumsum(0).tolist()], dtype=torch.int32)
    kv_indices = torch.randperm(total, generator=g, device="cuda").to(torch.int32)
    last_page_len = torch.tensor([(s - 1) % _PAGE + 1 for s in seq_lens], dtype=torch.int32)
    qo_indptr = torch.tensor([0, *torch.tensor(seq_lens).cumsum(0).tolist()], dtype=torch.int32)
    return qo_indptr, kv_indptr, kv_indices, last_page_len, kv_cache


def _run(wrapper, case, q, num_heads, num_kv_heads, head_dim, window_left):
    qo_indptr, kv_indptr, kv_indices, last_page_len, kv_cache = case
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        last_page_len,
        num_heads,
        num_kv_heads,
        head_dim,
        _PAGE,
        causal=True,
        # Gemma4 attends with scale 1 (q_scaling = 1 / sqrt(head_dim)).
        sm_scale=1.0,
        window_left=window_left,
        q_data_type=bf16,
        kv_data_type=e4m3,
    )
    return wrapper.run(q, kv_cache)


@_SM120
@pytest.mark.parametrize(
    "num_kv_heads, head_dim, window_left",
    [(4, 512, -1), (16, 256, 1023)],
    ids=["full-attention-d512", "sliding-d256-w1023"],
)
@pytest.mark.parametrize(
    "seq_lens", [(300,), (1000,), (2000,), (300, 1000, 2000)], ids=lambda s: "x".join(map(str, s))
)
def test_target_prefill_is_bitwise_flashinfers(seq_lens, num_kv_heads, head_dim, window_left):
    if not jit.available():
        pytest.skip(
            f"FlashInfer {jit.FLASHINFER_VERSION} is not installed; the wrapper would run "
            "FlashInfer's own module"
        )
    import flashinfer

    num_heads = 32
    case = _paged_case(seq_lens, num_kv_heads, head_dim, seed=sum(seq_lens) + head_dim)
    g = torch.Generator(device="cuda").manual_seed(head_dim)
    q = (
        torch.randn(sum(seq_lens), num_heads, head_dim, generator=g, device="cuda")
        * head_dim**-0.5
    ).to(bf16)

    def workspace():
        return torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device="cuda")

    ours = jit.Fa2PrefillWrapper(workspace(), "NHD", backend="fa2")
    out = _run(ours, case, q, num_heads, num_kv_heads, head_dim, window_left)
    assert jit.prefill_module.cache_info().currsize > 0, "the target's module was not built"
    stock = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace(), "NHD", backend="fa2")
    want = _run(stock, case, q, num_heads, num_kv_heads, head_dim, window_left)

    assert out.shape == want.shape == q.shape and out.dtype == bf16
    assert torch.isfinite(want).all()
    assert torch.equal(out.view(torch.int16), want.view(torch.int16))
