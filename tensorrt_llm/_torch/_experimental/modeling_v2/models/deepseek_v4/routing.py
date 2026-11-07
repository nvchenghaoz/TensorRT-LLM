# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a DeepseekV41ForCausalLM config lands. Read this file and you know.

``route`` is one forward-reading decision: the facts that fix which code runs,
compared against the deployments the targets below were written for. When none
fits it raises ``NoTarget`` stating those facts. In ``auto`` the engine then
uses the built-in implementation (``modeling_deepseekv41``); in ``require`` it
raises that error.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

from ..._router_index import NoTarget, routes

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

# (layers, hidden, routed experts) of DeepSeek-V4.1-Flash. Shape-sniffing also
# matches fine-tunes of the same shape; the accuracy gate pins identity.
_SHAPE = (40, 5120, 384)


@routes("DeepseekV41ForCausalLM")
def route(
    model_config: ModelConfig, llm_args: TorchLlmArgs, compute_capability: Tuple[int, int]
) -> type:
    """Return the target class for this deployment, or raise ``NoTarget`` saying why none fits.

    Each target is imported where it is returned, so declining costs no import
    of modeling code.
    """
    pretrained_config = model_config.pretrained_config
    text_config = getattr(pretrained_config, "text_config", None) or pretrained_config
    mapping = model_config.mapping
    spec_config = model_config.spec_config
    shape = (
        getattr(text_config, "num_hidden_layers", None),
        getattr(text_config, "hidden_size", None),
        getattr(text_config, "n_routed_experts", None),
    )
    # tp4_moetp2ep2: attention heads sharded four ways; the routed experts split
    # into two expert-parallel groups, each sharded two ways over the expert
    # intermediate dimension. A weight-layout question, which is why it selects a
    # target rather than a runtime branch.
    tp4_moetp2ep2 = (
        mapping.world_size == 4
        and mapping.tp_size == 4
        and mapping.moe_tp_size == 2
        and mapping.moe_ep_size == 2
        and not mapping.enable_attention_dp
        and mapping.pp_size == 1
        and mapping.cp_size == 1
    )
    # DeepSeek-V4.1-Flash-1017 on four RTX PRO 6000 (sm120), without speculative
    # decoding: the decode kernels are tuned for one query row per sequence.
    if compute_capability == (12, 0) and shape == _SHAPE and tp4_moetp2ep2 and spec_config is None:
        # One line on purpose: the claims suite reads the routed targets off
        # `from .<target>.modeling import <Class>` lines.
        # fmt: off
        # isort: off
        from .v41_flash_1017__sm_120__tp4_moetp2ep2.modeling import ModelingV2DeepseekV41Flash1017Sm120Tp4Moetp2ep2
        # isort: on
        # fmt: on
        return ModelingV2DeepseekV41Flash1017Sm120Tp4Moetp2ep2

    spec = "off" if spec_config is None else spec_config.spec_dec_mode.name
    raise NoTarget(
        f"compute capability {compute_capability}, config shape {shape}, "
        f"world_size={mapping.world_size}, tp={mapping.tp_size}, "
        f"moe_tp={mapping.moe_tp_size}, moe_ep={mapping.moe_ep_size}, "
        f"attention_dp={mapping.enable_attention_dp}, pp={mapping.pp_size}, "
        f"cp={mapping.cp_size}, speculative decoding {spec}; the only target is "
        f"DeepSeek-V4.1-Flash {_SHAPE} on sm120 with tp4, moe_tp2 x moe_ep2, no attention DP "
        "and no speculative decoding"
    )
