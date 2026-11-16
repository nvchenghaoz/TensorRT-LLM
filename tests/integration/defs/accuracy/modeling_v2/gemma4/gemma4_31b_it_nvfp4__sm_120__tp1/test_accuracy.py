# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Whole-model gate for the gemma4 modeling_v2 target gemma4_31b_it_nvfp4__sm_120__tp1.

Gemma-4-31B-IT-NVFP4 on sm_120, one rank (tp1), served the way the target was tuned: an fp8 KV
pool and breakable prefill CUDA graphs at the prompt sizes it targets.

The test passes ``modeling_v2="require"``: under ``"auto"`` a configuration that missed the
target's routing would fall back to the built-in model, pass, and report the built-in's numbers
as the target's.
"""

import pytest

from tensorrt_llm import LLM
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

from .....conftest import llm_models_root
from ....accuracy_core import GSM8K, LlmapiAccuracyTestHarness
from ..._env import scoped_env_overrides

skip_not_sm120 = pytest.mark.skipif(
    get_sm_version() != 120, reason="the gemma4 target is certified on sm_120 only"
)


@skip_not_sm120
class TestModelingV2Gemma431BItNvfp4Sm120Tp1(LlmapiAccuracyTestHarness):
    """gemma-4-31b-it-nvfp4 / sm_120 / tp1, text."""

    # References are keyed by model name plus the quantization the engine resolves for this
    # checkpoint (NVFP4) and the KV-cache algorithm (FP8).
    MODEL_NAME = "google/gemma-4-31B-it"
    MODEL_PATH = f"{llm_models_root()}/gemma/nvidia-Gemma-4-31B-IT-NVFP4"

    EXTRA_EVALUATOR_KWARGS = dict(chat_template_kwargs=dict(enable_thinking=False))

    ENGINE = dict(
        max_batch_size=128,
        max_num_tokens=8192,
        kv_cache_config=KvCacheConfig(
            dtype="fp8", enable_block_reuse=True, free_gpu_memory_fraction=0.8
        ),
        cuda_graph_config=CudaGraphConfig(max_batch_size=128),
        prefill_cuda_graph_backend="breakable",
        prefill_capture_num_tokens=[300, 600, 900, 1000, 1200, 1500, 1800, 2000],
        disable_mm_encoder=True,
    )

    def test_gsm8k(self):
        with (
            scoped_env_overrides({}) as rank_env,
            LLM(self.MODEL_PATH, **self.ENGINE, modeling_v2="require", env_overrides=rank_env) as llm,
        ):
            GSM8K(self.MODEL_NAME).evaluate(llm, extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)
