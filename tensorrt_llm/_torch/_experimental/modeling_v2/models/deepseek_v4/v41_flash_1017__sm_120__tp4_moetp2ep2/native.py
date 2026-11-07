# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""This target's native ops (csrc/), in torch.ops.mv2_deepseek_v4_v41_flash_1017_sm_120_tp4_moetp2ep2.

See NATIVE_OPS.md for the op list and the run call sites they replace.
"""

from tensorrt_llm._torch._experimental.modeling_v2._native import load_target_ops

# load_target_ops takes the target package's name (models.deepseek_v4.<target>), not this module's.
native_ops = load_target_ops(__name__.rpartition(".")[0])
