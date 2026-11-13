# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-denoising-step precision gate for MiniMax-H3.

Quantized paths (FP8 attention today; quantized GEMMs or collectives later) lose
fidelity when they run on the early denoising forwards, which fix the subject,
layout and framing, and keep it when those forwards run BF16. The pipeline sets
the index of the current denoising forward on one controller per transformer,
before the transformer call, and resets it at the start and end of denoising.
Each step-gated consumer compares that one index with its own count of BF16
forwards (``visual_gen_args.step_precision``).

The controller is host state read only inside ``torch.compiler.disable``'d code,
so changing the step never recompiles anything.
"""

from typing import Optional

# Keys in the transformer's attention_metadata_state, where attention backends find
# the controller and attention's own BF16 forward count.
STEP_PRECISION_STATE_KEY = "step_precision_controller"
ATTENTION_BF16_FORWARDS_STATE_KEY = "attention_bf16_forwards"


class DenoiseStepController:
    """Holds the index of the current denoising forward of one transformer."""

    def __init__(self) -> None:
        self.step_index: Optional[int] = None
        self.num_steps: Optional[int] = None
        self.warmup = False

    def set_step(self, step_index: int, num_steps: int, warmup: bool = False) -> None:
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        if step_index < 0 or step_index >= num_steps:
            raise IndexError(f"step_index must be in [0, {num_steps}), got {step_index}")
        self.step_index = step_index
        self.num_steps = num_steps
        self.warmup = bool(warmup)

    def reset(self) -> None:
        self.step_index = None
        self.num_steps = None
        self.warmup = False

    def runs_bf16(self, bf16_forwards: Optional[int]) -> bool:
        """Whether a consumer with ``bf16_forwards`` leading BF16 forwards runs BF16 now.

        ``None`` or 0 leaves the consumer on its low-precision recipe for every call.
        Otherwise the denoising forwards of the startup warmup run the low-precision
        recipe (consumers also exercise their BF16 path there), and calls outside a
        denoising loop (the once-per-request text refiner) run BF16.
        """
        if not bf16_forwards:
            return False
        if self.warmup:
            return False
        if self.step_index is None:
            return True
        return self.step_index < bf16_forwards
