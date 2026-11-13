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
ATTENTION_SAGE_ONSET_STATE_KEY = "attention_sage_onset"


class DenoiseStepController:
    """Holds the index of the current denoising forward of one transformer."""

    def __init__(self) -> None:
        self.step_index: Optional[int] = None
        self.num_steps: Optional[int] = None
        self.warmup = False
        # Gated attention calls in the current forward (host counter, logging only):
        # one per transformer block that ran, so a step cache shows as fewer calls.
        self.attention_calls = 0
        # Bumped on every set_step/reset: per-forward state keyed on it never crosses a forward.
        self.generation = 0
        # Callables run at every reset (start and end of denoising), e.g. per-request logging.
        self.reset_hooks = []

    def set_step(self, step_index: int, num_steps: int, warmup: bool = False) -> None:
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        if step_index < 0 or step_index >= num_steps:
            raise IndexError(f"step_index must be in [0, {num_steps}), got {step_index}")
        self.step_index = step_index
        self.num_steps = num_steps
        self.warmup = bool(warmup)
        self.attention_calls = 0
        self.generation += 1

    def reset(self) -> None:
        self.step_index = None
        self.num_steps = None
        self.warmup = False
        self.attention_calls = 0
        self.generation += 1
        for hook in self.reset_hooks:
            hook()

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

    def runs_sage(self, sage_onset: Optional[int], bf16_forwards: Optional[int]) -> bool:
        """Whether a gated BF16 call runs the Sage2 middle tier instead, on this same index.

        True only on denoising forwards in ``[sage_onset, bf16_forwards)``; never on the
        startup warmup (which runs the low-precision recipe) or outside a denoising loop
        (the text refiner stays BF16). ``sage_onset`` None leaves the BF16 phase unchanged.
        """
        if sage_onset is None or not self.runs_bf16(bf16_forwards):
            return False
        if self.step_index is None:
            return False
        return self.step_index >= sage_onset
