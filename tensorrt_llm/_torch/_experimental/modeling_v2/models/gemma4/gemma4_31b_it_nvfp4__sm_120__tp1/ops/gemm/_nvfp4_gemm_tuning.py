# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The target's NVFP4 GEMM: `trtllm::nvfp4_gemm`'s tactic choice plus three SM120 refinements.

`nvfp4_gemm` runs what the shared `trtllm::nvfp4_gemm` op runs (the AutoTuner's pick over the
allowed backends, op key ``trtllm::nvfp4_gemm::gemm``) except for the Gemma-4-31B MLP shapes:

* **Dense buckets** (``gemm_dense``). For (N, K) in `_DENSE_SHAPES` and M in its range, the
  tuner profiles at the dense grid point at or above M (`dense_bucket`) with L2 flushed before
  every timed repeat, over order-preserving tactics only (data-parallel CUTLASS and
  reduction-free cuBLASLt), instead of at the power of 2 below M with hot L2. A cuBLASLt pick
  runs the algorithm of the bucket's heuristic list (`CublasLtFP4BucketedRunner`).
* **A CUTLASS-only band** (``gemm_dense_dp``): down_proj at M 513-1023, where the standard
  bucket-512 pick (128x128 tiles, 1.79 waves at M=1000) is slower than 256x128 (one round).
* **Tactic pins.** Where the top tactics sit within 1-2% of each other the tuner's pick is a
  per-process lottery; a pin fixes the tactic of an M range by identity (a CUTLASS config
  index, or a cuBLASLt algorithm config looked up in the heuristic list the runner uses).

Every tactic these offer was found bitwise equal on real activations over the ranges listed
(the rsi-ascent census behind them), so they change speed, not output. Shapes and M outside
the tables keep the standard behavior exactly.
"""

from __future__ import annotations

import functools
from typing import Callable, List, Optional, Tuple

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 - registers torch.ops.trtllm.*
import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
from tensorrt_llm._torch.autotuner import (
    AutoTuner,
    ConstraintSpec,
    DistributedTuningStrategy,
    DynamicTensorSpec,
    OptimizationProfile,
    TuningConfig,
)
from tensorrt_llm._torch.cublaslt_utils import IS_CUBLASLT_AVAILABLE
from tensorrt_llm._torch.custom_ops.torch_custom_ops import (
    CublasLtFP4GemmRunner,
    FP4GemmRunner,
    NVFP4GemmUnifiedRunner,
    fp4_scale_infer_shape,
)
from tensorrt_llm._torch.utils import last_positive_power_of_2
from tensorrt_llm.logger import logger

# ------------------------------------------------------------------ dense buckets ---

#: (upper M, step): M in (previous upper, upper] maps up to the next multiple of step. Above the
#: last upper the standard power-of-2 rule applies.
DENSE_GRID = ((1024, 64), (2048, 128), (4096, 256), (8192, 512))
DENSE_MAX_M = DENSE_GRID[-1][0]


def dense_bucket(m: int, min_m: int) -> int:
    """The dense tuning bucket of ``m``: the smallest grid point >= m. M below ``min_m`` or above
    the grid keeps the standard rule (last power of 2)."""
    if m < min_m or m > DENSE_MAX_M:
        return last_positive_power_of_2(m)
    for upper, step in DENSE_GRID:
        if m <= upper:
            return max(min_m, -(-m // step) * step)
    return last_positive_power_of_2(m)


def dense_buckets(x: int, min_m: int) -> Tuple[int, ...]:
    """Every dense grid point from ``min_m`` up to the bucket of ``x``."""
    top = dense_bucket(min(x, DENSE_MAX_M), min_m)
    out = []
    lower = 0
    for upper, step in DENSE_GRID:
        b = -(-max(lower + 1, min_m) // step) * step
        while b <= upper:
            if min_m <= b <= top:
                out.append(b)
            b += step
        lower = upper
    return tuple(sorted(set(out)))


_COLD_L2_FLUSH_BYTES = 256 << 20

#: (N, K) -> (min M, max M) of the dense-bucket mode: gate_up N=43008 K=5376 at M 256-2048 (the
#: standard pick at M=300 is slower; above 2048 it is already the best), down N=5376 K=21504 at
#: M 4096-8192 (every tactic bitwise; M=8000 runs the 4096 pick otherwise).
_DENSE_SHAPES = {(43008, 5376): (256, 2048), (5376, 21504): (4096, 8192)}
#: (N, K) -> (min M, max M) of the CUTLASS-only band.
_DENSE_DP_SHAPES = {(5376, 21504): (513, 1023)}

OP_GEMM = "trtllm::nvfp4_gemm::gemm"
OP_DENSE = "trtllm::nvfp4_gemm::gemm_dense"
OP_DENSE_DP = "trtllm::nvfp4_gemm::gemm_dense_dp"


@functools.lru_cache(maxsize=None)
def dense_tuning_config(min_m: int, max_m: int) -> TuningConfig:
    return TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                0,
                0,
                functools.partial(_dense_gen, min_m=min_m, max_m=max_m),
                functools.partial(_dense_map, min_m=min_m),
            ),
        ),
        constraint_specs=(ConstraintSpec(2, 0, fp4_scale_infer_shape),),
        distributed_tuning_strategy=DistributedTuningStrategy.PARALLEL,
        cold_l2_flush_bytes=_COLD_L2_FLUSH_BYTES,
    )


def _dense_gen(x: int, min_m: int, max_m: int):
    return dense_buckets(min(x, max_m), min_m)


def _dense_map(m: int, min_m: int) -> int:
    return dense_bucket(m, min_m)


def dense_range(n: int, k: int) -> Optional[Tuple[int, int]]:
    """(min M, max M) of the dense-bucket mode for an (N, K) GEMM, else None."""
    return _DENSE_SHAPES.get((n, k))


def dense_dp_range(n: int, k: int) -> Optional[Tuple[int, int]]:
    """(min M, max M) of the CUTLASS-only band for an (N, K) GEMM, else None."""
    return _DENSE_DP_SHAPES.get((n, k))


# ------------------------------------------------------------------------- pins ---

_GATEUP_NK = (43008, 5376)
_DOWN_NK = (5376, 21504)
# cuBLASLt algorithm configs as get_algo_config()[2:] returns them:
# (algo_id, tile, stages, splitK, reduction, swizzle, custom, inner, cluster).
_LT_NVJET_256x128 = (67, 24, 36, 1, 0, 0, 31, 0, 0)  # nvjet ootst 256x128x128, data-parallel
_LT_CUTLASS3X_SK = (70, 20, 37, -2, 2, 0, 0, 0, 0)  # cutlass3x 128x128x256 stream-K
#: (N, K) -> [(op key, min M, max M, identity)]
_PINS = {
    _GATEUP_NK: [
        # dense bucket 320 (served M=300): nvjet 256x128x128, also on a dense cache miss and on
        # the standard (256) bucket.
        ("gemm_dense", 257, 320, ("cublaslt", _LT_NVJET_256x128)),
        ("gemm", 257, 320, ("cublaslt", _LT_NVJET_256x128)),
        # standard buckets 1024/2048/4096 (the chunked chain's lookup at M>=1664, and M>2048):
        # CUTLASS config 2 (256x128x128), which keeps the chunked chain on.
        ("gemm", 1024, 8191, ("cutlass", 2)),
    ],
    _DOWN_NK: [
        # standard bucket 1024: stream-K where it is fastest (served M 1200/1500), nvjet
        # 256x128x128 data-parallel above (served M 1800/2000).
        ("gemm", 1024, 1663, ("cublaslt", _LT_CUTLASS3X_SK)),
        ("gemm", 1664, 2047, ("cublaslt", _LT_NVJET_256x128)),
    ],
}
# (n, k, m, heuristic_m, config, dtype) -> heuristic-list index or None
_PIN_LT_INDEX: dict = {}


def _pin_entry(op: str, n: int, k: int, m: int) -> Optional[Tuple]:
    for p_op, lo, hi, ident in _PINS.get((n, k), ()):
        if p_op == op and lo <= m <= hi:
            return ident
    return None


def _pin_lt_index(
    inputs: List[torch.Tensor], output_dtype: torch.dtype, heuristic_m: int, cfg: Tuple
) -> Optional[int]:
    """Index of cuBLASLt config ``cfg`` in the heuristic list the runner uses at this M
    (heuristic_m > 0: that M's list, checked at the actual M)."""
    act_fp4, weight, act_sf, weight_scale, _ = inputs
    m, n, k = act_fp4.shape[0], weight.shape[0], act_fp4.shape[1] * 2
    key = (n, k, m, heuristic_m, cfg, output_dtype)
    if key in _PIN_LT_INDEX:
        return _PIN_LT_INDEX[key]
    runner = CublasLtFP4GemmRunner(0, output_dtype).cublaslt_runner
    found = None
    if heuristic_m <= 0 or heuristic_m == m:
        count = runner.get_num_heuristic_algos(act_fp4, weight, act_sf, weight_scale)
        hm = -1
    else:
        # The list size of heuristic_m (only the first operand's shape is read; no GEMM runs).
        shape_only = act_fp4.new_empty((heuristic_m, act_fp4.shape[1]))
        count = runner.get_num_heuristic_algos(shape_only, weight, act_sf, weight_scale)
        del shape_only
        hm = heuristic_m
    for i in range(count):
        c = runner.get_algo_config(act_fp4, weight, act_sf, weight_scale, i, hm)
        if not c[0]:
            break
        if not c[1] and tuple(c[2:]) == cfg:
            found = i
            break
    if found is None:
        logger.warning(
            f"NVFP4 tactic pin: cuBLASLt config {cfg} not in the heuristic list at m={m} "
            f"(list m={heuristic_m}, n={n}, k={k}); the tuned tactic runs instead"
        )
    _PIN_LT_INDEX[key] = found
    return found


def pinned_tactic(
    op: str,
    inputs: List[torch.Tensor],
    output_dtype: torch.dtype,
    tactic,
    allowed_backends: List[str],
    dense_min_m: Optional[int] = None,
):
    """The tactic to run for op key ``op`` ('gemm' or 'gemm_dense'): the pinned one for a pinned
    (N, K, M), else ``tactic``."""
    act_fp4, weight = inputs[0], inputs[1]
    m, n, k = act_fp4.shape[0], weight.shape[0], act_fp4.shape[1] * 2
    ident = _pin_entry(op, n, k, m)
    if ident is None:
        return tactic
    backend, val = ident
    if backend not in allowed_backends or (backend == "cublaslt" and not IS_CUBLASLT_AVAILABLE):
        return tactic
    if backend == "cutlass":
        return ("cutlass", val)
    heuristic_m = (
        dense_bucket(m, dense_min_m) if op == "gemm_dense" and dense_min_m is not None else -1
    )
    idx = _pin_lt_index(inputs, output_dtype, heuristic_m, val)
    if idx is None:
        return tactic
    return ("cublaslt", idx)


# ---------------------------------------------------------------------- runners ---


class CublasLtFP4BucketedRunner(CublasLtFP4GemmRunner):
    """cuBLASLt NVFP4 GEMM whose tactic indexes the heuristic list of the M it was tuned at.

    ``bucket_fn(m)`` gives that M; the algorithm is validated at the actual M and the actual
    M's top heuristic runs if it does not apply there. With ``bucket_fn=None`` this is the
    shared runner.
    """

    def __init__(
        self,
        output_buffer_kind: int,
        output_dtype: torch.dtype,
        group: Optional[List[int]] = None,
        bucket_fn: Optional[Callable[[int], int]] = None,
    ):
        super().__init__(output_buffer_kind, output_dtype, group=group)
        self.bucket_fn = bucket_fn

    def forward(self, inputs, tactic: int = -1, bias: Optional[torch.Tensor] = None):
        if self.bucket_fn is None or tactic is None or tactic < 0:
            return super().forward(inputs, tactic=tactic, bias=bias)
        mat1, mat2, mat1_scale, mat2_scale, alpha = inputs
        return self.cublaslt_runner.run_gemm_bucketed(
            mat1,
            mat2,
            mat1_scale,
            mat2_scale,
            alpha,
            self.output_buffer_kind,
            tactic,
            int(self.bucket_fn(mat1.shape[0])),
            self.group,
            bias,
        )

    def is_reduction_free(self, inputs: List[torch.Tensor], tactic: int) -> bool:
        """True when cuBLASLt tactic ``tactic`` at this M has no split-K and no reduction scheme,
        so each output is one K-ordered accumulation."""
        mat1, mat2, mat1_scale, mat2_scale, _ = inputs
        cfg = self.cublaslt_runner.get_algo_config(mat1, mat2, mat1_scale, mat2_scale, tactic, -1)
        # [has_algo, fallback, algo_id, tile, stages, splitK, reduction, ...]
        return bool(cfg[0]) and not cfg[1] and cfg[5] <= 1 and cfg[6] == 0


class Gemma4NVFP4GemmRunner(NVFP4GemmUnifiedRunner):
    """`NVFP4GemmUnifiedRunner` with a dense-bucket mode.

    With ``dense_min_m`` set, tactics are profiled at `dense_bucket(M, dense_min_m)`, only
    data-parallel CUTLASS and reduction-free cuBLASLt tactics are offered (CUTLASS only with
    ``dense_cutlass_only``), and a cuBLASLt tactic runs the algorithm of the bucket's heuristic
    list. Without it the runner is the shared one.
    """

    def __init__(
        self,
        output_buffer_kind: int,
        output_dtype: torch.dtype,
        allowed_backends: List[str],
        group: Optional[List[int]] = None,
        dense_min_m: Optional[int] = None,
        dense_cutlass_only: bool = False,
    ):
        super().__init__(output_buffer_kind, output_dtype, allowed_backends, group=group)
        self.dense_min_m = dense_min_m
        self.dense_cutlass_only = dense_cutlass_only
        self._dense_bucket_fn = (
            None if dense_min_m is None else (lambda m: dense_bucket(m, dense_min_m))
        )

    def get_valid_tactics(self, inputs: List[torch.Tensor], profile: OptimizationProfile, **kwargs):
        if self.dense_min_m is None:
            return super().get_valid_tactics(inputs, profile, **kwargs)
        tactics = []
        if self._is_backend_allowed("cutlass"):
            cutlass_runner = FP4GemmRunner(
                fp4_utils.FP4GemmType.W4A4_NVFP4_NVFP4,
                self.output_buffer_kind,
                self.output_dtype,
                group=self.group,
            )
            tactics.extend(("cutlass", t) for t in cutlass_runner.get_valid_tactics(inputs, profile))
        if self.dense_cutlass_only:
            return tactics
        if self._is_backend_allowed("cublaslt") and IS_CUBLASLT_AVAILABLE:
            lt = CublasLtFP4BucketedRunner(self.output_buffer_kind, self.output_dtype, group=self.group)
            for t in lt.get_valid_tactics(inputs, profile):
                if lt.is_reduction_free(inputs, t):
                    tactics.append(("cublaslt", t))
        return tactics

    def forward(self, inputs, tactic=-1, bias: Optional[torch.Tensor] = None, **kwargs):
        if tactic != -1 and tactic[0] == "cublaslt":
            return CublasLtFP4BucketedRunner(
                self.output_buffer_kind,
                self.output_dtype,
                group=self.group,
                bucket_fn=self._dense_bucket_fn,
            )(inputs, tactic=tactic[1], bias=bias)
        return super().forward(inputs, tactic=tactic, bias=bias, **kwargs)


# ------------------------------------------------------------------------ the GEMM ---


def nvfp4_gemm(
    act_fp4: torch.Tensor,
    weight: torch.Tensor,
    act_sf: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
    output_dtype: torch.dtype,
    allowed_backends: List[str],
) -> torch.Tensor:
    """out[M, N] = dequant(act) @ dequant(weight)^T * alpha with the tactic described above."""
    backends_list = list(allowed_backends)
    runner = Gemma4NVFP4GemmRunner(0, output_dtype, backends_list)
    tuner = AutoTuner.get()
    inputs = [act_fp4, weight, act_sf, weight_scale, alpha]
    n, k, m = weight.shape[0], act_fp4.shape[1] * 2, act_fp4.shape[0]
    tuning = tuner.is_tuning_mode

    dp_rng = dense_dp_range(n, k) if "cutlass" in backends_list else None
    dp_tuned = False

    def _dense_dp_choose():
        # The CUTLASS-only band is tuned after the standard and the dense entries, so those are
        # tuned exactly as they would be without it.
        nonlocal dp_tuned
        dp_tuned = True
        if tuner.is_tuning_mode:
            tuner.choose_one(OP_GEMM, [runner], NVFP4GemmUnifiedRunner.tuning_config, inputs)
        dp_runner = Gemma4NVFP4GemmRunner(
            0, output_dtype, backends_list, dense_min_m=dp_rng[0], dense_cutlass_only=True
        )
        _, dp_tactic = tuner.choose_one(
            OP_DENSE_DP, [dp_runner], dense_tuning_config(dp_rng[0], dp_rng[1]), inputs
        )
        return dp_runner, dp_tactic

    if dp_rng is not None and dp_rng[0] <= m <= dp_rng[1] and not tuning:
        dp_runner, dp_tactic = _dense_dp_choose()
        if dp_tactic != -1:
            return dp_runner(inputs=inputs, tactic=dp_tactic)

    rng = dense_range(n, k)
    if rng is not None:
        dense_min_m, dense_max_m = rng
        in_range = dense_min_m <= m <= dense_max_m
        if in_range or (tuning and m >= dense_min_m):
            if tuning:
                # Tune the standard entries first: M outside the dense range, and a dense cache
                # miss, run the standard pick.
                tuner.choose_one(OP_GEMM, [runner], NVFP4GemmUnifiedRunner.tuning_config, inputs)
            dense_runner = Gemma4NVFP4GemmRunner(
                0, output_dtype, backends_list, dense_min_m=dense_min_m
            )
            _, dense_tactic = tuner.choose_one(
                OP_DENSE, [dense_runner], dense_tuning_config(dense_min_m, dense_max_m), inputs
            )
            if in_range and not tuning:
                dense_tactic = pinned_tactic(
                    "gemm_dense", inputs, output_dtype, dense_tactic, backends_list, dense_min_m
                )
            if in_range and dense_tactic != -1:
                if dp_rng is not None and tuning and m >= dp_rng[0]:
                    _dense_dp_choose()
                return dense_runner(inputs=inputs, tactic=dense_tactic)

    if dp_rng is not None and tuning and not dp_tuned and m >= dp_rng[0]:
        dp_runner, dp_tactic = _dense_dp_choose()
        if dp_rng[0] <= m <= dp_rng[1] and dp_tactic != -1:
            return dp_runner(inputs=inputs, tactic=dp_tactic)

    _, best_tactic = tuner.choose_one(OP_GEMM, [runner], NVFP4GemmUnifiedRunner.tuning_config, inputs)
    if not tuning:
        best_tactic = pinned_tactic("gemm", inputs, output_dtype, best_tactic, backends_list)
    return runner(inputs=inputs, tactic=best_tactic)
