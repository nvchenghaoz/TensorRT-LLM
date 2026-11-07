# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint mapping, loading and sharding for this target.

The loaders live next to the modules whose parameters they fill (``_v4.py`` for the shared
DeepSeek-V4 layout, ``modeling.py`` for the V4.1 additions), exactly as the optimized build had
them; this module is the target's loading entry point.
"""

from ._v4 import DeepseekV4WeightLoader
from .modeling import DeepseekV41LoadCensus, DeepseekV41WeightLoader

__all__ = ["DeepseekV4WeightLoader", "DeepseekV41WeightLoader", "DeepseekV41LoadCensus"]
