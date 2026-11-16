# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Package marker.

Some test basenames here repeat the catalog's (gemm/test_modeling_v2_nvfp4_gemm.py). Without
packages, pytest's default import mode loads both under one module name and refuses the second.
"""
