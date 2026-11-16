#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#

# gemma4 native code. Each target keeps its own in <target>/csrc/; see the
# modeling_v2 README.
add_modeling_v2_family(gemma4)

# The SM120 sequential-K BF16 GEMM of the attention projections (TMA operand
# loads, mma.sync, one rounding per output). It looks cuTensorMapEncodeTiled up
# through the runtime's driver entry point, so it needs no driver link.
add_modeling_v2_target(
  gemma4 gemma4_31b_it_nvfp4__sm_120__tp1
  ARCHS 120f
  SOURCES *.cu *.cpp)
