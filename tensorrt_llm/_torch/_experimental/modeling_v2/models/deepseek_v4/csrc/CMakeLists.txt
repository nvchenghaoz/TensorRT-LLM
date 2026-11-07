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

# deepseek_v4 native code. Each target keeps its own in <target>/csrc/; see the
# modeling_v2 README.
add_modeling_v2_family(deepseek_v4)

# DeepSeek-V4.1-Flash-1017 on RTX PRO 6000 (SM120), TP4 attention with MoE
# TP2 x EP2. Its kernels guard on __CUDA_ARCH__ == 1200 (sm_120 family).
add_modeling_v2_target(
  deepseek_v4 v41_flash_1017__sm_120__tp4_moetp2ep2
  ARCHS 120f
  SOURCES *.cu *.cpp)
