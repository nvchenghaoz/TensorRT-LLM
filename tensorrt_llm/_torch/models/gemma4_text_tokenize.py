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
"""Exact fast tokenization of text-only Gemma4 prompts.

``Gemma4TextEncoder.encode(text)`` returns the same ids as
``tokenizer(text, return_tensors="pt")["input_ids"][0].tolist()`` without the tensor round trip.
"""

from typing import Any, List


class Gemma4TextEncoder:
    """Exact text-only encoder over a HF fast tokenizer (see module docstring)."""

    def __init__(self, tokenizer: Any):
        hf = getattr(tokenizer, "tokenizer", tokenizer)
        self._hf = hf
        self._backend = getattr(hf, "backend_tokenizer", None)

    def _encode_unsplit(self, text: str) -> List[int]:
        return self._backend.encode_batch([text], add_special_tokens=True)[0].ids

    def encode(self, text: str) -> List[int]:
        b = self._backend
        if b is None or b.truncation is not None or b.padding is not None:
            return list(self._hf(text)["input_ids"])
        return self._encode_unsplit(text)
