# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PrefixTokenCache serving ``add_special_tokens=True`` for tokenizers that add none.

Ids are compared with a fresh full tokenizer call, using a real Qwen fast tokenizer.
"""

import random

import pytest
from utils.llm_data import llm_models_root

from tensorrt_llm.inputs.prefix_token_cache import tokenizer_adds_no_special_tokens


@pytest.fixture(scope="module")
def tok():
    path = llm_models_root() / "Qwen3/Qwen3-0.6B"
    if not path.exists():
        pytest.skip(f"{path} is not available")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)


WORDS = (
    "the of and to in a is that for it as with was on be by at this which from or have an "
    "they not are his her had but were all their there one been has more who would will "
    "extraordinary extraterrestrial misunderstanding internationalization"
).split()


def english(rng, n_chars):
    out, size = [], 0
    while size < n_chars:
        w = rng.choice(WORDS)
        sep = rng.choice([" ", " ", " ", " ", ", ", ". ", "\n", "  ", "\n\n", " - "])
        out.append(w + sep)
        size += len(w) + len(sep)
    return "".join(out)


def test_eligibility_with_special_tokens(tok):
    assert tokenizer_adds_no_special_tokens(tok)

    class AddsBos:
        is_fast = True

        def num_special_tokens_to_add(self, pair=False):
            return 1

    assert not tokenizer_adds_no_special_tokens(AddsBos())
    assert not tokenizer_adds_no_special_tokens(object())
    # TRT-LLM's wrapper (its inherited num_special_tokens_to_add does not work).
    from tensorrt_llm.tokenizer import TransformersTokenizer

    assert tokenizer_adds_no_special_tokens(TransformersTokenizer(tok))


def test_default_input_processor_add_special_tokens(tok):
    from tensorrt_llm.inputs.registry import DefaultInputProcessor
    from tensorrt_llm.sampling_params import SamplingParams

    rng = random.Random(11)
    prefix = english(rng, 30_000)
    proc = DefaultInputProcessor(None, None, tok, enable_tokenization_cache=True)
    cache = proc._prefix_token_cache
    assert cache.serves_special_tokens
    # Prompts sharing a long document and ending differently, with and without
    # special tokens: the first is tokenized in full, the others are spliced.
    for special in (True, False, True):
        p = prefix + " " + english(rng, 3000)
        ids, _ = proc({"prompt": p}, SamplingParams(add_special_tokens=special))
        assert ids == tok.encode(p, add_special_tokens=special)
    assert cache.misses == 1 and cache.hits == 2
    # Truncation still bypasses the cache.
    ids, _ = proc({"prompt": prefix}, SamplingParams(truncate_prompt_tokens=100))
    assert len(ids) == 100
    assert cache.hits + cache.misses == 3
