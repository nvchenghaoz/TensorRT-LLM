# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the shared-prefix (longest-common-prefix) splice of PrefixTokenCache.

Every test compares the cache's ids with a fresh full tokenizer call on the same
text (the reference), using a real Qwen fast tokenizer.
"""

import random
import threading

import pytest
from utils.llm_data import llm_models_root

from tensorrt_llm.inputs import prefix_token_cache as ptc
from tensorrt_llm.inputs.prefix_token_cache import (
    PrefixTokenCache,
    PrefixTokenCacheConfig,
    tokenizer_adds_no_special_tokens,
)


@pytest.fixture(scope="module")
def tok():
    path = llm_models_root() / "Qwen3/Qwen3-0.6B"
    if not path.exists():
        pytest.skip(f"{path} is not available")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)


def ref(tok, text):
    return tok(text, add_special_tokens=False)["input_ids"]


def new_cache(**kw):
    return PrefixTokenCache(PrefixTokenCacheConfig(**kw))


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


CJK = (
    "的一是不了人我在有他这中大来上国个到说们为子和你地出道也时年得就那要下以生会自着去之过家学对可里后小"
    "么心多天而能好都然没日于起还发成事只作当想看文无开手十用主行方又如前所本见经头面公同三已老从动两长知"
    "民样现分将外但身些与高意进把法此实回二理美点月明其种声全工己话儿者向情部正名定女问力机给等几很业最间"
    "新什打便位因重被走电四第门相次东政海口使教西再平真听世气信北少关并内加化由却代军产入先山五太水万市眼"
    "体别处总才场师书比住员九笑性通目华报立马命张活难神数件安表原车白应路期叫死常提感金何更反合放做系计或"
    "司利受光王果亲界及今京务制解各任至清物台象记边共风战干接它许八特觉望直服毛林题建南度统色字请交爱让认"
    "算论百吃义科怎元社术结六功指思非流每青管夫连远资队跟带花快条院变联言权往展该领传近留红治决周保达办运"
    "武半候七必城父强步完革深区"
)


def cjk(rng, n_chars):
    return "".join(rng.choice(CJK) for _ in range(n_chars))


def assert_same(tok, cache, text):
    got = cache.encode(tok, text)
    want = ref(tok, text)
    first_diff = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), "length")
    assert got == want, f"mismatch at {first_diff}"
    return got


def test_lcp_splice_hits_and_is_exact(tok):
    rng = random.Random(0)
    prefix = english(rng, 60_000)
    cache = new_cache()
    a = prefix + " " + english(rng, 8000)
    b = prefix + " " + english(rng, 8000)
    assert_same(tok, cache, a)
    assert (cache.hits, cache.shared_prefix_hits, cache.misses) == (0, 0, 1)
    assert_same(tok, cache, b)
    assert (cache.hits, cache.shared_prefix_hits, cache.misses) == (0, 1, 1)
    # The new prompt is its own entry; the old one stays.
    assert len(cache._entries) == 2
    # Same prompt twice: a whole-prompt hit.
    assert_same(tok, cache, b)
    assert (cache.hits, cache.shared_prefix_hits, cache.misses) == (1, 1, 1)
    assert "2/3 hits" in cache.summary()


def test_lcp_split_is_at_space(tok):
    rng = random.Random(1)
    prefix = english(rng, 30_000)
    cache = new_cache()
    a = prefix + "alpha " + english(rng, 3000)
    cache.encode(tok, a)
    ((eid, entry),) = cache._entries.items()
    b = prefix + "beta " + english(rng, 3000)
    common = ptc._common_prefix_len(a, b)
    assert common == len(prefix)
    split = cache._split(entry, b, common, extends=False)
    assert split is not None
    tok_index, char = split
    assert tok_index % ptc.GRID_STRIDE == 0
    assert char <= common - ptc.LCP_MARGIN_CHARS
    assert ptc._is_space_split(b, char)
    assert_same(tok, cache, b)


def test_resync_failure_falls_back_to_full_encode(tok):
    rng = random.Random(2)
    prefix = english(rng, 30_000)

    class Perturbing:
        """Returns wrong ids for any text that starts with a space (the tail)."""

        def __call__(self, text, **kw):
            enc = tok(text, **kw)
            if text.startswith(" "):
                ids = list(enc["input_ids"])
                ids[0] = (ids[0] + 1) % 1000
                return {"input_ids": ids, "offset_mapping": enc["offset_mapping"]}
            return enc

    cache = new_cache()
    a = prefix + " " + english(rng, 5000)
    b = prefix + " " + english(rng, 5000)
    assert cache.encode(Perturbing(), a) == ref(tok, a)
    assert cache.encode(Perturbing(), b) == ref(tok, b)
    assert cache.shared_prefix_fallbacks == 1 and cache.resync_failures == 0
    assert cache.hits == 0 and cache.shared_prefix_hits == 0 and cache.misses == 2


@pytest.mark.parametrize(
    "pair",
    [
        ("extraordinary", "extraterrestrial"),
        ("internationalization", "international"),
        ("123456789", "123498765"),
        ("hello.\n\n", "hello. \n"),
    ],
)
def test_divergence_mid_word(tok, pair):
    rng = random.Random(3)
    prefix = english(rng, 20_000)
    cache = new_cache()
    for word in pair:
        assert_same(tok, cache, prefix + word + english(rng, 2000))
    assert cache.shared_prefix_hits == 1


def test_cjk_no_space_runs_at_seam(tok):
    rng = random.Random(4)
    prefix = english(rng, 10_000) + cjk(rng, 12_000)
    cache = new_cache()
    for _ in range(4):
        assert_same(tok, cache, prefix + cjk(rng, 3000))
    # Pure CJK prompt: no space to snap to anywhere.
    base = cjk(rng, 20_000)
    for _ in range(4):
        assert_same(tok, cache, base + cjk(rng, 1500))
    assert cache.hits + cache.shared_prefix_hits + cache.misses == 8


def test_special_tokens_near_seam(tok):
    rng = random.Random(5)
    turns = []
    for i in range(300):
        turns.append(
            f"<|im_start|>{'user' if i % 2 else 'assistant'}\n{english(rng, 120)}<|im_end|>\n"
        )
    prefix = "".join(turns)
    cache = new_cache()
    for _ in range(5):
        tail = "".join(
            f"<|im_start|>user\n{english(rng, rng.randrange(1, 80))}<|im_end|>\n"
            for _ in range(rng.randrange(1, 20))
        )
        assert_same(tok, cache, prefix + tail + "<|im_start|>assistant\n")
    assert cache.shared_prefix_hits >= 4


def test_sibling_entries_are_capped_per_bucket(tok):
    rng = random.Random(12)
    prefix = english(rng, 20_000)
    cache = new_cache()
    n = ptc.MAX_SHARED_PREFIX_SIBLINGS + 20
    prompts = [prefix + " " + english(rng, 1500) for _ in range(n)]
    for p in prompts:
        assert_same(tok, cache, p)
    assert (cache.misses, cache.shared_prefix_hits) == (1, n - 1)
    assert len(cache._entries) == ptc.MAX_SHARED_PREFIX_SIBLINGS
    # A cached sibling still extends to a whole-prompt hit; a late one splices.
    assert_same(tok, cache, prompts[5] + english(rng, 300))
    assert_same(tok, cache, prompts[-1] + english(rng, 300))
    assert cache.hits == 1 and cache.shared_prefix_hits == n


def test_text_shorter_than_cached_prompt(tok):
    rng = random.Random(6)
    long_text = english(rng, 40_000)
    cache = new_cache()
    assert_same(tok, cache, long_text)
    assert_same(tok, cache, long_text[:30_000])
    assert cache.shared_prefix_hits == 1


def test_short_common_prefix_is_a_miss(tok):
    rng = random.Random(7)
    head = english(rng, 5000)
    cache = new_cache()
    assert_same(tok, cache, head + english(rng, 20_000))
    assert_same(tok, cache, head + english(rng, 20_000))
    assert cache.hits == 0 and cache.shared_prefix_hits == 0 and cache.misses == 2


def test_multi_turn_extension_still_hits(tok):
    rng = random.Random(8)
    cache = new_cache()
    text = english(rng, 10_000)
    for _ in range(5):
        assert_same(tok, cache, text)
        text = text + english(rng, 500)
    assert cache.hits == 4 and cache.shared_prefix_hits == 0
    assert len(cache._entries) == 1


def test_concurrency(tok):
    rng = random.Random(9)
    prefix = english(rng, 40_000)
    prompts = [prefix + " " + english(rng, rng.randrange(1000, 6000)) for _ in range(48)]
    prompts += prompts[:8]  # repeats
    want = {p: ref(tok, p) for p in prompts}
    cache = new_cache()
    errors = []

    def worker(chunk):
        for p in chunk:
            if cache.encode(tok, p) != want[p]:
                errors.append(p[-50:])

    threads = [threading.Thread(target=worker, args=(prompts[i::8],)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert cache.hits + cache.shared_prefix_hits + cache.misses == len(prompts)
    assert not cache.disabled
    assert cache.hits + cache.shared_prefix_hits >= len(prompts) - 8


ALPHABET = (
    [" "] * 12
    + ["  ", "\n", "\n\n", "\t", " \n", "\r\n"]
    + list("abcdefghij")
    + ["word", "Word", "the", "'s", "'ll", "n't", "...", ",", ".", "!", "?", "(", ")", "-"]
    + list("0123456789")
    + ["2024", "3.14"]
    + ["中文", "日本語", "한국어"]
    + ["é", "́", "café", "naïve"]
    + ["😀", "👍🏽", "\u200b"]
    + ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]
)


def noise(rng, n_pieces):
    return "".join(rng.choice(ALPHABET) for _ in range(n_pieces))


@pytest.mark.parametrize("seed", range(6))
def test_fuzz_shared_prefixes(tok, seed):
    rng = random.Random(100 + seed)
    base = noise(rng, 9000)
    cache = new_cache()
    for _ in range(40):
        k = rng.randrange(len(base) // 2 + 700, len(base) + 1)
        text = base[:k] + noise(rng, rng.randrange(0, 1500))
        assert_same(tok, cache, text)
    assert cache.hits + cache.shared_prefix_hits > 0
    assert not cache.disabled


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
    for special in (True, False, True):
        p = prefix + " " + english(rng, 3000)
        ids, _ = proc({"prompt": p}, SamplingParams(add_special_tokens=special))
        assert ids == tok.encode(p, add_special_tokens=special)
    cache = proc._prefix_token_cache
    assert cache.shared_prefix_hits == 2 and cache.misses == 1
    # Truncation still bypasses the cache.
    ids, _ = proc({"prompt": prefix}, SamplingParams(truncate_prompt_tokens=100))
    assert len(ids) == 100
    assert cache.hits + cache.shared_prefix_hits + cache.misses == 3
