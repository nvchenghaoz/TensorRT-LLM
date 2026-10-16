# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Prefix-tokenization cache for ``DefaultInputProcessor``.

In multi-turn serving each prompt is usually the previous prompt plus a small
delta, so re-tokenizing the whole prompt every turn is wasted work. This cache
remembers the token ids of recent prompts and, when a new prompt extends a
cached one, tokenizes only the tail and splices it onto the cached ids.

Prompts that share a long prefix with a cached prompt but end differently (a
shared system prompt or document followed by different questions) are spliced
too: the cache finds the recent entry with the longest common text prefix,
splits a safe distance before the point where the texts diverge, and
re-tokenizes from there.

Tokenizing a tail in isolation is not generally identical to tokenizing the
whole prompt, because BPE merges can straddle the split. The cache therefore
backs off ``overlap`` tokens from the end of the cached prefix, re-tokenizes
from there, and accepts the splice only if the first ``resync`` re-tokenized ids
equal the cached ids over the same span. Otherwise the prompt is tokenized in
full. A shared-prefix split is placed, when one is near, at a token that starts
with a space following a non-space character: for byte-level BPE tokenizers whose
pre-tokenizer regex has no lookbehind (GPT-2, Llama-3, Qwen), every pre-token
match ends before such a space, so the tail tokenizes exactly as it does inside
the whole prompt.

Off by default; enable with the ``enable_tokenization_cache`` LLM argument.
Each ``DefaultInputProcessor`` owns its own cache, so cached ids are never
shared across tokenizers. Sizing knobs, eligibility rules, and measured effect
are documented in ``docs/source/features/prefix-tokenization-cache.md``.
"""

from __future__ import annotations

import os
import threading
from array import array
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from typing import ClassVar, Mapping, Protocol, Sequence

from ..logger import logger

__all__ = [
    "PrefixTokenCache",
    "PrefixTokenCacheConfig",
    "create_prefix_token_cache",
    "tokenizer_adds_no_special_tokens",
]

# Log the hit rate every this many cache-eligible requests, so an operator can
# see whether the cache is doing anything without reading its counters.
LOG_EVERY_REQUESTS = 1000
# Each entry records the start character of every GRID_STRIDE-th token, which
# is where a shared-prefix splice can split. Denser costs memory, sparser makes
# the re-tokenized tail longer.
GRID_STRIDE = 16
# Only the most recently inserted entries of a bucket are compared for the
# longest common prefix, so a lookup stays cheap however full the bucket is.
LCP_CANDIDATES = 8
# Characters compared per slice when measuring a common prefix.
LCP_BLOCK_CHARS = 8192
# A shared-prefix split is at least this many characters before the point where
# the texts diverge, so that the resync span lies inside the common prefix.
LCP_MARGIN_CHARS = 512
# Grid points searched backwards for a split at a space.
SNAP_SEARCH = 64
# A prompt spliced at a prefix shared with a cached prompt becomes an entry of
# its own (so its own later turns extend it) only while its bucket holds fewer
# entries than this. Past that the matched entry keeps serving the bucket and
# nothing is inserted, so a workload of many prompts sharing one long prefix
# keeps a small bucket and a constant-time lookup.
MAX_SHARED_PREFIX_SIBLINGS = 32
# Agentic prompts share a system/tool preamble that can run to tens of thousands
# of characters, so entries in one bucket cannot be told apart by their start,
# and they end with the same chat-template suffix, so CPython's own last-character
# shortcut in str.startswith does not help either. Each entry keeps its last
# PROBE_CHARS characters; a candidate is only compared in full when the incoming
# text has that probe at the same position. The probe must be longer than any
# suffix the chat template appends after the last message.
PROBE_CHARS = 256


class OffsetTokenizer(Protocol):
    """The subset of the HF fast-tokenizer call interface the cache relies on."""

    def __call__(
        self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool
    ) -> Mapping[str, Sequence]: ...


@dataclass(frozen=True)
class PrefixTokenCacheConfig:
    """Tunables for :class:`PrefixTokenCache`.

    The sizing fields have env overrides (see ``ENV_VARS``); ``overlap`` and
    ``resync`` are correctness internals and do not.
    """

    max_entries: int = 512
    # Total characters of cached prompt text. The cached ids are int32, about
    # one byte per character of English text, so this bounds host memory to
    # roughly twice this many bytes.
    max_total_chars: int = 64 * 1024 * 1024
    # Prompts shorter than this are tokenized normally and never cached. Also
    # the number of leading characters entries are bucketed by, so a lookup does
    # not scan every entry.
    min_chars: int = 4096
    # Tokens to back off from the end of the cached prefix before re-tokenizing.
    overlap: int = 64
    # Re-tokenized ids that must equal the cached ids for the splice to be used.
    resync: int = 32

    ENV_VARS: ClassVar[dict[str, str]] = {
        "max_entries": "TLLM_PREFIX_TOKEN_CACHE_ENTRIES",
        "max_total_chars": "TLLM_PREFIX_TOKEN_CACHE_MAX_CHARS",
        "min_chars": "TLLM_PREFIX_TOKEN_CACHE_MIN_CHARS",
    }

    @classmethod
    def from_env(cls) -> PrefixTokenCacheConfig:
        """Build a config from the environment.

        Raises:
            ValueError: if an override is not a positive integer.
        """
        overrides: dict[str, int] = {}
        for field_name, env_var in cls.ENV_VARS.items():
            raw = os.environ.get(env_var)
            if raw is None:
                continue
            if not raw.isdigit() or int(raw) == 0:
                raise ValueError(f"{env_var} must be a positive integer, got {raw!r}")
            overrides[field_name] = int(raw)
        return cls(**overrides)


@dataclass(slots=True, frozen=True)
class _Entry:
    text: str
    ids: array  # int32
    # Every prompt that extends ``text`` is re-tokenized from the same place:
    # token ``split_token``, which starts at character ``split_char``.
    split_token: int
    split_char: int
    probe: str  # the last PROBE_CHARS characters of ``text``
    # grid[g] is the start character of token g * GRID_STRIDE (int32).
    grid: array


def _common_prefix_len(a: str, b: str) -> int:
    """Length of the longest common prefix of ``a`` and ``b``.

    Compares slices (memcmp in C), never characters in a Python loop.
    """
    n = min(len(a), len(b))
    i = 0
    while i < n:
        j = min(i + LCP_BLOCK_CHARS, n)
        if a[i:j] != b[i:j]:
            # a[:i] == b[:i] and the first difference is in [i, j).
            lo, hi = i, j
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if a[lo:mid] == b[lo:mid]:
                    lo = mid
                else:
                    hi = mid
            return lo
        i = j
    return n


def _is_space_split(text: str, char: int) -> bool:
    """Whether every pre-token match of ``text`` ends at ``char``.

    True when ``text[char]`` is a space and the character before it is not
    whitespace: GPT-2-style pre-tokenizer patterns only put a space at the start
    of a match or inside an all-whitespace match.
    """
    return 0 < char < len(text) and text[char] == " " and not text[char - 1].isspace()


class PrefixTokenCache:
    """Splice a cached prefix's token ids with a freshly tokenized tail.

    Thread-safe. Eviction is least-recently-used, bounded by both entry count
    and total cached characters. When a prompt extends a cached entry, the new
    prompt replaces that entry rather than sitting alongside it, so a
    conversation costs one entry however many turns it has. A prompt that only
    shares a prefix with a cached entry is added as an entry of its own while
    its bucket holds fewer than ``MAX_SHARED_PREFIX_SIBLINGS`` entries.

    An unexpected error disables the cache, with a warning, and every prompt is
    tokenized in full from then on: a cache bug must never fail a request.
    """

    def __init__(self, config: PrefixTokenCacheConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._entries: OrderedDict[int, _Entry] = OrderedDict()  # LRU order
        self._buckets: dict[int, dict[int, _Entry]] = {}
        self._total_chars = 0
        self._next_id = 0
        # Read and written without the lock: a bool flip is atomic, and the
        # worst case of a race is two threads each logging the disable warning.
        self.disabled = False
        # Counters, reported by periodic log lines and read by tests. ``hits``
        # extended a whole cached prompt; ``shared_prefix_hits`` spliced at a
        # prefix shared with a cached prompt; ``misses`` were tokenized in full,
        # ``resync_failures`` and ``shared_prefix_fallbacks`` of them after a
        # failed splice of either kind.
        self.hits = 0
        self.misses = 0
        self.resync_failures = 0
        self.shared_prefix_hits = 0
        self.shared_prefix_fallbacks = 0

    def encode(self, tokenizer: OffsetTokenizer, text: str) -> list[int]:
        """Return the token ids of ``text`` without special tokens."""
        if self.disabled or len(text) < self._config.min_chars:
            return self._tokenize(tokenizer, text)[0]
        try:
            return self._encode(tokenizer, text)
        except Exception as e:
            self.disabled = True
            logger.warning(
                f"Disabling the prefix token cache after an error: {e!r} ({self.summary()})"
            )
            return self._tokenize(tokenizer, text)[0]

    def summary(self) -> str:
        """One-line hit-rate and occupancy summary. Caller need not hold the lock."""
        hits = self.hits + self.shared_prefix_hits
        total = hits + self.misses
        rate = hits / total if total else 0.0
        return (
            f"{hits}/{total} hits ({rate:.1%}), "
            f"{self.resync_failures + self.shared_prefix_fallbacks} resync fallbacks, "
            f"{len(self._entries)} entries, {self._total_chars} cached chars, "
            f"{self.shared_prefix_hits} hits and {self.shared_prefix_fallbacks} fallbacks "
            f"at a shared prefix"
        )

    def _encode(self, tokenizer: OffsetTokenizer, text: str) -> list[int]:
        found = self._lookup(text)
        ids = None
        attempted = False
        # True when ``text`` extends the whole cached prompt rather than only
        # sharing a prefix with it.
        extends = found is not None and found[2] == len(found[1].text)
        if found is not None:
            eid, entry, common = found
            split = self._split(entry, text, common, extends)
            if split is not None:
                attempted = True
                split_token, split_char = split
                tail_ids, tail_offsets = self._tokenize(tokenizer, text[split_char:])
                # A shared-prefix splice only compares ids inside the common
                # prefix, and needs a full resync span there.
                limit, required = (
                    (None, 1) if extends else (common - split_char, self._config.resync)
                )
                if self._resynced(entry, split_token, tail_ids, tail_offsets, limit, required):
                    ids = entry.ids[:split_token]
                    ids.extend(tail_ids)
                    base_token, base_char = split_token, split_char
        if ids is None:
            tail_ids, tail_offsets = self._tokenize(tokenizer, text)
            ids, base_token, base_char = array("i", tail_ids), 0, 0

        # Where a prompt extending this one will be re-tokenized from.
        split_token = len(ids) - self._config.overlap
        tail_index = split_token - base_token
        # A prompt that only shares a prefix with the entry it was spliced from
        # is kept, for its own later turns, only while its bucket is small;
        # otherwise the matched entry keeps serving the bucket.
        store = split_token > 0 and tail_index >= 0
        if store and base_token > 0 and not extends:
            with self._lock:
                bucket = self._buckets.get(self._bucket_key(text), ())
                store = len(bucket) < MAX_SHARED_PREFIX_SIBLINGS
        grid = None
        if store and base_token == 0:
            grid = array("i", [offset[0] for offset in tail_offsets[::GRID_STRIDE]])
        elif store:
            # Grid points before the split are shared with the entry.
            first = -(-base_token // GRID_STRIDE)
            grid = entry.grid[:first]
            grid.extend(
                base_char + offset[0]
                for offset in tail_offsets[first * GRID_STRIDE - base_token :: GRID_STRIDE]
            )

        with self._lock:
            if base_token == 0:
                self.misses += 1
                if attempted and extends:
                    self.resync_failures += 1
                elif attempted:
                    self.shared_prefix_fallbacks += 1
            elif extends:
                self.hits += 1
            else:
                self.shared_prefix_hits += 1
            if extends:
                self._remove(found[0])
            if grid is not None:
                self._insert(
                    text, ids, split_token, base_char + tail_offsets[tail_index][0], grid
                )
            requests = self.hits + self.shared_prefix_hits + self.misses
            report = requests % LOG_EVERY_REQUESTS == 0
            summary = self.summary() if report else None
        if summary is not None:
            logger.info(f"Prefix token cache: {summary}")
        return tail_ids if base_token == 0 else ids.tolist()

    @staticmethod
    def _tokenize(
        tokenizer: OffsetTokenizer, text: str
    ) -> tuple[list[int], Sequence[tuple[int, int]]]:
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        return enc["input_ids"], enc["offset_mapping"]

    def _split(
        self, entry: _Entry, text: str, common: int, extends: bool
    ) -> tuple[int, int] | None:
        """Token and character to re-tokenize ``text`` from, or None.

        ``common`` is the length of the common prefix of ``text`` and the entry.
        """
        if extends:
            return entry.split_token, entry.split_char
        limit = common - LCP_MARGIN_CHARS
        grid = entry.grid
        last = bisect_right(grid, limit) - 1
        fallback = None
        for g in range(last, max(last - SNAP_SEARCH, 0), -1):
            char = grid[g]
            if char > limit:
                continue
            if _is_space_split(text, char):
                return g * GRID_STRIDE, char
            if fallback is None:
                fallback = g * GRID_STRIDE, char
        # No space nearby (e.g. CJK text): split anywhere and rely on resync.
        return fallback

    def _resynced(
        self,
        entry: _Entry,
        split_token: int,
        tail_ids: Sequence[int],
        tail_offsets: Sequence[tuple[int, int]],
        limit: int | None,
        required: int,
    ) -> bool:
        """Whether the tokenizer re-synchronized with the cached ids at the seam.

        Compares up to ``resync`` re-tokenized ids with the cached ids from
        ``split_token``, only over tail tokens that end within ``limit``
        characters of the split (all of them if ``limit`` is None), and needs at
        least ``required`` of them.
        """
        cached = entry.ids
        span = min(self._config.resync, len(tail_ids), len(cached) - split_token)
        if limit is not None:
            while span > 0 and tail_offsets[span - 1][1] > limit:
                span -= 1
        return (
            span >= required
            and list(tail_ids[:span]) == cached[split_token : split_token + span].tolist()
        )

    def _bucket_key(self, text: str) -> int:
        return hash(text[: self._config.min_chars])

    def _lookup(self, text: str) -> tuple[int, _Entry, int] | None:
        """Cached entry to splice ``text`` from, promoted to MRU.

        Returns ``(id, entry, common)``, where ``common`` is the length of the
        common prefix of ``text`` and the entry. The longest entry that is a
        whole prefix of ``text`` wins (``common == len(entry.text)``); otherwise
        the recent entry sharing the longest prefix with ``text``, if that prefix
        leaves at least half of ``text`` (and ``min_chars``) to reuse.
        """
        key = self._bucket_key(text)
        with self._lock:
            candidates = list(self._buckets.get(key, {}).items())
        # Entries are immutable, so the comparisons run outside the lock. The
        # probe check rejects other conversations in O(PROBE_CHARS); only a true
        # prefix pays the full compare over the shared preamble.
        best: tuple[int, _Entry] | None = None
        for eid, entry in candidates:
            n = len(entry.text)
            if (
                n <= len(text)
                and (best is None or n > len(best[1].text))
                and text[n - len(entry.probe) : n] == entry.probe
                and text.startswith(entry.text)
            ):
                best = (eid, entry)
        common = len(best[1].text) if best is not None else 0
        if best is None:
            # Prompts sharing a long prefix but ending differently. All entries
            # in a bucket share their first min_chars characters, so only the
            # most recently used ones are measured.
            floor = max(self._config.min_chars, len(text) // 2) + LCP_MARGIN_CHARS
            for eid, entry in reversed(candidates[-LCP_CANDIDATES:]):
                n = _common_prefix_len(text, entry.text)
                if n >= floor and n > common:
                    best, common = (eid, entry), n
        if best is not None:
            with self._lock:
                if best[0] in self._entries:
                    self._entries.move_to_end(best[0])
                    # Buckets are kept in use order too.
                    bucket = self._buckets.get(key)
                    if bucket is not None and best[0] in bucket:
                        bucket[best[0]] = bucket.pop(best[0])
            # If it was evicted meanwhile the entry is still a valid prefix to
            # splice from; _remove() on it is a no-op.
            return best[0], best[1], common
        return None

    def _insert(
        self, text: str, ids: array, split_token: int, split_char: int, grid: array
    ) -> None:
        """Caller holds the lock."""
        key = self._bucket_key(text)
        bucket = self._buckets.setdefault(key, {})
        if any(entry.text == text for entry in bucket.values()):
            return  # concurrent misses on the same prompt
        eid, self._next_id = self._next_id, self._next_id + 1
        entry = _Entry(text, ids, split_token, split_char, text[-PROBE_CHARS:], grid)
        self._entries[eid] = entry
        bucket[eid] = entry
        self._total_chars += len(text)
        while self._entries and (
            len(self._entries) > self._config.max_entries
            or self._total_chars > self._config.max_total_chars
        ):
            self._remove(next(iter(self._entries)))

    def _remove(self, eid: int) -> None:
        """Caller holds the lock. A no-op if another thread already removed it."""
        entry = self._entries.pop(eid, None)
        if entry is None:
            return
        self._total_chars -= len(entry.text)
        key = self._bucket_key(entry.text)
        bucket = self._buckets[key]
        bucket.pop(eid, None)
        if not bucket:
            del self._buckets[key]


def create_prefix_token_cache(tokenizer: object) -> PrefixTokenCache | None:
    """Return a cache for ``tokenizer`` if the tokenizer can support one.

    Returns None when there is no tokenizer or it cannot report offsets.
    Raises ``ValueError`` for a malformed env override.
    """
    if tokenizer is None:
        return None
    if not getattr(tokenizer, "is_fast", False):
        logger.warning(
            "enable_tokenization_cache is set but the tokenizer is not a fast "
            "tokenizer, which the prefix token cache needs for offset mappings; "
            "disabling it."
        )
        return None
    config = PrefixTokenCacheConfig.from_env()
    logger.info(f"Prefix token cache enabled: {config}")
    return PrefixTokenCache(config)


def tokenizer_adds_no_special_tokens(tokenizer: object) -> bool:
    """Whether ``add_special_tokens=True`` gives the same ids as False for ``tokenizer``.

    True when the tokenizer's post-processor adds no special tokens to a single
    sequence (Qwen, GPT-2 style), checked on a probe string as well. The prefix
    token cache tokenizes with ``add_special_tokens=False``, so for such a
    tokenizer it can also serve requests that ask for special tokens, such as
    ``/v1/completions`` by default. Evaluated once, when a processor is built.
    """
    # A wrapper such as TransformersTokenizer delegates to ``.tokenizer``; ask
    # the wrapped tokenizer first, since the wrapper's inherited
    # num_special_tokens_to_add need not work.
    inner = getattr(tokenizer, "tokenizer", None)
    for candidate in (inner, tokenizer):
        count = getattr(candidate, "num_special_tokens_to_add", None)
        if candidate is None or not callable(count):
            continue
        try:
            if count() != 0:
                return False
            probe = "Hello world, tokenization probe."
            with_special = candidate(probe, add_special_tokens=True)["input_ids"]
            without = candidate(probe, add_special_tokens=False)["input_ids"]
        except Exception as e:  # noqa: BLE001 - any failure means "not known safe"
            logger.debug(f"Special-token probe failed on {type(candidate).__name__}: {e!r}")
            continue
        return list(with_special) == list(without)
    return False
