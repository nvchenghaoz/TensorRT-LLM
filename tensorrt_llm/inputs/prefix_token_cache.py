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

The same kind of cut makes the tokenization of a long tail (and of a missed
prompt) parallel: for a tokenizer of that proven shape, the text is cut at such
spaces into pieces that go to the tokenizer's backend in one ``encode_batch``
call, which runs them in parallel with the GIL released, and the pieces' ids and
shifted offsets are concatenated. ``TLLM_PREFIX_TOKEN_CACHE_PARALLEL=0`` turns
this off.

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
# A long tail (or a missed prompt) is cut, at spaces that follow a non-space,
# into pieces tokenized by one backend encode_batch call, which a fast
# tokenizer runs in parallel without the GIL. The cut is a pre-token boundary
# (see the module docstring and _space_cut_is_exact), so the ids and offsets are
# the ones a single call returns. Tests and benchmarks flip the attribute to
# tokenize every text in one call.
PARALLEL_TAIL = True
# Pieces are at least this many characters. A text gets at most
# PARALLEL_MAX_PIECES // (texts being tokenized at once) pieces, so concurrent
# requests share the cores instead of oversubscribing them.
PARALLEL_PIECE_CHARS = 1024
PARALLEL_MAX_PIECES = 8
# Characters searched past a piece's nominal start for a cut.
PARALLEL_CUT_SEARCH = 256
# Pre-tokenizer Split patterns, as stored in tokenizer.json, for which a space
# that follows a non-space is a pre-token boundary: no lookbehind, and a space
# only starts a match or sits in an all-whitespace match.
_CUT_SAFE_SPLIT_PATTERNS = frozenset(
    {
        # Qwen3 family.
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}"
        r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        # Qwen2 / GPT-4 (cl100k) style.
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
        r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    }
)


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


def _is_cut(text: str, char: int) -> bool:
    """A stricter ``_is_space_split``.

    ``text[char]`` is U+0020 and the character before it is a letter, mark,
    number, punctuation or symbol.

    Those categories are never whitespace in any Unicode version or regex
    engine, so no whitespace run of the pre-tokenizer reaches across the cut.
    """
    if not 0 < char < len(text) or text[char] != " ":
        return False
    prev = text[char - 1]
    return prev != " " and prev.isprintable()


def _space_cuts(text: str, pieces: int) -> list[int]:
    """Start characters of up to ``pieces`` pieces of ``text``, the first 0.

    Every other start is an ``_is_cut`` point within ``PARALLEL_CUT_SEARCH``
    characters after an equal-length cut; a cut with no such point nearby is
    skipped.
    """
    n = len(text)
    cuts = [0]
    for j in range(1, pieces):
        start = max(j * n // pieces, cuts[-1] + 1)
        end = min(start + PARALLEL_CUT_SEARCH, n)
        char = text.find(" ", start, end)
        while char != -1 and not _is_cut(text, char):
            char = text.find(" ", char + 1, end)
        if char != -1:
            cuts.append(char)
    return cuts


def _backend_of(tokenizer: object) -> object | None:
    """The ``tokenizers.Tokenizer`` behind a HF fast tokenizer or TRT-LLM's wrapper."""
    for candidate in (tokenizer, getattr(tokenizer, "tokenizer", None)):
        backend = getattr(candidate, "backend_tokenizer", None)
        if backend is not None:
            return backend
    return None


def _space_cut_is_exact(tokenizer: object) -> bool:
    """Whether pieces cut by ``_space_cuts`` tokenize as the whole text does.

    That is: the pieces give the ids, and the offsets shifted by each piece's
    start, of the whole text.

    True for a byte-level BPE fast tokenizer of a known shape: no normalizer or
    NFC (a space is a starter that never composes), a Split pre-tokenizer with
    a pattern from ``_CUT_SAFE_SPLIT_PATTERNS`` followed by ByteLevel without a
    prefix space or its own regex, no post-processor or a ByteLevel one that
    keeps offsets, BPE without dropout, and added tokens that hold no
    whitespace and strip none (they are split out before pre-tokenization).
    """
    import json

    backend = _backend_of(tokenizer)
    if backend is None:
        return False

    def state(component: object) -> dict | None:
        return None if component is None else json.loads(component.__getstate__())

    normalizer = state(backend.normalizer)
    if normalizer is not None and normalizer.get("type") != "NFC":
        return False
    pre = state(backend.pre_tokenizer) or {}
    steps = pre.get("pretokenizers") if pre.get("type") == "Sequence" else None
    if not steps or len(steps) != 2:
        return False
    split, byte_level = steps
    if (
        split.get("type") != "Split"
        or split.get("behavior") != "Isolated"
        or split.get("invert")
        or (split.get("pattern") or {}).get("Regex") not in _CUT_SAFE_SPLIT_PATTERNS
        or byte_level.get("type") != "ByteLevel"
        or byte_level.get("add_prefix_space")
        or byte_level.get("use_regex")
    ):
        return False
    post = state(backend.post_processor)
    if post is not None and (post.get("type") != "ByteLevel" or post.get("trim_offsets")):
        return False
    if type(backend.model).__name__ != "BPE" or getattr(backend.model, "dropout", None):
        return False
    for added in backend.get_added_tokens_decoder().values():
        if (
            added.lstrip
            or added.rstrip
            or added.single_word
            or any(ch.isspace() for ch in added.content)
        ):
            return False
    return True


class _PieceOffsets:
    """Offsets of a text tokenized in pieces, as one call on the text returns them.

    Each piece's offsets are shifted by the piece's start character.

    Read lazily from the pieces' encodings, because the cache reads only a few
    of them (the resync span, every GRID_STRIDE-th start, the next split), and
    building every tuple in Python would hold the GIL for longer than the
    parallel tokenization saves when several requests arrive together.
    """

    __slots__ = ("_encodings", "_starts", "_cuts", "_len")

    def __init__(self, encodings: list, starts: list[int], cuts: list[int], length: int):
        self._encodings = encodings
        self._starts = starts  # first token index of each piece
        self._cuts = cuts  # first character of each piece
        self._len = length

    def __len__(self) -> int:
        return self._len

    def __iter__(self):
        return (self._at(i) for i in range(self._len))

    def _at(self, index: int) -> tuple[int, int]:
        piece = bisect_right(self._starts, index) - 1
        start, end = self._encodings[piece].token_to_chars(index - self._starts[piece])
        shift = self._cuts[piece]
        return start + shift, end + shift

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self._at(i) for i in range(*index.indices(self._len))]
        if index < 0:
            index += self._len
        if not 0 <= index < self._len:
            raise IndexError(index)
        return self._at(index)


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
        # Whether requests that ask for special tokens get the same ids, so the
        # cache can serve them too (set by ``create_prefix_token_cache``).
        self.serves_special_tokens = False
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
        # Texts tokenized in parallel pieces, and parallel attempts that raised
        # and were tokenized in one call instead.
        self.parallel_tokenizations = 0
        self.parallel_errors = 0
        # (tokenizer, its backend if _space_cut_is_exact holds for it, else
        # None), set on first use.
        self._cut_backend: tuple[object, object | None] | None = None
        # Texts being tokenized by _tokenize_long right now. Concurrent requests
        # split into fewer pieces each, so they do not oversubscribe the cores.
        self._inflight = 0

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
                tail_ids, tail_offsets = self._tokenize_long(tokenizer, text[split_char:])
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
            tail_ids, tail_offsets = self._tokenize_long(tokenizer, text)
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
                self._insert(text, ids, split_token, base_char + tail_offsets[tail_index][0], grid)
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

    def parallel_backend(self, tokenizer: OffsetTokenizer) -> object | None:
        """The backend of ``tokenizer`` if ``_space_cut_is_exact`` holds for it (cached)."""
        known = self._cut_backend
        if known is not None and known[0] is tokenizer:
            return known[1]
        try:
            backend = _backend_of(tokenizer) if _space_cut_is_exact(tokenizer) else None
        except Exception as e:  # noqa: BLE001 - unknown shape means "not known exact"
            logger.debug(f"Prefix token cache: cannot inspect the tokenizer: {e!r}")
            backend = None
        self._cut_backend = (tokenizer, backend)
        state = "on" if backend is not None else "off (tokenizer shape not known exact)"
        logger.info(
            f"Prefix token cache: parallel tail tokenization {state} "
            f"(PARALLEL_TAIL={PARALLEL_TAIL}, "
            f"TOKENIZERS_PARALLELISM={os.environ.get('TOKENIZERS_PARALLELISM')})"
        )
        return backend

    def _tokenize_long(
        self, tokenizer: OffsetTokenizer, text: str
    ) -> tuple[list[int], Sequence[tuple[int, int]]]:
        """``_tokenize``, run on parallel pieces of ``text`` when that is exact.

        The pieces are cut at pre-token boundaries (``_space_cuts``) and go to
        the tokenizer's backend in one ``encode_batch`` call, which runs them in
        parallel with the GIL released; their ids are concatenated and their
        offsets shifted by each piece's start, which is what one call on the
        whole text returns. Falls back to one call when the tokenizer is not of
        a shape the cut is proven for, the text is short or has no cut point,
        enough other texts are being tokenized that it would get one piece, or
        the batched call fails or does not cover every piece.
        """
        backend = None
        if PARALLEL_TAIL and len(text) >= 2 * PARALLEL_PIECE_CHARS:
            backend = self.parallel_backend(tokenizer)
        if backend is not None:
            with self._lock:
                self._inflight += 1
                pieces = min(
                    PARALLEL_MAX_PIECES // self._inflight, len(text) // PARALLEL_PIECE_CHARS
                )
            out = None
            try:
                if pieces > 1:
                    out = self._tokenize_pieces(backend, text, pieces)
            except Exception as e:  # noqa: BLE001 - never fail a request on this path
                with self._lock:
                    self.parallel_errors += 1
                logger.debug(f"Prefix token cache: parallel tokenization failed: {e!r}")
                out = None
            finally:
                with self._lock:
                    self._inflight -= 1
            if out is not None:
                with self._lock:
                    self.parallel_tokenizations += 1
                return out
        return self._tokenize(tokenizer, text)

    @staticmethod
    def _tokenize_pieces(
        backend: object, text: str, pieces: int
    ) -> tuple[list[int], _PieceOffsets] | None:
        # A truncation, padding or special-token splitting left on the backend
        # by another caller would change the ids; the HF call resets them,
        # encode_batch does not.
        if (
            backend.truncation is not None
            or backend.padding is not None
            or getattr(backend, "encode_special_tokens", False)
        ):
            return None
        cuts = _space_cuts(text, pieces)
        if len(cuts) < 2:
            return None
        ends = cuts[1:]
        ends.append(len(text))
        encodings = backend.encode_batch(
            [text[start:end] for start, end in zip(cuts, ends)], add_special_tokens=False
        )
        if len(encodings) != len(cuts):
            return None
        ids: list[int] = []
        starts = []
        for start, end, encoding in zip(cuts, ends, encodings):
            piece_ids = encoding.ids
            # Byte-level BPE gives every character to a token, so a piece's
            # last token ends at the piece's end; anything else is not the
            # tokenization of the whole piece.
            if not piece_ids:
                return None
            last = encoding.token_to_chars(len(piece_ids) - 1)
            if last is None or last[1] != end - start:
                return None
            starts.append(len(ids))
            ids.extend(piece_ids)
        return ids, _PieceOffsets(encodings, starts, cuts, len(ids))

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
    cache = PrefixTokenCache(config)
    cache.serves_special_tokens = tokenizer_adds_no_special_tokens(tokenizer)
    return cache


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
