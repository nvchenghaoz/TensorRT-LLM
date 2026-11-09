# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Prefix-tokenization cache for ``DefaultInputProcessor``.

In multi-turn serving each prompt is usually the previous prompt plus a small
delta, so re-tokenizing the whole prompt every turn is wasted work. This cache
remembers the token ids of recent prompts and, when a new prompt keeps a cached
one up to its re-tokenization point, tokenizes only the tail and splices it onto
the cached ids. The re-tokenization point lies ``overlap`` tokens before the end
of the cached prompt, so the new prompt may also rewrite that end, as a chat
template does when the previous turn's generation prefix is replaced by the
response.

Tokenizing a tail in isolation is not generally identical to tokenizing the
whole prompt, because BPE merges can straddle the split. The cache therefore
backs off ``overlap`` tokens from the end of the cached prefix, re-tokenizes
from there, and accepts the splice only if the first ``resync`` re-tokenized ids
equal the cached ids over the same span. Otherwise the prompt is tokenized in
full.

Off by default; enable with the ``enable_tokenization_cache`` LLM argument.
Each ``DefaultInputProcessor`` owns its own cache, so cached ids are never
shared across tokenizers. Sizing knobs, eligibility rules, and measured effect
are documented in ``docs/source/features/prefix-tokenization-cache.md``.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import unicodedata
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
]

# Log the hit rate every this many cache-eligible requests, so an operator can
# see whether the cache is doing anything without reading its counters.
LOG_EVERY_REQUESTS = 1000
# Agentic prompts share a system/tool preamble that can run to tens of thousands
# of characters, so entries in one bucket cannot be told apart by their start.
# Each entry keeps the PROBE_CHARS characters before its re-tokenization point; a
# candidate is only compared in full when the incoming text has that probe at the
# same position.
PROBE_CHARS = 256

# Shared-prefix (longest-common-prefix) splice. Serving workloads often send
# many prompts that share a long preamble and then diverge (a different
# question after the same document). The stock lookup above never hits them,
# because a sibling does not keep its cached neighbour's text up to the
# neighbour's re-tokenization point. When the tokenizer's shape makes a cut at
# "U+0020 after a non-space, non-digit, non-CJK character" provably exact (see
# ``_lcp_splice_gate``), the cache instead measures the common prefix with the
# most recent entries of the bucket, cuts at such a token start before the
# divergence, and tokenizes only the rest.
LCP_CANDIDATES = 8  # most recent entries of the bucket to measure
LCP_MARGIN_CHARS = 512  # the cut lies at least this far before the divergence
LCP_CHUNK_CHARS = 8192  # slice size of the common-prefix scan
GRID_STRIDE = 16  # entries keep the start char of every GRID_STRIDE-th token
SNAP_SEARCH = 64  # grid points searched backwards for an exact cut
MAX_SHARED_PREFIX_SIBLINGS = 32  # a spliced prompt becomes an entry only below this

# Cross-process sharing of full tokenizations. Several serving frontend
# processes (num_serve_frontends > 1) each own a cache; without sharing, each
# one pays a full tokenization of the shared preamble on its first request.
# When this env var names a directory, a process that tokenizes a prompt in
# full publishes the entry to <dir>/<tokenizer fingerprint>/, and a process
# whose shared-prefix lookup misses loads the entries it has not seen yet and
# looks again. The launcher sets it to a directory inside its multi-frontend
# ipc dir (serve.py ``_share_prefix_token_dir``). Unset: no file I/O.
SHARED_DIR_ENV = "TLLM_PREFIX_TOKEN_CACHE_SHARED_DIR"
# Parallel tail tokenization. With the shared-prefix splice on, the cut rule of
# ``_is_exact_cut`` also splits a text into pieces whose ids concatenate to the
# ids of the whole text. ``_tokenize_tail`` cuts a tail (or a missed prompt)
# into up to PARALLEL_MAX_PIECES // texts-in-flight pieces of at least
# PARALLEL_PIECE_CHARS, searching up to PARALLEL_CUT_SEARCH chars past each
# nominal cut, and encodes them with ONE ``backend_tokenizer.encode_batch``
# call, which releases the GIL and runs the pieces on the backend's thread
# pool. Offsets are read lazily (``_PieceOffsets``): the cache only reads the
# grid points and the next split.
PARALLEL_PIECE_CHARS = 1024
PARALLEL_CUT_SEARCH = 256
PARALLEL_MAX_PIECES = 8
SHARED_MAX_FILES = 64  # entries published per fingerprint directory
SHARED_SUFFIX = ".ptc"
SHARED_VERSION = 1
# Byte-level BPE split patterns, as in tokenizer.json, for which a cut at a
# U+0020 whose previous character is not whitespace, a \p{N} digit or CJK/kana
# is a pre-token boundary that depends on neither side's other characters:
# none has a lookbehind or anchor, and a space can only start a match or sit
# inside an all-whitespace match. Digit runs group in threes from their left
# end and CJK runs are isolated whole, so the cut never lands in or right
# after one.
_DEEPSEEK_V3_SPLIT_PATTERNS = (
    "\\p{N}{1,3}",
    "[一-龥぀-ゟ゠-ヿ]+",
    "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+"
    "| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+",
)
_CUT_SAFE_SPLIT_SEQUENCES = (_DEEPSEEK_V3_SPLIT_PATTERNS,)


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (
        0x2E80 <= o <= 0x9FFF
        or 0xF900 <= o <= 0xFAFF
        or 0xFE30 <= o <= 0xFE4F
        or 0xFF00 <= o <= 0xFFEF
        or o >= 0x20000
    )


def _is_exact_cut(text: str, cut: int) -> bool:
    """Whether ``ids(text) == ids(text[:cut]) + ids(text[cut:])`` under the gated shape."""
    if cut <= 0 or cut >= len(text) or text[cut] != " ":
        return False
    prev = text[cut - 1]
    return not (
        prev.isspace()
        or prev in "​⁠﻿"
        or unicodedata.category(prev)[0] == "N"
        or _is_cjk(prev)
    )


def _common_prefix_len(a: str, b: str) -> int:
    """Length of the longest common prefix, by slice compares (no per-char loop)."""
    n = min(len(a), len(b))
    lo = 0
    while lo < n:
        hi = min(lo + LCP_CHUNK_CHARS, n)
        if a[lo:hi] != b[lo:hi]:
            break
        lo = hi
    else:
        return n
    # a[:lo] == b[:lo] and a[lo:hi] != b[lo:hi]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if a[lo:mid] == b[lo:mid]:
            lo = mid
        else:
            hi = mid
    return lo


def _offset_grid(offsets: Sequence[tuple[int, int]], base_char: int, base_token: int) -> array:
    """Start chars of the grid tokens among ``offsets``, whose first token is ``base_token``."""
    first = -base_token % GRID_STRIDE
    return array("i", [base_char + off[0] for off in offsets[first::GRID_STRIDE]])


def _piece_cuts(text: str, n_pieces: int) -> list[int]:
    """Exact cuts splitting ``text`` into at most ``n_pieces`` pieces of >= PARALLEL_PIECE_CHARS.

    Each cut is the first ``_is_exact_cut`` point at or after the nominal cut,
    looking PARALLEL_CUT_SEARCH chars ahead; if a window has none, the piece
    grows into the next window. The last piece is also >= PARALLEL_PIECE_CHARS.
    """
    n = len(text)
    if n_pieces <= 1 or n < 2 * PARALLEL_PIECE_CHARS:
        return []
    size = max(PARALLEL_PIECE_CHARS, -(-n // n_pieces))
    last = n - PARALLEL_PIECE_CHARS  # a cut may lie at most here
    cuts: list[int] = []
    lo = size
    while len(cuts) < n_pieces - 1 and lo <= last:
        hi = min(lo + PARALLEL_CUT_SEARCH, last)
        c = text.find(" ", lo, hi + 1)
        while c != -1 and not _is_exact_cut(text, c):
            c = text.find(" ", c + 1, hi + 1)
        if c == -1:
            lo = hi + 1
            continue
        cuts.append(c)
        lo = c + size
    return cuts


class _PieceOffsets:
    """Offsets of concatenated piece encodings, read lazily through ``token_to_chars``.

    Indexing (int or slice) returns the same ``(start, end)`` tuples, in chars
    of the whole text, as the single call's ``offset_mapping``.
    """

    __slots__ = ("_encs", "_char_starts", "_tok_starts", "_n")

    def __init__(self, encs: Sequence, char_starts: Sequence[int], lengths: Sequence[int]):
        self._encs = encs
        self._char_starts = char_starts
        tok_starts = []
        n = 0
        for length in lengths:
            tok_starts.append(n)
            n += length
        self._tok_starts = tok_starts
        self._n = n

    def __len__(self) -> int:
        return self._n

    def _at(self, i: int) -> tuple[int, int]:
        p = bisect_right(self._tok_starts, i) - 1
        start, end = self._encs[p].token_to_chars(i - self._tok_starts[p])
        base = self._char_starts[p]
        return (base + start, base + end)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self._at(j) for j in range(*i.indices(self._n))]
        if i < 0:
            i += self._n
        if not 0 <= i < self._n:
            raise IndexError("offset index out of range")
        return self._at(i)


def _encode_pieces(
    backend: object, text: str, n_pieces: int
) -> tuple[list[int], _PieceOffsets] | None:
    """Ids and lazy offsets of ``text`` from one ``encode_batch`` over exact pieces.

    None when a piece's encoding does not span exactly its piece (the caller
    then makes the single HF call).
    """
    cuts = _piece_cuts(text, n_pieces)
    starts = [0, *cuts]
    ends = [*cuts, len(text)]
    pieces = [text[a:b] for a, b in zip(starts, ends)]
    encs = backend.encode_batch(pieces, add_special_tokens=False)
    ids: list[int] = []
    lengths = []
    for enc, piece in zip(encs, pieces):
        n = len(enc)
        first = enc.token_to_chars(0) if n else None
        end = enc.token_to_chars(n - 1) if n else None
        if first is None or end is None or first[0] != 0 or end[1] != len(piece):
            return None
        ids.extend(enc.ids)
        lengths.append(n)
    return ids, _PieceOffsets(encs, starts, lengths)


def _backend_state_is_plain(tokenizer: object, backend: object) -> bool:
    """Whether ``encode_batch`` sees the state the HF call would reset it to.

    ``encode_batch`` skips the HF call's truncation/padding/special-token reset.
    """
    return (
        backend.truncation is None
        and backend.padding is None
        and not getattr(backend, "encode_special_tokens", False)
        and not getattr(tokenizer, "split_special_tokens", False)
    )


def _parallel_tail_gate(tokenizer: object) -> tuple[object | None, str]:
    """The backend to encode pieces with, or None, and why.

    Called only for a tokenizer that passed ``_lcp_splice_gate`` (so the cut
    rule is exact); checks the backend API and an empirical probe against the
    single HF call, ids and every offset.
    """
    try:
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is None or not hasattr(backend, "encode_batch"):
            return None, "no backend encode_batch"
        if not _backend_state_is_plain(tokenizer, backend):
            return None, "backend truncation/padding/special-token state set"
        unit = (
            "Alpha beta,  gamma\t delta 12345 x中文 y こん z.  "
            "<|end|> été \U0001f600 ok\n\n  next (paren) 'quote' end. "
        )
        probe = unit * (6 * PARALLEL_PIECE_CHARS // len(unit) + 1)
        enc = tokenizer(probe, add_special_tokens=False, return_offsets_mapping=True)
        ref_ids, ref_offsets = list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]
        got = _encode_pieces(backend, probe, 4)
        if got is None:
            return None, "probe pieces do not span their text"
        ids, offsets = got
        if ids != ref_ids:
            return None, "probe piece ids differ from the single call"
        if list(offsets[0:]) != ref_offsets:
            return None, "probe piece offsets differ from the single call"
        return backend, "encode_batch over exact pieces matches the single call"
    except Exception as e:  # noqa: BLE001 - any doubt keeps the single call
        return None, f"probe failed ({e!r})"


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
    # Every prompt that keeps ``text[:split_char]`` is re-tokenized from the same
    # place: token ``split_token``, which starts at character ``split_char``.
    split_token: int
    split_char: int
    probe: str  # the PROBE_CHARS characters of ``text`` before ``split_char``
    # Start char of tokens 0, GRID_STRIDE, 2 * GRID_STRIDE, ... (int32), kept
    # only when the shared-prefix splice is on.
    grid: array | None = None


class PrefixTokenCache:
    """Splice a cached prefix's token ids with a freshly tokenized tail.

    Thread-safe. Eviction is least-recently-used, bounded by both entry count
    and total cached characters. When a prompt reuses a cached entry, the new
    prompt replaces that entry rather than sitting alongside it, so a
    conversation costs one entry however many turns it has.

    An unexpected error disables the cache, with a warning, and every prompt is
    tokenized in full from then on: a cache bug must never fail a request.
    """

    def __init__(
        self,
        config: PrefixTokenCacheConfig,
        lcp_splice: bool = False,
        fingerprint: str | None = None,
        parallel_backend: object | None = None,
    ) -> None:
        self._config = config
        # Backend for parallel tail tokenization (see _tokenize_tail); only for
        # a tokenizer that passed _lcp_splice_gate and _parallel_tail_gate.
        self._parallel_backend = parallel_backend if lcp_splice else None
        self._inflight_lock = threading.Lock()
        self._inflight = 0  # texts inside _tokenize_tail's encode_batch
        self.parallel_tails = 0
        self.parallel_fallbacks = 0
        # Shared-prefix splice; only for a tokenizer that passed _lcp_splice_gate.
        self._lcp_splice = lcp_splice
        # Tokenizer fingerprint naming the cross-process directory; None keeps
        # the cache process-local. Resolved lazily (SHARED_DIR_ENV may be set
        # after the cache is built), see _shared_dir.
        self._fingerprint = fingerprint if lcp_splice else None
        self._shared_path: str | None = None
        self._shared_seen: set[str] = set()  # file names loaded, published or rejected
        self._shared_load_lock = threading.Lock()  # one directory scan at a time
        self.shared_loads = 0
        self.shared_publishes = 0
        self.shared_skipped = 0
        self._lock = threading.Lock()
        self._entries: OrderedDict[int, _Entry] = OrderedDict()  # LRU order
        self._buckets: dict[int, dict[int, _Entry]] = {}
        self._total_chars = 0
        self._next_id = 0
        # Read and written without the lock: a bool flip is atomic, and the
        # worst case of a race is two threads each logging the disable warning.
        self.disabled = False
        # Counters, reported by periodic log lines and read by tests.
        self.hits = 0
        self.misses = 0
        self.resync_failures = 0
        # Shared-prefix splice counters. A splice is not a hit: ``misses``
        # counts every prompt the stock lookup missed, and ``splices`` how many
        # of those the shared-prefix splice still served.
        self.splices = 0
        self.splice_fallbacks = 0  # splices refused by the resync check
        self.splice_no_cut = 0  # long common prefix but no exact cut found
        self._splice_ms = 0.0
        self._full_ms = 0.0
        self._full_count = 0

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
        total = self.hits + self.misses
        rate = self.hits / total if total else 0.0
        text = (
            f"{self.hits}/{total} hits ({rate:.1%}), "
            f"{self.resync_failures} resync fallbacks, "
            f"{len(self._entries)} entries, {self._total_chars} cached chars"
        )
        if self._lcp_splice:
            splice_ms = self._splice_ms / self.splices if self.splices else 0.0
            full_ms = self._full_ms / self._full_count if self._full_count else 0.0
            text += (
                f"; pid {os.getpid()}: {self.splices} shared-prefix splices, "
                f"{self._full_count} full tokenizations, "
                f"{self.splice_fallbacks} splice resync fallbacks, "
                f"{self.splice_no_cut} without an exact cut, "
                f"{splice_ms:.2f} ms per spliced vs {full_ms:.2f} ms per full request"
            )
            if self._parallel_backend is not None:
                text += (
                    f"; parallel tails: {self.parallel_tails}, "
                    f"{self.parallel_fallbacks} fallbacks"
                )
            if self._shared_path is not None:
                text += (
                    f"; shared: {self.shared_loads} loaded, "
                    f"{self.shared_publishes} published, {self.shared_skipped} skipped"
                )
        return text

    def _encode(self, tokenizer: OffsetTokenizer, text: str) -> list[int]:
        start = time.perf_counter()
        found = self._lookup(text)
        if (
            found is not None
            and self._lcp_splice
            and not (
                _is_exact_cut(text, found[1].split_char)
                and _is_exact_cut(found[1].text, found[1].split_char)
            )
        ):
            # With the splice on, re-tokenize only from provably exact cuts;
            # the shared-prefix path below serves this prompt instead.
            found = None
        ids = None
        spliced = None  # (entry, lcp chars) when the shared-prefix splice served it
        splice_fallback = splice_no_cut = False
        if found is not None:
            eid, entry = found
            tail_ids, tail_offsets = self._tokenize_tail(tokenizer, text[entry.split_char :])
            if self._resynced(entry, tail_ids):
                ids = entry.ids[: entry.split_token].tolist() + list(tail_ids)
                base_token, base_char = entry.split_token, entry.split_char
                base_entry = entry
        elif self._lcp_splice:
            loads_seen = self.shared_loads
            cand = self._lcp_lookup(text)
            if (
                cand is None
                and self._fingerprint is not None
                and self._load_shared(loads_seen)
            ):
                # Another process may already have tokenized this preamble.
                cand = self._lcp_lookup(text)
            if cand is not None:
                entry, tok, cut, lcp = cand
                if tok < 0:
                    splice_no_cut = True
                else:
                    tail_ids, tail_offsets = self._tokenize_tail(tokenizer, text[cut:])
                    if self._lcp_resynced(entry, tok, tail_ids):
                        ids = entry.ids[:tok].tolist() + list(tail_ids)
                        base_token, base_char = tok, cut
                        base_entry = entry
                        spliced = (entry, lcp)
                    else:
                        splice_fallback = True
        if ids is None:
            tail_ids, tail_offsets = self._tokenize_tail(tokenizer, text)
            ids, base_token, base_char = tail_ids, 0, 0
            base_entry = None

        # Where a prompt extending this one will be re-tokenized from.
        split_token = len(ids) - self._config.overlap
        tail_index = split_token - base_token
        grid = None
        if self._lcp_splice and split_token > 0 and tail_index >= 0:
            grid = _offset_grid(tail_offsets, base_char, base_token)
            if base_entry is not None:
                grid = base_entry.grid[: (base_token + GRID_STRIDE - 1) // GRID_STRIDE] + grid
        elapsed_ms = (time.perf_counter() - start) * 1e3

        with self._lock:
            if found is None:
                self.misses += 1
            elif base_token == 0:
                self.misses += 1
                self.resync_failures += 1
            else:
                self.hits += 1
            if spliced is not None:
                self.splices += 1
                self._splice_ms += elapsed_ms
            elif base_token == 0:
                self._full_count += 1
                self._full_ms += elapsed_ms
            self.splice_fallbacks += splice_fallback
            self.splice_no_cut += splice_no_cut
            if found is not None:
                self._remove(found[0])
            insert = split_token > 0 and tail_index >= 0
            if spliced is not None:
                # One shared-prefix workload must not grow its bucket without
                # bound: lookups scan the bucket.
                bucket = self._buckets.get(self._bucket_key(text))
                insert = insert and (bucket is None or len(bucket) < MAX_SHARED_PREFIX_SIBLINGS)
            publish = None
            if insert:
                split_char = base_char + tail_offsets[tail_index][0]
                self._insert(text, ids, split_token, split_char, grid)
                if (
                    self._fingerprint is not None
                    and found is None
                    and base_entry is None
                    and grid is not None
                ):
                    publish = (split_token, split_char)
            n = self.hits + self.misses
            # Power-of-two counts as well, so a short serving lifetime still
            # shows whether the splice works.
            report = n % LOG_EVERY_REQUESTS == 0 or (self._lcp_splice and n & (n - 1) == 0)
            summary = self.summary() if report else None
        if publish is not None:
            self._publish_shared(text, ids, publish[0], publish[1], grid)
        if summary is not None:
            logger.info(f"Prefix token cache: {summary}")
        return ids

    # ------------------------------------------------------------------
    # Cross-process sharing (SHARED_DIR_ENV). Every failure here only skips
    # sharing; it never disables the cache or fails a request.

    def _shared_dir(self) -> str | None:
        """This tokenizer's shared directory, or None while SHARED_DIR_ENV is unset."""
        if self._shared_path is None and self._fingerprint is not None:
            root = os.environ.get(SHARED_DIR_ENV)
            if root:
                self._shared_path = os.path.join(root, self._fingerprint)
        return self._shared_path

    def _publish_shared(
        self, text: str, ids: Sequence[int], split_token: int, split_char: int, grid: array
    ) -> None:
        """Write a fully tokenized entry for other processes; skip if already there."""
        path = self._shared_dir()
        if path is None:
            return
        start = time.perf_counter()
        text_bytes = text.encode("utf-8")
        name = hashlib.sha1(text_bytes).hexdigest()[:16] + SHARED_SUFFIX
        with self._lock:
            if name in self._shared_seen:
                return
            self._shared_seen.add(name)
        try:
            os.makedirs(path, mode=0o700, exist_ok=True)
            target = os.path.join(path, name)
            if os.path.exists(target):
                return
            if sum(1 for e in os.scandir(path) if e.name.endswith(SHARED_SUFFIX)) >= (
                SHARED_MAX_FILES
            ):
                return
            ids_arr = array("i", ids)
            grid_arr = array("i", grid)
            payload = text_bytes + ids_arr.tobytes() + grid_arr.tobytes()
            header = {
                "v": SHARED_VERSION,
                "fp": self._fingerprint,
                "pid": os.getpid(),
                "itemsize": ids_arr.itemsize,
                "n_chars": len(text),
                "n_text_bytes": len(text_bytes),
                "n_ids": len(ids_arr),
                "n_grid": len(grid_arr),
                "split_token": split_token,
                "split_char": split_char,
                "sha1": hashlib.sha1(payload).hexdigest(),
            }
            tmp = f"{target}.{os.getpid()}.{threading.get_ident()}.tmp"
            with open(tmp, "wb") as f:
                f.write(json.dumps(header).encode("utf-8") + b"\n")
                f.write(payload)
            os.replace(tmp, target)
        except OSError as e:
            logger.warning(f"Prefix token cache: pid {os.getpid()} could not publish {name}: {e!r}")
            return
        self.shared_publishes += 1
        logger.info(
            f"Prefix token cache: pid {os.getpid()} published shared entry {name} "
            f"({len(text)} chars, {len(ids_arr)} ids, {len(payload)} bytes) in "
            f"{(time.perf_counter() - start) * 1e3:.2f} ms"
        )

    def _load_shared(self, loads_seen: int) -> bool:
        """Insert the shared entries not seen yet.

        Returns whether any entry was inserted since ``shared_loads`` was
        ``loads_seen``, by this thread or by a concurrent one: loads are
        serialized, so a request that misses while another thread is loading
        waits for it and then looks again instead of tokenizing in full.
        """
        path = self._shared_dir()
        if path is None:
            return False
        with self._shared_load_lock:
            self._load_shared_locked(path)
        return self.shared_loads > loads_seen

    def _load_shared_locked(self, path: str) -> None:
        """Caller holds ``_shared_load_lock``."""
        try:
            names = sorted(
                e.name for e in os.scandir(path) if e.name.endswith(SHARED_SUFFIX)
            )
        except OSError:
            return
        with self._lock:
            fresh = [n for n in names if n not in self._shared_seen]
            self._shared_seen.update(fresh)
        for name in fresh:
            start = time.perf_counter()
            entry, why = self._read_shared(os.path.join(path, name))
            elapsed_ms = (time.perf_counter() - start) * 1e3
            if entry is None:
                self.shared_skipped += 1
                logger.info(
                    f"Prefix token cache: pid {os.getpid()} skipped shared entry {name}: {why}"
                )
                continue
            text, ids, split_token, split_char, grid, src_pid = entry
            with self._lock:
                self._insert(text, ids, split_token, split_char, grid)
                self.shared_loads += 1
            logger.info(
                f"Prefix token cache: pid {os.getpid()} loaded shared entry {name} "
                f"from pid {src_pid} ({len(text)} chars, {len(ids)} ids) in {elapsed_ms:.2f} ms"
            )

    def _read_shared(self, file: str) -> tuple[tuple | None, str]:
        """Parse and validate one shared file: ((text, ids, split_token, split_char, grid, pid), why)."""
        try:
            with open(file, "rb") as f:
                data = f.read()
            nl = data.find(b"\n")
            if nl < 0:
                return None, "no header"
            header = json.loads(data[:nl])
            if not isinstance(header, dict) or header.get("v") != SHARED_VERSION:
                return None, "unknown version"
            if header.get("fp") != self._fingerprint:
                return None, "foreign tokenizer fingerprint"
            ids = array("i")
            if header.get("itemsize") != ids.itemsize:
                return None, "itemsize mismatch"
            fields = ("n_chars", "n_text_bytes", "n_ids", "n_grid", "split_token", "split_char")
            if not all(type(header.get(k)) is int and header[k] >= 0 for k in fields):
                return None, "malformed header"
            n_chars, n_text_bytes, n_ids, n_grid, split_token, split_char = (
                header[k] for k in fields
            )
            payload = memoryview(data)[nl + 1 :]
            if len(payload) != n_text_bytes + ids.itemsize * (n_ids + n_grid):
                return None, "truncated or oversized payload"
            if hashlib.sha1(payload).hexdigest() != header.get("sha1"):
                return None, "checksum mismatch"
            text = bytes(payload[:n_text_bytes]).decode("utf-8")
            if len(text) != n_chars or n_chars < self._config.min_chars:
                return None, "text length mismatch"
            ids.frombytes(payload[n_text_bytes : n_text_bytes + ids.itemsize * n_ids])
            grid = array("i")
            grid.frombytes(payload[n_text_bytes + ids.itemsize * n_ids :])
            if not (
                0 < split_token < n_ids
                and 0 < split_char < n_chars
                and n_grid == (n_ids + GRID_STRIDE - 1) // GRID_STRIDE
                and grid[0] == 0
                and all(a <= b for a, b in zip(grid, grid[1:]))
                and grid[-1] < n_chars
            ):
                return None, "inconsistent split or grid"
            return (text, ids, split_token, split_char, grid, header.get("pid")), "ok"
        except (OSError, ValueError, TypeError, KeyError, UnicodeDecodeError) as e:
            return None, f"unreadable ({e!r})"

    @staticmethod
    def _tokenize(
        tokenizer: OffsetTokenizer, text: str
    ) -> tuple[list[int], Sequence[tuple[int, int]]]:
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        return enc["input_ids"], enc["offset_mapping"]

    def _tokenize_tail(
        self, tokenizer: OffsetTokenizer, text: str
    ) -> tuple[list[int], Sequence[tuple[int, int]]]:
        """``_tokenize``, through one GIL-releasing ``encode_batch`` over exact pieces when on."""
        backend = self._parallel_backend
        if (
            backend is None
            or len(text) < 2 * PARALLEL_PIECE_CHARS
            or getattr(tokenizer, "backend_tokenizer", None) is not backend
            or not _backend_state_is_plain(tokenizer, backend)
        ):
            return self._tokenize(tokenizer, text)
        with self._inflight_lock:
            self._inflight += 1
            inflight = self._inflight
        try:
            got = _encode_pieces(backend, text, max(1, PARALLEL_MAX_PIECES // inflight))
        finally:
            with self._inflight_lock:
                self._inflight -= 1
        if got is None:
            self.parallel_fallbacks += 1
            return self._tokenize(tokenizer, text)
        self.parallel_tails += 1
        return got

    def _resynced(self, entry: _Entry, tail_ids: Sequence[int]) -> bool:
        """Whether the tokenizer re-synchronized with the cached ids at the seam."""
        cached_tail = entry.ids[entry.split_token :]
        span = min(self._config.resync, len(tail_ids), len(cached_tail))
        return span > 0 and list(tail_ids[:span]) == cached_tail[:span].tolist()

    def _lcp_lookup(self, text: str) -> tuple[_Entry, int, int, int] | None:
        """The recent entry sharing the longest prefix with ``text``, and where to cut.

        Returns None if no entry shares a long enough prefix, else
        ``(entry, token, cut, lcp)``: ``text[:cut]`` is exactly the cached
        tokens ``entry.ids[:token]``, and ``lcp`` is the common prefix length in
        chars. ``token`` is -1 if no exact cut was found near the divergence.
        """
        with self._lock:
            bucket = self._buckets.get(self._bucket_key(text))
            candidates = list(bucket.values())[-LCP_CANDIDATES:] if bucket else []
        best, best_lcp = None, 0
        for entry in reversed(candidates):  # most recent first
            if entry.grid is None:
                continue
            lcp = _common_prefix_len(text, entry.text)
            if lcp > best_lcp:
                best, best_lcp = entry, lcp
        if best is None or best_lcp < max(self._config.min_chars, len(text) // 2) + LCP_MARGIN_CHARS:
            return None
        grid = best.grid
        g = bisect_right(grid, best_lcp - LCP_MARGIN_CHARS) - 1
        for g in range(g, max(g - SNAP_SEARCH, 0), -1):
            cut = grid[g]
            # The resync span (the next 2 grid strides) must lie inside the
            # common prefix, where the cached ids are the ids of ``text``.
            if (
                g + 2 < len(grid)
                and grid[g + 2] <= best_lcp
                and _is_exact_cut(text, cut)
                and (g == 0 or grid[g - 1] < cut)
            ):
                return best, g * GRID_STRIDE, cut, best_lcp
        return best, -1, -1, best_lcp

    def _lcp_resynced(self, entry: _Entry, token: int, tail_ids: Sequence[int]) -> bool:
        """Whether the tail's first ``resync`` ids equal the cached ids after ``token``."""
        span = min(self._config.resync, len(tail_ids), len(entry.ids) - token)
        return span > 0 and list(tail_ids[:span]) == entry.ids[token : token + span].tolist()

    def _bucket_key(self, text: str) -> int:
        return hash(text[: self._config.min_chars])

    def _lookup(self, text: str) -> tuple[int, _Entry] | None:
        """Cached entry with the longest re-tokenization prefix ``text`` keeps, promoted to MRU."""
        with self._lock:
            candidates = list(self._buckets.get(self._bucket_key(text), {}).items())
        # Entries are immutable, so the comparisons run outside the lock. The
        # probe check rejects other conversations in O(PROBE_CHARS); only a true
        # prefix pays the full compare over the shared preamble.
        best: tuple[int, _Entry] | None = None
        for eid, entry in candidates:
            n = entry.split_char
            if (
                n <= len(text)
                and (best is None or n > best[1].split_char)
                and text[n - len(entry.probe) : n] == entry.probe
                and text.startswith(entry.text[:n])
            ):
                best = (eid, entry)
        if best is not None:
            with self._lock:
                if best[0] in self._entries:
                    self._entries.move_to_end(best[0])
            # If it was evicted meanwhile the entry is still a valid prefix to
            # splice from; _remove() on it is a no-op.
        return best

    def _insert(
        self,
        text: str,
        ids: Sequence[int],
        split_token: int,
        split_char: int,
        grid: array | None = None,
    ) -> None:
        """Caller holds the lock."""
        key = self._bucket_key(text)
        bucket = self._buckets.setdefault(key, {})
        if any(entry.text == text for entry in bucket.values()):
            return  # concurrent misses on the same prompt
        eid, self._next_id = self._next_id, self._next_id + 1
        probe = text[max(split_char - PROBE_CHARS, 0) : split_char]
        entry = _Entry(text, array("i", ids), split_token, split_char, probe, grid)
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
    lcp_splice, why = _lcp_splice_gate(tokenizer)
    fingerprint = _tokenizer_fingerprint(tokenizer) if lcp_splice else None
    parallel_backend, parallel_why = (
        _parallel_tail_gate(tokenizer) if lcp_splice else (None, "")
    )
    logger.info(
        f"Prefix token cache enabled: {config}; shared-prefix splice "
        f"{'on' if lcp_splice else 'off'} ({why})"
        + (f"; cross-process fingerprint {fingerprint}" if fingerprint else "")
        + (
            f"; parallel tail tokenization {'on' if parallel_backend is not None else 'off'} "
            f"({PARALLEL_MAX_PIECES} pieces of >= {PARALLEL_PIECE_CHARS} "
            f"chars: {parallel_why})"
            if parallel_why
            else ""
        )
    )
    return PrefixTokenCache(
        config, lcp_splice=lcp_splice, fingerprint=fingerprint, parallel_backend=parallel_backend
    )


def _tokenizer_fingerprint(tokenizer: object) -> str | None:
    """Short hash of the backend tokenizer's serialized form; None if unavailable."""
    try:
        spec = tokenizer.backend_tokenizer.to_str()
        return hashlib.sha1(spec.encode("utf-8")).hexdigest()[:12]
    except Exception:  # noqa: BLE001 - no fingerprint keeps the cache process-local
        return None


def _lcp_splice_gate(tokenizer: object) -> tuple[bool, str]:
    """Whether a U+0020 cut (see ``_is_exact_cut``) is exact for ``tokenizer``, and why.

    Requires, from the backend's serialized form: no normalizer (or an empty
    Sequence, or NFC, which never composes across U+0020); a pre-tokenizer
    ``Sequence[Split(Isolated) for a pattern set in _CUT_SAFE_SPLIT_SEQUENCES,
    ByteLevel(add_prefix_space=False, use_regex=False)]``; a BPE model without
    dropout; a post-processor that keeps untrimmed offsets; and added tokens
    with no whitespace and no strip or single-word flags. Then checks a probe
    text empirically. Anything else keeps the splice off.
    """
    try:
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is None:
            return False, "no backend tokenizer"
        spec = json.loads(backend.to_str())
        norm = spec.get("normalizer")
        if norm is not None and not (
            norm.get("type") == "NFC"
            or (norm.get("type") == "Sequence" and not norm.get("normalizers"))
        ):
            return False, f"normalizer {norm.get('type')}"
        pre = spec.get("pre_tokenizer") or {}
        steps = pre.get("pretokenizers") or []
        if pre.get("type") != "Sequence" or len(steps) < 2:
            return False, f"pre_tokenizer {pre.get('type')}"
        *splits, last = steps
        if not (
            last.get("type") == "ByteLevel"
            and last.get("add_prefix_space") is False
            and last.get("use_regex") is False
        ):
            return False, "pre_tokenizer does not end in ByteLevel without regex/prefix space"
        patterns = []
        for split in splits:
            if (
                split.get("type") != "Split"
                or split.get("behavior") != "Isolated"
                or split.get("invert")
                or "Regex" not in (split.get("pattern") or {})
            ):
                return False, f"pre_tokenizer step {split.get('type')}"
            patterns.append(split["pattern"]["Regex"])
        if tuple(patterns) not in _CUT_SAFE_SPLIT_SEQUENCES:
            return False, f"split patterns {patterns!r} not known cut-safe"
        model = spec.get("model") or {}
        if model.get("type") != "BPE" or model.get("dropout"):
            return False, f"model {model.get('type')} dropout {model.get('dropout')}"
        post = spec.get("post_processor")
        if post is not None:
            kind = post.get("type")
            if kind == "ByteLevel":
                if post.get("trim_offsets"):
                    return False, "post_processor trims offsets"
            elif kind != "TemplateProcessing":
                return False, f"post_processor {kind}"
        for added in spec.get("added_tokens") or []:
            content = added.get("content", "")
            if any(c.isspace() for c in content) or (
                added.get("lstrip") or added.get("rstrip") or added.get("single_word")
            ):
                return False, f"added token {content!r} may straddle a cut"
        # Empirical probe: contiguous offsets, and the split identity at every
        # qualifying cut.
        probe = (
            "Alpha beta,  gamma\t delta 12345 x中文 y こん z.  "
            "<|end|> été \U0001f600 ok\n\n  next (paren) 'quote' end"
        )
        enc = tokenizer(probe, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = list(enc["input_ids"]), list(enc["offset_mapping"])
        ascii_offsets = [o for o in offsets if o[1] <= 33]  # probe[:33] is ASCII
        if not ascii_offsets or ascii_offsets[0][0] != 0:
            return False, "probe offsets do not start at 0"
        for a, b in zip(ascii_offsets, ascii_offsets[1:]):
            if a[1] != b[0]:
                return False, "probe offsets are not contiguous"
        for cut in range(1, len(probe)):
            if not _is_exact_cut(probe, cut):
                continue
            left = tokenizer(probe[:cut], add_special_tokens=False)["input_ids"]
            right = tokenizer(probe[cut:], add_special_tokens=False)["input_ids"]
            if list(left) + list(right) != ids:
                return False, f"probe split at {cut} is not exact"
            if cut not in {o[0] for o in offsets}:
                return False, f"probe cut {cut} is not a token start"
        return True, "byte-level BPE with cut-safe split patterns"
    except Exception as e:  # noqa: BLE001 - any doubt keeps the splice off
        return False, f"gate error {e!r}"
