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

Long prompts are additionally split into pieces that are tokenized in parallel (the
``tokenizers`` ``encode_batch`` call releases the GIL). This is exact only for tokenizers of
the Gemma shape, which is checked once at construction:

- normalizer ``Replace(" " -> "▁")`` (char-local, length-preserving);
- pre-tokenizer absent or ``Split(" ")``, which never matches the normalized text, so the whole
  text is one BPE word;
- a BPE model without dropout, continuing-subword prefix or end-of-word suffix;
- added tokens contain no whitespace and have no lstrip/rstrip/single_word behaviour.

BPE merges only adjacent symbols into vocabulary tokens. A cut before position ``i`` with
``text[i] == " "`` can only be crossed by a token containing ``text[i-1] + "▁"``. The
vocabulary is scanned for every character that ever precedes ``"▁"`` inside a token
(for Gemma4 only ``"▁"`` itself and ``">"``); cuts are taken only after other characters, so
both sides merge independently and their concatenation equals the whole-text encoding.
Byte-fallback pieces are per character, and added tokens (which contain no space) never span a
cut. Any tokenizer that does not match this shape uses the unsplit path.
"""

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional

from ...logger import logger

# Prompts shorter than this are encoded in one call.
SPLIT_MIN_CHARS = 1024
# Target characters per piece and the most pieces per prompt. BPE over one long word costs
# more than linear time, so pieces are cheaper even when encoded one after another.
PIECE_CHARS = 768
MAX_PIECES = 8
# How far past a target cut point to look for a safe boundary.
CUT_SEARCH_CHARS = 512
# At most this many prompts use the piece pool at once; others encode their pieces on the
# calling thread, so concurrent requests never queue behind each other's pieces.
MAX_CONCURRENT_SPLITS = 2
POOL_WORKERS = (MAX_PIECES - 1) * MAX_CONCURRENT_SPLITS

_SPACE = " "
_META = "▁"


class _PiecePool:
    """Thread pool for prompt pieces, recreated after fork."""

    def __init__(self):
        self._lock = threading.Lock()
        self._pool: Optional[ThreadPoolExecutor] = None
        self._pid = None

    def get(self) -> ThreadPoolExecutor:
        pid = os.getpid()
        if self._pool is None or self._pid != pid:
            with self._lock:
                if self._pool is None or self._pid != pid:
                    self._pool = ThreadPoolExecutor(
                        max_workers=POOL_WORKERS, thread_name_prefix="gemma4_tok"
                    )
                    self._pid = pid
        return self._pool


_PIECE_POOL = _PiecePool()


def _state(component) -> Optional[dict]:
    if component is None:
        return None
    return json.loads(component.__getstate__())


class Gemma4TextEncoder:
    """Exact text-only encoder over a HF fast tokenizer (see module docstring)."""

    def __init__(self, tokenizer: Any):
        hf = getattr(tokenizer, "tokenizer", tokenizer)
        self._hf = hf
        self._backend = getattr(hf, "backend_tokenizer", None)
        self._bad_prev: Optional[frozenset] = None
        self._prefix: List[int] = []
        self._suffix: List[int] = []
        self._inflight = 0
        self._inflight_lock = threading.Lock()
        if self._backend is not None:
            try:
                self._bad_prev = self._prove_cut_rule()
            except Exception as e:  # noqa: BLE001 - any surprise disables splitting only
                logger.debug(f"Gemma4TextEncoder: parallel split disabled ({e!r})")
                self._bad_prev = None

    @property
    def can_split(self) -> bool:
        return self._bad_prev is not None

    def _prove_cut_rule(self) -> Optional[frozenset]:
        b = self._backend
        norm = _state(b.normalizer)
        if not (
            norm
            and norm.get("type") == "Replace"
            and norm.get("pattern") == {"String": _SPACE}
            and norm.get("content") == _META
        ):
            return None
        pre = _state(b.pre_tokenizer)
        if pre is not None and not (
            pre.get("type") == "Split"
            and pre.get("pattern") == {"String": _SPACE}
            and not pre.get("invert", False)
        ):
            return None
        from tokenizers.models import BPE

        model = b.model
        if not (
            isinstance(model, BPE)
            and model.dropout is None
            and not model.continuing_subword_prefix
            and not model.end_of_word_suffix
        ):
            return None
        added = b.get_added_tokens_decoder()
        for tok in added.values():
            content = tok.content
            if any(ch.isspace() or ch == _META for ch in content):
                return None
            if tok.lstrip or tok.rstrip or tok.single_word:
                return None
        vocab = b.get_vocab(with_added_tokens=True)
        max_tok_len = 0
        bad = {_SPACE, _META}
        for t in vocab:
            if len(t) > max_tok_len:
                max_tok_len = len(t)
            j = t.find(_META, 1)
            while j != -1:
                bad.add(t[j - 1])
                j = t.find(_META, j + 1)
        # ignore_merges looks a whole word up in the vocabulary; pieces are far longer than any
        # vocabulary token, so they never hit that lookup.
        if PIECE_CHARS // 2 <= max_tok_len:
            return None
        # Special tokens the post-processor adds: prefix and suffix around the sequence.
        probe = "a"
        on = b.encode_batch([probe], add_special_tokens=True)[0].ids
        off = b.encode_batch([probe], add_special_tokens=False)[0].ids
        idx = None
        for k in range(len(on) - len(off) + 1):
            if on[k : k + len(off)] == off:
                idx = k
                break
        if idx is None:
            return None
        self._prefix = list(on[:idx])
        self._suffix = list(on[idx + len(off) :])
        return frozenset(bad)

    def _cuts(self, text: str) -> List[int]:
        n = len(text)
        if n < SPLIT_MIN_CHARS:
            return []
        pieces = min(MAX_PIECES, max(2, n // PIECE_CHARS))
        bad = self._bad_prev
        cuts = []
        last = 0
        for p in range(1, pieces):
            pos = max(last + 1, (n * p) // pieces)
            stop = min(n, pos + CUT_SEARCH_CHARS)
            i = text.find(_SPACE, pos, stop)
            while i != -1 and text[i - 1] in bad:
                i = text.find(_SPACE, i + 1, stop)
            if i == -1:
                continue
            cuts.append(i)
            last = i
        return cuts

    def _encode_unsplit(self, text: str) -> List[int]:
        return self._backend.encode_batch([text], add_special_tokens=True)[0].ids

    def encode(self, text: str) -> List[int]:
        b = self._backend
        if b is None or b.truncation is not None or b.padding is not None:
            return list(self._hf(text)["input_ids"])
        if self._bad_prev is None:
            return self._encode_unsplit(text)
        cuts = self._cuts(text)
        if not cuts:
            return self._encode_unsplit(text)
        bounds = [0] + cuts + [len(text)]
        pieces = [text[bounds[k] : bounds[k + 1]] for k in range(len(bounds) - 1)]
        ids = list(self._prefix)
        with self._inflight_lock:
            parallel = self._inflight < MAX_CONCURRENT_SPLITS
            if parallel:
                self._inflight += 1
        if parallel:
            try:
                pool = _PIECE_POOL.get()
                futs = [
                    pool.submit(b.encode_batch, [piece], add_special_tokens=False)
                    for piece in pieces[1:]
                ]
                ids += b.encode_batch([pieces[0]], add_special_tokens=False)[0].ids
                for f in futs:
                    ids += f.result()[0].ids
            finally:
                with self._inflight_lock:
                    self._inflight -= 1
        else:
            # Other prompts are using the pool: encode the pieces on this thread.
            for piece in pieces:
                ids += b.encode_batch([piece], add_special_tokens=False)[0].ids
        if self._suffix:
            ids += self._suffix
        return ids
