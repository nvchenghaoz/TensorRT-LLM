# SPDX-License-Identifier: Apache-2.0
"""Diagnostic for idea 109: which parallel-tail variant keeps the c4 wave makespan.

On the real benchmark hit tails (~11K chars), time per variant: sequential latency
(c1) and 4-thread concurrent wave makespan / per-request (c4), including the
consumer's offset reads (grid every 16th start + first 32 ends), so variants that
build offsets lazily are compared fairly. CPU only. Last stdout line: JSON.
"""

import json
import sys
import threading
import time
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor

from tok_common import (PromptMaker, common_prefix, hf_tokenizer, load_real_prompts,
                        load_tokenizer, log, pct)


def main():
    ckpt = sys.argv[1]
    from tensorrt_llm.inputs import prefix_token_cache as ptc
    tok = load_tokenizer(ckpt)
    hf = hf_tokenizer(tok)
    bt = hf.backend_tokenizer
    real, _ = load_real_prompts(ckpt)
    real_all = real[1] + real[2] + real[4]
    tails = []
    probe = ptc.PrefixTokenCache(ptc.PrefixTokenCacheConfig())
    orig = probe._tokenize_long

    def grab(tkz, text):
        tails.append(text)
        return orig(tkz, text)

    probe._tokenize_long = grab
    probe.encode(tok, f"{common_prefix(real_all).rstrip()} {PromptMaker(hf, 3).sample(2048)}")
    for p in real[1]:
        probe.encode(tok, p)
    tails = tails[1:]
    log(f"{len(tails)} tails, chars {min(map(len, tails))}-{max(map(len, tails))}")

    inflight = [0]
    lock = threading.Lock()

    def consume(ids, offs):
        n = len(ids)
        g = [offs[i][0] for i in range(0, n, 16)]
        r = [offs[i][1] for i in range(min(32, n))]
        return len(g) + len(r)

    def hf_single(t):
        enc = tok(t, add_special_tokens=False, return_offsets_mapping=True)
        return consume(enc["input_ids"], enc["offset_mapping"])

    def hf_batch(t, k):
        cuts = ptc._space_cuts(t, k)
        ends = cuts[1:] + [len(t)]
        enc = tok([t[a:b] for a, b in zip(cuts, ends)], add_special_tokens=False,
                  return_offsets_mapping=True)
        ids = list(enc["input_ids"][0])
        offs = list(enc["offset_mapping"][0])
        for s, ci, co in zip(cuts[1:], enc["input_ids"][1:], enc["offset_mapping"][1:]):
            ids.extend(ci)
            offs.extend([(a + s, b + s) for a, b in co])
        return consume(ids, offs)

    class Lazy:
        __slots__ = ("encs", "starts", "cuts")

        def __init__(self, encs, starts, cuts):
            self.encs, self.starts, self.cuts = encs, starts, cuts

        def __getitem__(self, i):
            p = bisect_right(self.starts, i) - 1
            s, e = self.encs[p].token_to_chars(i - self.starts[p])
            c = self.cuts[p]
            return s + c, e + c

    def be_lazy(t, k):
        cuts = ptc._space_cuts(t, k)
        ends = cuts[1:] + [len(t)]
        encs = bt.encode_batch([t[a:b] for a, b in zip(cuts, ends)], add_special_tokens=False)
        ids = []
        starts = []
        for e in encs:
            starts.append(len(ids))
            ids.extend(e.ids)
        return consume(ids, Lazy(encs, starts, cuts))

    def be_full(t, k):
        cuts = ptc._space_cuts(t, k)
        ends = cuts[1:] + [len(t)]
        encs = bt.encode_batch([t[a:b] for a, b in zip(cuts, ends)], add_special_tokens=False)
        ids, offs = [], []
        for c, e in zip(cuts, encs):
            ids.extend(e.ids)
            offs.extend([(a + c, b + c) for a, b in e.offsets] if c else e.offsets)
        return consume(ids, offs)

    def adaptive(fn):
        def run(t):
            with lock:
                inflight[0] += 1
                k = max(1, 8 // inflight[0])
            try:
                return fn(t, k) if k > 1 else hf_single(t)
            finally:
                with lock:
                    inflight[0] -= 1
        return run

    variants = {"hf_single": hf_single}
    for k in (2, 4, 8):
        variants[f"hf_batch_{k}"] = (lambda kk: lambda t: hf_batch(t, kk))(k)
        variants[f"be_lazy_{k}"] = (lambda kk: lambda t: be_lazy(t, kk))(k)
        variants[f"be_full_{k}"] = (lambda kk: lambda t: be_full(t, kk))(k)
    variants["hf_batch_adapt"] = adaptive(hf_batch)
    variants["be_lazy_adapt"] = adaptive(be_lazy)

    out = {}
    pool = ThreadPoolExecutor(max_workers=4)
    names = list(variants)
    c1 = {n: [] for n in names}
    c4m = {n: [] for n in names}
    c4r = {n: [] for n in names}
    c4lead = {n: [] for n in names}
    for rep in range(24):
        for n in (names if rep % 2 == 0 else names[::-1]):
            fn = variants[n]
            t = tails[rep % len(tails)]
            t0 = time.perf_counter()
            fn(t)
            c1[n].append((time.perf_counter() - t0) * 1e3)
            wave = [tails[(rep + j) % len(tails)] for j in range(4)]

            def timed(x, fn=fn):
                a = time.perf_counter()
                fn(x)
                return (time.perf_counter() - a) * 1e3

            t0 = time.perf_counter()
            res = list(pool.map(timed, wave))
            c4m[n].append((time.perf_counter() - t0) * 1e3)
            c4r[n].extend(res)
            c4lead[n].append(min(res))
    for n in names:
        if len(c1[n]) > 4:
            c1[n], c4m[n] = c1[n][4:], c4m[n][4:]
        out[n] = {"c1_median": round(pct(c1[n], 0.5), 4),
                  "c4_makespan_median": round(pct(c4m[n], 0.5), 4),
                  "c4_per_request_median": round(pct(c4r[n], 0.5), 4),
                  "c4_leader_median": round(pct(c4lead[n], 0.5), 4)}
        log(f"{n}: {out[n]}")
    print(json.dumps({"variants": out}), flush=True)


if __name__ == "__main__":
    main()
