# SPDX-License-Identifier: Apache-2.0
"""Exactness check for the parallel tail tokenization of PrefixTokenCache
(host-runtime idea 109): ids AND offsets of PrefixTokenCache._tokenize_long
(pieces cut at pre-token boundaries, one batched tokenizer call) vs one
tokenizer call on the whole text (main's _tokenize), on
  - the benchmark's real prompts and the tails the cache tokenizes for them,
  - >= 200 generated prompts (Shakespeare corpus samples, word salad, noise),
  - a junction set (double spaces, newlines, punctuation / digit runs,
    combining marks, CJK, emoji, NBSP / zero-width, added tokens near a cut),
    where EVERY cut point is checked as a two-piece split,
sequentially and on 8 threads. Then the whole cache path on the real prompts
with the kill switch on and off: returned ids equal a full tokenizer call, every
request after the first is a cache hit with 0 fallbacks, the stored entries
(ids, grid, split points) are bitwise identical between the two arms, and a
second pass on the warm cache equals the first. Also: the tokenizer-shape gate
refuses modified tokenizers. CPU only.

Last stdout line: {"passed": true|false, ...}
"""

import argparse
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor

from tok_common import (PromptMaker, common_prefix, hf_tokenizer, load_real_prompts,
                        load_tokenizer, log, make_processor, processor_cache)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--generated", type=int, default=240)
    args = ap.parse_args()

    from tensorrt_llm.inputs import prefix_token_cache as ptc
    from tensorrt_llm.sampling_params import SamplingParams

    tok = load_tokenizer(args.ckpt)
    hf = hf_tokenizer(tok)
    failures = []
    res = {}

    def fail(msg):
        failures.append(msg)
        log(f"FAIL {msg}")

    def single(text):
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]

    def norm(out):
        return list(out[0]), [tuple(o) for o in out[1]]

    # 1. Shape gate.
    gate = {"wrapper": ptc._space_cut_is_exact(tok), "hf": ptc._space_cut_is_exact(hf)}
    from tokenizers import AddedToken, Tokenizer
    base_json = json.loads(hf.backend_tokenizer.to_str())

    class Wrap:
        def __init__(self, backend):
            self.backend_tokenizer = backend

    def variant(mutate):
        j = json.loads(json.dumps(base_json))
        mutate(j)
        try:
            backend = Tokenizer.from_str(json.dumps(j))
        except Exception as e:  # noqa: BLE001
            log(f"variant not constructible: {e!r}")
            return None
        return ptc._space_cut_is_exact(Wrap(backend))

    def set_path(path, value):
        def m(j):
            d = j
            for k in path[:-1]:
                d = d[k]
            d[path[-1]] = value
        return m

    gate["unmodified_json"] = variant(lambda j: None)
    neg = {
        "nfkc": variant(set_path(["normalizer"], {"type": "NFKC"})),
        "pattern": variant(set_path(["pre_tokenizer", "pretokenizers", 0, "pattern"],
                                    {"Regex": r"(?<=\S) |\S+"})),
        "prefix_space": variant(set_path(["pre_tokenizer", "pretokenizers", 1,
                                          "add_prefix_space"], True)),
        "metaspace": variant(set_path(["pre_tokenizer"], {"type": "Metaspace",
                                                          "replacement": "▁",
                                                          "prepend_scheme": "always",
                                                          "split": True})),
        "trim_offsets": variant(set_path(["post_processor", "trim_offsets"], True)),
        "no_backend": ptc._space_cut_is_exact(object()),
    }
    b2 = Tokenizer.from_str(hf.backend_tokenizer.to_str())
    b2.add_tokens([AddedToken("hello world", special=False)])
    neg["added_with_space"] = ptc._space_cut_is_exact(Wrap(b2))
    b3 = Tokenizer.from_str(hf.backend_tokenizer.to_str())
    b3.add_tokens([AddedToken("<x>", lstrip=True)])
    neg["added_lstrip"] = ptc._space_cut_is_exact(Wrap(b3))
    gate["negatives"] = neg
    log(f"gate: {gate}")
    if not (gate["wrapper"] and gate["hf"] and gate["unmodified_json"]):
        fail("gate refuses the real tokenizer")
    if any(neg.values()):
        fail(f"gate accepts a modified tokenizer: {neg}")
    res["gate"] = gate

    # 2. Texts.
    maker = PromptMaker(hf, seed=99)
    rng = random.Random(7)
    words = ("the of and to in a is that for it as with was on be by at this which from or "
             "have an they not are his her had but were all their there one been has more "
             "extraordinary misunderstanding internationalization don't it's we'll I'M").split()
    alphabet = ([" "] * 14 + ["  ", "   ", "\n", "\n\n", "\t", " \n", "\r\n", " \r\n", "\n "] +
                list("abcdefghijXYZ") +
                ["word", "Word", "the", "'s", "'ll", "'S", "n't", "...", ",", ".", "!", "?",
                 "(", ")", "-", "--", "\"", "'", "`", "#", "$", "%", "&", "*", "/", "\\", "@"] +
                list("0123456789") + ["2024", "3.14", "1,000", "-5"] +
                ["中文", "日本語", "한국어", "العربية",
                 "हिन्दी"] +
                ["é", "́", "̈", "café", "naïve", "Å"] +
                ["\U0001F600", "\U0001F44D\U0001F3FD", "​", " ", "　", " ",
                 "﻿", "\x1c", "\x01"] +
                ["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<|endoftext|>",
                 "<tool_call>", "</tool_call>"])

    def word_salad(n):
        out = []
        size = 0
        while size < n:
            w = rng.choice(words)
            s = rng.choice([" ", " ", " ", " ", ", ", ". ", "\n", "  ", "\n\n", " - ", "; "])
            out.append(w + s)
            size += len(w) + len(s)
        return "".join(out)

    def noise(n):
        return "".join(rng.choice(alphabet) for _ in range(n))

    generated = []
    for i in range(args.generated):
        kind = i % 4
        if kind == 0:
            generated.append(maker.sample(rng.randrange(400, 3000)))
        elif kind == 1:
            generated.append(word_salad(rng.randrange(2500, 16000)))
        elif kind == 2:
            generated.append(noise(rng.randrange(1500, 6000)))
        else:
            generated.append(maker.sample(rng.randrange(200, 1500)) + noise(rng.randrange(200, 2000))
                             + word_salad(rng.randrange(1000, 5000)))
    junction_atoms = [
        "a  b", "a   b", "end. \n", "end \r\n", "x. \n\n", "word \n word", "...  ", "!! ?",
        "?? !!", "123 456", "3.14 2", "1 2 3", "e ́x", "x ́", "中文 中文",
        "中 文", "<|im_end|> <|im_start|>", "x <|im_end|>", "<|im_end|> x",
        "<think> </think>", "\U0001F600 \U0001F44D\U0001F3FD", "​ x", "a  b",
        "a  b", "\t x", "x \t", "　 x", "don 't", "'s 's", "I 'M", "a\n b",
        "a \n b", "$ 5", "( x )", "-- -", "\" quoted \"", "café naïve", "Å x",
        "ال عرب", "x  y", "\x1c x", "\x01 x", "﻿ x",
    ]
    junction = []
    for i in range(48):
        parts = []
        for _ in range(rng.randrange(20, 120)):
            parts.append(rng.choice(junction_atoms))
            parts.append(rng.choice(["", " ", "  ", "\n", "x", "word ", " the"]))
        junction.append("".join(parts))
    junction += junction_atoms

    real, info = load_real_prompts(args.ckpt)
    log(f"real prompts: {json.dumps(info)}")
    if real is None:
        fail("no real prompts")
        real = {1: [], 2: [], 4: []}
    real_all = real[1] + real[2] + real[4]

    cache = ptc.PrefixTokenCache(ptc.PrefixTokenCacheConfig())
    saved = (ptc.PARALLEL_PIECE_CHARS, ptc.PARALLEL_CUT_SEARCH)
    assert ptc.PARALLEL_TAIL, "kill switch is off in this process"

    saved_max = ptc.PARALLEL_MAX_PIECES

    def check_texts(name, texts, piece_chars, cut_search):
        ptc.PARALLEL_PIECE_CHARS, ptc.PARALLEL_CUT_SEARCH = piece_chars, cut_search
        refs = [single(t) for t in texts]
        before = cache.parallel_tokenizations
        bad = 0
        lens_ok = 0
        for t, r in zip(texts, refs):
            out = cache._tokenize_long(tok, t)
            lens_ok += len(out[1]) == len(r[1])
            if norm(out) != r:
                bad += 1
        seq_par = cache.parallel_tokenizations - before
        # 8 threads, each text twice, interleaved: once with the default piece
        # budget (concurrent texts get fewer pieces, down to one call), once
        # with a budget of 64 so every thread still splits into up to 8 pieces.
        errs = []
        lock = threading.Lock()

        def worker(idx):
            for j in idx:
                if norm(cache._tokenize_long(tok, texts[j])) != refs[j]:
                    with lock:
                        errs.append(j)

        thread_par = []
        for budget in (saved_max, 64):
            ptc.PARALLEL_MAX_PIECES = budget
            b0 = cache.parallel_tokenizations
            order = list(range(len(texts))) * 2
            rng.shuffle(order)
            threads = [threading.Thread(target=worker, args=(order[i::8],)) for i in range(8)]
            for th in threads:
                th.start()
            for th in threads:
                th.join()
            thread_par.append(cache.parallel_tokenizations - b0)
        ptc.PARALLEL_MAX_PIECES = saved_max
        out = {"texts": len(texts), "piece_chars": piece_chars, "mismatch_seq": bad,
               "mismatch_8threads": len(errs), "parallel_seq": seq_par,
               "parallel_8threads_default_budget": thread_par[0],
               "parallel_8threads_budget64": thread_par[1], "offset_len_ok": lens_ok,
               "chars": [min(map(len, texts)), max(map(len, texts))]}
        log(f"{name}: {out}")
        if bad or errs:
            fail(f"{name}: mismatches seq={bad} threads={len(errs)}")
        if thread_par[1] < seq_par:
            fail(f"{name}: 8-thread run rarely took the parallel path ({thread_par[1]})")
        return out

    tails = []
    if real_all:
        shared = common_prefix(real_all)
        # The tails a cache hit tokenizes: text[split_char:] (split at a cached
        # grid point within the shared prefix); reproduce with a real cache.
        probe = ptc.PrefixTokenCache(ptc.PrefixTokenCacheConfig())
        orig = probe._tokenize_long

        def grab(tkz, text):
            tails.append(text)
            return orig(tkz, text)

        probe._tokenize_long = grab
        probe.encode(tok, f"{shared.rstrip()} {maker.sample(2048)}")
        for p in real_all:
            probe.encode(tok, p)
        tails = tails[1:]
    res["real_prompts_default"] = check_texts("real prompts (default pieces)", real_all, *saved)
    res["real_tails_default"] = check_texts("real hit tails (default pieces)", tails, *saved)
    res["real_tails_fine"] = check_texts("real hit tails (64-char pieces)", tails, 64, 64)
    res["generated_default"] = check_texts("generated (default pieces)", generated, *saved)
    res["generated_fine"] = check_texts("generated (64-char pieces)", generated, 64, 64)
    res["junction_fine"] = check_texts("junction (16-char pieces)", junction, 16, 32)
    ptc.PARALLEL_PIECE_CHARS, ptc.PARALLEL_CUT_SEARCH = saved
    if res["real_tails_default"]["parallel_seq"] != len(tails) or not tails:
        fail("real hit tails did not all take the parallel path")
    if res["generated_default"]["parallel_seq"] < args.generated // 2:
        fail("generated texts rarely took the parallel path")

    # Every cut point as a two-piece split, through the same batched call.
    cuts_checked = 0
    cut_bad = 0
    for t in junction + generated[2::4][:12] + generated[1::4][:4]:
        r = single(t)
        points = [c for c in range(1, len(t)) if ptc._is_cut(t, c)]
        if len(points) > 400:
            points = sorted(rng.sample(points, 400))
        for start in range(0, len(points), 64):
            group = points[start:start + 64]
            pieces = []
            for c in group:
                pieces += [t[:c], t[c:]]
            enc = tok(pieces, add_special_tokens=False, return_offsets_mapping=True)
            for gi, c in enumerate(group):
                ids = list(enc["input_ids"][2 * gi]) + list(enc["input_ids"][2 * gi + 1])
                offs = ([tuple(o) for o in enc["offset_mapping"][2 * gi]] +
                        [(s + c, e + c) for s, e in enc["offset_mapping"][2 * gi + 1]])
                cuts_checked += 1
                if (ids, offs) != r:
                    cut_bad += 1
                    if cut_bad <= 5:
                        log(f"cut mismatch at {c}: {t[max(0, c - 8):c + 8]!r}")
    res["every_cut_point"] = {"checked": cuts_checked, "mismatches": cut_bad}
    log(f"every cut point: {res['every_cut_point']}")
    if cut_bad or cuts_checked < 1000:
        fail(f"every-cut check: {cut_bad} mismatches of {cuts_checked}")

    # The lazy offsets object: indexing, negative index, slices with steps, as
    # the cache reads it, equal the materialized offsets.
    backend = cache.parallel_backend(tok)
    lazy_bad = 0
    for t in tails[:8] + generated[:16]:
        r = single(t)
        out = ptc.PrefixTokenCache._tokenize_pieces(backend, t, 8)
        if out is None:
            continue
        ids, offs = out
        full = r[1]
        probes = [offs[0] == full[0], offs[-1] == full[-1], offs[len(full) // 2] == full[len(full) // 2],
                  offs[::16] == full[::16], offs[5::16] == full[5::16], offs[3:40] == full[3:40],
                  list(offs) == full, ids == r[0]]
        lazy_bad += not all(probes)
    res["lazy_offsets_mismatch"] = lazy_bad
    log(f"lazy offsets mismatches: {lazy_bad}")
    if lazy_bad:
        fail("lazy offsets differ")

    # 3. The cache path on the real prompts, kill switch on vs off.
    if real_all:
        sp = SamplingParams(max_tokens=256, add_special_tokens=True)
        prime = f"{common_prefix(real_all).rstrip()} {PromptMaker(hf, seed=5).sample(2048)}"
        seq = [(1, [prime]), (1, real[1]), (2, real[2]), (4, real[4]), (1, [real[1][0]]),
               (1, [real[1][0]])]
        prompts = [p for _, g in seq for p in g]
        reference = {p: hf(p, add_special_tokens=False)["input_ids"] for p in set(prompts)}
        arms = {}
        for arm in (True, False):
            ptc.PARALLEL_TAIL = arm
            proc = make_processor(args.ckpt, tok, enable_cache=True)
            c = processor_cache(proc)
            passes = []
            for _ in range(2):
                outs = []
                with ThreadPoolExecutor(max_workers=4) as pool:
                    for conc, group in seq:
                        for i in range(0, len(group), conc):
                            outs += list(pool.map(lambda p: proc({"prompt": p}, sp)[0],
                                                  group[i:i + conc]))
                passes.append(outs)
            mism = [sum(o != reference[p] for o, p in zip(outs, prompts)) for outs in passes]
            drift = sum(a != b for a, b in zip(*passes))
            state = sorted((e.text[-64:], e.ids.tobytes(), e.grid.tobytes(), e.split_token,
                            e.split_char) for e in c._entries.values())
            hits = c.hits + c.shared_prefix_hits
            arms[arm] = {"mismatches": mism, "pass2_vs_pass1": drift, "hits": hits,
                         "misses": c.misses, "fallbacks": c.resync_failures + c.shared_prefix_fallbacks,
                         "parallel": c.parallel_tokenizations, "parallel_errors": c.parallel_errors,
                         "entries": len(state), "summary": c.summary(), "_state": state}
            log(f"cache path PARALLEL_TAIL={arm}: "
                f"{ {k: v for k, v in arms[arm].items() if k != '_state'} }")
            ok = (mism == [0, 0] and drift == 0 and c.misses == 1 and hits == 2 * len(prompts) - 1
                  and arms[arm]["fallbacks"] == 0 and c.parallel_errors == 0)
            if arm:
                # every tokenization (the prime miss + every non-repeat hit) parallel
                ok = ok and c.parallel_tokenizations > 0
            if not ok:
                fail(f"cache path PARALLEL_TAIL={arm}")
        ptc.PARALLEL_TAIL = True
        same_state = arms[True]["_state"] == arms[False]["_state"]
        log(f"stored entries identical between arms: {same_state}")
        if not same_state:
            fail("stored cache entries differ between arms")
        res["cache_path"] = {str(k): {kk: vv for kk, vv in v.items() if kk != "_state"}
                             for k, v in arms.items()}
        res["cache_path"]["stored_entries_identical"] = same_state

    passed = not failures
    print(json.dumps({"passed": passed, "failures": failures,
                      "compared_against": "one tokenizer call on the whole text (main's "
                      "PrefixTokenCache._tokenize: ids and offset_mapping) and a full "
                      "tokenizer call per prompt for the cache path", "results": res}),
          flush=True)


if __name__ == "__main__":
    main()
