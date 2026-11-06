"""Exactness / wiring / kill-switch check for parallel tail tokenization (branch 233, idea 133).

CPU only, no model load:
    python experiments/ascent/tok_parallel/check.py <checkpoint> [--skip-pytest]

1. pytest tests/unittest/inputs/test_prefix_token_cache.py (protected, unchanged).
2. Builds the DeepseekV41InputProcessor via create_input_processor(enable_tokenization_cache=True),
   the server's path, and replays in server order benchmark-shaped prompts (prime, warmups,
   measured, check + repeats, accuracy) plus >= 320 seam-stress variants and short prompts,
   3 passes in one process on the same processor. Every result is compared with the single-call
   encode tokenizer(text, add_special_tokens=True).input_ids ("main's ids").
3. Every piece seam _piece_cuts chose is checked against the DeepSeek U+0020 rule independently
   (text[c] == ' ', previous char not whitespace / \\p{N} / CJK-kana / zero-width) and, for every
   distinct seam, ids(text[:c]) + ids(text[c:]) == ids(text) by the HF single call.
4. Every parallel tail's lazy offsets are compared in full with the single call's offset_mapping.
5. Concurrency: 4-thread barrier waves (texts in flight > 1 -> fewer pieces), ids compared.
6. Kill switch: PARALLEL_TAIL = False (module attribute) on a fresh processor replays pass 1; it
   must make 0 parallel calls, return the same ids and build byte-identical cache entries
   (text, ids, split_token, split_char, grid) as the parallel processor's pass 1.
Last stdout line: {"passed": ..., ...}.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import threading
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import PromptGen, host_receipt, log  # noqa: E402


def seam_rule_ok(text, c):
    if not (0 < c < len(text)) or text[c] != " ":
        return False
    prev = text[c - 1]
    o = ord(prev)
    cjk = (0x2E80 <= o <= 0x9FFF or 0xF900 <= o <= 0xFAFF or 0xFE30 <= o <= 0xFE4F
           or 0xFF00 <= o <= 0xFFEF or o >= 0x20000)
    return not (prev.isspace() or prev in "​⁠﻿"
                or unicodedata.category(prev)[0] == "N" or cjk)


def snapshot(cache):
    with cache._lock:
        return [(e.text, e.ids.tobytes(), e.split_token, e.split_char,
                 e.grid.tobytes() if e.grid is not None else None)
                for e in cache._entries.values()]


def main():
    ckpt = sys.argv[1]
    skip_pytest = "--skip-pytest" in sys.argv
    result = {"passed": False, "host": host_receipt()}
    log(f"host receipt: {result['host']}")

    if not skip_pytest:
        log("running pytest tests/unittest/inputs/test_prefix_token_cache.py")
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
             "tests/unittest/inputs/test_prefix_token_cache.py"],
            capture_output=True, text=True, timeout=900)
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-15:]
        for line in tail:
            print("  pytest| " + line, flush=True)
        result["pytest_returncode"] = proc.returncode
        result["pytest_tail"] = tail[-1] if tail else ""

    from tensorrt_llm.inputs import prefix_token_cache as ptc
    from tensorrt_llm.inputs.registry import create_input_processor
    from tensorrt_llm.sampling_params import SamplingParams
    from tensorrt_llm.tokenizer import load_hf_tokenizer

    assert ptc.PARALLEL_TAIL, "TLLM_PREFIX_TOKEN_CACHE_PARALLEL must be unset for this check"
    tok = load_hf_tokenizer(ckpt, trust_remote_code=True)
    hf = tok.tokenizer
    log(f"tokenizer {type(hf).__name__}; parallel gate: "
        f"{ptc._parallel_tail_gate(hf)[1]}; splice gate: {ptc._lcp_splice_gate(hf)}")

    # Record every seam and every parallel tail.
    seams = []  # (text, cuts)
    orig_cuts = ptc._piece_cuts

    def rec_cuts(text, n):
        cuts = orig_cuts(text, n)
        seams.append((text, tuple(cuts), n))
        return cuts

    ptc._piece_cuts = rec_cuts

    sp = SamplingParams(max_tokens=1)
    assert sp.add_special_tokens
    gen = PromptGen(hf, random.Random(1234))
    prefix = gen.prefix()
    leads = gen.rng.sample(gen.caps, 200)
    names = (["prime"] + [f"warm{i}" for i in range(6)] + [f"meas{i}" for i in range(52)]
             + [f"check{i}" for i in range(30)])
    prompts = [(n, prefix + gen.suffix(leads[i])) for i, n in enumerate(names)]
    for i in range(9):
        prompts.append((f"repeat{i}", prompts[len(names) - 30 + i][1]))
    for i in range(40):
        prompts.append((f"acc{i}", prefix + gen.suffix(leads[len(names) + i])))
    n_bench = len(prompts)
    tprefix = gen.tricky_prefix()
    for i in range(320):
        prompts.append((f"var{i}", gen.variant(tprefix, i)))
    for i in range(8):
        prompts.append((f"short{i}", gen.tricky(gen.rng.randint(3, 40))))
    log(f"{len(prompts)} prompts ({n_bench} benchmark-shaped); computing single-call references")
    refs = {name: list(hf(text, add_special_tokens=True).input_ids) for name, text in prompts}
    off34816 = sum(1 for n, _ in prompts[:n_bench] if len(refs[n]) != 34816)
    log(f"references done; benchmark-shaped prompts not 34816 tokens: {off34816}")

    proc_ = create_input_processor(ckpt, tok, enable_tokenization_cache=True)
    cache = proc_._prefix_token_cache
    result["processor"] = type(proc_).__name__
    if cache is None or not cache._lcp_splice or cache._parallel_backend is None:
        print(json.dumps(dict(result, reason="cache, splice or parallel backend missing")))
        return

    offset_checks = offset_bad = 0
    orig_tail = cache._tokenize_tail

    def checked_tail(tokenizer, text):
        nonlocal offset_checks, offset_bad
        ids, offs = orig_tail(tokenizer, text)
        if isinstance(offs, ptc._PieceOffsets):
            enc = hf(text, add_special_tokens=False, return_offsets_mapping=True)
            offset_checks += 1
            if list(ids) != list(enc["input_ids"]) or \
                    offs[0:] != [tuple(o) for o in enc["offset_mapping"]] or \
                    len(offs) != len(enc["offset_mapping"]):
                offset_bad += 1
                print(f"OFFSET/ID MISMATCH on a {len(text)}-char tail", flush=True)
        return ids, offs

    cache._tokenize_tail = checked_tail

    mismatches = []
    pass_stats = []
    pass1_snapshot = None
    for rep in range(3):
        before = (cache.splices, cache.hits, cache.parallel_tails)
        for k, (name, text) in enumerate(prompts):
            ids, _ = proc_.call_with_text_prompt({"prompt": text}, sp)
            if list(ids) != refs[name]:
                mismatches.append(f"pass{rep}:{name}")
                print(f"MISMATCH pass{rep} {name}: len {len(ids)} vs {len(refs[name])}", flush=True)
            if k % 100 == 0:
                log(f"pass {rep} {k}/{len(prompts)}: {cache.summary()}")
        pass_stats.append({"splices": cache.splices - before[0], "hits": cache.hits - before[1],
                           "parallel_tails": cache.parallel_tails - before[2]})
        if rep == 0:
            pass1_snapshot = snapshot(cache)
        log(f"pass {rep} done: {pass_stats[-1]}; {cache.summary()}")
    seams_main = list(seams)

    # Concurrency waves: texts in flight > 1.
    conc_bad = 0
    conc_calls = 0
    for w in range(20):
        texts = [prefix + gen.suffix(gen.rng.choice(gen.caps)) for _ in range(4)]
        if w % 2:
            texts = [gen.variant(tprefix, gen.rng.randint(0, 999)) for _ in range(4)]
        barrier = threading.Barrier(4)
        outs = [None] * 4

        def worker(j):
            barrier.wait()
            outs[j] = proc_.call_with_text_prompt({"prompt": texts[j]}, sp)[0]

        ths = [threading.Thread(target=worker, args=(j,)) for j in range(4)]
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        for j in range(4):
            conc_calls += 1
            if list(outs[j]) != list(hf(texts[j], add_special_tokens=True).input_ids):
                conc_bad += 1
                print(f"CONCURRENT MISMATCH wave {w} thread {j}", flush=True)
    piece_counts = sorted({n for _, _, n in seams[len(seams_main):]})
    log(f"concurrency: {conc_calls} calls, {conc_bad} mismatches, n_pieces seen {piece_counts}")

    # Seam rule.
    rule_bad = 0
    n_seams = 0
    distinct = {}
    for text, cuts, _ in seams:
        prev = 0
        for c in cuts:
            n_seams += 1
            if not seam_rule_ok(text, c) or c - prev < ptc.PARALLEL_PIECE_CHARS:
                rule_bad += 1
                print(f"SEAM RULE VIOLATION at {c}: {text[max(c - 3, 0):c + 3]!r}", flush=True)
            prev = c
        if cuts and len(text) - cuts[-1] < ptc.PARALLEL_PIECE_CHARS:
            rule_bad += 1
        for c in cuts:
            distinct.setdefault((hash(text), c), (text, c))
    log(f"{n_seams} seams checked against the U+0020 rule, {rule_bad} violations; "
        f"verifying {len(distinct)} distinct seams by split identity")
    split_bad = 0
    full_cache = {}
    for i, (text, c) in enumerate(distinct.values()):
        key = hash(text)
        if key not in full_cache:
            full_cache[key] = list(hf(text, add_special_tokens=False).input_ids)
        left = list(hf(text[:c], add_special_tokens=False).input_ids)
        right = list(hf(text[c:], add_special_tokens=False).input_ids)
        if left + right != full_cache[key]:
            split_bad += 1
            print(f"SEAM SPLIT VIOLATION at {c}: {text[max(c - 3, 0):c + 3]!r}", flush=True)
        if i % 500 == 0:
            log(f"split identity {i}/{len(distinct)}")
    log(f"split identity: {split_bad} violations")

    # Kill switch.
    ptc._piece_cuts = orig_cuts
    ptc.PARALLEL_TAIL = False
    kproc = create_input_processor(ckpt, tok, enable_tokenization_cache=True)
    kcache = kproc._prefix_token_cache
    kill_bad = 0
    for name, text in prompts:
        ids, _ = kproc.call_with_text_prompt({"prompt": text}, sp)
        if list(ids) != refs[name]:
            kill_bad += 1
    kill_snapshot = snapshot(kcache)
    kill_entries_equal = kill_snapshot == pass1_snapshot
    # Flip on the already-built parallel processor too.
    before_tails = cache.parallel_tails
    for name, text in prompts[:20]:
        if list(proc_.call_with_text_prompt({"prompt": text}, sp)[0]) != refs[name]:
            kill_bad += 1
    flipped_tails = cache.parallel_tails - before_tails
    ptc.PARALLEL_TAIL = True
    log(f"kill switch: {kill_bad} mismatches, parallel_backend={kcache._parallel_backend}, "
        f"parallel_tails={kcache.parallel_tails}, entries equal: {kill_entries_equal} "
        f"({len(kill_snapshot)} vs {len(pass1_snapshot)}); flipped processor tails {flipped_tails}")

    result.update({
        "prompts_per_pass": len(prompts), "benchmark_shaped": n_bench,
        "variants": len(prompts) - n_bench, "passes": 3, "pass_stats": pass_stats,
        "benchmark_shaped_not_34816": off34816,
        "mismatches": len(mismatches), "mismatch_names": mismatches[:20],
        "offset_checks": offset_checks, "offset_mismatches": offset_bad,
        "seams_checked": n_seams, "seam_rule_violations": rule_bad,
        "distinct_seams_split_checked": len(distinct), "seam_split_violations": split_bad,
        "concurrent_calls": conc_calls, "concurrent_mismatches": conc_bad,
        "concurrent_piece_counts": piece_counts,
        "kill_switch_mismatches": kill_bad, "kill_switch_parallel_tails": kcache.parallel_tails,
        "kill_switch_backend_none": kcache._parallel_backend is None,
        "kill_switch_entries_equal": kill_entries_equal, "kill_switch_entries": len(kill_snapshot),
        "flipped_processor_parallel_tails": flipped_tails,
        "parallel_fallbacks": cache.parallel_fallbacks, "disabled": cache.disabled,
        "splice_resync_fallbacks": cache.splice_fallbacks, "summary": cache.summary(),
    })
    result["passed"] = bool(
        not mismatches and offset_bad == 0 and offset_checks > 0 and rule_bad == 0
        and split_bad == 0 and n_seams > 0 and conc_bad == 0 and kill_bad == 0
        and kcache.parallel_tails == 0 and kcache._parallel_backend is None
        and kill_entries_equal and flipped_tails == 0 and cache.parallel_fallbacks == 0
        and not cache.disabled and not kcache.disabled and cache.splice_fallbacks == 0
        and all(s["splices"] > 0 for s in pass_stats[:1])
        and (skip_pytest or result.get("pytest_returncode") == 0))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
