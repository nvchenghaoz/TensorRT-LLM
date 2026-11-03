"""Exactness replay for the shared-prefix (LCP) splice in tensorrt_llm/inputs/prefix_token_cache.py.

CPU only, no model load. Usage:
    python experiments/ascent/tok_cache_check.py <checkpoint> [--skip-pytest]

1. pytest tests/unittest/inputs/test_prefix_token_cache.py (protected, unchanged).
2. Quotes the tokenizer.json fields the splice gate relies on.
3. Builds the DeepseekV41InputProcessor through create_input_processor(enable_tokenization_cache=True)
   (the server's path) and replays, in server order:
   - benchmark-shaped prompts: BOS + one ~32768-token shared prefix, then 2048-token suffixes whose
     first token differs (prime, warmups, measured, check + repeat, accuracy);
   - >= 300 random variants over a second prefix dense with digits, CJK/kana, whitespace runs,
     newlines, tabs, special tokens, BOS text, combining marks and emoji, each diverging at a random
     point (so the cut lands in that material) or carrying it right after the divergence.
   Every result is compared with tokenizer(text, add_special_tokens=True).input_ids.
4. Every chosen cut is re-verified independently: the cut is a token start of the full
   tokenization and ids(text[:cut]) + ids(text[cut:]) == ids(text); plus a fuzz of the cut rule.
Last stdout line: {"passed": ..., ...}.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time

T0 = time.time()


def log(msg):
    print(f"[+{time.time() - T0:6.1f}s] {msg}", flush=True)


def main():
    ckpt = sys.argv[1]
    skip_pytest = "--skip-pytest" in sys.argv
    result = {"passed": False}

    # ---------------------------------------------------------------- 1. protected test
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

    # ---------------------------------------------------------------- 2. tokenizer.json quote
    spec = json.load(open(os.path.join(ckpt, "tokenizer.json"), encoding="utf-8"))
    quote = {k: spec.get(k) for k in ("normalizer", "pre_tokenizer", "post_processor")}
    print("tokenizer.json:", json.dumps(quote, ensure_ascii=False), flush=True)
    ws_added = [a["content"] for a in spec.get("added_tokens", [])
                if any(c.isspace() for c in a["content"])]
    strip_added = [a["content"] for a in spec.get("added_tokens", [])
                   if a.get("lstrip") or a.get("rstrip") or a.get("single_word")]
    print(f"added tokens: {len(spec.get('added_tokens', []))}, with whitespace: {ws_added[:5]}, "
          f"strip/single_word: {strip_added[:5]}", flush=True)

    # ---------------------------------------------------------------- 3. processor
    from tensorrt_llm.inputs import prefix_token_cache as ptc
    from tensorrt_llm.inputs.registry import create_input_processor
    from tensorrt_llm.sampling_params import SamplingParams
    from tensorrt_llm.tokenizer import load_hf_tokenizer

    tok = load_hf_tokenizer(ckpt, trust_remote_code=True)
    hf = tok.tokenizer
    log(f"tokenizer {type(hf).__name__} fast={hf.is_fast} "
        f"num_special_tokens_to_add={hf.num_special_tokens_to_add()}")
    gate = ptc._lcp_splice_gate(hf)
    log(f"gate: {gate}")
    result["gate"] = gate[1]
    proc_ = create_input_processor(ckpt, tok, enable_tokenization_cache=True)
    log(f"processor {type(proc_).__name__}")
    cache = getattr(proc_, "_prefix_token_cache", None)
    if cache is None or not cache._lcp_splice:
        print(json.dumps(dict(result, passed=False, reason="cache missing or splice off")))
        return

    # record every chosen cut
    cuts = []
    orig_lookup = cache._lcp_lookup

    def recording_lookup(text):
        r = orig_lookup(text)
        if r is not None and r[1] >= 0:
            cuts.append((text, r[2], r[1]))
        return r

    cache._lcp_lookup = recording_lookup
    sp = SamplingParams(max_tokens=1)  # add_special_tokens defaults to True, as /v1/completions
    assert sp.add_special_tokens

    bos = hf.bos_token or "<｜begin▁of▁sentence｜>"
    specials = sorted({t for t in hf.all_special_tokens if t} | {bos})
    vocab = hf.get_vocab()
    words = sorted({hf.convert_tokens_to_string([t]) for t in vocab
                    if t.startswith("Ġ") and t[1:].isalpha() and t[1:].isascii()
                    and 3 <= len(t) - 1 <= 10})
    caps = sorted({w for w in words if w[1].isupper()})
    log(f"{len(words)} space-led words, {len(caps)} capitalized; bos={bos!r}; "
        f"{len(specials)} special tokens")
    rng = random.Random(1234)

    def prose(rng, n_words):
        out = []
        for _ in range(n_words):
            out.append(rng.choice(words))
            r = rng.random()
            if r < 0.06:
                out.append(".")
            elif r < 0.09:
                out.append(",")
            elif r < 0.10:
                out.append("\n\n")
            elif r < 0.11:
                out.append(f" {rng.randint(0, 99999)}")
        return "".join(out)

    tricky_bits = [
        lambda r: f" {r.randint(0, 10**r.randint(1, 9))}",
        lambda r: f"{r.randint(0, 999)}{r.randint(0, 999)} ",
        lambda r: " " * r.randint(2, 6),
        lambda r: "\t" * r.randint(1, 3) + " ",
        lambda r: "\n" * r.randint(1, 3) + " " * r.randint(0, 3),
        lambda r: " 中文字符",
        lambda r: "日本語のテキスト ",
        lambda r: " ひらがな" + r.choice(["", " ", "。"]),
        lambda r: r.choice(specials) + r.choice(["", " ", "\n"]),
        lambda r: " été café ",
        lambda r: " \U0001f600\U0001f680 " + r.choice(["", "‍\U0001f525"]),
        lambda r: " ½ Ⅳ ٣٤ ",
        lambda r: " (x) [y] {z}; \"q\" 'r' ",
        lambda r: r.choice([" ", "　", " ", "​"]) + " ",
        lambda r: " " + r.choice(caps).strip() + r.choice([":", "?", "!", "..."]),
    ]

    def tricky(rng, n_items):
        out = []
        for _ in range(n_items):
            if rng.random() < 0.5:
                out.append(rng.choice(tricky_bits)(rng))
            else:
                out.append(prose(rng, rng.randint(1, 8)))
        return "".join(out)

    def fit_tokens(text, n):
        offs = hf(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        if len(offs) <= n:
            return text
        return text[: offs[n][0]]

    # shared prefix ~32768 tokens, BOS text first (as the harness)
    prefix = fit_tokens(bos + prose(rng, 40000), 32768)
    log(f"benchmark prefix: {len(prefix)} chars, "
        f"{len(hf(prefix, add_special_tokens=True).input_ids)} tokens")

    leads = rng.sample(caps, 200)

    def suffix(i):
        return fit_tokens(leads[i] + prose(rng, 2600), 2048)

    prompts = []
    # prime, warmups, measured c1/c2/c4, check 30 + 9 repeats, accuracy 40
    names = ["prime"] + [f"warm{i}" for i in range(6)] + [f"meas{i}" for i in range(52)] \
        + [f"check{i}" for i in range(30)]
    for i, name in enumerate(names):
        prompts.append((name, prefix + suffix(i)))
    for i in range(9):
        prompts.append((f"repeat{i}", prompts[len(names) - 30 + i][1]))
    for i in range(40):
        prompts.append((f"acc{i}", prefix + suffix(len(names) + i)))
    n_bench = len(prompts)

    # tricky prefix and variants
    tprefix = fit_tokens(bos + tricky(rng, 9000), 32768)
    log(f"tricky prefix: {len(tprefix)} chars")
    for i in range(320):
        mode = i % 4
        if mode in (0, 1):
            d = rng.randint(len(tprefix) // 2 + 600, len(tprefix) - 1)
            text = tprefix[:d] + rng.choice(tricky_bits)(rng) + tricky(rng, rng.randint(5, 400))
        elif mode == 2:
            text = tprefix + rng.choice(tricky_bits)(rng) + tricky(rng, rng.randint(50, 600))
        else:
            # divergence right inside a digit / CJK / whitespace / combining run
            d = rng.randint(len(tprefix) // 2 + 600, len(tprefix) - 1)
            text = tprefix[:d] + rng.choice(["7", "中", " ", "\t", "́", "\n"]) \
                + tricky(rng, rng.randint(5, 300))
        prompts.append((f"var{i}", text))
    for i in range(8):
        prompts.append((f"short{i}", tricky(rng, rng.randint(3, 40))))
    log(f"{len(prompts)} prompts ({n_bench} benchmark-shaped)")

    mismatches = []
    spliced_ms, full_ms = [], []
    off_len = 0
    stock_resync_names = []  # stock-lookup resync fallbacks (stock guard; full re-tokenization)
    for k, (name, text) in enumerate(prompts):
        before = cache.splices
        before_resync = cache.resync_failures
        t = time.perf_counter()
        ids, _ = proc_.call_with_text_prompt({"prompt": text}, sp)
        ms = (time.perf_counter() - t) * 1e3
        (spliced_ms if cache.splices > before else full_ms).append(ms)
        if cache.resync_failures > before_resync:
            stock_resync_names.append(name)
            print(f"stock-lookup resync fallback on {name} (full tokenization)", flush=True)
        ref = hf(text, add_special_tokens=True).input_ids
        if list(ids) != list(ref):
            mismatches.append(name)
            print(f"MISMATCH {name}: len {len(ids)} vs {len(ref)}", flush=True)
        if k < n_bench and len(ref) != 34816:
            off_len += 1
        if k % 50 == 0:
            log(f"{k}/{len(prompts)} {name}: {cache.summary()}")
    log(f"replay done: {cache.summary()}; benchmark-shaped prompts not 34816 tokens: {off_len}")

    # ---------------------------------------------------------------- 4. independent cut checks
    violations = 0
    for text, cut, token in cuts:
        enc = hf(text, add_special_tokens=False, return_offsets_mapping=True)
        starts = [o[0] for o in enc["offset_mapping"]]
        left = hf(text[:cut], add_special_tokens=False).input_ids
        right = hf(text[cut:], add_special_tokens=False).input_ids
        ok = (text[cut] == " " and not text[cut - 1].isspace()
              and not text[cut - 1].isdigit() and token < len(starts) and starts[token] == cut
              and list(left) + list(right) == list(enc["input_ids"]))
        if not ok:
            violations += 1
            print(f"CUT VIOLATION at {cut} ({text[cut - 3:cut + 3]!r})", flush=True)
    log(f"{len(cuts)} chosen cuts checked, {violations} violations")

    # fuzz the rule itself on windows of the tricky text
    fuzz, fuzz_bad = 0, 0
    src = tprefix + tricky(rng, 3000)
    while fuzz < 2000:
        a = rng.randint(0, len(src) - 3000)
        w = src[a:a + rng.randint(200, 3000)]
        c = rng.randint(1, len(w) - 1)
        while c < len(w) and not ptc._is_exact_cut(w, c):
            c += 1
        if c >= len(w):
            continue
        fuzz += 1
        full = hf(w, add_special_tokens=False).input_ids
        if list(hf(w[:c], add_special_tokens=False).input_ids) + \
                list(hf(w[c:], add_special_tokens=False).input_ids) != list(full):
            fuzz_bad += 1
            print(f"FUZZ VIOLATION {w[c - 3:c + 3]!r}", flush=True)
    log(f"rule fuzz: {fuzz} cuts, {fuzz_bad} violations")

    def med(xs):
        return round(sorted(xs)[len(xs) // 2], 3) if xs else None

    result.update({
        "prompts": len(prompts),
        "benchmark_shaped": n_bench,
        "variants": len(prompts) - n_bench,
        "mismatches": len(mismatches),
        "mismatch_names": mismatches[:20],
        "hits": cache.hits,
        "misses": cache.misses,
        "splices": cache.splices,
        "splice_resync_fallbacks": cache.splice_fallbacks,
        "stock_resync_fallbacks": cache.resync_failures,
        "stock_resync_fallback_prompts": stock_resync_names,
        "stock_resync_fallbacks_benchmark_shaped": sum(
            1 for n in stock_resync_names if not n.startswith(("var", "short"))),
        "splice_no_cut": cache.splice_no_cut,
        "cuts_checked": len(cuts),
        "cut_violations": violations,
        "rule_fuzz_cuts": fuzz,
        "rule_fuzz_violations": fuzz_bad,
        "disabled": cache.disabled,
        "spliced_call_ms_median": med(spliced_ms),
        "full_call_ms_median": med(full_ms),
        "summary": cache.summary(),
    })
    result["passed"] = bool(
        len(mismatches) == 0 and violations == 0 and fuzz_bad == 0 and not cache.disabled
        and cache.splices > 0 and cache.splice_fallbacks == 0
        and result["stock_resync_fallbacks_benchmark_shaped"] == 0
        and (skip_pytest or result.get("pytest_returncode") == 0))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
