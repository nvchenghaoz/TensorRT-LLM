# SPDX-License-Identifier: Apache-2.0
"""Correctness check: token ids from the input processor with the prefix token
cache enabled vs a fresh full tokenizer call (main's text-only path).

Two request sequences, each replayed untimed through a fresh processor (cold
cache), for /v1/completions (add_special_tokens=True, the benchmark's case) and
chat (False), plus DefaultInputProcessor:
  real: the frozen benchmark's own prompts (proxy artifacts c1/c2/c4-input
        prompts.jsonl), after a prime request of the shared prefix plus a fresh
        suffix, in closed-loop waves of 1/2/4, then one prompt sent twice;
  bench: exactly tok_bench.py's sequence (same seed, warmups, measured).
A second pass over the same sequence on the now-warm cache must equal pass 1.
Every request is compared with the reference; the cache must hit on every
request after the first.

Last stdout line: {"passed": true|false, ...}
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from tok_common import (PromptMaker, hf_tokenizer, load_real_prompts, load_tokenizer, log,
                        make_processor, processor_cache, request_plan)


def common_prefix(texts):
    from tensorrt_llm.inputs.prefix_token_cache import _common_prefix_len
    p = texts[0]
    for t in texts[1:]:
        p = p[:_common_prefix_len(p, t)]
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--measured", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    import torch

    from tensorrt_llm.inputs.registry import DefaultInputProcessor
    from tensorrt_llm.sampling_params import SamplingParams

    tok = load_tokenizer(args.ckpt)
    hf = hf_tokenizer(tok)
    maker = PromptMaker(hf, seed=args.seed)

    sequences = {}
    real, info = load_real_prompts(args.ckpt)
    log(f"real prompts: {json.dumps(info)}")
    if real is not None:
        all_real = real[1] + real[2] + real[4]
        shared = common_prefix(all_real)
        log(f"real prompts share a {len(shared)}-char prefix; chars "
            f"{min(map(len, all_real))}-{max(map(len, all_real))}")
        prime = f"{shared.rstrip()} {maker.sample(2048)}"
        seq = [(1, [prime]), (1, real[1]), (2, real[2]), (4, real[4]), (1, [real[1][0]]),
               (1, [real[1][0]])]
        sequences["real"] = seq

    # tok_bench.py's sequence: fresh maker with the same seed, same call order.
    maker = PromptMaker(hf, seed=args.seed)
    prefix = maker.prefix()
    seq = [(1, [maker.request(prefix)])]
    for conc, w, m in request_plan(maker, prefix, args.warmup, args.measured):
        seq += [(conc, w), (conc, m)]
    rep = maker.request(prefix)
    seq += [(1, [rep]), (1, [rep])]
    sequences["bench"] = seq

    reference = {}
    for name, seq in sequences.items():
        prompts = [p for _, group in seq for p in group]
        for p in set(prompts) - set(reference):
            ids = hf(p, return_tensors="pt").input_ids[0].to(torch.int32).tolist()
            plain = hf(p, add_special_tokens=False)["input_ids"]
            assert ids == plain, "tokenizer adds special tokens?"
            reference[p] = ids
        log(f"{name}: {len(prompts)} requests ({len(set(prompts))} distinct), token counts "
            f"{min(len(reference[p]) for p in prompts)}-{max(len(reference[p]) for p in prompts)}")

    results = {}
    passed = True

    def replay(make, sp, label, seq):
        nonlocal passed
        proc = make()
        cache = processor_cache(proc)
        prompts = [p for _, group in seq for p in group]
        passes = []
        cold_misses = None
        for pass_index in range(2):
            outs = []
            with ThreadPoolExecutor(max_workers=4) as pool:
                for gi, (conc, group) in enumerate(seq):
                    for i in range(0, len(group), conc):
                        outs += list(pool.map(lambda p: proc({"prompt": p}, sp)[0],
                                              group[i:i + conc]))
                    if pass_index == 0 and gi == 0 and cache is not None:
                        cold_misses = cache.misses
            passes.append(outs)
        mismatches = [sum(o != reference[p] for o, p in zip(outs, prompts)) for outs in passes]
        drift = sum(a != b for a, b in zip(passes[0], passes[1]))
        res = {"processor": type(proc).__name__, "requests_per_pass": len(prompts),
               "mismatches_pass1": mismatches[0], "mismatches_pass2": mismatches[1],
               "pass2_vs_pass1_diffs": drift, "cache_enabled": cache is not None}
        ok_hits = False
        if cache is not None:
            hits = cache.hits + cache.shared_prefix_hits
            res.update(hits=cache.hits, shared_prefix_hits=cache.shared_prefix_hits,
                       misses=cache.misses, resync_failures=cache.resync_failures,
                       shared_prefix_fallbacks=cache.shared_prefix_fallbacks,
                       cold_first_request_misses=cold_misses, summary=cache.summary())
            # Only the cold first request may miss.
            ok_hits = cache.misses == 1 and hits == 2 * len(prompts) - 1
        res["hits_after_first"] = ok_hits
        ok = mismatches == [0, 0] and drift == 0 and ok_hits
        res["ok"] = ok
        passed = passed and ok
        results[label] = res
        log(f"{label}: {json.dumps(res)}")

    for name, seq in sequences.items():
        for special in (True, False):
            replay(lambda: make_processor(args.ckpt, tok, enable_cache=True),
                   SamplingParams(max_tokens=256, add_special_tokens=special),
                   f"{name}/input_processor/add_special_tokens={special}", seq)
        replay(lambda: DefaultInputProcessor(None, None, tok, enable_tokenization_cache=True),
               SamplingParams(max_tokens=256), f"{name}/DefaultInputProcessor/add_special_tokens=True",
               seq)

    print(json.dumps({"passed": passed and "real" in sequences, "real_prompts": info,
                      "compared_against": "fresh full tokenizer call per prompt (main's "
                      "text-only path: tokenizer(prompt, return_tensors='pt') -> int32 -> list)",
                      "results": results}), flush=True)


if __name__ == "__main__":
    main()
