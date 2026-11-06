"""Processor-call timing for the shared-prefix splice hit path (branch 233, idea 133).

CPU only, no model load. Runs unchanged on main and on the branch (run_gpu_pair):
    python experiments/ascent/tok_parallel/bench.py <checkpoint> [--reps N]

Builds the DeepseekV41InputProcessor through create_input_processor(enable_tokenization_cache=True),
primes it with one benchmark-shaped prompt (BOS + 32768-token shared prefix + 2048-token suffix),
then times DeepseekV41InputProcessor.call_with_text_prompt on FRESH prompts only:
  c1: N sequential calls -> per-call median / p90 / (p90-p10)/median;
  c2, c4: waves of 2 / 4 threads released by a barrier -> wave makespan (first start to last end)
  and the leader's (fastest) call.
Every timed call is also compared with tokenizer(text, add_special_tokens=True).input_ids
(after timing). Last stdout line: evidence JSON (step_ms = c1 call median, c2/c4 makespan median).
"""

from __future__ import annotations

import gc
import json
import os
import random
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import PromptGen, host_receipt, log  # noqa: E402


def stats(xs):
    s = sorted(xs)
    n = len(s)

    def q(p):
        return s[min(n - 1, max(0, int(round(p * (n - 1)))))]

    med = q(0.5)
    return {"median": round(med, 4), "p10": round(q(0.1), 4), "p90": round(q(0.9), 4),
            "mean": round(sum(s) / n, 4), "n": n,
            "noise_floor": round((q(0.9) - q(0.1)) / med, 4) if med else None}


def main():
    ckpt = sys.argv[1]
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 120
    waves = int(sys.argv[sys.argv.index("--waves") + 1]) if "--waves" in sys.argv else 40
    receipt = host_receipt()
    log(f"host receipt: {receipt}")

    from tensorrt_llm.inputs import prefix_token_cache as ptc
    from tensorrt_llm.inputs.registry import create_input_processor
    from tensorrt_llm.sampling_params import SamplingParams
    from tensorrt_llm.tokenizer import load_hf_tokenizer

    tok = load_hf_tokenizer(ckpt, trust_remote_code=True)
    hf = tok.tokenizer
    proc = create_input_processor(ckpt, tok, enable_tokenization_cache=True)
    cache = proc._prefix_token_cache
    tree = {"parallel_attr": getattr(ptc, "PARALLEL_TAIL", None),
            "parallel_backend": getattr(cache, "_parallel_backend", None) is not None,
            "processor": type(proc).__name__}
    log(f"tree: {tree}")
    sp = SamplingParams(max_tokens=1)

    gen = PromptGen(hf, random.Random(4321))
    prefix = gen.prefix()
    n_prompts = 1 + 8 + reps + 2 * waves + 4 * waves
    leads = [gen.rng.choice(gen.caps) for _ in range(n_prompts)]
    prompts = [prefix + gen.suffix(leads[i]) for i in range(n_prompts)]
    log(f"{n_prompts} prompts, prefix {len(prefix)} chars, suffix chars ~{len(prompts[1]) - len(prefix)}")
    it = iter(prompts)

    def call(text):
        ids, _ = proc.call_with_text_prompt({"prompt": text}, sp)
        return ids

    results = []  # (text, ids)
    t = time.perf_counter()
    results.append((p := next(it), call(p)))
    log(f"prime (full) {1e3 * (time.perf_counter() - t):.1f} ms")
    for _ in range(8):
        p = next(it)
        results.append((p, call(p)))
    log(f"warmup done: {cache.summary()}")

    gc.collect()
    gc.disable()
    c1 = []
    for i in range(reps):
        p = next(it)
        t = time.perf_counter()
        ids = call(p)
        c1.append((time.perf_counter() - t) * 1e3)
        results.append((p, ids))
        if i % 40 == 0:
            gc.collect()
            log(f"c1 {i}/{reps}: {c1[-1]:.2f} ms")

    def wave(k):
        texts = [next(it) for _ in range(k)]
        barrier = threading.Barrier(k)
        spans = [None] * k
        outs = [None] * k

        def worker(j):
            barrier.wait()
            s = time.perf_counter()
            outs[j] = call(texts[j])
            spans[j] = (s, time.perf_counter())

        ths = [threading.Thread(target=worker, args=(j,)) for j in range(k)]
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        for j in range(k):
            results.append((texts[j], outs[j]))
        make = (max(e for _, e in spans) - min(s for s, _ in spans)) * 1e3
        lead = min(e - s for s, e in spans) * 1e3
        return make, lead

    cw = {}
    for k in (2, 4):
        mk, ld = [], []
        for w in range(waves):
            gc.collect()
            m, l_ = wave(k)
            mk.append(m)
            ld.append(l_)
            if w % 10 == 0:
                log(f"c{k} wave {w}/{waves}: makespan {m:.2f} ms leader {l_:.2f} ms")
        cw[k] = (mk, ld)
    gc.enable()
    log(f"timing done: {cache.summary()}")

    mism = 0
    for text, ids in results:
        if list(ids) != list(hf(text, add_special_tokens=True).input_ids):
            mism += 1
    log(f"exactness of all {len(results)} timed/primed calls: {mism} mismatches")

    s1 = stats(c1)
    out = {"batch": {}, "output": {"class": "exact"}}
    out["batch"]["1"] = {"step_ms": s1["median"], "noise_floor": s1["noise_floor"],
                         "p90_ms": s1["p90"], "p10_ms": s1["p10"], "mean_ms": s1["mean"],
                         "n": s1["n"]}
    for k, (mk, ld) in cw.items():
        sm, sl = stats(mk), stats(ld)
        out["batch"][str(k)] = {"step_ms": sm["median"], "noise_floor": sm["noise_floor"],
                                "p90_ms": sm["p90"], "p10_ms": sm["p10"], "mean_ms": sm["mean"],
                                "leader_median_ms": sl["median"], "leader_p90_ms": sl["p90"],
                                "waves": sm["n"]}
    out["tree"] = tree
    out["host"] = receipt
    out["splices"] = cache.splices
    out["full_tokenizations"] = cache._full_count
    out["mismatches"] = mism
    out["parallel_tails"] = getattr(cache, "parallel_tails", None)
    out["parallel_fallbacks"] = getattr(cache, "parallel_fallbacks", None)
    if mism:
        out["output"]["class"] = "MISMATCH"
    print(json.dumps(out))


if __name__ == "__main__":
    main()
