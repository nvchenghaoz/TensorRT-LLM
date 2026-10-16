# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark: server-side prompt tokenization through the active input processor.

Runs the same request sequence the frozen serving benchmark sends (one primed
32768-token shared prefix, then requests of prefix + unique 2048-token suffix at
closed-loop concurrency 1/2/4) through ``create_input_processor(ckpt, tok,
enable_tokenization_cache=True)``, i.e. exactly what trtllm-serve calls in its
``_input_proc_executor`` for /v1/completions. Times each request's processor
call (ms). No GPU work, no model weights.

Last stdout line: {"batch": {"1": {"step_ms": ..., "noise_floor": ...}, ...},
"output": {"class": "exact"}, ...}
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

from tok_common import (PromptMaker, env_info, hf_tokenizer, load_real_prompts,
                        load_tokenizer, log, make_processor, pct, processor_cache,
                        request_plan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--add-special-tokens", type=int, default=1,
                    help="1 = /v1/completions default, 0 = chat default")
    ap.add_argument("--measured", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    from tensorrt_llm.sampling_params import SamplingParams

    log(f"env {env_info()}")
    tok = load_tokenizer(args.ckpt)
    hf = hf_tokenizer(tok)
    log(f"tokenizer {type(tok).__name__}/{type(hf).__name__} is_fast={hf.is_fast} "
        f"num_special_tokens_to_add={hf.num_special_tokens_to_add()}")
    bt = hf.backend_tokenizer
    log(f"normalizer={bt.normalizer!r}")
    log(f"pre_tokenizer={bt.pre_tokenizer!r}")
    proc = make_processor(args.ckpt, tok, enable_cache=True)
    log(f"input processor {type(proc).__name__} cache={processor_cache(proc) is not None}")

    sp = SamplingParams(max_tokens=256, add_special_tokens=bool(args.add_special_tokens))
    maker = PromptMaker(hf, seed=args.seed)
    log(f"corpus tokens {len(maker.corpus)}")
    prefix = maker.prefix()
    log(f"prefix chars {len(prefix)}")

    def run_one(prompt):
        t0 = time.perf_counter()
        ids, _ = proc({"prompt": prompt}, sp)
        return (time.perf_counter() - t0) * 1e3, len(ids)

    # Prime (cold first request), then warmups on fresh suffixes.
    prime = maker.request(prefix)
    ms, n = run_one(prime)
    log(f"prime request: {ms:.2f} ms, {n} tokens, {len(prime)} chars")
    # Reference: raw HF tokenizer call on the same prompt, as main's text path does.
    raw = []
    for _ in range(5):
        t0 = time.perf_counter()
        hf(prime, return_tensors="pt")
        raw.append((time.perf_counter() - t0) * 1e3)
    log(f"raw hf(prompt, return_tensors=pt): median {pct(raw, 0.5):.2f} ms")
    raw_be = []
    for _ in range(5):
        t0 = time.perf_counter()
        bt.encode(prime, add_special_tokens=False)
        raw_be.append((time.perf_counter() - t0) * 1e3)
    log(f"raw backend encode: median {pct(raw_be, 0.5):.2f} ms")

    batch = {}
    detail = {}
    for conc, prompts_w, prompts_m in request_plan(maker, prefix, args.warmup, args.measured):
        lat = []
        with ThreadPoolExecutor(max_workers=conc) as pool:
            for group, record in ((prompts_w, False), (prompts_m, True)):
                for i in range(0, len(group), conc):
                    wave = group[i:i + conc]
                    res = list(pool.map(run_one, wave))
                    if record:
                        lat.extend(r[0] for r in res)
        med = pct(lat, 0.5)
        batch[str(conc)] = {"step_ms": round(med, 4),
                            "noise_floor": round((pct(lat, 0.9) - pct(lat, 0.1)) / med, 4)}
        detail[str(conc)] = {"p10": round(pct(lat, 0.1), 3), "p90": round(pct(lat, 0.9), 3),
                             "mean": round(sum(lat) / len(lat), 3), "n": len(lat)}
        log(f"c{conc}: median {med:.2f} ms p10 {pct(lat, 0.1):.2f} p90 {pct(lat, 0.9):.2f} "
            f"n={len(lat)}")

    # Same prompt twice.
    rep = maker.request(prefix)
    a = run_one(rep)
    b = run_one(rep)
    log(f"repeated prompt: first {a[0]:.2f} ms, second {b[0]:.2f} ms")
    cache = processor_cache(proc)
    summary = cache.summary() if cache is not None else None
    log(f"cache summary: {summary}")

    # The benchmark's own prompts, through a fresh processor (cold cache): a
    # prime request of the shared prefix plus a fresh suffix, then the c1/c2/c4
    # rows in closed-loop waves.
    real, info = load_real_prompts(args.ckpt)
    real_detail = {"info": info}
    if real is not None:
        from tensorrt_llm.inputs.prefix_token_cache import _common_prefix_len
        rows = real[1] + real[2] + real[4]
        shared = rows[0]
        for t in rows[1:]:
            shared = shared[:_common_prefix_len(shared, t)]
        proc = make_processor(args.ckpt, tok, enable_cache=True)
        ms, n = run_one(f"{shared.rstrip()} {maker.sample(2048)}")
        real_detail["prime_ms"] = round(ms, 3)
        for conc in (1, 2, 4):
            lat = []
            with ThreadPoolExecutor(max_workers=conc) as pool:
                for i in range(0, len(real[conc]), conc):
                    lat.extend(r[0] for r in pool.map(run_one, real[conc][i:i + conc]))
            real_detail[str(conc)] = {"median": round(pct(lat, 0.5), 3),
                                      "p90": round(pct(lat, 0.9), 3),
                                      "mean": round(sum(lat) / len(lat), 3), "n": len(lat),
                                      "all_ms": [round(x, 2) for x in lat]}
            log(f"real c{conc}: median {pct(lat, 0.5):.2f} ms p90 {pct(lat, 0.9):.2f} "
                f"n={len(lat)}")
        rc = processor_cache(proc)
        real_detail["cache_summary"] = rc.summary() if rc is not None else None
        log(f"real prompts cache summary: {real_detail['cache_summary']}")
    out = {"batch": batch, "output": {"class": "exact"}, "detail": detail,
           "real_prompts": real_detail,
           "prime_ms": round(ms, 3), "raw_hf_ms": round(pct(raw, 0.5), 3),
           "raw_backend_ms": round(pct(raw_be, 0.5), 3), "processor": type(proc).__name__,
           "cache_enabled": cache is not None, "cache_summary": summary,
           "add_special_tokens": bool(args.add_special_tokens)}
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
