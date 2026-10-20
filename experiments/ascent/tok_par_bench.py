# SPDX-License-Identifier: Apache-2.0
# Paired intake microbenchmark for host-runtime idea 109 (parallel tail tokenization in
# tensorrt_llm/inputs/prefix_token_cache.py). BRACE-FREE on purpose: run_gpu_pair's
# base arm lacks this file, so the pair passes this text inline via python -c.
#
# Window: the input-processor call trtllm-serve runs in _input_proc_executor for
# /v1/completions (create_input_processor(ckpt, tok, enable_tokenization_cache=True)),
# timed per request. Prompts: the benchmark's real shared prefix (from the proxy
# artifacts) + a fresh 2048-token AIPerf-style suffix per request, in closed-loop
# waves of 1/2/4 after one prime request; then the real benchmark prompts themselves.
# In a tree whose prefix_token_cache has PARALLEL_TAIL, arms alternate per wave
# (ABBA) through that attribute in this one process: on = parallel pieces, off = one
# call (main's code). In the base tree there is one arm (main). The last stdout line
# is JSON: batch.<c>.step_ms = median of the tree's native arm (on, or main).
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.getcwd(), 'experiments', 'ascent'))
from tok_common import (PromptMaker, common_prefix, hf_tokenizer, load_real_prompts,  # noqa: E402
                        load_tokenizer, log, make_processor, pct, processor_cache)

WARM_PER_ARM = 12
TIMED_PER_ARM = 60


def git(*a):
    try:
        return subprocess.run(['git'] + list(a), capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return '?' + repr(e)


def stats(xs):
    if not xs:
        return dict(n=0)
    m = pct(xs, 0.5)
    return dict(n=len(xs), median=round(m, 4), p10=round(pct(xs, 0.1), 4),
                p90=round(pct(xs, 0.9), 4), mean=round(sum(xs) / len(xs), 4),
                noise=round((pct(xs, 0.9) - pct(xs, 0.1)) / m, 4))


def main():
    ckpt = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else 'unlabelled'
    log('TREE_COMMIT=%s BASE_COMMIT=07a287408d9a3a2313ae5f3ec22730f94f932b51 LABEL=%s DIRTY=%d'
        % (git('rev-parse', 'HEAD'), label,
           len(git('status', '--porcelain', '--untracked-files=no').splitlines())))
    from tensorrt_llm.inputs import prefix_token_cache as ptc
    from tensorrt_llm.sampling_params import SamplingParams
    paired = hasattr(ptc, 'PARALLEL_TAIL')
    log('paired arms in process: %s; PARALLEL_TAIL default %s; env %s'
        % (paired, getattr(ptc, 'PARALLEL_TAIL', None),
           os.environ.get('TLLM_PREFIX_TOKEN_CACHE_PARALLEL')))
    log('cpu_count %s affinity %s loadavg %s' % (os.cpu_count(), len(os.sched_getaffinity(0)),
                                                 os.getloadavg()))
    tok = load_tokenizer(ckpt)
    hf = hf_tokenizer(tok)
    real, info = load_real_prompts(ckpt)
    assert real is not None, 'no real prompts'
    real_all = real[1] + real[2] + real[4]
    shared = common_prefix(real_all).rstrip()
    log('real shared prefix chars %d; real rows %s' % (len(shared), info.get('rows')))
    maker = PromptMaker(hf, seed=2468)
    sp = SamplingParams(max_tokens=256, add_special_tokens=True)
    proc = make_processor(ckpt, tok, enable_cache=True)
    cache = processor_cache(proc)
    assert cache is not None
    default_arm = getattr(ptc, 'PARALLEL_TAIL', None)

    def set_arm(arm):
        if paired:
            ptc.PARALLEL_TAIL = arm

    def run_one(prompt):
        t0 = time.perf_counter()
        ids, _ = proc(dict(prompt=prompt), sp)
        return (time.perf_counter() - t0) * 1e3, len(ids)

    t0 = time.perf_counter()
    _, n = run_one(shared + ' ' + maker.sample(2048))
    log('prime: %.2f ms, %d tokens; %s' % ((time.perf_counter() - t0) * 1e3, n, cache.summary()))

    arms = [True, False] if paired else ['main']
    batch = dict()
    detail = dict()
    for conc in (1, 2, 4):
        warm_waves = 2 * math.ceil(WARM_PER_ARM / conc) if paired else math.ceil(WARM_PER_ARM / conc)
        timed_waves = 2 * math.ceil(TIMED_PER_ARM / conc) if paired else math.ceil(TIMED_PER_ARM / conc)
        lat = dict((a, []) for a in arms)
        lead = dict((a, []) for a in arms)
        span = dict((a, []) for a in arms)
        h0 = cache.hits + cache.shared_prefix_hits
        m0 = cache.misses
        with ThreadPoolExecutor(max_workers=conc) as pool:
            for w in range(warm_waves + timed_waves):
                # ABBA over waves.
                arm = arms[(w + (w // 2)) % len(arms)] if paired else 'main'
                set_arm(arm)
                wave = [shared + ' ' + maker.sample(2048) for _ in range(conc)]
                t0 = time.perf_counter()
                res = list(pool.map(run_one, wave))
                makespan = (time.perf_counter() - t0) * 1e3
                if w >= warm_waves:
                    lat[arm].extend(r[0] for r in res)
                    lead[arm].append(min(r[0] for r in res))
                    span[arm].append(makespan)
        set_arm(default_arm)
        hits = cache.hits + cache.shared_prefix_hits - h0
        misses = cache.misses - m0
        d = dict(hits=hits, misses=misses, requests=conc * (warm_waves + timed_waves))
        for a in arms:
            d[str(a)] = dict(per_request=stats(lat[a]), wave_leader=stats(lead[a]),
                             wave_makespan=stats(span[a]))
            log('c%d arm %s: per-request %s | leader %s | makespan %s'
                % (conc, a, stats(lat[a]), stats(lead[a]), stats(span[a])))
        native = True if paired else 'main'
        s = stats(lat[native])
        batch[str(conc)] = dict(step_ms=s['median'], noise_floor=s['noise'])
        if paired:
            on, off = stats(lat[True]), stats(lat[False])
            d['delta_median_ms'] = round(on['median'] - off['median'], 4)
            d['ratio_median'] = round(on['median'] / off['median'], 4)
            log('c%d in-process: on %.3f vs off %.3f ms median (ratio %.3f)'
                % (conc, on['median'], off['median'], d['ratio_median']))
        log('c%d cache: %d hits, %d misses of %d requests' % (conc, hits, misses, d['requests']))
        detail[str(conc)] = d

    # The benchmark's own prompts, each once, arms alternating per wave.
    real_detail = dict()
    for conc in (1, 2, 4):
        lat = dict((a, []) for a in arms)
        with ThreadPoolExecutor(max_workers=conc) as pool:
            rows = real[conc]
            for wi, i in enumerate(range(0, len(rows), conc)):
                arm = arms[(wi + (wi // 2)) % len(arms)] if paired else 'main'
                set_arm(arm)
                lat[arm].extend(r[0] for r in pool.map(run_one, rows[i:i + conc]))
        set_arm(default_arm)
        real_detail[str(conc)] = dict((str(a), stats(lat[a])) for a in arms)
        log('real c%d: %s' % (conc, real_detail[str(conc)]))
    summary = cache.summary()
    log('cache summary: %s' % summary)
    fallbacks = cache.resync_failures + cache.shared_prefix_fallbacks
    out = dict(batch=batch, output=dict(cls='exact'), detail=detail, real_prompts=real_detail,
               paired_in_process=paired, cache_summary=summary, misses=cache.misses,
               fallbacks=fallbacks, parallel=getattr(cache, 'parallel_tokenizations', None),
               parallel_errors=getattr(cache, 'parallel_errors', None),
               tree_commit=git('rev-parse', 'HEAD'), label=label)
    out['output'] = dict([('class', 'exact')])
    print(json.dumps(out), flush=True)


main()
