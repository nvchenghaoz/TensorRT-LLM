# Idea 133 (XQA PDL overlap): per-layer nsys spans of the decode full-attention chain
#   P (same-stream predecessor of R) -> R = applyBiasRopeUpdateKVCacheV2 -> K = kernel_mha
# in real CUDA-graph decode at batch 1/2/4 (FP8 KV, block reuse, 32768-token shared prefix + 2048 unique tokens).
# Usage: python nsys_span.py <checkpoint>          (re-runs itself under nsys, then analyzes)
#        python nsys_span.py <checkpoint> run      (the profiled model run only)
# Per layer: R duration, R exposed = end(R) - max(start(R), end(P)), gap = start(K) - end(R),
# SPAN = end(K) - end(P), K duration. kernel_ms = median SPAN per batch; noise_floor = relative
# (p90 - p10) spread of the SPAN medians of 8 consecutive chunks of that batch's layers.
# NOTE: written without curly braces so that run_gpu_pair can pass it inline to the base tree.
import json
import os
import random
import shutil
import sqlite3
import subprocess
import sys
import threading
import time

OUT = '/tmp/xqa_pdl_nsys'
PREFIX = 32768
SUFFIX = 2048
GEN = 64
ROPE = 'applyBiasRopeUpdateKVCacheV2'
MHA = 'kernel_mha'


def log(*a):
    print(*a, flush=True)


def heartbeat(tag):
    t0 = time.time()

    def loop():
        while True:
            time.sleep(30)
            log('[heartbeat %s] %.0fs' % (tag, time.time() - t0))

    threading.Thread(target=loop, daemon=True).start()


def run_model(ckpt):
    os.environ['TLLM_WORKER_USE_SINGLE_PROCESS'] = '1'
    import torch

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig
    heartbeat('model')
    t0 = time.time()
    llm = LLM(model=ckpt, max_seq_len=36864, max_batch_size=8, max_num_tokens=8192, enable_chunked_prefill=True,
              cuda_graph_config=CudaGraphConfig(max_batch_size=8),
              kv_cache_config=KvCacheConfig(dtype='fp8', enable_block_reuse=True,
                                            mamba_state_config=dict(additional_snapshot_offsets_from_start=[32768])))
    log('LLM ready in %.1fs' % (time.time() - t0))
    rng = random.Random(0)
    prefix = [rng.randrange(1000, 100000) for _ in range(PREFIX)]

    def prompts(n):
        return [prefix + [rng.randrange(1000, 100000) for _ in range(SUFFIX)] for _ in range(n)]

    sp_w = SamplingParams(max_tokens=8, ignore_eos=True, temperature=0.0)
    sp = SamplingParams(max_tokens=GEN, ignore_eos=True, temperature=0.0)
    t0 = time.time()
    llm.generate(prompts(1), sp_w)  # primes the shared prefix (block reuse + mamba snapshot at 32768)
    llm.generate(prompts(2), sp_w)
    llm.generate(prompts(4), sp_w)
    torch.cuda.synchronize()
    log('warmup done in %.1fs' % (time.time() - t0))
    torch.cuda.profiler.start()
    for n in (1, 2, 4):
        t1 = time.time()
        llm.generate(prompts(n), sp)
        torch.cuda.synchronize()
        log('profiled c%d in %.1fs' % (n, time.time() - t1))
        time.sleep(0.3)
    torch.cuda.profiler.stop()
    log('model run done')
    llm.shutdown()


def pct(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]


def med(xs):
    return pct(xs, 0.5)


def analyze(db):
    con = sqlite3.connect(db)
    cols = [r[1] for r in con.execute('PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)')]
    has_graph = 'graphId' in cols
    sel = 'SELECT k.start, k.end, k.streamId, k.gridX, k.gridY, k.gridZ, s.value'
    sel += ', k.graphId' if has_graph else ', 0'
    sel += ' FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName = s.id ORDER BY k.start'
    rows = con.execute(sel).fetchall()
    log('kernels in capture: %d (graphId column %s)' % (len(rows), has_graph))
    streams = dict()
    for r in rows:
        streams.setdefault(r[2], []).append(r)
    layers = dict()
    pred_names = dict()
    n_dec_rope, n_dec_rope_then_mha = 0, 0
    next_names = dict()
    for sid, lst in streams.items():
        for i, r in enumerate(lst):
            name = r[6]
            if ROPE in name:
                in_graph = bool(r[7]) if has_graph else (r[3] * r[4] * r[5] <= 96)
                if in_graph:
                    n_dec_rope += 1
                    nxt = lst[i + 1][6] if i + 1 < len(lst) else 'END'
                    if MHA in nxt:
                        n_dec_rope_then_mha += 1
                    else:
                        key = nxt[:80]
                        next_names[key] = next_names.get(key, 0) + 1
            if MHA not in name or i < 2:
                continue
            R, P, K = lst[i - 1], lst[i - 2], r
            if ROPE not in R[6]:
                key = R[6][:80]
                pred_names[key] = pred_names.get(key, 0) + 1
                continue
            b = K[5]
            layers.setdefault(b, []).append(dict(
                r_dur=R[1] - R[0], r_exposed=R[1] - max(R[0], P[1]), gap=K[0] - R[1], span=K[1] - P[1],
                k_dur=K[1] - K[0], k_early=K[0] < R[1], p_name=P[6][:60], m=K[3]))
    res = dict()
    for b in sorted(layers):
        L = layers[b]
        spans = [x['span'] for x in L]
        nch = 8
        csz = max(1, len(L) // nch)
        chunk_meds = [med(spans[c * csz:(c + 1) * csz]) for c in range(nch) if spans[c * csz:(c + 1) * csz]]
        cm = med(chunk_meds)
        pnames = dict()
        for x in L:
            pnames[x['p_name']] = pnames.get(x['p_name'], 0) + 1
        ent = dict(
            kernel_ms=med(spans) / 1e6,
            noise_floor=(pct(chunk_meds, 0.9) - pct(chunk_meds, 0.1)) / cm if cm else None,
            layers=len(L),
            m_values=sorted(set(x['m'] for x in L)),
            span_us_median=med(spans) / 1e3, span_us_p10=pct(spans, 0.1) / 1e3, span_us_p90=pct(spans, 0.9) / 1e3,
            span_chunk_medians_us=[c / 1e3 for c in chunk_meds],
            span_per_call_noise=(pct(spans, 0.9) - pct(spans, 0.1)) / med(spans),
            rope_dur_us_median=med([x['r_dur'] for x in L]) / 1e3,
            rope_exposed_us_median=med([x['r_exposed'] for x in L]) / 1e3,
            rope_exposed_us_p10=pct([x['r_exposed'] for x in L], 0.1) / 1e3,
            rope_exposed_us_p90=pct([x['r_exposed'] for x in L], 0.9) / 1e3,
            gap_us_median=med([x['gap'] for x in L]) / 1e3,
            gap_us_p10=pct([x['gap'] for x in L], 0.1) / 1e3,
            gap_us_p90=pct([x['gap'] for x in L], 0.9) / 1e3,
            mha_dur_us_median=med([x['k_dur'] for x in L]) / 1e3,
            frac_mha_starts_before_rope_ends=sum(1 for x in L if x['k_early']) / len(L),
            p_names=pnames)
        res[str(b)] = ent
        log('B=%d: %s' % (b, json.dumps(ent)))
    all_exp = [x['r_exposed'] for L in layers.values() for x in L]
    frac = n_dec_rope_then_mha / max(1, n_dec_rope)
    summary = dict(decode_rope_calls=n_dec_rope, decode_rope_followed_by_kernel_mha=n_dec_rope_then_mha,
                   frac_followed=frac, rope_next_other=next_names, mha_pred_not_rope=pred_names,
                   rope_exposed_us_median_all=(med(all_exp) / 1e3) if all_exp else None)
    log('summary: %s' % json.dumps(summary))
    return res, summary


def main():
    if len(sys.argv) > 2 and sys.argv[2] == 'run':
        run_model(sys.argv[1])
        return
    heartbeat('driver')
    nsys = shutil.which('nsys') or '/usr/local/cuda/bin/nsys'
    log('nsys: %s' % nsys)
    os.makedirs(OUT, exist_ok=True)
    rep = os.path.join(OUT, 'decode_%d' % os.getpid())
    cmd = [nsys, 'profile', '--trace=cuda', '--cuda-graph-trace=node', '--capture-range=cudaProfilerApi',
           '--capture-range-end=stop', '--sample=none', '--cpuctxsw=none', '--force-overwrite=true', '-o', rep,
           sys.executable, os.path.abspath(__file__), sys.argv[1], 'run']
    log('running: %s' % ' '.join(cmd))
    t0 = time.time()
    p = subprocess.run(cmd)
    log('nsys exit %d in %.0fs' % (p.returncode, time.time() - t0))
    db = rep + '.sqlite'
    subprocess.run([nsys, 'export', '--type=sqlite', '--force-overwrite=true', '-o', db, rep + '.nsys-rep'])
    res, summary = analyze(db)
    exp = summary['rope_exposed_us_median_all']
    gate = summary['frac_followed'] >= 0.95 and exp is not None and exp >= 0.8
    out = dict(batch=res, summary=summary, passed=bool(gate),
               gate='STEP 0: decode RoPE followed by kernel_mha in >= 95% of layers and median RoPE exposed >= 0.8 us',
               measure='kernel_ms = median per-layer SPAN end(P) -> end(kernel_mha) from nsys --cuda-graph-trace=node; '
               'noise_floor = (p90 - p10) / median of 8 consecutive chunk medians')
    out['output'] = dict([('class', 'exact'), ('basis', 'trace only; numerics are checked by xqa_evidence.py --mode check')])
    art = os.environ.get('AUTO_DEV_ARTIFACTS') or os.environ.get('ASCENT_ARTIFACTS_DIR')
    if art:
        try:
            os.makedirs(art, exist_ok=True)
            with open(os.path.join(art, 'nsys_span.json'), 'w') as f:
                json.dump(out, f, indent=1)
        except Exception as e:  # noqa
            log('artifact write failed: %s' % e)
    print(json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
