# Idea 30 gain harness: CUDA-event decode period on the real model, in process.
# BRACE-FREE on purpose: run_gpu_pair passes it via python -c to the base tree, which lacks this file.
# Loads Qwen3.8-27B-NVFP4 with the benchmark engine settings (fp8 KV, block reuse, recurrent snapshot at 32768,
# chunked prefill, max_num_tokens 8192, batch/graph 32), primes a 32768-token shared prefix, then for each
# concurrency 8/16/32 runs waves of conc requests (2048 unique suffix tokens, OSL tokens each) alternating arm X
# (tree default) and arm Y (module flag kv_cache_manager_v2._BLOCK_OFFSET_SNAPSHOT_COPY = False; in the base tree the
# flag does not exist, so Y is identical to X there and X-vs-Y is an in-process A/A).
# Decode period = CUDA event recorded on the execution stream right before CUDAGraphRunner.replay of step K to the
# same event of step K+1, kept only when both replays are at batch == conc and exactly one generation-only
# copy_batch_block_offsets call (no context) happened in between. The first WARM periods of each wave are dropped.
# Last stdout line: evidence JSON (step_ms per batch = arm X median, noise_floor = (p90 - p10) / median).
import asyncio
import json
import os
import random
import statistics
import sys
import threading
import time

os.environ.setdefault("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
T0 = time.time()
OSL = int(os.environ.get("GAIN_OSL", "96"))
WARM = int(os.environ.get("GAIN_WARM", "10"))
REPS = int(os.environ.get("GAIN_REPS", "2"))
CONCS = [int(x) for x in os.environ.get("GAIN_CONCS", "8,16,32").split(",")]


def log(*a):
    print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


def hb():
    while True:
        time.sleep(30)
        log("heartbeat")


threading.Thread(target=hb, daemon=True).start()

import torch  # noqa: E402

from tensorrt_llm._torch.pyexecutor import cuda_graph_runner as cgr  # noqa: E402
from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2 as kvm  # noqa: E402

REC = dict(phase=None, copies=0, ctx=0, rows=[], copy_mode=[])
HAS_FLAG = hasattr(kvm, "_BLOCK_OFFSET_SNAPSHOT_COPY")
orig_replay = cgr.CUDAGraphRunner.replay


def replay(self, key, *a, **kw):
    if REC["phase"] is not None:
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        REC["rows"].append((REC["phase"], cgr.KeyType(*key).batch_size, REC["copies"], REC["ctx"], ev))
    return orig_replay(self, key, *a, **kw)


cgr.CUDAGraphRunner.replay = replay
orig_copy = kvm.KVCacheManagerV2.copy_batch_block_offsets


def copy(self, dst, rids, bw, nctx, nseq, max_blocks=None):
    REC["copies"] += 1
    if nctx:
        REC["ctx"] += 1
    return orig_copy(self, dst, rids, bw, nctx, nseq, max_blocks)


kvm.KVCacheManagerV2.copy_batch_block_offsets = copy


async def wave(llm, prompts, sp):
    outs = await asyncio.gather(*(llm.generate_async(p, sp).aresult() for p in prompts))
    return [o.outputs[0].token_ids for o in outs]


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def periods(phase, conc):
    rows = [r for r in REC["rows"] if r[0] == phase]
    out = []
    for a, b in zip(rows, rows[1:]):
        if a[1] == conc and b[1] == conc and b[2] - a[2] == 1 and b[3] == a[3]:
            out.append(a[4].elapsed_time(b[4]))
    return out[WARM:]


def main():
    ckpt = sys.argv[1]
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

    eng = json.load(open("experiments/ascent-config/engine.json"))
    llm = LLM(model=ckpt, max_batch_size=32, max_num_tokens=8192, max_seq_len=36864, enable_chunked_prefill=True,
              cuda_graph_config=CudaGraphConfig(max_batch_size=32),
              kv_cache_config=KvCacheConfig(**eng["kv_cache_config"]))
    log("LLM ready; tree has snapshot flag:", HAS_FLAG)
    rng = random.Random(1001)
    prefix = [rng.randrange(1000, 100000) for _ in range(32768)]

    def req():
        return prefix + [rng.randrange(1000, 100000) for _ in range(2048)]

    llm.generate([req()], SamplingParams(max_tokens=4, ignore_eos=True, temperature=0.0))
    log("primed")
    sp = SamplingParams(max_tokens=OSL, ignore_eos=True, temperature=0.0)
    tokens = dict()
    for conc in CONCS:
        for rep in range(REPS):
            for arm in ("X", "Y"):
                if HAS_FLAG:
                    kvm._BLOCK_OFFSET_SNAPSHOT_COPY = (arm == "X")
                phase = "c%d_%s_%d" % (conc, arm, rep)
                REC["phase"] = phase
                prompts = [req() for _ in range(conc)]
                t = time.time()
                toks = asyncio.run(wave(llm, prompts, sp))
                REC["phase"] = None
                tokens[phase] = toks
                log(phase, "done in %.1fs" % (time.time() - t))
        if HAS_FLAG:
            kvm._BLOCK_OFFSET_SNAPSHOT_COPY = True
    torch.cuda.synchronize()
    batch = dict()
    arms = dict()
    for conc in CONCS:
        per_arm = dict()
        for arm in ("X", "Y"):
            xs = []
            for rep in range(REPS):
                xs += periods("c%d_%s_%d" % (conc, arm, rep), conc)
            med = statistics.median(xs)
            per_arm[arm] = dict(n=len(xs), median_ms=round(med, 4), p10_ms=round(pct(xs, 0.1), 4),
                                p90_ms=round(pct(xs, 0.9), 4), noise_floor=round((pct(xs, 0.9) - pct(xs, 0.1)) / med, 4))
        per_arm["X_minus_Y_median_us"] = round((per_arm["X"]["median_ms"] - per_arm["Y"]["median_ms"]) * 1000, 2)
        arms[str(conc)] = per_arm
        batch[str(conc)] = dict(step_ms=per_arm["X"]["median_ms"], noise_floor=per_arm["X"]["noise_floor"],
                                p90_ms=per_arm["X"]["p90_ms"], n=per_arm["X"]["n"])
        log("c%d" % conc, json.dumps(per_arm))
    # exactness across arms within this process: same prompts are not reused, so compare token counts only here;
    # bitwise dst equality is proven by check.py and the in-situ shadow compare.
    ntok = sorted(set(len(t) for v in tokens.values() for t in v))
    llm.shutdown()
    print(json.dumps(dict(batch=batch, output=dict(**dict([("class", "exact")])), arms=arms, tree_has_snapshot_flag=HAS_FLAG,
                          osl=OSL, warm_dropped_per_wave=WARM, reps=REPS, token_counts=ntok,
                          measure="CUDA-event decode period (graph K start to K+1 start), pure decode at batch == conc")),
          flush=True)
    os._exit(0)


main()
