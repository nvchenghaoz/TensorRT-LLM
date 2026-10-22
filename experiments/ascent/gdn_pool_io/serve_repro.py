# Repro of the benchmark's serving shape in-process (idea 5 debugging):
# Qwen3.8-27B-NVFP4, benchmark engine settings (fp8 KV, block reuse, recurrent
# snapshot at 32768, chunked prefill, max_num_tokens 8192, batch/graph 32), a
# primed 32768-token shared prefix + 2048 unique suffix tokens per request,
# closed loop at c16 then c32. Every FlashInfer GDN chunk call is wrapped: the
# slot ids are validated (range, uniqueness, count) and the stream is synced
# after the call, so a fault is attributed to the exact call.
#
# Usage: python serve_repro.py <checkpoint> [n_requests_per_conc] [out_tokens]
# Env TLLM_GDN_SM120_POOL_IO=0 runs main's gather/scatter path.
import asyncio
import faulthandler
import json
import os
import random
import sys
import time

faulthandler.enable()
os.environ.setdefault("TLLM_WORKER_USE_SINGLE_PROCESS", "1")

T0 = time.time()


def log(*a):
    print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


STATS = dict(calls=0, pool_calls=0, max_seqs=0, max_long=0, max_slot=-1, anomalies=[])


def install_wrapper():
    import torch

    import tensorrt_llm._torch.modules.fla.flashinfer_chunk as fc

    orig = fc.chunk_gated_delta_rule

    def wrapped(*args, **kwargs):
        st = kwargs.get("initial_state")
        idx = kwargs.get("initial_state_indices")
        cu = kwargs.get("cu_seqlens")
        info = None
        if st is not None and idx is not None and cu is not None:
            n = cu.shape[0] - 1
            ids = idx.tolist()
            lens = (cu[1:] - cu[:-1]).tolist()
            n_long = sum(1 for x in lens if x > 1)
            info = dict(n=n, n_ids=len(ids), n_long=n_long, hint=kwargs.get("num_long_seqs"),
                        lens=lens[:8], min_id=min(ids) if ids else None,
                        max_id=max(ids) if ids else None, pool=tuple(st.shape),
                        stride0=st.stride(0), dtype=str(st.dtype),
                        eligible=fc.sm120_pool_io_eligible(st) if hasattr(fc, "sm120_pool_io_eligible") else None)
            bad = []
            if len(ids) != n:
                bad.append("len(ids)!=n")
            if ids and (min(ids) < 0 or max(ids) >= st.shape[0]):
                bad.append("id out of range")
            if len(set(ids)) != len(ids):
                bad.append("duplicate ids")
            if min(lens) < 0:
                bad.append("negative len")
            STATS["calls"] += 1
            STATS["max_seqs"] = max(STATS["max_seqs"], n)
            STATS["max_long"] = max(STATS["max_long"], n_long)
            if ids:
                STATS["max_slot"] = max(STATS["max_slot"], max(ids))
            if bad and len(STATS["anomalies"]) < 20:
                STATS["anomalies"].append(dict(bad=bad, **info))
                log("ANOMALY", bad, json.dumps(info))
        try:
            res = orig(*args, **kwargs)
            torch.cuda.synchronize()
        except Exception:
            log("EXCEPTION in chunk call", json.dumps(info))
            raise
        return res

    fc.chunk_gated_delta_rule = wrapped


async def closed_loop(llm, prompts, conc, sp):
    q = list(prompts)
    lat = []
    sem = asyncio.Semaphore(conc)

    async def one(p):
        async with sem:
            t = time.time()
            out = await llm.generate_async(p, sp).aresult()
            lat.append(time.time() - t)
            return out.outputs[0].token_ids[:4]

    res = await asyncio.gather(*(one(p) for p in q))
    return lat, res


def main():
    ckpt = sys.argv[1]
    n_req = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    out_tok = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    install_wrapper()
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

    kv = KvCacheConfig(dtype="fp8", enable_block_reuse=True,
                       mamba_state_config=dict(additional_snapshot_offsets_from_start=[32768]))
    llm = LLM(model=ckpt, max_batch_size=32, max_num_tokens=8192, max_seq_len=36864,
              enable_chunked_prefill=True, cuda_graph_config=CudaGraphConfig(max_batch_size=32),
              kv_cache_config=kv)
    log("LLM ready; pool io env", os.environ.get("TLLM_GDN_SM120_POOL_IO"))
    rng = random.Random(1001)
    prefix = [rng.randrange(1000, 100000) for _ in range(32768)]

    def req():
        return prefix + [rng.randrange(1000, 100000) for _ in range(2048)]

    sp_prime = SamplingParams(max_tokens=4, ignore_eos=True, temperature=0.0)
    llm.generate([req()], sp_prime)
    log("primed; stats", json.dumps(STATS))
    sp = SamplingParams(max_tokens=out_tok, ignore_eos=True, temperature=0.0)
    results = dict()
    for conc in (16, 32):
        prompts = [req() for _ in range(n_req)]
        t = time.time()
        lat, toks = asyncio.run(closed_loop(llm, prompts, conc, sp))
        results[str(conc)] = dict(wall_s=time.time() - t, n=len(lat),
                                  mean_latency_s=sum(lat) / len(lat), first_tokens=toks[0])
        log("c%d done" % conc, json.dumps(results[str(conc)]), "stats", json.dumps(STATS))
    llm.shutdown()
    print(json.dumps(dict(passed=not STATS["anomalies"], results=results, stats=STATS,
                          pool_io_env=os.environ.get("TLLM_GDN_SM120_POOL_IO", "1"))), flush=True)


if __name__ == "__main__":
    main()
