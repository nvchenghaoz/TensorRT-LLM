"""In-situ probe of KVCacheManagerV2.copy_batch_block_offsets on the real model (idea 30 step 0).

Loads Qwen3.8-27B-NVFP4 in-process with the benchmark engine settings (fp8 KV, block reuse,
recurrent snapshot at 32768, chunked prefill, max_num_tokens 8192, batch/graph 32), primes a
32768-token shared prefix, then runs closed-loop c8, c16, c32 phases (2048 unique suffix tokens,
OUT tokens each).  Every copy_batch_block_offsets call is wrapped:
  - records pool count, host table / dst shapes, index scales / kv offsets, call site, num_seqs;
  - times the stock kernel with CUDA events on the manager's stream (in situ, i.e. right after
    the sampler and right before the decode graph replay);
  - runs one alternative arm into a scratch dst with the same inputs, alternating whether it runs
    before or after stock, and times it too (P1o / P4o: by-value split-row kernel from pinned host;
    A2: fresh pinned snapshot + H2D DMA + stock kernel on device pointers);
  - every 25th call checks scratch == real dst bitwise for rows [:num_seqs].
Usage: python insitu.py <checkpoint> [out_tokens]
"""
import asyncio
import collections
import json
import os
import random
import statistics
import sys
import threading
import time

os.environ.setdefault("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

T0 = time.time()


def log(*a):
    print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


def heartbeat():
    def loop():
        while True:
            time.sleep(30)
            log("[heartbeat]")

    threading.Thread(target=loop, daemon=True).start()


REC = dict(info=None, sites=collections.Counter(), events=[], mismatches=0, checks=0, phase="load")


def install(ext):
    import torch

    from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2 as m
    from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils import \
        copy_batch_block_offsets_to_device

    orig = m.KVCacheManagerV2.copy_batch_block_offsets
    state = dict(k=0, scratch=None, dev=None)
    arms = ["P1o", "P4o", "A2"]

    def wrapped(self, dst_tensor, request_ids, beam_width, num_contexts, num_seqs, max_blocks=None):
        f = sys._getframe(1)
        REC["sites"]["%s:%d" % (os.path.basename(f.f_code.co_filename), f.f_lineno)] += 1
        if self._use_per_layer_page_tables or not dst_tensor.is_cuda or beam_width != 1:
            return orig(self, dst_tensor, request_ids, beam_width, num_contexts, num_seqs, max_blocks)
        host = self.host_kv_cache_block_offsets
        P, srcMax, _, B = host.shape
        if REC["info"] is None:
            REC["info"] = dict(num_pools=int(self.num_pools), host_shape=list(host.shape),
                               dst_shape=list(dst_tensor.shape), index_scales=self.index_scales.tolist(),
                               kv_offset=self.kv_offset.tolist(), cls=type(self).__name__,
                               host_pinned=bool(host.is_pinned()), max_blocks_arg=max_blocks)
            log("INFO", json.dumps(REC["info"]))
        stream = self._stream
        if state["scratch"] is None:
            state["scratch"] = torch.empty_like(dst_tensor)
            state["dev"] = torch.empty(host.numel(), dtype=torch.int32, device=dst_tensor.device)
            state["scales_d"] = self.index_scales.to(dst_tensor.device)
            state["offs_d"] = self.kv_offset.to(dst_tensor.device)
            state["ident"] = torch.arange(srcMax, dtype=torch.int32, device=dst_tensor.device)
        scratch = state["scratch"]
        k = state["k"]
        state["k"] += 1
        arm = arms[(k // 2) % len(arms)]
        x_first = (k % 2 == 0)
        cidx = self.index_mapper.get_copy_index(request_ids, num_contexts, beam_width,
                                                self.kv_cache_type == m.CacheTypeCpp.CROSS)
        cl = cidx.tolist()
        snap = torch.empty((P, num_seqs, 2, B), dtype=torch.int32, pin_memory=True) if arm == "A2" else None
        if snap is not None:
            torch.index_select(host, 1, cidx.clone(), out=snap)

        def run_x():
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(stream):
                e0.record(stream)
                if arm == "A2":
                    sd = state["dev"][:P * num_seqs * 2 * B].view(P, num_seqs, 2, B)
                    sd.copy_(snap, non_blocking=True)
                    copy_batch_block_offsets_to_device(sd, scratch, state["ident"][:num_seqs], state["scales_d"],
                                                       state["offs_d"], stream.cuda_stream)
                else:
                    S = 1 if arm == "P1o" else 4
                    ext.p_copy(host, scratch, cl, self.index_scales.tolist(), self.kv_offset.tolist(), S, True,
                               srcMax, 2 * B)
                e1.record(stream)
            return e0, e1

        if x_first:
            ex = run_x()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(stream)
        orig(self, dst_tensor, request_ids, beam_width, num_contexts, num_seqs, max_blocks)
        e1.record(stream)
        if not x_first:
            ex = run_x()
        REC["events"].append((REC["phase"], num_seqs, num_contexts, arm, x_first, e0, e1, ex[0], ex[1]))
        if k % 25 == 0:
            stream.synchronize()
            REC["checks"] += 1
            if not torch.equal(scratch[:, :num_seqs], dst_tensor[:, :num_seqs]):
                REC["mismatches"] += 1
                log("MISMATCH at call", k, arm)

    m.KVCacheManagerV2.copy_batch_block_offsets = wrapped


async def closed_loop(llm, prompts, conc, sp):
    sem = asyncio.Semaphore(conc)

    async def one(p):
        async with sem:
            out = await llm.generate_async(p, sp).aresult()
            return out.outputs[0].token_ids[:2]

    return await asyncio.gather(*(one(p) for p in prompts))


def summarize():
    import torch
    torch.cuda.synchronize()
    out = {}
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for (ph, n, nc, arm, xf, e0, e1, x0, x1) in REC["events"]:
        if nc != 0:
            key = "%s_mixed" % ph
        else:
            key = "%s_decode_n%d" % (ph, n)
        g = groups[key]
        g["stock"].append(e0.elapsed_time(e1) * 1000)
        g["stock_" + ("second" if xf else "first")].append(e0.elapsed_time(e1) * 1000)
        g[arm + ("_first" if xf else "_second")].append(x0.elapsed_time(x1) * 1000)
    for key, g in groups.items():
        out[key] = {a: dict(n=len(v), med_us=round(statistics.median(v), 2),
                            p10_us=round(sorted(v)[len(v) // 10], 2), p90_us=round(sorted(v)[(len(v) * 9) // 10], 2))
                    for a, v in g.items() if v}
    return out


def main():
    ckpt = sys.argv[1]
    out_tok = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    heartbeat()
    from step0 import load_ext
    ext = load_ext()
    install(ext)
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

    eng = json.load(open("experiments/ascent-config/engine.json"))
    kv = KvCacheConfig(**eng["kv_cache_config"])
    llm = LLM(model=ckpt, max_batch_size=32, max_num_tokens=8192, max_seq_len=36864,
              enable_chunked_prefill=True, cuda_graph_config=CudaGraphConfig(max_batch_size=32),
              kv_cache_config=kv)
    log("LLM ready")
    rng = random.Random(1001)
    prefix = [rng.randrange(1000, 100000) for _ in range(32768)]

    def req():
        return prefix + [rng.randrange(1000, 100000) for _ in range(2048)]

    REC["phase"] = "prime"
    llm.generate([req()], SamplingParams(max_tokens=4, ignore_eos=True, temperature=0.0))
    log("primed", json.dumps(REC["info"]))
    sp = SamplingParams(max_tokens=out_tok, ignore_eos=True, temperature=0.0)
    for conc in (8, 16, 32):
        REC["phase"] = "c%d" % conc
        t = time.time()
        asyncio.run(closed_loop(llm, [req() for _ in range(conc)], conc, sp))
        log("c%d done in %.1fs, calls so far %d" % (conc, time.time() - t, len(REC["events"])))
    summ = summarize()
    for k in sorted(summ):
        log(k, json.dumps(summ[k]))
    llm.shutdown()
    res = dict(passed=REC["mismatches"] == 0, checks=REC["checks"], mismatches=REC["mismatches"],
               info=REC["info"], sites=dict(REC["sites"]), summary=summ)
    print(json.dumps(res), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
