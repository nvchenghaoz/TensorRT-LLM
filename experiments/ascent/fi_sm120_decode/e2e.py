# Idea 43 model-level check (acceptance item 4 + numerics hint). Usage:
#   python e2e.py <checkpoint>            -> runs arm "fi" and arm "base" (TRTLLM_NVFP4_FI_CUTLASS=0)
#                                            in two subprocesses, compares them, prints JSON
#   python e2e.py <checkpoint> <arm>      -> one arm (internal)
# Per arm: LLM API, max_batch_size 128, decode CUDA graphs to 128, max_num_tokens 8192,
# chunked prefill, engine.json KV config (FP8 KV, block reuse). 192 prompts = 32768-token shared
# prefix + 2048 unique tokens, 40 greedy tokens with logprobs, c64@64 then c128@128.
# Records: CUDA graph batch sizes captured / replayed, nvfp4_gemm AutoTuner picks at M=64/128
# per (N, K), and how many fi_cutlass launches were issued during graph capture.
import glob
import json
import os
import subprocess
import sys
import threading
import time

os.environ.setdefault("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
PREFIX, SUFFIX, OSL = 32768, 2048, 40
GROUPS = (("c64", 64), ("c128", 128))


def log(*a):
    print("[%s]" % time.strftime("%H:%M:%S"), *a, flush=True)


def heartbeat(tag):
    t0 = time.time()

    def loop():
        while True:
            time.sleep(30)
            log("[heartbeat %s] %.0fs" % (tag, time.time() - t0))

    threading.Thread(target=loop, daemon=True).start()


def corpus_ids(tok, n_tokens):
    files = sorted(glob.glob("docs/source/**/*.md", recursive=True))
    files += sorted(glob.glob("tensorrt_llm/_torch/**/*.md", recursive=True))
    files += sorted(glob.glob("examples/**/*.md", recursive=True))
    files += sorted(glob.glob("tensorrt_llm/**/*.py", recursive=True))
    ids = []
    for f in files:
        try:
            with open(f, encoding="utf-8", errors="ignore") as fh:
                ids.extend(tok.encode(fh.read(), add_special_tokens=False))
        except OSError:
            continue
        if len(ids) >= n_tokens:
            break
    assert len(ids) >= n_tokens, (len(ids), n_tokens)
    return ids[:n_tokens]


def run_arm(ckpt, arm, out_path):
    heartbeat(arm)
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.autotuner import AutoTuner
    from tensorrt_llm._torch.custom_ops import torch_custom_ops as tco
    from tensorrt_llm._torch.pyexecutor import cuda_graph_runner as cgr
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

    captures, replays, cur = set(), {}, {"phase": "load"}
    o_rep, o_cap = cgr.CUDAGraphRunner.replay, cgr.CUDAGraphRunner.capture

    def replay(self, key, *a, **kw):
        k = "%s:bs%d" % (cur["phase"], cgr.KeyType(*key).batch_size)
        replays[k] = replays.get(k, 0) + 1
        return o_rep(self, key, *a, **kw)

    def capture(self, key, *a, **kw):
        captures.add(cgr.KeyType(*key).batch_size)
        return o_cap(self, key, *a, **kw)

    cgr.CUDAGraphRunner.replay, cgr.CUDAGraphRunner.capture = replay, capture
    fi_calls = {}
    if hasattr(tco, "FlashInferCutlassNVFP4Runner"):
        o_fwd = tco.FlashInferCutlassNVFP4Runner.forward

        def fwd(self, inputs, tactic=-1, **kw):
            k = "M%d_N%d_K%d_t%d" % (inputs[0].shape[0], inputs[1].shape[0],
                                     inputs[0].shape[1] * 2, tactic)
            fi_calls[k] = fi_calls.get(k, 0) + 1
            return o_fwd(self, inputs, tactic=tactic, **kw)

        tco.FlashInferCutlassNVFP4Runner.forward = fwd

    eng = json.load(open("experiments/ascent-config/engine.json"))
    kvc = dict(eng.get("kv_cache_config", {}))
    t0 = time.time()
    llm = LLM(model=ckpt, max_seq_len=36864, max_batch_size=128, max_num_tokens=8192,
              enable_chunked_prefill=True, cuda_graph_config=CudaGraphConfig(max_batch_size=128),
              kv_cache_config=KvCacheConfig(**kvc), batch_wait_timeout_ms=200)
    log("%s LLM ready in %.1fs; graphs captured at %s" % (arm, time.time() - t0, sorted(captures)))
    # AutoTuner picks of trtllm::nvfp4_gemm at the decode buckets
    picks = {}
    for key, val in AutoTuner.get().profiling_cache.cache.items():
        if "nvfp4_gemm" not in str(key[0]):
            continue
        s = str(key)
        if "(64," in s or "(128," in s:
            picks[s[:160]] = str(val)[:120]
    tok = llm.tokenizer
    n = sum(c for _, c in GROUPS)
    ids = corpus_ids(tok, PREFIX + n * SUFFIX)
    prompts = [ids[:PREFIX] + ids[PREFIX + i * SUFFIX:PREFIX + (i + 1) * SUFFIX] for i in range(n)]
    sp = SamplingParams(max_tokens=OSL, temperature=0.0, top_k=1, ignore_eos=True, logprobs=1)
    res, base = {}, 0
    fi_after_load = dict(fi_calls)
    for g, conc in GROUPS:
        cur["phase"] = g
        t = time.time()
        outs = llm.generate(prompts[base:base + conc], sp, use_tqdm=False)
        for j, o in enumerate(outs):
            c = o.outputs[0]
            lps = []
            for tid, d in zip(c.token_ids, c.logprobs or []):
                lp = d.get(tid) if isinstance(d, dict) else None
                lps.append(float(lp.logprob) if lp is not None else None)
            res["%s-p%d" % (g, j)] = {"ids": list(c.token_ids), "lp": lps}
        log("%s %s: %d prompts %.1fs" % (arm, g, conc, time.time() - t))
        base += conc
    dec = sorted(set(int(k.split(":bs")[1]) for k in replays if not k.startswith("load")))
    json.dump({"arm": arm, "captured": sorted(captures), "replayed_bs": dec,
               "replays": replays, "fi_calls_by_load_end": fi_after_load,
               "fi_calls_total": fi_calls, "picks": picks, "res": res}, open(out_path, "w"))
    llm.shutdown()
    os._exit(0)


def main():
    ckpt = sys.argv[1]
    if len(sys.argv) > 2:
        run_arm(ckpt, sys.argv[2], sys.argv[3])
        return
    heartbeat("driver")
    art = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
    os.makedirs(art, exist_ok=True)
    arms = {}
    for arm, env in (("fi", {}), ("base", {"TRTLLM_NVFP4_FI_CUTLASS": "0"})):
        path = os.path.join(art, "e2e_%s.json" % arm)
        e = dict(os.environ, **env)
        log("starting arm", arm)
        rc = subprocess.call([sys.executable, __file__, ckpt, arm, path], env=e)
        log("arm", arm, "rc", rc)
        arms[arm] = json.load(open(path))
    a, b = arms["fi"], arms["base"]
    same_ids, max_dlp, first_div = 0, 0.0, {}
    for k in a["res"]:
        x, y = a["res"][k], b["res"][k]
        if x["ids"] == y["ids"]:
            same_ids += 1
        else:
            first_div[k] = next(i for i, (p, q) in enumerate(zip(x["ids"], y["ids"])) if p != q)
        for i, (p, q) in enumerate(zip(x["ids"], y["ids"])):
            if p != q:
                break
            if x["lp"][i] is not None and y["lp"][i] is not None:
                max_dlp = max(max_dlp, abs(x["lp"][i] - y["lp"][i]))
    fi_used = sum(a["fi_calls_total"].values())
    ok = (128 in a["captured"] and 64 in a["captured"] and 64 in a["replayed_bs"]
          and 128 in a["replayed_bs"] and fi_used > 0 and sum(b["fi_calls_total"].values()) == 0)
    print(json.dumps({
        "passed": ok,
        "fi_arm": {"captured": a["captured"], "replayed_bs": a["replayed_bs"],
                   "fi_calls_by_load_end": a["fi_calls_by_load_end"],
                   "fi_calls_total": a["fi_calls_total"], "picks": a["picks"]},
        "base_arm": {"captured": b["captured"], "replayed_bs": b["replayed_bs"],
                     "fi_calls_total": b["fi_calls_total"], "picks": b["picks"]},
        "prompts": len(a["res"]), "identical_token_sequences": same_ids,
        "first_divergence_index": first_div, "max_abs_dlogprob_before_divergence": max_dlp,
        "compared": ("decode CUDA graphs at bs 64 and 128 captured and replayed with fi_cutlass "
                     "launches captured (FI arm) and none in the kill-switch arm; tokens/logprobs "
                     "of 192 prompts (c64@64, c128@128, 32768 shared prefix + 2048 unique, 40 "
                     "greedy) FI arm vs kill-switch arm, informational (reordered)")}))


if __name__ == "__main__":
    main()
