"""Idea 10 (branch 18) end-to-end: real model, CUDA-graph decode at 17-32 rows.

Two arms, one process each (LLM API, serving engine settings: FP8 KV cache,
block reuse, chunked prefill, max_batch_size 32, CUDA graphs up to 32), one
shared AutoTuner cache (written by the first arm):
  main  TRTLLM_FP8_NVFP4_DECODE_WIDE=0  (main's routing: Marlin <= 16 rows)
  cand  default                         (per-(N,K) Marlin cap 32)
Each arm generates 32 prompts concurrently (decode batch 32), then 20 (graph
batch 24), 64 greedy tokens, ignore_eos, logprobs=1.
Instrumentation (wrapping FP8QDQLinearMethod.apply / CUDAGraphRunner):
  - route of every FP8-origin linear call during graph capture, by (N,K) and
    rows: 'w4a4' (prefill copy called) or 'decode' (Marlin / GEMV copy);
  - eager (non-capturing) calls with 17..32 rows after engine init: must be 0
    (pure-decode steps replay graphs, no eager fallback);
  - graph replays per batch size.
Pass: cand captures route (14336,5120) and (5120,6144) to 'decode' and
(16384,5120) to 'w4a4' at rows 17..32; main routes all to 'w4a4' there; both
route rows <= 16 identically to 'decode'; no eager 17..32-row calls; replays
at batch 24 and 32 > 0. Token / logprob agreement is reported.
Usage: e2e.py <checkpoint> <workdir>
"""
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = [
    "List the first ten prime numbers: 2, 3, 5,",
    "Count from 20 to 30: 20, 21,",
    "The days of the week in order are Monday, Tuesday,",
    "The months of the year in order are January, February, March,",
    "The planets of the solar system, in order from the Sun, are Mercury,",
    "Q: What is 12 * 12? Show the result only.\nA: 12 * 12 =",
    "Q: What is 365 - 120?\nA: 365 - 120 =",
    "Q: What is 1000 / 8?\nA: 1000 / 8 =",
    "Q: What are the capital cities of France, Japan and Italy?\nA: The capitals are",
    "Q: Name the author of 'Pride and Prejudice' and the author of 'Hamlet'.\nA:",
    "The powers of two are 1, 2, 4, 8,",
    "The square numbers are 1, 4, 9,",
    "Write a detailed explanation of how a refrigerator works.\n\n",
    "Explain the causes and consequences of the French Revolution.\n\n",
    "Describe step by step how to implement binary search in Python.\n\n",
    "Write a short story about a lighthouse keeper.\n\n",
]
PROMPTS = BASE + [f"Answer briefly. {p}" for p in BASE]


def log(*a):
    print(*a, flush=True)


def heartbeat(tag):
    t0 = time.time()

    def loop():
        while True:
            time.sleep(30)
            log(f"[heartbeat {tag}] {time.time() - t0:.0f}s")

    threading.Thread(target=loop, daemon=True).start()


def run_arm(arm, ckpt, out_path):
    heartbeat(arm)
    if arm == "main":
        os.environ["TRTLLM_FP8_NVFP4_DECODE_WIDE"] = "0"
    else:
        os.environ.pop("TRTLLM_FP8_NVFP4_DECODE_WIDE", None)
    os.environ["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"
    import torch

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.modules import fp8_prefill_nvfp4 as fpn
    from tensorrt_llm._torch.modules import linear as L
    from tensorrt_llm._torch.pyexecutor import cuda_graph_runner as cgr
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

    state = {"init": True}
    capture_routes = {}  # "N,K" -> {rows: route}
    eager_mid = {}  # "N,K,rows" -> count (17..32 rows, not capturing)
    replays = {}
    orig_apply = L.FP8QDQLinearMethod.apply

    def apply(self, module, input, bias):
        rows = input.reshape(-1, input.shape[-1]).shape[0]
        c0 = fpn._stats["calls"]
        out = orig_apply(self, module, input, bias)
        if getattr(module, "_nvfp4_decode_weight", None) is None:
            return out
        key = f"{module.out_features},{module.in_features}"
        route = "w4a4" if fpn._stats["calls"] > c0 else "decode"
        if torch.cuda.is_current_stream_capturing():
            capture_routes.setdefault(key, {})[rows] = route
        elif not state["init"] and 17 <= rows <= 32:
            k2 = f"{key},{rows},{route}"
            eager_mid[k2] = eager_mid.get(k2, 0) + 1
        return out

    L.FP8QDQLinearMethod.apply = apply
    runner_cls = cgr.CUDAGraphRunner
    orig_replay = runner_cls.replay

    def replay(self, key, *a, **kw):
        bs = str(getattr(key, "batch_size", key))
        replays[bs] = replays.get(bs, 0) + 1
        return orig_replay(self, key, *a, **kw)

    runner_cls.replay = replay

    t0 = time.time()
    llm = LLM(model=ckpt, max_seq_len=36864, max_batch_size=32,
              max_num_tokens=8192, enable_chunked_prefill=True,
              cuda_graph_config=CudaGraphConfig(max_batch_size=32),
              kv_cache_config=KvCacheConfig(
                  dtype="fp8", enable_block_reuse=True,
                  mamba_state_config={
                      "additional_snapshot_offsets_from_start": [32768]}))
    state["init"] = False
    log(f"[{arm}] LLM ready in {time.time() - t0:.1f}s; capture routes "
        f"{json.dumps(capture_routes, sort_keys=True)}")

    def lp(c):
        return [list(d.values())[0].logprob if isinstance(d, dict) else d
                for d in (c.logprobs or [])]

    items = {}
    sp = SamplingParams(max_tokens=64, temperature=0.0, ignore_eos=True,
                        logprobs=1)
    for tag, prompts in (("b32", PROMPTS), ("b20", PROMPTS[:20])):
        outs = llm.generate(prompts, sp)
        for i, o in enumerate(outs):
            c = o.outputs[0]
            items[f"{tag}_{i}"] = {"ids": list(c.token_ids), "lp": lp(c),
                                   "text": c.text}
        log(f"[{arm}] {tag} done; replays {replays}; eager 17-32 {eager_mid}")
    torch.cuda.synchronize()
    snap = None
    if hasattr(torch.ops.trtllm, "marlin_nvfp4_persistent_workspace_snapshot"):
        s = torch.ops.trtllm.marlin_nvfp4_persistent_workspace_snapshot(
            torch.empty(1, device="cuda"))
        snap = {"ints": int(s.numel()),
                "all_zero": bool((s == 0).all().item()) if s.numel() else None}
    res = {"arm": arm, "items": items, "capture_routes": capture_routes,
           "eager_mid_rows": eager_mid, "replays": replays,
           "workspace": snap}
    with open(out_path, "w") as f:
        json.dump(res, f)
    log(f"[{arm}] done")
    llm.shutdown()


def compare(a, b):
    n_same, max_dlp, n_tok, n_match, firsts = 0, 0.0, 0, 0, []
    for k in a["items"]:
        ia, ib = a["items"][k], b["items"][k]
        n_same += ia["ids"] == ib["ids"]
        d = next((i for i, (x, y) in enumerate(zip(ia["ids"], ib["ids"]))
                  if x != y), None)
        upto = len(ia["ids"]) if d is None else d + 1
        for x, y in zip(ia["lp"][:upto], ib["lp"][:upto]):
            if x is not None and y is not None:
                max_dlp = max(max_dlp, abs(x - y))
        n_tok += len(ia["ids"])
        n_match += upto if d is None else d
        if d is not None:
            firsts.append((k, d))
    return {"n_sequences": len(a["items"]), "identical_sequences": n_same,
            "tokens_before_first_divergence_frac": n_match / max(n_tok, 1),
            "max_abs_dlogprob_up_to_first_divergence": max_dlp,
            "first_divergence": firsts}


def check_routes(r, wide):
    bad = []
    for key, by_rows in r["capture_routes"].items():
        for rows, route in by_rows.items():
            rows = int(rows)
            if rows <= 16:
                want = "decode"
            elif rows <= 32:
                want = ("decode" if wide and key in ("14336,5120", "5120,6144")
                        else "w4a4")
            else:
                want = "w4a4"
            if route != want:
                bad.append((key, rows, route, want))
    return bad


def main(ckpt, workdir):
    os.makedirs(workdir, exist_ok=True)
    cache = os.path.join(workdir, "autotuner_cache.json")
    if os.path.exists(cache):
        os.remove(cache)
    os.environ["TLLM_AUTOTUNER_CACHE_PATH"] = cache
    paths = {}
    for arm in ("main", "cand"):
        p = os.path.join(workdir, f"e2e_{arm}.json")
        paths[arm] = p
        t0 = time.time()
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "arm",
                            arm, ckpt, p])
        log(f"arm {arm} exit {r.returncode} in {time.time() - t0:.0f}s")
        if r.returncode != 0:
            print(json.dumps({"passed": False,
                              "error": f"arm {arm} rc={r.returncode}"}))
            return
    a = json.load(open(paths["main"]))
    b = json.load(open(paths["cand"]))
    bad_main = check_routes(a, False)
    bad_cand = check_routes(b, True)
    captured_mid = sorted({int(r) for v in b["capture_routes"].values()
                           for r in v if 17 <= int(r) <= 32})
    ok = (not bad_main and not bad_cand and not a["eager_mid_rows"]
          and not b["eager_mid_rows"] and b["replays"].get("32", 0) > 0
          and b["replays"].get("20", 0) > 0
          and captured_mid == list(range(17, 33))
          and (b["workspace"] is None or b["workspace"]["all_zero"]
               is not False))
    res = {"passed": ok,
           "compared": "real model, CUDA-graph decode at batch 32 and 24 "
                       "(20 reqs padded): FP8-origin linear routes per "
                       "captured graph, eager 17-32-row calls, graph replays; "
                       "cand vs main (kill switch) greedy tokens/logprobs",
           "route_violations_main": bad_main, "route_violations_cand": bad_cand,
           "captured_rows_17_32": captured_mid,
           "capture_routes_cand": b["capture_routes"],
           "capture_routes_main": a["capture_routes"],
           "eager_17_32_main": a["eager_mid_rows"],
           "eager_17_32_cand": b["eager_mid_rows"],
           "replays_main": a["replays"], "replays_cand": b["replays"],
           "workspace_cand": b["workspace"],
           "cand_vs_main": compare(a, b),
           "sample_text": {k: [a["items"][k]["text"][:120],
                               b["items"][k]["text"][:120]]
                           for k in ("b32_5", "b32_8", "b20_2")}}
    print(json.dumps(res))


if __name__ == "__main__":
    if sys.argv[1] == "arm":
        run_arm(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        main(sys.argv[1], sys.argv[2])
