#!/usr/bin/env python3
"""Branch 23 under breakable prefill CUDA graphs (BCG, main 8c2c2cb5): replay evidence.

Two model loads, each in its own subprocess, with this tree's engine.json overlay (breakable
backend, disable_mm_encoder, buckets 300/600/900/1000/1200/1500/1800/2000) and ONE fresh
TLLM_AUTOTUNER_CACHE_PATH shared by both (arm epi0 tunes and writes it, arm epi1 reads it,
so both serve the same tactics):
  epi1: TRTLLM_GEMMA4_GATEUP_GELU_EPI=1 (candidate ON, gate_up rows permuted)
  epi0: TRTLLM_GEMMA4_GATEUP_GELU_EPI=0 (main's MLP path)
Each arm runs passes R (replay), E (eager: BreakableCUDAGraphRunner.has_graph gated off from
this script, as bcg_ab.py inproc does), R2 (replay) over the same prompts: c=1 prompts of
exactly every bucket size (2 each), 2 x isl8000 (eager, > largest bucket), and batches
2x300 / 3x300 / 4x300 / 2x1000 (600 / 900 / 1200 / 2000 tokens). Token ids + top-20 logprobs.
Arm epi1 also wraps modeling_gemma4.gate_up_gelu_fp4_permuted to count, per path and M,
calls made while the stream is capturing, and records any pick-memo miss, tuning-mode call or
missing fused-op workspace inside capture.
Each arm logs its gate_up (N = 2*intermediate) tuner entries, standard and dense-bucket
(trtllm::nvfp4_gemm::gemm / gemm_dense), and the cache file's sha256 is recorded before and
after each arm, so both arms provably served the same picks.
Checks: per arm R == E == R2 bitwise; epi1 == epi0 bitwise for R and E; replayed > 0 in R/R2;
the two arms' gate_up picks identical.
usage: bcg_check.py <checkpoint> [arm <epi0|epi1> <out.json>]
"""
import hashlib
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("TLLM_WORKER_USE_SINGLE_PROCESS", "1")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE_JSON = os.path.join(HERE, "..", "..", "ascent-config", "engine.json")
BUCKETS = [300, 600, 900, 1000, 1200, 1500, 1800, 2000]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def overlay():
    with open(ENGINE_JSON) as f:
        return json.load(f)


def sha(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def make_prompts(seed, isl, n):
    rng = np.random.default_rng(seed * 100003 + isl)
    return [[int(x) for x in rng.integers(1000, 200000, size=isl)] for _ in range(n)]


def summarize(o):
    c = o.outputs[0]
    lps = []
    for d in (c.logprobs or []):
        if isinstance(d, dict):
            lps.append(sorted((int(k), float(getattr(v, "logprob", v))) for k, v in d.items()))
        else:
            lps.append(float(d))
    return {"ids": list(c.token_ids), "lp": lps}


def gate_up_picks(n_gate_up):
    from tensorrt_llm._torch.autotuner import AutoTuner
    out = {}
    for k, v in AutoTuner.get().profiling_cache.cache.items():
        if k[0] in ("trtllm::nvfp4_gemm::gemm", "trtllm::nvfp4_gemm::gemm_dense") and \
                str(n_gate_up) in str(k[3]):
            out[f"{k[0].split('::')[-1]}|{k[3]}"] = list(v[1]) if isinstance(v[1], (list, tuple)) else v[1]
    return dict(sorted(out.items()))


def arm(ck, name, out_path):
    import torch

    from tensorrt_llm._torch.kernels.fused_ops import nvfp4_gelu_gemm_sm120 as ge
    from tensorrt_llm._torch.models import modeling_gemma4 as mg

    cfg = json.load(open(os.path.join(ck, "config.json")))
    tc = cfg.get("text_config", cfg)
    n_gate_up = 2 * int(tc["intermediate_size"])

    cap = {"capture_calls": {}, "eager_calls": {}, "memo_miss_in_capture": 0,
           "tuning_mode_in_capture": 0, "ws_missing_at_capture": 0}
    orig = mg.gate_up_gelu_fp4_permuted

    def counted(act_fp4, *a, **k):
        from tensorrt_llm._torch.autotuner import AutoTuner
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing:
            if AutoTuner.get().is_tuning_mode:
                cap["tuning_mode_in_capture"] += 1
            if not ge._ws_cache:
                cap["ws_missing_at_capture"] += 1
            n0 = len(ge._epi_decisions)
        res = orig(act_fp4, *a, **k)
        if capturing and len(ge._epi_decisions) != n0:
            cap["memo_miss_in_capture"] += 1
        d = cap["capture_calls" if capturing else "eager_calls"]
        key = f"{res[2]}@{act_fp4.shape[0]}"
        d[key] = d.get(key, 0) + 1
        return res

    mg.gate_up_gelu_fp4_permuted = counted

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import KvCacheConfig
    ov = overlay()
    log(f"{name}: EPI={os.environ.get('TRTLLM_GEMMA4_GATEUP_GELU_EPI')} overlay {ov} "
        f"cache {os.environ.get('TLLM_AUTOTUNER_CACHE_PATH')}")
    t0 = time.time()
    llm = LLM(model=ck, kv_cache_config=KvCacheConfig(dtype="fp8", enable_block_reuse=False,
                                                      free_gpu_memory_fraction=0.8),
              max_num_tokens=8192, max_batch_size=64, max_seq_len=9216, batch_wait_timeout_ms=1500.0,
              **ov)
    log(f"{name}: startup {time.time() - t0:.1f}s")
    picks = gate_up_picks(n_gate_up)
    log(f"{name}: gate_up picks {picks}")
    from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph_runner import BreakableCUDAGraphRunner
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import graphs as _g
    replay_on = [True]
    orig_has_graph = BreakableCUDAGraphRunner.has_graph
    BreakableCUDAGraphRunner.has_graph = lambda self, n: replay_on[0] and orig_has_graph(self, n)
    counts = getattr(_g, "_BCG_STEP_COUNTS", None)

    import gc
    permuted = sorted({bool(getattr(o, "_gelu_epi_permuted", False)) for o in gc.get_objects()
                       if isinstance(o, mg._Gemma4GeluQuantMLP)})
    params = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True, logprobs=20)
    groups = []
    for isl in BUCKETS + [8000]:
        groups += [(f"1x{isl}", [p]) for p in make_prompts(1, isl, 2)]
    for n, isl in ((2, 300), (3, 300), (4, 300), (2, 1000)):
        groups.append((f"{n}x{isl}", make_prompts(2 + n, isl, n)))
    for isl in (300, 1000, 2000, 8000):  # warmup
        llm.generate(make_prompts(5, isl, 1), sampling_params=params, use_tqdm=False)
    cap_after_startup = json.loads(json.dumps(cap))
    passes, steps = {}, {}
    for tag, on in (("R", True), ("E", False), ("R2", True)):
        replay_on[0] = on
        before = dict(counts) if counts is not None else None
        res = []
        for _, grp in groups:
            outs = llm.generate(grp, sampling_params=params, use_tqdm=False)
            res.append([summarize(o) for o in outs])
        passes[tag] = res
        if counts is not None:
            steps[tag] = {k: counts[k] - before[k] for k in counts}
        log(f"{name}: pass {tag} replay={on} steps={steps.get(tag)}")
    BreakableCUDAGraphRunner.has_graph = orig_has_graph
    picks_end = gate_up_picks(n_gate_up)
    with open(out_path, "w") as f:
        json.dump({"labels": [g[0] for g in groups], "passes": passes, "steps": steps,
                   "counters_after_startup": cap_after_startup, "counters_end": cap,
                   "mlp_permuted_values": permuted, "path_calls": dict(ge.path_calls),
                   "gate_up_picks": picks, "gate_up_picks_end": picks_end}, f)
    log(f"{name}: wrote {out_path}")


def diff(labels, a, b):
    per = {}
    for lab, x, y in zip(labels, a, b):
        d = per.setdefault(lab, [0, 0])
        d[1] += len(x)
        d[0] += sum(1 for u, v in zip(x, y) if u != v)
    return sum(v[0] for v in per.values()), {k: f"{v[0]}/{v[1]}" for k, v in per.items() if v[0]}


def main():
    ck = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "arm":
        arm(ck, sys.argv[3], sys.argv[4])
        return
    art = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
    os.makedirs(art, exist_ok=True)
    cache = os.path.join(art, f"autotuner_cache_{int(time.time())}.json")
    res, cache_sha = {}, {}
    for name, epi in (("epi0", "0"), ("epi1", "1")):
        env = dict(os.environ, TRTLLM_GEMMA4_GATEUP_GELU_EPI=epi, TLLM_AUTOTUNER_CACHE_PATH=cache)
        path = os.path.join(art, f"bcg_{name}.json")
        log(f"arm {name}")
        before = sha(cache)
        p = subprocess.run([sys.executable, __file__, ck, "arm", name, path], env=env)
        cache_sha[name] = {"before": before, "after": sha(cache)}
        if p.returncode != 0:
            print(json.dumps({"passed": False, "error": f"arm {name} rc={p.returncode}"}))
            return
        res[name] = json.load(open(path))
    labels = res["epi1"]["labels"]
    cmp, ok = {}, True
    for name in ("epi0", "epi1"):
        P = res[name]["passes"]
        for a, b in (("R", "E"), ("R2", "E"), ("R2", "R")):
            n, per = diff(labels, P[a], P[b])
            cmp[f"{name}_{a}_vs_{b}"] = {"n_diff": n, "per_group": per}
            ok &= n == 0
        for tag in ("R", "R2"):
            ok &= res[name]["steps"].get(tag, {}).get("replayed", 0) > 0
    for tag in ("R", "E", "R2"):
        n, per = diff(labels, res["epi1"]["passes"][tag], res["epi0"]["passes"][tag])
        cmp[f"epi1_vs_epi0_{tag}"] = {"n_diff": n, "per_group": per}
        ok &= n == 0
    picks_same = (res["epi0"]["gate_up_picks"] == res["epi1"]["gate_up_picks"]
                  and res["epi0"]["gate_up_picks_end"] == res["epi1"]["gate_up_picks_end"]
                  and len(res["epi0"]["gate_up_picks"]) > 0)
    ok &= picks_same
    c1 = res["epi1"]["counters_end"]
    cap_calls = c1["capture_calls"]
    ok &= (c1["memo_miss_in_capture"] == 0 and c1["tuning_mode_in_capture"] == 0
           and c1["ws_missing_at_capture"] == 0)
    ok &= res["epi1"]["mlp_permuted_values"] == [True] and res["epi0"]["mlp_permuted_values"] == [False]
    ok &= sum(cap_calls.values()) > 0
    ok &= sum((res["epi0"]["counters_end"]["capture_calls"] or {}).values()) == 0
    n_out = sum(len(x) for x in res["epi1"]["passes"]["R"])
    has_lp = bool(res["epi1"]["passes"]["R"][0][0]["lp"])
    print(json.dumps({
        "passed": bool(ok and has_lp),
        "compared": "token ids + top-20 logprobs bitwise, BCG overlay, %d outputs per pass: per arm "
                    "replay R / eager E / replay R2 in one process; candidate (EPI=1) vs main path "
                    "(EPI=0) on a shared tuner cache; groups 1x{300..2000 buckets, 8000}, 2/3/4x300, "
                    "2x1000" % n_out,
        "comparisons": cmp, "steps": {k: v["steps"] for k, v in res.items()},
        "epi1_dispatch_in_capture": cap_calls,
        "epi1_dispatch_eager": c1["eager_calls"],
        "epi1_capture_checks": {k: c1[k] for k in ("memo_miss_in_capture", "tuning_mode_in_capture",
                                                   "ws_missing_at_capture")},
        "epi1_capture_calls_during_startup": res["epi1"]["counters_after_startup"]["capture_calls"],
        "mlp_permuted": {k: v["mlp_permuted_values"] for k, v in res.items()},
        "path_calls": {k: v["path_calls"] for k, v in res.items()},
        "gate_up_picks": {k: v["gate_up_picks"] for k, v in res.items()},
        "gate_up_picks_same": bool(picks_same), "tuner_cache_sha256": cache_sha,
        "overlay": overlay(), "logprobs_present": has_lp}), flush=True)


if __name__ == "__main__":
    main()
