#!/usr/bin/env python3
"""Idea 65 full-model numerics gate: main's FlashInfer sources vs the branch tree, in-process LLM.

  driver:  model_ab.py drive --arms main,tree,main
  worker:  model_ab.py dump --arm <arm> --out f

Arms: 'tree' = the job's FlashInfer tree (v2m + hardware cvt); 'main' = bench.py's text-patched
copy that reverts both edits (== main's sources). Every arm sets FlashInfer's JIT include/csrc
dirs BEFORE TRT-LLM imports FlashInfer, disables AOT artifacts and uses a fresh
FLASHINFER_WORKSPACE_BASE (the two 'main' arms share theirs), so every module is built from its
arm's sources.
Engine: TLLM_WORKER_USE_SINGLE_PROCESS=1, FP8 KV, block reuse off, max_num_tokens 8192,
max_batch_size 64, batch_wait_timeout_ms 1500 (groups sent together are scheduled together).
TRTLLM_FI_HOST_PREFILL_PLAN, TRTLLM_GEMMA4_LAST_LAYER_ATTN_ROWS and FLASHINFER_FA2_D512_TILE64
stay at main's defaults (unset) in every arm and are printed with READ_ROWS_STATS per arm.
Prompts (fresh random tokens in the chat template, fixed seed): c=1 isl300 x4, isl1k x4, isl2k x3,
isl4k x3, isl8k x3; groups sent together: 2 x (4 x isl8k), 2 x (4 x isl1k). Greedy, 2 output
tokens (prefill + one decode step), top-20 logprobs per token. Each arm runs the set twice.
"""
import argparse
import json
import os
import pathlib
import struct
import subprocess
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ENV_NAMES = ("TRTLLM_FI_HOST_PREFILL_PLAN", "TRTLLM_GEMMA4_LAST_LAYER_ATTN_ROWS",
             "FLASHINFER_FA2_D512_TILE64")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def prompt_sets(tok, seed=6565):
    import numpy as np
    rng = np.random.default_rng(seed)

    def tmpl(text):
        out = tok.apply_chat_template([{"role": "user", "content": text}],
                                      add_generation_prompt=True, tokenize=True)
        if hasattr(out, "keys"):
            out = out["input_ids"]
        if out and isinstance(out[0], (list, tuple)):
            out = out[0]
        return [int(t) for t in out]
    overhead = len(tmpl("x"))

    def mk(isl):
        n = max(8, isl - overhead)
        ids = [int(x) for x in rng.integers(1000, 200000, size=n)]
        for _ in range(6):
            out = tmpl(tok.decode(ids))
            if len(out) <= isl:
                break
            ids = ids[:max(8, len(ids) - (len(out) - isl) - 4)]
        return out
    sets = []
    for isl, nm, n in ((300, "isl300", 4), (1000, "isl1k", 4), (2000, "isl2k", 3),
                       (4000, "isl4k", 3), (8000, "isl8k", 3)):
        for i in range(n):
            sets.append((f"{nm}@1#{i}", [mk(isl)]))
    for isl, nm in ((8000, "isl8k"), (1000, "isl1k")):
        for g in range(2):
            sets.append((f"{nm}@4#g{g}", [mk(isl) for _ in range(4)]))
    return sets


def summarize(o):
    c = o.outputs[0]
    lps = []
    if c.logprobs:
        for d in c.logprobs:
            if isinstance(d, dict):
                lps.append(sorted((int(k), float(getattr(v, "logprob", v))) for k, v in d.items()))
            else:
                lps.append(float(d))
    return {"ids": [int(t) for t in c.token_ids], "lp": lps}


def bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def same(a, b):
    if a["ids"] != b["ids"] or len(a["lp"]) != len(b["lp"]):
        return False
    for la, lb in zip(a["lp"], b["lp"]):
        if len(la) != len(lb):
            return False
        for (ta, va), (tb, vb) in zip(la, lb):
            if ta != tb or bits(va) != bits(vb):
                return False
    return True


def cmp_runs(x, y):
    n = eq = flips = 0
    maxd = 0.0
    first = []
    for name in x:
        for i, (a, b) in enumerate(zip(x[name], y[name])):
            n += 1
            if same(a, b):
                eq += 1
            elif len(first) < 5:
                first.append(f"{name}[{i}]")
            flips += a["ids"] != b["ids"]
            for la, lb in zip(a["lp"], b["lp"]):
                da, db = dict(la), dict(lb)
                for k in set(da) & set(db):
                    maxd = max(maxd, abs(da[k] - db[k]))
    return dict(prompts=n, bitwise_equal=eq, token_flips=flips, max_abs_dlogprob_top20=maxd,
                first_diffs=first)


def dump(a):
    import bench
    if a.arm != "tree":
        d, files = bench.make_arm_tree(a.arm, pathlib.Path(tempfile.mkdtemp(
            prefix=f"idea65_mab_{a.arm}_")) / "tree")
        log(f"arm {a.arm} sources {json.dumps(files)} at {d}")
    os.environ["FLASHINFER_WORKSPACE_BASE"] = a.base or tempfile.mkdtemp(
        prefix=f"idea65_mab_{a.arm}_ws_")
    os.environ["FLASHINFER_JIT_DEBUG"] = "0"
    os.environ.setdefault("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    from flashinfer.jit import env as jit_env
    if a.arm != "tree":
        jit_env.FLASHINFER_INCLUDE_DIR = pathlib.Path(d) / "include"
        jit_env.FLASHINFER_CSRC_DIR = pathlib.Path(d) / "csrc"
    jit_env.get_aot_artifacts = lambda name: ()
    import hashlib
    inc = pathlib.Path(jit_env.FLASHINFER_INCLUDE_DIR)
    for rel in ("flashinfer/attention/prefill.cuh", "flashinfer/vec_dtypes.cuh"):
        txt = (inc / rel).read_text()
        log(f"SRC {rel} sha {hashlib.sha256(txt.encode()).hexdigest()[:16]} "
            f"v2m={'quarter-warp takes 4 columns' in txt} "
            f"hwcvt_active={'FLASHINFER_HARDWARE_FP8_E4M3_TO_BF16_CONVERSION_ENABLED' in txt and 'idea65 arm: software' not in txt}")
    log("ENV " + json.dumps({k: os.environ.get(k, "<unset: main default>") for k in ENV_NAMES}))
    log(f"FLASHINFER_WORKSPACE_BASE={os.environ['FLASHINFER_WORKSPACE_BASE']} "
        f"load={os.getloadavg()}")
    import threading
    stop = threading.Event()

    def hb():
        t0 = time.time()
        while not stop.wait(60):
            log(f"[hb {a.arm}] {time.time() - t0:.0f}s load={os.getloadavg()}")
    threading.Thread(target=hb, daemon=True).start()

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import KvCacheConfig
    cfg = json.load(open(os.path.join(a.ckpt, "config.json")))
    tc = cfg.get("text_config", cfg)
    log("config: " + json.dumps({k: tc.get(k) for k in (
        "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "head_dim", "global_head_dim", "num_global_key_value_heads", "sliding_window")}))
    llm = LLM(model=a.ckpt,
              kv_cache_config=KvCacheConfig(dtype="fp8", enable_block_reuse=False,
                                            free_gpu_memory_fraction=0.8),
              max_num_tokens=8192, max_batch_size=64, max_seq_len=9216,
              batch_wait_timeout_ms=1500.0)
    params = SamplingParams(max_tokens=2, temperature=0.0, ignore_eos=True, logprobs=20)
    import numpy as np
    wrng = np.random.default_rng(5)
    for isl in (300, 1000, 8000):
        llm.generate([[int(x) for x in wrng.integers(1000, 200000, size=isl)] for _ in range(2)],
                     sampling_params=params, use_tqdm=False)
    from tensorrt_llm._torch.attention.backends import flashinfer as fib
    stats0 = dict(fib.READ_ROWS_STATS)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    sets = prompt_sets(tok)
    lens = [len(p) for _, ps in sets for p in ps]
    log(f"prompts: {len(lens)}, len min {min(lens)} max {max(lens)}")
    res = {}
    for p in range(a.passes):
        out = {}
        for name, prompts in sets:
            outs = llm.generate(prompts, sampling_params=params, use_tqdm=False)
            out[name] = [summarize(o) for o in outs]
        res[f"pass{p}"] = out
        log(f"pass {p} done; first {out[sets[0][0]][0]['ids']}")
    stats = {k: fib.READ_ROWS_STATS[k] - stats0.get(k, 0) for k in fib.READ_ROWS_STATS}
    log(f"READ_ROWS_STATS (measured passes) {json.dumps(stats)}")
    jd = jit_env.FLASHINFER_JIT_DIR
    log(f"JIT modules built here: {sorted(os.listdir(jd)) if os.path.isdir(jd) else []}")
    stop.set()
    with open(a.out, "w") as f:
        json.dump({"res": res, "read_rows_stats": stats,
                   "env": {k: os.environ.get(k) for k in ENV_NAMES}}, f)
    log(f"wrote {a.out} load={os.getloadavg()}")
    print(json.dumps({"passed": True, "arm": a.arm, "out": a.out}), flush=True)
    os._exit(0)


def drive(a):
    arms = a.arms.split(",")
    outs, meta = [], []
    main_base = tempfile.mkdtemp(prefix="idea65_mab_main_ws_")
    for idx, arm in enumerate(arms):
        out = os.path.join(tempfile.mkdtemp(prefix="idea65_mab_out_"), f"{idx}_{arm}.json")
        cmd = [sys.executable, os.path.abspath(__file__), "dump", "--arm", arm, "--ckpt", a.ckpt,
               "--out", out, "--passes", str(a.passes)]
        if arm == "main":
            cmd += ["--base", main_base]
        env = {k: v for k, v in os.environ.items() if k not in ENV_NAMES}
        log("SPAWN " + " ".join(cmd))
        t0 = time.time()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             env=env)
        for line in p.stdout:
            line = line.rstrip()
            if "nvcc" in line and line.startswith("["):
                continue
            print(f"[{arm}#{idx}] {line[:1500]}", flush=True)
        rc = p.wait()
        log(f"arm {arm}#{idx} rc {rc} wall {time.time() - t0:.0f}s")
        if rc != 0 or not os.path.exists(out):
            print(json.dumps({"passed": False, "error": f"arm {arm}#{idx} failed rc={rc}"}))
            return
        d = json.load(open(out))
        outs.append(d["res"])
        meta.append({"arm": arm, "read_rows_stats": d["read_rows_stats"], "env": d["env"]})
    res = {}
    for i, (arm, o) in enumerate(zip(arms, outs)):
        if len(o) > 1:
            res[f"{arm}#{i} pass0 vs pass1 (repeated execution)"] = cmp_runs(o["pass0"],
                                                                             o["pass1"])
    for i in range(1, len(arms)):
        res[f"{arms[i]}#{i} vs {arms[0]}#0 (pass0)"] = cmp_runs(outs[0]["pass0"],
                                                                outs[i]["pass0"])
    passed = all(v["bitwise_equal"] == v["prompts"] for v in res.values())
    rr = [m["read_rows_stats"] for m in meta]
    rr_ok = all(r.get("ran", 0) > 0 for r in rr) and len({json.dumps(r, sort_keys=True)
                                                          for r in rr}) == 1
    for k, v in res.items():
        log(f"{k}: {json.dumps(v)}")
    log(f"READ_ROWS_STATS per arm {json.dumps(meta)}")
    has_lp = bool(next(iter(outs[0]["pass0"].values()))[0]["lp"])
    print(json.dumps({"passed": bool(passed and has_lp and rr_ok),
                      "compared": "token ids (2 greedy tokens) + top-20 logprobs per token, "
                      f"bitwise, arms {a.arms}, one in-process LLM per arm, 33 chat-template "
                      "prompts: c=1 isl300x4/1kx4/2kx3/4kx3/8kx3 + 2x(4xisl8k) + 2x(4xisl1k) "
                      "groups sent together",
                      "env_both_arms": {k: "<unset: main default>" for k in ENV_NAMES},
                      "read_rows_stats_equal_and_ran": rr_ok, "logprobs_present": has_lp,
                      "results": res, "arms_meta": meta}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["drive", "dump"])
    ap.add_argument("--arms", default="main,tree,main")
    ap.add_argument("--arm", default="tree")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--base", default="")
    ap.add_argument("--passes", type=int, default=2)
    a = ap.parse_args()
    if a.mode == "dump":
        dump(a)
    else:
        drive(a)


if __name__ == "__main__":
    main()
