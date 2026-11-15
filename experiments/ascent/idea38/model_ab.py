#!/usr/bin/env python3
"""Idea 38 full-model numerics gate: in-process LLM A/B of the FlashInfer prefill header.

  driver:  model_ab.py drive --arms pin,tree      (each arm its own process, then compare)
  worker:  model_ab.py dump --arm <arm> --out f   (set FlashInfer env BEFORE importing it)

Arms: 'tree' = the job's FlashInfer tree as is; 'pin' / 'v2m' / ... = overlay of patch.py on a
copy of the include tree with its own fresh FLASHINFER_WORKSPACE_BASE (so no .so is shared).
Engine: TLLM_WORKER_USE_SINGLE_PROCESS=1, FP8 KV, block reuse off, max_num_tokens 8192,
max_batch_size 64, batch_wait_timeout_ms 1500 (repeatable batch composition, as fi_plan_ab.py).
Prompts (fresh random tokens, fixed seed): isl300@1 x12, isl1k@4 x3 groups, isl8k@1 x8,
isl8k@4 x2 groups, plus one mixed batch (10 x isl1k + 4 x isl8k) = 54 prompts.
Each arm runs the whole set twice (A/A inside the process). Per prompt: generated token id and
top-20 logprobs, compared bitwise across arms; max |d logprob| reported.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def prompt_sets(tok, seed=4242):
    """ISL {300,1k,2k,4k,8k} x c {1,4}, 10 fresh chat-template prompts each = 100 prompts.
    c=1: 10 single requests; c=4: groups of 4,4,2 submitted together."""
    import numpy as np
    rng = np.random.default_rng(seed)

    def tmpl(text):
        out = tok.apply_chat_template([{"role": "user", "content": text}],
                                      add_generation_prompt=True, tokenize=True)
        if hasattr(out, "keys"):  # BatchEncoding / dict
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
    for isl, nm in ((300, "isl300"), (1000, "isl1k"), (2000, "isl2k"), (4000, "isl4k"),
                    (8000, "isl8k")):
        for i in range(10):
            sets.append((f"{nm}@1#{i}", [mk(isl)]))
        ps = [mk(isl) for _ in range(10)]
        for g, (s, e) in enumerate(((0, 4), (4, 8), (8, 10))):
            sets.append((f"{nm}@4#g{g}", ps[s:e]))
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
    return {"ids": list(c.token_ids), "lp": lps}


def dump(a):
    import patch
    if a.arm != "tree":
        tmp = tempfile.mkdtemp(prefix=f"idea38_mab_{a.arm}_inc_")
        inc, tv = patch.make_overlay(a.arm, tmp)
        os.environ["FLASHINFER_EXTRA_CUDAFLAGS"] = f"-I{inc}"
        log(f"overlay {a.arm} over tree variant {tv}: {inc}")
    else:
        with open(os.path.join(patch.include_dir(), "flashinfer", "attention", "prefill.cuh")) as f:
            log(f"tree prefill.cuh variant: {patch.tree_variant(f.read())}")
    os.environ["FLASHINFER_WORKSPACE_BASE"] = tempfile.mkdtemp(prefix=f"idea38_mab_{a.arm}_ws_")
    os.environ.setdefault("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    log(f"FLASHINFER_WORKSPACE_BASE={os.environ['FLASHINFER_WORKSPACE_BASE']} "
        f"EXTRA={os.environ.get('FLASHINFER_EXTRA_CUDAFLAGS', '<none>')} load={os.getloadavg()}")
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
              batch_wait_timeout_ms=1500.0,
              **({"enable_autotuner": False} if os.environ.get("MAB_AUTOTUNER") == "0" else {}))
    log(f"autotuner {'OFF' if os.environ.get('MAB_AUTOTUNER') == '0' else 'default (on)'}")
    params = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True, logprobs=20)
    import numpy as np
    wrng = np.random.default_rng(5)
    for isl in (300, 1000, 8000):
        llm.generate([[int(x) for x in wrng.integers(1000, 200000, size=isl)] for _ in range(2)],
                     sampling_params=params, use_tqdm=False)
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
    try:
        import tensorrt_llm._torch.attention.backends.flashinfer  # noqa: F401
        from flashinfer.jit import env as jit_env
        jd = jit_env.FLASHINFER_JIT_DIR
        log(f"JIT modules built here: {sorted(os.listdir(jd)) if os.path.isdir(jd) else []}")
    except Exception as e:  # noqa: BLE001
        log(f"jit listing failed: {e}")
    stop.set()
    with open(a.out, "w") as f:
        json.dump(res, f)
    log(f"wrote {a.out} load={os.getloadavg()}")
    print(json.dumps({"passed": True, "arm": a.arm, "out": a.out}), flush=True)
    os._exit(0)


def drive(a):
    res1 = drive_once(a, autotuner=True)
    if res1 is None:
        return
    passed, res, has_lp = res1
    aa = next((v for k, v in res.items() if k.startswith("A_vs_A'")), None)
    out = {"autotuner_on": {"passed": passed, "results": res}}
    if aa is not None and aa["bitwise_equal"] != aa["prompts"]:
        log("A vs A' not bitwise with the autotuner on (cross-process tactic picks?): "
            "repeating all arms with enable_autotuner=False")
        res2 = drive_once(a, autotuner=False)
        if res2 is None:
            return
        passed, res, has_lp = res2
        out["autotuner_off"] = {"passed": passed, "results": res}
    print(json.dumps({"passed": bool(passed and has_lp),
                      "compared": "first-token ids + top-20 logprobs, bitwise, arms "
                      f"{a.arms} (A,B,A'), one in-process LLM per arm, 100 chat-template prompts "
                      "ISL 300/1k/2k/4k/8k x c1/c4", "logprobs_present": has_lp, **out}),
          flush=True)


def drive_once(a, autotuner):
    from model_ab_cmp import report
    arms = a.arms.split(",")
    outs = []
    if not autotuner:
        os.environ["MAB_AUTOTUNER"] = "0"
    for idx, arm in enumerate(arms):
        out = os.path.join(tempfile.mkdtemp(prefix="idea38_mab_out_"), f"{idx}_{arm}.json")
        cmd = [sys.executable, os.path.abspath(__file__), "dump", "--arm", arm, "--ckpt", a.ckpt,
               "--out", out]
        log("SPAWN " + " ".join(cmd))
        t0 = time.time()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            print(f"[{arm}] {line.rstrip()}", flush=True)
        rc = p.wait()
        log(f"arm {arm} rc {rc} wall {time.time() - t0:.0f}s")
        if rc != 0 or not os.path.exists(out):
            print(json.dumps({"passed": False, "error": f"arm {arm} failed rc={rc}"}))
            return None
        outs.append(json.load(open(out))["pass0"])
    passed, res = report(outs, arms)
    for k, v in res.items():
        log(f"[autotuner {'on' if autotuner else 'off'}] {k}: {json.dumps(v)}")
    has_lp = bool(next(iter(outs[0].values()))[0]["lp"])
    return passed, res, has_lp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["drive", "dump"])
    ap.add_argument("--arms", default="pin,tree,pin")
    ap.add_argument("--arm", default="tree")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--passes", type=int, default=1)
    a = ap.parse_args()
    if a.mode == "dump":
        dump(a)
    else:
        drive(a)


if __name__ == "__main__":
    main()
