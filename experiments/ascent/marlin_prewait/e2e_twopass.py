# Two-pass end-to-end exactness predictor for branch 21 (idea 28), one arm per process (NOT the frozen check).
# BRACE-FREE on purpose: run_gpu_pair passes it verbatim via python -c to both trees (the base tree lacks this file).
# Protocol shapes: 56 prompts = one 32768-token shared prefix (real text) + 2048 unique suffix tokens each;
# groups c8 (prompts 0-7 in one call of 8), c16 (8-23, one call of 16), c32 (24-55, one call of 32);
# 40 greedy tokens with logprobs=1; max_batch_size 32, decode CUDA graphs up to 32, max_num_tokens 8192, chunked
# prefill, experiments/ascent-config/engine.json kv_cache_config (FP8 KV, block reuse, snapshot at 32768).
# pass 1 starts from an empty cache; pass 2 reruns all 56 prompts reusing the cached prefix and snapshot.
# Compare per-prompt ids/logprob digests across the pair arms (exact change: equal in pass 1 and in pass 2).
# Usage: python e2e_twopass.py <checkpoint>
#
# Idea 67 pinned mode (knowledge 124; diagnosis only, the pin never goes in engine.json):
#   base arm : python -c <this> <ckpt> tune <nonce>     tunes into its own fresh cache, publishes it as the master
#   cand arm : python e2e_twopass.py <ckpt> cand <nonce> run 1 pinned (own copy of the master), run 2 unpinned;
#              compares both with the base arm per prompt (both passes) and names AutoTuner keys whose pick differs.
import glob
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time

os.environ.setdefault("TLLM_WORKER_USE_SINGLE_PROCESS", "1")

PREFIX, SUFFIX = 32768, 2048
OSL = int(os.environ.get("E2E_OSL", "40"))
BATCH_WAIT_MS = float(os.environ.get("E2E_BATCH_WAIT_MS", "200"))
GROUPS = (("c8", 8), ("c16", 16), ("c32", 32))


def log(*a):
    print("[%s]" % time.strftime("%H:%M:%S"), *a, flush=True)


def heartbeat():
    t0 = time.time()

    def loop():
        while True:
            time.sleep(30)
            log("[heartbeat] %.0fs" % (time.time() - t0))

    threading.Thread(target=loop, daemon=True).start()


def corpus_ids(tok, n_tokens):
    files = sorted(glob.glob("docs/source/**/*.md", recursive=True))
    files += sorted(glob.glob("tensorrt_llm/_torch/**/*.md", recursive=True))
    files += sorted(glob.glob("examples/**/*.md", recursive=True))
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


def digest_ids(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def digest_lp(lps):
    return hashlib.sha256(b"".join(struct.pack("<d", float("nan") if x is None else x) for x in lps)).hexdigest()


def pin_dirs(ckpt, nonce):
    root = os.path.dirname(os.path.dirname(os.path.abspath(ckpt)))
    return [os.path.join(root, "ascent_pin_b37", nonce), os.path.join("/tmp", "ascent_pin_b37", nonce)]


def picks(path):
    if not path or not os.path.exists(path):
        return dict()
    d = json.load(open(path))
    out = dict()
    for sect, ent in d.items():
        if sect == "metadata":
            continue
        for k, v in ent.items():
            out[sect + "|" + k] = "%s/%s" % (v.get("runner_id"), v.get("tactic"))
    return out


def run_engine(ckpt, cache_path, tag):
    os.environ["TLLM_AUTOTUNER_CACHE_PATH"] = cache_path
    import torch

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.pyexecutor import cuda_graph_runner as cgr
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

    captures = set()
    orig_capture = cgr.CUDAGraphRunner.capture

    def capture(self, key, *a, **kw):
        captures.add(cgr.KeyType(*key).batch_size)
        return orig_capture(self, key, *a, **kw)

    cgr.CUDAGraphRunner.capture = capture
    engine_json = json.load(open("experiments/ascent-config/engine.json"))
    pre = picks(cache_path)
    log(tag, "cache", cache_path, "preloaded picks", len(pre), "PDL", os.environ.get("TRTLLM_ENABLE_PDL"),
        "PREWAIT", os.environ.get("TRTLLM_MARLIN_PREWAIT_B"), "NORM_EARLY", os.environ.get("TRTLLM_NORM_EARLY_TRIGGER"))
    kvc = dict(engine_json.get("kv_cache_config", dict()))
    assert kvc.get("dtype") == "fp8", kvc
    t = time.time()
    llm = LLM(model=ckpt, max_seq_len=36864, max_batch_size=32, max_num_tokens=8192, enable_chunked_prefill=True,
              cuda_graph_config=CudaGraphConfig(max_batch_size=32), kv_cache_config=KvCacheConfig(**kvc),
              batch_wait_timeout_ms=BATCH_WAIT_MS)
    log(tag, "LLM ready in %.1fs; graphs at %s" % (time.time() - t, sorted(captures)))
    post_init = picks(cache_path)
    tok = llm.tokenizer
    n = sum(c for _, c in GROUPS)
    ids = corpus_ids(tok, PREFIX + n * SUFFIX)
    prefix = ids[:PREFIX]
    prompts = [prefix + ids[PREFIX + i * SUFFIX:PREFIX + (i + 1) * SUFFIX] for i in range(n)]
    sp = SamplingParams(max_tokens=OSL, temperature=0.0, top_k=1, ignore_eos=True, logprobs=1)

    def run_pass(ptag):
        res = dict()
        base = 0
        for g, conc in GROUPS:
            idx = list(range(base, base + conc))
            base += conc
            t1 = time.time()
            outs = llm.generate([prompts[i] for i in idx], sp, use_tqdm=False)
            for j, o in enumerate(outs):
                c = o.outputs[0]
                lps = []
                for tid, d in zip(c.token_ids, c.logprobs or []):
                    lp = d.get(tid) if isinstance(d, dict) else None
                    lps.append(float(lp.logprob) if lp is not None else None)
                res["%s-p%d" % (g, j)] = dict(ids=list(c.token_ids), lp=lps)
            log(tag, ptag, g, "%d prompts %.1fs" % (conc, time.time() - t1))
        return res

    p1 = run_pass("pass1")
    p2 = run_pass("pass2")
    torch.cuda.synchronize()
    p2eq = sum(int(p1[k] == p2[k]) for k in p1)
    out = dict(tag=tag, pass1=p1, pass2=p2, pass2_equals_pass1=p2eq, n=len(p1), graphs=sorted(captures),
               prompt_ids_sha=digest_ids(prompts)[:16], picks_pre=len(pre), picks_post_init=post_init,
               new_keys_vs_pre=sorted(k for k in post_init if k not in pre),
               changed_vs_pre=sorted(k for k in post_init if k in pre and pre[k] != post_init[k]))
    log(tag, "done; pass2==pass1 %d/%d; new keys vs preload %d, changed %d" % (
        p2eq, len(p1), len(out["new_keys_vs_pre"]), len(out["changed_vs_pre"])))
    try:
        llm.shutdown()
    except Exception as e:
        log("shutdown error", repr(e))
    return out


def compare(a, b):
    d1 = sorted(k for k in a["pass1"] if a["pass1"][k] != b["pass1"].get(k))
    d2 = sorted(k for k in a["pass2"] if a["pass2"][k] != b["pass2"].get(k))
    first = dict()
    for k in sorted(set(d1) | set(d2)):
        rec = dict()
        for p in ("pass1", "pass2"):
            x, y = a[p][k], b[p][k]
            rec[p + "_first_token_div"] = next((i for i, (u, v) in enumerate(zip(x["ids"], y["ids"])) if u != v), None)
            rec[p + "_first_lp_div"] = next((i for i, (u, v) in enumerate(zip(x["lp"], y["lp"])) if u != v), None)
            li = rec[p + "_first_lp_div"]
            if li is not None:
                rec[p + "_lp_at_div"] = [x["lp"][li], y["lp"][li]]
        first[k] = rec
    return dict(pass1_equal=len(a["pass1"]) - len(d1), pass2_equal=len(a["pass2"]) - len(d2), n=len(a["pass1"]),
                pass1_diff=d1, pass2_diff=d2, first_div=first)


def pinned_main(ckpt, mode, nonce):
    heartbeat()
    art = os.environ.get("AUTO_DEV_ARTIFACTS") or tempfile.mkdtemp()
    os.makedirs(art, exist_ok=True)
    dirs = pin_dirs(ckpt, nonce)
    if mode == "tune":
        own = os.path.join(tempfile.mkdtemp(prefix="at_base_"), "cache.json")
        r = run_engine(ckpt, own, "base_tuned")
        published = []
        for d in dirs:
            try:
                os.makedirs(d, exist_ok=True)
                shutil.copy(own, os.path.join(d, "master.json"))
                with open(os.path.join(d, "base_result.json"), "w") as f:
                    json.dump(r, f)
                published.append(d)
            except Exception as e:
                log("publish failed", d, repr(e))
        shutil.copy(own, os.path.join(art, "master_base.json"))
        with open(os.path.join(art, "base_result.json"), "w") as f:
            json.dump(r, f)
        print(json.dumps(dict(mode="tune", published=published, n_picks=len(r["picks_post_init"]),
                              pass2_equals_pass1=r["pass2_equals_pass1"], n=r["n"], graphs=r["graphs"],
                              prompt_ids_sha=r["prompt_ids_sha"])), flush=True)
        os._exit(0)
    src = None
    for d in dirs:
        if os.path.exists(os.path.join(d, "master.json")) and os.path.exists(os.path.join(d, "base_result.json")):
            src = d
            break
    if src is None:
        print(json.dumps(dict(passed=False, error="no master published by the base arm", looked=dirs)), flush=True)
        os._exit(0)
    log("master from", src)
    master = os.path.join(src, "master.json")
    shutil.copy(master, os.path.join(art, "master_from_base.json"))
    base = json.load(open(os.path.join(src, "base_result.json")))
    results = dict()
    for tag in ("cand_pinned", "cand_unpinned"):
        own = os.path.join(tempfile.mkdtemp(prefix="at_%s_" % tag), "cache.json")
        if tag == "cand_pinned":
            shutil.copy(master, own)
        outp = os.path.join(art, tag + "_result.json")
        t = time.time()
        rc = subprocess.run([sys.executable, os.path.abspath(__file__), ckpt, "child", nonce, own, tag,
                             outp]).returncode
        log(tag, "exit", rc, "in %.0fs" % (time.time() - t))
        if rc != 0 or not os.path.exists(outp):
            print(json.dumps(dict(passed=False, error="%s failed rc=%s" % (tag, rc))), flush=True)
            os._exit(0)
        shutil.copy(own, os.path.join(art, tag + "_cache.json"))
        results[tag] = json.load(open(outp))
    mp = picks(master)
    up = results["cand_unpinned"]["picks_post_init"]
    flipped = dict((k, dict(master=mp.get(k), cand_unpinned=up.get(k))) for k in sorted(set(mp) | set(up))
                   if mp.get(k) != up.get(k))
    pin = results["cand_pinned"]
    unp = results["cand_unpinned"]
    cmp_pin = compare(base, pin)
    cmp_unp = compare(base, unp)
    exact_pinned = cmp_pin["pass1_equal"] == cmp_pin["n"] and cmp_pin["pass2_equal"] == cmp_pin["n"]
    res = dict(passed=exact_pinned, base_vs_cand_pinned=cmp_pin, base_vs_cand_unpinned=cmp_unp,
               pinned_new_keys=pin["new_keys_vs_pre"], pinned_changed_keys=pin["changed_vs_pre"],
               flipped_keys_master_vs_unpinned=flipped, n_master_picks=len(mp),
               pass2_equals_pass1=dict(base=base["pass2_equals_pass1"], cand_pinned=pin["pass2_equals_pass1"],
                                       cand_unpinned=unp["pass2_equals_pass1"]),
               prompt_ids_sha=dict(base=base["prompt_ids_sha"], pinned=pin["prompt_ids_sha"],
                                   unpinned=unp["prompt_ids_sha"]),
               graphs=dict(base=base["graphs"], cand_pinned=pin["graphs"], cand_unpinned=unp["graphs"]),
               master_dir=src,
               compared="base tree (tuned; publishes master) vs candidate pinned to a per-process copy of that "
                        "master, and vs candidate unpinned; 56 prompts x 40 greedy tokens + logprobs, both passes")
    print(json.dumps(res), flush=True)
    os._exit(0)


def child_main(ckpt, cache_path, tag, out_path):
    heartbeat()
    r = run_engine(ckpt, cache_path, tag)
    with open(out_path, "w") as f:
        json.dump(r, f)
    os._exit(0)


def main(ckpt):
    heartbeat()
    import torch

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.pyexecutor import cuda_graph_runner as cgr
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

    replays, captures = dict(), set()
    cur = dict(phase="load")
    orig_replay, orig_capture = cgr.CUDAGraphRunner.replay, cgr.CUDAGraphRunner.capture

    def replay(self, key, *a, **kw):
        k = "%s:bs%d" % (cur["phase"], cgr.KeyType(*key).batch_size)
        replays[k] = replays.get(k, 0) + 1
        return orig_replay(self, key, *a, **kw)

    def capture(self, key, *a, **kw):
        captures.add(cgr.KeyType(*key).batch_size)
        return orig_capture(self, key, *a, **kw)

    cgr.CUDAGraphRunner.replay, cgr.CUDAGraphRunner.capture = replay, capture

    engine_json = json.load(open("experiments/ascent-config/engine.json"))
    log("engine.json:", json.dumps(engine_json))
    log("TRTLLM_ENABLE_PDL=%s TRTLLM_MARLIN_PREWAIT_B=%s batch_wait_timeout_ms=%s osl=%d" % (
        os.environ.get("TRTLLM_ENABLE_PDL"), os.environ.get("TRTLLM_MARLIN_PREWAIT_B"), BATCH_WAIT_MS, OSL))
    kvc = dict(engine_json.get("kv_cache_config", dict()))
    assert kvc.get("dtype") == "fp8", kvc
    t0 = time.time()
    llm = LLM(model=ckpt, max_seq_len=36864, max_batch_size=32, max_num_tokens=8192, enable_chunked_prefill=True,
              cuda_graph_config=CudaGraphConfig(max_batch_size=32), kv_cache_config=KvCacheConfig(**kvc),
              batch_wait_timeout_ms=BATCH_WAIT_MS)
    log("LLM ready in %.1fs; decode graphs captured at batch sizes %s" % (time.time() - t0, sorted(captures)))
    tok = llm.tokenizer
    n = sum(c for _, c in GROUPS)
    ids = corpus_ids(tok, PREFIX + n * SUFFIX)
    prefix = ids[:PREFIX]
    prompts = [prefix + ids[PREFIX + i * SUFFIX:PREFIX + (i + 1) * SUFFIX] for i in range(n)]
    sp = SamplingParams(max_tokens=OSL, temperature=0.0, top_k=1, ignore_eos=True, logprobs=1)

    def run_pass(tag):
        res = dict()
        base = 0
        for g, conc in GROUPS:
            cur["phase"] = "%s:%s" % (tag, g)
            idx = list(range(base, base + conc))
            t = time.time()
            outs = llm.generate([prompts[i] for i in idx], sp, use_tqdm=False)
            dt = time.time() - t
            names = []
            for j, o in enumerate(outs):
                c = o.outputs[0]
                lps = []
                for tid, d in zip(c.token_ids, c.logprobs or []):
                    lp = d.get(tid) if isinstance(d, dict) else None
                    lps.append(float(lp.logprob) if lp is not None else None)
                name = "%s-p%d" % (g, j)
                names.append(name)
                res[name] = dict(ids=list(c.token_ids), lp=lps, cached=getattr(o, "cached_tokens", None))
            log("%s %s: %d prompts %.2fs" % (tag, g, conc, dt))
            base += conc
        return res

    p1 = run_pass("pass1")
    p2 = run_pass("pass2")
    torch.cuda.synchronize()
    per = dict()
    p2_eq_p1 = 0
    for k in p1:
        same = p1[k]["ids"] == p2[k]["ids"] and p1[k]["lp"] == p2[k]["lp"]
        p2_eq_p1 += int(same)
        per[k] = dict(p1_ids_sha=digest_ids(p1[k]["ids"])[:16], p1_lp_sha=digest_lp(p1[k]["lp"])[:16],
                      p2_ids_sha=digest_ids(p2[k]["ids"])[:16], p2_lp_sha=digest_lp(p2[k]["lp"])[:16],
                      n_tokens=len(p1[k]["ids"]), pass2_equals_pass1=same)
    art = os.environ.get("AUTO_DEV_ARTIFACTS")
    if art:
        os.makedirs(art, exist_ok=True)
        with open(os.path.join(art, "e2e_twopass_tokens.json"), "w") as f:
            json.dump(dict(pass1=p1, pass2=p2), f)
    dec_bs = sorted(set(int(k.split(":bs")[1]) for k in replays if k.startswith("pass")))
    res = dict(pass2_equals_pass1_prompts=p2_eq_p1, n_prompts=len(p1),
               graphs_captured_batch_sizes=sorted(captures), graph_replays=replays,
               decode_graph_batch_sizes_replayed=dec_bs,
               pass1_all_ids_sha=digest_ids([p1[k]["ids"] for k in sorted(p1)]),
               pass1_all_lp_sha=hashlib.sha256("".join(per[k]["p1_lp_sha"] for k in sorted(per)).encode()).hexdigest(),
               pass2_all_ids_sha=digest_ids([p2[k]["ids"] for k in sorted(p2)]),
               pass2_all_lp_sha=hashlib.sha256("".join(per[k]["p2_lp_sha"] for k in sorted(per)).encode()).hexdigest(),
               prompt_ids_sha=digest_ids(prompts)[:16],
               engine_json=engine_json, batch_wait_timeout_ms=BATCH_WAIT_MS, osl=OSL,
               prompts="%d x (%d shared prefix + %d unique), %d greedy tokens, logprobs=1; c8@8, c16@16, c32@32" % (
                   n, PREFIX, SUFFIX, OSL), per_prompt=per,
               compared="compare pass1/pass2 digests and per_prompt across the pair arms")
    print(json.dumps(res), flush=True)
    llm.shutdown()
    os._exit(0)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "child":
        child_main(sys.argv[1], sys.argv[4], sys.argv[5], sys.argv[6])
    elif len(sys.argv) > 2:
        pinned_main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1])
