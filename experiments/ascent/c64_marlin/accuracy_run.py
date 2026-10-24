# Accuracy + greedy-token agreement proxy for idea 33 (branch 26): attn qkv_proj (14336, 5120)
# runs its NVFP4 W4A16 Marlin decode copy at rows 33-64 (precision; main runs W4A4 Q+gate +
# FP8 K/V there). Arms (same branch build, LLM API in-process worker, no engine change):
#   A: TRTLLM_FP8_NVFP4_DECODE_QKV64=0 -> qkv Marlin cap 32 (= main's routing, bitwise per
#      check job 221)
#   B: default (qkv Marlin cap 64)
# Engine as the harness base except the batch limit: max_batch_size 64 and CUDA graphs up to 64,
# so decode runs at <= 64 rows the way c64 serving does (closed loop of 64). FP8 KV, chunked
# prefill, max_num_tokens 8192, block reuse with a 32768 snapshot.
# Items (experiments/ascent/gdn_nvfp4/accuracy.py): 3 seeds x 32 long-context retrieval items
# (33.1K haystack, 160 output tokens), 16 short multi-token items (48), 4 x 256-token
# generations. One prime request per haystack, then the other 113 in one generate().
# qkv route hits are counted per arm (apply calls at capture by row bucket, graph replays by
# batch size).
# usage: accuracy_run.py all <ckpt> <outdir> | run <arm> <ckpt> <out.json>
import collections
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
T0 = time.time()
SEEDS = (4401, 5501, 6601)
MAX_BS = 64


def log(*a):
    print(*a, flush=True)


def heartbeat(tag, stop, period=30.0):
    while not stop.wait(period):
        log(f"[hb {tag} t={time.time() - T0:.0f}s]")


def lp(c):
    return [list(d.values())[0].logprob if isinstance(d, dict) else d for d in (c.logprobs or [])]


def run(arm, ckpt, out_path):
    if arm == "A":
        os.environ["TRTLLM_FP8_NVFP4_DECODE_QKV64"] = "0"
    else:
        os.environ.pop("TRTLLM_FP8_NVFP4_DECODE_QKV64", None)
    os.environ["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"
    stop = threading.Event()
    threading.Thread(target=heartbeat, args=(f"arm {arm}", stop), daemon=True).start()
    sys.path.insert(0, os.path.join(HERE, "..", "gdn_nvfp4"))
    import accuracy as acc  # noqa: E402
    import torch
    from transformers import AutoTokenizer

    import tensorrt_llm
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.modules import linear as lin_mod
    from tensorrt_llm._torch.pyexecutor import cuda_graph_runner as cgr
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

    log(f"[{arm}] tensorrt_llm {tensorrt_llm.__file__} pid {os.getpid()}; marker "
        f"{getattr(lin_mod, '_FP8_NVFP4_DECODE_QKV64_ENV', None)} caps "
        f"{lin_mod._FP8_NVFP4_DECODE_ROW_CAPS}")
    counts = collections.Counter()
    replays = collections.Counter()
    orig_apply = lin_mod.FP8QDQLinearMethod.apply

    def apply(self, module, input, bias):
        if (getattr(module, "out_features", None), getattr(module, "in_features", None)) == (14336, 5120) \
                and isinstance(input, torch.Tensor):
            rows = input.reshape(-1, input.shape[-1]).shape[0]
            cap = lin_mod.fp8_nvfp4_decode_max_rows(module)
            where = "capture" if torch.cuda.is_current_stream_capturing() else "eager"
            b = "le32" if rows <= 32 else ("33-64" if rows <= 64 else "gt64")
            counts[f"qkv/{'marlin' if rows <= cap else 'w4a4'}/{where}/{b}"] += 1
        return orig_apply(self, module, input, bias)

    lin_mod.FP8QDQLinearMethod.apply = apply
    orig_replay = cgr.CUDAGraphRunner.replay

    def replay(self, key, current_inputs, **kw):
        try:
            bs = key[0] if isinstance(key, tuple) else key
            replays[str(int(bs))] += 1
        except Exception:  # noqa: BLE001
            replays["?" + str(key)[:40]] += 1
        return orig_replay(self, key, current_inputs, **kw)

    cgr.CUDAGraphRunner.replay = replay
    tok = AutoTokenizer.from_pretrained(ckpt)
    t0 = time.time()
    llm = LLM(model=ckpt, max_seq_len=36864, max_batch_size=MAX_BS, max_num_tokens=8192,
              enable_chunked_prefill=True,
              cuda_graph_config=CudaGraphConfig(max_batch_size=MAX_BS),
              kv_cache_config=KvCacheConfig(dtype="fp8", enable_block_reuse=True,
                                            mamba_state_config={
                                                "additional_snapshot_offsets_from_start": [32768]}))
    log(f"[{arm}] LLM ready in {time.time() - t0:.1f}s; mem free/total "
        f"{[round(x / 1e9, 2) for x in torch.cuda.mem_get_info()]} GB; qkv counts after "
        f"warmup/capture {dict(counts)}")
    results = {"mode": arm, "items": {}, "counts_after_warmup": dict(counts)}
    counts.clear()
    replays.clear()
    prompts = []
    sp_ret = SamplingParams(max_tokens=160, temperature=0.0, logprobs=1)
    sp_short = SamplingParams(max_tokens=48, temperature=0.0, logprobs=1)
    sp_gen = SamplingParams(max_tokens=256, temperature=0.0, ignore_eos=True, logprobs=1)
    primes = []
    for seed in SEEDS:
        text, names, codes = acc.build_haystack(tok, seed)
        for j, (nm, cd) in enumerate(zip(names, codes)):
            p = (text + f"\n\nBased on the document above, answer the question.\n"
                 f"Question: What is the access code for the {nm} vault?\n"
                 f"Answer: The access code for the {nm} vault is")
            item = (p, f"ret{seed}:{nm}", [cd], sp_ret)
            (primes if j == 0 else prompts).append(item)
    for i, (p, e) in enumerate(acc.SHORT):
        prompts.append((p, f"short{i}", e, sp_short))
    for i, p in enumerate(acc.LONG_GEN):
        prompts.append((p, f"gen{i}", [], sp_gen))
    t1 = time.time()
    outs_p = llm.generate([x[0] for x in primes], [x[3] for x in primes])
    log(f"[{arm}] primes done {time.time() - t1:.1f}s")
    t1 = time.time()
    outs = llm.generate([x[0] for x in prompts], [x[3] for x in prompts])
    log(f"[{arm}] main batch ({len(prompts)} requests) done {time.time() - t1:.1f}s")
    for (p, k, e, _), o in zip(primes + prompts, list(outs_p) + list(outs)):
        c = o.outputs[0]
        results["items"][k] = {"text": c.text, "ids": list(c.token_ids), "expect": e,
                               "ok": all(x in c.text for x in e) if e else True, "lp": lp(c)}
    results["counts_run"] = dict(counts)
    results["replays_run"] = dict(replays)
    results["replays_33_64"] = sum(v for k, v in replays.items() if k.isdigit() and 33 <= int(k) <= 64)
    n_ok = sum(v["ok"] for k, v in results["items"].items() if not k.startswith("gen"))
    n = sum(1 for k in results["items"] if not k.startswith("gen"))
    log(f"[{arm}] correct {n_ok}/{n}; qkv counts {dict(counts)}; replays {dict(replays)}")
    for k, v in results["items"].items():
        if not v["ok"]:
            log(f"[{arm}] WRONG {k}: expect {v['expect']} got {v['text'][-200:]!r}")
    with open(out_path, "w") as f:
        json.dump(results, f)
    stop.set()
    llm.shutdown()


def compare(a_path, b_path):
    a = json.load(open(a_path))
    b = json.load(open(b_path))
    ia, ib = a["items"], b["items"]
    scored = [k for k in ia if not k.startswith("gen")]
    lost = [k for k in scored if ia[k]["ok"] and not ib[k]["ok"]]
    gained = [k for k in scored if not ia[k]["ok"] and ib[k]["ok"]]
    same_seq, tok_total, tok_same_prefix, firstdiv, lpd = 0, 0, 0, [], []
    tok_agree_pos = 0
    for k in ia:
        x, y = ia[k]["ids"], ib[k]["ids"]
        d = next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), None)
        if d is None and len(x) != len(y):
            d = min(len(x), len(y))
        if d is None:
            same_seq += 1
            tok_same_prefix += len(x)
        else:
            firstdiv.append([k, d])
            tok_same_prefix += d
        tok_total += len(x)
        tok_agree_pos += sum(int(u == v) for u, v in zip(x, y))
        la, lb = ia[k]["lp"], ib[k]["lp"]
        upto = d if d is not None else min(len(la), len(lb))
        for i in range(min(upto, len(la), len(lb))):
            if isinstance(la[i], (int, float)) and isinstance(lb[i], (int, float)):
                lpd.append(abs(la[i] - lb[i]))
    n = len(scored)
    res = {"A": "TRTLLM_FP8_NVFP4_DECODE_QKV64=0 (qkv Marlin cap 32 = main's routing)",
           "B": "default (qkv Marlin W4A16 at rows <= 64)",
           "correct_A": f"{sum(ia[k]['ok'] for k in scored)}/{n}",
           "correct_B": f"{sum(ib[k]['ok'] for k in scored)}/{n}",
           "retrieval_A": sum(ia[k]["ok"] for k in scored if k.startswith("ret")),
           "retrieval_B": sum(ib[k]["ok"] for k in scored if k.startswith("ret")),
           "lost": lost, "gained": gained, "allowed_lost": max(1, -(-n * 5 // 100)),
           "identical_sequences": f"{same_seq}/{len(ia)}",
           "greedy_token_agreement_until_first_divergence": round(tok_same_prefix / max(1, tok_total), 4),
           "greedy_token_positional_agreement": round(tok_agree_pos / max(1, tok_total), 4),
           "tokens_A": tok_total, "first_divergence": firstdiv,
           "logprob_absdiff_before_divergence": {
               "n": len(lpd), "mean": (sum(lpd) / len(lpd)) if lpd else None,
               "max": max(lpd) if lpd else None, "tolerance": 11.957},
           "qkv_counts_A": a.get("counts_run"), "qkv_counts_B": b.get("counts_run"),
           "replays_33_64_A": a.get("replays_33_64"), "replays_33_64_B": b.get("replays_33_64"),
           "warmup_counts_A": a.get("counts_after_warmup"),
           "warmup_counts_B": b.get("counts_after_warmup")}
    wb = b.get("counts_after_warmup") or {}
    wa = a.get("counts_after_warmup") or {}
    res["B_captures_marlin_33_64"] = any(k.startswith("qkv/marlin/capture/33-64") for k in wb)
    res["A_captures_w4a4_33_64"] = any(k.startswith("qkv/w4a4/capture/33-64") for k in wa)
    lpmax = res["logprob_absdiff_before_divergence"]["max"]
    res["passed"] = (len(lost) <= res["allowed_lost"] and res["B_captures_marlin_33_64"]
                     and res["A_captures_w4a4_33_64"] and (b.get("replays_33_64") or 0) > 0
                     and (lpmax is None or lpmax <= 11.957))
    for k, d in firstdiv:
        log(f"diverge {k} at token {d}: A {ia[k]['text'][:100]!r} | B {ib[k]['text'][:100]!r}")
    return res


def run_all(ckpt, outdir):
    os.makedirs(outdir, exist_ok=True)
    stop = threading.Event()
    threading.Thread(target=heartbeat, args=("driver", stop), daemon=True).start()
    paths = {}
    for arm in ("A", "B"):
        out = os.path.join(outdir, f"{arm}.json")
        log(f"=== arm {arm} start t={time.time() - T0:.0f}s ===")
        p = subprocess.Popen([sys.executable, "-u", os.path.abspath(__file__), "run", arm, ckpt, out],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        tail = collections.deque(maxlen=40)
        for line in p.stdout:
            tail.append(line.rstrip()[:400])
            if line.startswith(f"[{arm}]") or line.startswith("[hb") or "nvfp4_decode_copy" in line \
                    or "Traceback" in line or "Error" in line:
                log(line.rstrip()[:600])
        rc = p.wait()
        log(f"=== arm {arm} rc={rc} t={time.time() - T0:.0f}s ===")
        if rc != 0 or not os.path.exists(out):
            for l in tail:
                log("  | " + l)
            stop.set()
            print(json.dumps({"passed": False, "error": f"arm {arm} failed rc={rc}"}))
            return
        paths[arm] = out
    res = compare(paths["A"], paths["B"])
    stop.set()
    print(json.dumps(dict(res, elapsed_s=round(time.time() - T0))))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3], sys.argv[4])
    elif sys.argv[1] == "all":
        run_all(sys.argv[2], sys.argv[3])
    else:
        print(json.dumps(compare(sys.argv[2], sys.argv[3])))
