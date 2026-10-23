# Accuracy + greedy-token agreement proxy for the SM120 lm_head W4A4 rows (mode_switch).
# Arms (same branch build, LLM API in-process worker):
#   A: TRTLLM_LMHEAD_W4A4_MIN_ROWS=0  -> Marlin W4A16 lm_head at every row (= main's path)
#   B: default (W4A4 NVFP4 with the checkpoint's input_scale at rows >= 64, FP32 logits)
# Engine as the harness base: max_batch_size 128, CUDA graphs max_batch_size 128 (no padding),
# max_num_tokens 8192, chunked prefill, FP8 KV, block reuse with a 32768 snapshot.
# Items (reuse experiments/ascent/gdn_nvfp4/accuracy.py): 3 seeds x 32 long-context retrieval
# items (33.1K haystack), 16 short multi-token items, 4 x 256-token generations. One prime
# request per haystack, then all remaining 113 requests in one generate() so decode runs at
# 64-128 rows (W4A4 path in arm B). lm_head path hits are counted per arm.
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


def log(*a):
    print(*a, flush=True)


def heartbeat(tag, stop, period=30.0):
    while not stop.wait(period):
        log(f"[hb {tag} t={time.time() - T0:.0f}s]")


def lp(c):
    return [list(d.values())[0].logprob if isinstance(d, dict) else d for d in (c.logprobs or [])]


def run(arm, ckpt, out_path):
    if arm == "A":
        os.environ["TRTLLM_LMHEAD_W4A4_MIN_ROWS"] = "0"
    else:
        os.environ.pop("TRTLLM_LMHEAD_W4A4_MIN_ROWS", None)
    os.environ["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"
    stop = threading.Event()
    threading.Thread(target=heartbeat, args=(f"arm {arm}", stop), daemon=True).start()
    sys.path.insert(0, os.path.join(HERE, "..", "gdn_nvfp4"))
    import accuracy as acc  # noqa: E402
    import torch
    from transformers import AutoTokenizer

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.modules import embedding as emb
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

    counts = collections.Counter()
    orig_fwd = emb.LMHead.forward

    def fwd(self, input, *a, **kw):
        rows = input.reshape(-1, input.shape[-1]).shape[0]
        mr = self.__dict__.get("_w4a4_min_rows", 0)
        where = "capture" if torch.cuda.is_current_stream_capturing() else "eager"
        counts[f"{type(self.quant_method).__name__}/{'w4a4' if mr and rows >= mr else 'marlin'}"
               f"/{where}/rows{'>=64' if rows >= 64 else '<64'}"] += 1
        return orig_fwd(self, input, *a, **kw)

    emb.LMHead.forward = fwd
    tok = AutoTokenizer.from_pretrained(ckpt)
    t0 = time.time()
    llm = LLM(model=ckpt, max_seq_len=36864, max_batch_size=128, max_num_tokens=8192,
              enable_chunked_prefill=True,
              cuda_graph_config=CudaGraphConfig(max_batch_size=128),
              kv_cache_config=KvCacheConfig(dtype="fp8", enable_block_reuse=True,
                                            mamba_state_config={
                                                "additional_snapshot_offsets_from_start": [32768]}))
    log(f"[{arm}] LLM ready in {time.time() - t0:.1f}s; mem free/total "
        f"{[round(x / 1e9, 2) for x in torch.cuda.mem_get_info()]} GB; lm_head counts after "
        f"warmup/capture {dict(counts)}")
    results = {"mode": arm, "items": {}, "counts_after_warmup": dict(counts)}
    counts.clear()
    prompts, keys, expects, sps = [], [], [], []
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
    n_ok = sum(v["ok"] for k, v in results["items"].items() if not k.startswith("gen"))
    n = sum(1 for k in results["items"] if not k.startswith("gen"))
    log(f"[{arm}] correct {n_ok}/{n}; lm_head counts {dict(counts)}")
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
    res = {"A": "TRTLLM_LMHEAD_W4A4_MIN_ROWS=0 (Marlin W4A16 at all rows = main's path)",
           "B": "default (checkpoint NVFP4 W4A4 at rows >= 64, FP32 logits)",
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
               "max": max(lpd) if lpd else None},
           "lm_head_counts_A": a.get("counts_run"), "lm_head_counts_B": b.get("counts_run"),
           "warmup_counts_B": b.get("counts_after_warmup")}
    res["passed"] = len(lost) <= res["allowed_lost"] and any(
        "/w4a4/" in k for k in (b.get("counts_run") or {}))
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
            if line.startswith(f"[{arm}]") or line.startswith("[hb") or "lm_head" in line \
                    or "Traceback" in line or "Error" in line or "KV cache" in line.replace("kv cache", "KV cache"):
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
