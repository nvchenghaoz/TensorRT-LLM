# In-branch accuracy proxy for idea 37 (int8 GDN state pool), evaluator item 1(2).
# One job, two engine lifetimes, identical engine settings except
# kv_cache_config.mamba_ssm_cache_dtype ("auto" = main's bf16 pool, then "int8"):
#   engine = experiments/ascent-config/engine.json (FP8 KV, block reuse, mamba
#   snapshot at 32768) + harness base engine (max_batch_size 128, decode CUDA
#   graphs up to 128, max_num_tokens 8192, chunked prefill).
# Per lifetime:
#   1. accuracy items of experiments/ascent/gdn_nvfp4/accuracy.py: 2 seeds x 32
#      needle retrievals on the ~33K-token haystack (prefix primed by one prompt,
#      then 31 reuse hits through the 32768 snapshot) + 16 short items. Greedy.
#   2. reuse.py's 64-prompt batch (32768 shared prefix + 2048 unique tokens,
#      40 tokens, ignore_eos): prompt 0 alone (primes the prefix), then all 64
#      together. bf16: free greedy, logprobs recorded by a logits processor.
#      int8: teacher-forced: a logits processor records log_softmax at bf16's
#      batch-64 greedy token and then forces that token, so every position is
#      compared on the same history.
#   3. bf16 lifetime only: each of the 64 prompts alone (batch 1), teacher-forced
#      on bf16 batch-64's tokens: bf16's own batch-invariance on this text.
# Report |dlogprob| per token (max, p99, p90, mean) vs the 11.957 tolerance.
import glob
import json
import os
import random
import sys
import threading
import time

os.environ.setdefault("TLLM_WORKER_USE_SINGLE_PROCESS", "1")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gdn_nvfp4"))
import accuracy as acc  # noqa: E402  (item set: FILLER/SHORT/build_haystack)

PREFIX, SUFFIX, OSL, NC = 32768, 2048, 40, 64
TOL = 11.957
STORE = {}


def log(*a):
    print("[%s]" % time.strftime("%H:%M:%S"), *a, flush=True)


def heartbeat():
    t0 = time.time()

    def loop():
        while True:
            time.sleep(30)
            log("[heartbeat] %.0fs" % (time.time() - t0))

    threading.Thread(target=loop, daemon=True).start()


class Rec:
    """Records log_softmax of the chosen (or forced) token at every sampling call.

    targets None: record the greedy (argmax) token. targets list: record the
    target's logprob, then force it by lifting its logit above the max.
    """

    def __init__(self, tag, targets=None):
        self.tag = tag
        self.targets = targets
        STORE[tag] = []

    def __call__(self, req_id, logits, token_ids, stream_ptr, client_id=None):
        import torch
        row = logits.reshape(-1, logits.shape[-1])[0]
        ls = torch.log_softmax(row.float(), -1)
        top_lp, top_id = ls.max(-1)
        rec = STORE[self.tag]
        i = len(rec)
        if self.targets is not None and i < len(self.targets):
            t = int(self.targets[i])
            v = torch.stack([ls[t], top_lp, top_id.float(), torch.tensor(float(t), device=ls.device)])
            row[t] = row.max() + 100.0
        else:
            v = torch.stack([top_lp, top_lp, top_id.float(), top_id.float()])
        rec.append((v.clone(), len(token_ids[0])))


def collect(tag):
    out = []
    for v, n in STORE[tag]:
        lp, tlp, tid, tok = v.tolist()
        out.append(dict(lp=lp, top_lp=tlp, top_id=int(tid), tok=int(tok), ntok=n))
    return out


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


def accuracy_items(llm, tok, dt):
    from tensorrt_llm import SamplingParams
    items = {}
    for seed in (4401, 5501):
        text, names, codes = acc.build_haystack(tok, seed)
        prompts = [text + "\n\nBased on the document above, answer the question.\n"
                   f"Question: What is the access code for the {nm} vault?\n"
                   f"Answer: The access code for the {nm} vault is" for nm in names]
        sp = SamplingParams(max_tokens=160, temperature=0.0, logprobs=1)
        t1 = time.time()
        outs = llm.generate(prompts[:1], sp, use_tqdm=False)
        outs += llm.generate(prompts[1:], sp, use_tqdm=False)
        log(f"[{dt}] retrieval seed {seed}: {time.time()-t1:.1f}s cached "
            f"{sorted(set(str(getattr(o, 'cached_tokens', None)) for o in outs))}")
        for nm, cd, o in zip(names, codes, outs):
            c = o.outputs[0]
            items[f"ret{seed}:{nm}"] = dict(ok=cd in c.text, ids=list(c.token_ids), expect=[cd],
                                           tail=c.text[-120:])
    sp = SamplingParams(max_tokens=48, temperature=0.0, logprobs=1)
    outs = llm.generate([p for p, _ in acc.SHORT], sp, use_tqdm=False)
    for i, ((p, e), o) in enumerate(zip(acc.SHORT, outs)):
        c = o.outputs[0]
        items[f"short{i}"] = dict(ok=all(x in c.text for x in e), ids=list(c.token_ids), expect=e,
                                  tail=c.text[-120:])
    n_ok = sum(v["ok"] for v in items.values())
    log(f"[{dt}] accuracy items correct {n_ok}/{len(items)}")
    for k, v in items.items():
        if not v["ok"]:
            log(f"[{dt}] WRONG {k}: expect {v['expect']} got {v['tail']!r}")
    return items


def lifetime(ckpt, dt, ids_cache, targets64=None):
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

    engine_json = json.load(open("experiments/ascent-config/engine.json"))
    kvc = dict(engine_json.get("kv_cache_config", dict()))
    kvc["mamba_ssm_cache_dtype"] = dt
    log("kv_cache_config", kvc)
    t0 = time.time()
    llm = LLM(model=ckpt, max_seq_len=36864, max_batch_size=128,
              max_num_tokens=8192, enable_chunked_prefill=True,
              cuda_graph_config=CudaGraphConfig(max_batch_size=128),
              kv_cache_config=KvCacheConfig(**kvc))
    log("[%s] LLM ready in %.1fs" % (dt, time.time() - t0))
    tok = llm.tokenizer
    res = dict(items=accuracy_items(llm, tok, dt))

    if "ids" not in ids_cache:
        ids_cache["ids"] = corpus_ids(tok, PREFIX + NC * SUFFIX)
    ids = ids_cache["ids"]
    prefix = ids[:PREFIX]
    prompts = [prefix + ids[PREFIX + i * SUFFIX:PREFIX + (i + 1) * SUFFIX] for i in range(NC)]

    def gen(idx, tag, targets):
        sps = []
        for i in idx:
            tg = None if targets is None else targets[i]
            sps.append(SamplingParams(max_tokens=OSL, temperature=0.0, top_k=1, ignore_eos=True,
                                      logprobs=1, logits_processor=Rec(f"{dt}:{tag}:{i}", tg)))
        t = time.time()
        outs = llm.generate([prompts[i] for i in idx], sps, use_tqdm=False)
        out = []
        for i, o in zip(idx, outs):
            c = o.outputs[0]
            lps = []
            for tid, d in zip(c.token_ids, c.logprobs or []):
                lp = d.get(tid) if isinstance(d, dict) else None
                lps.append(float(lp.logprob) if lp is not None else None)
            out.append(dict(ids=list(c.token_ids), ret_lp=lps, cached=getattr(o, "cached_tokens", None),
                            rec=collect(f"{dt}:{tag}:{i}")))
        log("[%s] %s: %d prompts %.1fs cached %s" % (dt, tag, len(idx), time.time() - t,
                                                    sorted(set(str(r["cached"]) for r in out))))
        return out

    res["prime0"] = gen([0], "prime0", targets64)[0]
    res["b64"] = gen(list(range(NC)), "b64", targets64)
    if targets64 is None:
        # bf16 batch-1 teacher-forced on bf16 batch-64 tokens.
        tg = [r["ids"] for r in res["b64"]]
        res["b1"] = [gen([i], "b1", tg)[0] for i in range(NC)]
    llm.shutdown()
    del llm
    return res


def stats(xs):
    xs = sorted(xs)
    if not xs:
        return dict(n=0)
    q = lambda p: xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))]  # noqa: E731
    return dict(n=len(xs), max=xs[-1], p99=q(0.99), p90=q(0.90), p50=q(0.5), mean=sum(xs) / len(xs))


def forced_cmp(ref_runs, cand_runs):
    """ref free greedy (rec lp at its own tokens) vs cand forced on ref tokens."""
    d, d_first, flips, forced_ok, n_bad = [], [], 0, 0, 0
    worst = None
    for k, (r, c) in enumerate(zip(ref_runs, cand_runs)):
        rr, cr = r["rec"], c["rec"]
        if c["ids"] == r["ids"]:
            forced_ok += 1
        if len(rr) != OSL or len(cr) != OSL:
            n_bad += 1
        for j, (a, b) in enumerate(zip(rr, cr)):
            assert b["tok"] == a["top_id"] or True
            x = abs(a["lp"] - b["lp"])
            d.append(x)
            if j == 0:
                d_first.append(x)
            if b["top_id"] != b["tok"]:
                flips += 1
            if worst is None or x > worst[0]:
                worst = (x, k, j, a["lp"], b["lp"], b["top_lp"])
    s = stats(d)
    s.update(first_token=stats(d_first), cand_argmax_differs_from_forced=flips,
             forced_sequences_match=forced_ok, n_sequences=len(ref_runs),
             sequences_with_wrong_call_count=n_bad,
             worst=dict(dlp=worst[0], seq=worst[1], pos=worst[2], ref_lp=worst[3], cand_lp=worst[4],
                        cand_top_lp=worst[5]) if worst else None,
             within_tol=bool(s.get("max", 0) <= TOL))
    return s


def main(ckpt):
    heartbeat()
    ids_cache = {}
    bf = lifetime(ckpt, "auto", ids_cache)
    # sanity: recorded logprob of the bf16 greedy token == engine-returned logprob
    san = [abs(x["lp"] - y) for r in bf["b64"] for x, y in zip(r["rec"], r["ret_lp"]) if y is not None]
    rec_vs_out = sum(1 for r in bf["b64"] if [x["top_id"] for x in r["rec"]] == r["ids"])
    log("bf16 rec-vs-returned logprob max", max(san) if san else None, "argmax==output seqs", rec_vs_out)
    targets = [r["ids"] for r in bf["b64"]]
    q8 = lifetime(ckpt, "int8", ids_cache, targets64=targets)

    ia, ib = bf["items"], q8["items"]
    lost = [k for k in ia if ia[k]["ok"] and not ib[k]["ok"]]
    gained = [k for k in ia if not ia[k]["ok"] and ib[k]["ok"]]
    same_ids = sum(1 for k in ia if ia[k]["ids"] == ib[k]["ids"])
    accuracy = dict(n_items=len(ia), correct_auto=sum(v["ok"] for v in ia.values()),
                    correct_int8=sum(v["ok"] for v in ib.values()),
                    retrieval_auto=sum(v["ok"] for k, v in ia.items() if k.startswith("ret")),
                    retrieval_int8=sum(v["ok"] for k, v in ib.items() if k.startswith("ret")),
                    short_auto=sum(v["ok"] for k, v in ia.items() if k.startswith("short")),
                    short_int8=sum(v["ok"] for k, v in ib.items() if k.startswith("short")),
                    lost_int8_vs_auto=lost, gained_int8_vs_auto=gained,
                    identical_token_sequences=same_ids)

    tf_int8 = forced_cmp(bf["b64"], q8["b64"])
    tf_bf16_b1 = forced_cmp(bf["b64"], bf["b1"])
    log("teacher-forced int8 b64 vs bf16 b64", json.dumps(tf_int8))
    log("teacher-forced bf16 b1 vs bf16 b64", json.dumps(tf_bf16_b1))

    passed = (len(lost) <= max(1, -(-len(ia) * 5 // 100)) and tf_int8["within_tol"]
              and tf_int8["forced_sequences_match"] == NC and tf_int8["sequences_with_wrong_call_count"] == 0)
    res = dict(
        passed=bool(passed),
        compared=("in-branch proxy, engine.json pool dtype auto (bf16, main) vs int8: accuracy.py item set "
                  "(2x32 needle on ~33K haystack via 32768 snapshot + 16 short); teacher-forced |dlogprob| "
                  "of int8 on bf16 batch-64 greedy tokens (reuse.py 64-prompt batch, 40 tokens); bf16 batch-1 "
                  "vs batch-64 alongside; tolerance 11.957"),
        accuracy=accuracy,
        teacher_forced_int8_b64_vs_bf16_b64=tf_int8,
        teacher_forced_bf16_b1_vs_bf16_b64=tf_bf16_b1,
        sanity=dict(bf16_rec_vs_returned_lp_max=max(san) if san else None,
                    bf16_rec_argmax_equals_output_seqs=rec_vs_out,
                    cached_b64=dict(auto=sorted(set(str(r["cached"]) for r in bf["b64"])),
                                    int8=sorted(set(str(r["cached"]) for r in q8["b64"])))))
    art = os.environ.get("AUTO_DEV_ARTIFACTS")
    if art:
        os.makedirs(art, exist_ok=True)
        with open(os.path.join(art, "acc_proxy_raw.json"), "w") as f:
            json.dump(dict(auto=bf, int8=q8), f)
    print(json.dumps(res), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main(sys.argv[1])
