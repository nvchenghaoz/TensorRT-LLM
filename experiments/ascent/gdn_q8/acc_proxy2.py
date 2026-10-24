# In-branch accuracy proxy for idea 37 (int8 GDN state pool), acceptance item 1,
# in the shape of the top-lead ruling of 17:10 (supersedes acc_proxy.py / job 199,
# whose call-counter teacher forcing misaligned when a prompt's prefill was split
# across chunks and under the overlap scheduler's one-step token lag).
#   * accuracy-adapter item style: 96 needle retrievals + 32 arithmetic items over
#     a primed 32768-token prefix (34816-token prompts = 32768 prefix + 2048
#     unique suffix ending in the question), greedy, at concurrency 128 and at
#     concurrency 64 (2 x 64); items lost int8 vs base (<= 7);
#   * check-style teacher-forced |dlogprob| vs 11.957, c64@64 (reuse.py's
#     64-prompt batch, 32768 prefix + 2048 unique, 40 tokens) and c128@128 (the
#     128 accuracy prompts, first 40 tokens): the int8 engine is fed the base
#     arm's greedy tokens (a logits processor records log_softmax at the forced
#     token, then forces it), so every position is compared on the same history;
#     token flips (cand argmax != forced token) are flagged with the reference's
#     top-2 logprob gap against the near-tie margin 23.915;
#   * context: base batch-1 vs base batch-64 teacher-forced, same statistics.
# Two engine lifetimes in one job, base first:
#   base      = experiments/ascent-config/engine.json with mamba_ssm_cache_dtype
#               removed (= main's engine.json: bf16 pool);
#   candidate = experiments/ascent-config/engine.json unmodified (int8 pool).
# No TRTLLM_* env var. Harness base engine: max_batch_size 128, decode CUDA graphs
# up to 128, max_num_tokens 8192, chunked prefill. The overlap scheduler is off in
# both arms so the logits processor sees each request's full token history
# (output position = len(tokens) - prompt_len; intermediate prefill chunks also
# call at position 0 and the final chunk's call overwrites them).
import glob
import json
import os
import random
import sys
import threading
import time

os.environ.setdefault("TLLM_WORKER_USE_SINGLE_PROCESS", "1")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gdn_nvfp4"))
import accuracy as acc  # noqa: E402  (FILLER / COLORS / ANIMALS)

PREFIX, SUFFIX, OSL, NC = 32768, 2048, 40, 64
N_RET, N_ARI = 96, 32
# Job 212 used 48: the model often opens a <think> block first, so a 48-token
# budget scored that near-tie (think vs answer) instead of retrieval.
ACC_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 192
TOL, TIE = 11.957, 23.915
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
    def __init__(self, tag, plen, targets=None):
        self.tag, self.plen, self.targets = tag, plen, targets
        STORE[tag] = {}

    def __call__(self, req_id, logits, token_ids, stream_ptr, client_id=None):
        import torch
        row = logits.reshape(-1, logits.shape[-1])[0]
        ls = torch.log_softmax(row.float(), -1)
        t2v, t2i = torch.topk(ls, 2)
        j = len(token_ids[0]) - self.plen
        if self.targets is not None and 0 <= j < len(self.targets):
            t = int(self.targets[j])
            v = torch.stack([ls[t], t2v[0], t2v[1], t2i[0].float(),
                             torch.tensor(float(t), device=ls.device)])
            row[t] = row.max() + 100.0
        else:
            v = torch.stack([t2v[0], t2v[0], t2v[1], t2i[0].float(), t2i[0].float()])
        STORE[self.tag][j] = v.clone()


def collect(tag):
    d = STORE[tag]
    out = []
    for j in sorted(d):
        lp, top1, top2, tid, tok = d[j].tolist()
        out.append(dict(j=j, lp=lp, top1=top1, gap=top1 - top2, top_id=int(tid), tok=int(tok)))
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


def build_items(tok, seed=8128):
    """96 retrieval + 32 arithmetic prompts, each exactly PREFIX + SUFFIX tokens."""
    rng = random.Random(seed)
    combos = [f"{c} {a}" for c in acc.COLORS for a in acc.ANIMALS]
    rng.shuffle(combos)
    names = combos[:N_RET]
    codes = [str(rng.randrange(1000000, 9999999)) for _ in names]
    sents, total = [], 0
    flen = {s: len(tok.encode(" " + s, add_special_tokens=False)) for s in acc.FILLER}
    while total < PREFIX + 1500:
        s = rng.choice(acc.FILLER)
        sents.append(s)
        total += flen[s]
    n = len(sents)
    for i, (nm, cd) in enumerate(zip(names, codes)):
        pos = int((i + 0.5) * n * 0.85 / N_RET)
        sents.insert(pos + i, f"Remember this: the access code for the {nm} vault is {cd}.")
    doc = "Read the following document carefully.\n\n" + " ".join(sents)
    pids = tok.encode(doc, add_special_tokens=False)[:PREFIX]
    assert len(pids) == PREFIX, len(pids)
    ptxt = tok.decode(pids)
    missing = [cd for cd in codes if cd not in ptxt]
    assert not missing, ("needles cut from prefix", missing)

    def suffix(qtext, srng):
        q = tok.encode(qtext, add_special_tokens=False)
        fill = " ".join(srng.choice(acc.FILLER) for _ in range(400))
        f = tok.encode("\n\n" + fill, add_special_tokens=False)[:SUFFIX - len(q)]
        s = f + q
        assert len(s) == SUFFIX
        return s

    items = []
    for i, (nm, cd) in enumerate(zip(names, codes)):
        q = ("\n\nBased on the document above, answer the question.\n"
             f"Question: What is the access code for the {nm} vault?\n"
             f"Answer: The access code for the {nm} vault is")
        items.append(dict(key=f"ret:{nm}", ids=pids + suffix(q, random.Random(seed * 1000 + i)),
                          expect=cd))
    for i in range(N_ARI):
        r = random.Random(seed * 7000 + i)
        if i % 2 == 0:
            a, b = r.randrange(100, 1000), r.randrange(100, 1000)
            q, ans, op = f"\n\nQ: What is {a} + {b}?\nA: {a} + {b} =", a + b, "+"
        else:
            a, b = r.randrange(11, 100), r.randrange(11, 100)
            q, ans, op = f"\n\nQ: What is {a} * {b}?\nA: {a} * {b} =", a * b, "*"
        items.append(dict(key=f"ari{i}:{a}{op}{b}",
                          ids=pids + suffix(q, random.Random(seed * 9000 + i)), expect=str(ans)))
    log("built %d items of %s tokens" % (len(items), sorted(set(len(it["ids"]) for it in items))))
    return items


def provenance(arm):
    """Top-lead 17:22: tree identity and marker from inside the executor process
    (TLLM_WORKER_USE_SINGLE_PROCESS=1, so this pid runs the executor), plus the
    live GDN state pool's dtype/shape read from the cache manager object."""
    import gc
    import subprocess

    import tensorrt_llm
    root = os.path.dirname(os.path.dirname(os.path.abspath(tensorrt_llm.__file__)))
    try:
        sha = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001
        sha = f"error {e}"
    marker_file = os.path.join(os.path.dirname(tensorrt_llm.__file__), "_torch", "modules", "fla",
                               "gdn_state_q8.py")
    marker = None
    if os.path.exists(marker_file):
        marker = next((ln.strip() for ln in open(marker_file) if "8-bit GDN recurrent-state pool" in ln), None)
    pools = []
    for o in gc.get_objects():
        try:
            if type(o).__name__.startswith(("MambaHybridCacheManager", "CppMambaHybridCacheManager")) \
                    and hasattr(o, "ssm_state_dtype"):
                pools.append(dict(cls=type(o).__name__, ssm_state_dtype=str(o.ssm_state_dtype),
                                  ssm_state_shape=list(getattr(o, "ssm_state_shape", []))))
        except Exception:  # noqa: BLE001
            continue
    p = dict(pid=os.getpid(), tensorrt_llm_file=tensorrt_llm.__file__, git_sha=sha,
             marker_file=marker_file, marker=marker, single_process=os.environ.get("TLLM_WORKER_USE_SINGLE_PROCESS"),
             trtllm_env=sorted(k for k in os.environ if k.startswith("TRTLLM_")), pools=pools)
    log(f"[{arm}] provenance", json.dumps(p))
    return p


def lifetime(ckpt, arm, cache, ref=None):
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

    engine_json = json.load(open("experiments/ascent-config/engine.json"))
    kvc = dict(engine_json.get("kv_cache_config", dict()))
    if arm == "base":
        kvc.pop("mamba_ssm_cache_dtype", None)
    else:
        assert kvc.get("mamba_ssm_cache_dtype") == "int8", kvc
    assert not [k for k in os.environ if k.startswith("TRTLLM_GDN")], "no TRTLLM_GDN env var"
    log(arm, "kv_cache_config", kvc)
    t0 = time.time()
    llm = LLM(model=ckpt, max_seq_len=36864, max_batch_size=128,
              max_num_tokens=8192, enable_chunked_prefill=True,
              disable_overlap_scheduler=True,
              cuda_graph_config=CudaGraphConfig(max_batch_size=128),
              kv_cache_config=KvCacheConfig(**kvc))
    log("[%s] LLM ready in %.1fs" % (arm, time.time() - t0))
    out = {"provenance": provenance(arm)}
    tok = llm.tokenizer
    if "items" not in cache:
        cache["items"] = build_items(tok)
        cache["ids"] = corpus_ids(tok, PREFIX + NC * SUFFIX)
    items = cache["items"]
    ids = cache["ids"]
    rprompts = [ids[:PREFIX] + ids[PREFIX + i * SUFFIX:PREFIX + (i + 1) * SUFFIX] for i in range(NC)]

    def gen(prompts, tag, max_tokens, targets=None, record=True):
        sps = []
        for i, p in enumerate(prompts):
            lp = None
            if record:
                lp = Rec(f"{arm}:{tag}:{i}", len(p), None if targets is None else targets[i])
            sps.append(SamplingParams(max_tokens=max_tokens, temperature=0.0, top_k=1,
                                      ignore_eos=True, logits_processor=lp))
        t = time.time()
        outs = llm.generate(prompts, sps, use_tqdm=False)
        res = []
        for i, o in enumerate(outs):
            c = o.outputs[0]
            res.append(dict(ids=list(c.token_ids), text=c.text, cached=getattr(o, "cached_tokens", None),
                            rec=collect(f"{arm}:{tag}:{i}") if record else None))
        log("[%s] %s: %d prompts %.1fs cached %s" % (arm, tag, len(prompts), time.time() - t,
                                                    sorted(set(str(r["cached"]) for r in res))))
        return res

    def score(res):
        return {it["key"]: dict(ok=it["expect"] in r["text"], ids=r["ids"], tail=r["text"][:80],
                                cached=r["cached"])
                for it, r in zip(items, res)}

    ap = [it["ids"] for it in items]
    gen(ap[:1], "prime", 1, record=False)
    c128 = gen(ap, "acc_c128", ACC_TOKENS)
    out["acc_c128"] = score(c128)
    out["acc_c64"] = score(gen(ap[:64], "acc_c64a", ACC_TOKENS, record=False)
                           + gen(ap[64:], "acc_c64b", ACC_TOKENS, record=False))
    for k in ("acc_c128", "acc_c64"):
        bad = [x for x, v in out[k].items() if not v["ok"]]
        log(f"[{arm}] {k}: correct {len(items) - len(bad)}/{len(items)} wrong {bad}")
    gen(rprompts[:1], "rprime", 1, record=False)
    if ref is None:
        out["tf_c128"] = [dict(ids=r["ids"][:OSL], rec=[x for x in r["rec"] if x["j"] < OSL],
                               cached=r["cached"]) for r in c128]
        out["tf_c64"] = gen(rprompts, "b64", OSL)
        tg = [r["ids"] for r in out["tf_c64"]]
        out["b1"] = [gen([rprompts[i]], f"b1_{i}", OSL, targets=[tg[i]])[0] for i in range(NC)]
    else:
        out["tf_c128"] = gen(ap, "tf_c128", OSL, targets=[r["ids"] for r in ref["tf_c128"]])
        out["tf_c64"] = gen(rprompts, "tf_c64", OSL, targets=[r["ids"] for r in ref["tf_c64"]])
    out["cached"] = dict(acc_c128=sorted(set(str(v["cached"]) for v in out["acc_c128"].values())),
                         tf_c64=sorted(set(str(r.get("cached")) for r in out["tf_c64"])))
    llm.shutdown()
    del llm
    return out


def stats(xs):
    xs = sorted(xs)
    if not xs:
        return dict(n=0)
    q = lambda p: xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))]  # noqa: E731
    return dict(n=len(xs), max=xs[-1], p99=q(0.99), p90=q(0.90), p50=q(0.5), mean=sum(xs) / len(xs))


def forced_cmp(ref_runs, cand_runs):
    d, d0, flips, ok_seq, bad_cnt, worst = [], [], [], 0, 0, None
    ref_rec_vs_ids, cand_forced_vs_target = 0, 0
    for k, (r, c) in enumerate(zip(ref_runs, cand_runs)):
        rr = {x["j"]: x for x in r["rec"] if x["j"] < OSL}
        cr = {x["j"]: x for x in c["rec"] if x["j"] < OSL}
        if c["ids"][:OSL] == r["ids"][:OSL]:
            ok_seq += 1
        if sorted(rr) != list(range(OSL)) or sorted(cr) != list(range(OSL)):
            bad_cnt += 1
        for j in range(OSL):
            if j not in rr or j not in cr:
                continue
            a, b = rr[j], cr[j]
            ref_rec_vs_ids += int(a["top_id"] != r["ids"][j])
            cand_forced_vs_target += int(b["tok"] != r["ids"][j])
            x = abs(a["lp"] - b["lp"])
            d.append(x)
            if j == 0:
                d0.append(x)
            if b["top_id"] != b["tok"]:
                flips.append(dict(seq=k, pos=j, dlp=x, ref_gap=a["gap"],
                                  cand_gap_to_forced=b["top1"] - b["lp"]))
            if worst is None or x > worst["dlp"]:
                worst = dict(dlp=x, seq=k, pos=j, ref_lp=a["lp"], cand_lp=b["lp"], ref_gap=a["gap"])
    s = stats(d)
    s.update(first_token=stats(d0), forced_sequences_match=ok_seq, n_sequences=len(ref_runs),
             sequences_missing_positions=bad_cnt, ref_rec_argmax_ne_output=ref_rec_vs_ids,
             cand_forced_ne_target=cand_forced_vs_target, worst=worst,
             token_flips=len(flips),
             flips_ref_gap_max=max((f["ref_gap"] for f in flips), default=None),
             flips_outside_near_tie=sum(1 for f in flips if f["ref_gap"] >= TIE),
             top_flips=sorted(flips, key=lambda f: -f["ref_gap"])[:5],
             within_tol=bool(s.get("max", 0) <= TOL))
    return s


def main(ckpt):
    heartbeat()
    cache = {}
    base = lifetime(ckpt, "base", cache)
    cand = lifetime(ckpt, "int8", cache, ref=base)

    accuracy = {}
    for k in ("acc_c128", "acc_c64"):
        a, b = base[k], cand[k]
        lost = [x for x in a if a[x]["ok"] and not b[x]["ok"]]
        accuracy[k] = dict(
            n_items=len(a), base_correct=sum(v["ok"] for v in a.values()),
            int8_correct=sum(v["ok"] for v in b.values()),
            base_retrieval=sum(v["ok"] for x, v in a.items() if x.startswith("ret")),
            int8_retrieval=sum(v["ok"] for x, v in b.items() if x.startswith("ret")),
            base_arith=sum(v["ok"] for x, v in a.items() if x.startswith("ari")),
            int8_arith=sum(v["ok"] for x, v in b.items() if x.startswith("ari")),
            lost=lost, gained=[x for x in a if not a[x]["ok"] and b[x]["ok"]],
            base_wrong={x: v["tail"] for x, v in a.items() if not v["ok"]},
            int8_wrong={x: v["tail"] for x, v in b.items() if not v["ok"]},
            identical_token_sequences=sum(1 for x in a if a[x]["ids"] == b[x]["ids"]))
    tf = dict(c64_at_64=forced_cmp(base["tf_c64"], cand["tf_c64"]),
              c128_at_128=forced_cmp(base["tf_c128"], cand["tf_c128"]))
    b1 = forced_cmp(base["tf_c64"], base["b1"])
    for k, v in tf.items():
        log("teacher-forced int8 vs base", k, json.dumps(v))
    log("teacher-forced base b1 vs base b64", json.dumps(b1))
    tf_ok = all(v["within_tol"] and v["forced_sequences_match"] == v["n_sequences"]
                and v["sequences_missing_positions"] == 0 and v["cand_forced_ne_target"] == 0
                for v in tf.values())
    lost_ok = all(len(v["lost"]) <= 7 for v in accuracy.values())
    res = dict(
        passed=bool(tf_ok and lost_ok),
        compared=("in-branch adapter proxy (top-lead 17:10 shape), base = engine.json minus "
                  "mamba_ssm_cache_dtype (bf16 pool) vs candidate = engine.json (int8 pool), same job, "
                  "no env var: 96 retrieval + 32 arithmetic over primed 32768 prefix (34816-token "
                  "prompts) at c128 and c64, items lost (<= 7); teacher-forced max |dlogprob| c64@64 "
                  "and c128@128 vs 11.957, flips vs near-tie margin 23.915; base b1 vs b64 alongside"),
        accuracy=accuracy, teacher_forced_int8_vs_base=tf,
        teacher_forced_base_b1_vs_base_b64=b1,
        cached=dict(base=base["cached"], int8=cand["cached"]),
        acc_max_tokens=ACC_TOKENS,
        provenance=dict(base=base["provenance"], int8=cand["provenance"]))
    art = os.environ.get("AUTO_DEV_ARTIFACTS")
    if art:
        os.makedirs(art, exist_ok=True)
        with open(os.path.join(art, "acc_proxy2_raw.json"), "w") as f:
            json.dump(dict(base=base, int8=cand), f)
    print(json.dumps(res), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main(sys.argv[1])
