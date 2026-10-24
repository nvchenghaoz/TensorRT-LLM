# Three-arm version of acc_proxy2.py (same items, same teacher forcing), answering
# top-lead ruling 17:30 on job 212: base (bf16 pool) vs int8-row vs e4m3-row in ONE
# job, every output scored at a 48-token budget (job 212's) and at the full budget
# (192, job 230's), with lost items that are cut off inside <think> counted apart
# from wrong answers.
#   base = engine.json minus mamba_ssm_cache_dtype; int8 = engine.json unmodified;
#   e4m3 = engine.json unmodified + gdn_state_q8.Q8_FORMAT = 1 set in-process before
#   the engine is built (single-process executor; gdn_decode_t1 binds the name at
#   import, so both module globals are set; no source or env change).
# The 48-token score reads the first 48 greedy tokens of the 192-token run.
# Usage: acc_proxy3.py <ckpt> [192]
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import acc_proxy2 as P  # noqa: E402

SHORT = 48


def set_format(fmt):
    import tensorrt_llm._torch.modules.fla.gdn_decode_t1 as t1
    import tensorrt_llm._torch.modules.fla.gdn_state_q8 as q8
    q8.Q8_FORMAT = fmt
    t1.Q8_FORMAT = fmt
    return dict(gdn_state_q8=q8.Q8_FORMAT, gdn_decode_t1=t1.Q8_FORMAT)


def lifetime(ckpt, arm, cache, ref=None):
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

    engine_json = json.load(open("experiments/ascent-config/engine.json"))
    kvc = dict(engine_json.get("kv_cache_config", dict()))
    if arm == "base":
        kvc.pop("mamba_ssm_cache_dtype", None)
    else:
        assert kvc.get("mamba_ssm_cache_dtype") == "int8", kvc
    fmt = set_format(1 if arm == "e4m3" else 0)
    assert not [k for k in os.environ if k.startswith("TRTLLM_GDN")]
    P.log(arm, "kv_cache_config", kvc, "Q8_FORMAT", fmt)
    t0 = time.time()
    llm = LLM(model=ckpt, max_seq_len=36864, max_batch_size=128,
              max_num_tokens=8192, enable_chunked_prefill=True,
              disable_overlap_scheduler=True,
              cuda_graph_config=CudaGraphConfig(max_batch_size=128),
              kv_cache_config=KvCacheConfig(**kvc))
    P.log("[%s] LLM ready in %.1fs" % (arm, time.time() - t0))
    out = {"provenance": dict(P.provenance(arm), q8_format=fmt)}
    tok = llm.tokenizer
    if "items" not in cache:
        cache["items"] = P.build_items(tok)
        cache["ids"] = P.corpus_ids(tok, P.PREFIX + P.NC * P.SUFFIX)
    items, ids = cache["items"], cache["ids"]
    rprompts = [ids[:P.PREFIX] + ids[P.PREFIX + i * P.SUFFIX:P.PREFIX + (i + 1) * P.SUFFIX]
                for i in range(P.NC)]

    def gen(prompts, tag, max_tokens, targets=None, record=True):
        sps = []
        for i, p in enumerate(prompts):
            lp = P.Rec(f"{arm}:{tag}:{i}", len(p), None if targets is None else targets[i]) if record else None
            sps.append(SamplingParams(max_tokens=max_tokens, temperature=0.0, top_k=1,
                                      ignore_eos=True, logits_processor=lp))
        t = time.time()
        outs = llm.generate(prompts, sps, use_tqdm=False)
        res = []
        for i, o in enumerate(outs):
            c = o.outputs[0]
            res.append(dict(ids=list(c.token_ids), text=c.text, cached=getattr(o, "cached_tokens", None),
                            rec=P.collect(f"{arm}:{tag}:{i}") if record else None))
        P.log("[%s] %s: %d prompts %.1fs cached %s" % (arm, tag, len(prompts), time.time() - t,
                                                      sorted(set(str(r["cached"]) for r in res))))
        return res

    def score(res):
        d = {}
        for it, r in zip(items, res):
            short = tok.decode(r["ids"][:SHORT])
            d[it["key"]] = dict(ok=it["expect"] in r["text"], ok48=it["expect"] in short,
                                think48=("<think>" in short), ids=r["ids"], head=r["text"][:80],
                                cached=r["cached"])
        return d

    ap = [it["ids"] for it in items]
    gen(ap[:1], "prime", 1, record=False)
    c128 = gen(ap, "acc_c128", P.ACC_TOKENS)
    out["acc_c128"] = score(c128)
    out["acc_c64"] = score(gen(ap[:64], "acc_c64a", P.ACC_TOKENS, record=False)
                           + gen(ap[64:], "acc_c64b", P.ACC_TOKENS, record=False))
    for k in ("acc_c128", "acc_c64"):
        for b in ("ok", "ok48"):
            bad = [x for x, v in out[k].items() if not v[b]]
            P.log(f"[{arm}] {k} {b}: correct {len(items) - len(bad)}/{len(items)}")
    gen(rprompts[:1], "rprime", 1, record=False)
    if ref is None:
        out["tf_c128"] = [dict(ids=r["ids"][:P.OSL], rec=[x for x in r["rec"] if x["j"] < P.OSL],
                               cached=r["cached"]) for r in c128]
        out["tf_c64"] = gen(rprompts, "b64", P.OSL)
    else:
        out["tf_c128"] = gen(ap, "tf_c128", P.OSL, targets=[r["ids"] for r in ref["tf_c128"]])
        out["tf_c64"] = gen(rprompts, "tf_c64", P.OSL, targets=[r["ids"] for r in ref["tf_c64"]])
    llm.shutdown()
    del llm
    return out


def acc_table(a, b):
    res = {}
    for budget, key in (("192", "ok"), ("48", "ok48")):
        lost = [x for x in a if a[x][key] and not b[x][key]]
        gained = [x for x in a if not a[x][key] and b[x][key]]
        res[budget] = dict(
            base_correct=sum(v[key] for v in a.values()), cand_correct=sum(v[key] for v in b.values()),
            base_retrieval=sum(v[key] for x, v in a.items() if x.startswith("ret")),
            cand_retrieval=sum(v[key] for x, v in b.items() if x.startswith("ret")),
            base_arith=sum(v[key] for x, v in a.items() if x.startswith("ari")),
            cand_arith=sum(v[key] for x, v in b.items() if x.startswith("ari")),
            lost=len(lost), gained=len(gained), net=len(gained) - len(lost),
            lost_cut_in_think=sum(1 for x in lost if (b[x]["think48"] if budget == "48"
                                                       else "<think>" in b[x]["head"])),
            lost_items={x: b[x]["head"] for x in lost},
            base_wrong_cut_in_think=sum(1 for x in a if not a[x][key] and a[x]["think48"]))
    return res


def main(ckpt):
    P.heartbeat()
    cache = {}
    base = lifetime(ckpt, "base", cache)
    arms = {a: lifetime(ckpt, a, cache, ref=base) for a in ("int8", "e4m3")}
    res = dict(compared=("three-arm adapter proxy, one job: base (bf16 pool) vs int8-row vs e4m3-row; "
                         "96 retrieval + 32 arithmetic over primed 32768 prefix (34816-token prompts), "
                         "c128 and c64, scored at 48 and %d tokens, lost <= 7; teacher-forced |dlogprob| "
                         "c64@64 and c128@128 vs 11.957" % P.ACC_TOKENS),
               acc_max_tokens=P.ACC_TOKENS, accuracy={}, teacher_forced={},
               provenance={k: v["provenance"] for k, v in dict(base=base, **arms).items()})
    for a, c in arms.items():
        res["accuracy"][a] = {k: acc_table(base[k], c[k]) for k in ("acc_c128", "acc_c64")}
        res["teacher_forced"][a] = {
            "c64_at_64": P.forced_cmp(base["tf_c64"], c["tf_c64"]),
            "c128_at_128": P.forced_cmp(base["tf_c128"], c["tf_c128"])}
        for k, v in res["teacher_forced"][a].items():
            v.pop("top_flips", None)
    ok = {}
    for a in arms:
        acc_ok = all(res["accuracy"][a][k]["192"]["lost"] <= 7 for k in ("acc_c128", "acc_c64"))
        tf_ok = all(v["within_tol"] and v["forced_sequences_match"] == v["n_sequences"]
                    and v["sequences_missing_positions"] == 0 and v["cand_forced_ne_target"] == 0
                    for v in res["teacher_forced"][a].values())
        ok[a] = bool(acc_ok and tf_ok)
    res["arm_passed"] = ok
    res["passed"] = ok["int8"]
    art = os.environ.get("AUTO_DEV_ARTIFACTS")
    if art:
        os.makedirs(art, exist_ok=True)
        with open(os.path.join(art, "acc_proxy3_raw.json"), "w") as f:
            json.dump(dict(base=base, **arms), f)
    print(json.dumps(res), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main(sys.argv[1])
