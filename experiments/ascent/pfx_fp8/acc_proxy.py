import json, os, random, sys, threading, time
os.environ.setdefault("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
# In-branch accuracy PROXY for idea 22 (not the frozen adapter; seeds disjoint from 1001-6006). Run inline
# (python -c "<this>" <checkpoint>) in both trees via run_gpu_pair, one LLM per arm with the serving engine settings
# (max_batch_size 128, decode CUDA graphs to 128, max_num_tokens 8192, chunked prefill, kv_cache_config from the
# tree's experiments/ascent-config/engine.json: FP8 KV, block reuse, snapshot at 32768).
# 128 items, ALL on one shared ~33.1K-token haystack so the cascade engages (P = 32768 needs every request of a batch
# on the same prefix pages): 96 needle retrievals (7-digit vault codes) + 32 two-digit multiplications asked after the
# document. One priming request, then phase c128: all 128 at once (decode at B = 128, graph); phase c64: the first 64
# retrieval items again at once (decode at B = 64, graph). Greedy, 160 tokens, ignore_eos, logprobs 1. Reports correct
# counts, every item's ok flag, token-id digests and the decode graphs captured / replayed by batch size.
SEED = 7707
HAY_TOKENS = 33100
OSL = 160
T0 = time.time()


def log(*a):
    print("[%6.0fs]" % (time.time() - T0), *a, flush=True)


def heartbeat():
    def loop():
        while True:
            time.sleep(30)
            log("[heartbeat]")
    threading.Thread(target=loop, daemon=True).start()


FILLER = [
    "The morning fog settled over the harbor while fishermen prepared their nets.",
    "Local historians note that the old mill was rebuilt twice during the last century.",
    "A gentle breeze carried the scent of pine across the quiet valley.",
    "The committee met on Thursday to review the annual budget for the library.",
    "Several species of migratory birds rest near the lake every autumn.",
    "The bakery on the corner is known for its rye bread and almond pastries.",
    "Engineers inspected the bridge and recommended minor repairs to the railings.",
    "Children in the village learn to swim in the shallow river during summer.",
    "The museum recently acquired a collection of hand-painted ceramic tiles.",
    "Farmers in the region rotate barley, beans and clover to keep the soil healthy.",
    "A narrow trail winds up the hill to an abandoned stone watchtower.",
    "The town council approved a plan to plant two hundred new oak trees.",
    "Rain fell steadily through the night, filling the gutters and ditches.",
    "The orchestra rehearsed a new symphony in the old concert hall.",
    "Traders once carried salt and wool along the mountain pass to the coast.",
    "The lighthouse keeper recorded wind speeds in a leather-bound logbook.",
    "Students gathered in the courtyard to discuss the results of the science fair.",
    "The market opens at dawn, and the first stalls sell fresh herbs and eggs.",
    "An old ferry still crosses the estuary twice each day in calm weather.",
    "The botanical garden added a greenhouse for tropical ferns and orchids.",
    "Workers repaved the main square with granite stones from a nearby quarry.",
    "A small observatory on the ridge opens to visitors on clear evenings.",
    "The library's reading room has tall windows facing the western hills.",
    "Volunteers cleaned the beach after the storm washed up driftwood and kelp.",
]
COLORS = ["red", "blue", "green", "amber", "violet", "silver", "golden", "black", "white", "orange", "crimson", "teal",
          "ivory", "scarlet", "indigo", "bronze"]
ANIMALS = ["falcon", "otter", "tiger", "heron", "badger", "lynx", "raven", "salmon", "beetle", "walrus", "gecko",
           "panther", "sparrow", "bison", "marten", "cobra"]


def build_items(tok):
    rng = random.Random(SEED)
    combos = [c + " " + a for c in COLORS for a in ANIMALS]
    rng.shuffle(combos)
    names = combos[:96]
    codes = [str(rng.randrange(1000000, 9999999)) for _ in names]
    flen = dict((s, len(tok.encode(" " + s, add_special_tokens=False))) for s in FILLER)
    budget = HAY_TOKENS - 96 * 22 - 16
    sents, total = [], 0
    while total < budget:
        s = rng.choice(FILLER)
        sents.append(s)
        total += flen[s]
    n = len(sents)
    for i, (nm, cd) in enumerate(zip(names, codes)):
        sents.insert(int((i + 0.5) * n / 96) + i, "Remember this: the access code for the %s vault is %s." % (nm, cd))
    hay = "Read the following document carefully.\n\n" + " ".join(sents)
    items = []
    for nm, cd in zip(names, codes):
        items.append(("ret:" + nm, hay + "\n\nBased on the document above, answer the question.\nQuestion: What is the "
                      "access code for the %s vault?\nAnswer: The access code for the %s vault is" % (nm, nm), cd))
    for i in range(32):
        x, y = rng.randrange(12, 99), rng.randrange(12, 99)
        items.append(("mul%d:%dx%d" % (i, x, y), hay + "\n\nIgnore the document above and answer the arithmetic "
                      "question.\nQuestion: What is %d * %d?\nAnswer: %d * %d =" % (x, y, x, y), str(x * y)))
    log("haystack tokens", len(tok.encode(hay, add_special_tokens=False)), "items", len(items))
    return items


def main(ckpt):
    heartbeat()
    import hashlib
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.pyexecutor import cuda_graph_runner as cgr
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig
    captured, replays = dict(), dict()
    orig_capture, orig_replay = cgr.CUDAGraphRunner.capture, cgr.CUDAGraphRunner.replay

    def capture(self, key, *a, **kw):
        bs = str(cgr.KeyType(*key).batch_size)
        captured[bs] = captured.get(bs, 0) + 1
        return orig_capture(self, key, *a, **kw)

    def replay(self, key, *a, **kw):
        bs = str(cgr.KeyType(*key).batch_size)
        replays[bs] = replays.get(bs, 0) + 1
        return orig_replay(self, key, *a, **kw)

    cgr.CUDAGraphRunner.capture = capture
    cgr.CUDAGraphRunner.replay = replay
    kvc = dict(json.load(open("experiments/ascent-config/engine.json")).get("kv_cache_config", dict()))
    log("kv_cache_config", json.dumps(kvc))
    llm = LLM(model=ckpt, max_seq_len=36864, max_batch_size=128, max_num_tokens=8192, enable_chunked_prefill=True,
              cuda_graph_config=CudaGraphConfig(max_batch_size=128), kv_cache_config=KvCacheConfig(**kvc))
    log("LLM ready; graphs captured by batch size", json.dumps(captured))
    items = build_items(llm.tokenizer)
    sp = SamplingParams(max_tokens=OSL, temperature=0.0, top_k=1, ignore_eos=True, logprobs=1)
    llm.generate([items[0][1]], SamplingParams(max_tokens=1, temperature=0.0), use_tqdm=False)
    log("primed")

    def run(idx, tag):
        r0 = dict(replays)
        t = time.time()
        outs = llm.generate([items[i][1] for i in idx], sp, use_tqdm=False)
        res = dict()
        for i, o in zip(idx, outs):
            c = o.outputs[0]
            key, _, exp = items[i]
            lps = []
            for tid, d in zip(c.token_ids, c.logprobs or []):
                lp = d.get(tid) if isinstance(d, dict) else None
                lps.append(round(float(lp.logprob), 5) if lp is not None else None)
            res[key] = dict(ok=exp in c.text[:120], expect=exp, text=c.text[:120], ids=list(c.token_ids), lp=lps,
                            cached=getattr(o, "cached_tokens", None))
        dr = dict((k, replays.get(k, 0) - r0.get(k, 0)) for k in replays if replays.get(k, 0) != r0.get(k, 0))
        log("%s: %d items in %.1fs, correct %d, decode graph replays by batch %s" % (
            tag, len(idx), time.time() - t, sum(v["ok"] for v in res.values()), json.dumps(dr)))
        return res, dr

    p128, r128 = run(list(range(128)), "c128")
    p64, r64 = run(list(range(64)), "c64")
    out = dict(seed=SEED, osl=OSL, n_items=128,
               correct_c128=sum(v["ok"] for v in p128.values()),
               correct_retrieval_c128=sum(v["ok"] for k, v in p128.items() if k.startswith("ret")),
               correct_mul_c128=sum(v["ok"] for k, v in p128.items() if k.startswith("mul")),
               correct_c64=sum(v["ok"] for v in p64.values()),
               graphs_captured_by_batch=captured, replays_c128_phase=r128, replays_c64_phase=r64,
               wrong_c128=sorted(k for k, v in p128.items() if not v["ok"]),
               wrong_c64=sorted(k for k, v in p64.items() if not v["ok"]),
               ok_c128=dict((k, v["ok"]) for k, v in p128.items()),
               ids_sha_c128=dict((k, hashlib.sha256(json.dumps(v["ids"]).encode()).hexdigest()[:12])
                                 for k, v in p128.items()),
               c64_equals_c128_ids=sum(1 for k in p64 if p64[k]["ids"] == p128[k]["ids"]))
    for k in out["wrong_c128"]:
        log("WRONG c128", k, "expect", p128[k]["expect"], "got", repr(p128[k]["text"][:100]))
    art = os.environ.get("AUTO_DEV_ARTIFACTS")
    if art:
        os.makedirs(art, exist_ok=True)
        with open(os.path.join(art, "acc_proxy.json"), "w") as f:
            json.dump(dict(c128=p128, c64=p64, summary=out), f)
    print(json.dumps(out), flush=True)
    llm.shutdown()
    os._exit(0)


main(sys.argv[1])
