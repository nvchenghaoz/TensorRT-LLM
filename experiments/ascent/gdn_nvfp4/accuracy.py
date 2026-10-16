# Accuracy proxy for the decode-only NVFP4 W4A16 copy of GDN in_proj_qkvz.
# Runs the model (LLM API, serving engine settings) with the copy disabled
# (TRTLLM_GDN_INPROJ_NVFP4_DECODE=0, original FP8 math) or enabled, on:
#   - 2 seeds x 32 long-context retrieval items: a ~33K-token haystack with 32
#     embedded 7-digit codes (shared prefix, snapshot at 32768), one question
#     per code; greedy; correct if the code is in the output;
#   - 16 short items with multi-token answers (decode-dependent);
#   - 4 long greedy generations (256 tokens) for divergence statistics.
# Usage: accuracy.py run <mode> <out.json> | accuracy.py compare a.json b.json
import json
import os
import random
import sys
import time

CKPT = os.environ.get("CKPT")
HAY_TOKENS = 33100


def log(*a):
    print(*a, flush=True)


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
    "The train station was painted green and cream during the restoration.",
    "Beekeepers in the valley sell honey flavored by heather and wild thyme.",
    "The annual regatta draws sailors from towns along the entire coastline.",
    "An exhibition of maps showed how the river changed course over centuries.",
    "The weaving guild still uses wooden looms built more than a hundred years ago.",
    "Snow closed the northern road for three days in the middle of winter.",
]
COLORS = ["red", "blue", "green", "amber", "violet", "silver", "golden", "black",
          "white", "orange", "crimson", "teal", "ivory", "scarlet", "indigo", "bronze"]
ANIMALS = ["falcon", "otter", "tiger", "heron", "badger", "lynx", "raven", "salmon",
           "beetle", "walrus", "gecko", "panther", "sparrow", "bison", "marten", "cobra"]

SHORT = [
    ("List the first ten prime numbers: 2, 3, 5,", ["7", "11", "13", "17", "19", "23", "29"]),
    ("Count from 20 to 30: 20, 21,", ["22", "23", "24", "25", "26", "27", "28", "29", "30"]),
    ("The days of the week in order are Monday, Tuesday,", ["Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]),
    ("The months of the year in order are January, February, March,", ["April", "May", "June", "July", "August", "September"]),
    ("The planets of the solar system, in order from the Sun, are Mercury,", ["Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]),
    ("Q: What is 12 * 12? Show the result only.\nA: 12 * 12 =", ["144"]),
    ("Q: What is 365 - 120?\nA: 365 - 120 =", ["245"]),
    ("Q: What is 1000 / 8?\nA: 1000 / 8 =", ["125"]),
    ("Q: What is 7 + 8 + 9?\nA: 7 + 8 + 9 =", ["24"]),
    ("Q: What are the capital cities of France, Japan and Italy?\nA: The capitals are", ["Paris", "Tokyo", "Rome"]),
    ("Q: Name the author of 'Pride and Prejudice' and the author of 'Hamlet'.\nA:", ["Austen", "Shakespeare"]),
    ("The first five letters of the English alphabet are A, B,", ["C", "D", "E"]),
    ("Q: What is the chemical formula of water, and what is the chemical symbol of gold?\nA:", ["H", "2", "O", "Au"]),
    ("The powers of two are 1, 2, 4, 8,", ["16", "32", "64", "128", "256"]),
    ("The square numbers are 1, 4, 9,", ["16", "25", "36", "49", "64"]),
    ("Q: Spell the word 'planet' backwards, letter by letter.\nA: t, e,", ["n", "a", "l", "p"]),
]

LONG_GEN = [
    "Write a detailed explanation of how a refrigerator works, covering the compressor, condenser, expansion valve and evaporator.\n\n",
    "Explain the causes and consequences of the French Revolution in a structured essay.\n\n",
    "Describe step by step how to implement binary search in Python, including edge cases.\n\n",
    "Write a short story about a lighthouse keeper who discovers a message in a bottle.\n\n",
]


def build_haystack(tok, seed):
    rng = random.Random(seed)
    names, used = [], set()
    while len(names) < 32:
        n = f"{rng.choice(COLORS)} {rng.choice(ANIMALS)}"
        if n not in used:
            used.add(n)
            names.append(n)
    codes = [str(rng.randrange(1000000, 9999999)) for _ in names]
    flen = {s: len(tok.encode(" " + s, add_special_tokens=False)) for s in FILLER}
    budget = HAY_TOKENS - 32 * 22 - 16
    sents, total = [], 0
    while total < budget:
        s = rng.choice(FILLER)
        sents.append(s)
        total += flen[s]
    n = len(sents)
    for i, (nm, cd) in enumerate(zip(names, codes)):
        pos = int((i + 0.5) * n / 32)
        sents.insert(pos + i, f"Remember this: the access code for the {nm} vault is {cd}.")
    text = "Read the following document carefully.\n\n" + " ".join(sents)
    ids = tok.encode(text, add_special_tokens=False)
    log("haystack tokens (untrimmed)", len(ids))
    return text, names, codes


def run(mode, out_path):
    os.environ["TRTLLM_GDN_INPROJ_NVFP4_DECODE"] = "1" if mode == "nvfp4" else "0"
    os.environ["TRTLLM_GDN_INPROJ_NVFP4_DECODE_REPORT"] = "1" if mode == "nvfp4" else "0"
    from transformers import AutoTokenizer
    import torch
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig
    tok = AutoTokenizer.from_pretrained(CKPT)
    t0 = time.time()
    llm = LLM(model=CKPT, max_seq_len=36864, max_batch_size=8, max_num_tokens=8192,
              enable_chunked_prefill=True,
              cuda_graph_config=CudaGraphConfig(max_batch_size=8),
              kv_cache_config=KvCacheConfig(enable_block_reuse=True, mamba_state_config={
                  "additional_snapshot_offsets_from_start": [32768]}))
    log(f"[{mode}] LLM ready in {time.time()-t0:.1f}s; torch allocated in this process "
        f"{torch.cuda.memory_allocated()/1e9:.2f} GB; free/total {[x/1e9 for x in torch.cuda.mem_get_info()]}")
    results = {"mode": mode, "items": {}}
    for seed in (4401, 5501):
        text, names, codes = build_haystack(tok, seed)
        prompts, keys, expects = [], [], []
        for nm, cd in zip(names, codes):
            prompts.append(text + f"\n\nBased on the document above, answer the question.\n"
                           f"Question: What is the access code for the {nm} vault?\n"
                           f"Answer: The access code for the {nm} vault is")
            keys.append(f"ret{seed}:{nm}")
            expects.append([cd])
        sp = SamplingParams(max_tokens=int(os.environ.get("RET_TOKENS", "160")), temperature=0.0, logprobs=1)
        # Prime the shared prefix once, then the rest (block reuse).
        t1 = time.time()
        outs = llm.generate(prompts[:1], sp)
        outs += llm.generate(prompts[1:], sp)
        log(f"[{mode}] retrieval seed {seed}: {time.time()-t1:.1f}s, prompt tokens {len(outs[0].prompt_token_ids)}")
        for k, e, o in zip(keys, expects, outs):
            c = o.outputs[0]
            results["items"][k] = {"text": c.text, "ids": list(c.token_ids), "expect": e,
                                   "ok": all(x in c.text for x in e),
                                   "lp": [list(d.values())[0].logprob if isinstance(d, dict) else d for d in (c.logprobs or [])],
                                   "cached": getattr(o, "cached_tokens", None)}
    sp = SamplingParams(max_tokens=48, temperature=0.0, logprobs=1)
    outs = llm.generate([p for p, _ in SHORT], sp)
    for i, ((p, e), o) in enumerate(zip(SHORT, outs)):
        c = o.outputs[0]
        results["items"][f"short{i}"] = {"text": c.text, "ids": list(c.token_ids), "expect": e,
                                         "ok": all(x in c.text for x in e),
                                         "lp": [list(d.values())[0].logprob if isinstance(d, dict) else d for d in (c.logprobs or [])]}
    sp = SamplingParams(max_tokens=256, temperature=0.0, ignore_eos=True, logprobs=1)
    outs = llm.generate(LONG_GEN, sp)
    for i, o in enumerate(outs):
        c = o.outputs[0]
        results["items"][f"gen{i}"] = {"text": c.text, "ids": list(c.token_ids), "expect": [], "ok": True,
                                       "lp": [list(d.values())[0].logprob if isinstance(d, dict) else d for d in (c.logprobs or [])]}
    n_ok = sum(v["ok"] for k, v in results["items"].items() if not k.startswith("gen"))
    n = sum(1 for k in results["items"] if not k.startswith("gen"))
    log(f"[{mode}] correct {n_ok}/{n}")
    for k, v in results["items"].items():
        if not v["ok"]:
            log(f"[{mode}] WRONG {k}: expect {v['expect']} got {v['text'][-200:]!r}")
    with open(out_path, "w") as f:
        json.dump(results, f)
    llm.shutdown()


def compare(a_path, b_path):
    a = json.load(open(a_path))["items"]
    b = json.load(open(b_path))["items"]
    lost = [k for k in a if a[k]["ok"] and not b[k]["ok"]]
    gained = [k for k in a if not a[k]["ok"] and b[k]["ok"]]
    same, firstdiv, lpd = 0, [], []
    for k in a:
        ia, ib = a[k]["ids"], b[k]["ids"]
        if ia == ib:
            same += 1
        else:
            d = next((i for i, (x, y) in enumerate(zip(ia, ib)) if x != y), min(len(ia), len(ib)))
            firstdiv.append((k, d))
        la, lb = a[k]["lp"], b[k]["lp"]
        dpos = next((i for i, (x, y) in enumerate(zip(ia, ib)) if x != y), min(len(ia), len(ib)))
        for i in range(min(dpos, len(la), len(lb))):
            if isinstance(la[i], (int, float)) and isinstance(lb[i], (int, float)):
                lpd.append(abs(la[i] - lb[i]))
    n = sum(1 for k in a if not k.startswith("gen"))
    n_ret = sum(1 for k in a if k.startswith("ret"))
    res = {
        "fp8_correct": sum(a[k]["ok"] for k in a if not k.startswith("gen")),
        "nvfp4_correct": sum(b[k]["ok"] for k in b if not k.startswith("gen")),
        "n_items": n, "n_retrieval": n_ret,
        "retrieval_fp8": sum(a[k]["ok"] for k in a if k.startswith("ret")),
        "retrieval_nvfp4": sum(b[k]["ok"] for k in b if k.startswith("ret")),
        "lost": lost, "gained": gained, "allowed_lost": max(1, -(-n * 5 // 100)),
        "identical_token_sequences": same, "n_sequences": len(a),
        "first_divergence": firstdiv,
        "logprob_absdiff_before_divergence": {
            "n": len(lpd), "mean": (sum(lpd) / len(lpd)) if lpd else None,
            "max": max(lpd) if lpd else None},
    }
    res["passed"] = len(lost) <= res["allowed_lost"]
    for k, d in firstdiv:
        log(f"diverge {k} at token {d}: fp8 {a[k]['text'][:120]!r} | nvfp4 {b[k]['text'][:120]!r}")
    print(json.dumps(res))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3])
    else:
        compare(sys.argv[2], sys.argv[3])
