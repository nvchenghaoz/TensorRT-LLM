#!/usr/bin/env python3
"""Accuracy proxy for idea 47 (NVFP4 W4A4 prefill copy of FP8 GDN in_proj_qkvz).

One arm per process. Arm is selected by the env kill switch
TRTLLM_FP8_PREFILL_NVFP4_GDN (0 = arm A = main behaviour, unset = arm C).

Items:
  * 2 seeds (4401, 5501) x 32 needle-retrieval items on a ~33.1K-token shared
    haystack (32 needles per haystack, one question per needle); the haystack
    is primed once per seed through this arm's own prefill;
  * 16 short QA items.
Greedy decoding, max_tokens 200 (>= 160), engine.json serving KV settings
(FP8 KV, block reuse, mamba snapshot at 32768). Writes per-item tokens,
logprobs and scores to --out; prints one JSON line.
"""
import argparse
import glob
import json
import os
import random
import time

os.environ.setdefault("TLLM_WORKER_USE_SINGLE_PROCESS", "1")

import torch  # noqa: E402

SHORT = [
    ("Question: What is the capital of France?\nAnswer:", ["paris"]),
    ("Question: What is 12 multiplied by 12?\nAnswer:", ["144"]),
    ("Question: Which planet is known as the Red Planet?\nAnswer:", ["mars"]),
    ("Question: What is the chemical symbol for gold?\nAnswer:", ["au"]),
    ("Question: How many days are there in a leap year?\nAnswer:", ["366"]),
    ("Question: Who wrote the play Romeo and Juliet?\nAnswer:",
     ["shakespeare"]),
    ("Question: What is the largest ocean on Earth?\nAnswer:", ["pacific"]),
    ("Question: What is 256 divided by 8?\nAnswer:", ["32"]),
    ("Question: What is the boiling point of water in degrees Celsius at sea "
     "level?\nAnswer:", ["100"]),
    ("Question: What is the capital city of Japan?\nAnswer:", ["tokyo"]),
    ("Question: How many sides does a hexagon have?\nAnswer:", ["6", "six"]),
    ("Question: What gas do plants absorb from the air for photosynthesis?"
     "\nAnswer:", ["carbon dioxide", "co2", "co₂"]),
    ("Question: What is the square root of 81?\nAnswer:", ["9", "nine"]),
    ("Question: In which continent is Egypt located?\nAnswer:", ["africa"]),
    ("Question: What is 7 plus 15?\nAnswer:", ["22"]),
    ("Question: What is the freezing point of water in degrees Fahrenheit?"
     "\nAnswer:", ["32"]),
]

ADJ = ["Amber", "Basalt", "Cobalt", "Dune", "Ember", "Falcon", "Granite",
       "Harbor", "Indigo", "Juniper", "Krypton", "Lumen", "Meridian", "Nimbus",
       "Onyx", "Polaris", "Quartz", "Raven", "Sierra", "Tundra", "Umber",
       "Vortex", "Willow", "Xenon", "Yarrow", "Zephyr", "Aster", "Birch",
       "Cinder", "Delta", "Echo", "Fjord"]
NOUN = ["Lantern", "Bridge", "Compass", "Anvil", "Beacon", "Citadel", "Drift",
        "Engine", "Forge", "Garden", "Horizon", "Island", "Junction", "Keel",
        "Ledger", "Mosaic", "Needle", "Orchard", "Pinnacle", "Quarry",
        "Rampart", "Signal", "Tower", "Uplink", "Valley", "Warden"]


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def corpus_ids(tok, n_tokens, offset_files=0):
    files = sorted(glob.glob("docs/source/**/*.md", recursive=True))
    files += sorted(glob.glob("tensorrt_llm/_torch/**/*.md", recursive=True))
    files = files[offset_files:] + files[:offset_files]
    ids = []
    for f in files:
        try:
            with open(f, encoding="utf-8", errors="ignore") as fh:
                txt = fh.read()
        except OSError:
            continue
        ids.extend(tok.encode(txt, add_special_tokens=False))
        if len(ids) >= n_tokens:
            break
    assert len(ids) >= n_tokens, (len(ids), n_tokens)
    return ids[:n_tokens]


def build_seed(tok, seed, n_items=32, target=33100):
    rng = random.Random(seed)
    names = set()
    while len(names) < n_items:
        names.add(f"{rng.choice(ADJ)} {rng.choice(NOUN)}")
    names = sorted(names)
    rng.shuffle(names)
    codes = [str(rng.randint(1000000, 9999999)) for _ in names]
    needles = [
        tok.encode(f"\n\nThe access code for project {n} is {c}.\n\n",
                   add_special_tokens=False) for n, c in zip(names, codes)
    ]
    n_needle = sum(len(x) for x in needles)
    filler = corpus_ids(tok, target - n_needle,
                        offset_files=(seed * 7) % 50)
    # Needle positions: sorted random offsets in the filler.
    pos = sorted(rng.sample(range(200, len(filler) - 200), n_items))
    order = list(range(n_items))
    rng.shuffle(order)
    hay, last = [], 0
    for p, k in zip(pos, order):
        hay.extend(filler[last:p])
        hay.extend(needles[k])
        last = p
    hay.extend(filler[last:])
    items = []
    for n, c in zip(names, codes):
        q = tok.encode(
            f"\n\nQuestion: According to the document above, what is the "
            f"access code for project {n}?\nAnswer: The access code for "
            f"project {n} is",
            add_special_tokens=False)
        items.append({"name": n, "code": c, "q": q})
    return hay, items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--seeds", default="4401,5501")
    args = ap.parse_args()
    assert args.max_tokens >= 160

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

    kvc = {"dtype": "fp8", "enable_block_reuse": True,
           "mamba_state_config": {
               "additional_snapshot_offsets_from_start": [32768]}}
    eng = "experiments/ascent-config/engine.json"
    if os.path.exists(eng):
        with open(eng) as fh:
            kvc.update(json.load(fh).get("kv_cache_config", {}))
    envs = {k: os.environ.get(k) for k in (
        "TRTLLM_FP8_PREFILL_NVFP4_GDN", "TRTLLM_FP8_PREFILL_NVFP4_ATTN_QKV",
        "TRTLLM_FP8_PREFILL_NVFP4_ATTN_O", "TRTLLM_FP8_PREFILL_NVFP4_GDN_OUT")}
    log(f"arm={args.arm} env {envs} kvc={kvc}")
    t0 = time.time()
    llm = LLM(model=args.ckpt,
              kv_cache_config=KvCacheConfig(**kvc),
              cuda_graph_config=CudaGraphConfig(max_batch_size=8),
              enable_chunked_prefill=True,
              max_batch_size=8,
              max_num_tokens=8192,
              max_seq_len=36864,
              tensor_parallel_size=1)
    t_ready = time.time() - t0
    log(f"LLM ready in {t_ready:.1f}s; mem allocated "
        f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB")
    tok = llm.tokenizer

    from tensorrt_llm._torch.modules import fp8_prefill_nvfp4 as fpn
    stats_load = fpn.prefill_nvfp4_stats()
    log("nvfp4 prefill copies:", stats_load)

    # Per-layer weight rel-RMS of the NVFP4 copy (arm C only, a few layers).
    relrms = {}
    try:
        model = llm._executor.engine.model_engine.model
        lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2,
                            -3, -4, -6], device="cuda")
        mods = []
        for suf in ("in_proj_qkvz", "qkv_proj", "o_proj", "out_proj"):
            ms = [(n, m) for n, m in model.named_modules()
                  if n.endswith(suf)
                  and fpn.get_prefill_nvfp4_copy(m) is not None]
            mods += ms[::max(1, len(ms) // 3)][:3]
        for n, m in mods:
            w = (m.weight.float() * m.weight_scale.float()).to(torch.bfloat16)
            g = fpn.get_prefill_nvfp4_copy(m)[4]
            q, sf = torch.ops.trtllm.fp4_quantize(
                w, torch.tensor([1.0 / g], device="cuda"), 16, False, False)
            N, K = w.shape
            v = torch.stack([lut[(q & 0xF).long()], lut[(q >> 4).long()]],
                            -1).reshape(N, K)
            s = sf.reshape(N, -1)[:, :K // 16].view(
                torch.float8_e4m3fn).float().repeat_interleave(16, 1)
            wq = v * s * g
            wf = w.float()
            relrms[n] = float(((wq - wf).norm() / wf.norm()).item())
            del w, q, sf, v, s, wq, wf
        log("weight rel-RMS (NVFP4 copy vs FP8 dequant):", relrms)
    except Exception as e:  # noqa: BLE001
        log("rel-RMS failed:", repr(e))

    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0, top_k=1,
                        logprobs=1)

    def run(prompts):
        outs = llm.generate(prompts, sp, use_tqdm=False)
        res = []
        for o in outs:
            c = o.outputs[0]
            lps = []
            for tid, d in zip(c.token_ids, c.logprobs or []):
                lp = d.get(tid) if isinstance(d, dict) else None
                lps.append(float(lp.logprob) if lp is not None else None)
            res.append({"tokens": list(c.token_ids), "logprobs": lps,
                        "text": c.text,
                        "cached": getattr(o, "cached_tokens", None)})
        return res

    items = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        hay, its = build_seed(tok, seed)
        t = time.time()
        llm.generate([hay], SamplingParams(max_tokens=1, temperature=0.0,
                                           top_k=1), use_tqdm=False)
        tp = time.time() - t
        res = run([hay + it["q"] for it in its])
        log(f"seed {seed}: haystack {len(hay)} tok, prime {tp:.1f}s, "
            f"items {time.time() - t - tp:.1f}s")
        for it, r in zip(its, res):
            ok = it["code"] in r["text"]
            items.append({"id": f"r{seed}_{it['name']}", "kind": "retrieval",
                          "ok": ok, **r})
        log(f"seed {seed}: {sum(1 for i in items if i['id'].startswith(f'r{seed}') and i['ok'])}/32")
    sres = run([tok.encode(p, add_special_tokens=False) for p, _ in SHORT])
    for k, ((p, ans), r) in enumerate(zip(SHORT, sres)):
        ok = any(a in r["text"].lower() for a in ans)
        items.append({"id": f"s{k}", "kind": "short", "ok": ok, **r})

    nret = sum(1 for i in items if i["kind"] == "retrieval" and i["ok"])
    nsh = sum(1 for i in items if i["kind"] == "short" and i["ok"])
    stats = fpn.prefill_nvfp4_stats()
    out = {"arm": args.arm, "llm_ready_s": t_ready, "retrieval": nret,
           "short": nsh, "nvfp4_copies": stats_load["built"],
           "nvfp4_copy_bytes": stats_load["bytes"],
           "nvfp4_calls": stats["calls"], "weight_relrms": relrms,
           "nvfp4_groups": stats.get("groups"), "env": envs,
           "items": items}
    with open(args.out, "w") as fh:
        json.dump(out, fh)
    summary = {k: v for k, v in out.items() if k != "items"}
    print(json.dumps(summary), flush=True)
    llm.shutdown()


if __name__ == "__main__":
    main()
