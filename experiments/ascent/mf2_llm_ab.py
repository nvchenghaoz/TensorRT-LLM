"""In-process LLM A/B for branch 2 (knowledge 372 method).

One process (TLLM_WORKER_USE_SINGLE_PROCESS=1), one LLM, batch_wait_timeout_ms
so batch composition does not depend on arrival timing. Passes A (fused
post-attn junction off = main's chain), B (on), A' (off) over the same
prompts: isl300@1, isl1k@4, isl8k@1, isl8k@4, N prompts each, 1 greedy
output token, top-20 logprobs. Reports A-vs-A' and A-vs-B bitwise counts,
token flips and max |dlogprob|, plus how many times each kernel path ran.

usage: mf2_llm_ab.py <checkpoint> [n_per_case=10]
The __main__ guard matters (LLM API workers re-import this module).
"""
import json
import os
import random
import sys
import threading
import time


def _heartbeat():
    t0 = time.time()
    while True:
        time.sleep(60)
        print(f"[heartbeat] {time.time() - t0:.0f}s", flush=True)


WORDS = ("the of and to in is was for on that with as by at from his her it an were are which this be "
         "or has had not but have one their they its new first after who been also two time more year "
         "city river music school system state world history data model energy market water light "
         "village company science paper season team game church station family garden island bridge "
         "language theory number field road power north south east west small large early late").split()


def make_text(rng, n_words):
    out = []
    while len(out) < n_words:
        k = rng.randint(6, 18)
        s = [rng.choice(WORDS) for _ in range(k)]
        s[0] = s[0].capitalize()
        if rng.random() < 0.3:
            s.append(str(rng.randint(1, 2030)))
        out.append(" ".join(s) + ".")
    return " ".join(out)


def main():
    os.environ["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"
    os.environ.setdefault("TLLM_LOG_LEVEL", "INFO")
    threading.Thread(target=_heartbeat, daemon=True).start()
    ck = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    from transformers import AutoTokenizer

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.models import modeling_gemma4 as mg
    from tensorrt_llm.llmapi import KvCacheConfig

    tok = AutoTokenizer.from_pretrained(ck)
    cases = [("isl300@1", 300, 1), ("isl1k@4", 1000, 4), ("isl8k@1", 8000, 1), ("isl8k@4", 8000, 4)]
    prompts = []  # (case, conc, ids)
    rng = random.Random(20261114)
    for label, isl, conc in cases:
        for i in range(n):
            ids = tok(make_text(rng, isl * 2), add_special_tokens=False)["input_ids"][: isl - 1]
            ids = [tok.bos_token_id or 2] + ids
            assert len(ids) == isl, (len(ids), isl)
            prompts.append((label, conc, ids))
    print(f"built {len(prompts)} prompts", flush=True)

    # Count which junction path runs.
    counts = {"fused": 0, "cute_fp4": 0}
    _fused, _cute = mg.rmsnorm_residual_add_fp4_quant, mg.rmsnorm_fp4_quant

    def fused(*a, **k):
        counts["fused"] += 1
        return _fused(*a, **k)

    def cute(*a, **k):
        counts["cute_fp4"] += 1
        return _cute(*a, **k)

    mg.rmsnorm_residual_add_fp4_quant = fused
    mg.rmsnorm_fp4_quant = cute

    llm = LLM(
        model=ck,
        max_num_tokens=8192,
        max_seq_len=8300,
        max_batch_size=64,
        batch_wait_timeout_ms=1500,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.85, enable_block_reuse=False),
    )
    print("loaded", flush=True)
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)

    def run_pass(tag):
        res = []
        i = 0
        t0 = time.time()
        while i < len(prompts):
            label, conc, _ = prompts[i]
            group = prompts[i: i + conc]
            outs = llm.generate([{"prompt_token_ids": g[2]} for g in group], sp, use_tqdm=False)
            for g, o in zip(group, outs):
                c = o.outputs[0]
                lp = c.logprobs[0] if c.logprobs else {}
                top = {int(t): float(v.logprob) for t, v in lp.items()}
                res.append({"label": g[0], "tok": int(c.token_ids[0]), "top": top})
            i += conc
        print(f"[{tag}] {len(res)} prompts in {time.time() - t0:.1f}s counts={counts}", flush=True)
        return res

    # warmup (also JIT) on both paths
    mg._FUSE_POST_ATTN_FP4 = False
    llm.generate([{"prompt_token_ids": prompts[0][2]}], sp, use_tqdm=False)
    mg._FUSE_POST_ATTN_FP4 = True
    llm.generate([{"prompt_token_ids": prompts[0][2]}], sp, use_tqdm=False)

    passes = {}
    calls = {}
    for tag, flag in (("A", False), ("B", True), ("A2", False)):
        mg._FUSE_POST_ATTN_FP4 = flag
        c0 = dict(counts)
        passes[tag] = run_pass(tag)
        calls[tag] = {k: counts[k] - c0[k] for k in counts}
    llm.shutdown()

    def compare(a, b):
        eq = flips = 0
        maxd = 0.0
        per = {}
        for x, y in zip(a, b):
            same = x["tok"] == y["tok"] and x["top"] == y["top"]
            eq += same
            flips += x["tok"] != y["tok"]
            common = set(x["top"]) & set(y["top"])
            d = max((abs(x["top"][t] - y["top"][t]) for t in common), default=float("inf"))
            if set(x["top"]) != set(y["top"]):
                d = max(d, float("inf") if not common else d)
            maxd = max(maxd, d)
            e = per.setdefault(x["label"], [0, 0])
            e[0] += 1
            e[1] += same
        return {"bitwise_equal": eq, "total": len(a), "token_flips": flips, "max_abs_dlogprob": maxd,
                "per_case_equal": per}

    aa = compare(passes["A"], passes["A2"])
    ab = compare(passes["A"], passes["B"])
    print("A vs A2:", json.dumps(aa), flush=True)
    print("A vs B :", json.dumps(ab), flush=True)
    ok = (aa["bitwise_equal"] == aa["total"] and ab["bitwise_equal"] == ab["total"]
          and calls["B"]["fused"] > 0 and calls["B"]["cute_fp4"] == 0 and calls["A"]["fused"] == 0)
    print(json.dumps({"passed": bool(ok), "compared": "in-process LLM, fused junction off/on/off, "
                      "first-token id + top-20 logprobs bitwise", "A_vs_A2": aa, "A_vs_B": ab,
                      "kernel_calls": calls}))


if __name__ == "__main__":
    main()
