"""In-process LLM A/X/A' + own-prompt re-send for idea 60 (branch 26).

Copied from experiments/ascent/mf2_llm_ab.py (not edited). One process
(TLLM_WORKER_USE_SINGLE_PROCESS=1), one LLM, batch_wait_timeout_ms 1500, own
make_text prompts (seed 20261114), 1 greedy token, top-20 logprobs.
TRTLLM_FI_HOST_PREFILL_PLAN is left at main's default (1) in every arm.

Cases (n per case): isl300@1, isl1k@1, isl1k@4, isl2k@1, isl8k@1, isl8k@4.
Passes: A (chunk path off = main), X (on), A' (off). A/X/A' judges X only if
A/A' is bitwise. Then the re-send, same process, no reset: groups 4x8k, 4x1k,
4x8k(second group) and 2 c=1 prompts per ISL, sent once and re-sent, first
with the switch ON (judged vs its own first send and vs the OFF send), then
with it OFF. Counts chunk-path dispatches per pass, and checks the autotuner
cache (keys + picks) is unchanged by the passes.

usage: llm_ab.py <checkpoint> [n_per_case=17]
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
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 17
    print("TRTLLM_FI_HOST_PREFILL_PLAN =", os.environ.get("TRTLLM_FI_HOST_PREFILL_PLAN", "<unset: main default 1>"),
          flush=True)

    from transformers import AutoTokenizer

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.autotuner import AutoTuner
    from tensorrt_llm._torch.models import modeling_gemma4 as mg
    from tensorrt_llm.llmapi import KvCacheConfig

    tok = AutoTokenizer.from_pretrained(ck)
    cases = [("isl300@1", 300, 1), ("isl1k@1", 1000, 1), ("isl1k@4", 1000, 4), ("isl2k@1", 2000, 1),
             ("isl8k@1", 8000, 1), ("isl8k@4", 8000, 4)]
    rng = random.Random(20261114)

    def mk(isl):
        ids = tok(make_text(rng, isl * 2), add_special_tokens=False)["input_ids"][: isl - 1]
        ids = [tok.bos_token_id or 2] + ids
        assert len(ids) == isl, (len(ids), isl)
        return ids

    prompts = []  # (case, conc, ids)
    for label, isl, conc in cases:
        nn = (n + conc - 1) // conc * conc
        for _ in range(nn):
            prompts.append((label, conc, mk(isl)))
    # re-send set: list of groups (label, [ids...]) in order 8k-group, 1k-group, 8k-group, then c=1 pairs
    resend = [("rs_8k@4", [mk(8000) for _ in range(4)]), ("rs_1k@4", [mk(1000) for _ in range(4)]),
              ("rs_8k@4b", [mk(8000) for _ in range(4)])]
    for isl, lab in ((300, "rs_300@1"), (1000, "rs_1k@1"), (2000, "rs_2k@1"), (8000, "rs_8k@1")):
        for _ in range(2):
            resend.append((lab, [mk(isl)]))
    print(f"built {len(prompts)} A/B prompts, {sum(len(g) for _, g in resend)} re-send prompts", flush=True)

    counts = {"chunked": 0, "fallback": 0}
    _orig = mg.gate_up_gelu_fp4_chunked

    def counted(*a, **k):
        r = _orig(*a, **k)
        counts["chunked" if r is not None else "fallback"] += 1
        return r

    mg.gate_up_gelu_fp4_chunked = counted

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

    def snap_tuner():
        c = AutoTuner.get().profiling_cache.cache
        return {str(k): str(v[:2]) for k, v in c.items()}

    tuner0 = snap_tuner()
    nv = {k: v for k, v in tuner0.items() if "nvfp4_gemm" in k}
    print(f"tuner cache after warmup: {len(tuner0)} entries, {len(nv)} nvfp4_gemm", flush=True)
    for k, v in sorted(nv.items())[:40]:
        print("  ", k[:160], "->", v, flush=True)

    def gen(groups_ids):
        outs = llm.generate([{"prompt_token_ids": ids} for ids in groups_ids], sp, use_tqdm=False)
        res = []
        for o in outs:
            c = o.outputs[0]
            lp = c.logprobs[0] if c.logprobs else {}
            res.append({"tok": int(c.token_ids[0]), "top": {int(t): float(v.logprob) for t, v in lp.items()}})
        return res

    def run_pass(tag):
        res = []
        i = 0
        t0 = time.time()
        while i < len(prompts):
            label, conc, _ = prompts[i]
            group = prompts[i: i + conc]
            for g, r in zip(group, gen([g[2] for g in group])):
                r["label"] = g[0]
                res.append(r)
            i += conc
        print(f"[{tag}] {len(res)} prompts in {time.time() - t0:.1f}s counts={counts}", flush=True)
        return res

    # warmup both paths (JIT)
    mg._MLP_CHUNK_ENABLED = False
    gen([prompts[-1][2]])
    mg._MLP_CHUNK_ENABLED = True
    gen([prompts[-1][2]])

    passes, calls = {}, {}
    for tag, flag in (("A", False), ("X", True), ("A2", False)):
        mg._MLP_CHUNK_ENABLED = flag
        c0 = dict(counts)
        passes[tag] = run_pass(tag)
        calls[tag] = {k: counts[k] - c0[k] for k in counts}

    def same(x, y):
        return x["tok"] == y["tok"] and x["top"] == y["top"]

    def compare(a, b):
        eq = flips = 0
        maxd = 0.0
        per = {}
        first_diff = None
        for idx, (x, y) in enumerate(zip(a, b)):
            s = same(x, y)
            eq += s
            flips += x["tok"] != y["tok"]
            common = set(x["top"]) & set(y["top"])
            d = max((abs(x["top"][t] - y["top"][t]) for t in common), default=float("inf"))
            maxd = max(maxd, d)
            e = per.setdefault(x["label"], [0, 0])
            e[0] += 1
            e[1] += s
            if not s and first_diff is None:
                first_diff = {"index": idx, "label": x["label"], "tok": [x["tok"], y["tok"]]}
        return {"bitwise_equal": eq, "total": len(a), "token_flips": flips, "max_abs_dlogprob": maxd,
                "per_case_equal": per, "first_diff": first_diff}

    aa = compare(passes["A"], passes["A2"])
    ax = compare(passes["A"], passes["X"])
    print("A vs A2:", json.dumps(aa), flush=True)
    print("A vs X :", json.dumps(ax), flush=True)

    # re-send: ON first, then OFF; each group sent then re-sent immediately (no reset)
    rs = {}
    for flag in (True, False):
        mg._MLP_CHUNK_ENABLED = flag
        c0 = dict(counts)
        first, second = [], []
        for lab, ids in resend:
            r1 = gen(ids)
            r2 = gen(ids)
            for x in r1:
                x["label"] = lab
            for y in r2:
                y["label"] = lab
            first += r1
            second += r2
        rs["on" if flag else "off"] = {"first": first, "second": second,
                                        "calls": {k: counts[k] - c0[k] for k in counts}}
    rs_report = {
        "on_second_vs_first": compare(rs["on"]["first"], rs["on"]["second"]),
        "off_second_vs_first": compare(rs["off"]["first"], rs["off"]["second"]),
        "on_first_vs_off_first": compare(rs["off"]["first"], rs["on"]["first"]),
        "on_second_vs_off_first": compare(rs["off"]["first"], rs["on"]["second"]),
        "calls": {k: v["calls"] for k, v in rs.items()},
    }
    print("re-send:", json.dumps(rs_report), flush=True)

    tuner1 = snap_tuner()
    tuner_same = tuner0 == tuner1
    llm.shutdown()

    def all_eq(c):
        return c["bitwise_equal"] == c["total"]

    ok = (all_eq(aa) and all_eq(ax) and calls["X"]["chunked"] > 0 and calls["A"]["chunked"] == 0
          and calls["A2"]["chunked"] == 0 and all(all_eq(v) for k, v in rs_report.items() if k != "calls")
          and rs_report["calls"]["on"]["chunked"] > 0 and rs_report["calls"]["off"]["chunked"] == 0
          and tuner_same)
    print(json.dumps({"passed": bool(ok),
                      "compared": "in-process LLM, chunk path off/on/off + re-send on/off, first-token id + "
                                  "top-20 logprobs bitwise; TRTLLM_FI_HOST_PREFILL_PLAN=main default (1)",
                      "A_vs_A2": aa, "A_vs_X": ax, "kernel_calls": calls, "resend": rs_report,
                      "tuner_cache_entries": len(tuner1), "tuner_cache_unchanged": tuner_same}))


if __name__ == "__main__":
    main()
