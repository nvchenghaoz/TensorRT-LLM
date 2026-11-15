"""Idea 116 (branch 50): own-prompt in-process A/X/A'/Y for the NVFP4 tactic pins.

One process (TLLM_WORKER_USE_SINGLE_PROCESS=1), one eager LLM (no BCG overlay, so the module
flags reach every NVFP4 GEMM call), FP8 KV, block reuse off, max_num_tokens 8192. Autotuner on
(tunes in the warmup; the same cache serves every pass). Passes over the same prompts:
  A  : both pins off (TRTLLM_NVFP4_GATEUP_PIN / _DOWN_PIN module flags 0) = main's path
  X  : gate_up pins only               -> must be bitwise equal to A (gate_up pins are exact)
  A2 : both off                        -> must be bitwise equal to A
  Y  : both pins (the branch default)  -> token agreement and max |d logprob| vs A reported
  Y2 : both pins again                 -> must be bitwise equal to Y (repeat, no reset)
1 greedy token, top-20 logprobs. Prompts: c=1 isl 300 x20, 1000 x20, 2000 x20 (down M=2000 is
pinned), 4000 x10; 4x300 groups x5 (M=1200), 6x300 groups x2 (M=1800, pinned), all unique
random token ids. Group members are enqueued before the executor thread can fetch any.
NVFP4 dispatches per pass are counted by (op, N, tactic) to show the pins ran.
"""
import json
import os
import sys
import threading
import time

os.environ.setdefault("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
CKPT = os.environ.get("CKPT") or sys.argv[1]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _hb():
    t0 = time.time()
    while True:
        time.sleep(60)
        print(f"[heartbeat] {time.time() - t0:.0f}s", flush=True)


def main():
    threading.Thread(target=_hb, daemon=True).start()
    cfg = json.load(open(os.path.join(CKPT, "config.json")))
    tc = cfg.get("text_config", cfg)
    h, inter = tc["hidden_size"], tc["intermediate_size"]
    GU, DN = (2 * inter, h), (h, inter)
    log(f"[dims] hidden={h} intermediate={inter} layers={tc['num_hidden_layers']}")

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.custom_ops import torch_custom_ops as tco
    from tensorrt_llm.llmapi import KvCacheConfig

    counts = {}
    _fwd = tco.NVFP4GemmUnifiedRunner.forward

    def fwd(self, inputs, tactic=-1, bias=None, **kw):
        nk = (int(inputs[1].shape[0]), int(inputs[1].shape[1]) * 2)
        if nk in (GU, DN):
            key = f"{'gate_up' if nk == GU else 'down'}|M{int(inputs[0].shape[0])}|{tactic}"
            counts[key] = counts.get(key, 0) + 1
        return _fwd(self, inputs, tactic=tactic, bias=bias, **kw)

    tco.NVFP4GemmUnifiedRunner.forward = fwd
    llm = LLM(model=CKPT, kv_cache_config=KvCacheConfig(dtype="fp8", enable_block_reuse=False,
                                                        free_gpu_memory_fraction=0.8),
              max_num_tokens=8192, max_batch_size=64, max_seq_len=9216,
              # every group's members land in one step (fixed M per group in every pass)
              batch_wait_timeout_ms=300.0)
    import random

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CKPT)
    rng = random.Random(116)
    words = ("the of and to in is was for on that with as by at from his her it an were are which this be "
             "or has had not but have one their they its new first after who been also two time more year "
             "city river music school system state world history data model energy market water light "
             "village company science paper season team game church station family garden island bridge "
             "language theory number field road power north south east west small large early late").split()

    def mk(isl):
        out = []
        while len(out) < isl * 2:
            s = [rng.choice(words) for _ in range(rng.randint(6, 18))]
            s[0] = s[0].capitalize()
            out.extend(s)
            out[-1] += "."
        ids = tok(" ".join(out), add_special_tokens=False)["input_ids"][: isl - 1]
        ids = [tok.bos_token_id or 2] + ids
        assert len(ids) == isl
        return ids

    groups = []
    for isl, n in ((300, 20), (1000, 20), (2000, 20), (4000, 10)):
        groups += [(f"isl{isl}@1", [mk(isl)]) for _ in range(n)]
    groups += [("4x300", [mk(300) for _ in range(4)]) for _ in range(5)]
    groups += [("6x300", [mk(300) for _ in range(6)]) for _ in range(2)]
    n_prompts = sum(len(g[1]) for g in groups)
    sp = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True, logprobs=20)

    def run(prompts):
        old = sys.getswitchinterval()
        sys.setswitchinterval(0.5)
        try:
            futs = [llm.generate_async(p, sampling_params=sp) for p in prompts]
        finally:
            sys.setswitchinterval(old)
        return [f.result() for f in futs]

    def summarize(o):
        c = o.outputs[0]
        lp = c.logprobs[0] if c.logprobs else {}
        return {"tok": int(c.token_ids[0]),
                "top": {int(t): float(getattr(v, "logprob", v)) for t, v in lp.items()}}

    def send(tag):
        c0 = dict(counts)
        res = []
        for label, ps in groups:
            for o in run(ps):
                r = summarize(o)
                r["label"] = label
                res.append(r)
        calls = {k: counts[k] - c0.get(k, 0) for k in counts if counts[k] - c0.get(k, 0)}
        log(f"[{tag}] {len(res)} outputs; nvfp4 dispatches: {json.dumps(calls)}")
        return res, calls

    def compare(a, b):
        eq = flips = 0
        maxd = 0.0
        per = {}
        for x, y in zip(a, b):
            same = x["tok"] == y["tok"] and x["top"] == y["top"]
            eq += same
            flips += x["tok"] != y["tok"]
            common = set(x["top"]) & set(y["top"])
            d = max((abs(x["top"][t] - y["top"][t]) for t in common), default=0.0)
            maxd = max(maxd, d)
            e = per.setdefault(x["label"], [0, 0, 0, 0.0])
            e[0] += 1
            e[1] += same
            e[2] += x["tok"] != y["tok"]
            e[3] = max(e[3], d)
        return {"bitwise_equal": eq, "total": len(a), "token_agreement": len(a) - flips,
                "max_abs_dlogprob": maxd,
                "per_case_n_equal_flips_maxd": {k: [v[0], v[1], v[2], round(v[3], 6)] for k, v in per.items()}}

    def set_flags(gu, dn):
        tco.NVFP4_GATEUP_PIN = gu
        tco.NVFP4_DOWN_PIN = dn

    set_flags(True, True)
    for _ in range(2):
        for label, ps in groups[:: max(1, len(groups) // 12)]:
            run(ps)
    passes, calls = {}, {}
    for tag, gu, dn in (("A", False, False), ("X", True, False), ("A2", False, False), ("Y", True, True),
                        ("Y2", True, True)):
        set_flags(gu, dn)
        passes[tag], calls[tag] = send(tag)
    set_flags(True, True)
    ax = compare(passes["A"], passes["X"])
    aa = compare(passes["A"], passes["A2"])
    ay = compare(passes["A"], passes["Y"])
    yy = compare(passes["Y"], passes["Y2"])
    for k, v in (("A_vs_X", ax), ("A_vs_A2", aa), ("A_vs_Y", ay), ("Y_vs_Y2", yy)):
        log(f"{k}: {json.dumps(v)}")
    pinned_down = sum(v for k, v in calls["Y"].items() if k.startswith("down|M2000") or k.startswith("down|M1800"))
    ok = (ax["bitwise_equal"] == ax["total"] and aa["bitwise_equal"] == aa["total"]
          and yy["bitwise_equal"] == yy["total"] and pinned_down > 0)
    llm.shutdown()
    print(json.dumps({"passed": bool(ok), "n_prompts": n_prompts, "A_vs_X_gate_up_pins": ax, "A_vs_A2": aa,
                      "A_vs_Y_both_pins": ay, "Y_vs_Y2": yy, "dispatches": calls,
                      "compared": "in-process eager LLM, first token id + top-20 logprobs: X (gate_up pins) and A2 "
                                  "bitwise vs A (pins off), Y2 vs Y bitwise; Y (both pins) token agreement and "
                                  "max |dlogprob| vs A reported"}), flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
