"""In-process LLM A/B for branch 25 (idea 28), copied from experiments/ascent/mf2_llm_ab.py.

One process (TLLM_WORKER_USE_SINGLE_PROCESS=1), one LLM, batch_wait_timeout_ms 1500
so batch composition does not depend on arrival timing. Passes A (all branch-25 host
trims off = main's path), B (on), A2 (off) over the same prompts: isl300@1, isl1k@1,
isl1k@4, isl8k@1, isl8k@4, N prompts each, 1 greedy output token, top-20 logprobs.
Then a same-process re-send (no reset) with trims on: 2 prompts per class, classes
ordered 8k, 1k, 8k (and 300), judged against the switch-OFF run (pass A).
Counts how often each trimmed path was taken (or bypassed) per pass.

usage: llm_ab.py <checkpoint> [n_per_case=100]
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
    os.environ.setdefault("TLLM_LOG_LEVEL", "WARNING")
    threading.Thread(target=_heartbeat, daemon=True).start()
    ck = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    print("TRTLLM_FI_HOST_PREFILL_PLAN =", os.environ.get("TRTLLM_FI_HOST_PREFILL_PLAN", "1 (default)"),
          flush=True)

    from transformers import AutoTokenizer

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.attention import attention as attn_mod
    from tensorrt_llm._torch.attention.backends import flashinfer as fib
    from tensorrt_llm._torch.models import modeling_gemma4 as mg
    from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2 as kvm
    from tensorrt_llm.llmapi import KvCacheConfig

    switches = [(m, a) for m, a in ((mg, "_HOST_TRIMS"), (kvm, "_KV_VIEW_CACHE"),
                                    (fib, "_PLAN_PARAMS_MEMO_ENABLED")) if hasattr(m, a)]
    print("switches:", [a for _, a in switches], flush=True)

    tok = AutoTokenizer.from_pretrained(ck)
    cases = [("isl300@1", 300, 1), ("isl1k@1", 1000, 1), ("isl1k@4", 1000, 4), ("isl8k@1", 8000, 1),
             ("isl8k@4", 8000, 4)]
    prompts = []  # (case, conc, ids)
    rng = random.Random(20261114)
    for label, isl, conc in cases:
        for i in range(n):
            ids = tok(make_text(rng, isl * 2), add_special_tokens=False)["input_ids"][: isl - 1]
            ids = [tok.bos_token_id or 2] + ids
            assert len(ids) == isl, (len(ids), isl)
            prompts.append((label, conc, ids))
    print(f"built {len(prompts)} prompts", flush=True)

    # Dispatch counters: zero-copy KV views built, PlanParams constructed,
    # base-class quant-flag derivations, attention layer calls.
    counts = {"kv_view_built": 0, "get_buffers": 0, "planparams_built": 0, "plan_calls": 0,
              "quant_flags_derived": 0, "attn_impl": 0}
    _conv = kvm.convert_to_torch_tensor

    def conv(*a, **k):
        counts["kv_view_built"] += 1
        return _conv(*a, **k)

    kvm.convert_to_torch_tensor = conv
    _gb = kvm.KVCacheManagerV2.get_buffers

    def gb(self, *a, **k):
        counts["get_buffers"] += 1
        return _gb(self, *a, **k)

    kvm.KVCacheManagerV2.get_buffers = gb
    _ppinit = fib.PlanParams.__init__

    def ppinit(self, *a, **k):
        counts["planparams_built"] += 1
        _ppinit(self, *a, **k)

    fib.PlanParams.__init__ = ppinit
    _plan = fib.FlashInferAttentionMetadata.plan

    def plan(self, *a, **k):
        counts["plan_calls"] += 1
        return _plan(self, *a, **k)

    fib.FlashInferAttentionMetadata.plan = plan
    _uqo = attn_mod.Attention._use_quantize_output

    def uqo(self):
        counts["quant_flags_derived"] += 1
        return _uqo(self)

    attn_mod.Attention._use_quantize_output = uqo
    _ai = attn_mod.Attention._attn_impl

    def ai(self, *a, **k):
        counts["attn_impl"] += 1
        return _ai(self, *a, **k)

    attn_mod.Attention._attn_impl = ai
    counts["final_norm_direct"] = 0
    _fin = mg._flashinfer_rmsnorm

    def fin(*a, **k):
        counts["final_norm_direct"] += 1
        return _fin(*a, **k)

    mg._flashinfer_rmsnorm = fin

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

    def set_switches(on):
        for m, a in switches:
            setattr(m, a, on)

    def gen(group):
        outs = llm.generate([{"prompt_token_ids": g[2]} for g in group], sp, use_tqdm=False)
        res = []
        for g, o in zip(group, outs):
            c = o.outputs[0]
            lp = c.logprobs[0] if c.logprobs else {}
            top = {int(t): float(v.logprob) for t, v in lp.items()}
            res.append({"label": g[0], "tok": int(c.token_ids[0]), "top": top})
        return res

    def run_pass(tag):
        res = []
        i = 0
        t0 = time.time()
        while i < len(prompts):
            label, conc, _ = prompts[i]
            group = prompts[i: i + conc]
            res.extend(gen(group))
            i += conc
        print(f"[{tag}] {len(res)} prompts in {time.time() - t0:.1f}s counts={counts}", flush=True)
        return res

    # warmup (also JIT) on both paths
    for on in (False, True):
        set_switches(on)
        gen([prompts[0]])

    passes = {}
    calls = {}
    for tag, flag in (("A", False), ("B", True), ("A2", False)):
        set_switches(flag)
        c0 = dict(counts)
        passes[tag] = run_pass(tag)
        calls[tag] = {k: counts[k] - c0[k] for k in counts}

    # Same-process re-send, trims on, no reset: 2 prompts per class (groups kept
    # whole), classes ordered 8k, 1k, 8k, then 1k@1 and 300@1.
    set_switches(True)
    first_idx = {}
    for idx, (label, conc, _) in enumerate(prompts):
        first_idx.setdefault(label, idx)
    order = ["isl8k@4", "isl8k@1", "isl1k@4", "isl1k@1", "isl8k@4", "isl8k@1", "isl300@1"]
    conc_of = {c[0]: c[2] for c in cases}
    resend = []  # (prompt index, result)
    c0 = dict(counts)
    visit = {}
    for label in order:
        conc = conc_of[label]
        v = visit.get(label, 0)
        visit[label] = v + 1
        if conc == 1:
            for i in (first_idx[label] + 2 * v, first_idx[label] + 2 * v + 1):
                resend.append((i, gen([prompts[i]])[0]))
        else:
            # one whole group, exactly as it was batched in the passes
            start = first_idx[label] + conc * v
            for j, r in enumerate(gen(prompts[start: start + conc])):
                resend.append((start + j, r))
    calls["resend"] = {k: counts[k] - c0[k] for k in counts}
    llm.shutdown()

    def same(x, y):
        return x["tok"] == y["tok"] and x["top"] == y["top"]

    def compare(a, b):
        eq = flips = 0
        maxd = 0.0
        per = {}
        for x, y in zip(a, b):
            s = same(x, y)
            eq += s
            flips += x["tok"] != y["tok"]
            common = set(x["top"]) & set(y["top"])
            d = max((abs(x["top"][t] - y["top"][t]) for t in common), default=float("inf"))
            maxd = max(maxd, d)
            e = per.setdefault(x["label"], [0, 0])
            e[0] += 1
            e[1] += s
        return {"bitwise_equal": eq, "total": len(a), "token_flips": flips, "max_abs_dlogprob": maxd,
                "per_case_equal": per}

    aa = compare(passes["A"], passes["A2"])
    ab = compare(passes["A"], passes["B"])
    rs_per = {}
    for idx, r in resend:
        e = rs_per.setdefault(r["label"], [0, 0])
        e[0] += 1
        e[1] += same(r, passes["A"][idx]) and same(r, passes["B"][idx])
    rs_ok = all(v[0] == v[1] for v in rs_per.values()) and len(resend) > 0
    print("A vs A2:", json.dumps(aa), flush=True)
    print("A vs B :", json.dumps(ab), flush=True)
    print("resend vs A and B (equal/total per class):", json.dumps(rs_per), flush=True)
    print("calls:", json.dumps(calls), flush=True)
    ok = (aa["bitwise_equal"] == aa["total"] and ab["bitwise_equal"] == ab["total"] and rs_ok
          and calls["B"]["attn_impl"] > 0
          # trims taken in B: no per-call KV views / PlanParams / flag derivations
          and calls["B"]["kv_view_built"] == 0 and calls["B"]["planparams_built"] == 0
          and calls["B"]["quant_flags_derived"] == 0
          and calls["B"]["final_norm_direct"] > 0 and calls["A"]["final_norm_direct"] == 0
          # bypassed in A: main's per-call work runs
          and calls["A"]["kv_view_built"] == calls["A"]["get_buffers"] > 0
          and calls["A"]["quant_flags_derived"] == calls["A"]["attn_impl"])
    print(json.dumps({"passed": bool(ok), "compared": "in-process LLM, branch-25 host trims off/on/off + "
                      "same-process re-send (trims on, no reset) vs off run; first-token id + top-20 "
                      "logprobs bitwise; TRTLLM_FI_HOST_PREFILL_PLAN default (1) in all passes",
                      "A_vs_A2": aa, "A_vs_B": ab, "resend_vs_A_and_B": rs_per, "calls": calls}))


if __name__ == "__main__":
    main()
