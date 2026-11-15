#!/usr/bin/env python3
"""Idea 95 (branch 38): in-process full-model A/B/A'/B' for the FA2 longest-first CTA order
and the per-row split-KV merge.

ONE LLM (this tree's experiments/ascent-config/engine.json: breakable prefill CUDA graphs,
disable_mm_encoder, buckets 300..2000) serves four passes over the same prompt groups:
  A   branch default (FLASHINFER_FA2_LPT / FLASHINFER_FA2_MERGE_ROWS unset = on)
  B   FLASHINFER_FA2_LPT=0 FLASHINFER_FA2_MERGE_ROWS=0 (main's order, grid and merge)
  A2  on again, B2 off again
FlashInfer reads both switches on every plan() and every launch, and attention runs as the
eager break of the breakable graph, so a flip between passes changes the attention kernels
of replayed steps too. Token ids + top-20 logprobs are compared bitwise across passes; the
BCG replayed/eager context-step counts are reported per pass; a repeated sequence
(2 warmup + 10 calls of 1x300 and 3x1000, no reset) must equal its first call and pass A.
Usage: python llm_ab.py <checkpoint>
"""
import json
import os
import sys
import time

if len(sys.argv) > 1:
    os.environ["CKPT"] = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bcg_prefill"))

import bcg_ab as B  # noqa: E402  (sets TLLM_WORKER_USE_SINGLE_PROCESS=1)

KEYS = ("FLASHINFER_FA2_LPT", "FLASHINFER_FA2_MERGE_ROWS")


def set_on(on):
    for k in KEYS:
        os.environ.pop(k, None)
        if not on:
            os.environ[k] = "0"


def main():
    B.print_dims()
    set_on(True)
    llm = B.build_llm(batch_wait_timeout_ms=1500.0)
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import graphs as _g
    counts = getattr(_g, "_BCG_STEP_COUNTS", None)
    params = B.sp(logprobs=20)
    ps = B.Prompts(9595)
    groups = []
    for isl, n in ((300, 4), (1000, 4), (2000, 3), (4000, 1), (8000, 2)):
        groups += [("1x%d" % isl, [ps.make(isl)]) for _ in range(n)]
    for n, isl, reps in ((4, 300, 2), (16, 300, 1), (3, 1000, 2), (4, 2000, 1), (2, 700, 1)):
        groups += [("%dx%d" % (n, isl), [ps.make(isl) for _ in range(n)]) for _ in range(reps)]
    warm = B.Prompts(5)
    for isl in (300, 1000, 2000, 8000):
        B.run_batch(llm, [warm.make(isl)], params)
    passes, steps = {}, {}
    for tag, on in (("A", True), ("B", False), ("A2", True), ("B2", False)):
        set_on(on)
        before = dict(counts) if counts is not None else None
        res = []
        t0 = time.time()
        for _, grp in groups:
            _, outs = B.run_batch(llm, grp, params)
            res.append([B.summarize(o) for o in outs])
        passes[tag] = res
        if counts is not None:
            steps[tag] = {k: counts[k] - before[k] for k in counts}
        B.log(f"pass {tag} lpt/merge={'on' if on else 'off'} {time.time() - t0:.1f}s "
              f"steps={steps.get(tag)}")
    # repeated execution with the branch default, no reset
    set_on(True)
    rep1 = [g for g in groups if g[0] == "1x300"][0][1]
    rep3 = [g for g in groups if g[0] == "3x1000"][0][1]
    i1 = [k for k, g in enumerate(groups) if g[1] is rep1][0]
    i3 = [k for k, g in enumerate(groups) if g[1] is rep3][0]
    r1, r3 = [], []
    for _ in range(12):
        _, o = B.run_batch(llm, rep1, params)
        r1.append([B.summarize(x) for x in o])
        _, o = B.run_batch(llm, rep3, params)
        r3.append([B.summarize(x) for x in o])
    rep_ok = all(x == r1[0] for x in r1) and all(x == r3[0] for x in r3)
    rep_eq_a = r1[0] == passes["A"][i1] and r3[0] == passes["A"][i3]

    def diff(a, b):
        out = {}
        for (label, _), x, y in zip(groups, a, b):
            d = out.setdefault(label, [0, 0])
            d[1] += len(x)
            d[0] += sum(1 for u, v in zip(x, y) if u != v)
        return {k: f"{v[0]}/{v[1]} differ" for k, v in out.items()}, sum(v[0] for v in out.values())

    cmp, total = {}, 0
    for a, b in (("B", "A"), ("A2", "A"), ("B2", "A"), ("B2", "B")):
        per, n = diff(passes[a], passes[b])
        cmp[f"{a}_vs_{b}"] = {"n_diff": n, "per_group": per}
        total += n
    n_out = sum(len(x) for x in passes["A"])
    has_lp = bool(passes["A"][0][0]["lp"])
    replayed = sum((steps.get(t) or {}).get("replayed", 0) for t in ("A", "A2"))
    print(json.dumps({
        "passed": bool(total == 0 and rep_ok and rep_eq_a and has_lp and
                       (counts is None or replayed > 0)),
        "compared": "token ids + top-20 logprobs bitwise in ONE BCG process: branch default "
                    "(A, A2) vs FLASHINFER_FA2_LPT=0 + FLASHINFER_FA2_MERGE_ROWS=0 (B, B2, main's "
                    "paths); %d outputs per pass; repeated 12 calls of 1x300 and 3x1000 == first "
                    "== pass A" % n_out,
        "comparisons": cmp, "repeat_all_equal_first": rep_ok, "repeat_equal_pass_A": rep_eq_a,
        "logprobs_present": has_lp, "context_steps_per_pass": steps,
        "overlay": B.overlay()}), flush=True)


if __name__ == "__main__":
    main()
