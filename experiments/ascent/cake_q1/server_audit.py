"""Branch 145/157 (idea 245 Q1, resubmit idea 296) end-to-end audit: ONE 4-GPU in-process run, not timing evidence.

Settings: the protocol server settings (TP4, CUDA graphs [1,2,4,8] padded, KV 0.5/reuse/128, max_batch 8,
max_num_tokens 8192), overlaid by this tree's experiments/ascent-config/engine.json and then by MAIN's engine.json
layout since landing 51 (moe_tensor_parallel_size 2 x moe_expert_parallel_size 2; top-lead ruling session 7 (1)).
Every arm runs MoE in deterministic mode (moe_config.disable_finalize_fusion=True): the served EP2xTP2 chain is not
bitwise repeatable at M=1 with finalize fusion on (pitfall 2658, ruling), and this audit judges ATTENTION switches.

Tactics are pinned across lifetimes (method 2663/2678): every arm gets the same TLLM_AUTOTUNER_CACHE_PATH under
$AUTO_DEV_ARTIFACTS. Arm W (writer) tunes and saves it, and also records the Kineto decode traces for the route
checks. Every compared arm (A1, A1b, A2..A5) LOADS it, so A1 and A1b are symmetric. TLLM_LOG_LEVEL=INFO so the
'[Autotuner] Loading cache from' line and the CUDA-graph capture lines are printed; each arm's full log is kept in
$AUTO_DEV_ARTIFACTS/phase_<arm>.log and counted per rank.

Arms (each a fresh subprocess / LLM lifetime, so every env switch is read at worker start and graph capture):
  W    default, writes the cache, then Kineto decode windows c1 and c4 (LLM.start_profile, 40 steps)
  A1   default                                   (control)
  A1b  default again                             (control; GATE A1 == A1b bitwise)
  A2   TRTLLM_CSA2_ATTN_PDL=0 (Q1 off)           -> bitwise vs A1
  A3   TRTLLM_ENABLE_PDL=0                       -> bitwise vs A1
  A4   TRTLLM_CSA2_CAKE_DECODE=0                 -> bitwise vs A5
  A5   TRTLLM_CSA2_CAKE_DECODE=0 + ATTN_PDL=0
Generation: 5 fixed text prompts, c1 = prompt 0 alone, c4 = prompts 1-4 together; 64 greedy tokens (63 decode
steps), token ids + the sampled token's logprob.
Gates: (G1) cache-load line on all 4 ranks in every loading arm; (G2) prefill first-token logprob equal across arms;
(G3) A1 == A1b bitwise; then A2/A3 == A1 and A4 == A5 bitwise. If G3 fails, a LABELED spread comparison is reported
(never called bitwise): per prompt, first divergence index and max |dlogprob| over the common prefix, each arm vs A1
against A1b vs A1; if prefill already differs between A1 and A1b, the nondeterminism is in PREFILL and teacher
forcing cannot help, so only the spread is reported.
Route checks (as job 1441) on every rank's W traces, plus graph-capture lines for batch 1/2/4/8 and eager-fallback
warnings per arm. Usage: server_audit.py <checkpoint>. Last stdout line: JSON.
"""
import glob
import gzip
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

STEPS = 40
MAX_TOKENS = 64
ART = Path(os.environ.get("AUTO_DEV_ARTIFACTS", "artifacts")).resolve()
CACHE = ART / "autotuner_cache" / "cache.json"
WORKTREE = Path(__file__).resolve().parents[3]
MERGE_RE = re.compile(r"CakeMergeF32Kernel")
K2_RE = re.compile(r"csa2_lowm_wo_a")
K3_RE = re.compile(r"csa2_mxfp8_low_m_gemm")
BODY_RE = re.compile(r"decode_body|kernel_sm120_dsv41_mixed_sparse_mla")
PREP_RE = re.compile(r"_cake_index_prep")
AR_RE = re.compile(r"allreduce|all_reduce|lamport|oneshot|twoshot|nccl", re.I)
ALL_ARMS = {
    "W": {},
    "A1": {},
    "A1b": {},
    "A2": {"TRTLLM_CSA2_ATTN_PDL": "0"},
    "A3": {"TRTLLM_ENABLE_PDL": "0"},
    "A4": {"TRTLLM_CSA2_CAKE_DECODE": "0"},
    "A5": {"TRTLLM_CSA2_CAKE_DECODE": "0", "TRTLLM_CSA2_ATTN_PDL": "0"},
}
# Model loads take ~17-20 min each on node smc521ge-0109. Default: W (writes the cache, records the traces), then
# A1, A1b, A2, A3, which all LOAD the cache (acceptance item 1: the load line on all 4 ranks in every compared arm).
# AUDIT_ARMS=W,A1,A1b,A2,A3,A4,A5 runs the full set.
ARMS = {n: ALL_ARMS[n] for n in os.environ.get("AUDIT_ARMS", "W,A1,A1b,A2,A3").split(",")}
WRITER = next(iter(ARMS))
# main's experiments/ascent-config/engine.json since landing 51 (branch 147); this branch's base predates it.
MAIN_LAYOUT = {"moe_tensor_parallel_size": 2, "moe_expert_parallel_size": 2}
PROTOCOL = dict(tensor_parallel_size=4, moe_expert_parallel_size=4, pipeline_parallel_size=1,
                enable_attention_dp=False, enable_chunked_prefill=True, max_batch_size=8, max_num_tokens=8192,
                max_seq_len=36864, trust_remote_code=True)
LOG_PATTERNS = re.compile(r"Loading cache from|Saving cache to|Running autotuner warmup|CUDA graph (capture|warmup)|"
                          r"eager|Traceback|Error|error:", re.I)

TOPICS = ["the history of river trade in medieval Europe", "how a compiler turns loops into vector instructions",
          "the life cycle of stars and the origin of heavy elements", "designing a small vegetable garden in a dry climate",
          "the rules and strategy of the board game Go"]


def make_prompts():
    import random
    rng = random.Random(245)
    words = ("people often notice that small changes in practice lead to large differences in outcome over many "
             "years because each step builds on the previous one and the record of what worked is kept carefully "
             "by those who came before").split()
    prompts = []
    for t in TOPICS:
        body = []
        for i in range(40):
            body.append(" ".join(rng.choice(words) for _ in range(12)).capitalize() + ".")
        prompts.append(f"You are writing a detailed essay about {t}. Background notes: " + " ".join(body)
                       + f"\n\nNow write the essay about {t}, starting with a clear introduction.\n\n")
    return prompts


def llm_kwargs():
    kw = dict(PROTOCOL)
    eng = WORKTREE / "experiments" / "ascent-config" / "engine.json"
    if eng.exists():
        kw.update(json.loads(eng.read_text()))
    kw.update(MAIN_LAYOUT)
    return kw


def run_phase(ckpt, name):
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, MoeConfig

    import random
    t0 = time.time()
    kw = llm_kwargs()
    print(f"[{name}] LLM kwargs: {kw} env-cache={os.environ.get('TLLM_AUTOTUNER_CACHE_PATH')} "
          f"cache_exists={CACHE.exists()}", flush=True)
    llm = LLM(model=ckpt,
              kv_cache_config=KvCacheConfig(enable_block_reuse=True, free_gpu_memory_fraction=0.5,
                                            tokens_per_block=128),
              cuda_graph_config=CudaGraphConfig(batch_sizes=[1, 2, 4, 8], enable_padding=True),
              moe_config=MoeConfig(disable_finalize_fusion=True), **kw)
    print(f"[{name}] LLM up in {time.time() - t0:.0f}s", flush=True)
    prompts = make_prompts()
    sp = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0, ignore_eos=True, logprobs=1)
    res = {}
    for c, group in ((1, [prompts[0]]), (4, prompts[1:5])):
        outs = llm.generate(group, sp)
        rows = []
        for o in outs:
            toks = list(o.outputs[0].token_ids)
            lps = []
            for t, d in zip(toks, o.outputs[0].logprobs or []):
                if isinstance(d, dict):
                    v = d.get(t, next(iter(d.values())))
                    lps.append(float(getattr(v, "logprob", v)))
                else:
                    lps.append(float(d))
            rows.append({"prompt_len": len(o.prompt_token_ids), "tokens": toks, "logprobs": lps})
        res[f"c{c}"] = rows
        print(f"[{name}] c{c}: tokens[:6]={[r['tokens'][:6] for r in rows]} lp0={[r['logprobs'][:1] for r in rows]}",
              flush=True)
    if name == WRITER:
        rng = random.Random(245)
        root = ART / "traces"
        for c in (1, 4):
            d = root / f"decode_c{c}"
            d.mkdir(parents=True, exist_ok=True)
            spl = SamplingParams(max_tokens=1200, temperature=0.0, ignore_eos=True)
            futs = [llm.generate_async([rng.randrange(1000, 60000) for _ in range(512)], spl) for _ in range(c)]
            time.sleep(3.0)
            llm.start_profile(output_dir=str(d), num_steps=STEPS, activities=["GPU"])
            print(f"[{name}] decode_c{c}: profile started", flush=True)
            [f.result() for f in futs]
            try:
                llm.stop_profile()
            except Exception as exc:  # noqa: BLE001
                print(f"stop_profile: {exc!r}", flush=True)
            deadline = time.time() + 300
            while time.time() < deadline and len(list(d.glob("*.json*"))) < 4:
                time.sleep(5)
            time.sleep(3)
    llm.shutdown()
    (ART / f"phase_{name}.json").write_text(json.dumps(res))


_EV = re.compile(rb'"ph":\s*"X",\s*"cat":\s*"kernel",\s*"name":\s*"((?:[^"\\]|\\.)*)",\s*"pid":\s*"?([^,"]*)"?,'
                 rb'\s*"tid":\s*"?([^,"]*)"?,\s*"ts":\s*([0-9.eE+-]+),\s*"dur":\s*([0-9.eE+-]+)')


def load(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rb") as f:
        buf = f.read()
    evs = []
    for m in _EV.finditer(buf):
        try:
            name = json.loads(b'"' + m.group(1) + b'"')
        except Exception:  # noqa: BLE001
            name = m.group(1).decode(errors="replace")
        evs.append((float(m.group(4)), float(m.group(5)), name, m.group(2).decode() + "/" + m.group(3).decode()))
    if not evs and b'"kernel"' in buf:
        data = json.loads(buf)
        raw = data.get("traceEvents", data) if isinstance(data, dict) else data
        evs = [(float(e["ts"]), float(e["dur"]), e.get("name", ""), f"{e.get('pid')}/{e.get('tid')}") for e in raw
               if isinstance(e, dict) and e.get("ph") == "X" and e.get("cat") == "kernel" and "dur" in e]
    evs.sort()
    return evs, {"graph_node_id_keys": buf.count(b'"graph node id"'), "graph_id_keys": buf.count(b'"graph id"')}


def reduce_rank(evs):
    c = {"prep": 0, "decode_body": 0, "merge_true": 0, "merge_false": 0, "k2": 0, "k3": 0}
    for _, _, n, _ in evs:
        if PREP_RE.search(n):
            c["prep"] += 1
        if BODY_RE.search(n):
            c["decode_body"] += 1
        if MERGE_RE.search(n):
            c["merge_true" if "<true>" in n or "<(bool)1>" in n else "merge_false"] += 1
        if K2_RE.search(n):
            c["k2"] += 1
        if K3_RE.search(n):
            c["k3"] += 1
    by = {}
    for e in evs:
        by.setdefault(e[3], []).append(e)
    after_merge, after_k3 = {}, {}
    merges = follow = early = 0
    gaps = []
    for lst in by.values():
        for i, (ts, dur, n, _) in enumerate(lst):
            if MERGE_RE.search(n):
                merges += 1
                if i + 1 < len(lst):
                    nx = lst[i + 1]
                    after_merge[nx[2][:70]] = after_merge.get(nx[2][:70], 0) + 1
                    if K2_RE.search(nx[2]):
                        follow += 1
                        early += int(nx[0] < ts + dur)
                        gaps.append(round(nx[0] - (ts + dur), 2))
            if K3_RE.search(n) and i + 1 < len(lst):
                a = lst[i + 1][2][:70]
                after_k3[a] = after_k3.get(a, 0) + 1
    gaps.sort()
    top = lambda d: sorted(d.items(), key=lambda kv: -kv[1])[:5]  # noqa: E731
    ar_after_k3 = sum(v for k, v in after_k3.items() if AR_RE.search(k))
    return {"kernels": len(evs), "per_step": {k: round(v / STEPS, 2) for k, v in c.items()}, "merges": merges,
            "merge_then_k2": follow, "k2_starts_before_merge_ends": early,
            "k2_start_minus_merge_end_us_p10_p50_p90": [gaps[int(q * (len(gaps) - 1))] for q in (0.1, 0.5, 0.9)]
            if gaps else None,
            "after_merge": top(after_merge), "after_k3": top(after_k3), "ar_after_k3": ar_after_k3,
            "after_k3_is_ar_share": round(ar_after_k3 / max(1, sum(after_k3.values())), 3)}


def first_div(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) == len(b) else n


def compare(a, b):
    out = {}
    for c in a:
        tok = all(x["tokens"] == y["tokens"] for x, y in zip(a[c], b[c])) and len(a[c]) == len(b[c])
        lp = all(x["logprobs"] == y["logprobs"] for x, y in zip(a[c], b[c])) and len(a[c]) == len(b[c])
        per = []
        for x, y in zip(a[c], b[c]):
            d = first_div(x["tokens"], y["tokens"])
            upto = min(d + 1, len(x["logprobs"]), len(y["logprobs"]))
            dl = [abs(p - q) for p, q in zip(x["logprobs"][:upto], y["logprobs"][:upto])]
            per.append({"first_token_div": d if d < len(x["tokens"]) else None,
                        "prefill_lp_delta": abs(x["logprobs"][0] - y["logprobs"][0]) if x["logprobs"] and
                        y["logprobs"] else None,
                        "max_abs_dlp_common_prefix": max(dl) if dl else 0.0,
                        "n_lp_unequal_common_prefix": sum(int(v != 0.0) for v in dl)})
        out[c] = {"tokens_bitwise": tok, "logprobs_bitwise": lp, "decode_tokens": [len(x["tokens"]) - 1 for x in a[c]],
                  "per_prompt": per}
    return out


def bitwise(cmpr):
    return all(v["tokens_bitwise"] and v["logprobs_bitwise"] for v in cmpr.values())


def within_spread(cmpr, ctrl):
    """Labeled spread test (NOT bitwise): per prompt, the arm diverges no earlier and deviates no more than A1b."""
    ok = True
    for c in cmpr:
        for p, q in zip(cmpr[c]["per_prompt"], ctrl[c]["per_prompt"]):
            pd = p["first_token_div"] if p["first_token_div"] is not None else 10 ** 9
            qd = q["first_token_div"] if q["first_token_div"] is not None else 10 ** 9
            ok &= pd >= qd and p["max_abs_dlp_common_prefix"] <= q["max_abs_dlp_common_prefix"] + 0.0
    return bool(ok)


def scan_log(path):
    r = {"cache_load_ranks": [], "cache_save_ranks": [], "capture_bs_by_rank": {}, "eager_fallback_lines": 0,
         "autotuner_warmup_lines": [], "cache_load_lines": 0}
    if not path.exists():
        return r
    rk = re.compile(r"\[RANK (\d+)\]")
    bs_re = re.compile(r"Run generation-only CUDA graph capture .*?batch size=(\d+)")
    loads, saves, caps = set(), set(), {}
    with open(path, errors="replace") as fh:
        for line in fh:
            m = rk.search(line)
            rank = int(m.group(1)) if m else -1
            if "Loading cache from" in line:
                loads.add(rank)
                r["cache_load_lines"] += 1
            if "Saving cache to" in line:
                saves.add(rank)
            b = bs_re.search(line)
            if b:
                caps.setdefault(str(rank), set()).add(int(b.group(1)))
                r.setdefault("capture_lines_by_bs", {})
                r["capture_lines_by_bs"][b.group(1)] = r["capture_lines_by_bs"].get(b.group(1), 0) + 1
            if "falling back to eager" in line.lower() or "fall back to eager" in line.lower():
                r["eager_fallback_lines"] += 1
            if "Running autotuner warmup" in line and len(r["autotuner_warmup_lines"]) < 8:
                r["autotuner_warmup_lines"].append(line.strip()[-160:])
    r["cache_load_ranks"] = sorted(loads)
    r["cache_save_ranks"] = sorted(saves)
    r["capture_bs_by_rank"] = {k: sorted(v) for k, v in caps.items()}
    return r


def run_arm(ckpt, name, env):
    e = dict(os.environ)
    e.update(env)
    e["TLLM_AUTOTUNER_CACHE_PATH"] = str(CACHE)
    e["TLLM_LOG_LEVEL"] = "INFO"
    logf = ART / f"phase_{name}.log"
    t0 = time.time()
    last = t0
    n = 0
    print(f"[{time.strftime('%H:%M:%S')}] arm {name} start env={env} cache_exists={CACHE.exists()}", flush=True)
    with open(logf, "w") as fh:
        p = subprocess.Popen([sys.executable, __file__, ckpt, "--phase", name], env=e, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1)
        for line in p.stdout:
            fh.write(line)
            n += 1
            if line.startswith(f"[{name}]") or (LOG_PATTERNS.search(line) and len(line) < 400):
                print(f"  {name}| {line.rstrip()[:300]}", flush=True)
            if time.time() - last > 60:
                print(f"[{time.strftime('%H:%M:%S')}] arm {name}: {n} log lines, {time.time() - t0:.0f}s", flush=True)
                last = time.time()
        p.wait()
    rc = {"rc": p.returncode, "s": round(time.time() - t0)}
    print(f"[{time.strftime('%H:%M:%S')}] arm {name}: {rc}", flush=True)
    return rc


def main():
    ckpt = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "--phase":
        run_phase(ckpt, sys.argv[3])
        return
    ART.mkdir(parents=True, exist_ok=True)
    if CACHE.exists():
        CACHE.unlink()
    print(f"settings: {llm_kwargs()} + moe_config.disable_finalize_fusion=True; cache {CACHE}", flush=True)
    rc = {}
    for name, env in ARMS.items():
        rc[name] = run_arm(ckpt, name, env)
        if name == WRITER and not CACHE.exists():
            print("WARNING: writer arm did not create the autotuner cache file", flush=True)
    phases = {n: (json.loads((ART / f"phase_{n}.json").read_text()) if (ART / f"phase_{n}.json").exists() else None)
              for n in ARMS}
    logs = {n: scan_log(ART / f"phase_{n}.log") for n in ARMS}
    summary = {"settings": {**llm_kwargs(), "moe_config.disable_finalize_fusion": True},
               "cache_path": str(CACHE), "arms_rc": rc, "logs": logs}
    all_rc = all(v["rc"] == 0 for v in rc.values()) and all(phases.values())
    # G1: cache-load line on all 4 ranks in every loading arm.
    g1 = {n: logs[n]["cache_load_ranks"] for n in ARMS if n != WRITER}
    # Lines without a [RANK n] tag (logger rank unset) count as -1; then require >= 4 occurrences instead.
    g1_ok = all(set(v) >= {0, 1, 2, 3} or (v == [-1] and logs[n]["cache_load_lines"] >= 4) for n, v in g1.items())
    # G2: prefill first-token logprob (and first token) equal across arms.
    first = {n: {c: [(r["tokens"][0], r["logprobs"][0]) for r in rows] for c, rows in ph.items()}
             for n, ph in phases.items() if ph}
    ref_first = first.get("A1")
    g2 = {n: (v == ref_first) for n, v in first.items()}
    g2_ok = ref_first is not None and all(g2.values())
    cmp = {}
    if phases.get("A1"):
        for n in ("W", "A1b", "A2", "A3"):
            if phases.get(n):
                cmp[f"{n}_vs_A1"] = compare(phases["A1"], phases[n])
    if phases.get("A4") and phases.get("A5"):
        cmp["A5_vs_A4"] = compare(phases["A4"], phases["A5"])
    if phases.get("A1") and phases.get("A4"):
        cmp["A4_vs_A1_info"] = compare(phases["A1"], phases["A4"])
    g3_ok = "A1b_vs_A1" in cmp and bitwise(cmp["A1b_vs_A1"])
    judged = {}
    if g3_ok:
        for k in ("A2_vs_A1", "A3_vs_A1", "A5_vs_A4"):
            judged[k] = {"mode": "bitwise", "pass": k in cmp and bitwise(cmp[k])}
        path = "primary (shared cache, A1 == A1b bitwise)"
    else:
        prefill_diff = "A1b_vs_A1" in cmp and any(
            p["prefill_lp_delta"] not in (0.0, None) for v in cmp["A1b_vs_A1"].values() for p in v["per_prompt"])
        path = ("fallback: A1/A1b SPREAD comparison, not bitwise; nondeterminism is in "
                + ("PREFILL (teacher forcing cannot help)" if prefill_diff else "DECODE only"))
        ctrl = cmp.get("A1b_vs_A1")
        for k in ("A2_vs_A1", "A3_vs_A1"):
            judged[k] = {"mode": "spread_vs_A1b", "pass": bool(ctrl) and k in cmp and within_spread(cmp[k], ctrl)}
        judged["A5_vs_A4"] = {"mode": "spread_vs_A1b",
                              "pass": bool(ctrl) and "A5_vs_A4" in cmp and within_spread(cmp["A5_vs_A4"], ctrl)}
    traces = {}
    route_ok = True
    for w in ("decode_c1", "decode_c4"):
        files = sorted(glob.glob(str(ART / "traces" / w / "*.json*")))
        res = {"files": len(files), "ranks": {}}
        for f in files:
            m = re.search(r"rank-(\d+)", os.path.basename(f))
            evs, gk = load(f)
            rr = reduce_rank(evs)
            rr.update(gk)
            res["ranks"][int(m.group(1)) if m else -1] = rr
        ok_r = {}
        for r, rr in res["ranks"].items():
            ps = rr["per_step"]
            ok_r[r] = (ps.get("merge_true", 0) >= 38 and ps.get("merge_false", 1) == 0
                       and rr["merge_then_k2"] == rr["merges"] and rr["k2_starts_before_merge_ends"] > 0
                       and rr["ar_after_k3"] >= rr["merges"])
        res["route_ok_by_rank"] = ok_r
        res["route_ok"] = len(files) >= 4 and all(ok_r.values())
        route_ok &= res["route_ok"]
        traces[w] = res
        print(w, json.dumps(res, default=str)[:4000], flush=True)
    caps = {n: logs[n]["capture_bs_by_rank"] for n in ARMS}
    def _cap(n):
        v = caps[n]
        if all(set(v.get(str(r), [])) >= {1, 2, 4, 8} for r in range(4)):
            return True
        by = logs[n].get("capture_lines_by_bs", {})
        return list(v) == ["-1"] and all(by.get(str(b), 0) >= 4 for b in (1, 2, 4, 8))
    cap_ok = all(_cap(n) for n in ARMS) and all(logs[n]["eager_fallback_lines"] == 0 for n in ARMS)
    judged_ok = bool(judged) and all(v["pass"] for v in judged.values())
    passed = all_rc and g1_ok and g2_ok and g3_ok and judged_ok and route_ok and cap_ok
    summary.update({"gates": {"all_rc": all_rc, "G1_cache_load_all_ranks": g1_ok, "G1_detail": g1,
                              "G2_prefill_first_token_equal": g2_ok, "G2_detail": g2, "first_token": first,
                              "G3_A1_eq_A1b_bitwise": g3_ok, "path": path, "judged": judged,
                              "route_ok": route_ok, "graph_capture_ok": cap_ok, "capture_bs": caps},
                    "compare": cmp, "traces": traces})
    (ART / "server_audit_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    short_cmp = {k: {c: {"tok": v["tokens_bitwise"], "lp": v["logprobs_bitwise"],
                         "div": [p["first_token_div"] for p in v["per_prompt"]],
                         "maxdlp": [round(p["max_abs_dlp_common_prefix"], 5) for p in v["per_prompt"]]}
                     for c, v in cv.items()} for k, cv in cmp.items()}
    print(json.dumps({"passed": bool(passed), "path": path, "gates": {
        "all_rc": all_rc, "G1": g1_ok, "G2": g2_ok, "G3": g3_ok, "judged": judged, "route_ok": route_ok,
        "graph_capture_ok": cap_ok}, "arms_rc": rc, "compare": short_cmp,
        "route": {w: {"ok": t["route_ok"], "by_rank": t["route_ok_by_rank"]} for w, t in traces.items()},
        "reference": "A1 (default, same tree, shared autotuner cache, deterministic MoE)"}, default=str))


if __name__ == "__main__":
    main()
