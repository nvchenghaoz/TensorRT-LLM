"""Branch 145 (idea 245 Q1) acceptance item 3: ONE 4-GPU in-process run (TP4/EP4, protocol server settings,
CUDA graphs [1,2,4,8] padded), not timing evidence.

Phases (each a fresh subprocess so every env switch is read at worker start / graph capture):
  default        : Q1 on. Greedy generation of fixed synthetic prompts at batch 1 and 4 (token ids + top-1 logprobs),
                   then Kineto decode windows c1 and c4 (LLM.start_profile, 40 steps).
  attn_pdl0      : TRTLLM_CSA2_ATTN_PDL=0 (Q1 off)          -> tokens and logprobs must equal 'default' bitwise
  pdl0           : TRTLLM_ENABLE_PDL=0 (no PDL anywhere)    -> tokens and logprobs must equal 'default' bitwise
  cake0          : TRTLLM_CSA2_CAKE_DECODE=0 (split-sink)   -> must run; no CakeMergeF32Kernel; agreement reported
Trace reduction on every rank: per decode step counts of Cake prep / decode_body / CakeMergeF32Kernel<true|false> /
K2' / K3'; on each rank's streams, the kernel right after every merge (must be K2'), K2'.start < merge.end share,
the kernel right after K3' (the attention all-reduce).
Usage: server_audit.py <checkpoint>. Last stdout line: JSON.
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
ART = Path(os.environ.get("AUTO_DEV_ARTIFACTS", "artifacts")).resolve()
MERGE_RE = re.compile(r"CakeMergeF32Kernel")
K2_RE = re.compile(r"csa2_lowm_wo_a")
K3_RE = re.compile(r"csa2_mxfp8_low_m_gemm")
BODY_RE = re.compile(r"decode_body|kernel_sm120_dsv41_mixed_sparse_mla")
PREP_RE = re.compile(r"_cake_index_prep")
AR_RE = re.compile(r"allreduce|all_reduce|lamport|oneshot|twoshot|nccl", re.I)
PHASES = {
    "default": {},
    "attn_pdl0": {"TRTLLM_CSA2_ATTN_PDL": "0"},
    "pdl0": {"TRTLLM_ENABLE_PDL": "0"},
    "cake0": {"TRTLLM_CSA2_CAKE_DECODE": "0"},
}


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def run_phase(ckpt, name):
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

    import random
    t0 = time.time()
    llm = LLM(model=ckpt, tensor_parallel_size=4, moe_expert_parallel_size=4, pipeline_parallel_size=1,
              enable_attention_dp=False, enable_chunked_prefill=True,
              kv_cache_config=KvCacheConfig(enable_block_reuse=True, free_gpu_memory_fraction=0.5,
                                            tokens_per_block=128),
              cuda_graph_config=CudaGraphConfig(batch_sizes=[1, 2, 4, 8], enable_padding=True),
              max_batch_size=8, max_num_tokens=8192, max_seq_len=36864, trust_remote_code=True)
    log(f"[{name}] LLM up in {time.time() - t0:.0f}s")
    rng = random.Random(245)
    prompts = [[rng.randrange(1000, 60000) for _ in range(768)] for _ in range(5)]
    sp = SamplingParams(max_tokens=48, temperature=0.0, ignore_eos=True, logprobs=1)
    res = {}
    for c, group in ((1, [prompts[0]]), (4, prompts[1:5])):
        outs = llm.generate(group, sp)
        res[f"c{c}"] = [{"tokens": list(o.outputs[0].token_ids),
                         "logprobs": [float(next(iter(d.values())).logprob) if isinstance(d, dict) else float(d)
                                      for d in (o.outputs[0].logprobs or [])]} for o in outs]
        log(f"[{name}] c{c}: {[r['tokens'][:6] for r in res[f'c{c}']]}")
    if name == "default":
        root = ART / "traces"
        for c in (1, 4):
            d = root / f"decode_c{c}"
            d.mkdir(parents=True, exist_ok=True)
            spl = SamplingParams(max_tokens=1200, temperature=0.0, ignore_eos=True)
            futs = [llm.generate_async([rng.randrange(1000, 60000) for _ in range(512)], spl) for _ in range(c)]
            time.sleep(3.0)
            llm.start_profile(output_dir=str(d), num_steps=STEPS, activities=["GPU"])
            log(f"[{name}] decode_c{c}: profile started")
            [f.result() for f in futs]
            try:
                llm.stop_profile()
            except Exception as exc:  # noqa: BLE001
                log(f"stop_profile: {exc!r}")
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
    return evs


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
    return {"per_step": {k: round(v / STEPS, 2) for k, v in c.items()}, "merges": merges,
            "merge_then_k2": follow, "k2_starts_before_merge_ends": early,
            "k2_start_minus_merge_end_us_p10_p50_p90": [gaps[int(q * (len(gaps) - 1))] for q in (0.1, 0.5, 0.9)]
            if gaps else None,
            "after_merge": top(after_merge), "after_k3": top(after_k3),
            "after_k3_is_ar_share": round(sum(v for k, v in after_k3.items() if AR_RE.search(k))
                                          / max(1, sum(after_k3.values())), 3)}


def compare(a, b):
    out = {}
    for c in a:
        tok = all(x["tokens"] == y["tokens"] for x, y in zip(a[c], b[c]))
        lp = all(x["logprobs"] == y["logprobs"] for x, y in zip(a[c], b[c]))
        agree = [sum(int(p == q) for p, q in zip(x["tokens"], y["tokens"])) / max(1, len(x["tokens"]))
                 for x, y in zip(a[c], b[c])]
        out[c] = {"tokens_bitwise": tok, "logprobs_bitwise": lp, "token_agreement": [round(v, 3) for v in agree]}
    return out


def main():
    ckpt = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "--phase":
        run_phase(ckpt, sys.argv[3])
        return
    ART.mkdir(parents=True, exist_ok=True)
    rc = {}
    for name, env in PHASES.items():
        e = dict(os.environ)
        e.update(env)
        t0 = time.time()
        p = subprocess.run([sys.executable, __file__, ckpt, "--phase", name], env=e)
        rc[name] = {"rc": p.returncode, "s": round(time.time() - t0)}
        log(f"phase {name}: {rc[name]}")
    phases = {}
    for name in PHASES:
        f = ART / f"phase_{name}.json"
        phases[name] = json.loads(f.read_text()) if f.exists() else None
    summary = {"phases_rc": rc, "compare": {}}
    ok = all(v["rc"] == 0 for v in rc.values()) and phases["default"] is not None
    for name in ("attn_pdl0", "pdl0", "cake0"):
        if phases.get(name) and phases["default"]:
            cmpr = compare(phases["default"], phases[name])
            summary["compare"][name] = cmpr
            if name != "cake0":
                ok &= all(v["tokens_bitwise"] and v["logprobs_bitwise"] for v in cmpr.values())
    traces = {}
    for w in ("decode_c1", "decode_c4"):
        files = sorted(glob.glob(str(ART / "traces" / w / "*.json*")))
        res = {"files": len(files), "ranks": {}}
        for f in files:
            m = re.search(r"rank-(\d+)", os.path.basename(f))
            res["ranks"][int(m.group(1)) if m else -1] = reduce_rank(load(f))
        r0 = res["ranks"].get(0) or (next(iter(res["ranks"].values())) if res["ranks"] else {})
        ps = r0.get("per_step", {}) if r0 else {}
        res["route_ok"] = bool(files) and ps.get("merge_true", 0) >= 38 and ps.get("merge_false", 1) == 0 \
            and r0.get("merge_then_k2", 0) == r0.get("merges", -1) and r0.get("k2_starts_before_merge_ends", 0) > 0
        ok &= res["route_ok"]
        traces[w] = res
        log(w, json.dumps(res)[:3000])
    summary["traces"] = traces
    hits = []
    pat = re.compile(r"cuda graph|capture", re.I)
    for lf in sorted(glob.glob(str(ART / "**" / "*.log"), recursive=True)):
        try:
            with open(lf, errors="replace") as fh:
                for line in fh:
                    if pat.search(line):
                        hits.append(f"{os.path.basename(lf)}: {line.strip()[:200]}")
        except OSError:
            pass
    summary["log_hits"] = hits[:30]
    (ART / "server_audit_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({"passed": bool(ok), "phases_rc": rc, "compare": summary["compare"],
                      "traces": {w: {"route_ok": t["route_ok"], "rank0": t["ranks"].get(0)} for w, t in
                                 traces.items()}}, default=str))


if __name__ == "__main__":
    main()
