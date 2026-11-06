"""Kernel-name trace of one c1 prefill step + a few decode steps (idea 94, acceptance 1/3). NOT timing evidence.

The frozen profile adapter cannot be copied whole (common.py is not readable in full), so this follows method 797
with the LLM API in-process: the protocol's server settings (as in experiments/ascent/ar_trace_job.py, from proxy
job 230) deep-merged with this tree's experiments/ascent-config/engine.json; prime a 32768-token prefix, warm up,
then LLM.start_profile(GPU) around one request with a fresh 2048-token suffix (34816 prompt tokens, 32768 cached)
and a few output tokens. Per rank: count lowPrecision* kernels, ncclDevKernel_AllReduce*, and
allreduce_fusion_kernel_oneshot_lamport per executor step, split by the step[EXTEND/DECODE] record_function scopes.
Usage: python trace.py <checkpoint>. Last stdout line: JSON summary.
"""
import glob
import gzip
import json
import os
import random
import re
import sys
import threading
import time
import traceback
from pathlib import Path

SERVER_SETTINGS = {
    "context_parallel_size": 1, "cuda_graph_config": {"batch_sizes": [1, 2, 4, 8], "enable_padding": True},
    "enable_attention_dp": False, "enable_chunked_prefill": True,
    "kv_cache_config": {"enable_block_reuse": True, "free_gpu_memory_fraction": 0.5, "tokens_per_block": 128},
    "max_batch_size": 8, "max_num_tokens": 8192, "max_seq_len": 36864, "moe_expert_parallel_size": 4,
    "pipeline_parallel_size": 1, "tensor_parallel_size": 4, "trust_remote_code": True}
PREFIX, SUFFIX, OUT_TOKENS, PROFILE_STEPS = 32768, 2048, 6, 6
LP_RE = re.compile(r"lowPrecision", re.I)
NCCL_AR_RE = re.compile(r"ncclDevKernel_AllReduce|ncclKernel_AllReduce")
LAMPORT_RE = re.compile(r"allreduce_fusion_kernel_oneshot_lamport|oneshot_lamport")
OTHER_AR_RE = re.compile(r"allreduce|all_reduce|twoshot|nccl", re.I)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def deep_merge(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def heartbeat(stage):
    def run():
        while True:
            time.sleep(60)
            log(f"heartbeat {stage[0]}")
    threading.Thread(target=run, daemon=True).start()


def load_kernels(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        data = json.load(f)
    evs = data.get("traceEvents", data) if isinstance(data, dict) else data
    ks = [e for e in evs if isinstance(e, dict) and e.get("ph") == "X" and e.get("cat") == "kernel"]
    ks.sort(key=lambda e: e["ts"])
    return ks


STEP_RE = re.compile(r"^(step\[(EXTEND|DECODE)[^\]]*\])")


def load_events(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        data = json.load(f)
    evs = data.get("traceEvents", data) if isinstance(data, dict) else data
    return [e for e in evs if isinstance(e, dict) and e.get("ph") == "X"]


def split_steps(evs, ks):
    """Assign each kernel to its executor step via the step[...] record_function scopes (py_executor
    _build_step_scope_label). Primary: kernel correlation -> CUDA runtime launch -> enclosing CPU user_annotation.
    Fallback: GPU-side gpu_user_annotation span containing the kernel start."""
    cpu_steps = sorted([e for e in evs if e.get("cat") == "user_annotation" and STEP_RE.match(e.get("name", ""))],
                       key=lambda e: e["ts"])
    gpu_steps = sorted([e for e in evs if e.get("cat") == "gpu_user_annotation" and STEP_RE.match(e.get("name", ""))],
                       key=lambda e: e["ts"])
    launches = {}
    for e in evs:
        if e.get("cat") in ("cuda_runtime", "cuda_driver"):
            c = (e.get("args") or {}).get("correlation")
            if c is not None:
                launches[c] = e
    assign, method = [], {"correlation": 0, "gpu_annotation": 0, "none": 0}
    for k in ks:
        lab = None
        ln = launches.get((k.get("args") or {}).get("correlation"))
        if ln is not None:
            for s in cpu_steps:
                if s.get("tid") == ln.get("tid") and s["ts"] <= ln["ts"] <= s["ts"] + s["dur"]:
                    lab = (s["ts"], STEP_RE.match(s["name"]).group(1))
            if lab:
                method["correlation"] += 1
        if lab is None:
            for s in gpu_steps:
                if s["ts"] <= k["ts"] <= s["ts"] + s["dur"]:
                    lab = (s["ts"], STEP_RE.match(s["name"]).group(1))
            if lab:
                method["gpu_annotation"] += 1
        if lab is None:
            method["none"] += 1
        assign.append(lab)
    return assign, method, len(cpu_steps), len(gpu_steps)


def reduce_rank(path):
    evs = load_events(path)
    ks = sorted([e for e in evs if e.get("cat") == "kernel"], key=lambda e: e["ts"])
    assign, method, n_cpu_steps, n_gpu_steps = split_steps(evs, ks)
    # group kernels by step. A step's bare scope and its ' sample' scope share the label body; key by
    # label body + order of first appearance of a new EXTEND/DECODE scope start.
    groups, order = {}, []
    for k, lab in zip(ks, assign):
        key = lab if lab else (None, "unassigned")
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(k)
    order.sort(key=lambda key: (key[0] is None, key[0] or 0))

    def count(lst):
        c = {"lowPrecision": 0, "ncclDevKernel_AllReduce": 0, "oneshot_lamport": 0, "other_ar": {}}
        lp_names = {}
        for e in lst:
            n = e["name"]
            if LP_RE.search(n):
                c["lowPrecision"] += 1
                lp_names[n[:80]] = lp_names.get(n[:80], 0) + 1
            elif NCCL_AR_RE.search(n):
                c["ncclDevKernel_AllReduce"] += 1
            elif LAMPORT_RE.search(n):
                c["oneshot_lamport"] += 1
            elif OTHER_AR_RE.search(n):
                c["other_ar"][n[:80]] = c["other_ar"].get(n[:80], 0) + 1
        c["lp_kernel_names"] = lp_names
        return c

    steps = []
    for key in order:
        lst = groups[key]
        c = count(lst)
        c["lp_calls"] = sum(v for n, v in c["lp_kernel_names"].items() if "TwoShot" in n)
        c["label"] = key[1]
        c["kernels"] = len(lst)
        c["nccl_ar_us"] = [round(e["dur"], 1) for e in lst if NCCL_AR_RE.search(e["name"])][:100]
        c["lp_twoshot_us"] = [round(e["dur"], 1) for e in lst if LP_RE.search(e["name"]) and "TwoShot" in e["name"]][:100]
        c["lp_twoshot_grid"] = sorted({str((e.get("args") or {}).get("grid")) for e in lst
                                       if LP_RE.search(e["name"]) and "TwoShot" in e["name"]})
        steps.append(c)
    ext = [s for s in steps if s["label"].startswith("step[EXTEND")]
    dec = [s for s in steps if s["label"].startswith("step[DECODE")]
    pc = ext[0] if ext else {}
    return {"kernels": len(ks), "assign_method": method, "cpu_step_scopes": n_cpu_steps,
            "gpu_step_scopes": n_gpu_steps, "steps": steps, "prefill_window": pc,
            "decode_steps_summary": [{"label": s["label"], "lowPrecision": s["lowPrecision"],
                                      "ncclDevKernel_AllReduce": s["ncclDevKernel_AllReduce"],
                                      "oneshot_lamport": s["oneshot_lamport"]} for s in dec],
            "decode_window": {"lowPrecision": sum(s["lowPrecision"] for s in dec),
                              "ncclDevKernel_AllReduce": sum(s["ncclDevKernel_AllReduce"] for s in dec),
                              "oneshot_lamport": sum(s["oneshot_lamport"] for s in dec)}}


def main():
    ckpt = sys.argv[1]
    art = Path(os.environ.get("AUTO_DEV_ARTIFACTS", "artifacts"))
    tdir = art / "lp_trace"
    tdir.mkdir(parents=True, exist_ok=True)
    stage = ["init"]
    heartbeat(stage)
    eng = {}
    p = Path.cwd() / "experiments" / "ascent-config" / "engine.json"
    if p.exists():
        eng = json.loads(p.read_text())
    settings = deep_merge(SERVER_SETTINGS, {k: v for k, v in eng.items() if k not in ("env", "serve_args")})
    log(f"settings: {settings}")
    cfg = json.loads((Path(ckpt) / "config.json").read_text())
    vocab = int(cfg.get("vocab_size", 100000))
    rng = random.Random(1234)
    lo, hi = 1000, min(vocab - 1000, 120000)
    prefix = [rng.randrange(lo, hi) for _ in range(PREFIX)]
    lead = list(range(lo + 7, lo + 7 + 64 * 37, 37))  # a distinct first suffix token per request

    def prompt(i):
        r = random.Random(77 + i)
        return prefix + [lead[i]] + [r.randrange(lo, hi) for _ in range(SUFFIX - 1)]

    from tensorrt_llm import LLM, SamplingParams
    stage[0] = "load"
    t0 = time.time()
    llm = LLM(model=ckpt, **settings)
    log(f"LLM up in {time.time()-t0:.0f}s")
    sp1 = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
    spn = SamplingParams(max_tokens=OUT_TOKENS, temperature=0.0, ignore_eos=True)
    summary = {"settings": settings}
    try:
        stage[0] = "prime"
        o = llm.generate([prompt(0)], sp1)
        log("prime done")
        stage[0] = "warmup"
        for i in (1, 2):
            o = llm.generate([prompt(i)], spn)
            log(f"warm {i}: cached={getattr(o[0], 'cached_tokens', None)}")
        stage[0] = "profile"
        llm.start_profile(output_dir=str(tdir), num_steps=PROFILE_STEPS, activities=["CPU", "GPU"])
        o = llm.generate([prompt(3)], spn)
        try:
            llm.stop_profile()
        except Exception as exc:  # noqa: BLE001
            log(f"stop_profile: {exc!r}")
        summary["profiled_request"] = {"prompt_tokens": len(prompt(3)),
                                       "cached_tokens": getattr(o[0], "cached_tokens", None),
                                       "out_tokens": len(o[0].outputs[0].token_ids)}
    except Exception as exc:  # noqa: BLE001
        summary["error"] = repr(exc)[:800]
        log(traceback.format_exc())
    finally:
        stage[0] = "shutdown"
        try:
            llm.shutdown()
        except Exception as exc:  # noqa: BLE001
            log(f"shutdown: {exc!r}")
    time.sleep(5)
    files = sorted(glob.glob(str(tdir / "**" / "*.json*"), recursive=True))
    log(f"trace files: {files}")
    per_rank = {}
    for f in files:
        m = re.search(r"rank[-_]?(\d+)", os.path.basename(f))
        r = m.group(1) if m else os.path.basename(f)
        try:
            per_rank[r] = reduce_rank(f)
        except Exception as exc:  # noqa: BLE001
            per_rank[r] = {"error": repr(exc)[:300]}
        log(f"rank {r}: {json.dumps(per_rank[r])[:3000]}")
    summary["per_rank"] = per_rank
    r0 = per_rank.get("0") or (next(iter(per_rank.values())) if per_rank else {})
    pw = r0.get("prefill_window", {})
    summary["rank0_prefill"] = {"lp_calls": pw.get("lp_calls"), "nccl_allreduce": pw.get("ncclDevKernel_AllReduce"),
                                "lamport_before_decode": pw.get("oneshot_lamport")}
    dw = r0.get("decode_window", {})
    summary["rank0_decode"] = dw
    summary["rank0_steps"] = [s["label"] for s in r0.get("steps", [])]
    # Per-rank AR census (landing 51 / EP2xTP2 re-check): any reduce-scatter / all-gather / other AR kernel shows
    # up in other_ar; LP two-shot and NCCL AR durations show whether the [2048,5120] / tail shapes changed.
    summary["layout"] = {k: settings.get(k) for k in ("tensor_parallel_size", "moe_tensor_parallel_size",
                                                      "moe_expert_parallel_size", "allreduce_strategy")}
    census = {}
    for r, d in per_rank.items():
        pwr, dwr = d.get("prefill_window", {}) or {}, d.get("decode_window", {}) or {}
        ln = sorted(pwr.get("lp_twoshot_us", []))
        nn = sorted(pwr.get("nccl_ar_us", []))
        census[r] = {"prefill_lp_calls": pwr.get("lp_calls"), "prefill_nccl_ar": pwr.get("ncclDevKernel_AllReduce"),
                     "prefill_lamport": pwr.get("oneshot_lamport"), "prefill_other_ar": pwr.get("other_ar"),
                     "prefill_lp_us_median": ln[len(ln) // 2] if ln else None,
                     "prefill_lp_twoshot_grid": pwr.get("lp_twoshot_grid"),
                     "prefill_nccl_us_median": nn[len(nn) // 2] if nn else None,
                     "decode": dwr,
                     "decode_other_ar": [s.get("other_ar") for s in d.get("steps", [])
                                         if str(s.get("label", "")).startswith("step[DECODE") and s.get("other_ar")][:4]}
    summary["per_rank_census"] = census
    summary["passed"] = (bool(pw) and pw.get("lp_calls") == 40 and pw.get("ncclDevKernel_AllReduce") == 40
                         and dw.get("lowPrecision", 1) == 0)
    (art / "lp_trace_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
