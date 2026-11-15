#!/usr/bin/env python3
"""BCG (idea 19) exactness proof with pinned AutoTuner tactics (plans 679 / 701).

ONE GPU job, four sequential processes, each with its OWN explicit TLLM_AUTOTUNER_CACHE_PATH:
  A  main eager (no engine overlay; branch code is inert without a prefill graph backend),
     fresh cache file FA: it tunes and saves at autotune() exit.
  B  BCG overlay, UNPINNED, fresh cache file FB (mandatory: FB key set must equal FA's).
  C  BCG overlay loading a COPY of FA. Passes: N (replay allowed), E (BreakableCUDAGraphRunner
     .has_graph gated off -> eager in the same process), S (replay allowed, shuffled item order
     = mixed bucket order), then a repeated-call sequence and the per-step timing diag.
  D  main eager loading a COPY of FA (pinned A/A control), plus the per-step timing diag.
Every output is compared bitwise (token ids + top-20 logprobs as int64 views of the float).
The cache files C and D save at exit are diffed against FA, every search_cache lookup
(phase, tuning mode, op, profile, hit, runner, tactic) is recorded in-process, and the
'[AutoTuner] ... fallback tactic' warnings are grepped from each process log.

usage: pin_proof.py            (orchestrator; last stdout line is the result JSON)
       pin_proof.py proc NAME OVERLAY(0|1) CACHE_PATH OUT_JSON PLAN
"""
import hashlib
import json
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
WORKSPACE_BYTES = 33554432  # CUBLAS_WORKSPACE_SIZE (cpp/include/tensorrt_llm/common/cudaUtils.h)
SHARED_ROOT = ("/home/scratch.chengzhang/trtllm-dev-gemma4-31b-pf-20261114-163048/shared/"
               "host_runtime_bcg_pinproof")
BASE_MAIN = "e58d0587"  # main this branch is synced to
SRC_FILES = [
    "tensorrt_llm/_torch/pyexecutor/engine/runners/common.py",
    "tensorrt_llm/_torch/pyexecutor/engine/runners/decoder/graphs.py",
    "tensorrt_llm/_torch/pyexecutor/breakable_cuda_graph_runner.py",
    "tensorrt_llm/_torch/models/modeling_gemma4.py",
    "tensorrt_llm/_torch/modules/gemma4/fused_qkv.py",
    "tensorrt_llm/_torch/kernels/fused_ops/gate_up_gelu_fp4_chunked.py",
    "tensorrt_llm/_torch/autotuner.py",
    "experiments/ascent-config/engine.json",
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance():
    out = {}
    for args in (["git", "rev-parse", "HEAD"], ["git", "rev-parse", "HEAD^{tree}"]):
        try:
            out[" ".join(args[1:])] = subprocess.run(
                args, cwd=REPO, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception as e:  # noqa: BLE001
            out[" ".join(args[1:])] = f"unavailable: {e!r}"
    out["src_sha256"] = {f: (sha256_file(os.path.join(REPO, f))
                             if os.path.exists(os.path.join(REPO, f)) else None)
                         for f in SRC_FILES}
    return out


# ----------------------------------------------------------------------------------------------
# stress set (identical in every process)
# ----------------------------------------------------------------------------------------------
def build_items(Prompts):
    """Return [(item_id, label, [prompt, ...])]. Same prompts as bcg_ab.py mode_arm (job 259)
    plus singles/batches at every captured size [300, 600, 900, 1000, 1200, 1500, 1800, 2000]."""
    pf = Prompts(777)
    groups = [(300, 64), (1000, 16), (2000, 8), (4000, 4), (8000, 2)]
    lists = [[pf.make(isl) for _ in range(n)] for isl, n in groups]
    mixed = []
    i = 0
    while any(lists):
        lst = lists[i % len(lists)]
        if lst:
            mixed.append(lst.pop())
        i += 1
    items = [("mixed", "mixed94", mixed)]
    ps = Prompts(4242)
    k = 0
    for isl in (300, 1000, 2000, 4000, 8000):
        for _ in range(20):
            items.append((f"s{k}", f"1x{isl}", [ps.make(isl)]))
            k += 1
    for L in (310, 901, 1010, 2047, 257, 333, 1500):
        items.append((f"s{k}", f"1x{L}", [ps.make(L)]))
        k += 1
    small = [(4, 300), (2, 300), (2, 1000)]
    for j, (n, isl) in enumerate(small):
        items.append((f"b{j}", f"{n}x{isl}", [ps.make(isl) for _ in range(n)]))
    px = Prompts(4343)
    for isl in (600, 900, 1200, 1800):
        for _ in range(2):
            items.append((f"s{k}", f"1x{isl}", [px.make(isl)]))
            k += 1
    for j, (n, isl) in enumerate([(3, 300), (5, 300), (6, 300), (2, 900)], start=len(small)):
        items.append((f"b{j}", f"{n}x{isl}", [px.make(isl) for _ in range(n)]))
    return items


def canon(o):
    c = o.outputs[0]
    lpb, lpf = [], []
    for d in (c.logprobs or []):
        row_b, row_f = [], []
        for tok, v in sorted(d.items(), key=lambda kv: int(kv[0])):
            f = float(getattr(v, "logprob", v))
            row_b.append([int(tok), struct.unpack("<q", struct.pack("<d", f))[0]])
            row_f.append([int(tok), f])
        lpb.append(row_b)
        lpf.append(row_f)
    return {"ids": [int(x) for x in c.token_ids], "lpb": lpb, "lpf": lpf}


# ----------------------------------------------------------------------------------------------
# child process
# ----------------------------------------------------------------------------------------------
class Tee:
    def __init__(self, inner):
        self.inner = inner
        self.lines = []
        self._buf = ""

    def write(self, s):
        self.inner.write(s)
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if "PREFILL_STEP_TIMING" in line:
                self.lines.append(line)
        return len(s)

    def flush(self):
        self.inner.flush()


def proc(name, overlay_on, cache_path, out_path, plan):
    os.environ["BCG_OVERLAY"] = "1" if overlay_on else "0"
    os.environ["TLLM_AUTOTUNER_CACHE_PATH"] = cache_path
    os.environ.setdefault("TRTLLM_FI_HOST_PREFILL_PLAN", "1")
    sys.path.insert(0, HERE)
    import bcg_ab  # noqa: E402  (sets TLLM_WORKER_USE_SINGLE_PROCESS=1)

    env_print = {k: os.environ.get(k) for k in (
        "TLLM_AUTOTUNER_CACHE_PATH", "TRTLLM_CACHE_DIR", "TRTLLM_FI_HOST_PREFILL_PLAN",
        "TLLM_WORKER_USE_SINGLE_PROCESS", "TRTLLM_PREFILL_GRAPH_EXACT_ONLY",
        "TRTLLM_GEMMA4_MLP_CHUNK", "BCG_OVERLAY", "BCG_PACK")}
    log(f"proc {name}: env {json.dumps(env_print)}")
    log(f"proc {name}: cache file exists at start: {os.path.exists(cache_path)}"
        + (f" sha256 {sha256_file(cache_path)}" if os.path.exists(cache_path) else ""))
    prov = provenance()
    log(f"proc {name}: provenance {json.dumps(prov)}")
    bcg_ab.print_dims()

    import torch
    from tensorrt_llm._torch.autotuner import AutoTuner, AutoTunerProfilingCache
    log(f"proc {name}: after import TLLM_AUTOTUNER_CACHE_PATH="
        f"{os.environ.get('TLLM_AUTOTUNER_CACHE_PATH')}")

    phase = ["startup"]
    rec = {}
    orig_search = AutoTunerProfilingCache.search_cache

    def search_cache(self, custom_op, runners, input_shapes, tuning_config,
                     apply_map_to_tuning_buckets=True):
        res = orig_search(self, custom_op, runners, input_shapes, tuning_config,
                          apply_map_to_tuning_buckets)
        try:
            hit, idx, tactic, _ = res
            at = AutoTuner.get()
            prof = at._find_nearest_profile(input_shapes, tuning_config.dynamic_tensor_specs,
                                            tuning_config.constraint_specs,
                                            tuning_config.tune_max_num_tokens,
                                            apply_map_to_tuning_buckets)
            runner = runners[idx] if hit else None
            key = (phase[0], bool(at.is_tuning_mode), custom_op, str(prof), bool(hit),
                   runner.__class__.__name__ if runner is not None else "-",
                   str(runner.unique_id()) if runner is not None else "-", repr(tactic))
            rec[key] = rec.get(key, 0) + 1
        except Exception as e:  # noqa: BLE001
            k = (phase[0], None, custom_op, f"record-error {e!r}", None, "-", "-", "-")
            rec[k] = rec.get(k, 0) + 1
        return res

    AutoTunerProfilingCache.search_cache = search_cache

    mem_hist = overlay_on and "C" in name
    if mem_hist:
        try:
            torch.cuda.memory._record_memory_history(max_entries=200000, stacks="python")
        except Exception as e:  # noqa: BLE001
            log(f"memory history unavailable: {e!r}")
            mem_hist = False

    t0 = time.time()
    llm = bcg_ab.build_llm(use_overlay=overlay_on, batch_wait_timeout_ms=1500.0)
    startup_s = time.time() - t0
    free_b, total_b = torch.cuda.mem_get_info()
    mem = {"startup_s": round(startup_s, 1), "free_gib_after_startup": round(free_b / 2**30, 3),
           "torch_reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 3)}
    log(f"proc {name}: {mem}")

    ws = None
    if mem_hist:
        try:
            snap = torch.cuda.memory._snapshot()
            ws = []
            for seg in snap.get("segments", []):
                for blk in seg.get("blocks", []):
                    if blk.get("size") != WORKSPACE_BYTES or blk.get("state") != "active_allocated":
                        continue
                    frames = blk.get("frames") or []
                    names = [f"{os.path.basename(fr.get('filename', ''))}:{fr.get('line')}:"
                             f"{fr.get('name')}" for fr in frames[:12]]
                    ws.append({"segment_pool_id": str(seg.get("segment_pool_id")),
                               "stream": seg.get("stream"), "frames": names,
                               "cublas_mm_site": any("linear" in n or "cublas" in n
                                                     for n in names)})
            torch.cuda.memory._record_memory_history(enabled=None)
            log(f"proc {name}: 32 MiB active blocks: {json.dumps(ws)[:4000]}")
        except Exception as e:  # noqa: BLE001
            ws = {"error": repr(e)}

    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import graphs as _g
    counts = getattr(_g, "_BCG_STEP_COUNTS", None)
    replay_on = [True]
    orig_has_graph = None
    if overlay_on:
        from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph_runner import (
            BreakableCUDAGraphRunner,
        )
        orig_has_graph = BreakableCUDAGraphRunner.has_graph
        BreakableCUDAGraphRunner.has_graph = (
            lambda self, n: replay_on[0] and orig_has_graph(self, n))
    import tensorrt_llm._torch.attention.backends.flashinfer as _fi
    host_plan = getattr(_fi, "_HOST_PREFILL_PLAN", None)
    log(f"proc {name}: flashinfer _HOST_PREFILL_PLAN={host_plan}")

    params = bcg_ab.sp(logprobs=20)
    items = build_items(bcg_ab.Prompts)
    labels = {iid: lab for iid, lab, _ in items}
    phase[0] = "warmup"
    warm = bcg_ab.Prompts(5)
    for isl in (300, 1000, 8000):
        bcg_ab.run_batch(llm, [warm.make(isl) for _ in range(2)], params)

    passes, steps, pass_s = {}, {}, {}
    for p_i, kind in enumerate(plan.split(",")):
        tag = f"{name}{p_i}{kind}"
        phase[0] = tag
        replay_on[0] = kind != "E"
        order = list(range(len(items)))
        if kind == "S":
            random.Random(99).shuffle(order)
        before = dict(counts) if counts is not None else None
        ts = time.time()
        res = {}
        for j in order:
            iid, lab, prompts = items[j]
            _, outs = bcg_ab.run_batch(llm, prompts, params)
            res[iid] = [canon(o) for o in outs]
        passes[tag] = res
        pass_s[tag] = round(time.time() - ts, 1)
        if counts is not None:
            steps[tag] = {k: counts[k] - before[k] for k in counts}
        log(f"proc {name}: pass {tag} replay_allowed={replay_on[0]} steps={steps.get(tag)} "
            f"{pass_s[tag]} s; first mixed ids {res['mixed'][0]['ids']}")
    replay_on[0] = True

    repeat = None
    if overlay_on and "C" in name:
        phase[0] = "repeat"
        rp = bcg_ab.Prompts(9191)
        rep_groups = [("1x300", [rp.make(300)]), ("1x1800", [rp.make(1800)]),
                      ("1x2000", [rp.make(2000)]), ("4x300", [rp.make(300) for _ in range(4)])]
        repeat = {}
        for lab, grp in rep_groups:
            b0 = dict(counts)
            outs = []
            for _ in range(25):  # 5 warmup + 20, no reset
                _, o = bcg_ab.run_batch(llm, grp, params)
                outs.append([canon(x) for x in o])
            nrep = counts["replayed"] - b0["replayed"]
            replay_on[0] = False
            _, o = bcg_ab.run_batch(llm, grp, params)
            eager = [canon(x) for x in o]
            replay_on[0] = True
            repeat[lab] = {"calls": 25, "replayed_steps": nrep,
                           "all_equal_first": all(x == outs[0] for x in outs),
                           "first_equal_eager": outs[0] == eager}
            log(f"proc {name}: repeat {lab} {repeat[lab]}")

    diag = None
    if name in ("C", "D"):
        phase[0] = "diag"
        tee = Tee(sys.stdout)
        sys.stdout = tee
        _g._STEP_TIMING_DIAG = True
        dp = bcg_ab.Prompts(31337)
        try:
            modes = [("replay", True), ("eager", False)] if overlay_on else [("eager", True)]
            for mode, on in modes:
                replay_on[0] = on
                for isl, n in ((300, 12), (1000, 8)):
                    for _ in range(n):
                        bcg_ab.run_batch(llm, [dp.make(isl)], params)
        finally:
            _g._STEP_TIMING_DIAG = False
            sys.stdout = tee.inner
            replay_on[0] = True
        rows = {}
        for line in tee.lines:
            kv = dict(t.split("=", 1) for t in line.split("PREFILL_STEP_TIMING", 1)[1].split()
                      if "=" in t)
            key = f"{kv['tokens']}:{kv['mode']}"
            rows.setdefault(key, {"host": [], "gpu": []})
            rows[key]["host"].append(float(kv["host_enqueue_ms"]))
            rows[key]["gpu"].append(float(kv["gpu_ms"]))
        diag = {}
        for k, v in rows.items():
            h = sorted(v["host"][2:] or v["host"])  # drop first 2 per key as warmup
            g = sorted(v["gpu"][2:] or v["gpu"])
            diag[k] = {"n": len(h), "host_enqueue_ms_median": h[len(h) // 2],
                       "gpu_ms_median": g[len(g) // 2]}
        log(f"proc {name}: step timing diag (unprofiled, sync per step) {json.dumps(diag)}")

    if orig_has_graph is not None:
        BreakableCUDAGraphRunner.has_graph = orig_has_graph
    AutoTunerProfilingCache.search_cache = orig_search
    lookups = [list(k) + [v] for k, v in sorted(rec.items(), key=lambda kv: str(kv[0]))]
    out = {"name": name, "overlay": bcg_ab.overlay() if overlay_on else {}, "env": env_print,
           "provenance": prov, "memory": mem, "passes": passes, "steps": steps,
           "pass_s": pass_s, "labels": labels, "repeat": repeat, "diag": diag,
           "workspace_blocks": ws, "lookups": lookups, "host_prefill_plan": host_plan,
           "inmem_cache_size": len(AutoTuner.get().profiling_cache)}
    with open(out_path, "w") as f:
        json.dump(out, f)
    log(f"proc {name}: wrote {out_path}")
    # autotune() contexts saved the cache file at their exit; nothing else to flush.
    sys.stdout.flush()
    os._exit(0)


# ----------------------------------------------------------------------------------------------
# orchestrator
# ----------------------------------------------------------------------------------------------
def load_cache_file(p):
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return {sec: d.get(sec, {}) for sec in d if sec != "metadata"}


def diff_caches(ref, other):
    """Per section ('shared', 'rank_0'): sizes, missing/extra keys, (runner, tactic) diffs."""
    if ref is None or other is None:
        return {"error": "missing file", "ref": ref is not None, "other": other is not None}
    out = {}
    for sec in sorted(set(ref) | set(other)):
        a, b = ref.get(sec, {}), other.get(sec, {})
        only_a = sorted(set(a) - set(b))
        only_b = sorted(set(b) - set(a))
        pick = []
        same_time = 0
        for k in sorted(set(a) & set(b)):
            va, vb = a[k], b[k]
            if (va["runner_id"], va["tactic"]) != (vb["runner_id"], vb["tactic"]):
                pick.append({"key": k, "ref": [va["runner_id"], va["tactic"], va["min_time"]],
                             "other": [vb["runner_id"], vb["tactic"], vb["min_time"]]})
            elif va["min_time"] == vb["min_time"]:
                same_time += 1
        out[sec] = {"n_ref": len(a), "n_other": len(b), "only_ref": only_a[:40],
                    "n_only_ref": len(only_a), "only_other": only_b[:40],
                    "n_only_other": len(only_b), "n_pick_diff": len(pick), "pick_diffs": pick,
                    "n_identical_incl_time": same_time}
    return out


def cmp_passes(ref, other, labels):
    per = {}
    n_diff = 0
    lp_max = 0.0
    first = []
    for iid, ra in ref.items():
        rb = other.get(iid)
        lab = labels[iid]
        d = per.setdefault(lab, [0, 0])
        for x, y in zip(ra, rb or []):
            d[1] += 1
            if x["ids"] != y["ids"] or x["lpb"] != y["lpb"]:
                d[0] += 1
                n_diff += 1
                if len(first) < 8:
                    first.append(iid)
                for la, lb in zip(x["lpf"], y["lpf"]):
                    da, db = dict((t, v) for t, v in la), dict((t, v) for t, v in lb)
                    for t in set(da) & set(db):
                        lp_max = max(lp_max, abs(da[t] - db[t]))
    return {"n_diff": n_diff, "n": sum(v[1] for v in per.values()), "first": first,
            "lp_maxdiff": lp_max,
            "per_label_diff": {k: f"{v[0]}/{v[1]}" for k, v in per.items() if v[0]}}


def grep_log(path):
    pats = {"fallback": re.compile(r"using the fallback tactic"),
            "load": re.compile(r"\[Autotuner\] Loading cache from"),
            "save": re.compile(r"Saving cache to|Successfully saved cache"),
            "cache_size": re.compile(r"Cache size after warmup"),
            "kv": re.compile(r"(?i)(kv.?cache|KVCM).*(GiB|GB|tokens|blocks|pages)"),
            "capture": re.compile(r"(?i)(breakable|prefill cuda graph|captur)")}
    hits = {k: [] for k in pats}
    ansi = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
    with open(path, errors="replace") as f:
        for line in f:
            line = ansi.sub("", line.rstrip())
            for k, p in pats.items():
                if p.search(line):
                    if k == "fallback":  # drop timestamp/prefix so sets compare across logs
                        line = line[line.find("[AutoTuner]"):] if "[AutoTuner]" in line else line
                    hits[k].append(line[-400:])
    return {k: {"n": len(v), "lines": (v if k == "fallback" else v[:6] + (v[-4:] if len(v) > 10 else []))}
            for k, v in hits.items()}


def lookup_sets(lookups):
    inf_miss, inf_keys, tune_miss = set(), set(), set()
    for ph, tuning, op, prof, hit, runner, uid, tactic, n in lookups:
        if tuning is None:
            continue
        if tuning:
            if not hit:
                tune_miss.add((op, prof))
        else:
            inf_keys.add((op, prof))
            if not hit:
                inf_miss.add((op, prof))
    return inf_miss, inf_keys, tune_miss


def run_child(name, overlay, cache_path, out_json, plan, art):
    logp = os.path.join(art, f"proc_{name}.log")
    log(f"=== process {name}: overlay={overlay} cache={cache_path} plan={plan}")
    env = dict(os.environ)
    env["TLLM_AUTOTUNER_CACHE_PATH"] = cache_path
    env["TRTLLM_FI_HOST_PREFILL_PLAN"] = env.get("TRTLLM_FI_HOST_PREFILL_PLAN", "1")
    t0 = time.time()
    with open(logp, "w") as lf:
        p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "proc", name,
                              str(overlay), cache_path, out_json, plan],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        for line in p.stdout:
            lf.write(line)
            if line.startswith("[") and ("proc " in line or "startup" in line) \
                    or "fallback tactic" in line or "Error" in line or "Traceback" in line:
                print(line, end="", flush=True)
        p.wait()
    log(f"=== process {name} rc={p.returncode} in {time.time() - t0:.0f} s")
    return p.returncode, logp


def orchestrate():
    art = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp/pin_proof")
    os.makedirs(art, exist_ok=True)
    cdir = os.path.join(art, "caches")
    os.makedirs(cdir, exist_ok=True)
    prov = provenance()
    log(f"orchestrator provenance {json.dumps(prov)}")
    tcd = os.environ.get("TRTLLM_CACHE_DIR")
    default_cache = os.path.join(os.path.expanduser(tcd), "autotuner", "cache.json") if tcd else None
    default_info = None
    if default_cache and os.path.exists(default_cache):
        try:
            dd = load_cache_file(default_cache)
            default_info = {"path": default_cache, "size": os.path.getsize(default_cache),
                            "mtime": os.path.getmtime(default_cache),
                            "n_keys": {k: len(v) for k, v in dd.items()}}
        except Exception as e:  # noqa: BLE001
            default_info = {"path": default_cache, "error": repr(e)}
    log(f"inherited TRTLLM_CACHE_DIR={tcd!r}; its default autotuner cache: "
        f"{json.dumps(default_info) if default_info else 'absent'} (NOT used: every process "
        f"gets an explicit TLLM_AUTOTUNER_CACHE_PATH)")
    FA, FB = os.path.join(cdir, "FA.json"), os.path.join(cdir, "FB.json")
    FC, FD = os.path.join(cdir, "FC_copy_of_FA.json"), os.path.join(cdir, "FD_copy_of_FA.json")
    FA_ref = os.path.join(cdir, "FA_pristine.json")
    for p in (FA, FB, FC, FD, FA_ref):
        for q in (p, p + ".lock"):
            if os.path.exists(q):
                os.remove(q)
    res, logs, rcs = {}, {}, {}
    sel = os.environ.get("PIN_PROCS", "A,B,C,D").split(",")
    plans = {"A": (0, FA, "N,N"), "B": (1, FB, "N"), "C": (1, FC, "N,E,S"), "D": (0, FD, "N")}
    for name in sel:
        overlay, cpath, plan = plans[name]
        if name in ("C", "D"):
            if not os.path.exists(FA):
                print(json.dumps({"passed": False, "error": "FA missing before " + name}))
                return
            shutil.copyfile(FA, cpath)
        outj = os.path.join(art, f"proc_{name}.json")
        rc, logp = run_child(name, overlay, cpath, outj, plan, art)
        rcs[name] = rc
        logs[name] = grep_log(logp)
        if rc != 0 or not os.path.exists(outj):
            print(json.dumps({"passed": False, "error": f"process {name} rc={rc}",
                              "log_grep": logs[name]}))
            return
        res[name] = json.load(open(outj))
        if name == "A":
            shutil.copyfile(FA, FA_ref)
            log(f"FA saved: sha256 {sha256_file(FA)} sections "
                f"{ {k: len(v) for k, v in load_cache_file(FA).items()} }")
    labels = res["A"]["labels"]
    A0 = res["A"]["passes"]["A0N"]
    cmp = {"A1_vs_A0": cmp_passes(A0, res["A"]["passes"]["A1N"], labels)}
    for name in ("B", "C", "D"):
        for tag, pr in res.get(name, {}).get("passes", {}).items():
            cmp[f"{tag}_vs_A0"] = cmp_passes(A0, pr, labels)
    if "C" in res:
        cp = res["C"]["passes"]
        cmp["C1E_vs_C0N"] = cmp_passes(cp["C0N"], cp["C1E"], labels)
        cmp["C2S_vs_C0N"] = cmp_passes(cp["C0N"], cp["C2S"], labels)
    fa = load_cache_file(FA_ref)
    caches = {"sizes": {n: ({k: len(v) for k, v in (load_cache_file(p) or {}).items()})
                        for n, p in (("FA", FA_ref), ("FB", FB), ("FC_after", FC),
                                     ("FD_after", FD))},
              "FB_vs_FA": diff_caches(fa, load_cache_file(FB)) if "B" in res else None,
              "FC_after_vs_FA": diff_caches(fa, load_cache_file(FC)) if "C" in res else None,
              "FD_after_vs_FA": diff_caches(fa, load_cache_file(FD)) if "D" in res else None,
              "FA_after_A_equals_pristine": sha256_file(FA) == sha256_file(FA_ref)}
    lk = {n: lookup_sets(r["lookups"]) for n, r in res.items()}
    a_miss, a_keys, _ = lk["A"]
    lookups = {}
    for n, (miss, keys, tmiss) in lk.items():
        lookups[n] = {"inference_keys": len(keys), "inference_misses": sorted(map(list, miss))[:40],
                      "n_inference_misses": len(miss),
                      "new_inference_misses_vs_A": sorted(map(list, miss - a_miss))[:40],
                      "inference_keys_not_in_A": sorted(map(list, keys - a_keys))[:40],
                      "n_tuning_mode_misses": len(tmiss),
                      "tuning_mode_misses": sorted(map(list, tmiss))[:20]}
    # pass criteria (plan 701)
    def zero(k):
        return k in cmp and cmp[k]["n_diff"] == 0

    keyset_eq = {}
    for k in ("FB_vs_FA", "FC_after_vs_FA", "FD_after_vs_FA"):
        d = caches.get(k) or {}
        keyset_eq[k] = bool(d) and "error" not in d and all(
            s["n_only_ref"] == 0 and s["n_only_other"] == 0 for s in d.values())
    pinned_equal = {k: keyset_eq[k] and all(s["n_pick_diff"] == 0 for s in caches[k].values())
                    for k in ("FC_after_vs_FA", "FD_after_vs_FA") if caches.get(k)}
    c_steps = res.get("C", {}).get("steps", {})
    rep = res.get("C", {}).get("repeat") or {}
    crit = {
        "A_repeatable(A1==A0)": zero("A1_vs_A0"),
        "D==A (pinned main A/A)": zero("D0N_vs_A0"),
        "C_replay==A": zero("C0N_vs_A0"),
        "C_eager_gated==A": zero("C1E_vs_A0"),
        "C_replay_shuffled==A": zero("C2S_vs_A0"),
        "C_ABA_in_process": zero("C1E_vs_C0N") and zero("C2S_vs_C0N"),
        "C_replayed_in_N_and_S": c_steps.get("C0N", {}).get("replayed", 0) > 0
                                  and c_steps.get("C2S", {}).get("replayed", 0) > 0,
        "C_E_pass_all_eager": c_steps.get("C1E", {}).get("replayed", 1) == 0,
        "FB_keyset==FA": keyset_eq.get("FB_vs_FA", False),
        "FC_after==FA": pinned_equal.get("FC_after_vs_FA", False),
        "FD_after==FA": pinned_equal.get("FD_after_vs_FA", False),
        "no_new_inference_misses_B_C": all(not lookups[n]["new_inference_misses_vs_A"]
                                           for n in ("B", "C") if n in lookups),
        "no_tuning_misses_C_D(pinned)": all(lookups[n]["n_tuning_mode_misses"] == 0
                                            for n in ("C", "D") if n in lookups),
        "fallback_warnings_B_C_subset_of_A": all(
            set(logs[n]["fallback"]["lines"]) <= set(logs["A"]["fallback"]["lines"])
            for n in ("B", "C") if n in logs),
        "repeat_replay_all_equal_first_and_eager": bool(rep) and all(
            v["all_equal_first"] and v["first_equal_eager"] and v["replayed_steps"] > 0
            for v in rep.values()),
        "logprobs_present": bool(A0["s0"][0]["lpb"]),
    }
    passed = all(crit.values())
    summary = {
        "passed": passed,
        "compared": "token ids + top-20 logprobs, bitwise (int64 views), against process A = main "
                    "eager with a fresh AutoTuner cache FA; C = BCG with a copy of FA (passes: "
                    "replay N, has_graph-gated eager E, replay in shuffled bucket order S), D = "
                    "main eager with a copy of FA, B = BCG unpinned with a fresh FB; %d outputs "
                    "per pass (mixed 94-prompt batch, %d singles incl. every captured size "
                    "300..2000 plus 1500/2047/off-bucket, small batches 2x300..6x300, 2x900, "
                    "2x1000)" % (cmp["A1_vs_A0"]["n"],
                                 sum(1 for i in labels if i.startswith("s"))),
        "criteria": crit, "comparisons": cmp, "caches": caches, "lookups": lookups,
        "fallback_warnings": {n: logs[n]["fallback"] for n in logs},
        "log_grep": {n: {k: v for k, v in logs[n].items() if k != "fallback"} for n in logs},
        "steps_per_pass": {n: r["steps"] for n, r in res.items()},
        "pass_seconds": {n: r["pass_s"] for n, r in res.items()},
        "repeat": rep, "diag_step_timing": {n: r["diag"] for n, r in res.items()},
        "memory": {n: r["memory"] for n, r in res.items()},
        "workspace_blocks_C": res.get("C", {}).get("workspace_blocks"),
        "env": {n: r["env"] for n, r in res.items()},
        "provenance": prov, "inherited_default_autotuner_cache": default_info,
        "host_prefill_plan": {n: r["host_prefill_plan"] for n, r in res.items()},
        "overlay": {n: r["overlay"] for n, r in res.items()},
    }
    sp = os.path.join(art, "pin_proof_summary.json")
    with open(sp, "w") as f:
        json.dump(summary, f, indent=1)
    # shared copy (top-lead ruling 01:06): .tmp-<id> dir, manifest last, rename into place
    try:
        dst = os.path.join(SHARED_ROOT, BASE_MAIN)
        tmp = os.path.join(SHARED_ROOT, f".tmp-{os.getpid()}")
        os.makedirs(tmp, exist_ok=True)
        files = {"pin_proof_summary.json": sp, "FA.json": FA_ref}
        man = {"provenance": prov, "created": time.time(), "files": {}}
        for fn, src in files.items():
            shutil.copyfile(src, os.path.join(tmp, fn))
            man["files"][fn] = sha256_file(os.path.join(tmp, fn))
        with open(os.path.join(tmp, "manifest.json"), "w") as f:
            json.dump(man, f, indent=1)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        os.rename(tmp, dst)
        shutil.copyfile(os.path.join(dst, "manifest.json"), os.path.join(art, "shared_manifest.json"))
        summary["shared_dir"] = dst
        log(f"shared copy at {dst}")
    except Exception as e:  # noqa: BLE001
        summary["shared_dir"] = f"failed: {e!r}"
    slim = {k: summary[k] for k in ("passed", "compared", "criteria", "steps_per_pass",
                                    "memory", "diag_step_timing", "repeat", "shared_dir",
                                    "host_prefill_plan")}
    slim["comparisons"] = {k: {kk: v[kk] for kk in ("n_diff", "n", "lp_maxdiff",
                                                     "per_label_diff")}
                           for k, v in cmp.items()}
    slim["caches"] = {"sizes": caches["sizes"],
                      **{k: ({s: {kk: vv for kk, vv in d.items()
                                  if kk in ("n_ref", "n_other", "n_only_ref", "n_only_other",
                                            "n_pick_diff")}
                              for s, d in caches[k].items()} if caches.get(k) else None)
                         for k in ("FB_vs_FA", "FC_after_vs_FA", "FD_after_vs_FA")}}
    slim["n_fallback_warnings"] = {n: logs[n]["fallback"]["n"] for n in logs}
    slim["summary_file"] = sp
    print(json.dumps(slim), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "proc":
        proc(sys.argv[2], sys.argv[3] == "1", sys.argv[4], sys.argv[5], sys.argv[6])
    else:
        orchestrate()
