# Branch 46 (idea 80) decode-window eager-kernel census at c8/c16 under nsys --cuda-graph-trace=node.
# BRACE-FREE on purpose: run_gpu_pair inlines it in the base tree via a bash heredoc (the base lacks this file).
# Usage: python b46_nsys_eager.py <checkpoint>        (re-runs itself under nsys, then analyzes the sqlite)
#        python b46_nsys_eager.py <checkpoint> run    (the profiled model run only)
# Model run: engine.json KV config (FP8 KV, block reuse, snapshot 32768), max_batch_size 32, decode graphs <= 32,
# overlap on. Primes the shared prefix, warms c8/c16, then inside one cudaProfilerStart/Stop region runs a c8 call
# and a c16 call (NVTX ranges b46_c8 / b46_c16; 32768 cached + 2048 unique tokens, GEN greedy tokens each).
# Analysis per NVTX range: the decode graph = the graphId with the most kernels in the range (full-batch decode);
# window = first..last kernel of that graph; steps = cudaGraphLaunch calls in the window; eager kernels = kernels
# with no graphId that start in the window, counted by name per step. indexSelect* / scatter_gather / copy shown.
import json
import os
import random
import shutil
import sqlite3
import subprocess
import sys
import threading
import time

OUT = "/tmp/b46_nsys"
PREFIX = 32768
SUFFIX = 2048
GEN = 48


def log(*a):
    print(*a, flush=True)


def heartbeat(tag):
    t0 = time.time()

    def loop():
        while True:
            time.sleep(30)
            log("[heartbeat %s] %.0fs" % (tag, time.time() - t0))

    threading.Thread(target=loop, daemon=True).start()


def run_model(ckpt):
    os.environ["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"
    import torch

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig
    heartbeat("model")
    engine_json = json.load(open("experiments/ascent-config/engine.json"))
    kvc = dict(engine_json.get("kv_cache_config", dict()))
    t0 = time.time()
    llm = LLM(model=ckpt, max_seq_len=36864, max_batch_size=32, max_num_tokens=8192, enable_chunked_prefill=True,
              cuda_graph_config=CudaGraphConfig(max_batch_size=32), kv_cache_config=KvCacheConfig(**kvc),
              batch_wait_timeout_ms=200)
    log("LLM ready in %.1fs" % (time.time() - t0))
    rng = random.Random(0)
    prefix = [rng.randrange(1000, 100000) for _ in range(PREFIX)]

    def prompts(n):
        return [prefix + [rng.randrange(1000, 100000) for _ in range(SUFFIX)] for _ in range(n)]

    sp_w = SamplingParams(max_tokens=8, ignore_eos=True, temperature=0.0)
    sp = SamplingParams(max_tokens=GEN, ignore_eos=True, temperature=0.0)
    llm.generate(prompts(1), sp_w)
    llm.generate(prompts(8), sp_w)
    llm.generate(prompts(16), sp_w)
    torch.cuda.synchronize()
    log("warmup done")
    torch.cuda.profiler.start()
    for n in (8, 16):
        torch.cuda.nvtx.range_push("b46_c%d" % n)
        llm.generate(prompts(n), sp)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        log("profiled c%d" % n)
    torch.cuda.profiler.stop()
    llm.shutdown()


def nvtx_ranges(con):
    cols = [r[1] for r in con.execute("PRAGMA table_info(NVTX_EVENTS)")]
    q = "SELECT start, end, text" + (", textId" if "textId" in cols else ", NULL") + " FROM NVTX_EVENTS"
    strings = dict(con.execute("SELECT id, value FROM StringIds").fetchall())
    out = []
    for s, e, text, tid in con.execute(q).fetchall():
        name = text if text is not None else strings.get(tid)
        if name and name.startswith("b46_c") and e is not None:
            out.append((name[4:], s, e))
    return sorted(out, key=lambda x: x[1])


def analyze(db):
    con = sqlite3.connect(db)
    rows = con.execute("SELECT k.start, k.end, s.value, k.graphId FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                       "JOIN StringIds s ON k.demangledName = s.id ORDER BY k.start").fetchall()
    launch_ids = [r[0] for r in con.execute("SELECT id FROM StringIds WHERE value LIKE 'cudaGraphLaunch%'")]
    launches = []
    for lid in launch_ids:
        launches += [r[0] for r in con.execute("SELECT start FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE nameId = ?",
                                               (lid,)).fetchall()]
    res = dict()
    for tag, s, e in nvtx_ranges(con):
        in_rng = [r for r in rows if r[0] >= s and r[1] <= e]
        per_graph = dict()
        for r in in_rng:
            if r[3]:
                per_graph[r[3]] = per_graph.get(r[3], 0) + 1
        if not per_graph:
            continue
        gid = max(per_graph, key=lambda g: per_graph[g])
        gk = [r for r in in_rng if r[3] == gid]
        lo, hi = gk[0][0], gk[-1][1]
        steps = sum(1 for t in launches if lo - 2000000 <= t <= hi)
        counts = dict()
        durs = dict()
        for r in in_rng:
            if r[3] or r[0] < lo or r[0] > hi:
                continue
            nm = r[2][:140]
            counts[nm] = counts.get(nm, 0) + 1
            durs[nm] = durs.get(nm, 0) + (r[1] - r[0])
        st = max(1, steps)
        per = dict()
        for nm in sorted(counts, key=lambda x: -durs[x]):
            per[nm] = dict(count=counts[nm], per_step=counts[nm] / st, us_per_step=durs[nm] / st / 1e3)
        res[tag] = dict(
            decode_graph_id=gid, decode_graph_kernels=per_graph[gid], graph_launches_in_window=steps,
            eager_kernels_per_step=sum(counts.values()) / st, eager_us_per_step=sum(durs.values()) / st / 1e3,
            indexSelect_kernels=sum(v["count"] for k, v in per.items() if "indexSelect" in k),
            scatter_gather_kernels=sum(v["count"] for k, v in per.items() if "scatter_gather" in k),
            copy_kernels_per_step=sum(v["count"] for k, v in per.items() if "copy" in k.lower()) / st,
            eager_by_name=per)
    return res


def main():
    if len(sys.argv) > 2 and sys.argv[2] == "run":
        run_model(sys.argv[1])
        return
    heartbeat("driver")
    tree_op = "gather" if "previous_slots_i64" in open(
        "tensorrt_llm/_torch/pyexecutor/engine/runners/decoder/runner.py").read() else "index_select"
    log("tree op: %s" % tree_op)
    nsys = shutil.which("nsys") or "/usr/local/cuda/bin/nsys"
    os.makedirs(OUT, exist_ok=True)
    rep = os.path.join(OUT, "dec_%d" % os.getpid())
    cmd = [nsys, "profile", "--trace=cuda,nvtx", "--cuda-graph-trace=node", "--capture-range=cudaProfilerApi",
           "--capture-range-end=stop", "--sample=none", "--cpuctxsw=none", "--force-overwrite=true", "-o", rep,
           sys.executable, os.path.abspath(__file__), sys.argv[1], "run"]
    log("running: %s" % " ".join(cmd))
    p = subprocess.run(cmd)
    log("nsys exit %d" % p.returncode)
    subprocess.run([nsys, "export", "--type=sqlite", "--force-overwrite=true", "-o", rep + ".sqlite",
                    rep + ".nsys-rep"])
    res = analyze(rep + ".sqlite")
    for tag, r in res.items():
        log("%s: graph %s launches %d eager/step %.2f (%.2f us) indexSelect %d scatter_gather %d copy/step %.2f" % (
            tag, r["decode_graph_id"], r["graph_launches_in_window"], r["eager_kernels_per_step"],
            r["eager_us_per_step"], r["indexSelect_kernels"], r["scatter_gather_kernels"],
            r["copy_kernels_per_step"]))
        for nm, v in list(r["eager_by_name"].items())[:30]:
            log("   %6.2f/step %7.2f us/step  %s" % (v["per_step"], v["us_per_step"], nm))
    ok = len(res) >= 2 and all(v["graph_launches_in_window"] >= 10 for v in res.values())
    if tree_op == "gather":
        ok = ok and all(v["indexSelect_kernels"] == 0 for v in res.values())
    out = dict(passed=bool(ok), tree_op=tree_op, census=res,
               note="eager (non-graph) kernels starting inside each range's full-batch decode-graph window")
    art = os.environ.get("AUTO_DEV_ARTIFACTS")
    if art:
        os.makedirs(art, exist_ok=True)
        with open(os.path.join(art, "b46_nsys_eager.json"), "w") as f:
            json.dump(out, f, indent=1)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
