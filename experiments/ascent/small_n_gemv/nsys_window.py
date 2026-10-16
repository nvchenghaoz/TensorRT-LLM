# nsys check (acceptance item 4): in real decode at c2/c4, does in_proj_ba
# finish inside the parallel in_proj_qkvz window?
#
# Usage: python nsys_window.py <checkpoint>
# Re-runs itself under `nsys profile --cuda-graph-trace=node` with an in-process
# (TLLM_WORKER_USE_SINGLE_PROCESS=1) LLM using the benchmark's engine settings,
# profiles batch-2 and batch-4 greedy decode (capture range = cudaProfilerApi),
# exports SQLite and, for every in_proj_ba kernel of a decode step, finds the
# concurrent FP8 in_proj_qkvz GEMV (cudaCoreGemm with gridY = 16384/2 = 8192)
# and reports ba_end - qkvz_end.
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time

OUT = "/tmp/small_n_gemv_nsys"


def log(*a):
    print(*a, flush=True)


def run_model(ckpt):
    os.environ["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"
    import random

    import torch

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

    llm = LLM(model=ckpt,
              max_batch_size=8,
              max_num_tokens=8192,
              max_seq_len=36864,
              enable_chunked_prefill=True,
              cuda_graph_config=CudaGraphConfig(max_batch_size=8),
              kv_cache_config=KvCacheConfig(enable_block_reuse=False,
                                            free_gpu_memory_fraction=0.6))
    rng = random.Random(0)

    def prompts(n, length=1024):
        return [[rng.randrange(1000, 100000) for _ in range(length)]
                for _ in range(n)]

    sp = SamplingParams(max_tokens=48, ignore_eos=True, temperature=0.0)
    log("warmup")
    llm.generate(prompts(4), sp)
    llm.generate(prompts(2), sp)
    torch.cuda.synchronize()
    log("profiling c2 then c4")
    torch.cuda.profiler.start()
    llm.generate(prompts(2), sp)
    time.sleep(0.5)
    llm.generate(prompts(4), sp)
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    log("done")
    llm.shutdown()


def analyze(db):
    con = sqlite3.connect(db)
    rows = con.execute(
        "SELECT k.start, k.end, k.streamId, k.gridX, k.gridY, k.gridZ, "
        "s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s "
        "ON k.demangledName = s.id ORDER BY k.start").fetchall()
    log("kernels in capture:", len(rows))
    ba = [r for r in rows if "small_n_gemv" in r[6]]
    qkvz = [r for r in rows if "cudaCoreGemm" in r[6] and r[4] == 8192]
    other_ba = [r for r in rows if ("wmma_tensorop_bf16" in r[6]
                                    or "dot_kernel" in r[6])]
    log("small_n_gemv kernels:", len(ba), "qkvz kernels:", len(qkvz),
        "cuBLAS wmma/dot kernels:", len(other_ba))
    res = {}
    qi = 0
    by_m = {}
    for b in ba:
        m = b[4]  # grid = (N, M)
        # concurrent qkvz: the one with the largest overlap
        best, best_ov = None, 0
        for q in qkvz:
            if q[0] > b[1]:
                break
            ov = min(b[1], q[1]) - max(b[0], q[0])
            if ov > best_ov:
                best, best_ov = q, ov
        if best is None:
            # no overlap: compare with the nearest preceding qkvz
            prev = [q for q in qkvz if q[1] <= b[0]]
            nxt = [q for q in qkvz if q[0] >= b[1]]
            by_m.setdefault(m, []).append(
                dict(delta=None, dur=b[1] - b[0],
                     gap_prev=(b[0] - prev[-1][1]) if prev else None,
                     gap_next=(nxt[0][0] - b[1]) if nxt else None))
            continue
        by_m.setdefault(m, []).append(
            dict(delta=b[1] - best[1], dur=b[1] - b[0],
                 qdur=best[1] - best[0], start_lag=b[0] - best[0]))
    for m, lst in sorted(by_m.items()):
        d = sorted(x["delta"] for x in lst if x["delta"] is not None)
        no_ov = [x for x in lst if x["delta"] is None]
        durs = sorted(x["dur"] for x in lst)
        qd = sorted(x["qdur"] for x in lst if x.get("qdur") is not None)
        inside = sum(1 for x in d if x <= 0)
        res[str(m)] = dict(
            ba_kernels=len(lst),
            overlapping=len(d),
            not_overlapping=len(no_ov),
            finished_inside_qkvz=inside,
            frac_inside=inside / max(1, len(d)),
            ba_end_minus_qkvz_end_ns_median=d[len(d) // 2] if d else None,
            ba_end_minus_qkvz_end_ns_max=d[-1] if d else None,
            ba_dur_ns_median=durs[len(durs) // 2],
            ba_dur_ns_p90=durs[int(0.9 * (len(durs) - 1))],
            qkvz_dur_ns_median=qd[len(qd) // 2] if qd else None,
        )
        log("M=%s" % m, res[str(m)])
        if no_ov:
            log("  non-overlapping examples:", no_ov[:5])
    return res, len(ba), len(other_ba)


def main():
    if len(sys.argv) > 2 and sys.argv[2] == "run":
        run_model(sys.argv[1])
        return
    nsys = shutil.which("nsys")
    log("nsys:", nsys)
    if nsys is None:
        print(json.dumps({"passed": False, "error": "nsys not found"}))
        return
    os.makedirs(OUT, exist_ok=True)
    rep = os.path.join(OUT, "decode")
    cmd = [
        nsys, "profile", "--trace=cuda", "--cuda-graph-trace=node",
        "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
        "--sample=none", "--cpuctxsw=none", "--force-overwrite=true", "-o",
        rep, sys.executable,
        os.path.abspath(__file__), sys.argv[1], "run"
    ]
    log("running:", " ".join(cmd))
    p = subprocess.run(cmd)
    log("nsys exit", p.returncode)
    db = rep + ".sqlite"
    subprocess.run([
        nsys, "export", "--type=sqlite", "--force-overwrite=true", "-o", db,
        rep + ".nsys-rep"
    ])
    res, n_ba, n_cublas = analyze(db)
    ok = (n_ba > 0 and all(
        v["frac_inside"] >= 0.95 for m, v in res.items() if m in ("2", "4")))
    art = os.environ.get("ASCENT_ARTIFACTS_DIR") or os.environ.get(
        "ARTIFACTS_DIR")
    if art:
        try:
            shutil.copy(rep + ".nsys-rep", art)
        except Exception as e:  # noqa
            log("copy failed", e)
    print(json.dumps({
        "passed": ok,
        "compared": "nsys decode trace c2/c4: in_proj_ba (small_n_gemv) end "
        "vs concurrent in_proj_qkvz cudaCoreGemm end",
        "by_M": res,
        "small_n_gemv_kernels": n_ba,
        "cublas_wmma_dot_kernels": n_cublas
    }), flush=True)


if __name__ == "__main__":
    main()
