# Acceptance item 2 (idea 5): nsys of real Qwen3.8-27B-NVFP4 prefill + mixed
# (prefill + decode riders) steps on SM120. Counts the Triton state gather /
# scatter launches (_gather_cast_vk_to_fp32_vk_kernel /
# _cast_scatter_fp32_vk_to_vk_kernel) and the FlashInfer GDN chunk kernels in the
# capture window, with the in-place pool path on (default) and off
# (TLLM_GDN_SM120_POOL_IO=0, main's path) in one job.
#
# Usage: python nsys_mixed.py <checkpoint>            (both modes, prints JSON)
#        python nsys_mixed.py <checkpoint> run         (the profiled workload)
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time

OUT = "/tmp/gdn_pool_io_nsys"


def log(*a):
    print(*a, flush=True)


def run_model(ckpt):
    os.environ["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"
    import random

    import torch

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

    llm = LLM(
        model=ckpt,
        max_batch_size=32,
        max_num_tokens=8192,
        max_seq_len=36864,
        enable_chunked_prefill=True,
        cuda_graph_config=CudaGraphConfig(max_batch_size=32),
        kv_cache_config=KvCacheConfig(
            dtype="fp8", enable_block_reuse=False, free_gpu_memory_fraction=0.6
        ),
    )
    rng = random.Random(0)

    def prompt(length=2048):
        return [rng.randrange(1000, 100000) for _ in range(length)]

    sp = SamplingParams(max_tokens=48, ignore_eos=True, temperature=0.0)
    log("warmup")
    llm.generate([prompt(512) for _ in range(4)], sp)
    torch.cuda.synchronize()
    log("profiling: 16 staggered 2048-token requests (prefill + mixed steps)")
    torch.cuda.profiler.start()
    futs = []
    for i in range(16):
        futs.append(llm.generate_async(prompt(), sp))
        time.sleep(0.12)
    outs = [f.result() for f in futs]
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    log("done", len(outs), "requests;", "first tokens", outs[0].outputs[0].token_ids[:8])
    llm.shutdown()


def analyze(db):
    con = sqlite3.connect(db)
    rows = con.execute(
        "SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s "
        "ON k.demangledName = s.id ORDER BY k.start"
    ).fetchall()
    n = len(rows)
    gather = [r for r in rows if "gather_cast_vk_to_fp32_vk" in r[2]]
    scatter = [r for r in rows if "cast_scatter_fp32_vk_to_vk" in r[2]]
    chunk = [r for r in rows if "deltarule" in r[2].lower()]
    names = sorted(set(r[2][:120] for r in chunk))
    tot = lambda lst: sum(r[1] - r[0] for r in lst) / 1e6
    return dict(
        kernels_in_capture=n,
        gather_launches=len(gather),
        scatter_launches=len(scatter),
        gather_ms=tot(gather),
        scatter_ms=tot(scatter),
        chunk_kernel_launches=len(chunk),
        chunk_kernel_ms=tot(chunk),
        chunk_kernel_names=names[:6],
    )


def profile(ckpt, mode):
    nsys = shutil.which("nsys")
    rep = os.path.join(OUT, "mixed_" + mode)
    env = dict(os.environ)
    env["TLLM_GDN_SM120_POOL_IO"] = "1" if mode == "on" else "0"
    cmd = [
        nsys, "profile", "--trace=cuda", "--cuda-graph-trace=node",
        "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
        "--sample=none", "--cpuctxsw=none", "--force-overwrite=true", "-o", rep,
        sys.executable, os.path.abspath(__file__), ckpt, "run",
    ]
    log("running (%s):" % mode, " ".join(cmd))
    p = subprocess.run(cmd, env=env)
    log("nsys exit", p.returncode)
    db = rep + ".sqlite"
    subprocess.run([nsys, "export", "--type=sqlite", "--force-overwrite=true", "-o", db,
                    rep + ".nsys-rep"])
    res = analyze(db)
    log(mode, json.dumps(res))
    art = os.environ.get("ASCENT_ARTIFACTS_DIR") or os.environ.get("ARTIFACTS_DIR")
    if art:
        try:
            shutil.copy(rep + ".nsys-rep", art)
        except Exception as e:  # noqa: BLE001
            log("copy failed", e)
    return res


def main():
    if len(sys.argv) > 2 and sys.argv[2] == "run":
        run_model(sys.argv[1])
        return
    os.makedirs(OUT, exist_ok=True)
    on = profile(sys.argv[1], "on")
    off = profile(sys.argv[1], "off")
    ok = (
        on["chunk_kernel_launches"] > 0
        and on["gather_launches"] == 0
        and on["scatter_launches"] == 0
        and off["gather_launches"] > 0
        and off["scatter_launches"] > 0
    )
    print(json.dumps(dict(
        passed=bool(ok),
        compared=("nsys of real-model 2048-token prefill + mixed steps (16 staggered requests): "
                  "Triton gather/scatter launches with the SM120 in-place pool path on (default) "
                  "vs off (TLLM_GDN_SM120_POOL_IO=0 = main's path)"),
        pool_io_on=on,
        pool_io_off=off,
    )), flush=True)


if __name__ == "__main__":
    main()
