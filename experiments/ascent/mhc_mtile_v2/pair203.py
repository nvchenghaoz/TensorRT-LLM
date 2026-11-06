"""Paired microbenchmark for idea 316 (branch 203): v2<12,2,SMEM,1,JROLLED> as mtile id tile_m = 18.

Full mhc_fused_hc op (pmap/GEMM kernel + bigfuse), V4.1-lagged call (norm_weight + pre_mix_ext), DeepSeek-V4.1
hc=4, hidden=5120, N=24. For each M the AutoTuner (cache cleared) picks the tactic exactly as production does;
that tactic is then timed: 5 warmup + REPS x ITERS (5 x 20 = 100) calls, 256 MB L2 flush before each (cold L2),
CUDA events around the op only. M=128 is also timed warm (no flush) and as a 20-call CUDA graph over 20 distinct
layer input sets (proxy of landing 44's L20-39 tail hc_ffn graph; ~170 MB working set > 128 MB L2), flushed before
each replay; per-call time reported. Diagnostic: fixed-tactic cold times of v1<12,2> (id 2), id 18, id 20, id 72.
Same script in both arms; the base arm does not offer id 18.
Last stdout line: {"batch": {key: {"kernel_ms", "noise_floor", "tactic"}}, "output": {"class": "exact"}}.
"""
import json
import mmap
import os
import socket
import subprocess
import sys

import torch

import tensorrt_llm
from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.modules.mhc import mhc_cuda

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mhc_fhc_bench import make_inputs  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from guard import dirty_guard  # noqa: E402

dirty_guard()


def env_log():
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,clocks.sm,clocks.max.sm,memory.used,utilization.gpu,"
                        "temperature.gpu,clocks_throttle_reasons.active", "--format=csv,noheader"],
                       capture_output=True, text=True).stdout.strip()
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()
    print(f"[env] host={socket.gethostname()} CVD={os.environ.get('CUDA_VISIBLE_DEVICES')} slot_cwd={os.getcwd()}",
          flush=True)
    print(f"[env] nvidia-smi: {q.replace(chr(10), ' | ')}", flush=True)
    print(f"[env] compute-apps: {apps or 'none'}", flush=True)
    libdir = os.path.join(os.path.dirname(tensorrt_llm.__file__), "libs")
    keys = [b"fused_pmap_gemm_fma_mtile_v3", b"fused_pmap_gemm_fma_mtile_v2ILi12ELi2E",
            b"fused_pmap_gemm_fma_mtile_v2ILi24ELi4E"]
    for name in ("libtensorrt_llm.so", "libth_common.so"):
        p = os.path.join(libdir, name)
        if not os.path.exists(p):
            print(f"[lib] {p} missing", flush=True)
            continue
        real = os.path.realpath(p)
        with open(real, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            cnt = {}
            for k in keys:
                c, i = 0, mm.find(k)
                while i >= 0:
                    c += 1
                    i = mm.find(k, i + 1)
                cnt[k.decode()] = c
            mm.close()
        print(f"[lib] {real} mtime={os.path.getmtime(real):.0f} counts={cnt}", flush=True)


env_log()

HC, HID = 4, 5120
ARGS = (HC, HID, 1e-6, 1e-6, 1e-6, 2.0, 20)
MS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "64,128,256,512").split(",")]
ITERS = int(os.environ.get("ITERS", "20"))
REPS = int(os.environ.get("REPS", "5"))
GRAPH_LAYERS = 20

flush = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")


def time_fn(fn, iters, cold=True):
    st = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        if cold:
            flush.zero_()
        st[i].record()
        fn()
        en[i].record()
    torch.cuda.synchronize()
    return sorted(s.elapsed_time(e) for s, e in zip(st, en))


def measure(fn, cold=True, scale=1.0):
    for _ in range(5):
        fn()
    meds, noises = [], []
    for _ in range(REPS):
        ts = [t / scale for t in time_fn(fn, ITERS, cold)]
        med = ts[len(ts) // 2]
        meds.append(med)
        noises.append((ts[int(0.9 * (len(ts) - 1))] - ts[int(0.1 * (len(ts) - 1))]) / med)
    meds.sort()
    return meds[len(meds) // 2], max(noises), [round(m * 1000, 1) for m in meds]


out = {}
for M in MS:
    inputs, nw = make_inputs(M)
    pre_ext = torch.rand((M, HC), device="cuda").float() + 0.5
    kw = dict(norm_weight=nw, norm_eps=1e-6, pre_mix_ext=pre_ext)
    AutoTuner.get().clear_cache()
    with autotune(skip_dynamic_tuning_buckets=True):
        mhc_cuda.mhc_fused_hc(*inputs, *ARGS, **kw)
    torch.cuda.synchronize()
    tac = tuple(mhc_cuda.mhc_fused_hc_tactic(*inputs, *ARGS, **kw))
    rn = mhc_cuda.MhcFusedHcRunner(*ARGS)
    offered = [tuple(t) for t in rn.get_valid_tactics(inputs, None)]
    print(f"M={M}: offered {offered}", flush=True)
    fn = lambda: mhc_cuda.mhc_fused_hc(*inputs, *ARGS, **kw)  # noqa: E731
    med, nz, reps = measure(fn)
    out[str(M)] = {"kernel_ms": med, "noise_floor": nz, "tactic": list(tac), "rep_medians_us": reps,
                   "id72_offered": any(t[4] == 72 for t in offered), "id18_offered": any(t[4] == 18 for t in offered)}
    print(f"M={M} cold: tactic {tac} {med*1000:.1f} us noise {nz:.3f} reps {reps}", flush=True)
    bs = 512 if M < 512 else 256
    fixed = {}
    for t in [("fused_half_fma", 12, 1, bs, 2), ("fused_half_fma", 12, 1, bs, 18),
              ("fused_half_fma", 24, 1, bs, 20), ("fused_half_fma", 24, 1, bs, 72)]:
        if t not in offered:
            continue
        ftn = lambda t=t: rn.forward(inputs, tactic=list(t), **kw)  # noqa: E731
        fm, _, _ = measure(ftn)
        fixed[str(t)] = round(fm * 1000, 1)
    out[str(M)]["fixed_tactic_cold_us"] = fixed
    print(f"M={M}: fixed tactics cold {fixed}", flush=True)
    if M == 128:
        med, nz, reps = measure(fn, cold=False)
        out["128_warm"] = {"kernel_ms": med, "noise_floor": nz, "tactic": list(tac), "rep_medians_us": reps}
        print(f"M=128 warm: {med*1000:.1f} us noise {nz:.3f} reps {reps}", flush=True)
        # L20-39 tail-graph proxy: 20 distinct layer input sets, one graph, tuned tactic from the cache.
        layers = []
        for li in range(GRAPH_LAYERS):
            li_inputs, li_nw = make_inputs(M, seed=1000 + li)
            li_pe = torch.rand((M, HC), device="cuda").float() + 0.5
            layers.append((li_inputs, dict(norm_weight=li_nw, norm_eps=1e-6, pre_mix_ext=li_pe)))
        gtac = tuple(mhc_cuda.mhc_fused_hc_tactic(*layers[0][0], *ARGS, **layers[0][1]))
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for li_inputs, li_kw in layers:
                mhc_cuda.mhc_fused_hc(*li_inputs, *ARGS, **li_kw)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            gouts = [mhc_cuda.mhc_fused_hc(*li_inputs, *ARGS, **li_kw) for li_inputs, li_kw in layers]
        med, nz, reps = measure(graph.replay, cold=True, scale=GRAPH_LAYERS)
        out["128_graph"] = {"kernel_ms": med, "noise_floor": nz, "tactic": list(gtac), "rep_medians_us": reps,
                            "note": f"per call, {GRAPH_LAYERS}-call graph replay / {GRAPH_LAYERS}"}
        print(f"M=128 graph: tactic {gtac} {med*1000:.1f} us/call noise {nz:.3f} reps {reps}", flush=True)
        del graph, gouts, layers
    del inputs
    torch.cuda.empty_cache()
env_log()
print(json.dumps({"batch": out, "output": {"class": "exact"}}))
