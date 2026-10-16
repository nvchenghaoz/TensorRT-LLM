import json, sys, time, torch
import tensorrt_llm
from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.custom_ops.torch_custom_ops import NVFP4GemmUnifiedRunner as U
# Tunes with this tree's NVFP4GemmUnifiedRunner.tuning_config, then times each picked
# (backend, tactic) cold: one CUDA graph of nb back-to-back calls over distinct weight copies
# (>= 400 MB rotated per replay), 10 warmup + 60 timed replays, per-call median/p90.
B = "cutlass,cublaslt,cuda_core"
MS = (1, 2, 3, 4, 8, 2048, 8192)
SH = (("gate_up", 34816, 5120), ("down", 5120, 17408))
def log(*a): print(*a, flush=True)
def pct(v, q):
    v = sorted(v); return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]
def q(x, gs): return torch.ops.trtllm.fp4_quantize(x, gs, 16, False, True)
cfg = U.tuning_config
log("tree use_cold_l2_cache=%s" % cfg.use_cold_l2_cache)
g = torch.Generator(device="cuda").manual_seed(0)
shapes = []
for nm, n, k in SH:
    w = (torch.randn(n, k, generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    gw = (2688.0 / w.float().abs().max()).reshape(1).float()
    wq, ws = q(w, gw); del w
    wb = wq.numel() + ws.numel(); nb = max(2, -(-(400 << 20) // wb))
    cps = [(wq, ws)] + [(wq.clone(), ws.clone()) for _ in range(nb - 1)]
    acts = dict()
    for m in MS:
        a = torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16)
        ga = (2688.0 / a.float().abs().max()).reshape(1).float()
        af, asf = q(a, ga)
        acts[m] = (af, asf, (1.0 / (ga * gw)).float())
    shapes.append((nm, n, k, wb, nb, cps, acts))
    log("%s N=%d K=%d %.1f MB x %d copies" % (nm, n, k, wb / 1e6, nb))
tuner = AutoTuner.get()
tuner.clear_cache()
torch.cuda.synchronize(); base = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
tune_s = dict()
for nm, n, k, wb, nb, cps, acts in shapes:
    t0 = time.perf_counter()
    with autotune():
        af, asf, al = acts[8192]
        torch.ops.trtllm.nvfp4_gemm(af, cps[0][0], asf, cps[0][1], al, torch.bfloat16, allowed_backends=B)
    torch.cuda.synchronize(); tune_s[nm] = time.perf_counter() - t0
peak = (torch.cuda.max_memory_allocated() - base) / 1e6
log("tune_s %s peak_extra_MB %.1f" % (tune_s, peak))
r = U(0, torch.bfloat16, B.split(","))
def inp(acts, cps, m, i):
    af, asf, al = acts[m]; w, ws = cps[i]
    return [af, w, asf, ws, al]
batch = dict(); detail = dict()
for m in MS:
    tot = 0.0; nz = 0.0; d = dict()
    for nm, n, k, wb, nb, cps, acts in shapes:
        hit, rid, tac, tms = tuner.profiling_cache.search_cache(
            "trtllm::nvfp4_gemm::gemm", [r], tuple(tuner._get_input_sizes(inp(acts, cps, m, 0))), cfg)
        for i in range(nb): r(inp(acts, cps, m, i), tactic=tac)
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for i in range(nb): r(inp(acts, cps, m, i), tactic=tac)
        torch.cuda.current_stream().wait_stream(st)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            outs = [r(inp(acts, cps, m, i), tactic=tac) for i in range(nb)]
        for _ in range(10): gr.replay()
        e0 = [torch.cuda.Event(enable_timing=True) for _ in range(60)]
        e1 = [torch.cuda.Event(enable_timing=True) for _ in range(60)]
        for i in range(60):
            torch.cuda._sleep(100000); e0[i].record(); gr.replay(); e1[i].record()
        torch.cuda.synchronize()
        ts = [e0[i].elapsed_time(e1[i]) / nb for i in range(60)]
        med = pct(ts, 0.5); p90 = pct(ts, 0.9); nzs = (p90 - pct(ts, 0.1)) / med
        tot += med; nz = max(nz, nzs)
        d[nm] = dict(tactic=list(tac), hit=bool(hit), tuner_ms=tms, median_ms=med, p90_ms=p90,
                     TBps=wb / (med * 1e-3) / 1e12, noise=nzs)
        log("M=%d %s tactic %s: median %.4f ms p90 %.4f ms %.3f TB/s noise %.3f" %
            (m, nm, tac, med, p90, wb / (med * 1e-3) / 1e12, nzs))
        del gr, outs
    batch[str(m)] = dict(kernel_ms=round(tot, 5), noise_floor=round(nz, 4))
    detail[str(m)] = d
print(json.dumps(dict(batch=batch, output=dict([("class", "reordered")]),
                      use_cold_l2_cache=cfg.use_cold_l2_cache, tune_s=tune_s,
                      tune_peak_extra_mb=peak, detail=detail)))
