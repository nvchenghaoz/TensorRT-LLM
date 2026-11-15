"""Idea 9 / branch 30: trtllm::nvfp4_gemm kernel pair and repeated-execution check.

Runs unchanged on main (base arm) and on the branch (candidate arm). Synthetic
operands of the model's MLP shapes (read from <checkpoint>/config.json),
quantized with trtllm::fp4_quantize. Tuning is what the engine's warmup does:
one call per shape at M=8192 under autotune(), on fresh inputs.

modes
  pair   time the op per (shape, M), L2 flushed (256 MiB write) before every
         warmup/timed iteration; 5 warmup + 30 timed, CUDA events, median/p10/p90.
  check  untimed sequence with the same warmup+iters per case, shape order A,B,A:
         every output int16-equal to call 1, to main's pick (switch OFF in the
         same process) and FP32-reference error within 1.10x main's + 1 ulp;
         inputs unchanged. Seeds 1, 2 and heavy-tailed randn*2^U(-8,8) rows.
  smoke  quick functional test of the dense policy (branch only).
"""
import argparse
import json
import os
import sys
import time

import torch

M_PREFILL = [300, 301, 1000, 1023, 1200, 1856, 2000, 4000, 4096, 4800, 8000, 8100, 8192]
M_DECODE = [1, 8, 32, 64]
WARMUP = 5
ITERS = 30


def log(*a):
    print(*a, flush=True)


def read_dims(ckpt):
    with open(os.path.join(ckpt, "config.json")) as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)
    hidden, inter = tc["hidden_size"], tc["intermediate_size"]
    q = {}
    p = os.path.join(ckpt, "hf_quant_config.json")
    if os.path.exists(p):
        with open(p) as f:
            q = json.load(f)
    q = q.get("quantization", q)
    log(f"[dims] hidden={hidden} intermediate={inter} layers={tc.get('num_hidden_layers')} "
        f"quant_algo={q.get('quant_algo')} group_size={q.get('group_size')} kv={q.get('kv_cache_quant_algo')}")
    return {"gate_up": (2 * inter, hidden), "down": (hidden, inter)}


def quantize(x):
    amax = x.abs().amax().float()
    gs = (448.0 * 6.0 / amax).reshape(1).to(torch.float32)
    q, sf = torch.ops.trtllm.fp4_quantize(x, gs, 16, False)
    return q, sf, gs


_LUT = None


def dequant(q, sf, gs, rows, k):
    global _LUT
    if _LUT is None:
        _LUT = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6],
                            dtype=torch.float32, device=q.device)
    qb = q.view(torch.uint8)
    lo = (qb & 0xF).long()
    hi = (qb >> 4).long()
    vals = torch.stack([_LUT[lo], _LUT[hi]], dim=-1).reshape(rows, k)
    rows_pad = (rows + 127) // 128 * 128
    sf_lin = torch.ops.trtllm.block_scale_interleave_reverse(sf.view(torch.uint8).reshape(rows_pad, -1))
    sf_f = sf_lin[:rows, :k // 16].view(torch.float8_e4m3fn).float()
    return vals * sf_f.repeat_interleave(16, dim=1) / gs


def make_weight(n, k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(n, k, generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    wq, wsf, wgs = quantize(w)
    return wq, wsf, wgs


def make_act(m, k, seed, heavy=False):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, generator=g, device="cuda")
    if heavy:
        e = torch.rand(m, 1, generator=g, device="cuda") * 16 - 8
        x = x * torch.exp2(e)
    return quantize(x.to(torch.bfloat16))


BACKENDS = "cutlass,cublaslt,cutedsl,cuda_core"


def gemm(xq, xsf, wq, wsf, alpha):
    return torch.ops.trtllm.nvfp4_gemm(xq, wq, xsf, wsf, alpha, torch.bfloat16, allowed_backends=BACKENDS)


def tco():
    from tensorrt_llm._torch.custom_ops import torch_custom_ops
    return torch_custom_ops


def has_dense():
    return hasattr(tco(), "NVFP4_DENSE_TUNING")


def set_dense(flag):
    if has_dense():
        tco().NVFP4_DENSE_TUNING = flag


def tune(weights, dims):
    from tensorrt_llm._torch.autotuner import autotune
    t0 = time.time()
    with autotune():
        for name, (wq, wsf, wgs) in weights.items():
            n, k = dims[name]
            xq, xsf, xgs = make_act(8192, k, 7)
            alpha = (1.0 / (xgs * wgs)).float()
            t1 = time.time()
            gemm(xq, xsf, wq, wsf, alpha)
            torch.cuda.synchronize()
            log(f"[tune] {name} {time.time() - t1:.1f}s")
    torch.cuda.synchronize()
    return time.time() - t0


def picks_for(name, m, n, k):
    """Tuner cache entries (main and dense) that the lookup of (name, M) hits."""
    from tensorrt_llm._torch.autotuner import AutoTuner
    t = tco()
    cache = AutoTuner.get().profiling_cache.cache
    res = {}
    m_main = t.last_positive_power_of_2(m)
    want = {("trtllm::nvfp4_gemm::gemm", m_main)}
    if has_dense():
        rng = t._NVFP4_DENSE_SHAPES.get((n, k))
        if rng is not None and rng[0] <= m <= rng[1]:
            want.add(("trtllm::nvfp4_gemm::gemm_dense", t.nvfp4_dense_bucket(m, rng[0])))
    for key, val in cache.items():
        op, shapes = key[0], key[3]
        if shapes[1][0] == n and shapes[0][1] * 2 == k and (op, shapes[0][0]) in want:
            res[op.split("::")[-1] + f"@{shapes[0][0]}"] = str(val[1])
    return res


def timed(fn, flush):
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    for _ in range(WARMUP):
        flush.zero_()
        fn()
    for i in range(ITERS):
        flush.zero_()
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    ts = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    q = lambda f: ts[int(round(f * (len(ts) - 1)))]  # noqa: E731
    return q(0.5), q(0.1), q(0.9)


def gpu_state():
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        return {"sm_clock_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
                "temp_c": pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
                "power_w": pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0,
                "throttle_reasons": int(pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(h))}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def warm_gpu(seconds=10.0):
    """Fixed back-to-back BF16 GEMM load so both arms start timing in the same power/thermal state."""
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    t0 = time.time()
    n = 0
    while time.time() - t0 < seconds:
        for _ in range(10):
            a @ b
        torch.cuda.synchronize()
        n += 10
    log(f"[warm] {n} bf16 8192^3 GEMMs in {time.time() - t0:.1f}s state={gpu_state()}")
    del a, b


REPS = 3


def mode_pair(args, dims, weights):
    """Driver: REPS fresh processes (each warms the GPU, tunes, times); per key the median of the
    process medians. Autotuner picks differ per process, so one process is one sample of them."""
    import subprocess
    reps = []
    for r in range(REPS):
        cmd = [sys.executable, os.path.abspath(__file__), "--mode", "pair1", "--checkpoint", args.checkpoint]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        last = None
        for line in p.stdout:
            sys.stdout.write(f"[rep{r}] {line}")
            sys.stdout.flush()
            if line.startswith("{"):
                last = line
        p.wait()
        reps.append(json.loads(last))
    batch = {}
    for key in reps[0]["batch"]:
        meds = sorted(rp["batch"][key]["kernel_ms"] for rp in reps)
        within = sorted(rp["batch"][key]["noise_floor"] for rp in reps)[len(reps) // 2]
        med = meds[len(meds) // 2]
        across = (meds[-1] - meds[0]) / med
        batch[key] = {"kernel_ms": med, "noise_floor": max(within, across), "within_process_noise": within,
                      "across_process_spread": across, "process_medians_ms": meds,
                      "p10_ms": sorted(rp["batch"][key]["p10_ms"] for rp in reps)[len(reps) // 2],
                      "p90_ms": sorted(rp["batch"][key]["p90_ms"] for rp in reps)[len(reps) // 2],
                      "picks_per_process": [rp["picks"][key] for rp in reps]}
        log(f"[agg] {key} med={med * 1e3:.1f}us procs={[round(m * 1e3, 1) for m in meds]} "
            f"noise={batch[key]['noise_floor']:.4f}")
    print(json.dumps({"batch": batch, "output": {"class": "exact"},
                      "tune_seconds_per_process": [rp["tune_seconds"] for rp in reps],
                      "dense_available": reps[0]["dense_available"], "processes": REPS}))


def mode_pair1(args, dims, weights):
    log(f"[gpu] start {gpu_state()}")
    warm_gpu()
    tune_s = tune(weights, dims)
    log(f"[tune] total {tune_s:.1f}s dense_available={has_dense()}")
    flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    batch = {}
    picks = {}
    for name, (wq, wsf, wgs) in weights.items():
        n, k = dims[name]
        for m in M_DECODE + M_PREFILL:
            xq, xsf, xgs = make_act(m, k, 11)
            alpha = (1.0 / (xgs * wgs)).float()
            med, p10, p90 = timed(lambda: gemm(xq, xsf, wq, wsf, alpha), flush)
            pk = picks_for(name, m, n, k)
            picks[f"{name}@{m}"] = pk
            log(f"[time] {name} M={m} med={med * 1e3:.1f}us p10={p10 * 1e3:.1f} p90={p90 * 1e3:.1f} picks={pk}")
            batch[f"{name}@{m}"] = {"kernel_ms": med, "p10_ms": p10, "p90_ms": p90,
                                    "noise_floor": (p90 - p10) / med}
    log(f"[gpu] end {gpu_state()}")
    print(json.dumps({"batch": batch, "output": {"class": "exact"}, "tune_seconds": tune_s, "picks": picks,
                      "dense_available": has_dense()}))


def int16_mismatch(a, b):
    d = a.view(torch.int16) != b.view(torch.int16)
    cnt = int(d.sum().item())
    first = None
    if cnt:
        idx = d.nonzero()[0].tolist()
        first = {"row": idx[0], "col": idx[1], "a": float(a[idx[0], idx[1]]), "b": float(b[idx[0], idx[1]])}
    return cnt, first


def mode_check(args, dims, weights):
    tune_s = tune(weights, dims)
    from tensorrt_llm._torch.autotuner import AutoTuner
    cache_before = dict(AutoTuner.get().profiling_cache.cache)
    passed = True
    results = {}
    order = []
    for m in M_DECODE + M_PREFILL:
        order += [("gate_up", m), ("down", m), ("gate_up", m)]  # A, B, A
    wrefs = {}
    for name, (wq, wsf, wgs) in weights.items():
        n, k = dims[name]
        wrefs[name] = dequant(wq, wsf, wgs, n, k)
    for variant, seed, heavy in (("seed1", 1, False), ("seed2", 2, False), ("heavy", 3, True)):
        for name, m in order:
            wq, wsf, wgs = weights[name]
            n, k = dims[name]
            xq, xsf, xgs = make_act(m, k, seed + m, heavy)
            alpha = (1.0 / (xgs * wgs)).float()
            snap = [t.clone() for t in (xq, xsf, wq, wsf, alpha)]
            set_dense(True)
            first = gemm(xq, xsf, wq, wsf, alpha)
            rep_bad = 0
            for _ in range(WARMUP + ITERS - 1):
                o = gemm(xq, xsf, wq, wsf, alpha)
                rep_bad += int16_mismatch(o, first)[0]
                del o
            set_dense(False)
            main_out = gemm(xq, xsf, wq, wsf, alpha)
            set_dense(True)
            torch.cuda.synchronize()
            vs_main, fm = int16_mismatch(first, main_out)
            unchanged = all(torch.equal(a.view(torch.uint8) if a.dtype != torch.float32 else a,
                                        b.view(torch.uint8) if b.dtype != torch.float32 else b)
                            for a, b in zip(snap, (xq, xsf, wq, wsf, alpha)))
            finite = bool(torch.isfinite(first.float()).all())
            ok = rep_bad == 0 and vs_main == 0 and unchanged and finite and first.shape == (m, n) \
                and first.dtype == torch.bfloat16
            key = f"{variant}|{name}@{m}"
            if key not in results:
                xref = dequant(xq, xsf, xgs, m, k)
                ref = xref @ wrefs[name].t()
                e_c = first.float() - ref
                e_m = main_out.float() - ref
                ulp = 2.0 ** (torch.floor(torch.log2(ref.abs().max())).item() - 7)
                r = {"maxabs": e_c.abs().max().item(), "relfro": (e_c.norm() / ref.norm()).item(),
                     "main_maxabs": e_m.abs().max().item(), "main_relfro": (e_m.norm() / ref.norm()).item()}
                tol = r["maxabs"] <= 1.10 * r["main_maxabs"] + ulp and r["relfro"] <= 1.10 * r["main_relfro"]
                ok = ok and tol
                r.update({"repeat_mismatch": rep_bad, "vs_main_int16_mismatch": vs_main, "first_mismatch": fm,
                          "inputs_unchanged": unchanged, "finite": finite, "tol_ok": tol, "ok": ok,
                          "picks": picks_for(name, m, n, k)})
                results[key] = r
                del xref, ref, e_c, e_m
                log(f"[check] {key} " + json.dumps(r))
            else:
                results[key]["ok"] = results[key]["ok"] and ok
                results[key]["repeat_mismatch"] += rep_bad
                results[key]["vs_main_int16_mismatch"] += vs_main
            passed &= ok
            if not ok:
                log(f"[check] FAIL {key} rep={rep_bad} vs_main={vs_main} first={fm} unchanged={unchanged}")
            del first, main_out
    cache_after = dict(AutoTuner.get().profiling_cache.cache)
    mutated = cache_before != cache_after
    passed &= not mutated
    print(json.dumps({"passed": bool(passed), "compared": "dense-policy trtllm::nvfp4_gemm vs main's pick "
                      "(switch OFF, same process) int16 on every row; every call vs call 1; FP32 dequant ref; "
                      "inputs unchanged; shape order A,B,A; seeds 1,2 + heavy-tailed",
                      "dense_available": has_dense(), "autotuner_cache_mutated": mutated,
                      "n_cases": len(results), "tune_seconds": tune_s,
                      "fails": [k for k, v in results.items() if not v["ok"]], "results": results}))


def mode_smoke(args, dims, weights):
    t = tco()
    from tensorrt_llm._torch import utils as u
    log(f"[smoke] buckets256={u.nvfp4_dense_buckets(8192, 256)}")
    log(f"[smoke] buckets4096={u.nvfp4_dense_buckets(8192, 4096)}")
    for m in (1, 255, 256, 300, 1000, 1025, 2000, 4000, 4096, 4097, 4800, 8000, 8192, 9000):
        log(f"[smoke] m={m} b256={u.nvfp4_dense_bucket(m, 256)} b4096={u.nvfp4_dense_bucket(m, 4096)}")
    tune_s = tune(weights, dims)
    log(f"[smoke] tune {tune_s:.1f}s")
    for name, (wq, wsf, wgs) in weights.items():
        n, k = dims[name]
        for m in (64, 300, 2000, 4800, 8000):
            xq, xsf, xgs = make_act(m, k, 5)
            alpha = (1.0 / (xgs * wgs)).float()
            set_dense(True)
            a = gemm(xq, xsf, wq, wsf, alpha)
            set_dense(False)
            b = gemm(xq, xsf, wq, wsf, alpha)
            set_dense(True)
            log(f"[smoke] {name} M={m} int16_mismatch={int16_mismatch(a, b)[0]} picks={picks_for(name, m, n, k)}")
    print(json.dumps({"passed": True, "tune_seconds": tune_s}))


def mode_cache(args, dims, weights):
    """phase save: tune, save the autotuner cache to --cache-path; phase load: load it without tuning.
    Both print the picks and an int16 hash of the op's output per (shape, M)."""
    import hashlib
    from tensorrt_llm._torch.autotuner import autotune
    if args.phase == "save" and os.path.exists(args.cache_path):
        os.remove(args.cache_path)
    with autotune(tune_mode=args.phase == "save", cache_path=args.cache_path):
        for name, (wq, wsf, wgs) in weights.items():
            n, k = dims[name]
            xq, xsf, xgs = make_act(8192, k, 7)
            gemm(xq, xsf, wq, wsf, (1.0 / (xgs * wgs)).float())
    torch.cuda.synchronize()
    res = {}
    for name, (wq, wsf, wgs) in weights.items():
        n, k = dims[name]
        for m in M_DECODE + M_PREFILL:
            xq, xsf, xgs = make_act(m, k, 21 + m)
            o = gemm(xq, xsf, wq, wsf, (1.0 / (xgs * wgs)).float())
            torch.cuda.synchronize()
            h = hashlib.sha256(o.view(torch.int16).cpu().numpy().tobytes()).hexdigest()[:16]
            res[f"{name}@{m}"] = {"picks": picks_for(name, m, n, k), "out_sha": h}
    print(json.dumps(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="pair")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--phase", default="save")
    ap.add_argument("--cache-path", default="/tmp/nvfp4_dense_cache.json")
    args = ap.parse_args()
    import tensorrt_llm  # noqa: F401
    dims = read_dims(args.checkpoint)
    if args.mode == "pair":  # driver only: the child processes touch the GPU
        return mode_pair(args, dims, None)
    log(f"[gpu] {torch.cuda.get_device_name()} sm={torch.cuda.get_device_capability()}")
    weights = {}
    for i, name in enumerate(("gate_up", "down")):
        n, k = dims[name]
        log(f"[case] {name} N={n} K={k}")
        weights[name] = make_weight(n, k, 100 + i)
    {"pair1": mode_pair1, "check": mode_check, "smoke": mode_smoke, "cache": mode_cache}[args.mode](
        args, dims, weights)


if __name__ == "__main__":
    main()
