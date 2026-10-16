# cudaCoreGemmFp4 (NVFP4 W4A4 decode GEMV) chain microbenchmark + exactness check.
# Self-contained: runs unchanged on main and on the candidate, so run_gpu_pair can pass it
# inline with `python -c`. The arm is detected from the op schema (act_sf_swizzled arg).
#
# Chain timed per call = what CudaCoreNVFP4Runner.forward runs in serving:
#   main:      fp4_quantize -> block_scale_interleave_reverse -> cuda_core_nvfp4_gemm
#   candidate: fp4_quantize -> cuda_core_nvfp4_gemm(act_sf_swizzled=True)
# Shapes (N, K): gate_up (34816, 5120), down (5120, 17408). Cold L2: one CUDA graph per
# (shape, M) holds ROUNDS passes over COPIES distinct weight copies (>= 400 MB rotated,
# L2 is 128 MiB), PDL on (TRTLLM_ENABLE_PDL unset). WARMUP replays, then ITERS timed replays
# with CUDA events; variants are interleaved per iteration.
# Variants: chain (runner.forward), lin (reverse + kernel with linear act-SF read: main's
# exact chain; on the candidate it isolates the pre-PDL-load part), quant (fp4_quantize only),
# tuned (torch.ops.trtllm.nvfp4_gemm with the AutoTuner's pick, default backends),
# cublaslt (nvfp4_gemm restricted to cublaslt).
# Modes (argv[1]): bench (default) | check <expected_hashes_json>
import hashlib
import json
import os
import sys

os.environ.setdefault("TLLM_AUTOTUNER_LOG_LEVEL_DEBUG_TO_INFO", "1")

import torch  # noqa: E402

import tensorrt_llm  # noqa: E402,F401
from tensorrt_llm._torch.autotuner import AutoTuner, autotune  # noqa: E402
from tensorrt_llm._torch.custom_ops.torch_custom_ops import (  # noqa: E402
    CudaCoreNVFP4Runner, NVFP4GemmUnifiedRunner)


def log(*a):
    print(*a, flush=True)


MODE = sys.argv[1] if len(sys.argv) > 1 else "bench"
dev = torch.device("cuda")
SHAPES = {"gate_up": (34816, 5120), "down": (5120, 17408)}
COPIES = {"gate_up": 4, "down": 8}
ROUNDS = 2
WARMUP, ITERS = 10, 60
TIME_MS = [1, 2, 3, 4]
BATCH_MS = [1, 2, 4]
CHECK_MS = [1, 2, 3, 4, 5, 6, 7, 8]
TUNE_MS = [1, 2, 4, 8]
FP8_MAX, E2M1_MAX = 448.0, 6.0
DEFAULT_BACKENDS = "cutlass,cublaslt,cuda_core"
ARGS = [a.name for a in torch.ops.trtllm.cuda_core_nvfp4_gemm.default._schema.arguments]
IS_CAND = "act_sf_swizzled" in ARGS
ARM = "candidate" if IS_CAND else "baseline"
runner = CudaCoreNVFP4Runner(0, torch.bfloat16)


def sha(t):
    t = t.contiguous()
    return hashlib.sha256(t.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:20]


def make_weight(n, k, seed):
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    w = torch.randn(n, k, device=dev, generator=g).to(torch.bfloat16)
    w.view(n, k // 16, 16)[::5, ::11, :] = 0  # zero-SF blocks
    gs = (FP8_MAX * E2M1_MAX / w.float().abs().amax()).reshape(1).float()
    wq, wsf = torch.ops.trtllm.fp4_quantize(w, gs, 16, False, True)
    del w
    return wq, wsf, gs


def make_act(m, k, seed):
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    x = (torch.randn(m, k, device=dev, generator=g) * 2.0).to(torch.bfloat16)
    x.view(m, k // 16, 16)[:, 3::7, :] = 0  # zero-SF blocks
    gs = (FP8_MAX * E2M1_MAX / x.float().abs().amax()).reshape(1).float()
    return x, gs


def quant(x, gsa):
    return torch.ops.trtllm.fp4_quantize(x, gsa, 16, False, True)


def v_chain(x, gsa, wq, wsf, alpha):
    aq, asf = quant(x, gsa)
    return runner.forward([aq, wq, asf, wsf, alpha], tactic=0)


def v_lin(x, gsa, wq, wsf, alpha):
    aq, asf = quant(x, gsa)
    m = aq.shape[0]
    lin = torch.ops.trtllm.block_scale_interleave_reverse(asf.view((m + 127) // 128 * 128, -1))
    return torch.ops.trtllm.cuda_core_nvfp4_gemm(aq, wq, lin, wsf, alpha, None, torch.bfloat16)


def v_quant(x, gsa, wq, wsf, alpha):
    return quant(x, gsa)[0]


def v_tuned(x, gsa, wq, wsf, alpha):
    aq, asf = quant(x, gsa)
    return torch.ops.trtllm.nvfp4_gemm(aq, wq, asf, wsf, alpha, torch.bfloat16)


def v_cublaslt(x, gsa, wq, wsf, alpha):
    aq, asf = quant(x, gsa)
    return torch.ops.trtllm.nvfp4_gemm(aq, wq, asf, wsf, alpha, torch.bfloat16,
                                       allowed_backends="cublaslt")


VARIANTS = {"chain": v_chain, "lin": v_lin, "quant": v_quant, "tuned": v_tuned,
            "cublaslt": v_cublaslt}


def build():
    data = {}
    for name, (n, k) in SHAPES.items():
        wq, wsf, gsw = make_weight(n, k, seed=17 if name == "gate_up" else 29)
        copies = [(wq, wsf)] + [(wq.clone(), wsf.clone()) for _ in range(COPIES[name] - 1)]
        wbytes = wq.numel() + wsf.numel()
        log(f"[{ARM}] {name} N={n} K={k}: {COPIES[name]} copies x {wbytes / 1e6:.1f} MB = "
            f"{COPIES[name] * wbytes / 1e6:.0f} MB rotated; wq {sha(wq)} wsf {sha(wsf)}")
        data[name] = dict(n=n, k=k, copies=copies, gsw=gsw, wbytes=wbytes)
    return data


def calls_for(d, m):
    ncall = COPIES_ROUNDS(d)
    out = []
    for i in range(ncall):
        x, gsa = make_act(m, d["k"], seed=1000 * m + i)
        alpha = (1.0 / (gsa * d["gsw"])).float()
        wq, wsf = d["copies"][i % len(d["copies"])]
        out.append((x, gsa, wq, wsf, alpha))
    return out


def COPIES_ROUNDS(d):
    return len(d["copies"]) * ROUNDS


def capture(fn, calls):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for c in calls[:2]:
            fn(*c)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = [fn(*c) for c in calls]
    torch.cuda.synchronize()
    return g, outs


def tune(data):
    picks = {}
    tuner = AutoTuner.get()
    for name, d in data.items():
        wq, wsf = d["copies"][0]
        for m in TUNE_MS:
            x, gsa = make_act(m, d["k"], seed=7 + m)
            aq, asf = quant(x, gsa)
            alpha = (1.0 / (gsa * d["gsw"])).float()
            for backends in (DEFAULT_BACKENDS, "cublaslt"):
                with autotune(skip_dynamic_tuning_buckets=True):
                    torch.ops.trtllm.nvfp4_gemm(aq, wq, asf, wsf, alpha, torch.bfloat16,
                                                allowed_backends=backends)
                torch.cuda.synchronize()
                r = NVFP4GemmUnifiedRunner(0, torch.bfloat16, backends.split(","))
                inputs = [aq, wq, asf, wsf, alpha]
                try:
                    hit, rid, tac, t = tuner.profiling_cache.search_cache(
                        "trtllm::nvfp4_gemm::gemm", [r], tuple(tuner._get_input_sizes(inputs)),
                        NVFP4GemmUnifiedRunner.tuning_config, apply_map_to_tuning_buckets=True)
                    rec = {"tactic": str(tac), "hit": bool(hit),
                           "tuner_ms": None if t is None else float(t)}
                except Exception as e:  # pragma: no cover - diagnostics only
                    _, tac = tuner.choose_one("trtllm::nvfp4_gemm::gemm", [r],
                                              NVFP4GemmUnifiedRunner.tuning_config, inputs)
                    rec = {"tactic": str(tac), "err": repr(e)[:80]}
                key = f"{name}/M{m}/" + ("default" if backends == DEFAULT_BACKENDS else backends)
                picks[key] = rec
                log(f"[{ARM}] autotuner pick {key}: {rec}")
    return picks


def stats(ts):
    s = sorted(ts)
    q = lambda p: s[min(len(s) - 1, int(round(p * (len(s) - 1))))]  # noqa: E731
    med = q(0.5)
    return med, q(0.9), q(0.1)


def probe():
    import subprocess
    res = {}
    try:
        q = ("name,ecc.mode.current,clocks.sm,clocks.mem,clocks_throttle_reasons.active,"
             "temperature.gpu,power.draw")
        cmd = ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"]
        if os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-"):
            cmd += ["-i", os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]]
        res["nvidia_smi"] = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # pragma: no cover
        res["nvidia_smi"] = repr(e)[:100]
    for label, nbytes, fn in (("read_2GB", 2 << 30, None), ("d2d_1GB", 1 << 30, "copy")):
        src = torch.ones(nbytes // 4, dtype=torch.float32, device=dev)
        dst = torch.empty_like(src) if fn == "copy" else None
        ts = []
        for it in range(25):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            if fn == "copy":
                dst.copy_(src)
            else:
                src.sum()
            e1.record()
            e1.synchronize()
            if it >= 5:
                ts.append(e0.elapsed_time(e1))
        med, p90, _ = stats(ts)
        mult = 2 if fn == "copy" else 1
        res[label + "_TBps_med"] = round(mult * nbytes / (med * 1e-3) / 1e12, 3)
        res[label + "_TBps_p90"] = round(mult * nbytes / (p90 * 1e-3) / 1e12, 3)
        del src, dst
    torch.cuda.empty_cache()
    log(f"[{ARM}] probe: " + json.dumps(res))
    return res


def bench():
    dram = probe()
    data = build()
    picks = tune(data)
    hashes = {}
    detail = {}
    batch = {}
    for name, d in data.items():
        for m in CHECK_MS:
            calls = calls_for(d, m)
            outs = [v_chain(*c) for c in calls]
            torch.cuda.synchronize()
            hashes[f"{name}/M{m}"] = sha(torch.stack(outs))
        for m in TIME_MS:
            calls = calls_for(d, m)
            ncall = len(calls)
            graphs = {}
            for vn, fn in VARIANTS.items():
                graphs[vn], _ = capture(fn, calls)
            ts = {vn: [] for vn in graphs}
            for _ in range(WARMUP):
                for g in graphs.values():
                    g.replay()
            torch.cuda.synchronize()
            for _ in range(ITERS):
                for vn, g in graphs.items():
                    e0 = torch.cuda.Event(enable_timing=True)
                    e1 = torch.cuda.Event(enable_timing=True)
                    e0.record()
                    g.replay()
                    e1.record()
                    e1.synchronize()
                    ts[vn].append(e0.elapsed_time(e1) / ncall)
            rec = {}
            for vn, t in ts.items():
                med, p90, p10 = stats(t)
                rec[vn] = {"us": round(med * 1e3, 2), "p90_us": round(p90 * 1e3, 2),
                           "p10_us": round(p10 * 1e3, 2), "nf": round((p90 - p10) / med, 4)}
                if vn != "quant":
                    tb = d["wbytes"] / (med * 1e-3) / 1e12
                    rec[vn]["TBps"] = round(tb, 3)
                    rec[vn]["pct_read"] = round(100 * tb / dram["read_2GB_TBps_med"], 1)
            rec["gemv_exposed_us"] = round(rec["chain"]["us"] - rec["quant"]["us"], 2)
            detail[f"{name}/M{m}"] = rec
            log(f"[{ARM}] {name} M={m} ({ncall} calls/replay): " + json.dumps(rec))
            if m in BATCH_MS:
                # kernel_ms = one dense-MLP layer's NVFP4 GEMV chains: gate_up + down (medians).
                med, p90, p10 = stats(ts["chain"])
                b = batch.setdefault(str(m), {"kernel_ms": 0.0, "noise_floor": 0.0, "p90_ms": 0.0})
                b["kernel_ms"] = round(b["kernel_ms"] + med, 6)
                b["p90_ms"] = round(b["p90_ms"] + p90, 6)
                b["noise_floor"] = max(b["noise_floor"], round((p90 - p10) / med, 4))
                b[name + "_ms"] = round(med, 6)
            del graphs
            torch.cuda.synchronize()
    res = {"batch": batch, "output": {"class": "exact"}, "arm": ARM, "hashes": hashes,
           "picks": picks, "dram": dram, "detail": detail}
    log(json.dumps(res))


def dequant(q, sf, gs, rows, k):
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
                       device=dev, dtype=torch.float32)
    lo = lut[(q & 0xF).long()]
    hi = lut[(q >> 4).long()]
    v = torch.stack([lo, hi], dim=-1).reshape(rows, k)
    ncol = k // 16
    ntk = (ncol + 3) // 4
    r = torch.arange(rows, device=dev).view(-1, 1)
    c = torch.arange(ncol, device=dev).view(1, -1)
    idx = ((r // 128) * ntk + c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4
    s = sf.view(-1)[idx].view(torch.float8_e4m3fn).float()
    return v * s.repeat_interleave(16, dim=1) / gs


def check(expected):
    data = build()
    ok = True
    report = {"arm": ARM, "cases": {}}
    for name, d in data.items():
        wsnap = [(sha(wq), sha(wsf)) for wq, wsf in d["copies"]]
        wdq = dequant(d["copies"][0][0], d["copies"][0][1], d["gsw"], d["n"], d["k"])
        for m in CHECK_MS:
            key = f"{name}/M{m}"
            calls = calls_for(d, m)
            xsnap = [sha(c[0]) for c in calls]
            eager = [v_chain(*c) for c in calls]
            torch.cuda.synchronize()
            h = sha(torch.stack(eager))
            match_main = expected.get(key) == h
            # (a) alone: swizzled read == reverse + linear read on the same kernel
            lin = [v_lin(*c) for c in calls]
            a_equal = all(torch.equal(e, l) for e, l in zip(eager, lin))
            # FP32 reference from the identically quantized operands (advisory)
            aq, asf = quant(calls[0][0], calls[0][1])
            adq = dequant(aq, asf, calls[0][1], m, d["k"])
            ref = adq @ wdq.t()
            err = (eager[0].float() - ref).abs()
            # repeated execution: CUDA graph, same warmup/iteration count as timing
            g, gouts = capture(v_chain, calls)
            first = None
            rep_ok = True
            max_dev_eager = 0.0
            for it in range(WARMUP + ITERS):
                for o in gouts:  # NaN-poison every output before each replay
                    o.fill_(float("nan"))
                g.replay()
                torch.cuda.synchronize()
                cur = torch.stack(gouts)
                if first is None:
                    first = cur.clone()
                if not torch.equal(cur, first):
                    rep_ok = False
                dev_e = (cur.float() - torch.stack(eager).float()).abs().max().item()
                max_dev_eager = max(max_dev_eager, dev_e)
            graph_eq_eager = torch.equal(first, torch.stack(eager))
            inputs_ok = ([sha(c[0]) for c in calls] == xsnap
                         and [(sha(wq), sha(wsf)) for wq, wsf in d["copies"]] == wsnap)
            # eager repeated calls identical too
            eager2 = [v_chain(*c) for c in calls]
            eager_rep = all(torch.equal(a, b) for a, b in zip(eager, eager2))
            case_ok = (match_main and a_equal and rep_ok and graph_eq_eager and inputs_ok
                       and eager_rep and bool(torch.isfinite(torch.stack(eager).float()).all()))
            ok = ok and case_ok
            report["cases"][key] = {
                "ok": case_ok, "hash": h, "main_hash": expected.get(key), "bitwise_vs_main": match_main,
                "swz_eq_reverse_lin": a_equal, "graph_replays": WARMUP + ITERS,
                "replays_eq_iter1": rep_ok, "graph_eq_eager": graph_eq_eager,
                "max_dev_vs_eager": max_dev_eager, "eager_repeat_eq": eager_rep,
                "inputs_unchanged": inputs_ok,
                "ref_max_abs": round(err.max().item(), 5), "ref_mean_abs": round(err.mean().item(), 6),
                "out_mean_abs": round(eager[0].float().abs().mean().item(), 4)}
            log(f"[{ARM}] check {key}: " + json.dumps(report["cases"][key]))
            del g, gouts
        del wdq
    report["passed"] = ok
    report["compared_against"] = ("main's fp4_quantize -> block_scale_interleave_reverse -> "
                                  "cuda_core_nvfp4_gemm output hashes (baseline arm of the pair)")
    log(json.dumps(report))


if MODE == "check":
    check(json.loads(sys.argv[2]))
else:
    bench()
