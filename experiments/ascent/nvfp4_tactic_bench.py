"""NVFP4 GEMM (trtllm::nvfp4_gemm) microbenchmark / checks for the cuBLASLt tactic-identity fix.

Modes:
  pair   - autotune once (default backends, buckets 1..8192), then time trtllm::nvfp4_gemm per
           (shape, M) with L2 flushed before every timed iteration. Runs on main and on the branch.
  diag   - branch only: per (shape, M) report the tuned tactic, the cuBLASLt algo main would run
           (heuristic list of actual M), the one the branch runs (list of the bucket M), the one the
           autotuner profiled (bucket M), and time main's algo, the branch's algo and the best cuBLASLt
           algo at that exact M (floor).
  check  - untimed repeated-execution check: the timed sequence (warmup+iters per case) replayed,
           every output compared bitwise with the first call and with an FP32 dequantized reference.
  cache  - acceptance 2: process A tunes and saves the autotuner cache, process B loads it; both
           record the resolved cuBLASLt algorithm per (shape, M); they must match.
Synthetic tensors only; dimensions read from <checkpoint>/config.json.
"""
import argparse
import json
import os
import subprocess
import sys

import torch

M_LIST = [300, 1000, 1200, 2000, 4000, 4800, 8000, 8100, 8192]
WARMUP = 5
ITERS = 30


def log(*a):
    print(*a, flush=True)


def read_dims(ckpt):
    with open(os.path.join(ckpt, "config.json")) as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)
    hidden = tc["hidden_size"]
    inter = tc["intermediate_size"]
    qcfg = {}
    p = os.path.join(ckpt, "hf_quant_config.json")
    if os.path.exists(p):
        with open(p) as f:
            qcfg = json.load(f)
    q = qcfg.get("quantization", qcfg)
    log(f"[dims] hidden={hidden} intermediate={inter} layers={tc.get('num_hidden_layers')} "
        f"quant_algo={q.get('quant_algo')} group_size={q.get('group_size')} "
        f"exclude_modules(sample)={str(q.get('exclude_modules'))[:300]}")
    # NVFP4 linears of the model (attention projections are BF16 per hf_quant_config)
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
    sf_lin = torch.ops.trtllm.block_scale_interleave_reverse(
        sf.view(torch.uint8).reshape(rows_pad, -1))
    sf_f = sf_lin[:rows, :k // 16].view(torch.float8_e4m3fn).float()
    return vals * sf_f.repeat_interleave(16, dim=1) / gs


def make_case(n, k, m_max, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(n, k, generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    x = torch.randn(m_max, k, generator=g, device="cuda").to(torch.bfloat16)
    wq, wsf, wgs = quantize(w)
    del w
    return x, wq, wsf, wgs


def act_for(x, m):
    xq, xsf, xgs = quantize(x[:m].contiguous())
    return xq, xsf, xgs


BACKENDS = "cutlass,cublaslt,cutedsl,cuda_core"


def gemm(xq, xsf, wq, wsf, alpha):
    return torch.ops.trtllm.nvfp4_gemm(xq, wq, xsf, wsf, alpha, torch.bfloat16,
                                       allowed_backends=BACKENDS)


def tune_all(cases, tune=True, cache_path=None):
    from tensorrt_llm._torch.autotuner import autotune
    with autotune(tune_mode=tune, cache_path=cache_path):
        for name, (x, wq, wsf, wgs) in cases.items():
            xq, xsf, xgs = act_for(x, 8192)
            alpha = (1.0 / (xgs * wgs)).float()
            log(f"[tune] {name} tune={tune}")
            gemm(xq, xsf, wq, wsf, alpha)
    torch.cuda.synchronize()


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
    med = ts[len(ts) // 2]
    p10 = ts[int(0.1 * (len(ts) - 1))]
    p90 = ts[int(0.9 * (len(ts) - 1))]
    return med, p10, p90


def tuned_tactic(xq, xsf, wq, wsf, alpha):
    from tensorrt_llm._torch.autotuner import AutoTuner
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import NVFP4GemmUnifiedRunner
    runner = NVFP4GemmUnifiedRunner(0, torch.bfloat16, [b for b in BACKENDS.split(",") if b])
    shapes = tuple(t.shape for t in [xq, wq, xsf, wsf, alpha])
    hit, _, tactic, t = AutoTuner.get().profiling_cache.search_cache(
        "trtllm::nvfp4_gemm::gemm", [runner], shapes, NVFP4GemmUnifiedRunner.tuning_config)
    return hit, tactic, t


def mode_pair(args, cases):
    tune_all(cases)
    flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    batch = {}
    for name, (x, wq, wsf, wgs) in cases.items():
        for m in M_LIST:
            xq, xsf, xgs = act_for(x, m)
            alpha = (1.0 / (xgs * wgs)).float()
            hit, tactic, _ = tuned_tactic(xq, xsf, wq, wsf, alpha)
            med, p10, p90 = timed(lambda: gemm(xq, xsf, wq, wsf, alpha), flush)
            log(f"[time] {name} M={m} tactic={tactic} med={med*1e3:.1f}us p10={p10*1e3:.1f} p90={p90*1e3:.1f}")
            batch[f"{name}@{m}"] = {"kernel_ms": med, "p10_ms": p10, "p90_ms": p90,
                                    "noise_floor": (p90 - p10) / med, "tactic": str(tactic)}
    tot = sum(v["kernel_ms"] for v in batch.values())
    log(f"[total] sum of medians {tot:.4f} ms")
    print(json.dumps({"batch": batch, "output": {"class": "reordered"}}))


def runner_obj():
    return torch.classes.trtllm.CublasLtFP4GemmRunner(torch.bfloat16)


def last_pow2(m):
    from tensorrt_llm._torch.utils import last_positive_power_of_2
    return last_positive_power_of_2(m)


def mode_diag(args, cases):
    tune_all(cases)
    flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    r = runner_obj()
    rows = []
    for name, (x, wq, wsf, wgs) in cases.items():
        for m in M_LIST:
            xq, xsf, xgs = act_for(x, m)
            alpha = (1.0 / (xgs * wgs)).float()
            hit, tactic, tt = tuned_tactic(xq, xsf, wq, wsf, alpha)
            b = last_pow2(m)
            bq, bsf, _ = act_for(x, b)
            n_alg = r.get_num_heuristic_algos(xq, wq, xsf, wsf)
            row = {"shape": name, "M": m, "bucket": b, "tuned": str(tactic), "num_algos_at_M": n_alg}
            sub = tactic[1] if isinstance(tactic, (tuple, list)) else -1
            if isinstance(tactic, (tuple, list)) and tactic[0] == "cublaslt":
                row["profiled_algo"] = r.get_algo_config(bq, wq, bsf, wsf, sub, b)
                row["main_algo"] = r.get_algo_config(xq, wq, xsf, wsf, sub, -1)
                row["branch_algo"] = r.get_algo_config(xq, wq, xsf, wsf, sub, b)
                row["branch_matches_profiled"] = row["branch_algo"][2:] == row["profiled_algo"][2:]
                row["main_matches_profiled"] = row["main_algo"][2:] == row["profiled_algo"][2:]
                row["main_us"] = timed(lambda: r.run_gemm_bucketed(xq, wq, xsf, wsf, alpha, 0, sub, -1, None, None), flush)[0] * 1e3
                row["branch_us"] = timed(lambda: r.run_gemm_bucketed(xq, wq, xsf, wsf, alpha, 0, sub, b, None, None), flush)[0] * 1e3
            # floor: best cuBLASLt algo at exact M (L2 flushed)
            best = (1e9, -1)
            for j in range(n_alg):
                try:
                    t = timed(lambda: r.run_gemm(xq, wq, xsf, wsf, alpha, 0, j, None, None), flush)[0] * 1e3
                except Exception as e:  # noqa
                    continue
                best = min(best, (t, j))
            row["floor_cublaslt_us"], row["floor_idx"] = best
            row["floor_algo"] = r.get_algo_config(xq, wq, xsf, wsf, best[1], -1)
            row["op_us"] = timed(lambda: gemm(xq, xsf, wq, wsf, alpha), flush)[0] * 1e3
            log("[diag] " + json.dumps(row))
            rows.append(row)
    cub = [x for x in rows if "branch_algo" in x]
    ok = all(x["branch_matches_profiled"] for x in cub)
    log(f"[diag] cublaslt-tuned cases={len(cub)} branch_matches_profiled_all={ok} "
        f"main_matches_profiled={sum(x['main_matches_profiled'] for x in cub)}/{len(cub)}")
    print(json.dumps({"passed": ok, "rows": rows}))


def mode_check(args, cases):
    tune_all(cases)
    from tensorrt_llm._torch.autotuner import AutoTuner
    cache_before = dict(AutoTuner.get().profiling_cache.cache)
    passed = True
    results = {}
    for name, (x, wq, wsf, wgs) in cases.items():
        n, k = wq.shape[0], wq.shape[1] * 2
        wref = dequant(wq, wsf, wgs, n, k)
        for m in M_LIST:
            xq, xsf, xgs = act_for(x, m)
            alpha = (1.0 / (xgs * wgs)).float()
            xref = dequant(xq, xsf, xgs, m, k)
            ref = xref @ wref.t()
            outs = []
            for _ in range(WARMUP + ITERS):
                outs.append(gemm(xq, xsf, wq, wsf, alpha))
            torch.cuda.synchronize()
            first = outs[0]
            bitwise = all(torch.equal(o, first) for o in outs[1:])
            finite = bool(torch.isfinite(first).all())
            err = (first.float() - ref)
            maxabs = err.abs().max().item()
            rel = (err.norm() / ref.norm()).item()
            # also the main-semantics cuBLASLt op output of the same tuned tactic for comparison
            ok = bitwise and finite and first.shape == (m, n) and first.dtype == torch.bfloat16 and rel < 3e-2
            passed &= ok
            results[f"{name}@{m}"] = {"bitwise_repeat": bitwise, "finite": finite,
                                      "maxabs_vs_fp32": maxabs, "relfro_vs_fp32": rel}
            log(f"[check] {name} M={m} repeat_bitwise={bitwise} finite={finite} maxabs={maxabs:.4g} rel={rel:.4g}")
            del ref, xref, outs
        del wref
    cache_after = dict(AutoTuner.get().profiling_cache.cache)
    mutated = cache_before != cache_after
    log(f"[check] autotuner cache entries={len(cache_after)} mutated_during_sequence={mutated}")
    passed &= not mutated
    print(json.dumps({"passed": bool(passed), "compared": "every call vs first (bitwise) and vs FP32 dequant reference",
                      "results": results, "autotuner_cache_mutated": mutated}))


def resolved_table(cases):
    r = runner_obj()
    tab = {}
    for name, (x, wq, wsf, wgs) in cases.items():
        for m in M_LIST:
            xq, xsf, xgs = act_for(x, m)
            alpha = (1.0 / (xgs * wgs)).float()
            hit, tactic, _ = tuned_tactic(xq, xsf, wq, wsf, alpha)
            is_seq = isinstance(tactic, (tuple, list))
            entry = {"hit": hit, "tactic": list(tactic) if is_seq else tactic}
            if is_seq and tactic[0] == "cublaslt":
                entry["algo"] = r.get_algo_config(xq, wq, xsf, wsf, tactic[1], last_pow2(m))
            else:
                # what the branch would run if cuBLASLt tactic i were picked: record bucket list top entries
                entry["algo_t0"] = r.get_algo_config(xq, wq, xsf, wsf, 0, last_pow2(m))
            tab[f"{name}@{m}"] = entry
    return tab


def mode_cache(args, cases):
    path = args.cache_path
    if args.phase == "save":
        if os.path.exists(path):
            os.remove(path)
        tune_all(cases, tune=True, cache_path=path)
        print(json.dumps(resolved_table(cases)))
        return
    if args.phase == "load":
        tune_all(cases, tune=False, cache_path=path)
        print(json.dumps(resolved_table(cases)))
        return
    # driver
    outs = {}
    for phase in ("save", "load"):
        cmd = [sys.executable, __file__, "--mode", "cache", "--phase", phase, "--cache-path", path,
               "--checkpoint", args.checkpoint, "--backends", BACKENDS]
        log(f"[cache] running {phase}")
        p = subprocess.run(cmd, capture_output=True, text=True)
        sys.stdout.write(p.stdout[-4000:])
        sys.stderr.write(p.stderr[-4000:])
        outs[phase] = json.loads(p.stdout.strip().splitlines()[-1])
    same = outs["save"] == outs["load"]
    for key in outs["save"]:
        log(f"[cache] {key} save={outs['save'][key]} load={outs['load'].get(key)}")
    print(json.dumps({"passed": same, "compared": "resolved cuBLASLt algo per (shape,M) after save vs after reload",
                      "cache_file_exists": os.path.exists(path)}))


def main():
    global BACKENDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="pair")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--phase", default="driver")
    ap.add_argument("--cache-path", default="/tmp/nvfp4_tactic_cache.json")
    ap.add_argument("--shapes", default="gate_up,down")
    ap.add_argument("--backends", default=BACKENDS)
    args = ap.parse_args()
    BACKENDS = args.backends
    import tensorrt_llm  # noqa: F401  (loads torch ops)
    dims = read_dims(args.checkpoint)
    log(f"[gpu] {torch.cuda.get_device_name()} sm={torch.cuda.get_device_capability()}")
    cases = {}
    for name in args.shapes.split(","):
        n, k = dims[name]
        log(f"[case] {name} N={n} K={k}")
        cases[name] = make_case(n, k, 8192, seed={"gate_up": 1, "down": 2}[name])
    {"pair": mode_pair, "diag": mode_diag, "check": mode_check, "cache": mode_cache}[args.mode](args, cases)


if __name__ == "__main__":
    main()
