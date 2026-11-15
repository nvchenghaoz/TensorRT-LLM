"""Idea 116 (branch 50): NVFP4 tactic picks across fresh engine lifetimes.

usage:
  lifetimes.py drive <n_lifetimes> [tag]   parent: n fresh processes in sequence, aggregate
  lifetimes.py life <out.json>              one engine lifetime (child)

Each lifetime: in-process LLM (TLLM_WORKER_USE_SINGLE_PROCESS=1) with the same settings as
experiments/ascent/bcg_prefill/bcg_ab.py (FP8 KV, block reuse off, max_num_tokens 8192,
max_batch_size 64, max_seq_len 9216) plus this tree's experiments/ascent-config/engine.json
(BCG prefill graphs). No TLLM_AUTOTUNER_CACHE_PATH / TRTLLM_CACHE_DIR: tunes from scratch.
TLLM_AUTOTUNER_LOG_LEVEL_DEBUG_TO_INFO=1 so every profiled candidate is logged; the parent
parses the NVFP4 MLP candidates (gate_up N=2I, down N=H) into per-bucket top-3 margins.

Per lifetime it records:
  - the tuner cache entries of trtllm::nvfp4_gemm::gemm / ::gemm_dense for both shapes, and
    for cuBLASLt picks the resolved algorithm config at every served M;
  - op-level replay with synthetic operands at served M (same tuner cache): the kernel that
    runs (torch.profiler name + grid), cold-L2 time, and whether the M-chunked gate_up chain
    engages (gate_up_gelu_fp4_chunked returns non-None);
  - the (op, N, M, tactic) dispatches of NVFP4GemmUnifiedRunner during served prompts
    (BCG captures included) and chunked-chain calls;
  - kernels in served steps (torch.profiler over one request of each case);
  - TTFT (llm.generate wall time, 1 output token) of isl300@1, isl2k@1, isl4k@1.
"""
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE_JSON = os.path.join(HERE, "..", "..", "ascent-config", "engine.json")
CKPT = os.environ.get("CKPT") or \
    "/home/scratch.chengzhang/trtllm-dev-gemma4-31b-pf-20261114-163048/models/Gemma-4-31B-IT-NVFP4"

SERVED_M = [300, 1000, 1200, 1500, 1800, 2000, 4000, 4800]
CHUNK_M = [1800, 2000, 4000, 4800]
TTFT_CASES = [("isl300@1", 300, 1), ("isl2k@1", 2000, 1), ("isl4k@1", 4000, 1)]
PROF_CASES = [("isl300@1", 300, 1), ("4x300", 300, 4), ("5x300", 300, 5), ("6x300", 300, 6), ("isl1k@1", 1000, 1),
              ("isl2k@1", 2000, 1), ("isl4k@1", 4000, 1), ("16x300", 300, 16)]
ROUNDS = int(os.environ.get("ROUNDS", "15"))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _hb():
    t0 = time.time()
    while True:
        time.sleep(60)
        print(f"[heartbeat] {time.time() - t0:.0f}s", flush=True)


def dims():
    cfg = json.load(open(os.path.join(CKPT, "config.json")))
    tc = cfg.get("text_config", cfg)
    h, i = tc["hidden_size"], tc["intermediate_size"]
    try:
        q = json.load(open(os.path.join(CKPT, "hf_quant_config.json")))
        q = q.get("quantization", q)
    except Exception:  # noqa: BLE001
        q = {}
    log(f"[dims] hidden={h} intermediate={i} layers={tc.get('num_hidden_layers')} "
        f"quant_algo={q.get('quant_algo')} group_size={q.get('group_size')} kv={q.get('kv_cache_quant_algo')}")
    return {"gate_up": (2 * i, h), "down": (h, i)}


def gpu_state():
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return {"sm_clock_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
                "temp_c": pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
                "power_w": pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0,
                "throttle_reasons": int(pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(h))}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


# ----------------------------------------------------------------------------- child
def life(out_path):
    os.environ["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"
    os.environ["TLLM_AUTOTUNER_LOG_LEVEL_DEBUG_TO_INFO"] = "1"
    os.environ["TLLM_LOG_LEVEL"] = "INFO"
    for v in ("TLLM_AUTOTUNER_CACHE_PATH", "TRTLLM_CACHE_DIR"):
        os.environ.pop(v, None)
    threading.Thread(target=_hb, daemon=True).start()
    D = dims()
    GU, DN = D["gate_up"], D["down"]
    ov = json.load(open(ENGINE_JSON)) if os.path.exists(ENGINE_JSON) else {}
    for k, v in (ov.pop("env", None) or {}).items():
        os.environ[k] = str(v)
    log(f"engine overlay {json.dumps(ov)}; env pins: "
        + json.dumps({k: os.environ.get(k) for k in ("TRTLLM_NVFP4_GATEUP_PIN", "TRTLLM_NVFP4_DOWN_PIN",
                                                     "TRTLLM_GEMMA4_MLP_CHUNK", "TRTLLM_NVFP4_DENSE_TUNING")}))
    log(f"[gpu] start {gpu_state()}")
    import numpy as np
    import torch

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.autotuner import AutoTuner
    from tensorrt_llm._torch.custom_ops import torch_custom_ops as tco
    from tensorrt_llm._torch.models import modeling_gemma4 as mg
    from tensorrt_llm.llmapi import KvCacheConfig

    # ---- dispatch recording
    disp = {}
    _fwd = tco.NVFP4GemmUnifiedRunner.forward

    def fwd(self, inputs, tactic=-1, bias=None, **kw):
        n = int(inputs[1].shape[0])
        if (n, int(inputs[1].shape[1]) * 2) in (GU, DN):
            key = f"{'dense' if self.dense_min_m is not None else 'gemm'}|N{n}|M{int(inputs[0].shape[0])}|{tactic}" \
                  f"|cap{int(torch.cuda.is_current_stream_capturing())}"
            disp[key] = disp.get(key, 0) + 1
        return _fwd(self, inputs, tactic=tactic, bias=bias, **kw)

    tco.NVFP4GemmUnifiedRunner.forward = fwd
    chunk_calls = {"engaged": {}, "declined": {}}
    _chunked = mg.gate_up_gelu_fp4_chunked

    def chunked(act_fp4, *a, **k):
        r = _chunked(act_fp4, *a, **k)
        d = chunk_calls["engaged" if r is not None else "declined"]
        m = int(act_fp4.shape[0])
        d[m] = d.get(m, 0) + 1
        return r

    mg.gate_up_gelu_fp4_chunked = chunked

    t0 = time.time()
    llm = LLM(model=CKPT, kv_cache_config=KvCacheConfig(dtype="fp8", enable_block_reuse=False,
                                                        free_gpu_memory_fraction=0.8),
              max_num_tokens=8192, max_batch_size=64, max_seq_len=9216, **ov)
    startup = time.time() - t0
    log(f"startup {startup:.1f}s; [gpu] {gpu_state()}")
    out = {"startup_s": startup, "overlay": ov, "dispatch_startup": dict(disp),
           "chunk_startup": json.loads(json.dumps(chunk_calls))}

    tuner = AutoTuner.get()
    picks = {}
    for key, val in tuner.profiling_cache.cache.items():
        if key[0] in ("trtllm::nvfp4_gemm::gemm", "trtllm::nvfp4_gemm::gemm_dense"):
            shapes = key[3]
            n, kk = shapes[1][0], shapes[0][1] * 2
            if (n, kk) in (GU, DN):
                picks[f"{key[0].split('::')[-1]}|N{n}|M{shapes[0][0]}"] = {
                    "tactic": str(val[1]), "time_ms": float(val[2]), "key": str(key)[:300]}
    out["picks"] = picks
    for k in sorted(picks):
        log(f"[pick] {k} -> {picks[k]['tactic']} {picks[k]['time_ms']:.4f}ms")

    # ---- synthetic operands of the real shapes
    def quantize(x):
        amax = x.abs().amax().float()
        gs = (448.0 * 6.0 / amax).reshape(1).to(torch.float32)
        q, sf = torch.ops.trtllm.fp4_quantize(x, gs, 16, False)
        return q, sf, gs

    W = {}
    for name, (n, k) in D.items():
        g = torch.Generator(device="cuda").manual_seed(5 + n)
        W[name] = quantize((torch.randn(n, k, generator=g, device="cuda") * 0.02).to(torch.bfloat16))

    def act(m, k, seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        return quantize(torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16))

    backends = "cutlass,cublaslt,cutedsl,cuda_core"
    bl = backends.split(",")
    runner_lt = tco.CublasLtFP4GemmRunner(0, torch.bfloat16)
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")

    def kernel_of(fn):
        from torch.profiler import ProfilerActivity, profile
        import tempfile
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        path = tempfile.mktemp(suffix=".json")
        prof.export_chrome_trace(path)
        tr = json.load(open(path))
        os.remove(path)
        ks = [(e["name"], e.get("args", {}).get("grid"), e.get("dur")) for e in tr.get("traceEvents", [])
              if e.get("cat") == "kernel"]
        return ks

    def cold_time(fn, warm=5, iters=20):
        s = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        e = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for _ in range(warm):
            flush.zero_()
            fn()
        for i in range(iters):
            flush.zero_()
            s[i].record()
            fn()
            e[i].record()
        torch.cuda.synchronize()
        ts = sorted(a.elapsed_time(b) * 1e3 for a, b in zip(s, e))
        return {"med_us": ts[len(ts) // 2], "p10_us": ts[int(0.1 * (len(ts) - 1))],
                "p90_us": ts[int(round(0.9 * (len(ts) - 1)))]}

    def lookups(name, m, n, k, xq, xsf):
        """The tuner entries nvfp4_gemm reads at this M, and the resolved cuBLASLt config."""
        res = {}
        shapes = (xq.size(), W[name][0].size(), xsf.size(), W[name][1].size(), torch.Size([1]))
        runner = tco.NVFP4GemmUnifiedRunner(0, torch.bfloat16, bl)
        hit, _, tac, t = tuner.profiling_cache.search_cache("trtllm::nvfp4_gemm::gemm", [runner], shapes,
                                                            tco.NVFP4GemmUnifiedRunner.tuning_config)
        res["gemm"] = {"hit": bool(hit), "tactic": str(tac)}
        if isinstance(tac, tuple) and tac[0] == "cublaslt":
            res["gemm"]["algo_cfg"] = list(runner_lt.cublaslt_runner.get_algo_config(
                xq, W[name][0], xsf, W[name][1], tac[1], -1))
        rng = tco._nvfp4_dense_range(n, k) if hasattr(tco, "_nvfp4_dense_range") else None
        if rng is not None and rng[0] <= m <= rng[1]:
            dr = tco.NVFP4GemmUnifiedRunner(0, torch.bfloat16, bl, dense_min_m=rng[0])
            hit, _, tac, t = tuner.profiling_cache.search_cache(
                "trtllm::nvfp4_gemm::gemm_dense", [dr], shapes, tco._nvfp4_dense_tuning_config(*rng))
            b = tco.nvfp4_dense_bucket(m, rng[0])
            res["dense"] = {"hit": bool(hit), "tactic": str(tac), "bucket": b}
            if isinstance(tac, tuple) and tac[0] == "cublaslt":
                res["dense"]["algo_cfg"] = list(runner_lt.cublaslt_runner.get_algo_config(
                    xq, W[name][0], xsf, W[name][1], tac[1], b))
        pin = getattr(tco, "nvfp4_pin_info", None)
        if pin is not None:
            res["pin"] = pin(n, k, m)
        return res

    def op_one(name, m):
        n, k = D[name]
        wq, wsf, wgs = W[name]
        xq, xsf, xgs = act(m, k, 11 + m)
        alpha = (1.0 / (xgs * wgs)).float()
        f = lambda: torch.ops.trtllm.nvfp4_gemm(xq, wq, xsf, wsf, alpha, torch.bfloat16,  # noqa: E731
                                                 allowed_backends=backends)
        f()
        ks = kernel_of(f)
        r = {"kernels": ks, "cold": cold_time(f), "lookup": lookups(name, m, n, k, xq, xsf)}
        if name == "gate_up" and m in CHUNK_M:
            dis = torch.ones(1, dtype=torch.float32, device="cuda")
            fc = lambda: _chunked(xq, xsf, wq, wsf, alpha, torch.bfloat16, bl, dis, 1280)  # noqa: E731
            res = fc()
            r["chunk_engaged"] = res is not None
            if res is not None:
                r["chunk_kernels"] = kernel_of(fc)
                r["chunk_cold"] = cold_time(fc)
        log(f"[op] {name}@{m} kernels={[(a[0][:90], a[1]) for a in ks]} cold={r['cold']['med_us']:.1f}us "
            f"lookup={r['lookup']} chunk={r.get('chunk_engaged')}")
        return r

    oplevel = {}
    for name in D:
        for m in SERVED_M:
            try:
                oplevel[f"{name}@{m}"] = op_one(name, m)
            except Exception as e:  # noqa: BLE001
                log(f"[op] {name}@{m} ERROR {e!r}")
                oplevel[f"{name}@{m}"] = {"error": repr(e)}
    out["oplevel"] = oplevel
    # Full cuBLASLt heuristic lists at the buckets of interest (index -> config).
    lists = {}
    for name, ms in (("gate_up", [(300, 320), (300, 256), (4000, 2048), (4000, 4000)]),
                     ("down", [(2000, 1024), (2000, 2000), (300, 256), (1000, 512), (4000, 2048)])):
        n, k = D[name]
        wq, wsf, _ = W[name]
        for m, hm in ms:
            try:
                xq, xsf, _ = act(m, k, 3)
                cnt = runner_lt.cublaslt_runner.get_num_heuristic_algos(xq, wq, xsf, wsf)
                hq, hsf, _ = act(hm, k, 3)
                cnt_h = runner_lt.cublaslt_runner.get_num_heuristic_algos(hq, wq, hsf, wsf)
                lst = [list(runner_lt.cublaslt_runner.get_algo_config(xq, wq, xsf, wsf, i,
                                                                       hm if hm != m else -1))
                       for i in range(cnt_h)]
                lists[f"{name}|M{m}|list_of_m{hm}"] = {"n_actual": cnt, "n_list": cnt_h, "cfgs": lst}
                log(f"[lt] {name}|M{m}|list_of_m{hm} n={cnt}/{cnt_h} cfgs={lst}")
            except Exception as e:  # noqa: BLE001
                log(f"[lt] {name} {m}/{hm} ERROR {e!r}")
    out["cublaslt_lists"] = lists
    del flush
    torch.cuda.empty_cache()

    # ---- served prompts
    sp = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
    rng = np.random.default_rng(int(time.time()) % 100000)

    def mk(isl):
        return [int(x) for x in rng.integers(1000, 200000, size=isl)]

    def run(prompts):
        old = sys.getswitchinterval()
        t = time.perf_counter()
        sys.setswitchinterval(0.5)
        try:
            futs = [llm.generate_async(p, sampling_params=sp) for p in prompts]
        finally:
            sys.setswitchinterval(old)
        [f.result() for f in futs]
        return time.perf_counter() - t

    try:
        from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import graphs as _g
        bcg = getattr(_g, "_BCG_STEP_COUNTS", None)
    except Exception:  # noqa: BLE001
        bcg = None
    disp.clear()
    chunk_calls["engaged"].clear()
    chunk_calls["declined"].clear()
    for _, isl, c in PROF_CASES:
        for _ in range(2):
            run([mk(isl) for _ in range(c)])
    ttft = {}
    for label, isl, c in TTFT_CASES:
        ts = sorted(run([mk(isl) for _ in range(c)]) * 1e3 for _ in range(ROUNDS))
        med = statistics.median(ts)
        ttft[label] = {"ms": med, "p10": ts[int(0.1 * (len(ts) - 1))], "p90": ts[int(round(0.9 * (len(ts) - 1)))],
                       "n": len(ts)}
        log(f"[ttft] {label} {ttft[label]}")
    out["ttft"] = ttft
    with open(out_path, "w") as f:
        json.dump(out, f)
    served = {}
    from torch.profiler import ProfilerActivity, profile
    import tempfile
    for label, isl, c in PROF_CASES:
      try:  # noqa: E111
        before = dict(bcg) if bcg is not None else None
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            run([mk(isl) for _ in range(c)])
            torch.cuda.synchronize()
        path = tempfile.mktemp(suffix=".json")
        prof.export_chrome_trace(path)
        tr = json.load(open(path))
        os.remove(path)
        agg = {}
        for e in tr.get("traceEvents", []):
            if e.get("cat") != "kernel":
                continue
            nm = e["name"]
            if not re.search(r"nvjet|cutlass|gemm|Gemm|sm120|xmma", nm):
                continue
            key = f"{nm[:160]}|grid{e.get('args', {}).get('grid')}"
            a = agg.setdefault(key, [0, 0.0])
            a[0] += 1
            a[1] += float(e.get("dur", 0))
        agg = {k: {"count": v[0], "avg_us": v[1] / v[0]} for k, v in agg.items() if v[0] >= 30}
        steps = {k: bcg[k] - before[k] for k in bcg} if bcg is not None else None
        served[label] = {"gemm_kernels": agg, "bcg_steps": steps}
        log(f"[served] {label} bcg={steps}")
        for k, v in sorted(agg.items(), key=lambda kv: -kv[1]["avg_us"] * kv[1]["count"]):
            log(f"    {v['count']:5d} x {v['avg_us']:8.1f}us  {k}")
      except Exception as e:  # noqa: BLE001,E111
        log(f"[served] {label} ERROR {e!r}")
    out["served_kernels"] = served
    out["dispatch"] = disp
    out["chunk_calls"] = chunk_calls
    out["gpu_end"] = gpu_state()
    with open(out_path, "w") as f:
        json.dump(out, f)
    log(f"[life] wrote {out_path}")
    try:
        llm.shutdown()
    except Exception:  # noqa: BLE001
        pass
    os._exit(0)


# ----------------------------------------------------------------------------- parent
CAND_RE = re.compile(r"\[Autotuner\] Candidate: custom_op=(trtllm::nvfp4_gemm::gemm(?:_dense)?), runner=.*?, "
                     r"tactic=(.*?), shapes=(.*?), time=([0-9.]+|inf)ms")
WARN_RE = re.compile(r"fallback tactic|not usable at m=|cache miss", re.I)


def parse_candidates(text, GU, DN):
    out = {}
    for m in CAND_RE.finditer(text):
        op, tac, shp, t = m.groups()
        ints = [int(x) for x in re.findall(r"\d+", shp)]
        if len(ints) < 4:
            continue
        mm, n, k = ints[0], ints[2], ints[1] * 2
        if (n, k) not in (GU, DN):
            continue
        key = f"{op.split('::')[-1]}|N{n}|M{mm}"
        out.setdefault(key, []).append((tac, float(t)))
    return out


ARM_ENV = {"main": {"TRTLLM_NVFP4_GATEUP_PIN": "0", "TRTLLM_NVFP4_DOWN_PIN": "0"}, "pin": {}}


def drive_ab(n_life):
    """Interleaved arms on one GPU: main (both pin kill switches 0), pin, main, pin, ..."""
    lives = {arm: [] for arm in ARM_ENV}
    for i in range(n_life):
        for arm in ARM_ENV:
            lives[arm].append(run_life(len(lives[arm]), arm, ARM_ENV[arm]))
    res = {arm: aggregate(lives[arm], arm) for arm in ARM_ENV}
    t = {arm: res[arm]["ttft"] for arm in res}
    better = {}
    for lab in t["pin"]:
        best_main = min(t["main"][lab])
        better[lab] = {"pin_median": statistics.median(t["pin"][lab]), "main_best": best_main,
                       "main_median": statistics.median(t["main"][lab]),
                       "pin_median_le_main_best": statistics.median(t["pin"][lab]) <= best_main * 1.0}
    ok = not res["pin"]["op_flip_keys"] and not res["pin"]["served_flip_cases"]
    log(f"[ab] ttft {json.dumps(better)}")
    print(json.dumps({"passed": bool(ok), "pin_kernels_identical_across_lifetimes": ok,
                      "pin_op_flip_keys": res["pin"]["op_flip_keys"],
                      "pin_served_flip_cases": res["pin"]["served_flip_cases"],
                      "main_op_flip_keys": res["main"]["op_flip_keys"],
                      "main_flipping_pick_keys": res["main"]["flipping_pick_keys"],
                      "ttft": t, "ttft_vs_main_best": better, "n_lifetimes_per_arm": n_life}))


def run_life(i, tag, env_extra=None):
    art = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
    os.makedirs(art, exist_ok=True)
    D = dims()
    GU, DN = D["gate_up"], D["down"]
    env = dict(os.environ)
    env.update(env_extra or {})
    if True:
        jp = os.path.join(art, f"{tag}_life{i}.json")
        lp = os.path.join(art, f"{tag}_life{i}.log")
        log(f"[drive] {tag} lifetime {i} env={env_extra} -> {lp}")
        with open(lp, "w") as lf:
            p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "life", jp],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
            for line in p.stdout:
                lf.write(line)
                if "[Autotuner]" in line:
                    continue
                if re.match(r"^\[\d\d:\d\d:\d\d\] ", line) or line.startswith(("[heartbeat]", "    ")):
                    sys.stdout.write(f"[L{i}] {line}")
                    sys.stdout.flush()
            p.wait()
        text = open(lp, errors="replace").read()
        cands = parse_candidates(text, GU, DN)
        warns = sorted(set(l.strip()[:300] for l in text.splitlines() if WARN_RE.search(l)
                           and ("nvfp4" in l or "CublasLtFP4" in l)))
        res = json.load(open(jp)) if os.path.exists(jp) else {"error": f"rc={p.returncode}"}
        res["candidates"] = cands
        res["warnings"] = warns
        with open(jp, "w") as f:
            json.dump(res, f)
        log(f"[drive] {tag} lifetime {i} rc={p.returncode} ncand_keys={len(cands)} warnings={warns[:6]}")
        return res


def drive(n_life, tag):
    lives = [run_life(i, tag) for i in range(n_life)]
    s = aggregate(lives, tag)
    print(json.dumps({"passed": True, "n_lifetimes": n_life, "flipping_pick_keys": s["flipping_pick_keys"],
                      "op_flip_keys": s["op_flip_keys"], "served_flip_cases": s["served_flip_cases"],
                      "ttft": s["ttft"], "summary": f"{tag}_summary.json"}))


def aggregate(lives, tag):
    art = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
    n_life = len(lives)
    summary = {"n": n_life, "picks": {}, "margins": {}, "op_kernels": {}, "served": {}, "ttft": {},
               "chunk": {}, "warnings": [l.get("warnings") for l in lives]}
    keys = sorted({k for l in lives for k in l.get("picks", {})})
    for k in keys:
        summary["picks"][k] = [(l.get("picks", {}).get(k) or {}).get("tactic") for l in lives]
        mg = []
        for l in lives:
            c = sorted(l.get("candidates", {}).get(k, []), key=lambda x: x[1])
            mg.append([(t, round(v * 1e3, 1)) for t, v in c[:3]])
        summary["margins"][k] = mg
        flip = len(set(summary["picks"][k])) > 1
        log(f"[agg] {'FLIP ' if flip else '     '}{k}: picks={summary['picks'][k]}")
        for i, x in enumerate(mg):
            log(f"          L{i} top3(us)={x}")
    for k in sorted({k for l in lives for k in l.get("oplevel", {})}):
        rows = []
        for l in lives:
            r = l.get("oplevel", {}).get(k, {})
            rows.append({"kernels": [(a[0][:100], a[1]) for a in r.get("kernels", [])],
                         "cold_us": round(r.get("cold", {}).get("med_us", -1), 1),
                         "chunk": r.get("chunk_engaged"),
                         "chunk_cold_us": round(r.get("chunk_cold", {}).get("med_us", -1), 1)
                         if r.get("chunk_cold") else None,
                         "lookup": r.get("lookup")})
        summary["op_kernels"][k] = rows
        names = [tuple(x[0] for x in rr["kernels"]) for rr in rows]
        if len(set(names)) > 1 or len({rr["chunk"] for rr in rows}) > 1:
            summary.setdefault("op_flip_keys", []).append(k)
        log(f"[agg] {tag} {'FLIP ' if len(set(names)) > 1 else '     '}op {k}: " +
            " | ".join(f"{rr['kernels'][0][0][:60] if rr['kernels'] else None} {rr['cold_us']}us chunk={rr['chunk']}"
                       f"/{rr['chunk_cold_us']}" for rr in rows))
    summary["served_flip_cases"] = []
    for lab in [c[0] for c in PROF_CASES]:
        summary["served"][lab] = [sorted(l.get("served_kernels", {}).get(lab, {}).get("gemm_kernels", {}).items(),
                                         key=lambda kv: -kv[1]["avg_us"] * kv[1]["count"])[:8] for l in lives]
        # NVFP4 GEMM kernels only (BF16 projection GEMM picks are outside this change).
        sets = [tuple(sorted((k, v["count"]) for k, v in
                             l.get("served_kernels", {}).get(lab, {}).get("gemm_kernels", {}).items()
                             if re.search(r"Fp4|ue4m3|e2m1", k)))
                for l in lives]
        # Single-request cases only: a multi-request group's step split can vary between
        # lifetimes (arrival timing), which changes M, not the pick.
        if len(set(sets)) > 1 and lab.startswith("isl"):
            summary["served_flip_cases"].append(lab)
        log(f"[agg] {tag} served {lab}: {'FLIP' if len(set(sets)) > 1 else 'identical'} across lifetimes")
    for lab in [c[0] for c in TTFT_CASES]:
        summary["ttft"][lab] = [round(l.get("ttft", {}).get(lab, {}).get("ms", -1), 2) for l in lives]
        log(f"[agg] {tag} ttft {lab}: {summary['ttft'][lab]}")
    summary["chunk"] = [l.get("chunk_calls") for l in lives]
    summary["dispatch"] = [l.get("dispatch") for l in lives]
    summary.setdefault("op_flip_keys", [])
    summary["flipping_pick_keys"] = [k for k in summary["picks"] if len(set(summary["picks"][k])) > 1]
    with open(os.path.join(art, f"{tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return summary


if __name__ == "__main__":
    if sys.argv[1] == "life":
        life(sys.argv[2])
    elif sys.argv[1] == "drive_ab":
        drive_ab(int(sys.argv[2]))
    else:
        drive(int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "life")
