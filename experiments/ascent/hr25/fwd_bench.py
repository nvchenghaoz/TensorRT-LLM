"""Branch 25 (idea 28) in-process eager-forward harness.

One process (TLLM_WORKER_USE_SINGLE_PROCESS=1), one LLM. Gemma4ForCausalLM.forward
is wrapped at class level: when recording, it syncs the GPU, then times
  enqueue_ms = host time from entry until forward returns (all launches queued),
  wall_ms    = entry until the GPU drained after it (torch.cuda.synchronize).
The sync before entry means each forward starts on an idle GPU.

usage: fwd_bench.py <checkpoint> <mode> [opts]
  mode time : warmup + timed single-request forwards at 300/1000/2000/8000 tokens,
              prints run_gpu_pair JSON (step_ms = wall median) as the last line.
  mode prof : time mode at 300 only, then cProfile over N 300-token forwards
              (category table + top functions) and one torch.profiler (kineto)
              forward listing every CUDA kernel (count, us).
  mode check: A,B,A (toggle mg._HOST_TRIMS if present) over 300, 1000, 4x2000
              repeated sequences; logits compared bitwise.
The __main__ guard matters (LLM API workers re-import this module).
"""
import json
import os
import random
import statistics
import sys
import threading
import time


def _heartbeat():
    t0 = time.time()
    while True:
        time.sleep(60)
        print(f"[heartbeat] {time.time() - t0:.0f}s", flush=True)


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main():
    os.environ["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"
    os.environ.setdefault("TLLM_LOG_LEVEL", "WARNING")
    threading.Thread(target=_heartbeat, daemon=True).start()
    ck = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "time"
    art = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")

    import torch

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.models import modeling_gemma4 as mg
    from tensorrt_llm.llmapi import KvCacheConfig

    cfg = json.load(open(os.path.join(ck, "config.json")))
    tcfg = cfg.get("text_config", cfg)
    print("config: hidden", tcfg.get("hidden_size"), "layers", tcfg.get("num_hidden_layers"),
          "vocab", tcfg.get("vocab_size"), "sliding_window", tcfg.get("sliding_window"), flush=True)
    vocab = int(tcfg.get("vocab_size", 262144))

    rec = {"on": False, "rows": [], "capture": None, "cprof": None, "kineto": None}
    orig = mg.Gemma4ForCausalLM.forward

    def wrapped(self, *a, **k):
        if not rec["on"]:
            return orig(self, *a, **k)
        md = k.get("attn_metadata", a[0] if a else None)
        n = int(md.num_tokens)
        torch.cuda.synchronize()
        cp = rec["cprof"]
        kin = rec["kineto"]
        if kin is not None:
            kin.__enter__()
        if cp is not None:
            cp.enable()
        t0 = time.perf_counter()
        out = orig(self, *a, **k)
        t1 = time.perf_counter()
        if cp is not None:
            cp.disable()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        if kin is not None:
            kin.__exit__(None, None, None)
            rec["kineto_done"] = kin
            rec["kineto"] = None
        rec["rows"].append((n, int(md.num_contexts), (t1 - t0) * 1e3, (t2 - t0) * 1e3))
        if rec["capture"] is not None:
            rec["capture"].append(out.detach().float().cpu().clone())
        return out

    mg.Gemma4ForCausalLM.forward = wrapped

    llm = LLM(
        model=ck,
        max_num_tokens=8192,
        max_seq_len=8300,
        max_batch_size=64,
        batch_wait_timeout_ms=(1500 if mode == "check" else 0),
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.80, enable_block_reuse=False),
    )
    print("loaded", flush=True)
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    rng = random.Random(25)

    def ids(n, seed=None):
        r = rng if seed is None else random.Random(seed)
        return {"prompt_token_ids": [2] + [r.randrange(1000, min(vocab, 200000)) for _ in range(n - 1)]}

    def run(n, reps, conc=1, seed_base=None):
        rec["rows"].clear()
        for i in range(reps):
            group = [ids(n, None if seed_base is None else seed_base + i * 100 + j) for j in range(conc)]
            llm.generate(group, sp, use_tqdm=False)
        return [r for r in rec["rows"] if r[1] > 0]

    def stats(rows):
        enq = [r[2] for r in rows]
        wall = [r[3] for r in rows]
        return {"n": len(rows), "tokens": sorted(set(r[0] for r in rows)),
                "enqueue_med": statistics.median(enq), "enqueue_p10": pct(enq, .1), "enqueue_p90": pct(enq, .9),
                "wall_med": statistics.median(wall), "wall_p10": pct(wall, .1), "wall_p90": pct(wall, .9)}

    # Untimed warmup at each size (JIT, autotuner caches).
    for n in (300, 1000, 2000, 8000):
        llm.generate([ids(n) for _ in range(2)], sp, use_tqdm=False)
        llm.generate([ids(n)], sp, use_tqdm=False)
    print("warm", flush=True)

    result = {}
    if mode in ("time", "prof"):
        sizes = (300, 1000, 2000, 8000) if mode == "time" else (300,)
        reps = int(os.environ.get("HR25_REPS", "30"))
        rec["on"] = True
        table = {}
        for rnd in range(2):  # two interleaved rounds; pooled per size
            for n in sizes:
                run(n, 5)  # warmup
                rows = run(n, reps // 2)
                table.setdefault(n, []).extend(rows)
                print(f"round {rnd} n={n}: {json.dumps(stats(rows))}", flush=True)
        batch = {}
        for n, rows in table.items():
            s = stats(rows)
            print(f"FINAL n={n}: {json.dumps(s)}", flush=True)
            batch[str(n)] = {"step_ms": s["wall_med"],
                             "noise_floor": (s["wall_p90"] - s["wall_p10"]) / s["wall_med"],
                             "wall_p10": s["wall_p10"], "wall_p90": s["wall_p90"],
                             "enqueue_ms": s["enqueue_med"], "enqueue_p10": s["enqueue_p10"],
                             "enqueue_p90": s["enqueue_p90"], "count": s["n"]}
        result = {"batch": batch, "output": {"class": "exact"}}

    if mode == "prof":
        import cProfile
        import io
        import pstats

        # Host us per Triton JITFunction.run call (binder, specialization key,
        # cache lookup, launcher), per kernel, on unprofiled forwards.
        from triton.runtime import jit as tjit
        _run = tjit.JITFunction.run
        jt = {}

        def timed_run(self, *a, **k):
            t = time.perf_counter()
            r = _run(self, *a, **k)
            d = time.perf_counter() - t
            name = getattr(getattr(self, "fn", None), "__name__", str(self))
            e = jt.setdefault(name, [0, 0.0])
            e[0] += 1
            e[1] += d
            return r

        tjit.JITFunction.run = timed_run
        rows = run(300, 10)
        tjit.JITFunction.run = _run
        nf = len(rows)
        print(f"=== Triton JITFunction.run host time over {nf} unprofiled 300-token forwards ===")
        tot = 0.0
        for name, (c, s) in sorted(jt.items(), key=lambda kv: -kv[1][1]):
            print(f"{c / nf:6.1f} calls/fwd {s * 1e6 / c:8.2f} us/call {s * 1e3 / nf:8.3f} ms/fwd  {name}")
            tot += s * 1e3 / nf
        print(f"Triton JIT run total: {tot:.3f} ms/fwd; enqueue med of these forwards "
              f"{statistics.median([r[2] for r in rows]):.2f} ms")
        result["triton_jit_ms_per_fwd"] = tot

        cp = cProfile.Profile()
        rec["cprof"] = cp
        rows = run(300, 10)
        rec["cprof"] = None
        nf = len(rows)
        print(f"cProfile over {nf} forwards: enqueue med {statistics.median([r[2] for r in rows]):.2f} ms", flush=True)
        cp.dump_stats(os.path.join(art, "fwd300.pstats"))
        st = pstats.Stats(cp)
        # Inclusive per-forward ms of chosen functions (by file:func substring).
        keys = {}
        for (fn, line, func), (cc, nc, tt, ct, callers) in st.stats.items():
            key = f"{os.path.basename(fn)}:{line}:{func}"
            keys[key] = (nc / nf, tt * 1e3 / nf, ct * 1e3 / nf)
        top_tt = sorted(keys.items(), key=lambda kv: -kv[1][1])[:90]
        top_ct = sorted(keys.items(), key=lambda kv: -kv[1][2])[:140]
        print("=== top by tottime (calls/fwd, tottime ms/fwd, cumtime ms/fwd) ===")
        for k, v in top_tt:
            print(f"{v[0]:8.1f} {v[1]:8.3f} {v[2]:8.3f}  {k}")
        print("=== top by cumtime ===")
        for k, v in top_ct:
            print(f"{v[0]:8.1f} {v[1]:8.3f} {v[2]:8.3f}  {k}")
        sys.stdout.flush()
        s = io.StringIO()
        pstats.Stats(cp, stream=s).sort_stats("cumulative").print_callers(60)
        open(os.path.join(art, "callers.txt"), "w").write(s.getvalue())

        # One kineto forward for the kernel list.
        from torch.profiler import ProfilerActivity, profile
        rec["kineto"] = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
        run(300, 1)
        kin = rec.get("kineto_done")
        if kin is not None:
            kern = {}
            nk = 0
            for e in kin.events():
                if e.device_type.name != "CUDA":
                    continue
                d = kern.setdefault(e.name[:110], [0, 0.0])
                d[0] += 1
                d[1] += getattr(e, "device_time", None) or getattr(e, "cuda_time", 0.0)
                nk += 1
            busy = sum(us for _, us in kern.values()) / 1e3
            print(f"=== kineto: {nk} GPU ops in one 300-token forward, sum of op durations "
                  f"{busy:.2f} ms ===")
            result["gpu_ops_sum_ms"] = busy
            for name, (c, us) in sorted(kern.items(), key=lambda kv: -kv[1][0]):
                print(f"{c:5d} {us / c:9.1f} us/each {us:10.1f} us  {name}")
            kin.export_chrome_trace(os.path.join(art, "fwd300_kineto.json"))
            result["kernels_per_forward"] = nk

    if mode == "check":
        from tensorrt_llm._torch.attention.backends import flashinfer as fib
        from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2 as kvm
        switches = [(m, a) for m, a in ((mg, "_HOST_TRIMS"), (kvm, "_KV_VIEW_CACHE"),
                                        (fib, "_PLAN_PARAMS_MEMO_ENABLED")) if hasattr(m, a)]
        flag = len(switches) > 0
        print("switches:", [a for _, a in switches], flush=True)
        seq = [(300, 1), (1000, 1), (2000, 4), (300, 1), (2000, 4), (1000, 1)]
        passes = {}
        for tag, on in (("A", False), ("B", True), ("A2", False)):
            for m, a in switches:
                setattr(m, a, on)
            rec["on"] = True
            rec["capture"] = []
            for si, (n, conc) in enumerate(seq):
                for rep in range(3):
                    run(n, 1, conc, seed_base=1000 * si + rep * 7 + 1)
            passes[tag] = rec["capture"]
            rec["capture"] = None
            print(f"[{tag}] trims={on if flag else 'n/a'} captured {len(passes[tag])} forwards "
                  f"shapes {[tuple(x.shape) for x in passes[tag][:6]]}", flush=True)

        def cmp(a, b):
            neq = 0
            first = None
            for i, (x, y) in enumerate(zip(a, b)):
                same = x.shape == y.shape and torch.equal(x, y)
                if not same:
                    neq += 1
                    if first is None:
                        first = (i, tuple(x.shape), float((x - y).abs().max()) if x.shape == y.shape else None)
            return {"forwards": len(a), "len_b": len(b), "not_bitwise": neq, "first_mismatch": first}

        aa = cmp(passes["A"], passes["A2"])
        ab = cmp(passes["A"], passes["B"])
        # Repeated calls within a pass: each prompt seed is used once per pass, so
        # compare passes; also check finiteness.
        finite = all(bool(torch.isfinite(x).all()) for p in passes.values() for x in p)
        ok = (aa["not_bitwise"] == 0 and ab["not_bitwise"] == 0 and finite and aa["len_b"] == aa["forwards"]
              and ab["len_b"] == ab["forwards"] and flag)
        result = {"passed": bool(ok), "compared": "in-process eager Gemma4ForCausalLM.forward logits, trims "
                  "off/on/off (A,B,A) over 300, 1000, 4x2000 sequence x3 reps, torch.equal bitwise",
                  "toggle_present": flag, "A_vs_A2": aa, "A_vs_B": ab, "finite": finite}

    llm.shutdown()
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
