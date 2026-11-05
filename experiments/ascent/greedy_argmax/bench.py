"""Idea 243 timing + repeated-execution check for fast_greedy_sample_kernel.

  bench.py time   -> times the sampler's eager trio (fast_greedy_sample_kernel) at B=1/2/4/8
                     after a ~6 ms dummy graph replay; prints step_ms JSON (both arms).
  bench.py check  -> untimed, same warmup/iteration/reset sequence; every iteration compares
                     tokens and the new_tokens buffer with the torch reference trio, checks the
                     fused kernel's slot/counter scratch is zero before each reset; plus edge cases.
"""
import json
import statistics
import sys

import torch

from tensorrt_llm._torch.pyexecutor.sampler import sampler_features as sf

V = 129280
DEV = "cuda"
WARMUP = 20
ITERS = 200
BATCHES = (1, 2, 4, 8)
STEPS, SLOTS = 4, 64


def ref_trio(logits, new_tokens, dest, mbw=1):
    nt = torch.argmax(logits, dim=-1).to(dtype=new_tokens.dtype)
    di = dest.unsqueeze(1).expand(-1, mbw)
    ne = nt.unsqueeze(1).expand(-1, mbw)
    new_tokens.view(-1, *new_tokens.shape[2:]).scatter_(0, di, ne)
    return nt


def make_dummy():
    # ~6 ms of elementwise work captured in a graph (no cuBLAS) to hide host launch cost
    a = torch.ones(16 * 1024 * 1024, device=DEV)
    a.mul_(1.0)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(68):
            a.mul_(1.0)
    g._buf = a
    return g


def setup(B, seed):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    logits = torch.randn(B, V, device=DEV, dtype=torch.float32, generator=gen)
    pristine = torch.randint(0, V, (STEPS, SLOTS, 1), device=DEV, dtype=torch.int32, generator=gen)
    new_tokens = pristine.clone()
    perm = torch.randperm(STEPS * SLOTS, device=DEV, generator=gen)[:B]
    dest = perm.to(torch.int64)
    return logits, pristine, new_tokens, dest


def run_time():
    if "--stage0" in sys.argv:
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import stage0
        stage0.main()
    dummy = make_dummy()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record(); dummy.replay(); e.record(); e.synchronize()
    print("dummy graph ms", s.elapsed_time(e), flush=True)
    out = {}
    fused = getattr(sf, "fused_greedy_argmax_enabled", lambda: False)()
    print("fused path available+enabled:", fused, flush=True)
    for B in BATCHES:
        logits, pristine, new_tokens, dest = setup(B, 1234 + B)
        ts = []
        for it in range(WARMUP + ITERS):
            new_tokens.copy_(pristine)  # reset point
            dummy.replay()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            sf.fast_greedy_sample_kernel(logits, new_tokens, dest, 1)
            e.record()
            e.synchronize()
            if it >= WARMUP:
                ts.append(s.elapsed_time(e))
        ts.sort()
        med = statistics.median(ts)
        nf = (ts[int(0.9 * len(ts))] - ts[int(0.1 * len(ts))]) / med
        out[str(B)] = {"step_ms": med, "noise_floor": nf}
        print(f"B={B} median_us={med*1000:.2f} p10={ts[int(0.1*len(ts))]*1000:.2f} "
              f"p90={ts[int(0.9*len(ts))]*1000:.2f} nf={nf:.3f}", flush=True)
    print(json.dumps({"batch": out, "output": {"class": "exact"}, "fused": fused}))


def case_pool(B, gen):
    """Varied logits rows [B, V]: random, bf16-cast, ties, inf, NaN, signed zeros."""
    pool = []
    pool.append(torch.randn(B, V, device=DEV, generator=gen))
    pool.append(torch.randn(B, V, device=DEV, generator=gen).bfloat16().float())
    # heavy ties: coarse values
    pool.append(torch.randint(-3, 4, (B, V), device=DEV, generator=gen).float())
    boundaries = [0, 1, V - 1, V - 2, 2047, 2048, 2049, 4095, 4096, 1023, 1024, 65535, 65536,
                  V - 2048, V // 2] + torch.randint(0, V, (6,), device=DEV, generator=gen).tolist()
    for k in (2, 3):
        x = torch.randn(B, V, device=DEV, generator=gen)
        for b in range(B):
            pos = torch.tensor(boundaries, device=DEV)[torch.randperm(len(boundaries), device=DEV,
                                                                       generator=gen)[:k]]
            x[b, pos] = 100.0
        pool.append(x)
    pool.append(torch.full((B, V), 0.5, device=DEV))  # all equal -> 0
    pool.append(torch.full((B, V), float("-inf"), device=DEV))  # all -inf -> 0
    x = torch.randn(B, V, device=DEV, generator=gen)
    x[:, 5000] = float("inf"); x[:, 7] = float("inf"); x[:, V - 1] = float("inf")
    pool.append(x)
    x = torch.randn(B, V, device=DEV, generator=gen)
    x[:, 9000] = float("nan"); x[:, 4096] = float("inf"); x[:, 100000] = float("nan")
    pool.append(x)
    x = torch.randn(B, V, device=DEV, generator=gen)
    x[:, V - 1] = float("nan")
    pool.append(x)
    x = torch.randn(B, V, device=DEV, generator=gen)
    x[:, 2048] = -float("nan")  # negative-sign NaN
    x[:, 3000] = float("nan")
    pool.append(x)
    x = torch.full((B, V), float("-inf"), device=DEV)
    x[:, 3000] = -0.0; x[:, 5000] = 0.0
    pool.append(x)
    x = torch.full((B, V), float("-inf"), device=DEV)
    x[:, 3000] = 0.0; x[:, 5000] = -0.0
    pool.append(x)
    x = torch.randn(B, V, device=DEV, generator=gen) - 10.0  # all negative
    pool.append(x)
    x = torch.zeros(B, V, device=DEV); x[:, ::2] = -0.0  # mixed signed zeros -> 0
    pool.append(x)
    x = torch.randn(B, V, device=DEV, generator=gen)
    x[:, 2049] = torch.finfo(torch.float32).max; x[:, 2050] = float("inf")
    pool.append(x)
    x = -torch.rand(B, V, device=DEV, generator=gen) * 1e-38  # denormals/tiny negatives
    pool.append(x)
    return pool


def scratch_zero():
    from tensorrt_llm._torch.pyexecutor.sampler.ops import greedy_argmax as ga
    for slot, count in ga._scratch.values():
        if int(slot.abs().sum()) != 0 or int(count.abs().sum()) != 0:
            return False
    return True


def run_check():
    from tensorrt_llm._torch.pyexecutor.sampler.ops import greedy_argmax as ga
    assert sf.fused_greedy_argmax_enabled()
    gen = torch.Generator(device=DEV).manual_seed(7)
    report = {}
    all_ok = True
    total_iters = 0
    # 1) repeated-execution sequence, same warmup/iters/reset points as timing
    for B in BATCHES:
        logits, pristine, new_tokens, dest = setup(B, 1234 + B)
        pool = case_pool(B, gen)
        mism = []
        scratch_bad = 0
        for it in range(WARMUP + ITERS):
            if not scratch_zero():
                scratch_bad += 1
            new_tokens.copy_(pristine)  # reset point
            src = pool[it % len(pool)] if it % 3 else torch.randn(B, V, device=DEV, generator=gen)
            logits.copy_(src)
            ref_buf = pristine.clone()
            ref = ref_trio(logits, ref_buf, dest)
            got = sf.fast_greedy_sample_kernel(logits, new_tokens, dest, 1)
            ok = (got.dtype == ref.dtype and got.shape == ref.shape and torch.equal(got, ref)
                  and torch.equal(new_tokens, ref_buf))
            if not ok:
                mism.append(it)
            total_iters += 1
        if not scratch_zero():
            scratch_bad += 1
        report[f"seq_B{B}"] = {"mismatch_iters": mism[:20], "n_mismatch": len(mism),
                               "scratch_not_zero": scratch_bad}
        all_ok &= (not mism) and scratch_bad == 0
        print(f"seq B={B}: mismatches={len(mism)} scratch_bad={scratch_bad}", flush=True)
    # 2) 10k random rows at B=1..8, fp32 and bf16 inputs, plus every edge pool at B=1..8
    rows = 0
    bad = 0
    for dtype in (torch.float32, torch.bfloat16):
        r = 0
        while r < 10000:
            for B in range(1, 9):
                x = torch.randn(B, V, device=DEV, generator=gen).to(dtype)
                nt = torch.randint(0, V, (STEPS, SLOTS, 1), device=DEV, dtype=torch.int32,
                                   generator=gen)
                dest = torch.randperm(STEPS * SLOTS, device=DEV, generator=gen)[:B]
                rb = nt.clone()
                ref = ref_trio(x, rb, dest)
                got = sf.fast_greedy_sample_kernel(x, nt, dest, 1)
                if not (torch.equal(got, ref) and torch.equal(nt, rb)):
                    bad += 1
                r += B
        rows += r
    report["random_rows"] = {"rows": rows, "bad_batches": bad}
    all_ok &= bad == 0
    print(f"random rows={rows} bad={bad}", flush=True)
    edge_bad = []
    for B in range(1, 9):
        for ci, x in enumerate(case_pool(B, gen)):
            for dtype in (torch.float32, torch.bfloat16):
                xx = x.to(dtype)
                nt = torch.randint(0, V, (STEPS, SLOTS, 1), device=DEV, dtype=torch.int32,
                                   generator=gen)
                dest = torch.randperm(STEPS * SLOTS, device=DEV, generator=gen)[:B]
                rb = nt.clone()
                ref = ref_trio(xx, rb, dest)
                got = sf.fast_greedy_sample_kernel(xx, nt, dest, 1)
                if not (torch.equal(got, ref) and torch.equal(nt, rb)):
                    edge_bad.append((B, ci, str(dtype), got.tolist()[:4], ref.tolist()[:4]))
    report["edge_cases_bad"] = edge_bad[:20]
    all_ok &= not edge_bad
    print(f"edge cases bad={len(edge_bad)}", flush=True)
    # 3) strided rows (row stride > V), large B, int64 buffer, beam>1 fallback
    extra = []
    big = torch.randn(8, V + 64, device=DEV, generator=gen)
    x = big[:, :V]
    nt = torch.zeros(STEPS, SLOTS, 1, dtype=torch.int32, device=DEV)
    dest = torch.arange(8, device=DEV) * 5
    rb = nt.clone(); ref = ref_trio(x, rb, dest); got = sf.fast_greedy_sample_kernel(x, nt, dest, 1)
    extra.append(("strided", torch.equal(got, ref) and torch.equal(nt, rb)))
    for B in (16, 64, 256):
        x = torch.randn(B, V, device=DEV, generator=gen)
        nt = torch.randint(0, V, (STEPS, SLOTS, 1), dtype=torch.int32, device=DEV, generator=gen)
        dest = torch.randperm(STEPS * SLOTS, device=DEV, generator=gen)[:B]
        rb = nt.clone(); ref = ref_trio(x, rb, dest); got = sf.fast_greedy_sample_kernel(x, nt, dest, 1)
        extra.append((f"B{B}", torch.equal(got, ref) and torch.equal(nt, rb)))
    x = torch.randn(4, V, device=DEV, generator=gen)
    nt = torch.zeros(STEPS, SLOTS, 1, dtype=torch.int64, device=DEV)
    dest = torch.tensor([3, 9, 1, 200], device=DEV)
    rb = nt.clone(); ref = ref_trio(x, rb, dest); got = sf.fast_greedy_sample_kernel(x, nt, dest, 1)
    extra.append(("int64_buf", torch.equal(got, ref) and torch.equal(nt, rb)))
    nt = torch.zeros(STEPS, SLOTS, 2, dtype=torch.int32, device=DEV)
    rb = nt.clone(); ref = ref_trio(x, rb, dest, 2); got = sf.fast_greedy_sample_kernel(x, nt, dest, 2)
    extra.append(("beam2_fallback", torch.equal(got, ref) and torch.equal(nt, rb)))
    # 4) CUDA-graph capture + replay with changing logits
    xg = torch.randn(4, V, device=DEV, generator=gen)
    ntg = torch.zeros(STEPS, SLOTS, 1, dtype=torch.int32, device=DEV)
    destg = torch.tensor([3, 9, 1, 200], device=DEV)
    sf.fast_greedy_sample_kernel(xg, ntg, destg, 1)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outg = sf.fast_greedy_sample_kernel(xg, ntg, destg, 1)
    gok = True
    for i in range(50):
        xg.copy_(case_pool(4, gen)[i % 5] if i % 2 else torch.randn(4, V, device=DEV, generator=gen))
        ntg.zero_()
        g.replay()
        rb = torch.zeros_like(ntg); ref = ref_trio(xg, rb, destg)
        gok &= torch.equal(outg, ref) and torch.equal(ntg, rb)
    extra.append(("graph_replay_50", gok))
    extra.append(("scratch_zero_end", scratch_zero()))
    report["extra"] = extra
    all_ok &= all(ok for _, ok in extra)
    print("extra", extra, flush=True)
    report["seq_iters_checked"] = total_iters
    report["compared_against"] = "torch.argmax(logits,-1).to(new_tokens.dtype) + scatter_ (main's trio)"
    print(json.dumps({"passed": bool(all_ok), **report}))


if __name__ == "__main__":
    {"time": run_time, "check": run_check}[sys.argv[1]]()
