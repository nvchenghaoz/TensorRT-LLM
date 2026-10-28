"""Idea 9 encoder-only check + interleaved microbenchmark (one process, no denoiser).

1. tokenize-once == main's two tokenizer calls (ids, mask, dtype) on many prompts.
2. warm buckets k*80 (k=1..TE_K, default 2) compile-only (TRTLLM_FLUX3_TEXT_GRAPH=0),
   then capture: startup seconds and GPU memory on vs off.
3. ctx bitwise (padded positions included) graph replay vs main's compiled graphs-off
   path, every warmed bucket, A/B/A order, 3 repeats each; no Dynamo compile.
4. after warmup: a bucket outside the warmed set compiles on demand (as main) and
   captures no graph; its ctx is bitwise main's.
5. interleaved timing (>=50 iters) at T=80 and T=160: graph vs off, plus off-vs-off A/A.
Usage: python te_check.py <checkpoint>
"""

import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from te_common import PROMPTS, load_embedder, main_encode, main_tokenize, prompt_for_bucket  # noqa

CKPT = sys.argv[1]
ITERS = int(os.environ.get("TE_ITERS", "60"))
fails = []


def note(ok, what):
    if not ok:
        fails.append(what)
        print("FAIL", what, flush=True)


def counters():
    from torch._dynamo.utils import counters as c

    return dict(c["frames"]), dict(c["stats"])


emb = load_embedder(CKPT)
print("loaded", flush=True)

# 1. tokenize-once equality
bucket_prompts = {k: prompt_for_bucket(emb, k) for k in range(1, 8)}
tok_ok = 0
for p in PROMPTS + list(bucket_prompts.values()):
    a_ids, a_mask = emb._tokenize(p)
    b_ids, b_mask = main_tokenize(emb, p)
    ok = (
        torch.equal(a_ids, b_ids)
        and torch.equal(a_mask, b_mask)
        and a_ids.dtype == b_ids.dtype
        and a_mask.dtype == b_mask.dtype
    )
    note(ok, f"tokenize {p[:30]!r}")
    tok_ok += ok
print(f"tokenize-once equal: {tok_ok}/{len(PROMPTS) + len(bucket_prompts)}", flush=True)

# 2. warmup: compile-only, then capture
torch.cuda.synchronize()
mem0 = torch.cuda.memory_reserved()
os.environ["TRTLLM_FLUX3_TEXT_GRAPH"] = "0"
K = int(os.environ.get("TE_K", "2"))
emb.params.warmup_buckets = K
t0 = time.time()
emb.warmup_buckets()
t_compile = time.time() - t0
torch.cuda.synchronize()
mem_off = torch.cuda.memory_reserved()
os.environ.pop("TRTLLM_FLUX3_TEXT_GRAPH")
lengths = sorted(emb._warm_lengths)
print(f"warm lengths {lengths} compile-only {t_compile:.1f}s", flush=True)
t0 = time.time()
for T in lengths:
    ids, mask = emb._tokenize("warmup")
    n = int(mask.sum())
    sid = torch.full((1, T), emb.processor.tokenizer.pad_token_id)
    sid[:, :n] = ids[:, :n]
    sm = torch.zeros((1, T), dtype=torch.long)
    sm[:, :n] = 1
    with torch.inference_mode():
        emb._capture(sid.cuda(), sm.cuda())
t_capture = time.time() - t0
torch.cuda.synchronize()
mem_on = torch.cuda.memory_reserved()
startup = {
    "compile_s_total": round(t_compile, 1),
    "capture_s_total": round(t_capture, 2),
    "mem_reserved_before_mb": mem0 / 2**20,
    "mem_reserved_graphs_off_mb": mem_off / 2**20,
    "mem_reserved_graphs_on_mb": mem_on / 2**20,
    "graph_pool_mb": (mem_on - mem_off) / 2**20,
}
print("startup", startup, flush=True)
note(sorted(emb._graphs) == lengths, "graphs for every warm bucket")

# 3. bitwise ctx, A/B/A order
order = [k for k in [1, 2, 1, 3, 4, 2, 1] if k <= K]
eq = {}
frames0, stats0 = counters()
for k in order:
    p = bucket_prompts[k]
    ref = main_encode(emb, p)
    outs = [emb.encode([p])[0] for _ in range(3)]
    ok = all(torch.equal(o, ref) for o in outs) and all(o.shape == ref.shape for o in outs)
    note(ok, f"ctx bucket {k}")
    eq.setdefault(k * 80, []).append(ok)
for p in PROMPTS:
    if emb._tokenize(p)[0].shape[1] > K * 80:
        continue
    ref = main_encode(emb, p)
    out = emb.encode([p])[0]
    note(torch.equal(out, ref), f"ctx prompt {p[:30]!r}")
print("ctx bitwise per bucket:", eq, flush=True)
frames1, stats1 = counters()
note(frames1 == frames0 and stats1 == stats0, "no compile while checking warm buckets")

# 4. a bucket outside the warmed set: compiles on demand (as main), no graph capture
ngraphs = len(emb._graphs)
unseen = {}
kx = K + 1
for k in (kx, kx, 1, kx):
    p = bucket_prompts[k]
    torch.cuda.synchronize()
    t0 = time.time()
    out = emb.encode([p])[0]
    torch.cuda.synchronize()
    unseen.setdefault(out.shape[1], []).append(round((time.time() - t0) * 1e3, 1))
frames2, stats2 = counters()
print("after-warmup call ms per T:", unseen, flush=True)
print("dynamo frames before/after:", frames1, frames2, stats1, stats2, flush=True)
note(stats2.get("unique_graphs", 0) == stats1.get("unique_graphs", 0) + 1,
     "exactly one on-demand compile for the unwarmed bucket")
note(len(emb._graphs) == ngraphs, "no graph capture after warmup")
outx = emb.encode([bucket_prompts[kx]])[0]
refx = main_encode(emb, bucket_prompts[kx])
note(torch.equal(outx, refx), "unwarmed bucket ctx bitwise main")
err = (outx.float() - refx.float()).abs().max().item()


# 5. interleaved timing
def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    return (t2 - t0) * 1e3, (t1 - t0) * 1e3


def stats(xs):
    xs = sorted(xs)
    q = lambda f: xs[min(len(xs) - 1, int(f * len(xs)))]  # noqa
    med = statistics.median(xs)
    return {"median": round(med, 3), "p10": round(q(0.1), 3), "p90": round(q(0.9), 3),
            "noise": round((q(0.9) - q(0.1)) / med, 4)}


def gpu_busy(fn, n=5):
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
    ev = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    return sum(e.time_range.elapsed_us() for e in ev) / n / 1e3, len(ev) / n


timing = {}
for k in (1, 2):
    p = bucket_prompts[k]
    on = lambda: emb.encode([p])  # noqa
    off = lambda: main_encode(emb, p)  # noqa
    for _ in range(5):
        on(), off()
    res = {"on": [], "off": [], "off2": [], "on_ret": [], "off_ret": []}
    eq_iters = 0
    for i in range(ITERS):
        seq = [("on", on), ("off", off), ("off2", off)] if i % 2 == 0 else [("off2", off), ("off", off), ("on", on)]
        for name, fn in seq:
            w, r = timed(fn)
            res[name].append(w)
            if name in ("on", "off"):
                res[name + "_ret"].append(r)
        eq_iters += torch.equal(emb.encode([p])[0], main_encode(emb, p))
    busy_on, kn_on = gpu_busy(on)
    busy_off, kn_off = gpu_busy(off)
    timing[f"T{k * 80}"] = {
        name: stats(v) for name, v in res.items()
    } | {"ctx_equal_iters": f"{eq_iters}/{ITERS}", "gpu_busy_ms_on": round(busy_on, 3),
         "gpu_busy_ms_off": round(busy_off, 3), "gpu_events_on": kn_on, "gpu_events_off": kn_off}
    note(eq_iters == ITERS, f"timing-loop equality T={k * 80}")
    print(f"T={k * 80}", json.dumps(timing[f"T{k * 80}"]), flush=True)

print(json.dumps({
    "passed": not fails,
    "failures": fails,
    "compared_against": "main's compiled graphs-off encode (two tokenizer calls, compiled _forward_tensors, clone) in the same process",
    "startup": startup,
    "after_warmup_ms": unseen,
    "unwarmed_bucket_max_abs_vs_main": err,
    "timing": timing,
}))
