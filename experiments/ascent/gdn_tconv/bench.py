# GDN prefill transpose + causal conv1d microbenchmark (Qwen3.8-27B, TP1).
# BRACE-FREE on purpose: run_gpu_pair passes this file's text inline as
#   python -c <text> BASE_COMMIT=... / CAND_COMMIT=...
# because the baseline tree has no experiments/ directory.
#   main arm:      extract_transpose_prefill_slice + in-place C++ causal_conv1d_fwd
#   candidate arm: gdn_prefill_conv.gdn_prefill_transpose_conv (one Triton kernel),
#                  plus an in-process kill-switch A/B of both chains.
# Shapes: x = in_proj_qkvz[:T, :10240] (row stride 16384, bf16), conv width 4,
# conv-state pool [16, 10240, 3] bf16, every sequence has an initial state
# (the benchmark's 32768-token cached prefix). n x 2048 tokens for n = 1, 2, 3, 4.
# Every iteration: restore the state pool from a pristine copy, optionally evict
# L2 (256 MiB write), torch.cuda._sleep so the start event is queued behind GPU
# work, then events around the op. 10 warmup + 60 timed per (path, L2 mode, round),
# two rounds in alternating order. batch[n].kernel_ms = the arm's production
# dispatch (gdn_fused_transpose_conv_enabled) at cold L2.
import json
import subprocess
import sys

print(sys.argv[1] if len(sys.argv) > 1 else "TREE_COMMIT=unlabelled", flush=True)
try:
    _h = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    _d = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout
    print("git_head=%s dirty=%d" % (_h.strip(), len(_d.splitlines())), flush=True)
except Exception as _e:  # noqa: BLE001
    print("git unavailable: %r" % (_e,), flush=True)

import torch  # noqa: E402

from tensorrt_llm._torch.modules.mamba.causal_conv1d import causal_conv1d_fn  # noqa: E402
from tensorrt_llm._torch.modules.mamba.fuse_elementwise_ops import (  # noqa: E402
    extract_transpose_prefill_slice,
)

try:
    from tensorrt_llm._torch.modules.mamba import gdn_prefill_conv as F
except ImportError:
    F = None

DIM, QKVZ, SLOTS, L = 10240, 16384, 16, 2048
WARMUP, ITERS = 10, 60
dev = torch.device("cuda")
flush = torch.empty(64 * 1024 * 1024, dtype=torch.float32, device=dev)


def log(*a):
    print(*a, flush=True)


def make(n, gen, seqlen=L):
    T = n * seqlen
    qkvz = (torch.randn(T, QKVZ, device=dev, generator=gen) * 1.5).to(torch.bfloat16)
    w = (torch.randn(DIM, 4, device=dev, generator=gen) * 0.5).to(torch.bfloat16)
    pool0 = torch.randn(SLOTS, DIM, 3, device=dev, generator=gen).to(torch.bfloat16)
    return dict(
        x=qkvz[:, :DIM],
        w=w,
        pool0=pool0,
        pool=pool0.clone(),
        qsl=torch.arange(0, T + 1, seqlen, dtype=torch.int32, device=dev),
        idx=torch.randperm(SLOTS, device=dev, generator=gen)[:n].to(torch.int32),
        hi=torch.ones(n, dtype=torch.bool, device=dev),
    )


def run_main(c):
    xt = extract_transpose_prefill_slice(c["x"], c["x"].shape[0], 0, c["x"].shape[1])
    return causal_conv1d_fn(
        xt,
        c["w"],
        None,
        activation="silu",
        conv_states=c["pool"],
        has_initial_state=c["hi"],
        cache_indices=c["idx"],
        query_start_loc=c["qsl"],
    )


def run_fused(c):
    return F.gdn_prefill_transpose_conv(
        c["x"], c["w"], None, c["pool"], c["qsl"], c["idx"], c["hi"], activation="silu"
    )


def timeit(fn, c, cold):
    ev = []
    for i in range(WARMUP + ITERS):
        c["pool"].copy_(c["pool0"])
        if cold:
            flush.fill_(float(i))
        torch.cuda._sleep(40000)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn(c)
        e.record()
        if i >= WARMUP:
            ev.append((s, e))
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in ev]


def stats(ts):
    ts = sorted(ts)
    n = len(ts)
    med = ts[n // 2] if n % 2 else 0.5 * (ts[n // 2 - 1] + ts[n // 2])
    p10, p90 = ts[int(0.1 * (n - 1))], ts[int(0.9 * (n - 1))]
    return dict(median_ms=med, p10_ms=p10, p90_ms=p90, noise=(p90 - p10) / med)


def main():
    gen = torch.Generator(device=dev)
    gen.manual_seed(7)
    paths = dict(main=run_main)
    if F is not None:
        paths["fused"] = run_fused
    out = dict(batch=dict(), output=dict([("class", "exact")]), extra=dict())
    log("fused module present: %s" % (F is not None,))
    for n in [1, 2, 3, 4]:
        # the arm's production dispatch (gdn_mixer._prefill_transpose_conv)
        active = "main"
        if F is not None and F.gdn_fused_transpose_conv_enabled(n * L):
            active = "fused"
        log("n=%d: active path %s" % (n, active))
        out["extra"]["active_%d" % n] = active
        c = make(n, gen)
        for fn in paths.values():  # JIT / autotune outside timing
            for _ in range(3):
                fn(c)
        torch.cuda.synchronize()
        res = dict()
        for cold in (False, True):
            mode = "cold" if cold else "warm"
            # two rounds with alternating order, pooled per path
            acc = dict((p, []) for p in paths)
            order = list(paths)
            for rnd in range(2):
                for p in order if rnd == 0 else order[::-1]:
                    acc[p] += timeit(paths[p], c, cold)
            for p in paths:
                st = stats(acc[p])
                res[p + "_" + mode] = st
                log(
                    "n=%d T=%d %-5s %s: median %8.2f us  p10 %8.2f  p90 %8.2f  noise %.4f"
                    % (
                        n,
                        n * L,
                        p,
                        mode,
                        st["median_ms"] * 1e3,
                        st["p10_ms"] * 1e3,
                        st["p90_ms"] * 1e3,
                        st["noise"],
                    )
                )
            if "fused" in paths:
                sv = (res["main_" + mode]["median_ms"] - res["fused_" + mode]["median_ms"]) * 1e3
                res["saved_us_" + mode] = sv
                log("n=%d %s: in-process saved %.2f us/layer" % (n, mode, sv))
        prim = res[active + "_cold"]
        out["batch"][str(n)] = dict(kernel_ms=prim["median_ms"], noise_floor=prim["noise"])
        ex = dict()
        for k, v in res.items():
            if isinstance(v, dict):
                ex[k] = dict((kk, round(vv, 5)) for kk, vv in v.items())
            else:
                ex[k] = round(v, 3)
        out["extra"][str(n)] = ex
        del c
        torch.cuda.empty_cache()
    print(json.dumps(out))


main()
