"""CSA2 SM120 decode attention: main's _forward_pools vs the 2-launch Triton split sink kernel.

Shapes: one TP4 rank. q [count, 16, 512] bf16, SWA pool [cap, 128, 512] and extra pool
[cap, 512, 512] bf16 (width 640), prepared lengths int32 (dense prefix), sinks [16] fp32,
sm_scale 1/sqrt(512). Counts 1/2/4/8. Pools hold STALE NON-ZERO data (incl. large values)
past each query's length.

Modes
  time  : either tree. The production generation path (candidate: _forward_pools_decode;
          base: _forward_pools = stage_rows_and_mask + FA2 split-KV(64) + merge +
          sink_epilogue) under CUDA-graph replay, cold L2 (256 MB flush before every replay),
          lengths 640 (production: 34K context fills SWA + top-512). Also reports random
          lengths. Prints the evidence JSON.
  sweep : candidate only. Timing per (chunk, block_n, num_warps) config + error vs FP64.
  check : candidate only. FP64 reference vs new and vs main's _forward_pools at special
          lengths; repeated graph replays (10 warmup + 100) vs reference and eager;
          replays with changing pools/lengths (incl. 0 and 640) vs eager.
"""

import argparse
import json
import math
import sys

import torch

from tensorrt_llm._torch.attention.backends.sparse.csa2.backend import CSA2FlashInfer

HEADS, DIM, SWA, EXTRA = 16, 512, 128, 512
WIDTH = SWA + EXTRA
CAP = 16
SCALE = 1.0 / math.sqrt(DIM)
COUNTS = (1, 2, 4, 8)
WARMUP, ITERS = 10, 100
NEW = hasattr(CSA2FlashInfer, "_forward_pools_decode")
SPECIAL = (0, 1, 63, 64, 65, 128, 129, 639, 640, 127, 31, 33)


def log(*a):
    print(*a, flush=True)


def make_inputs(count, seed, lens=None, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn((count, HEADS, DIM), generator=g).to(torch.bfloat16)
    swa = torch.randn((CAP, SWA, DIM), generator=g).to(torch.bfloat16)
    extra = torch.randn((CAP, EXTRA, DIM), generator=g).to(torch.bfloat16)
    if lens == "full":
        lens = torch.full((count,), WIDTH, dtype=torch.int64)
    elif lens is None:
        lens = torch.randint(0, WIDTH + 1, (count,), generator=g)
    else:
        lens = torch.tensor(lens, dtype=torch.int64)
    # Stale non-zero data past each length, including large values.
    for i in range(count):
        n = int(lens[i])
        stale = torch.randn((WIDTH, DIM), generator=g) * 50.0 + 3.0
        stale[::7] = 300.0
        stale = stale.to(torch.bfloat16)
        if n < SWA:
            swa[i, n:] = stale[n:SWA]
        extra[i, max(n - SWA, 0):] = stale[SWA + max(n - SWA, 0):]
    sink = torch.randn((HEADS,), generator=g) * 2.0
    return [t.to(device) for t in (q, swa, extra, lens.to(torch.int32), sink.float())]


def reference(q, swa, extra, lens, sink):
    count = q.shape[0]
    kv = torch.cat((swa[:count], extra[:count]), dim=1).double()
    valid = torch.arange(WIDTH, device=q.device)[None, :] < lens[:, None]
    kv = torch.where(valid[..., None], kv, 0.0)
    s = torch.einsum("bhd,bkd->bhk", q.double(), kv) * SCALE
    s = s.masked_fill(~valid[:, None, :], -math.inf)
    lse = torch.logsumexp(s, dim=-1)
    p = torch.nan_to_num(torch.softmax(s, dim=-1), nan=0.0)
    o = torch.einsum("bhk,bkd->bhd", p, kv)
    out = o * torch.sigmoid(lse - sink.double()[None, :])[..., None]
    return torch.where(valid.any(-1)[:, None, None], out, 0.0)


def prod(fi, q, swa, extra, lens, sink, out=None):
    fn = fi._forward_pools_decode if NEW else fi._forward_pools
    return fn(q, swa, extra, lens, sink, SCALE, out=out)


def old(fi, q, swa, extra, lens, sink, out=None):
    return fi._forward_pools(q, swa, extra, lens, sink, SCALE, out=out)


FLUSH = None


def time_graph(fn):
    global FLUSH
    if FLUSH is None:
        FLUSH = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    for _ in range(WARMUP):
        FLUSH.zero_()
        graph.replay()
    for i in range(ITERS):
        FLUSH.zero_()
        starts[i].record()
        graph.replay()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(a.elapsed_time(b) for a, b in zip(starts, ends))
    med = times[len(times) // 2]
    p10, p90 = times[len(times) // 10], times[(9 * len(times)) // 10]
    return med, (p90 - p10) / med, graph, out


def time_case(count, seed, lens, fn=prod):
    # Pools re-created from the same seed before each timed sequence.
    q, swa, extra, ln, sink = make_inputs(count, seed, lens)
    fi = CSA2FlashInfer()
    out = torch.empty_like(q)
    med, noise, graph, res = time_graph(lambda: fn(fi, q, swa, extra, ln, sink, out))
    graph.replay()
    torch.cuda.synchronize()
    ref = reference(q, swa, extra, ln, sink)
    err = (res.double() - ref).abs().max().item()
    fin = bool(torch.isfinite(res).all())
    del graph
    return med, noise, err, fin, ln.tolist()


def mode_time(args):
    batch = {}
    for count in COUNTS:
        med, noise, err, fin, _ = time_case(count, args.seed + count, "full")
        rmed, rnoise, rerr, rfin, rl = time_case(count, args.seed + 50 + count, None)
        log(f"count {count}: full {med*1e3:.2f}us noise {noise:.3f} err {err:.3e} | "
            f"random lens {rl}: {rmed*1e3:.2f}us noise {rnoise:.3f} err {rerr:.3e}")
        batch[str(count)] = {
            "kernel_ms": med, "noise_floor": noise, "max_abs_err_vs_fp64": err, "finite": fin,
            "random_lens_ms": rmed, "random_lens_noise": rnoise, "random_lens": rl,
            "random_lens_err": rerr,
        }
    print(json.dumps({
        "batch": batch,
        "path": "_forward_pools_decode (2-launch Triton split sink)" if NEW
        else "_forward_pools (stage_rows_and_mask + FA2 split-KV 64 + merge + sink_epilogue)",
        "output": {"class": "reordered"},
    }))


def mode_sweep(args):
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import split_sink_decode as ssd

    configs = [(64, 64, 4), (64, 32, 4), (32, 32, 4), (32, 16, 4), (16, 16, 4), (128, 32, 4),
               (128, 64, 4), (64, 64, 8), (32, 32, 8), (64, 32, 8), (32, 16, 2), (16, 16, 2)]
    res = {}
    for count in COUNTS:
        base = time_case(count, args.seed + count, "full", fn=old)
        log(f"count {count} main _forward_pools: {base[0]*1e3:.2f}us noise {base[1]:.3f} "
            f"err {base[2]:.3e}")
        res[f"main/{count}"] = base[:3]
        for cfg in configs:
            ssd.SPLIT_CONFIGS[count] = cfg
            try:
                r = time_case(count, args.seed + count, "full")
                log(f"count {count} cfg {cfg}: {r[0]*1e3:.2f}us noise {r[1]:.3f} err {r[2]:.3e}")
                res[f"{cfg}/{count}"] = r[:3]
            except Exception as e:  # noqa: BLE001
                log(f"count {count} cfg {cfg}: FAILED {type(e).__name__}: {str(e)[:200]}")
        ssd.SPLIT_CONFIGS.pop(count, None)
    print(json.dumps({"sweep": res}))


def mode_check(args):
    assert NEW, "check runs in the candidate tree"
    ok = True
    report = {}
    worst_ratio = 0.0
    for count in COUNTS:
        for trial in range(6):
            if trial < 3:
                lens = [SPECIAL[(i + 3 * trial) % len(SPECIAL)] for i in range(count)]
            elif trial == 3:
                lens = [0 if i % 2 == 0 else 640 for i in range(count)]
            elif trial == 4:
                lens = [64 if i % 2 == 0 else 65 for i in range(count)]
            else:
                lens = None
            q, swa, extra, ln, sink = make_inputs(count, args.seed + 100 * count + trial, lens)
            ref = reference(q, swa, extra, ln, sink)
            fi = CSA2FlashInfer()
            base = old(fi, q, swa, extra, ln, sink)
            new = prod(fi, q, swa, extra, ln, sink)
            again = prod(fi, q, swa, extra, ln, sink)
            torch.cuda.synchronize()
            e_old = (base.double() - ref).abs().max().item()
            e_new = (new.double() - ref).abs().max().item()
            d = (new.float() - base.float()).abs().max().item()
            fin = bool(torch.isfinite(new).all())
            live = ln > 0
            zero_ok = bool((new[~live] == 0).all())
            tol_ok = torch.allclose(new.double(), ref, rtol=1.6e-2, atol=1.6e-2)
            det = torch.equal(new, again)
            good = (fin and tol_ok and zero_ok and det
                    and e_new <= max(e_old * 1.25, e_old + 2 ** -8)
                    and new.shape == q.shape and new.stride() == q.stride())
            if e_old > 0:
                worst_ratio = max(worst_ratio, e_new / e_old)
            ok &= good
            k = f"acc/{count}/{trial}"
            report[k] = dict(lens=ln.tolist(), err_new_vs_fp64=e_new, err_main_vs_fp64=e_old,
                             max_abs_diff_new_vs_main=d, finite=fin, empty_rows_zero=zero_ok,
                             deterministic=det, passed=good)
            log(k, report[k])
    # Repeated graph replays: same warmup/iters as timing (10 + 100), every iteration
    # compared with the FP64 reference and with eager.
    for count in COUNTS:
        fi = CSA2FlashInfer()
        lens = [SPECIAL[i % len(SPECIAL)] for i in range(count)] if count > 1 else [640]
        q, swa, extra, ln, sink = make_inputs(count, args.seed + count, lens)
        ref = reference(q, swa, extra, ln, sink)
        ref_abs = ref.abs().clamp_min(1e-2)
        out = torch.empty_like(q)
        eager = prod(fi, q, swa, extra, ln, sink).clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            prod(fi, q, swa, extra, ln, sink, out)
        errs, rels = [], []
        bit = True
        for _ in range(WARMUP + ITERS):
            out.fill_(float("nan"))  # reset point: output poisoned, workspace left stale
            graph.replay()
            torch.cuda.synchronize()
            diff = (out.double() - ref).abs()
            errs.append(diff.max().item())
            rels.append((diff / ref_abs).max().item())
            bit &= torch.equal(out, eager)
        good = bit and max(errs) <= 1.6e-2 and bool(torch.isfinite(out).all())
        ok &= good
        report[f"replay/{count}"] = dict(
            lens=ln.tolist(), iters=len(errs), max_abs_err_per_iter_first=errs[0],
            max_abs_err_per_iter_last=errs[-1], max_abs_err_all=max(errs),
            min_abs_err_all=min(errs), max_rel_err_all=max(rels),
            bitwise_vs_eager_all_iters=bit, passed=good)
        log(f"replay/{count}", report[f"replay/{count}"])
        # Replays with changing pools/lengths (incl. 0 and 640) vs eager and reference.
        bit2 = True
        maxerr = 0.0
        for step in range(16):
            if step % 4 == 0:
                lens = [SPECIAL[(step + i) % len(SPECIAL)] for i in range(count)]
            elif step % 4 == 1:
                lens = [0] + [640] * (count - 1)
            elif step % 4 == 2:
                lens = [640] + [0] * (count - 1)
            else:
                lens = None
            nq, nswa, nextra, nln, nsink = make_inputs(count, args.seed + 1000 + step, lens)
            for dst, src in ((q, nq), (swa, nswa), (extra, nextra), (ln, nln), (sink, nsink)):
                dst.copy_(src)
            out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            got = out.clone()
            e = prod(CSA2FlashInfer(), nq, nswa, nextra, nln, nsink)
            torch.cuda.synchronize()
            bit2 &= torch.equal(got, e)
            maxerr = max(maxerr, (got.double() - reference(nq, nswa, nextra, nln, nsink)).abs().max().item())
        good = bit2 and maxerr <= 1.6e-2
        ok &= good
        report[f"changing/{count}"] = dict(steps=16, bitwise_vs_eager=bit2, max_abs_err=maxerr,
                                           passed=good)
        log(f"changing/{count}", report[f"changing/{count}"])
        del graph
    # SWA-only layers (no extra pool, width 128).
    for count in COUNTS:
        lens = [[0, 1, 64, 65, 127, 128, 31, 33][i % 8] for i in range(count)]
        q, swa, extra, ln, sink = make_inputs(count, args.seed + 7 * count, lens)
        fi = CSA2FlashInfer()
        new = fi._forward_pools_decode(q, swa, None, ln, sink, SCALE)
        base = fi._forward_pools(q, swa, None, ln, sink, SCALE)
        kv = swa[:count].double()
        valid = torch.arange(SWA, device=q.device)[None, :] < ln[:, None]
        kv = torch.where(valid[..., None], kv, 0.0)
        s = (torch.einsum("bhd,bkd->bhk", q.double(), kv) * SCALE).masked_fill(~valid[:, None, :], -math.inf)
        lse = torch.logsumexp(s, -1)
        o = torch.einsum("bhk,bkd->bhd", torch.nan_to_num(torch.softmax(s, -1), nan=0.0), kv)
        ref = torch.where(valid.any(-1)[:, None, None], o * torch.sigmoid(lse - sink.double())[..., None], 0.0)
        e_new = (new.double() - ref).abs().max().item()
        e_old = (base.double() - ref).abs().max().item()
        good = bool(torch.isfinite(new).all()) and e_new <= max(e_old * 1.25, e_old + 2 ** -8) \
            and e_new <= 1.6e-2
        ok &= good
        report[f"swa_only/{count}"] = dict(lens=ln.tolist(), err_new=e_new, err_main=e_old,
                                           passed=good)
        log(f"swa_only/{count}", report[f"swa_only/{count}"])
    print(json.dumps({
        "passed": bool(ok),
        "compared": "CSA2FlashInfer._forward_pools_decode (2-launch Triton split sink) vs FP64 "
        "sink-attention reference over valid rows and vs main's _forward_pools (FA2 split-KV); "
        "stale non-zero pool data past lengths; graph replays vs eager",
        "worst_err_ratio_new_over_main": worst_ratio,
        "report": report,
    }))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("time", "sweep", "check"))
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    log(f"new={NEW} device={torch.cuda.get_device_name()}")
    {"time": mode_time, "sweep": mode_sweep, "check": mode_check}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
