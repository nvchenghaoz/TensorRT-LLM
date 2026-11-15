"""Idea 116 (branch 50): the timed protocol of bench_pair.py, shared with the untimed checks.

WARMUP, ITERS and the L2 flush come from bench_pair.py (which sets them on nvfp4_dense/bench.py
and whose bench.timed runs: flush.zero_(); fn() for WARMUP calls, then flush.zero_(); fn() for
ITERS calls, no reset between calls). seq() runs that exact sequence untimed and compares every
call's output to call 1 (int16 for BF16, bytes otherwise) and, given a reference, records each
call's max-abs error against it. Output-buffer policy as the timed run: the op allocates its
output per call and it is freed before the next call (call 1 is kept as a clone).
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "nvfp4_dense"))
import bench  # noqa: E402
import bench_pair  # noqa: E402,F401  (sets bench.WARMUP, bench.ITERS)

WARMUP, ITERS = bench.WARMUP, bench.ITERS
FLUSH_BYTES = bench_pair.FLUSH_BYTES
make_flush = bench_pair.make_flush
CALLS = WARMUP + ITERS


def protocol_line():
    return (f"[protocol] bench WARMUP={bench.WARMUP} ITERS={bench.ITERS} flush={bench_pair.FLUSH_BYTES >> 20}MiB | "
            f"check WARMUP={WARMUP} ITERS={ITERS} flush={FLUSH_BYTES >> 20}MiB calls/key={CALLS}")


def _flat(o):
    return tuple(o) if isinstance(o, (tuple, list)) else (o,)


def _bits(x):
    return x.view(torch.int16) if x.dtype in (torch.bfloat16, torch.float16) else x.view(torch.uint8)


def _ne(o, first):
    tot = None
    for a, b in zip(_flat(o), _flat(first)):
        c = (_bits(a) != _bits(b)).sum()
        tot = c if tot is None else tot + c
    return tot


def seq(fn, flush, ref=None):
    """Run the bench.timed sequence untimed. Returns (call-1 output, calls run, per-call
    mismatch counts vs call 1, per-call max-abs error vs ref or None)."""
    nd = torch.zeros(CALLS, dtype=torch.int64, device="cuda")
    err = torch.zeros(CALLS, dtype=torch.float32, device="cuda") if ref is not None else None
    first = None
    calls = 0
    for i in range(CALLS):
        flush.zero_()
        o = fn()
        calls += 1
        if first is None:
            first = tuple(x.clone() for x in _flat(o)) if isinstance(o, (tuple, list)) else o.clone()
        else:
            nd[i] = _ne(o, first)
        if err is not None:
            err[i] = (o.float() - ref).abs().max()
        del o
    torch.cuda.synchronize()
    return first, calls, nd.tolist(), (err.tolist() if err is not None else None)


def snapshot(ts):
    return [t.clone() for t in ts]


def unchanged(snap, ts):
    return all(torch.equal(_bits(a) if a.dtype != torch.float32 else a, _bits(b) if b.dtype != torch.float32 else b)
               for a, b in zip(snap, ts))
