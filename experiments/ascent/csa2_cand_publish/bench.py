"""CSA2 candidate-source publication (layer 20): CSA2Indexer._publish_candidates.

Main caps CandidatePublishKernel at 4096 padded blocks, so width 36864 (4608
blocks of 8) and 34816 (4352 blocks) take the PyTorch fallback (amax, CUDA radix
TopK = topKPerRowPrefill + radixSortKVInPlace, sort, ...). The branch raises the
cap to 8192 and runs one CuTe launch.

Modes:
  bench  - graph replay with a 256 MB L2 flush before each replay; per-case us/call,
           launches/call and kernel names (eager profiler of one call).
  check  - untimed: the same warmup + iters graph-replay sequence, outputs reset to
           garbage before every replay; every replay compared with the PyTorch
           fallback (dsl_available patched False) on the same inputs.
"""

import argparse
import json
import sys
import types

import torch

import tensorrt_llm._torch.attention.backends.sparse.csa2.indexer as ix
from tensorrt_llm._torch.attention.backends.sparse.csa2.indexer import CSA2Indexer
from tensorrt_llm._torch.modules.top_k import TopK, TopKImplementation

BLOCK = 8
TOPK_BLOCKS = 2048
SPARSE = 8
LAYER = 20
CASES = ((1, 36864), (2, 36864), (4, 36864), (8, 36864), (2048, 34816))
WARMUP = 5
ITERS = 60


def log(*a):
    print(*a, flush=True)


def make_indexer():
    ind = object.__new__(CSA2Indexer)
    torch.nn.Module.__init__(ind)
    ind.layer_idx = LAYER
    ind.layout = types.SimpleNamespace(candidate_block_size=BLOCK)
    ind.candidate_top_k = TopK(TOPK_BLOCKS, prefill_implementation=TopKImplementation.CUDA_RADIX)
    ind.sparse_block = SPARSE
    return ind


def make_inputs(count, width, seed, kind="normal"):
    g = torch.Generator(device="cuda").manual_seed(seed)
    s = torch.randn(count, width, generator=g, device="cuda", dtype=torch.float32)
    if kind == "ties":
        s = torch.round(s * 4) / 4
    if count == 2048:  # prefill chunk: causal visibility over the cached prefix
        visible = 32768 + torch.arange(count, device="cuda", dtype=torch.int32) + 1
    else:  # decode: graph width is the admitted bound, context is shorter
        visible = torch.tensor(
            [34817 + 37 * r for r in range(count)], device="cuda", dtype=torch.int32
        )
    cols = torch.arange(width, device="cuda")
    s = s.masked_fill(cols[None, :] >= visible[:, None].long(), -torch.inf)
    if kind == "special":
        rows = count
        # row 0: all -inf, visible 0 (no pinned block)
        s[0] = -torch.inf
        visible[0] = 0
        if rows > 1:  # all reachable selected: short context (< 2048 blocks)
            visible[1] = 5000
            s[1, 5000:] = -torch.inf
        if rows > 2:  # masked pages: -inf holes in the middle
            s[2, 1000:9000] = -torch.inf
            s[2, 20000:20512] = -torch.inf
        if rows > 3:  # padding row: visible past width, pin beyond last block
            visible[3] = width + 100
        if rows > 4:  # tiny context, one partial block
            visible[4] = 3
            s[4, 3:] = -torch.inf
        if rows > 5:  # constant scores (all ties)
            s[5, : int(visible[5])] = 0.5
    width_pub = min(TOPK_BLOCKS, -(-width // BLOCK)) * BLOCK
    return s.contiguous(), visible, width_pub


def make_call(ind, scores, visible, width_pub):
    count = scores.shape[0]
    results = {LAYER: torch.full((count, width_pub), -7, dtype=torch.int32, device="cuda")}
    blocks = {LAYER: torch.zeros((count, width_pub // SPARSE), dtype=torch.int32, device="cuda")}
    counts = {LAYER: torch.zeros((count, 1), dtype=torch.int32, device="cuda")}

    def fn():
        ind._publish_candidates(scores, visible, results, 0, count, width_pub, blocks, counts)

    def outs():
        return results[LAYER], blocks[LAYER], counts[LAYER]

    def reset():
        results[LAYER].fill_(-7)
        blocks[LAYER].fill_(-7)
        counts[LAYER].fill_(-7)

    return fn, outs, reset


_FLUSH = {}


def l2_flush():
    buf = _FLUSH.get("b")
    if buf is None:
        buf = _FLUSH["b"] = torch.empty(64 * 1024 * 1024, dtype=torch.float32, device="cuda")
    buf.zero_()


def sm_clock():
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        return int(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))
    except Exception:  # noqa: BLE001
        return -1


def capture(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


def kernel_names(fn):
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = [e.name for e in prof.events() if e.device_type.name == "CUDA"
             and "Memcpy" not in e.name and "Memset" not in e.name]
    return names


def bench(args):
    ind = make_indexer()
    out = {}
    table = []
    for count, width in CASES:
        scores, visible, wp = make_inputs(count, width, 1234 + count)
        fn, _, _ = make_call(ind, scores, visible, wp)
        names = kernel_names(fn)
        g = capture(fn)
        for _ in range(WARMUP):
            l2_flush()
            g.replay()
        torch.cuda.synchronize()
        times, clocks = [], []
        for i in range(ITERS):
            l2_flush()
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record()
            g.replay()
            b.record()
            if i % 20 == 10:
                clocks.append(sm_clock())
            b.synchronize()
            times.append(a.elapsed_time(b) * 1000.0)
        times.sort()
        med = times[len(times) // 2]
        p10 = times[int(0.1 * (ITERS - 1))]
        p90 = times[int(0.9 * (ITERS - 1))]
        nf = (p90 - p10) / med
        short = sorted({n[:60] for n in names})
        log(f"count={count} width={width} us={med:.2f} p10={p10:.2f} p90={p90:.2f} "
            f"noise={nf:.4f} launches={len(names)} sm_clock_min={min(clocks)} kernels={short}")
        table.append(dict(count=count, width=width, us=med, p10=p10, p90=p90, launches=len(names),
                          sm_clock_min=min(clocks), kernels=short))
        out[str(count)] = {"kernel_ms": med / 1000.0, "noise_floor": nf}
        del g
    log(json.dumps({"table": table}))
    print(json.dumps({"batch": out, "output": {"class": "exact"}}))


def tie_rows(scores, visible, sel_blocks):
    """Rows whose K-th and (K+1)-th block values tie (selection set not unique)."""
    count, width = scores.shape
    pad = torch.nn.functional.pad(scores, (0, -width % BLOCK), value=-torch.inf)
    bm = pad.reshape(count, -1, BLOCK).amax(-1)
    latest = (visible.long() - 1) // BLOCK
    ids = torch.arange(bm.shape[1], device="cuda")
    bm = bm.masked_fill(ids[None, :] == latest[:, None], torch.inf)
    if sel_blocks >= bm.shape[1]:
        return set(), bm
    v = bm.sort(dim=1, descending=True).values
    tie = (v[:, sel_blocks - 1] == v[:, sel_blocks]) & (v[:, sel_blocks - 1] > -torch.inf)
    return set(torch.nonzero(tie).flatten().tolist()), bm


def check(args):
    ind = make_indexer()
    ok = True
    report = []
    torch.cuda.synchronize()
    for count, width in CASES:
        for kind in ("normal", "special", "ties"):
            scores, visible, wp = make_inputs(count, width, 99 + count, kind)
            # Reference: the PyTorch fallback (what main runs at these widths).
            ref_results = {}
            ref_blocks, ref_counts = {}, {}
            saved = ix.dsl_available
            ix.dsl_available = lambda: False
            try:
                ind._publish_candidates(scores, visible, ref_results, 0, count, wp,
                                        ref_blocks, ref_counts)
            finally:
                ix.dsl_available = saved
            torch.cuda.synchronize()
            ref = (ref_results[LAYER], ref_blocks[LAYER], ref_counts[LAYER])
            sel = min(TOPK_BLOCKS, -(-width // BLOCK))
            ties, bm = tie_rows(scores, visible, sel)
            fn, outs, reset = make_call(ind, scores, visible, wp)
            mem0 = None
            g = capture(fn)
            torch.cuda.synchronize()
            mem0 = torch.cuda.memory_allocated()
            bad_iters = 0
            tie_bad = 0
            nondet = 0
            first = None
            nontie = torch.ones(count, dtype=torch.bool, device="cuda")
            if ties:
                nontie[list(ties)] = False
            for it in range(WARMUP + ITERS):
                reset()
                g.replay()
                torch.cuda.synchronize()
                got = outs()
                eq = all(torch.equal(a[nontie], b[nontie]) for a, b in zip(got, ref))
                if not eq:
                    bad_iters += 1
                if first is None:
                    first = tuple(t.clone() for t in got)
                elif not all(torch.equal(a, b) for a, b in zip(got, first)):
                    nondet += 1
                if ties and it in (0, WARMUP + ITERS - 1):
                    # tie rows: same multiset of selected block values (an equally valid set),
                    # same valid count; non-boundary blocks identical.
                    for r in ties:
                        gc, rc = got[0][r], ref[0][r]
                        gb = gc[gc >= 0].reshape(-1)
                        rb = rc[rc >= 0].reshape(-1)
                        gbl = torch.unique(gb // BLOCK)
                        rbl = torch.unique(rb // BLOCK)
                        same_vals = torch.equal(bm[r, gbl].sort().values, bm[r, rbl].sort().values)
                        sorted_ok = bool((gb[1:] > gb[:-1]).all()) if gb.numel() > 1 else True
                        if not (same_vals and sorted_ok and int(got[2][r]) == int(ref[2][r])):
                            tie_bad += 1
            # Pure replay segment: the allocator must see no allocation at all.
            torch.cuda.synchronize()
            n0 = torch.cuda.memory_stats().get("allocation.all.allocated", 0)
            mem0 = torch.cuda.memory_allocated()
            for _ in range(20):
                g.replay()
            torch.cuda.synchronize()
            n1 = torch.cuda.memory_stats().get("allocation.all.allocated", 0)
            mem1 = mem0 + (n1 - n0)  # alloc_delta = allocations during replays
            mem1 = torch.cuda.memory_allocated() if n1 == n0 else mem1
            passed = bad_iters == 0 and tie_bad == 0 and nondet == 0 and mem1 == mem0
            ok &= passed
            line = dict(count=count, width=width, kind=kind, tie_rows=len(ties),
                        bad_iters=bad_iters, tie_bad=tie_bad, nondet_iters=nondet,
                        replays=WARMUP + ITERS,
                        alloc_delta=mem1 - mem0, passed=passed)
            log(json.dumps(line))
            report.append(line)
            del g
    # Decode graph capture: the capacity-8192 kernel must be compiled in eager warmup.
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import kernel as k

    keys = [key for key in getattr(k, "_COMPILED", {}) if key[0][0] == "candidate_publish"]
    log("compiled candidate_publish keys:", keys)
    names = {}
    for count in (1, 2, 4, 8):
        scores, visible, wp = make_inputs(count, 36864, 7)
        fn, _, _ = make_call(ind, scores, visible, wp)
        n = kernel_names(fn)
        names[count] = n
        bad = [x for x in n if "topKPerRow" in x or "radixSort" in x]
        log(f"decode count={count} launches={len(n)} kernels={sorted(set(n))}")
        if bad:
            ok = False
    print(json.dumps({"passed": bool(ok), "compared_against": "PyTorch fallback of "
                      "CSA2Indexer._publish_candidates (main's path at >4096 blocks)",
                      "cases": report, "decode_launches": {c: len(v) for c, v in names.items()}}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("bench", "check"))
    a = p.parse_args()
    log("device", torch.cuda.get_device_name(), "python", sys.version.split()[0])
    {"bench": bench, "check": check}[a.mode](a)
