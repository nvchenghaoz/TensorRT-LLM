"""Untimed correctness for the branch-43 SM120 BF16 LUT pins (candidate tree). Pitfall 629 gate (a), (b), (d), (e)
plus the dispatch census and CUDA-graph capture/replay; (c) real activations is real_check.py.

1. Op gate, every shape (sl_qkv, gl_qkv [q|k], sl_o, gl_o) at every census M (BCG buckets, bucket +-1, 301, 1023,
   and the edges of the edited LUT windows): the tree's dispatch (trtllm::cublas_mm where routed, else F.linear)
   vs main's dispatch (F.linear, or main's LUT config recomputed with ctypes cuBLASLt where main routed), raw
   int16, full shape, on randn seed 0, randn seed 1 and heavy-tailed randn*2^U(-8,8) per row (x and W).
   The cand output block is pre-filled 0xA5 (allocator reuse) before the call. First mismatch reported.
2. Dispatch census: torch.profiler kernel names of cand vs main per (shape, M); a kernel may change only at the
   (shape, M) this branch pins (NEW_PINS).
3. Repeated execution = bench_chain.py's exact sequence: same Chain (60 layers, own weights), one eager call and a
   CUDA-graph capture per bucket, then 3 passes (M order fwd, rev, fwd) of 3 warmup + 15 replays; after EVERY
   replay all 120 GEMM outputs are compared bitwise with main's F.linear outputs on the same inputs (the
   interleaved bucket order gives the A,B,A permutation, e.g. 300 -> 600 -> ... -> 2000 -> 2000 -> ... -> 300).
   Inputs (activations bitwise, weights by int64 checksum) unchanged at the end.

usage: python check_enum.py <checkpoint>
"""
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chain import Chain  # noqa: E402
from ltx import LtX, kernel_names, main_dispatch, read_shapes  # noqa: E402

BUCKETS = [300, 600, 900, 1000, 1200, 1500, 1800, 2000]
WARM, ITERS, PASSES = 3, 15, 3
T0 = time.time()

# (K, N) -> token counts this branch pins (new exact entries or a new config at an already-routed count)
NEW_PINS = {
    (5376, 16384): {300, 600, 900, 1200, 2000},
    (5376, 18432): {300, 1000, 1200, 1500, 1800, 2000},
    (8192, 5376): {900, 1200, 1500, 2000},
    (16384, 5376): {900, 1200, 1500, 2000},
}
ADDED_ROUTES = {(5376, 16384): {600}, (5376, 18432): {300, 1000}, (8192, 5376): {1200, 1500, 2000},
                (16384, 5376): {1200, 1500, 2000}}
_H = [-1, 0, 0, 0, 0, 0, 0, 0]
# main's (6dd1ebe2) sm120_bf16_range_algo_list entries for m <= 2099, the window this branch edits
MAIN_LUT = {
    (5376, 16384): [(299, _H), (300, [67, 467, 35, 1, 0, 0, 30, 0]), (899, _H), (900, [67, 24, 35, 1, 0, 0, 30, 2]),
                    (999, _H), (1000, [67, 24, 35, 1, 0, 0, 31, 2]), (1199, _H), (1200, [67, 409, 35, 1, 0, 0, 30, 0]),
                    (1344, _H), (1408, [67, 24, 35, 1, 0, 0, 31, 2]), (1536, [67, 202, 35, 1, 0, 0, 15, 0]),
                    (1600, [67, 29, 35, 1, 0, 0, 23, 0]), (1728, _H), (1792, [67, 24, 35, 1, 0, 0, 31, 2]),
                    (1799, [67, 463, 35, 1, 0, 0, 30, 0]), (1800, [67, 202, 35, 1, 0, 0, 15, 0]),
                    (1856, [67, 463, 35, 1, 0, 0, 30, 0]), (1920, [67, 202, 35, 1, 0, 0, 15, 0]), (1984, _H),
                    (2000, [67, 412, 35, 1, 0, 0, 31, 0]), (2240, _H)],
    (5376, 18432): [(599, _H), (600, [67, 24, 35, 1, 0, 0, 31, 2]), (899, _H), (900, [67, 202, 35, 1, 0, 0, 15, 0]),
                    (1199, _H), (1200, [67, 24, 35, 1, 0, 0, 31, 2]), (1279, _H), (1280, [67, 24, 35, 1, 0, 0, 31, 2]),
                    (1344, [67, 202, 35, 1, 0, 0, 15, 0]), (1408, [67, 29, 35, 1, 0, 0, 23, 0]), (1472, _H),
                    (1600, [67, 24, 35, 1, 0, 0, 31, 2]), (1792, _H), (1800, [67, 24, 35, 1, 0, 0, 31, 2]),
                    (1856, [67, 412, 35, 1, 0, 0, 31, 0]), (1984, _H), (1999, [67, 209, 35, 1, 0, 0, 31, 0]),
                    (2000, [67, 414, 35, 1, 0, 0, 22, 0]), (2112, [67, 209, 35, 1, 0, 0, 31, 0])],
    (8192, 5376): [(899, _H), (900, [67, 24, 35, 1, 0, 0, 30, 2]), (1792, _H), (1856, [67, 463, 35, 1, 0, 0, 30, 0]),
                   (2099, _H)],
    (16384, 5376): [(899, _H), (900, [67, 24, 35, 1, 0, 0, 144, 2]), (999, _H), (1000, [67, 24, 35, 1, 0, 0, 144, 2]),
                    (1279, _H), (1280, [67, 29, 35, 1, 0, 0, 22, 0]), (1792, _H),
                    (1856, [67, 463, 35, 1, 0, 0, 30, 0]), (2099, _H)],
}
CENSUS_EXTRA = [299, 301, 599, 601, 899, 901, 999, 1001, 1023, 1199, 1201, 1279, 1280, 1281, 1344, 1408, 1472,
                1473, 1499, 1501, 1600, 1792, 1793, 1799, 1801, 1856, 1984, 1985, 1999, 2001, 2099]


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def lut_lookup(ents, m):
    for m_max, cfg in ents:
        if m <= m_max:
            return cfg if cfg[0] >= 0 else None
    return None


def main():
    ckpt = sys.argv[1]
    shapes, info = read_shapes(ckpt)
    op, routed, mg = main_dispatch()
    lt = LtX()
    log("shapes", json.dumps(shapes), torch.cuda.get_device_name(), os.environ.get("CUDA_VISIBLE_DEVICES"))
    ok = True

    def main_routed(N, K, M):
        return routed(N, K, M) and M not in ADDED_ROUTES.get((K, N), set())

    def main_op(x, W):
        N, K = W.shape
        M = x.shape[0]
        if main_routed(N, K, M):
            cfg = lut_lookup(MAIN_LUT[(K, N)], M)
            assert cfg is not None, (K, N, M)
            out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
            lt.matmul(x, W, out, lt.make_algo(cfg))
            return out
        return F.linear(x, W)

    def cand_op(x, W):
        t = torch.empty(x.shape[0], W.shape[0], device="cuda", dtype=torch.bfloat16)
        t.view(torch.int16).fill_(-23131)  # 0xA5A5; the freed block is what the op's output reuses
        del t
        return op(x, W)

    # ---------------- 1 + 2: op gate and dispatch census
    gate = {"cases": 0, "bitwise": 0, "failures": []}
    census = {"changed": [], "unexpected_change": [], "missing_change": []}
    names = {"sl_qkv": None, "gl_qkv_dedup": None, "sl_o": None, "gl_o": None}
    ms_all = sorted(set(BUCKETS + CENSUS_EXTRA))
    for sname in names:
        N, K = shapes[sname]
        for di, (seed, heavy) in enumerate(((0, False), (1, False), (2, True))):
            g = torch.Generator(device="cuda").manual_seed(100 + seed)

            def mk(r, c, scale):
                t = torch.randn(r, c, device="cuda", dtype=torch.float32, generator=g)
                if heavy:
                    t = t * torch.exp2(torch.empty(r, 1, device="cuda").uniform_(-8, 8, generator=g))
                return (t * scale).to(torch.bfloat16)

            W = mk(N, K, 0.02)
            X = mk(max(ms_all), K, 1.0)
            for M in ms_all:
                x = X[:M]
                ref_main = main_op(x, W)
                ref_fl = F.linear(x, W)
                o = cand_op(x, W)
                gate["cases"] += 1
                d_main = o.view(torch.int16) != ref_main.view(torch.int16)
                d_fl = o.view(torch.int16) != ref_fl.view(torch.int16)
                n_main, n_fl = int(d_main.sum()), int(d_fl.sum())
                if n_main == 0 and n_fl == 0:
                    gate["bitwise"] += 1
                else:
                    r, c = (d_main if n_main else d_fl).nonzero()[0].tolist()
                    gate["failures"].append(dict(shape=sname, M=M, data=di, vs_main=n_main, vs_flinear=n_fl, row=r,
                                                 col=c, cand=float(o[r, c]), main=float(ref_main[r, c])))
                    log(f"!! MISMATCH {sname} M={M} data={di}: {gate['failures'][-1]}")
                if di == 0:
                    kc = kernel_names(lambda: op(x, W))
                    km = kernel_names(lambda: main_op(x, W))
                    pinned = M in NEW_PINS[(K, N)]
                    changed = kc != km
                    if changed:
                        census["changed"].append([sname, M, km, kc])
                    if changed and not pinned:
                        census["unexpected_change"].append([sname, M, km, kc])
                    if pinned and not changed:
                        census["missing_change"].append([sname, M, km, kc])
                    if pinned or changed:
                        log(f"census {sname} M={M}: pinned={pinned} main {km} -> cand {kc}")
            del W, X
            torch.cuda.empty_cache()
        log(f"op gate {sname} done: {gate['cases']} cases so far, {len(gate['failures'])} failures")
    gate_ok = gate["cases"] > 0 and not gate["failures"]
    census_ok = not census["unexpected_change"] and not census["missing_change"]
    log("op gate", json.dumps({k: v for k, v in gate.items() if k != "failures"}), "failures", gate["failures"][:5])
    log("census unexpected", census["unexpected_change"], "missing", census["missing_change"])
    ok &= gate_ok and census_ok

    # ---------------- 3: bench_chain.py's sequence, untimed, every replay checked
    chain = Chain(ckpt, max(BUCKETS))
    xs_snap = {"xh": chain.xh.clone(), **{k: v.clone() for k, v in chain.xo.items()}}
    w_sum = [(int(a.view(torch.int16).sum(dtype=torch.int64)), int(b.view(torch.int16).sum(dtype=torch.int64)))
             for a, b in chain.W]
    refs = {}
    graphs = {}
    eager_ok = True
    for M in BUCKETS:
        refs[M] = chain.run(main_op, M, keep=True)
        eo = chain.run(op, M, keep=True)
        eager_ok &= all(torch.equal(a.view(torch.int16), b.view(torch.int16)) for a, b in zip(eo, refs[M]))
        del eo
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            chain.run(op, M)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            outs = chain.run(op, M, keep=True)
        torch.cuda.synchronize()
        graphs[M] = (g, outs)
        log(f"chain M={M}: eager cand == main bitwise so far {eager_ok}; captured")
    rep = {"replays": 0, "bad_replays": 0, "first_bad": None}
    seq = []
    for p in range(PASSES):
        order = BUCKETS if p % 2 == 0 else list(reversed(BUCKETS))
        for M in order:
            g, outs = graphs[M]
            for i in range(WARM + ITERS):
                for t in outs:
                    t.view(torch.int16).fill_(-23131)
                g.replay()
                good = all(torch.equal(a.view(torch.int16), b.view(torch.int16)) for a, b in zip(outs, refs[M]))
                rep["replays"] += 1
                if not good:
                    rep["bad_replays"] += 1
                    if rep["first_bad"] is None:
                        bad = [j for j, (a, b) in enumerate(zip(outs, refs[M]))
                               if not torch.equal(a.view(torch.int16), b.view(torch.int16))]
                        rep["first_bad"] = dict(pass_=p, M=M, replay=i, gemm=bad[0], n_bad_gemms=len(bad))
            seq.append(M)
        log(f"chain pass {p} done: {rep}")
    inputs_ok = all(torch.equal(xs_snap["xh"], chain.xh) for _ in [0]) and all(
        torch.equal(xs_snap[k], v) for k, v in chain.xo.items())
    inputs_ok &= w_sum == [(int(a.view(torch.int16).sum(dtype=torch.int64)),
                            int(b.view(torch.int16).sum(dtype=torch.int64))) for a, b in chain.W]
    chain_ok = eager_ok and rep["bad_replays"] == 0 and rep["replays"] == PASSES * len(BUCKETS) * (WARM + ITERS)
    log("chain", json.dumps(rep), "eager", eager_ok, "inputs unchanged", inputs_ok, "bucket order", seq)
    ok &= chain_ok and inputs_ok
    print(json.dumps({
        "passed": bool(ok),
        "compared": "tree dispatch (trtllm::cublas_mm SM120 LUT where routed, else F.linear) vs main's dispatch "
                    "(F.linear / main LUT config via cuBLASLt) and F.linear, raw int16, full shape; randn x2 seeds + "
                    "heavy-tailed; kernel census; bench_chain.py's graph sequence with every replay checked",
        "op_gate": {"cases": gate["cases"], "bitwise": gate["bitwise"], "failures": gate["failures"][:10]},
        "census": {"changed": census["changed"], "unexpected_change": census["unexpected_change"],
                   "missing_change": census["missing_change"]},
        "chain": rep, "chain_eager_bitwise": eager_ok, "inputs_unchanged": inputs_ok}))


if __name__ == "__main__":
    main()
