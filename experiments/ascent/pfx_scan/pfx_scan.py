#!/usr/bin/env python3
# Idea 76: kernel_mha_prefix step 1 (P search + per-chunk page verification) with batched, coalesced page-table loads.
# Compiles the base tree (git archive of BASE_COMMIT) and the candidate tree (cwd) with NVRTC exactly like the XQA JIT
# (experiments/ascent/cascade/cascade_lib.py), no model load, synthetic serving-shape inputs (24q/4kv d256 e4m3 paged KV,
# 32768-token shared prefix + 2048 + 128 unique tokens, max_seq_len 36864).
#   --mode bench : STEP 0 table (step-1-only variant minus an empty launch of the same grid, base and candidate) and
#                  prefix-only and cascade-total (prefix + kernel_mha) times, base vs candidate interleaved in one process
#                  per round, cold L2 (384 MiB read flush + NaN poison before every graph replay), B=8/16/24/32.
#   --mode check : untimed; per B=8/16/24/32 360 calls (3 rounds x (20 + 100), poison + flush before each) of base and
#                  candidate graphs interleaved: output, FP32 records and cascadeInfo bitwise equal to the base's; P=0,
#                  single-chunk mismatch, tile-end mismatch, two-group, ragged-length and non-256-multiple prefix tables at
#                  B=8/16/24/32/40/64 (40/64 exercise the candidate's second P-search load group); one captured graph per
#                  arm replayed as P changes; read-only inputs unchanged.
import argparse
import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.getcwd(), "experiments/ascent/cascade"))
import cascade_check as cchk  # noqa: E402
import cascade_lib as cl  # noqa: E402

BASE_COMMIT = "8b7eaf6c79fa8fec66d8c0d667a52146fa1e0250"
log = cl.log
STEP1_MARK = "    if (nbSub == 0)\n    {\n        // kernel_mha ignores the records"
EMPTY_MARK = "    uint32_t* const misc = reinterpret_cast<uint32_t*>(pfxSmem + miscOffset);\n"


def variant_tree(src_dir, kind):
    """Copy of src_dir with kernel_mha_prefix cut after step 1 ('step1') or at entry ('empty')."""
    d = tempfile.mkdtemp(prefix=f"pfx_{kind}_")
    dst = os.path.join(d, "xqa")
    shutil.copytree(src_dir, dst)
    p = os.path.join(dst, "mha.cu")
    s = open(p).read()
    if kind == "step1":
        assert s.count(STEP1_MARK) == 1
        s = s.replace(STEP1_MARK, "    publishInfo();\n    return;\n" + STEP1_MARK)
    elif kind == "empty":
        assert s.count(EMPTY_MARK) == 1
        s = s.replace(
            EMPTY_MARK, EMPTY_MARK + "    if (tid == 0 && idxChunk == 0 && idxRowGrp == 0 && idxHeadGrp == 0)\n"
            "    {\n        cascadeInfo[0] = 0U;\n    }\n    return;\n")
    else:
        raise ValueError(kind)
    open(p, "w").write(s)
    return dst


def prefix_graph(kern, inp, b, m):
    pf = cl.Call(kern, inp, b, m, cascade=True)
    cl.Kernels._launch(kern.prefix, (pf.c, pf.g, cl.NB_K), (kern.pcta, 1, 1), kern.psmem, pf.pfx_args)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        cl.Kernels._launch(kern.prefix, (pf.c, pf.g, cl.NB_K), (kern.pcta, 1, 1), kern.psmem, pf.pfx_args)
    torch.cuda.synchronize()
    pf.graph = g
    return pf


def time_interleaved(graphs, inp, flush, rounds, warmup, iters):
    """graphs: dict name -> CUDAGraph. Each round runs every config's warmup+iters back to back, configs in a rotated
    order per round so no config always follows another."""
    names = list(graphs)
    ts = {n: [] for n in names}
    for k in range(rounds):
        order = names[k % len(names):] + names[:k % len(names)]
        if k % 2:
            order = order[::-1]
        for n in order:
            tot = warmup + iters
            st = [torch.cuda.Event(enable_timing=True) for _ in range(tot)]
            en = [torch.cuda.Event(enable_timing=True) for _ in range(tot)]
            for i in range(tot):
                inp["out"].fill_(float("nan"))
                flush()
                st[i].record()
                graphs[n].replay()
                en[i].record()
            torch.cuda.synchronize()
            ts[n].append(np.array([st[i].elapsed_time(en[i]) for i in range(warmup, tot)]))
        log(f"  round {k}: " + ", ".join(f"{n} {cl.stats(ts[n][-1])['kernel_ms'] * 1e3:.2f}" for n in names))
    out = {}
    for n in names:
        s = cl.stats(np.concatenate(ts[n]))
        out[n] = dict(trimmed_us=s["kernel_ms"] * 1e3, median_us=s["median_ms"] * 1e3, p10_us=s["p10_ms"] * 1e3,
                      p90_us=s["p90_ms"] * 1e3, noise_floor=s["noise_floor"], samples=s["samples"])
    return out


def mode_bench(a):
    base_dir = cl.extract_tree(BASE_COMMIT)
    cand_dir = os.path.join(os.getcwd(), "cpp/kernels/xqa")
    kerns = {}
    for arm, d in (("base", base_dir), ("cand", cand_dir)):
        kerns[arm] = cl.Kernels(d)
        if a.step0:
            kerns[arm + "_step1"] = cl.Kernels(variant_tree(d, "step1"))
            kerns[arm + "_empty"] = cl.Kernels(variant_tree(d, "empty"))
    regs = {n: dict(prefix_regs=k.pregs, prefix_local_bytes=k.plmem, mha_regs=k.regs) for n, k in kerns.items()}
    log(f"regs: {regs}")
    batches = [int(x) for x in a.batches.split(",")]
    inp = cl.build_inputs(max(batches))
    flush = cl.xe.Flusher(a.flush_mb)
    res = {"regs": regs, "batch": {}}
    for b in batches:
        m = cl.main_nb_sub_seq(b)
        graphs = {}
        for n, k in kerns.items():
            graphs[n + ":prefix"] = prefix_graph(k, inp, b, m).graph
        for arm in ("base", "cand"):
            graphs[arm + ":total"] = cl.Call(kerns[arm], inp, b, m, cascade=True).capture().graph
        c, g = cl.cascade_grid(b, m)
        log(f"B={b} m={m} C={c} G={g}: timing {len(graphs)} configs interleaved")
        t = time_interleaved(graphs, inp, flush, a.rounds, a.warmup, a.iters)
        r = dict(nb_sub_seq=m, chunks=c, row_groups=g, times=t)
        if a.step0:
            for arm in ("base", "cand"):
                r[f"{arm}_step1_us"] = t[f"{arm}_step1:prefix"]["trimmed_us"] - t[f"{arm}_empty:prefix"]["trimmed_us"]
                r[f"{arm}_step1_median_us"] = t[f"{arm}_step1:prefix"]["median_us"] - t[f"{arm}_empty:prefix"][
                    "median_us"]
        for what in ("prefix", "total"):
            r[f"{what}_delta_pct"] = 100.0 * (t[f"cand:{what}"]["trimmed_us"] / t[f"base:{what}"]["trimmed_us"] - 1)
        res["batch"][str(b)] = r
        log(f"B={b}: " + json.dumps({k: v for k, v in r.items() if k != "times"}))
        del graphs
    print(json.dumps(res), flush=True)


def scratch_region(inp, b, c):
    off = inp["scratch_ptr"] - inp["scratch_raw"].data_ptr()
    return inp["scratch_raw"][off:off + cl.mha_scratch_offset(b, c)]


class Arm:
    """One captured cascade graph (prefix + kernel_mha) whose call poisons output and cascade scratch first."""

    def __init__(self, kern, inp, b, m):
        self.call = cl.Call(kern, inp, b, m, cascade=True).capture()
        self.inp, self.b, self.c = inp, b, self.call.c

    def run(self, flush):
        self.inp["out"].fill_(float("nan"))
        scratch_region(self.inp, self.b, self.c).fill_(0xFF)
        flush()
        self.call.graph.replay()
        return (self.call.out().clone(), scratch_region(self.inp, self.b, self.c).clone())


def same(x, y):
    return torch.equal(x[0].view(torch.int16), y[0].view(torch.int16)) and torch.equal(x[1], y[1])


def info_of(res, c):
    return res[1][:4 * (1 + c)].view(torch.int32).cpu().tolist()


def set_case(inp, case, bmax, b):
    """cascade_check's tables plus 'tileend' (row b-1's last page of tile 70 differs: P = 70 tiles); 'ragged_len' here
    uses 34817 + 7 r tokens so 64 rows fit max_seq_len."""
    if case == "ragged_len":
        seqlens = [34817 + 7 * r for r in range(bmax)]
        pages = torch.zeros(bmax, 1, 2, cl.MAX_PAGES, dtype=torch.int32)
        for r in range(bmax):
            nbp = math.ceil(seqlens[r] / cl.TPP)
            own = 1024 + r * cl.MAX_PAGES
            blk = torch.tensor(list(range(1024)) + [own + j for j in range(1024, nbp)], dtype=torch.int32)
            pages[r, 0, 0, :nbp] = 2 * blk
            pages[r, 0, 1, :nbp] = 2 * blk + 1
        inp["pages"].copy_(pages.cuda())
        inp["seqlen"].copy_(torch.tensor(seqlens, dtype=torch.int32, device="cuda"))
        torch.cuda.synchronize()
        return
    if case != "tileend":
        cchk.set_case(inp, case, bmax)
        return
    cchk.set_case(inp, "shared", bmax)
    pages = inp["pages"]
    r, j = b - 1, 70 * 8 + 7
    new = 1024 + r * cl.MAX_PAGES + cl.MAX_PAGES - 2
    pages[r, 0, 0, j] = 2 * new
    pages[r, 0, 1, j] = 2 * new + 1
    torch.cuda.synchronize()


def mode_check(a):
    base = cl.Kernels(cl.extract_tree(BASE_COMMIT))
    cand = cl.Kernels(os.path.join(os.getcwd(), "cpp/kernels/xqa"))
    bmax = 64
    inp = cchk.build(bmax)
    flush = cl.xe.Flusher(a.flush_mb)
    rep = {"repeated": {}, "tables": {}, "graph_p_changes": {}}
    passed = True
    n_calls = a.rounds * (a.warmup + a.iters)
    set_case(inp, "shared", bmax, bmax)
    h0 = cl.readonly_hash(inp)
    for b in (8, 16, 24, 32):
        m = cl.main_nb_sub_seq(b)
        ba, ca = Arm(base, inp, b, m), Arm(cand, inp, b, m)
        ref = ba.run(flush)
        nb_eq = nc_eq = 0
        for i in range(n_calls):
            if same(ba.run(flush), ref):
                nb_eq += 1
            if same(ca.run(flush), ref):
                nc_eq += 1
            if i % 120 == 119:
                log(f"  B={b}: {i + 1}/{n_calls} calls, cand==base {nc_eq}, base==base {nb_eq}")
        info = info_of(ref, ba.c)
        fin = bool(torch.isfinite(ref[0].float()).all().item())
        ok = nb_eq == n_calls and nc_eq == n_calls and fin and info[0] == 32768 and not any(info[1:])
        rep["repeated"][f"B{b}"] = dict(calls=n_calls, chunks=ba.c, cand_bitwise_base=nc_eq, base_repeat=nb_eq,
                                        finite=fin, P=info[0], chunk_flags=info[1:], passed=ok)
        passed &= ok
        log(f"repeated B={b} C={ba.c}: cand==base {nc_eq}/{n_calls}, base repeat {nb_eq}/{n_calls}, P={info[0]} "
            f"flags {sum(info[1:])} -> {'PASS' if ok else 'FAIL'}")
        del ba, ca
    cases = ("none", "violate", "tileend", "groups2", "shared1003", "shared1027", "ragged_len", "shared")
    for b in (8, 16, 24, 32, 40, 64):
        m = cl.main_nb_sub_seq(b)
        for case in cases:
            set_case(inp, case, bmax, b)
            ba, ca = Arm(base, inp, b, m), Arm(cand, inp, b, m)
            ref = ba.run(flush)
            eq = sum(same(ca.run(flush), ref) for _ in range(a.table_calls))
            info = info_of(ref, ba.c)
            ok = eq == a.table_calls
            rep["tables"][f"B{b}_{case}"] = dict(cand_bitwise_base=eq, calls=a.table_calls, P=info[0],
                                                 flagged_chunks=[i for i, f in enumerate(info[1:]) if f], passed=ok)
            passed &= ok
            log(f"table B={b} {case}: P={info[0]} flagged {[i for i, f in enumerate(info[1:]) if f]} "
                f"cand==base {eq}/{a.table_calls} -> {'PASS' if ok else 'FAIL'}")
            del ba, ca
    seq = ["shared", "none", "shared1003", "tileend", "shared", "violate", "shared1027", "groups2", "ragged_len",
           "shared"]
    for b in (16, 32, 64):
        m = cl.main_nb_sub_seq(b)
        set_case(inp, seq[0], bmax, b)
        ba, ca = Arm(base, inp, b, m), Arm(cand, inp, b, m)
        steps, gok = [], True
        for case in seq * 2:
            set_case(inp, case, bmax, b)
            ok = True
            for _ in range(3):
                r = ba.run(flush)
                ok &= same(ca.run(flush), r)
            steps.append(dict(case=case, P=info_of(r, ba.c)[0], passed=ok))
            gok &= ok
        rep["graph_p_changes"][f"B{b}"] = dict(steps=steps, passed=gok)
        passed &= gok
        log(f"graph P changes B={b}: {'PASS' if gok else 'FAIL'} " + str([(s['case'], s['P']) for s in steps[:10]]))
        del ba, ca
    set_case(inp, "shared", bmax, bmax)
    ro = cl.readonly_hash(inp) == h0
    rep["readonly_inputs_unchanged"] = ro
    passed &= ro
    rep["regs"] = dict(base_prefix_regs=base.pregs, base_prefix_local=base.plmem, cand_prefix_regs=cand.pregs,
                       cand_prefix_local=cand.plmem)
    rep["compared_against"] = (f"base kernel_mha_prefix + kernel_mha compiled from {BASE_COMMIT}, same inputs: output "
                               "BF16, FP32 records + cascadeInfo (P and every chunk flag) bitwise")
    rep["passed"] = bool(passed)
    print(json.dumps(rep), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="bench")
    p.add_argument("--batches", default="8,16,24,32")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--table-calls", type=int, default=10)
    p.add_argument("--flush-mb", type=int, default=384)
    p.add_argument("--step0", type=int, default=1)
    a = p.parse_args()
    cl.init_cuda()
    {"bench": mode_bench, "check": mode_check}[a.mode](a)


if __name__ == "__main__":
    main()
