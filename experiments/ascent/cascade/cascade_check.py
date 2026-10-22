#!/usr/bin/env python3
# Correctness check for idea 20 (shared-prefix cascade decode). Untimed, same call sequence as the timed loop.
#  (1) repeated: per B in 8/16/32 (serving shape, 32768-token shared prefix, kernel_mha count 5/2/1), the CUDA graph
#      (kernel_mha_prefix + kernel_mha) is replayed rounds x (warmup + iters) times with NaN-poisoned output and an L2
#      flush before each replay: every replay bitwise equal to the first, semaphores back to 0 after every replay,
#      finite outputs; error vs the FP32 reference <= 1.5x main's kernel_mha (compiled from main) on the same inputs
#      and <= 1.25x main's + 1 BF16 ulp of the output magnitude (protocol's reordered class).
#  (2) ragged cases at B=16 (graph per case): no shared prefix (P=0) and two prefix groups -> bitwise equal to main;
#      shared prefix not a multiple of 256 tokens (1003 and 1027 shared pages), requests at different decode
#      positions, and a page table that shares the 256-token tile ends but not a page inside a tile (verification
#      must fall back to P=0 -> bitwise main).
#  (3) one captured graph, page table / sequence lengths rewritten in place between replays (P = 32768 -> 0 ->
#      32000 -> 32768 ...): every replay equals the eager result of its case (bitwise vs main where P=0, bitwise vs the
#      cascade's own first result otherwise).
#  (4) below the threshold: kernel_mha with cascadeInfo = nullptr at B=1/2/4/8/16/32 bitwise equal to main's kernel.
#  (5) interleaved configs on shared semaphores/scratch: each call equals the first call of its config.
#  (6) read-only inputs (pool, page table, q, seq lens, scale) unchanged.
import argparse
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cascade_lib as cl  # noqa: E402
import torch  # noqa: E402

log = cl.log
DEFAULT_LEN = 32768 + 2048 + 128


def build(bmax, seed=4321):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    nblocks = 1024 + bmax * cl.MAX_PAGES
    pool = (torch.randn(nblocks * 2, cl.NB_K, cl.TPP, cl.HD, generator=g, device="cuda") * 2.0).to(
        torch.float8_e4m3fn)
    scratch_raw = torch.zeros(96 << 20, dtype=torch.uint8, device="cuda")
    return dict(pool=pool, pages=torch.zeros(bmax, 1, 2, cl.MAX_PAGES, dtype=torch.int32, device="cuda"),
                q=torch.randn(bmax, cl.NB_Q, cl.HD, generator=g, device="cuda").to(torch.bfloat16),
                seqlen=torch.full((bmax, ), DEFAULT_LEN, dtype=torch.int32, device="cuda"),
                kvscale=torch.tensor([cl.KV_SCALE], dtype=torch.float32, device="cuda"),
                out=torch.zeros(bmax, cl.NB_Q, cl.HD, dtype=torch.bfloat16, device="cuda"),
                sem=torch.zeros(1024, dtype=torch.int32, device="cuda"), scratch_raw=scratch_raw,
                scratch_ptr=(scratch_raw.data_ptr() + 4095) // 4096 * 4096)


def set_case(inp, case, bmax):
    """Rewrites the page table and sequence lengths in place."""
    seqlens = [DEFAULT_LEN] * bmax
    shared, groups, violate = 1024, 1, False
    if case == "shared":
        pass
    elif case == "none":
        shared = 0
    elif case == "groups2":
        groups = 2
    elif case == "shared1003":
        shared = 1003
    elif case == "shared1027":
        shared = 1027
    elif case == "ragged_len":
        seqlens = [34817 + 37 * r for r in range(bmax)]
    elif case == "violate":
        violate = True
    else:
        raise ValueError(case)
    pages = torch.zeros(bmax, 1, 2, cl.MAX_PAGES, dtype=torch.int32)
    for r in range(bmax):
        nbp = math.ceil(seqlens[r] / cl.TPP)
        own = 1024 + r * cl.MAX_PAGES
        if groups == 2 and r % 2 == 1:
            common = [1024 + (bmax - 1) * cl.MAX_PAGES + j for j in range(shared)]
        else:
            common = list(range(shared))
        blk = common[:min(shared, nbp)] + [own + j for j in range(min(shared, nbp), nbp)]
        if violate and r == 3:
            blk[500] = own + cl.MAX_PAGES - 1  # inside tile 62 (pages 496-503); tile-end pages stay shared
        blk = torch.tensor(blk, dtype=torch.int32)
        pages[r, 0, 0, :nbp] = 2 * blk
        pages[r, 0, 1, :nbp] = 2 * blk + 1
    inp["pages"].copy_(pages.cuda())
    inp["seqlen"].copy_(torch.tensor(seqlens, dtype=torch.int32, device="cuda"))
    torch.cuda.synchronize()


def expected_p(case):
    return {"shared": 32768, "none": 0, "groups2": 0, "shared1003": 32000, "shared1027": 32768,
            "ragged_len": 32768, "violate": 0}[case]


def errs(o, ref):
    d = (o.float() - ref).abs()
    return float(d.max()), float(d.mean())


def tol_ok(ec, em, ref):
    ulp = float(2.0**(math.floor(math.log2(max(float(ref.abs().max()), 1e-30))) - 7))
    ok15 = ec[0] <= 1.5 * em[0] + 1e-12 and ec[1] <= 1.5 * em[1] + 1e-12
    ok125 = ec[0] <= 1.25 * em[0] + ulp and ec[1] <= 1.25 * em[1] + ulp
    return ok15, ok125, ulp


def run_calls(call, flush, n):
    outs, sems = [], []
    for _ in range(n):
        call.inp["out"].fill_(float("nan"))
        flush()
        call.graph.replay()
        outs.append(call.out().clone())
        sems.append((call.inp["sem"] == 0).all())
    torch.cuda.synchronize()
    return outs, [bool(s.item()) for s in sems]


def read_info(inp, c):
    info = torch.empty(1 + c, dtype=torch.int32, device="cuda")
    cl.ck(cl.cu.cuMemcpyDtoD(info.data_ptr(), inp["scratch_ptr"], 4 * (1 + c)), "dtod")
    return info.cpu().tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--flush-mb", type=int, default=384)
    a = p.parse_args()
    cl.init_cuda()
    cand = cl.Kernels(os.path.join(os.getcwd(), "cpp/kernels/xqa"))
    main_k = cl.Kernels(cl.extract_tree(cl.MAIN_COMMIT))
    assert cand.prefix is not None
    bmax = 32
    inp = build(bmax)
    flush = cl.xe.Flusher(a.flush_mb)
    rep = {"repeated": {}, "ragged": {}, "graph_p_changes": {}, "below_threshold": {}, "interleaved": {}}
    passed = True
    n_calls = a.rounds * (a.warmup + a.iters)

    # (1) repeated, serving shape
    set_case(inp, "shared", bmax)
    h0 = cl.readonly_hash(inp)
    for b in (8, 16, 32):
        m = cl.main_nb_sub_seq(b)
        cc = cl.Call(cand, inp, b, m, cascade=True).capture()
        outs, sems = run_calls(cc, flush, n_calls)
        stack = torch.stack(outs)
        rep_ok = bool((stack == stack[0:1]).all().item())
        fin = bool(torch.isfinite(stack.float()).all().item())
        info = read_info(inp, cc.c)
        mc = cl.Call(main_k, inp, b, m, cascade=False).capture()
        mouts, msems = run_calls(mc, flush, 3)
        ref = cl.reference(inp, b)
        per_call_max = [errs(o, ref)[0] for o in outs[::max(1, n_calls // 12)]]
        ec, em = errs(outs[0], ref), errs(mouts[0], ref)
        ok15, ok125, ulp = tol_ok(ec, em, ref)
        ok = rep_ok and all(sems) and fin and ok15 and ok125 and info[0] == 32768 and not any(info[1:])
        rep["repeated"][f"B{b}"] = dict(calls=n_calls, nb_sub_seq=m, prefix_chunks=cc.c, row_groups=cc.g,
                                        repeat_bitwise=rep_ok, semaphores_zero_every_call=all(sems), finite=fin,
                                        detected_prefix_tokens=info[0], chunk_flags_clear=not any(info[1:]),
                                        cand_max_abs=ec[0], cand_mean_abs=ec[1], main_max_abs=em[0],
                                        main_mean_abs=em[1], bf16_ulp_of_output=ulp, within_1p5x=ok15,
                                        within_1p25x_plus_ulp=ok125, sampled_per_call_max_abs=per_call_max,
                                        passed=ok)
        passed &= ok
        log(f"repeated B={b} m={m} C={cc.c}: {n_calls} calls repeat-bitwise {rep_ok} sem0 {all(sems)} finite {fin} "
            f"P={info[0]} | cand max {ec[0]:.3e} mean {ec[1]:.3e} | main max {em[0]:.3e} mean {em[1]:.3e} -> "
            f"{'PASS' if ok else 'FAIL'}")
        del stack, outs

    # (2) ragged cases at B=16
    b = 16
    m = cl.main_nb_sub_seq(b)
    for case in ("none", "groups2", "shared1003", "shared1027", "ragged_len", "violate"):
        set_case(inp, case, bmax)
        cc = cl.Call(cand, inp, b, m, cascade=True).capture()
        outs, sems = run_calls(cc, flush, 20)
        info = read_info(inp, cc.c)
        mc = cl.Call(main_k, inp, b, m, cascade=False).capture()
        mouts, _ = run_calls(mc, flush, 2)
        ref = cl.reference(inp, b)
        rep_ok = all(torch.equal(o, outs[0]) for o in outs)
        ec, em = errs(outs[0], ref), errs(mouts[0], ref)
        ok15, ok125, ulp = tol_ok(ec, em, ref)
        exp_p = expected_p(case)
        bad = any(info[1:])
        p_eff = 0 if bad else info[0]
        bitwise_main = bool(torch.equal(outs[0], mouts[0]))
        ok = rep_ok and all(sems) and p_eff == exp_p and (bitwise_main if exp_p == 0 else (ok15 and ok125))
        rep["ragged"][case] = dict(batch=b, detected_prefix_tokens=info[0], verification_flagged=bad,
                                   effective_prefix_tokens=p_eff, expected_prefix_tokens=exp_p,
                                   bitwise_equal_main=bitwise_main, repeat_bitwise=rep_ok,
                                   semaphores_zero=all(sems), cand_max_abs=ec[0], cand_mean_abs=ec[1],
                                   main_max_abs=em[0], main_mean_abs=em[1], within_1p5x=ok15,
                                   within_1p25x_plus_ulp=ok125, passed=ok)
        passed &= ok
        log(f"ragged {case}: P detected {info[0]} flagged {bad} effective {p_eff} (expect {exp_p}) bitwise-main "
            f"{bitwise_main} | cand max {ec[0]:.3e} mean {ec[1]:.3e} | main max {em[0]:.3e} -> "
            f"{'PASS' if ok else 'FAIL'}")

    # (3) one graph, P changes between replays
    b = 16
    m = cl.main_nb_sub_seq(b)
    seq = ["shared", "none", "shared1003", "shared", "violate", "shared1027", "none", "shared"]
    expect = {}
    for case in set(seq):
        set_case(inp, case, bmax)
        ec_ = cl.Call(cand, inp, b, m, cascade=True).capture()
        o, _ = run_calls(ec_, flush, 1)
        mc = cl.Call(main_k, inp, b, m, cascade=False).capture()
        mo, _ = run_calls(mc, flush, 1)
        expect[case] = (o[0], mo[0])
    set_case(inp, seq[0], bmax)
    g_call = cl.Call(cand, inp, b, m, cascade=True).capture()
    results = []
    gok = True
    for case in seq * 2:
        set_case(inp, case, bmax)
        o, sems = run_calls(g_call, flush, 3)
        same_case = all(torch.equal(x, expect[case][0]) for x in o)
        eq_main = all(torch.equal(x, expect[case][1]) for x in o)
        need_main = expected_p(case) == 0
        ok = same_case and all(sems) and (eq_main if need_main else True)
        results.append(dict(case=case, equals_fresh_capture=same_case, bitwise_main=eq_main, passed=ok))
        gok &= ok
    rep["graph_p_changes"] = dict(batch=b, sequence=seq * 2, replays_per_step=3, steps=results, passed=gok)
    passed &= gok
    log(f"graph with P changing between replays: {'PASS' if gok else 'FAIL'} "
        f"({sum(r['passed'] for r in results)}/{len(results)} steps)")

    # (4) below threshold / disabled: cascadeInfo = nullptr -> bitwise main
    set_case(inp, "shared", bmax)
    for b in (1, 2, 4, 8, 16, 32):
        m = 47 if b == 1 else cl.main_nb_sub_seq(b)
        cc = cl.Call(cand, inp, b, m, cascade=False).capture()
        mc = cl.Call(main_k, inp, b, m, cascade=False).capture()
        o, sems = run_calls(cc, flush, 5)
        mo, _ = run_calls(mc, flush, 5)
        ok = all(torch.equal(x, y) for x, y in zip(o, mo)) and all(sems)
        rep["below_threshold"][f"B{b}_m{m}"] = dict(bitwise_equal_main=ok, passed=ok)
        passed &= ok
        log(f"no-cascade path B={b} m={m}: bitwise main {ok}")

    # (5) interleaved
    cfgs = [(8, True), (16, True), (32, True), (4, False), (16, False), (1, False)]
    calls = {}
    for bb, casc in cfgs:
        mm = 47 if bb == 1 else cl.main_nb_sub_seq(bb)
        calls[(bb, casc)] = cl.Call(cand, inp, bb, mm, cascade=casc).capture()
    pairs = [(i, j) for i in range(len(cfgs)) for j in range(len(cfgs)) if i != j]
    random.Random(20261022).shuffle(pairs)
    order = [cfgs[k] for pr in pairs for k in pr]
    firsts, iok = {}, True
    for cfg in order:
        o, sems = run_calls(calls[cfg], flush, 1)
        if cfg not in firsts:
            firsts[cfg] = o[0]
        iok &= torch.equal(o[0], firsts[cfg]) and all(sems)
    rep["interleaved"] = dict(calls=len(order), configs=[f"B{bb}{'_cascade' if c else ''}" for bb, c in cfgs],
                              each_call_equals_first_of_config=iok, passed=iok)
    passed &= iok
    log(f"interleaved {len(order)} calls: {'PASS' if iok else 'FAIL'}")

    # (6) read-only inputs
    set_case(inp, "shared", bmax)
    ro = cl.readonly_hash(inp) == h0
    rep["readonly_inputs_unchanged"] = ro
    passed &= ro
    rep["passed"] = bool(passed)
    rep["compared_against"] = (f"FP32 reference over the stored e4m3 KV; main's kernel_mha compiled from {cl.MAIN_COMMIT} "
                               "on the same inputs (same sub-sequence counts)")
    rep["candidate_source_sha256"] = cand.src_sha
    rep["main_source_sha256"] = main_k.src_sha
    print(json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
