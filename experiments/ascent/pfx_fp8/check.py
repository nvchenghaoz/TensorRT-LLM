#!/usr/bin/env python3
# Untimed correctness check for idea 22 (kernel_mha_prefix in e4m3 MMA math, output class precision). Candidate =
# this tree's mha.cu; main = mha.cu of the branch base commit (git archive), compiled the same way (NVRTC, JIT macros).
#  (1) repeated, serving shape (32768 shared + 2176 unique tokens), B = 64 and 128 (+ B = 64 with q x4, peakier
#      softmax): the CUDA graph (prefix + kernel_mha) replayed rounds x (warmup + iters) times like the timed loop,
#      NaN-poisoned output + 384 MiB L2 flush before each replay: every replay bitwise equal to the first, semaphores 0
#      after every replay, finite, detected P = 32768 with no chunk flagged. Error vs the FP32 reference (the stored
#      e4m3 KV dequantized, FP32 softmax, rounded once to BF16) reported next to main's cascade error; precision gate:
#      rel. L2 error <= 0.05 and max abs error <= 0.05 (sanity bounds, accuracy decides).
#  (2) ragged cases at B=16: no shared prefix (P=0), two prefix groups, a tile with a non-shared page (verification ->
#      P=0) must be bitwise equal to main; 1003/1027 shared pages and ragged lengths: P detected as expected,
#      deterministic, within the precision bounds.
#  (3) one graph, page table / lengths rewritten between replays (P 32768 -> 0 -> 32000 -> ...): each replay equals a
#      fresh capture of its case, bitwise main where P = 0.
#  (4) cascade off (cascadeInfo = nullptr) at B = 1..32: bitwise equal to main's kernel_mha.
#  (5) interleaved configs on shared semaphores/scratch: each call equals the first call of its config.
#  (6) read-only inputs unchanged.
import argparse
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cascade"))
import cascade_check as cc  # noqa: E402
import cascade_lib as cl  # noqa: E402
import torch  # noqa: E402

log = cl.log
BASE_COMMIT = "0674e6a60795a3ad9a1324cf64948749b44c721a"
REL_L2_MAX = 0.05
MAX_ABS_MAX = 0.05


def err_stats(o, ref):
    d = (o.float() - ref).abs()
    return dict(max_abs=float(d.max()), mean_abs=float(d.mean()),
                rel_l2=float((o.float() - ref).norm() / ref.norm().clamp_min(1e-30)))


def prec_ok(e):
    return e["rel_l2"] <= REL_L2_MAX and e["max_abs"] <= MAX_ABS_MAX


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--rounds", type=int, default=3)
    a = p.parse_args()
    cl.init_cuda()
    cand = cl.Kernels(os.path.join(os.getcwd(), "cpp/kernels/xqa"))
    main_k = cl.Kernels(cl.extract_tree(BASE_COMMIT))
    assert cand.prefix is not None and main_k.prefix is not None
    log(f"candidate prefix regs {cand.pregs} local {cand.plmem}; main prefix regs {main_k.pregs}")
    flush = cl.xe.Flusher(384)
    rep = {"repeated": {}, "ragged": {}, "graph_p_changes": {}, "below_threshold": {}, "interleaved": {}}
    passed = True
    n_calls = a.rounds * (a.warmup + a.iters)

    # (1) repeated, serving shape
    big = cl.build_inputs(128)
    h_big = cl.readonly_hash(big)
    q0 = big["q"].clone()
    for b, qs in ((64, 1.0), (128, 1.0), (64, 4.0)):
        big["q"].copy_((q0.float() * qs).to(torch.bfloat16))
        m = cl.main_nb_sub_seq(b)
        call = cl.Call(cand, big, b, m, cascade=True).capture()
        outs, sems = cc.run_calls(call, flush, n_calls)
        first = outs[0]
        rep_ok = all(torch.equal(o, first) for o in outs)
        fin = all(bool(torch.isfinite(o.float()).all()) for o in outs[::30]) and bool(torch.isfinite(first.float()).all())
        info = cc.read_info(big, call.c)
        mcall = cl.Call(main_k, big, b, m, cascade=True).capture()
        mouts, msems = cc.run_calls(mcall, flush, 3)
        ref = cl.reference(big, b)
        ec, em = err_stats(first, ref), err_stats(mouts[0], ref)
        per_call = [err_stats(o, ref)["max_abs"] for o in outs[::max(1, n_calls // 12)]]
        vs_main = float((first.float() - mouts[0].float()).abs().max())
        ok = rep_ok and all(sems) and fin and info[0] == 32768 and not any(info[1:]) and prec_ok(ec)
        key = f"B{b}_q{qs:g}"
        rep["repeated"][key] = dict(calls=n_calls, nb_sub_seq=m, prefix_chunks=call.c, row_groups=call.g,
                                    repeat_bitwise=rep_ok, semaphores_zero_every_call=all(sems), finite=fin,
                                    detected_prefix_tokens=info[0], chunk_flags_clear=not any(info[1:]),
                                    cand_err=ec, main_err=em,
                                    mean_abs_ratio_vs_main=ec["mean_abs"] / max(em["mean_abs"], 1e-30),
                                    max_abs_vs_main=vs_main, sampled_per_call_max_abs=per_call, passed=ok)
        passed &= ok
        log(f"repeated {key} C={call.c}: {n_calls} replays bitwise-repeat {rep_ok} sem0 {all(sems)} finite {fin} "
            f"P={info[0]} | cand {ec} | main {em} -> {'PASS' if ok else 'FAIL'}")
        del outs
    big["q"].copy_(q0)
    ro_big = cl.readonly_hash(big) == h_big
    del big
    torch.cuda.empty_cache()

    # (2) ragged cases at B=16
    bmax = 32
    inp = cc.build(bmax)
    cc.set_case(inp, "shared", bmax)
    h0 = cl.readonly_hash(inp)
    b = 16
    m = cl.main_nb_sub_seq(b)
    for case in ("none", "groups2", "violate", "shared1003", "shared1027", "ragged_len"):
        cc.set_case(inp, case, bmax)
        call = cl.Call(cand, inp, b, m, cascade=True).capture()
        outs, sems = cc.run_calls(call, flush, 20)
        info = cc.read_info(inp, call.c)
        mcall = cl.Call(main_k, inp, b, m, cascade=True).capture()
        mouts, _ = cc.run_calls(mcall, flush, 2)
        ref = cl.reference(inp, b)
        rep_ok = all(torch.equal(o, outs[0]) for o in outs)
        ec, em = err_stats(outs[0], ref), err_stats(mouts[0], ref)
        exp_p = cc.expected_p(case)
        bad = any(info[1:])
        p_eff = 0 if bad else info[0]
        bitwise_main = bool(torch.equal(outs[0], mouts[0]))
        ok = rep_ok and all(sems) and p_eff == exp_p and (bitwise_main if exp_p == 0 else prec_ok(ec))
        rep["ragged"][case] = dict(batch=b, detected_prefix_tokens=info[0], verification_flagged=bad,
                                   effective_prefix_tokens=p_eff, expected_prefix_tokens=exp_p,
                                   bitwise_equal_main=bitwise_main, repeat_bitwise=rep_ok, semaphores_zero=all(sems),
                                   cand_err=ec, main_err=em, passed=ok)
        passed &= ok
        log(f"ragged {case}: P {info[0]} flagged {bad} eff {p_eff} (expect {exp_p}) bitwise-main {bitwise_main} | "
            f"cand {ec} | main {em} -> {'PASS' if ok else 'FAIL'}")

    # (3) one graph, P changes between replays
    seq = ["shared", "none", "shared1003", "shared", "violate", "shared1027", "none", "shared"]
    expect = {}
    for case in set(seq):
        cc.set_case(inp, case, bmax)
        c1 = cl.Call(cand, inp, b, m, cascade=True).capture()
        o, _ = cc.run_calls(c1, flush, 1)
        mc = cl.Call(main_k, inp, b, m, cascade=True).capture()
        mo, _ = cc.run_calls(mc, flush, 1)
        expect[case] = (o[0], mo[0])
    cc.set_case(inp, seq[0], bmax)
    g_call = cl.Call(cand, inp, b, m, cascade=True).capture()
    results, gok = [], True
    for case in seq * 2:
        cc.set_case(inp, case, bmax)
        o, sems = cc.run_calls(g_call, flush, 3)
        same = all(torch.equal(x, expect[case][0]) for x in o)
        eq_main = all(torch.equal(x, expect[case][1]) for x in o)
        ok = same and all(sems) and (eq_main if cc.expected_p(case) == 0 else True)
        results.append(dict(case=case, equals_fresh_capture=same, bitwise_main=eq_main, passed=ok))
        gok &= ok
    rep["graph_p_changes"] = dict(batch=b, steps=results, passed=gok)
    passed &= gok
    log(f"graph with P changing between replays: {'PASS' if gok else 'FAIL'} "
        f"({sum(r['passed'] for r in results)}/{len(results)})")

    # (4) cascade off -> bitwise main
    cc.set_case(inp, "shared", bmax)
    for bb in (1, 2, 4, 8, 16, 32):
        mm = 47 if bb == 1 else cl.main_nb_sub_seq(bb)
        c1 = cl.Call(cand, inp, bb, mm, cascade=False).capture()
        mc = cl.Call(main_k, inp, bb, mm, cascade=False).capture()
        o, sems = cc.run_calls(c1, flush, 5)
        mo, _ = cc.run_calls(mc, flush, 5)
        ok = all(torch.equal(x, y) for x, y in zip(o, mo)) and all(sems)
        rep["below_threshold"][f"B{bb}_m{mm}"] = dict(bitwise_equal_main=ok, passed=ok)
        passed &= ok
        log(f"no-cascade B={bb} m={mm}: bitwise main {ok}")

    # (5) interleaved
    cfgs = [(8, True), (16, True), (32, True), (4, False), (16, False), (1, False)]
    calls = {}
    for bb, casc in cfgs:
        mm = 47 if bb == 1 else cl.main_nb_sub_seq(bb)
        calls[(bb, casc)] = cl.Call(cand, inp, bb, mm, cascade=casc).capture()
    pairs = [(i, j) for i in range(len(cfgs)) for j in range(len(cfgs)) if i != j]
    random.Random(20261023).shuffle(pairs)
    order = [cfgs[kk] for pr in pairs for kk in pr]
    firsts, iok = {}, True
    for cfg in order:
        o, sems = cc.run_calls(calls[cfg], flush, 1)
        firsts.setdefault(cfg, o[0])
        iok &= torch.equal(o[0], firsts[cfg]) and all(sems)
    rep["interleaved"] = dict(calls=len(order), passed=iok)
    passed &= iok
    log(f"interleaved {len(order)} calls: {'PASS' if iok else 'FAIL'}")

    # (6) read-only inputs
    cc.set_case(inp, "shared", bmax)
    ro = cl.readonly_hash(inp) == h0 and ro_big
    rep["readonly_inputs_unchanged"] = ro
    passed &= ro
    rep["passed"] = bool(passed)
    rep["output_class"] = "precision"
    rep["compared_against"] = (f"FP32 reference over the stored e4m3 KV (protected-reference recipe of cascade_lib); "
                               f"main = mha.cu of base commit {BASE_COMMIT} compiled identically, same inputs")
    rep["candidate_source_sha256"] = cand.src_sha
    rep["main_source_sha256"] = main_k.src_sha
    print(json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
