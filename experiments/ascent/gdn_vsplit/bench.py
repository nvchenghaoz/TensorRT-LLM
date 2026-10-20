# SM120 GDN chunk prefill V-split x2 bench / probe / check (branch 112, idea 122).
# Synthetic tensors with the real shapes (16 qk heads, 48 v heads, D=128, bf16,
# fp32 state), recipe of fact 995 (seed 777, alpha shift 6). No model load.
#
#   python bench.py probe   [T=2048]        STEP 1: stock vs V-split, probe modes 0..3
#   python bench.py check                   bitwise SHA check + 10+50 repeated calls
#   python bench.py time    [shapes]        kernel pair stock vs V-split (+A/A arm)
#
# Cold L2 with a READ flush (sum over a 256 MiB buffer), never zero_() (fact 995).
import hashlib
import json
import os
import subprocess
import sys
import time

import torch

T0 = time.time()


def log(*a):
    print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


def tree_info():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        tree = subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"], capture_output=True, text=True, cwd=here
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, cwd=here
        ).stdout
        return tree, len([ln for ln in dirty.splitlines() if ln.strip()])
    except Exception as exc:  # pragma: no cover
        return "unknown(%s)" % exc, -1


H_QK, H_V, D = 16, 48, 128


def make_inputs(seq_lens, with_init=True, seed=777, alpha_shift=6.0):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    T = sum(seq_lens)
    N = len(seq_lens)
    F = torch.nn.functional
    q = F.normalize(
        torch.randn(T, H_QK, D, generator=g, device="cuda"), dim=-1
    ).to(torch.bfloat16)
    k = F.normalize(
        torch.randn(T, H_QK, D, generator=g, device="cuda"), dim=-1
    ).to(torch.bfloat16)
    v = torch.randn(T, H_V, D, generator=g, device="cuda").to(torch.bfloat16)
    alpha = torch.exp(
        -F.softplus(torch.randn(T, H_V, generator=g, device="cuda") - alpha_shift)
    ).contiguous()
    beta = torch.sigmoid(torch.randn(T, H_V, generator=g, device="cuda")).contiguous()
    init = (
        (torch.randn(N, H_V, D, D, generator=g, device="cuda") * 0.05).contiguous()
        if with_init
        else None
    )
    cu = [0]
    for s in seq_lens:
        cu.append(cu[-1] + s)
    cu_seqlens = torch.tensor(cu, dtype=torch.int64, device="cuda")
    return dict(q=q, k=k, v=v, alpha=alpha, beta=beta, init=init, cu=cu_seqlens, T=T, N=N)


def run(inp, o, state, vsplit, probe=0, num_long=None):
    from flashinfer.gdn_kernels.delta_rule_dsl.delta_rule_sm120 import (
        delta_rule_prefill_dsl,
    )

    delta_rule_prefill_dsl(
        o,
        state,
        inp["q"],
        inp["k"],
        inp["v"],
        inp["init"],
        inp["alpha"],
        inp["beta"],
        inp["cu"],
        D ** -0.5,
        num_long_seqs=num_long,
        vsplit=vsplit,
        _probe_mode=probe,
    )


def n_long_of(seq_lens):
    return sum(1 for s in seq_lens if s > 1)


def alloc_out(inp):
    o = torch.empty(inp["T"], H_V, D, dtype=torch.bfloat16, device="cuda")
    st = torch.empty(inp["N"], H_V, D, D, dtype=torch.float32, device="cuda")
    return o, st


def sha(t):
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    return hashlib.sha256(t.contiguous().cpu().numpy().tobytes()).hexdigest()[:8]


_FLUSH = None


def flush_l2():
    global _FLUSH
    if _FLUSH is None:
        _FLUSH = torch.ones(64 * 1024 * 1024, dtype=torch.float32, device="cuda")
    torch.sum(_FLUSH)


def time_arms(arms, warmup=10, iters=60, sleep_cycles=2_000_000):
    """arms: list of (name, fn). Arms rotate every iteration (ABAB...).
    Returns name -> list of per-call us."""
    for _ in range(warmup):
        for _, fn in arms:
            flush_l2()
            fn()
    torch.cuda.synchronize()
    res = {name: [] for name, _ in arms}
    for it in range(iters):
        order = arms if it % 2 == 0 else list(reversed(arms))
        for name, fn in order:
            torch.cuda._sleep(sleep_cycles)
            flush_l2()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            e.synchronize()
            res[name].append(s.elapsed_time(e) * 1000.0)
    return res


def stats(xs):
    ys = sorted(xs)
    n = len(ys)

    def pct(p):
        return ys[min(n - 1, max(0, int(round(p * (n - 1)))))]

    return dict(median=pct(0.5), p10=pct(0.1), p90=pct(0.9), n=n)


def cmd_probe(argv):
    T = int(argv[0]) if argv else 2048
    inp = make_inputs([T])
    outs = {}
    arms = []
    for vs in (False, True):
        for mode in (0, 1, 2, 3):
            name = ("vsplit" if vs else "stock") + "_m%d" % mode
            o, st = alloc_out(inp)
            outs[name] = (o, st)
            t = time.time()
            run(inp, o, st, vs, mode)
            torch.cuda.synchronize()
            log("compiled+ran", name, "%.1fs" % (time.time() - t))
            arms.append((name, (lambda o=o, st=st, vs=vs, mode=mode: run(inp, o, st, vs, mode))))
    # quick bitwise check of the full kernels
    so, ss = outs["stock_m0"]
    vo, vs_ = outs["vsplit_m0"]
    run(inp, so, ss, False, 0)
    run(inp, vo, vs_, True, 0)
    torch.cuda.synchronize()
    eq_o = torch.equal(so.view(torch.int16), vo.view(torch.int16))
    eq_s = torch.equal(ss.view(torch.int32), vs_.view(torch.int32))
    log("bitwise o", eq_o, "state", eq_s, "sha o", sha(so), sha(vo), "state", sha(ss), sha(vs_))
    if not eq_o:
        d = (so.float() - vo.float()).abs()
        log("o max abs diff", d.max().item(), "n diff", int((d > 0).sum().item()))
    if not eq_s:
        d = (ss - vs_).abs()
        log("state max abs diff", d.max().item(), "n diff", int((d > 0).sum().item()))
    res = time_arms(arms, warmup=10, iters=60)
    summary = {}
    for name, xs in res.items():
        summary[name] = stats(xs)
        log("%-10s median %.2f us  p10 %.2f  p90 %.2f" % (
            name, summary[name]["median"], summary[name]["p10"], summary[name]["p90"]))
    for arm in ("stock", "vsplit"):
        m = lambda k: summary["%s_m%d" % (arm, k)]["median"]
        log("%s: full %.1f  Pk %.1f  Pv %.1f  skel %.1f  V-dep(Pv-skel) %.1f  KK-dep(Pk-skel) %.1f" % (
            arm, m(0), m(1), m(2), m(3), m(2) - m(3), m(1) - m(3)))
    print(json.dumps(dict(T=T, bitwise_o=eq_o, bitwise_state=eq_s, summary=summary)), flush=True)


def run_killswitch(inp, o, st, num_long=None):
    """Stock reference through the production auto gate with the kill switch."""
    old = os.environ.get("FLASHINFER_GDN_SM120_VSPLIT")
    os.environ["FLASHINFER_GDN_SM120_VSPLIT"] = "0"
    try:
        run(inp, o, st, None, 0, num_long=num_long)
    finally:
        if old is None:
            del os.environ["FLASHINFER_GDN_SM120_VSPLIT"]
        else:
            os.environ["FLASHINFER_GDN_SM120_VSPLIT"] = old


def check_case(seq_lens, with_init, warm=10, iters=50):
    inp = make_inputs(seq_lens, with_init=with_init)
    ref_o, ref_s = alloc_out(inp)
    # stock reference: kill switch FLASHINFER_GDN_SM120_VSPLIT=0, same host hint
    run_killswitch(inp, ref_o, ref_s, num_long=n_long_of(seq_lens))
    torch.cuda.synchronize()
    # and the explicit stock kernel (vsplit=False) must agree with the kill switch
    x_o, x_s = alloc_out(inp)
    run(inp, x_o, x_s, False, 0)
    torch.cuda.synchronize()
    assert torch.equal(x_o.view(torch.int16), ref_o.view(torch.int16))
    assert torch.equal(x_s.view(torch.int32), ref_s.view(torch.int32))
    ref = (sha(ref_o), sha(ref_s))
    o, st = alloc_out(inp)
    first = None
    bad = 0
    init_pristine = inp["init"].clone() if inp["init"] is not None else None
    for it in range(warm + iters):
        o.fill_(float("nan"))
        st.fill_(float("nan"))
        if init_pristine is not None:
            inp["init"].copy_(init_pristine)
        # auto gate with the production host hint (number of >1-token sequences)
        run(inp, o, st, None, 0, num_long=n_long_of(seq_lens))
        torch.cuda.synchronize()
        eq = torch.equal(o.view(torch.int16), ref_o.view(torch.int16)) and torch.equal(
            st.view(torch.int32), ref_s.view(torch.int32)
        )
        if first is None:
            first = (o.clone(), st.clone())
        else:
            eq = eq and torch.equal(o.view(torch.int16), first[0].view(torch.int16))
        if not eq:
            bad += 1
    if init_pristine is not None:
        assert torch.equal(inp["init"], init_pristine)
    return dict(
        shape=list(seq_lens),
        init=with_init,
        ref_sha_o=ref[0],
        ref_sha_state=ref[1],
        cand_sha_o=sha(o),
        cand_sha_state=sha(st),
        calls=warm + iters,
        mismatching_calls=bad,
    )


def wrapper_check():
    """TRT-LLM adapter path (fla/flashinfer_chunk.py): in-place indexed state
    update on a state pool, mixed [2048, 1] step, hint vs kill switch."""
    from tensorrt_llm._torch.modules.fla.flashinfer_chunk import (
        chunk_gated_delta_rule as cg,
    )

    seq_lens = [2048, 1]
    inp = make_inputs(seq_lens)
    T = inp["T"]
    g = torch.Generator(device="cuda")
    g.manual_seed(1234)
    pool0 = (torch.randn(4, H_V, D, D, generator=g, device="cuda") * 0.05).contiguous()
    idx = torch.tensor([3, 1], dtype=torch.int32, device="cuda")
    q4 = inp["q"].unsqueeze(0)
    k4 = inp["k"].unsqueeze(0)
    v4 = inp["v"].unsqueeze(0)
    g_log = torch.log(inp["alpha"]).unsqueeze(0).contiguous()
    beta = inp["beta"].unsqueeze(0).to(torch.bfloat16)

    def call(pool, hint):
        out, _ = cg(
            q4,
            k4,
            v4,
            g_log,
            beta,
            scale=D ** -0.5,
            initial_state=pool,
            initial_state_indices=idx,
            inplace_indexed_state_update=True,
            output_final_state=False,
            cu_seqlens=inp["cu"],
            head_first=False,
            use_qk_l2norm_in_kernel=False,
            num_long_seqs=hint,
        )
        return out

    os.environ["FLASHINFER_GDN_SM120_VSPLIT"] = "0"
    pool_ref = pool0.clone()
    out_ref = call(pool_ref, 1).clone()
    del os.environ["FLASHINFER_GDN_SM120_VSPLIT"]
    pool_c = pool0.clone()
    out_c = call(pool_c, 1).clone()
    pool_n = pool0.clone()
    out_n = call(pool_n, None).clone()  # no hint: 2 seqs counted -> stock path
    torch.cuda.synchronize()
    r = dict(
        wrapper_mixed_2048_1=True,
        out_equal=torch.equal(out_ref.view(torch.int16), out_c.view(torch.int16)),
        pool_equal=torch.equal(pool_ref.view(torch.int32), pool_c.view(torch.int32)),
        nohint_out_equal=torch.equal(out_ref.view(torch.int16), out_n.view(torch.int16)),
        nohint_pool_equal=torch.equal(pool_ref.view(torch.int32), pool_n.view(torch.int32)),
        untouched_slots_equal=torch.equal(pool_c[0], pool0[0]) and torch.equal(pool_c[2], pool0[2]),
        sha_out=sha(out_c),
        sha_pool=sha(pool_c),
    )
    r["passed"] = all(
        r[k] for k in ("out_equal", "pool_equal", "nohint_out_equal", "nohint_pool_equal", "untouched_slots_equal")
    )
    return r


def cmd_check(argv):
    from flashinfer.gdn_kernels.delta_rule_dsl.delta_rule_sm120 import _vsplit_auto

    cases = []
    for T in (64, 100, 1000, 2048, 3000):
        for wi in (True, False):
            cases.append(([T], wi))
    cases.append(([2048, 1], True))
    cases.append(([2048, 1], False))
    cases.append(([2048, 1, 1, 1], True))
    cases.append(([2048, 2048], True))
    cases.append(([2048, 2048, 2048, 1], True))
    out = []
    ok = True
    for seq_lens, wi in cases:
        gated = _vsplit_auto(
            len(seq_lens), n_long_of(seq_lens), H_V, D, False, torch.device("cuda")
        )
        r = check_case(seq_lens, wi)
        r["vsplit_gated"] = gated
        r["passed"] = r["mismatching_calls"] == 0 and r["ref_sha_o"] == r["cand_sha_o"] and (
            r["ref_sha_state"] == r["cand_sha_state"]
        )
        ok = ok and r["passed"]
        log(json.dumps(r))
        out.append(r)
    w = wrapper_check()
    log(json.dumps(w))
    ok = ok and w["passed"]
    print(json.dumps(dict(
        passed=ok,
        compared=(
            "V-split (production auto gate + host hint) vs stock kernel via kill switch "
            "FLASHINFER_GDN_SM120_VSPLIT=0 (also == vsplit=False), bitwise o + final state, "
            "10 warmup + 50 calls with NaN-poisoned outputs and init restored, each call == call 1 "
            "== stock; plus TRT-LLM flashinfer_chunk adapter in-place pool update [2048,1]"
        ),
        cases=out,
        wrapper=w,
    )), flush=True)


def cmd_sweep(argv):
    """V-split stage configs at 1x2048 (sizing only; production uses the defaults)."""
    from flashinfer.gdn_kernels.delta_rule_dsl.delta_rule_sm120 import (
        delta_rule_prefill_dsl,
    )

    T = int(argv[0]) if argv else 2048
    inp = make_inputs([T])
    configs = [
        None,
        dict(kk_stage=1, qk_stage=1),
        dict(kk_stage=1, qk_stage=1, q_stage=2),
        dict(kk_stage=2, qk_stage=1, v_stage=2),
        dict(kk_stage=1, qk_stage=2, v_stage=2),
        dict(kk_stage=1, qk_stage=1, v_stage=2),
        dict(kk_stage=1, qk_stage=1, k_stage=3),
    ]
    ref_o, ref_s = alloc_out(inp)
    run(inp, ref_o, ref_s, False, 0)
    arms = [("stock", lambda: run(inp, ref_o, ref_s, False, 0))]
    for cfg in configs:
        name = "vs_" + ("default" if cfg is None else "_".join("%s%d" % (k.replace("_stage", ""), v) for k, v in cfg.items()))
        o, st = alloc_out(inp)

        def fn(o=o, st=st, cfg=cfg):
            delta_rule_prefill_dsl(
                o, st, inp["q"], inp["k"], inp["v"], inp["init"], inp["alpha"],
                inp["beta"], inp["cu"], D ** -0.5, vsplit=True, _vsplit_stages=cfg,
            )

        try:
            fn()
            torch.cuda.synchronize()
        except Exception as exc:
            log("config", name, "failed:", repr(exc)[:300])
            continue
        eq = torch.equal(o.view(torch.int16), ref_o.view(torch.int16)) and torch.equal(
            st.view(torch.int32), ref_s.view(torch.int32))
        log("config", name, "bitwise", eq)
        arms.append((name, fn))
    res = time_arms(arms, warmup=10, iters=40)
    out = {n: stats(x) for n, x in res.items()}
    for n, s in out.items():
        log("%-28s median %.2f  p10 %.2f  p90 %.2f" % (n, s["median"], s["p10"], s["p90"]))
    print(json.dumps(out), flush=True)


def cmd_time(argv):
    shapes = []
    for a in argv or ["2048", "512", "1024", "3072", "2048,1", "2048,1,1,1", "2048,2048", "2048,2048,2048", "2048,2048,2048,1", "2048,2048,2048,2048"]:
        shapes.append([int(x) for x in a.split(",")])
    result = {}
    for seq_lens in shapes:
        inp = make_inputs(seq_lens)
        o1, s1 = alloc_out(inp)
        o2, s2 = alloc_out(inp)
        o3, s3 = alloc_out(inp)
        nl = n_long_of(seq_lens)
        arms = [
            ("stock", lambda: run(inp, o1, s1, False, 0)),
            ("stock_aa", lambda: run(inp, o3, s3, False, 0)),
            ("cand", lambda: run(inp, o2, s2, None, 0, num_long=nl)),
        ]
        res = time_arms(arms, warmup=10, iters=60)
        key = "x".join(str(s) for s in seq_lens)
        result[key] = {n: stats(x) for n, x in res.items()}
        st = result[key]
        log("%-22s stock %.2f (p10 %.2f p90 %.2f)  A/A %.2f  cand %.2f (p10 %.2f p90 %.2f)  ratio %.3f" % (
            key, st["stock"]["median"], st["stock"]["p10"], st["stock"]["p90"],
            st["stock_aa"]["median"], st["cand"]["median"], st["cand"]["p10"], st["cand"]["p90"],
            st["cand"]["median"] / st["stock"]["median"]))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    tree, dirty = tree_info()
    log("TREE_COMMIT(tree)=%s DIRTY=%d" % (tree, dirty))
    import flashinfer

    log("flashinfer", flashinfer.__version__, flashinfer.__file__)
    log("gpu", torch.cuda.get_device_name(), "sms", torch.cuda.get_device_properties(0).multi_processor_count)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "probe"
    dict(probe=cmd_probe, check=cmd_check, time=cmd_time, sweep=cmd_sweep)[cmd](sys.argv[2:])
