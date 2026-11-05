"""Correctness of the SM120 CSA2 attention-input chain (idea 232, branch 127), all untimed.

A = trtllm::csa2_mxfp8_low_m_gemm wq_a||wkv with debug_flags bit 1 (trigger dependents at start);
B = trtllm::csa2_qnorm_low_m_gemm (q_norm + MXFP8 quant + wq_b(||index_wq_b) in one launch).

Sections:
  N  op-level q_norm/quant on 8448 strided rows (row stride 1792; zeros, a 30x group, bf16 subnormals, 3e4, 1e-20):
     qr bitwise vs FlashInfer rmsnorm_cute (trtllm::flashinfer_rmsnorm), quant bytes + SF vs mxfp8_quantize(qr).
  O  every built B config at M=1..8, both routes (n_split 0 / 8192): q bitwise vs csa2_mxfp8_low_m_gemm on the
     rmsnorm_cute qr; q / index max abs+rel vs FP64 beside b12x's own error; A trigger-at-start bitwise vs trigger 0.
  B  live-overlap PDL: a producer that triggers at once, sleeps ~50 us, then writes NaN-poisoned rows; B must match,
     the skip-wait control (debug_flags bit 0) must fail; eager and inside a CUDA graph.
  C  W-AGG: the A(start)->B chain on 4 weight copies x 8 beside mul256 / stream-copy / cuBLAS aggressors on a second
     stream, outputs NaN-poisoned before every replay, >= 60 replays per case (90 at M=1), each bitwise vs isolated;
     a b12x control (mxfp8_quantize + flashinfer_mm_mxfp8 wq_b) in the same job must fault somewhere.
  D  module route: DeepseekV41Attention._project_inputs (probe with real Linears) at M=1/2/4/8, index and SWA layers:
     eager outputs vs the reference composition (bitwise), route kernels are the chain; CUDA graph captured, then a
     >= 8-step decode window with fresh inputs, each replay bitwise vs eager; tolerance vs the separate-Linear path.
  E  repeated execution matching chain_pair.py (5 warmup + 30 iterations, fresh input per iteration): every
     iteration compared; weights unchanged.
  K  kill switches: TRTLLM_CSA2_QB_FUSED=0, TRTLLM_LOW_M_MXFP8=0 and M=9 run main's code (bitwise vs main route).
argv: [checkpoint]
"""

import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "csa2_lowm"))
import csa2_shared_proj_bench as pb  # noqa: E402
import lowm_bench as lb  # noqa: E402

A_N, HIDDEN, Q_RANK, QB_N, IDX_N = 1792, 5120, 1280, 8192, 4096
EPS = 1e-6
results = {}
failures = []


def log(*a):
    print(*a, flush=True)


def fail(msg):
    failures.append(msg)
    log("FAIL:", msg)


def lg():
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import low_m_gemm

    return low_m_gemm


def rms(x, w, eps=EPS):
    return torch.ops.trtllm.flashinfer_rmsnorm(x, w, eps)


def qnorm(x, w, sw, nw, cfg, n_split=0, flags=0, xq=None, xsf=None):
    return torch.ops.trtllm.csa2_qnorm_low_m_gemm(x, w, sw, nw, EPS, cfg[0], cfg[1], cfg[2], n_split, True,
                                                  debug_flags=flags, xq_out=xq, xsf_out=xsf)


def lowm(x, w, sw, cfg, n_split=0, flags=0):
    return torch.ops.trtllm.csa2_mxfp8_low_m_gemm(x, w, sw, cfg[0], cfg[1], cfg[2], n_split, True,
                                                debug_flags=flags)


QCFGS = [(4, 1, 1), (4, 1, 2), (4, 1, 4), (8, 1, 2), (8, 1, 4)]


def fits(cfg, m, k=Q_RANK):
    return lb.valid(cfg, QB_N, k, m)


def section_n(gen):
    rows = 8192 + 256
    base = torch.randn((rows, A_N), generator=gen, device="cuda")
    scale = torch.logspace(-3, 2, rows, device="cuda")[torch.randperm(rows, generator=gen, device="cuda")]
    base = base * scale.view(-1, 1)
    base[0:16] = 0.0
    base[16:48, 64:96] *= 30.0
    base[48:80] = torch.randn((32, A_N), generator=gen, device="cuda") * 1e-39
    base[80:112] = torch.randn((32, A_N), generator=gen, device="cuda") * 3e4
    base[112:144, :640] = 0.0
    base[144:176] = torch.randn((32, A_N), generator=gen, device="cuda") * 1e-20
    full = base.to(torch.bfloat16)
    xs = full[:, :Q_RANK]
    nw = (1.0 + 0.2 * torch.randn(Q_RANK, generator=gen, device="cuda")).to(torch.bfloat16)
    w, sw, _ = lb.make_weight(QB_N, Q_RANK, gen)
    ref = rms(xs, nw)
    torch.cuda.synchronize()
    bad_qr = bad_q = 0
    max_ulp = 0
    r0 = 0
    m = 1
    chunks = 0
    table = lg()._QNORM_CONFIGS[(QB_N, Q_RANK)]
    while r0 < rows:
        mm = min(m, rows - r0)
        cfg = table[mm]
        xq = torch.empty((mm, Q_RANK), dtype=torch.uint8, device="cuda")
        xsf = torch.zeros((128 * Q_RANK // 32,), dtype=torch.uint8, device="cuda")
        qr, _ = qnorm(xs[r0:r0 + mm], w, sw, nw, cfg, xq=xq, xsf=xsf)
        qref, sref = torch.ops.trtllm.mxfp8_quantize(ref[r0:r0 + mm].contiguous(), True)
        a = qr.view(torch.int16).int()
        b = ref[r0:r0 + mm].view(torch.int16).int()
        bad_qr += int((a != b).any(1).sum().item())
        max_ulp = max(max_ulp, int((a - b).abs().max().item()))
        okq = torch.equal(xq, qref.view(torch.uint8)) and torch.equal(lb.unswizzle(xsf, mm, Q_RANK),
                                                                       lb.unswizzle(sref, mm, Q_RANK))
        bad_q += 0 if okq else mm
        r0 += mm
        m = m % 8 + 1
        chunks += 1
    entry = {"rows": rows, "launches": chunks, "qr_bad_rows": bad_qr, "qr_max_ulp": max_ulp, "quant_bad_rows": bad_q}
    log("N", json.dumps(entry))
    if bad_qr or bad_q:
        fail(f"N: {entry}")
    results["N"] = entry


def section_o(gen):
    wa, swa, _ = lb.make_weight(A_N, HIDDEN, gen)
    wb, sb, rawb = lb.make_weight(QB_N + IDX_N, Q_RANK, gen)
    wq, sq, rawq = wb[:QB_N], sb[: QB_N * 40], rawb[:QB_N]
    nw = (1.0 + 0.2 * torch.randn(Q_RANK, generator=gen, device="cuda")).to(torch.bfloat16)
    out = {}
    L = lg()
    for m in range(1, 9):
        x = (torch.randn((m, HIDDEN), generator=gen, device="cuda") * 2).to(torch.bfloat16)
        ca = L._CHAIN_A_CONFIGS[(A_N, HIDDEN)][m]
        rows0 = lowm(x, wa, swa, ca)[0]
        rows1 = lowm(x, wa, swa, ca, flags=2)[0]
        a_ok = torch.equal(rows0, rows1)
        qr_ref = rms(rows0[:, :Q_RANK], nw)
        ref64 = lb.dequant_ref(qr_ref.contiguous(), wq, rawq).double()
        ref64_i = lb.dequant_ref(qr_ref.contiguous(), wb, rawb).double()
        b12x = lb.main_mm(qr_ref.contiguous(), wq, sq)
        bidx = lb.main_mm(qr_ref.contiguous(), wb, sb)
        torch.cuda.synchronize()

        def err(o, r):
            d = (o.double() - r).abs()
            return d.max().item(), d.max().item() / max(r.abs().max().item(), 1e-30)

        row = {"A_trigger_bitwise": a_ok, "b12x_q": err(b12x, ref64), "b12x_qi": err(bidx, ref64_i), "cfgs": {}}
        if not a_ok:
            fail(f"O M={m}: A trigger-at-start output differs")
        for cfg in QCFGS:
            if not fits(cfg, m):
                continue
            lm = lowm(qr_ref.contiguous(), wq, sq, cfg)[0]
            lmi = lowm(qr_ref.contiguous(), wb, sb, cfg, n_split=QB_N)
            qr, q = qnorm(rows1[:, :Q_RANK], wq, sq, nw, cfg)
            qr2, qi, ii = qnorm(rows1[:, :Q_RANK], wb, sb, nw, cfg, n_split=QB_N)
            torch.cuda.synchronize()
            c = {"qr_bitwise": torch.equal(qr.view(torch.int16), qr_ref.view(torch.int16))
                 and torch.equal(qr2.view(torch.int16), qr_ref.view(torch.int16)),
                 "q_bitwise_vs_lowm": torch.equal(q, lm),
                 "idx_bitwise_vs_lowm": torch.equal(qi, lmi[0]) and torch.equal(ii, lmi[1]),
                 "q_err_fp64": err(q, ref64), "qi_err_fp64": err(torch.cat([qi, ii], 1), ref64_i),
                 "shapes": [tuple(t.shape) for t in (qr, q, qi, ii)],
                 "finite": bool(torch.isfinite(q).all() and torch.isfinite(ii).all())}
            ok = c["qr_bitwise"] and c["q_bitwise_vs_lowm"] and c["idx_bitwise_vs_lowm"] and c["finite"] and \
                c["q_err_fp64"][1] < 1.6e-2 and c["qi_err_fp64"][1] < 1.6e-2
            if not ok:
                fail(f"O M={m} cfg={cfg}: {c}")
            row["cfgs"][str(cfg)] = c
        out[m] = row
        log(f"O M={m} {json.dumps(row)}")
    results["O"] = out


def section_b(gen):
    sys.path.insert(0, os.path.join(HERE, "..", "csa2_lowm"))
    import lowm_check as lc

    prod = lc._producer_kernel()
    wb, sb, _ = lb.make_weight(QB_N + IDX_N, Q_RANK, gen)
    wa, swa, _ = lb.make_weight(A_N, HIDDEN, gen)
    nw = (1.0 + 0.2 * torch.randn(Q_RANK, generator=gen, device="cuda")).to(torch.bfloat16)
    L = lg()
    out = {}
    for m in (1, 2, 4, 8):
        for n_split, table_key in ((0, (QB_N, Q_RANK)), (QB_N, (QB_N + IDX_N, Q_RANK))):
            cfg = L._QNORM_CONFIGS[table_key].get(m) or L._QNORM_CONFIGS[(QB_N, Q_RANK)][m]
            w = wb[:QB_N] if n_split == 0 else wb
            s = sb[: QB_N * 40] if n_split == 0 else sb
            x = (torch.randn((m, HIDDEN), generator=gen, device="cuda") * 2).to(torch.bfloat16)
            src = lowm(x, wa, swa, L._CHAIN_A_CONFIGS[(A_N, HIDDEN)][m])[0]
            ref = qnorm(src[:, :Q_RANK], w, s, nw, cfg, n_split)
            dst = torch.empty_like(src)
            numel = src.numel()

            def run(flags):
                dst.fill_(float("nan"))
                prod[((numel + 1023) // 1024,)](src, dst, numel, SPINS=50, BLOCK=1024)
                return qnorm(dst[:, :Q_RANK], w, s, nw, cfg, n_split, flags)

            def same(a, b):
                return all(torch.equal(u.view(torch.int16), v.view(torch.int16)) for u, v in zip(a, b))

            e_bad = c_bad = 0
            for _ in range(10):
                y = run(0)
                yc = run(1)
                torch.cuda.synchronize()
                e_bad += int(not same(y, ref))
                c_bad += int(not same(yc, ref))
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                run(0)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=st):
                    yg = run(0)
                gc = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gc, stream=st):
                    ygc = run(1)
            torch.cuda.current_stream().wait_stream(st)
            torch.cuda.synchronize()
            g_bad = gc_bad = 0
            for _ in range(20):
                for t in yg:
                    t.fill_(0)
                g.replay()
                torch.cuda.synchronize()
                g_bad += int(not same(yg, ref))
                gc.replay()
                torch.cuda.synchronize()
                gc_bad += int(not same(ygc, ref))
            entry = {"cfg": cfg, "eager_bad_of_10": e_bad, "eager_skipwait_bad_of_10": c_bad, "graph_bad_of_20": g_bad,
                     "graph_skipwait_bad_of_20": gc_bad}
            key = f"M{m}_{'index' if n_split else 'plain'}"
            log(f"B {key} {json.dumps(entry)}")
            if e_bad or g_bad:
                fail(f"B {key}: wrong under a late producer {entry}")
            if c_bad == 0 and gc_bad == 0:
                fail(f"B {key}: skip-wait control did not fail (test not sensitive) {entry}")
            out[key] = entry
            del g, gc
    results["B"] = out


def section_c(gen):
    sys.path.insert(0, os.path.join(HERE, "..", "csa2_lowm"))
    import lowm_check as lc

    aggs = lc._aggressors(gen)
    for agg in aggs.values():  # initialize cuBLAS handles/workspaces eagerly, never inside capture
        agg()
    torch.cuda.synchronize()
    L = lg()
    nw = (1.0 + 0.2 * torch.randn(Q_RANK, generator=gen, device="cuda")).to(torch.bfloat16)
    copies = []
    for _ in range(4):
        wa, swa, _ = lb.make_weight(A_N, HIDDEN, gen)
        wq, sq, _ = lb.make_weight(QB_N, Q_RANK, gen)
        copies.append((wa, swa, wq, sq))
    out = {}
    b12x_faults = 0
    for m in (1, 2, 4, 8):
        x = (torch.randn((m, HIDDEN), generator=gen, device="cuda") * 2).to(torch.bfloat16)
        ca = L._CHAIN_A_CONFIGS[(A_N, HIDDEN)][m]
        cb = L._QNORM_CONFIGS[(QB_N, Q_RANK)][m]
        replays = 90 if m == 1 else 60
        reps = 8

        def chain_victims():
            def one(c):
                rows = lowm(x, c[0], c[1], ca, flags=2)[0]
                return qnorm(rows[:, :Q_RANK], c[2], c[3], nw, cb)[1]
            return [(lambda c=c: one(c)) for _ in range(reps) for c in copies]

        def b12x_victims():
            qx = (torch.randn((m, Q_RANK), generator=gen, device="cuda")).to(torch.bfloat16)
            return [(lambda c=c: lb.main_mm(qx, c[2], c[3])) for _ in range(reps) for c in copies]

        entry = {"cfg_a": ca, "cfg_b": cb, "victims_per_replay": reps * len(copies), "replays": replays}
        for aname, agg in aggs.items():
            pbad, pe, pinfo = lc._overlap_case(chain_victims, agg, replays)
            bbad, be, binfo = lc._overlap_case(b12x_victims, agg, replays)
            entry[aname] = {"chain_bad_replays": pbad, "chain_bad_elems": pe, "chain_overlap": pinfo,
                            "b12x_control_bad_replays": bbad, "b12x_overlap": binfo}
            b12x_faults += bbad
            log(f"C M={m} {aname} {json.dumps(entry[aname])}")
            if pbad:
                fail(f"C M={m} {aname}: chain wrong under the cross-stream aggressor {entry[aname]}")
        out[f"M{m}"] = entry
    out["b12x_control_faults_total"] = b12x_faults
    if b12x_faults == 0:
        fail("C: b12x control never faulted (W-AGG not sensitive in this job)")
    results["C"] = out


def _reference_route(probe, h):
    """The chain's math composed from reference ops (rmsnorm_cute + csa2_mxfp8_low_m_gemm on the chain's A config)."""
    L = lg()
    m = h.shape[0]
    rows = lowm(h, *probe._wq_a_wkv, L._CHAIN_A_CONFIGS[(A_N, HIDDEN)][m])[0]
    qr = rms(rows[:, :Q_RANK], probe.q_norm.weight)
    if probe.with_index:
        cfg = L._QNORM_CONFIGS[(QB_N + IDX_N, Q_RANK)].get(m)
        if cfg is None:
            return None
        q, idx = lowm(qr.contiguous(), *probe._wq_b_index, cfg, n_split=QB_N)
    else:
        cfg = L._QNORM_CONFIGS[(QB_N, Q_RANK)][m]
        q = lowm(qr.contiguous(), probe.wq_b.weight, probe.wq_b.weight_scale, cfg)[0]
        idx = None
    return qr, q, rows[:, Q_RANK:], idx


def _kernel_names(fn):
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as p:
        fn()
        torch.cuda.synchronize()
    return [e.name for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA]


def section_d_e(gen):
    out_d, out_e = {}, {}
    names = ("qr", "q", "swa", "idx")
    kn = {}
    for with_index in (True, False):
        probe = pb.Probe(with_index, gen)
        hs = {m: torch.randn((m, HIDDEN), generator=gen, device="cuda").to(torch.bfloat16) for m in (1, 2, 4, 8)}
        sep = {m: probe.separate(h) for m, h in hs.items()}
        assert pb.attach_branch(probe)
        weights = [t.clone() for t in (probe._wq_a_wkv[0], probe._wq_a_wkv[1], probe.wq_b.weight)]
        for m in (1, 2, 4, 8):
            h = hs[m]
            eager = pb.branch(probe, h)
            ref = _reference_route(probe, h)
            torch.cuda.synchronize()
            routed = ref is not None
            key = f"{'index' if with_index else 'swa'}_M{m}"
            ref_ok = (not routed) or all(pb.equal(a, b) for a, b in zip(eager, ref))
            tol = {}
            for n, a, b in zip(names, eager, sep[m]):
                if a is None or b is None:
                    continue
                d = (a.float() - b.float()).abs().max().item()
                tol[n] = d / max(b.float().abs().max().item(), 1e-30)
            static_h = h.clone()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                pb.branch(probe, static_h)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=s):
                    gout = pb.branch(probe, static_h)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            names_k = _kernel_names(lambda: pb.branch(probe, h))
            kn[key] = sorted(set(n[:60] for n in names_k))
            bad = 0
            for step in range(10):
                hn = torch.randn((m, HIDDEN), generator=gen, device="cuda").to(torch.bfloat16)
                r = pb.branch(probe, hn)
                torch.cuda.synchronize()
                static_h.copy_(hn)
                for t in gout:
                    if t is not None:
                        t.fill_(float("nan"))
                g.replay()
                torch.cuda.synchronize()
                bad += int(not all(pb.equal(a, b) for a, b in zip(gout, r)))
            out_d[key] = {"routed_chain": routed, "eager_bitwise_vs_reference_composition": ref_ok,
                          "decode_window_steps": 10, "graph_bad_vs_eager": bad, "rel_vs_separate": tol,
                          "kernels": kn[key]}
            log(f"D {key} {json.dumps(out_d[key])}")
            no_b12x = not any("B12x" in n or "b12x" in n.lower() for n in kn[key]) if routed else True
            no_norm = not any("RMSNorm" in n or "rmsnorm" in n.lower() for n in kn[key]) if routed else True
            if routed and not (no_b12x and no_norm):
                fail(f"D {key}: gated route still runs b12x / rmsnorm kernels: {kn[key]}")
            if not ref_ok or bad or any(v > 1.6e-2 for v in tol.values()):
                fail(f"D {key}: {out_d[key]}")
            per_iter = []
            for i in range(5 + 30):
                gi = torch.Generator(device="cuda").manual_seed(pb.SEED + 17 * i + m)
                x = torch.randn((m, HIDDEN), generator=gi, device="cuda").to(torch.bfloat16)
                expect = pb.branch(probe, x)
                torch.cuda.synchronize()
                static_h.copy_(x)
                g.replay()
                torch.cuda.synchronize()
                errs = []
                for a, b in zip(gout, expect):
                    if a is None:
                        continue
                    d = (a.float() - b.float()).abs()
                    errs.append((d.max().item(), (d / b.float().abs().clamp_min(1e-30)).max().item()))
                per_iter.append((max(e[0] for e in errs), max(e[1] for e in errs)))
            unchanged = all(torch.equal(a, b) for a, b in zip(
                weights, (probe._wq_a_wkv[0], probe._wq_a_wkv[1], probe.wq_b.weight)))
            worst = max(p[0] for p in per_iter)
            out_e[key] = {"iterations": len(per_iter), "max_abs_per_iter": [p[0] for p in per_iter],
                          "max_rel_per_iter": [p[1] for p in per_iter], "weights_unchanged": unchanged}
            log(f"E {key} iterations={len(per_iter)} worst_abs={worst} unchanged={unchanged}")
            if worst != 0 or not unchanged:
                fail(f"E {key}: worst_abs={worst} unchanged={unchanged}")
            del g
            torch.cuda.empty_cache()
    results["D"] = out_d
    results["E"] = out_e


def section_k(gen):
    L = lg()
    out = {}
    for with_index in (True, False):
        probe = pb.Probe(with_index, gen)
        assert pb.attach_branch(probe)
        for env, m in (("TRTLLM_CSA2_QB_FUSED", 1), ("TRTLLM_LOW_M_MXFP8", 2), (None, 9)):
            h = torch.randn((m, HIDDEN), generator=gen, device="cuda").to(torch.bfloat16)
            if env:
                os.environ[env] = "0"
            L._state.clear()
            names_k = _kernel_names(lambda: pb.branch(probe, h))
            got = pb.branch(probe, h)
            if env:
                del os.environ[env]
            L._state.clear()
            fused = any("csa2_mxfp8_low_m_gemm_kernel<4, 4, 2, 1, true>" in n or ", true>" in n for n in names_k)
            key = f"{'index' if with_index else 'swa'}_{env or 'M9'}"
            out[key] = {"fused_kernel_launched": fused, "kernels": sorted(set(n[:60] for n in names_k))}
            log(f"K {key} {json.dumps(out[key])}")
            if fused:
                fail(f"K {key}: fused q_norm kernel launched with the route disabled")
    results["K"] = out


def main():
    import tensorrt_llm._torch.custom_ops  # noqa: F401

    assert hasattr(torch.ops.trtllm, "csa2_qnorm_low_m_gemm"), "csa2_qnorm_low_m_gemm not built"
    only = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
    gen = torch.Generator(device="cuda").manual_seed(lb.SEED)
    for name, fn in (("N", section_n), ("O", section_o), ("B", section_b), ("D", section_d_e), ("K", section_k),
                     ("C", section_c)):
        if only and name not in only:
            continue
        log(f"=== section {name}")
        try:
            fn(gen)
        except Exception as e:
            import traceback

            traceback.print_exc()
            fail(f"section {name} raised {e!r}")
    summary = {"passed": not failures, "failures": failures[:20],
               "compared_against": "FlashInfer rmsnorm_cute qr, trtllm::mxfp8_quantize bytes, csa2_mxfp8_low_m_gemm "
                                   "on the rmsnorm_cute qr (bitwise), FP64 dequant reference, eager _project_inputs "
                                   "and the separate-Linear (quantize + b12x) path"}
    with open(os.path.join(os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp"), "qb_op_check.json"), "w") as f:
        json.dump(results, f, indent=1, default=str)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
