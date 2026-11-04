"""Correctness / overlap checks for trtllm::csa2_mxfp8_low_m_gemm (SM120 port of MiniMax-M3 mxfp8LowMGemm).

Sections (all untimed):
  A  production dispatch (csa2/low_m_gemm.low_m_mm) and every shape's best config at M=1/2/4/8 vs the FP32
     dequant reference; x_q / SF bitwise vs trtllm::mxfp8_quantize(x, True); pre-quantized input bitwise equal.
  B  late NaN-poisoning PDL producer: a producer that triggers launch_dependents at once, sleeps, then writes x
     (poisoned with NaN before). The port must match isolation; the no-wait control (debug_flags=1) must not.
  C  cross-stream DRAM aggressor (pitfall 1353): graph = 4x mul_ on 256 MB fp32 on a forked stream beside a
     chain of victims; >= 30 replays, outputs poisoned before each replay, every replay bitwise vs isolated.
     The b12x control arm (mxfp8_quantize + flashinfer_mm_mxfp8) runs in the same job. Overlap is logged.
  D  decode graphs: DeepseekV41Attention._project_inputs captured at batch 1/2/4/8, 8 consecutive replays with
     fresh inputs, each bitwise vs eager and within tolerance of the separate-Linear path.
  E  repeated execution matching proj_pair.py (5 warmup + 30 iterations, L2 flush before each replay): every
     iteration compared to the reference, inputs/weights checked unchanged.
argv: [checkpoint]
"""

import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import lowm_bench as lb  # noqa: E402

BEST = {(1792, 5120): {1: (8, 2, 2), 2: (8, 2, 2), 4: (8, 4, 1), 8: (8, 4, 1)},
        (8192, 1280): {1: (4, 1, 2), 2: (4, 1, 2), 4: (8, 1, 4), 8: (8, 1, 2)},
        (12288, 1280): {1: (4, 1, 1), 2: (4, 1, 1), 4: (4, 1, 1), 8: (4, 1, 1)},
        (4096, 1280): {1: (4, 1, 2), 2: (4, 1, 2), 4: (8, 1, 2), 8: (8, 1, 2)}}
results = {}
failures = []


def log(*a):
    print(*a, flush=True)


def fail(msg):
    failures.append(msg)
    log("FAIL:", msg)


def port(x, w, sw, cfg, flags=0):
    return torch.ops.trtllm.csa2_mxfp8_low_m_gemm(x, w, sw, cfg[0], cfg[1], cfg[2], 0, True, debug_flags=flags)[0]


def section_a(gen):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.low_m_gemm import config_for, low_m_mm

    out = {}
    for (n, k), table in BEST.items():
        w, sw, raw = lb.make_weight(n, k, gen)
        for m in (1, 2, 4, 8):
            x = (torch.randn((m, k), generator=gen, device="cuda") * 2).to(torch.bfloat16)
            ref = lb.dequant_ref(x, w, raw)
            base = lb.main_mm(x, w, sw)
            cfg = table[m]
            y = port(x, w, sw, cfg)
            (yp,) = lb.port_preq_mm(x, w, sw, cfg)
            torch.cuda.synchronize()
            _, rel = lb.err(y, ref)
            _, rel_main = lb.err(base, ref)
            okq = lb.check_quant(x, w, sw, cfg)
            entry = {"cfg": cfg, "rel_err_vs_fp32": rel, "main_rel_err_vs_fp32": rel_main,
                     "max_abs_vs_main": (y.float() - base.float()).abs().max().item(),
                     "xq_sf_bitwise": okq, "preq_bitwise": torch.equal(y, yp),
                     "shape_ok": tuple(y.shape) == (m, n) and y.stride() == (n, 1)}
            if config_for(n, k, m) is not None:
                yd = low_m_mm(x, w, sw)
                entry["dispatch_bitwise_vs_cfg"] = yd is not None and torch.equal(yd, port(x, w, sw,
                                                                                         config_for(n, k, m)))
                # Dispatch must hand back the same math whatever the config table holds.
                entry["dispatch_rel_err"] = lb.err(yd, ref)[1]
            ok = (torch.isfinite(y).all().item() and rel < 1.6e-2 and okq and entry["preq_bitwise"]
                  and entry["shape_ok"] and entry.get("dispatch_bitwise_vs_cfg", True))
            if not ok:
                fail(f"A {n}x{k} M={m}: {entry}")
            out[f"{n}x{k}_M{m}"] = entry
            log(f"A {n}x{k} M={m} {json.dumps(entry)}")
    results["A"] = out


def _producer_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def late_producer(src, dst, n, SPINS: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        t = tl.zeros([BLOCK], dtype=tl.int32) + pid
        # Release the dependent grid at once (start trigger), then sleep, then publish x.
        t = tl.inline_asm_elementwise("griddepcontrol.launch_dependents;\n mov.b32 $0, $1;", "=r,r", [t],
                                      dtype=tl.int32, is_pure=False, pack=1)
        for _ in range(SPINS):
            t = tl.inline_asm_elementwise("nanosleep.u32 1000;\n mov.b32 $0, $1;", "=r,r", [t],
                                          dtype=tl.int32, is_pure=False, pack=1)
        mask = (offs < n) & (t >= 0)
        v = tl.load(src + offs, mask=mask)
        tl.store(dst + offs, v, mask=mask)

    return late_producer


def section_b(gen):
    try:
        prod = _producer_kernel()
    except Exception as e:  # pragma: no cover
        fail(f"B producer unavailable: {e!r}")
        return
    out = {}
    for (n, k), table in BEST.items():
        w, sw, raw = lb.make_weight(n, k, gen)
        for m in (1, 4):
            cfg = table[m]
            src = (torch.randn((m, k), generator=gen, device="cuda") * 2).to(torch.bfloat16)
            ref = port(src, w, sw, cfg)
            dst = torch.empty_like(src)
            numel = src.numel()
            block = 1024

            def run(flags):
                dst.fill_(float("nan"))
                prod[(triton_cdiv(numel, block),)](src, dst, numel, SPINS=100, BLOCK=block)
                return port(dst, w, sw, cfg, flags)

            # eager
            y = run(0)
            yc = run(1)
            torch.cuda.synchronize()
            eager_ok = torch.equal(y, ref)
            ctrl_bad = not torch.equal(yc, ref)
            # graph (PDL edges inside the captured graph)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                run(0)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=s):
                    yg = run(0)
                gc = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gc, stream=s):
                    ygc = run(1)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g_bad = c_bad = 0
            for _ in range(10):
                yg.fill_(0)
                g.replay()
                torch.cuda.synchronize()
                g_bad += int(not torch.equal(yg, ref))
                ygc.fill_(0)
                gc.replay()
                torch.cuda.synchronize()
                c_bad += int(not torch.equal(ygc, ref))
            entry = {"cfg": cfg, "eager_bitwise": eager_ok, "eager_nowait_control_wrong": ctrl_bad,
                     "graph_bad_of_10": g_bad, "graph_nowait_control_bad_of_10": c_bad}
            log(f"B {n}x{k} M={m} {json.dumps(entry)}")
            if not eager_ok or g_bad:
                fail(f"B {n}x{k} M={m} port wrong under late producer: {entry}")
            if not ctrl_bad and c_bad == 0:
                fail(f"B {n}x{k} M={m} no-wait control did not fail (test not sensitive): {entry}")
            out[f"{n}x{k}_M{m}"] = entry
            del g, gc
    results["B"] = out


def triton_cdiv(a, b):
    return (a + b - 1) // b


def _overlap_case(make_victims, agg, replays):
    """Graph = aggressor on a forked stream beside the victims. Returns (bad replays, overlap info)."""
    victims = make_victims()
    torch.cuda.synchronize()
    iso = [v() for v in victims]
    torch.cuda.synchronize()
    iso = [t.clone() for t in iso]
    s0 = torch.cuda.Stream()
    s1 = torch.cuda.Stream()
    s0.wait_stream(torch.cuda.current_stream())
    graphs = {}
    with torch.cuda.stream(s0):
        for v in victims:
            v()
        for name in ("both", "vic", "agg"):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s0):
                outs = []
                if name != "vic":
                    s1.wait_stream(s0)
                    with torch.cuda.stream(s1):
                        for _ in range(4):
                            agg.mul_(1.0)
                if name != "agg":
                    outs = [v() for v in victims]
                if name != "vic":
                    s0.wait_stream(s1)
            graphs[name] = (g, outs)
    torch.cuda.current_stream().wait_stream(s0)
    torch.cuda.synchronize()
    tm = {}
    for name, (g, _) in graphs.items():
        ts = []
        for _ in range(5):
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record()
            g.replay()
            b.record()
            b.synchronize()
            ts.append(a.elapsed_time(b))
        tm[name] = sorted(ts)[2] * 1000
    g, outs = graphs["both"]
    bad = 0
    bad_elems = 0
    for _ in range(replays):
        for o in outs:
            o.fill_(float("nan"))
        g.replay()
        torch.cuda.synchronize()
        wrong = [int((o != r).sum().item()) for o, r in zip(outs, iso)]
        bad += int(any(wrong))
        bad_elems += sum(wrong)
    info = {"us_both": round(tm["both"], 1), "us_victims": round(tm["vic"], 1), "us_aggressor": round(tm["agg"], 1),
            "overlap_us": round(tm["vic"] + tm["agg"] - tm["both"], 1)}
    del graphs
    return bad, bad_elems, info


def section_c(gen, replays=30):
    agg = torch.ones(64 * 1024 * 1024, dtype=torch.float32, device="cuda")
    out = {}
    for (n, k), table in BEST.items():
        w, sw, raw = lb.make_weight(n, k, gen)
        copies = [(w.clone(), sw.clone()) for _ in range(4)]
        for m in (1, 2, 4, 8):
            cfg = table[m]
            x = (torch.randn((m, k), generator=gen, device="cuda") * 2).to(torch.bfloat16)
            reps = 8  # 32 victim launches per replay, inside the ~1.5 ms aggressor window

            def port_victims():
                return [(lambda wc=wc, sc=sc: port(x, wc, sc, cfg)) for _ in range(reps) for wc, sc in copies]

            def b12x_victims():
                return [(lambda wc=wc, sc=sc: lb.main_mm(x, wc, sc)) for _ in range(reps) for wc, sc in copies]

            pb, pe, pinfo = _overlap_case(port_victims, agg, replays)
            bb, be, binfo = _overlap_case(b12x_victims, agg, replays)
            entry = {"cfg": cfg, "victims_per_replay": reps * len(copies), "replays": replays,
                     "port_bad_replays": pb, "port_bad_elems": pe, "port_overlap": pinfo,
                     "b12x_control_bad_replays": bb, "b12x_control_bad_elems": be, "b12x_overlap": binfo}
            log(f"C {n}x{k} M={m} {json.dumps(entry)}")
            if pb:
                fail(f"C {n}x{k} M={m}: port wrong under the cross-stream aggressor {entry}")
            # Two DRAM-bound workloads sharing bandwidth add up in time even when they run concurrently, so
            # overlap_us (logged) cannot show concurrency. The test is sensitive when the b12x control faults in the
            # same case, under the same aggressor graph.
            if bb == 0:
                fail(f"C {n}x{k} M={m}: b12x control did not fault, aggressor not shown to overlap {binfo}")
            out[f"{n}x{k}_M{m}"] = entry
        del copies
        torch.cuda.empty_cache()
    out["b12x_control_faulted_any"] = any(v["b12x_control_bad_replays"] for v in out.values()
                                          if isinstance(v, dict))
    results["C"] = out


def section_d_e(gen):
    import csa2_shared_proj_bench as pb

    out_d, out_e = {}, {}
    for with_index in (True, False):
        probe = pb.Probe(with_index, gen)
        hs = {m: torch.randn((m, pb.HIDDEN), generator=gen, device="cuda").to(torch.bfloat16) for m in (1, 2, 4, 8)}
        sep = {m: probe.separate(h) for m, h in hs.items()}
        assert pb.attach_branch(probe)
        weights = [t.clone() for t in (probe._wq_a_wkv[0], probe._wq_a_wkv[1])]
        for m in (1, 2, 4, 8):
            h = hs[m]
            eager = pb.branch(probe, h)
            torch.cuda.synchronize()
            # Tolerance vs the separate-Linear (quantize + b12x) path; bitwise is not expected (reordered).
            tol = {}
            for name, a, b in zip(("qr", "q", "swa", "idx"), eager, sep[m]):
                if a is None or b is None:
                    continue
                d = (a.float() - b.float()).abs().max().item()
                sc = b.float().abs().max().item()
                tol[name] = d / max(sc, 1e-30)
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
            bad = 0
            for step in range(8):
                hn = torch.randn((m, pb.HIDDEN), generator=gen, device="cuda").to(torch.bfloat16)
                ref = pb.branch(probe, hn)
                torch.cuda.synchronize()
                static_h.copy_(hn)
                g.replay()
                torch.cuda.synchronize()
                bad += int(not all(pb.equal(a, b) for a, b in zip(gout, ref)))
            key = f"{'index' if with_index else 'swa'}_M{m}"
            out_d[key] = {"replays": 8, "bad_vs_eager": bad, "rel_vs_separate": tol}
            log(f"D {key} {json.dumps(out_d[key])}")
            if bad or any(v > 1.6e-2 for v in tol.values()):
                fail(f"D {key}: {out_d[key]}")
            # E: the timed protocol of proj_pair.py, untimed, every iteration compared.
            static_h.copy_(h)
            l2_flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
            per_iter = []
            for i in range(5 + 30):
                l2_flush.zero_()
                g.replay()
                torch.cuda.synchronize()
                errs = []
                for a, b in zip(gout, eager):
                    if a is None:
                        continue
                    d = (a.float() - b.float()).abs()
                    errs.append((d.max().item(), (d / b.float().abs().clamp_min(1e-30)).max().item()))
                per_iter.append((max(e[0] for e in errs), max(e[1] for e in errs)))
            unchanged = torch.equal(static_h, h) and all(
                torch.equal(a, b) for a, b in zip(weights, (probe._wq_a_wkv[0], probe._wq_a_wkv[1])))
            worst = max(p[0] for p in per_iter)
            out_e[key] = {"iterations": len(per_iter), "max_abs_per_iter": [p[0] for p in per_iter],
                          "max_rel_per_iter": [p[1] for p in per_iter], "inputs_weights_unchanged": unchanged}
            log(f"E {key} iterations={len(per_iter)} worst_abs={worst} unchanged={unchanged}")
            if worst != 0 or not unchanged:
                fail(f"E {key}: worst_abs={worst} unchanged={unchanged}")
            del g, l2_flush
            torch.cuda.empty_cache()
    results["D"] = out_d
    results["E"] = out_e


def main():
    import tensorrt_llm._torch.custom_ops  # noqa: F401

    gen = torch.Generator(device="cuda").manual_seed(lb.SEED)
    for name, fn in (("A", section_a), ("B", section_b), ("C", section_c), ("D/E", section_d_e)):
        log(f"=== section {name}")
        try:
            fn(gen)
        except Exception as e:
            import traceback

            traceback.print_exc()
            fail(f"section {name} raised {e!r}")
    summary = {"passed": not failures, "failures": failures[:20],
               "b12x_control_faulted": results.get("C", {}).get("b12x_control_faulted_any"),
               "compared_against": "FP32 dequant reference, trtllm::mxfp8_quantize bytes, isolated port output, "
                                   "eager _project_inputs and the separate-Linear (quantize + b12x) path"}
    with open(os.path.join(os.environ.get("AUTO_DEV_ARTIFACTS", "."), "lowm_check.json"), "w") as f:
        json.dump(results, f, indent=1, default=str)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
