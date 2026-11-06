"""Idea 272 (branch 179): attention-input chain L2 prefetch, SM120 decode M<=8, TP4 rank shapes.

40-layer cold-L2 CUDA-graph chain per layer:
  sep (non-PDL elementwise kernel, the rest of the previous layer)
  -> REAL hc_attn K1 (mhc_lagged_layer_input, Triton, PDL: wait then trigger at start)
  -> A = low_m_mm(wq_a||wkv [1792,5120], trigger_at_start=True[, l2_prefetch=(wB, sfB)])
  -> B = qnorm_mm (fused q_norm + MXFP8 quant + wq_b [8192,1280], or wq_b||index_wq_b [12288,1280] n_split 8192)
32 plain + 8 indexer layers interleaved (fact 1587). Every layer has its own weights (~860 MB > 128 MB L2).
A and B are exactly the calls csa2/module.py _project_inputs makes on the chain route.

modes (argv[1]):
  arms  : candidate tree only; base (TRTLLM_CSA2_QB_L2PF=0) vs PF0/PF1/PF2 x frac x policy; WHOLE us/layer median,
          p10/p90; Kineto K1/A/B dur and K1 end -> B end; A per-CTA residency stamps.
  pair  : run_gpu_pair evidence (tree default; the base tree has no l2_prefetch and runs its own chain).
  check : untimed sequence (same warmup/iterations as timing), every iteration's A out, qr, q, index q bitwise vs
          the gate-off graph on the same fresh inputs, M = 1/2/4/8, PDL on, then again in a child with
          TRTLLM_ENABLE_PDL=0.
Prints one JSON object last.
"""

import inspect
import json
import os
import subprocess
import sys
import tempfile

import torch

SEED = 4321
HIDDEN = 5120
A_N = 1792
Q_RANK = 1280
QB_N = 8192
IDX_N = 4096
LAYERS = 40
DRAM_GBPS = 1506.0
EPS = 1e-6
MS = tuple(int(v) for v in os.environ.get("L2PF_MS", "1,2,3,4,8").split(","))
ROUNDS = int(os.environ.get("L2PF_ROUNDS", "30"))
WARMUP = 5


def log(*a):
    print(*a, flush=True)


COMPILES = {"n": 0, "in_capture": 0, "capturing": False}


def _count_compiles():
    """Wrap Triton / CuTe / cpp_extension compile entry points; count calls made while a graph is being captured."""
    def wrap(mod, name):
        fn = getattr(mod, name, None)
        if fn is None or getattr(fn, "_l2pf_wrapped", False):
            return

        def w(*a, **k):
            COMPILES["n"] += 1
            if COMPILES["capturing"]:
                COMPILES["in_capture"] += 1
            return fn(*a, **k)

        w._l2pf_wrapped = True
        setattr(mod, name, w)

    try:
        import triton
        import triton.compiler as tc
        import triton.compiler.compiler as tcc
        wrap(tcc, "compile")
        wrap(tc, "compile")  # JITFunction.create_binder does `from ..compiler import compile`
        wrap(triton, "compile")
    except Exception:  # noqa: BLE001
        pass
    try:
        import cutlass.cute as cute
        wrap(cute, "compile")
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch.utils.cpp_extension as ce
        wrap(ce, "load")
        wrap(ce, "load_inline")
    except Exception:  # noqa: BLE001
        pass


def occupancy_table(L):
    """S0: dynamic smem/CTA (csa2SmemBytes in mxfp8LowMGemm.cu, same formula), grid and smem-limited CTAs/SM of
    A (chain wq_a||wkv) and B (fused q_norm + wq_b, plain 8192 and indexer 12288) at M = 1/2/4/8."""
    props = torch.cuda.get_device_properties(0)
    per_sm = getattr(props, "shared_memory_per_multiprocessor", 102400)
    k_row_pad = 32

    def smem(warps, ck, rt, m, k):
        kc = k // ck
        kcb = kc // 32
        rows = 16 * rt
        w_bytes = rows * (kc + k_row_pad)
        red = warps * rows * 8 * 4
        return (w_bytes + rt * (kcb // 4) * 256 + m * kc + ((m * kcb + 15) // 16) * 16 + (ck - 1) * rows * m * 4
                + (0 if red <= w_bytes else red))

    out = {"smem_per_sm": per_sm, "reserved_per_cta": 1024}
    for m in (1, 2, 4, 8):
        row = {}
        for name, n, k, cfg in (("A_wq_a_wkv", A_N, HIDDEN, L.chain_a_config_for(A_N, HIDDEN, m)),
                                ("B_plain", QB_N, Q_RANK, L.qnorm_config_for(QB_N, Q_RANK, m)),
                                ("B_indexer", QB_N + IDX_N, Q_RANK, L.qnorm_config_for(QB_N + IDX_N, Q_RANK, m))):
            if cfg is None:
                row[name] = None
                continue
            s = smem(cfg[0], cfg[1], cfg[2], m, k)
            row[name] = {"cfg": list(cfg), "smem": s, "grid": n // (16 * cfg[2]) * cfg[1], "threads": cfg[0] * 32,
                         "ctas_per_sm": per_sm // (s + 1024)}
        row["K1_mhc_lagged"] = {"grid": m, "note": "Triton, 1 CTA per row; smem from its compiled metadata below"}
        out[str(m)] = row
    try:
        from tensorrt_llm._torch.modules.mhc import mhc_lagged_split as S
        kern = S._mhc_lagged_layer_input_kernel
        caches = getattr(kern, "device_caches", None) or {}
        for dev, c in caches.items():
            for key, ck in (c[0].items() if isinstance(c, tuple) else c.items()):
                md = getattr(ck, "metadata", None)
                out["K1_compiled"] = {"shared": getattr(md, "shared", None), "num_warps": getattr(md, "num_warps", None),
                                      "n_regs": getattr(ck, "n_regs", None)}
                break
    except Exception as e:  # noqa: BLE001
        out["K1_compiled"] = repr(e)[:200]
    return out


def make_weight(n, k, gen):
    w = (torch.randn((n, k), generator=gen, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    raw = torch.randint(118, 128, (n, k // 32), generator=gen, device="cuda", dtype=torch.int32).to(torch.uint8)
    sw = torch.ops.trtllm.block_scale_interleave(raw.contiguous()).reshape(-1)
    return w, sw


class Chain:
    def __init__(self, gen):
        import tensorrt_llm._torch.custom_ops  # noqa: F401
        from tensorrt_llm._torch.attention.backends.sparse.csa2 import low_m_gemm as L
        from tensorrt_llm._torch.modules.mhc.mhc_lagged_split import mhc_lagged_layer_input

        _count_compiles()
        self.L = L
        self.k1 = mhc_lagged_layer_input
        self.has_pf = "l2_prefetch" in inspect.signature(L.low_m_mm).parameters
        self.index = [(l % 5) == 2 for l in range(LAYERS)]  # 8 of 40
        self.A = [make_weight(A_N, HIDDEN, gen) for _ in range(LAYERS)]
        self.B = [make_weight(QB_N + IDX_N if ix else QB_N, Q_RANK, gen) for ix in self.index]
        self.gamma = (1.0 + 0.2 * torch.randn(Q_RANK, generator=gen, device="cuda")).to(torch.bfloat16)
        self.nw = (1.0 + 0.1 * torch.randn(HIDDEN, generator=gen, device="cuda")).to(torch.bfloat16)
        self.scratch = torch.zeros((8, 64), dtype=torch.bfloat16, device="cuda")
        # warm the chain kernels' attributes eagerly (same as module.py's qnorm_chain_warmup)
        for ix, (wb, sb) in zip((False, True), (self.B[0], self.B[2])):
            L.qnorm_chain_warmup(*self.A[0], wb, sb, self.gamma, EPS, QB_N if ix else 0)

    def inputs(self, m, gen):
        return {
            "x": (torch.randn((m, HIDDEN), generator=gen, device="cuda")).to(torch.bfloat16),
            "res": (torch.randn((m, 4, HIDDEN), generator=gen, device="cuda")).to(torch.bfloat16),
            "post": torch.rand((m, 4), generator=gen, device="cuda") + 0.5,
            "comb": torch.rand((m, 4, 4), generator=gen, device="cuda") * 0.5,
            "pre": torch.rand((m, 4), generator=gen, device="cuda") + 0.5,
        }

    def layer(self, l, inp, outs, dbg=None, flags_override=None):
        L = self.L
        m = inp["x"].shape[0]
        self.scratch[:m, :8].add_(1.0)  # non-PDL separator (rest of the previous layer)
        h = self.k1(inp["x"], inp["res"], inp["post"], inp["comb"], inp["pre"], self.nw, EPS, 224, enable_pdl=True)
        wa, sa = self.A[l]
        wb, sb = self.B[l]
        n_split = QB_N if self.index[l] else 0
        if n_split and L.qnorm_config_for(QB_N + IDX_N, Q_RANK, m) is None:
            # no fused wq_b||index_wq_b config at this M (module takes the unfused route there): plain wq_b slice
            wb, sb, n_split = wb[:QB_N], sb[: QB_N * (Q_RANK // 32)], 0
        if dbg is not None:
            # stamp run: same launch as low_m_mm with dbg_ts (A per-CTA residency)
            cfg = L.chain_a_config_for(A_N, HIDDEN, m)
            flags = L._TRIGGER_AT_START
            kw = {}
            pf = L.l2pf_setting(m) if self.has_pf else None
            if pf is not None:
                flags |= L._L2PF | (pf[0] << 3) | ((1 if pf[1] else 0) << 5)
                kw = dict(pf_w=wb, pf_sf=sb, pf_frac=pf[2])
            rows = torch.ops.trtllm.csa2_mxfp8_low_m_gemm(h, wa, sa, cfg[0], cfg[1], cfg[2], 0, True, dbg_ts=dbg[l],
                                                         debug_flags=flags, **kw)[0]
        elif self.has_pf:
            rows = L.low_m_mm(h, wa, sa, trigger_at_start=True, l2_prefetch=(wb, sb))
        else:
            rows = L.low_m_mm(h, wa, sa, trigger_at_start=True)
        o = L.qnorm_mm(rows[:, :Q_RANK], self.gamma, EPS, wb, sb, n_split)
        if outs is not None:
            outs.append((rows, *o))
        return o

    def capture(self, inp, dbg=None):
        outs = []
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for l in range(LAYERS):
                self.layer(l, inp, None, dbg)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        COMPILES["capturing"] = True
        try:
            with torch.cuda.graph(g):
                for l in range(LAYERS):
                    self.layer(l, inp, outs, dbg)
        finally:
            COMPILES["capturing"] = False
        torch.cuda.synchronize()
        return g, outs


def arm_list():
    arms = [("base", "0")]
    poss = [int(v) for v in os.environ.get("L2PF_POS", "0,1,2").split(",")]
    fracs = [int(v) for v in os.environ.get("L2PF_FRACS", "256,154,102").split(",")]
    pols = [int(v) for v in os.environ.get("L2PF_POLS", "0,1").split(",")]
    for pos in poss:
        for frac in fracs:
            for last in pols:
                arms.append((f"PF{pos}_f{frac}_{'el' if last else 'pl'}", f"{pos},{last},{frac}"))
    sel = os.environ.get("L2PF_ARMS")
    if sel:
        keep = set(sel.split(";"))
        arms = [a for a in arms if a[0] in keep or a[0] == "base"]
    if os.environ.get("L2PF_WITH_DEFAULT") == "1":
        arms.append(("default", "1"))
    return arms


def stats(v):
    v = sorted(v)
    med = v[len(v) // 2]
    p10 = v[int(0.1 * (len(v) - 1))]
    p90 = v[int(0.9 * (len(v) - 1))]
    return {"us": round(med, 3), "p10": round(p10, 3), "p90": round(p90, 3), "noise": round((p90 - p10) / med, 4)}


def time_graphs(graphs, rounds):
    names = list(graphs)
    t = {n: [] for n in names}
    for _ in range(WARMUP):
        for n in names:
            graphs[n].replay()
    torch.cuda.synchronize()
    for r in range(rounds):
        order = names if r % 2 == 0 else names[::-1]
        for n in order:
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record()
            graphs[n].replay()
            b.record()
            b.synchronize()
            t[n].append(a.elapsed_time(b) * 1000.0 / LAYERS)
    return {n: stats(t[n]) for n in names}, t


def kineto(graph, index):
    """K1 / A / B durations and K1 end -> B end per layer kind (us), from one profiled replay."""
    from torch.profiler import ProfilerActivity, profile

    graph.replay()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        graph.replay()
        torch.cuda.synchronize()
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    prof.export_chrome_trace(path)
    with open(path) as f:
        ev = json.load(f)["traceEvents"]
    os.unlink(path)
    ks = sorted([e for e in ev if e.get("cat") == "kernel"], key=lambda e: e["ts"])
    layers = []
    i = 0
    while i < len(ks):
        if "mhc_lagged" in ks[i]["name"] and i + 2 < len(ks) and "csa2" in ks[i + 1]["name"] \
                and "csa2" in ks[i + 2]["name"]:
            k1, a, b = ks[i], ks[i + 1], ks[i + 2]
            layers.append({"k1": k1["dur"], "a": a["dur"], "b": b["dur"],
                           "k1e_ae": a["ts"] + a["dur"] - k1["ts"] - k1["dur"],
                           "k1e_be": b["ts"] + b["dur"] - k1["ts"] - k1["dur"],
                           "a_start_rel_k1": a["ts"] - k1["ts"], "b_start_rel_k1": b["ts"] - k1["ts"]})
            i += 3
        else:
            i += 1
    out = {"layers_found": len(layers)}
    if len(layers) != LAYERS:
        return out
    for kind, sel in (("plain", False), ("indexer", True)):
        rows = [r for r, ix in zip(layers, index) if ix == sel]
        out[kind] = {k: round(sorted(r[k] for r in rows)[len(rows) // 2], 2) for k in rows[0]}
    return out


def stamp_summary(dbg, grid):
    """A per-CTA %globaltimer stamps (0 start, 2 weights landed, 4 end): residency."""
    res = []
    for buf in dbg:
        t = buf.view(grid, 5).double()
        t0 = t[:, 0].min()
        st, land, end = t[:, 0] - t0, t[:, 2] - t0, t[:, 4] - t0
        first_end = end.min()
        res.append({"start_spread": st.max().item() / 1e3, "land_med": land.median().item() / 1e3,
                    "end_max": end.max().item() / 1e3, "late_ctas": int((st > first_end).sum().item())})
    keys = res[0].keys()
    return {k: round(sorted(r[k] for r in res)[len(res) // 2], 2) for k in keys}


def mode_arms(ch, gen):
    table = {}
    for m in MS:
        try:
            table[str(m)] = _arms_one_m(ch, gen, m)
        except Exception as e:  # noqa: BLE001
            import traceback
            log(f"M={m} arms FAILED: {traceback.format_exc()}")
            table[str(m)] = {"error": repr(e)[:500]}
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    os.environ["TRTLLM_CSA2_QB_L2PF"] = "1"
    return {"arms": table}


def _arms_one_m(ch, gen, m):
    if True:
        inp = ch.inputs(m, gen)
        graphs, kin, stamps = {}, {}, {}
        cfg = ch.L.chain_a_config_for(A_N, HIDDEN, m)
        grid = A_N // (16 * cfg[2]) * cfg[1]
        for name, env in arm_list():
            os.environ["TRTLLM_CSA2_QB_L2PF"] = env
            g, _ = ch.capture(inp)
            graphs[name] = g
        times, _ = time_graphs(graphs, ROUNDS)
        for name, env in arm_list():
            os.environ["TRTLLM_CSA2_QB_L2PF"] = env
            kin[name] = kineto(graphs[name], ch.index)
            dbg = [torch.zeros(grid * 5, dtype=torch.int64, device="cuda") for _ in range(LAYERS)]
            gs, _ = ch.capture(inp, dbg)
            for _ in range(3):
                gs.replay()
            torch.cuda.synchronize()
            stamps[name] = stamp_summary(dbg, grid)
            del gs
        base = times["base"]["us"]
        row = {}
        for name in graphs:
            row[name] = {"whole": times[name], "delta_us_layer": round(times[name]["us"] - base, 3),
                         "kineto": kin[name], "a_stamps": stamps[name]}
            log(f"M={m} {name:16s} {times[name]['us']:7.3f} us/layer ({row[name]['delta_us_layer']:+.3f}) "
                f"p10 {times[name]['p10']:.3f} p90 {times[name]['p90']:.3f} kin {json.dumps(kin[name])} "
                f"stamps {json.dumps(stamps[name])}")
        del graphs
        torch.cuda.empty_cache()
        return row


def mode_pair(ch, gen):
    batch = {}
    for m in MS:
        inp = ch.inputs(m, gen)
        g, _ = ch.capture(inp)
        times, raw = time_graphs({"x": g}, ROUNDS)
        st = times["x"]
        batch[str(m)] = {"kernel_ms": st["us"] * LAYERS / 1000.0, "noise_floor": st["noise"],
                         "per_layer_us": st["us"], "p10_us": st["p10"], "p90_us": st["p90"]}
        log(f"M={m}: {st['us']:.3f} us/layer p10 {st['p10']:.3f} p90 {st['p90']:.3f} (has_pf={ch.has_pf}, "
            f"setting={ch.L.l2pf_setting(m) if ch.has_pf else None})")
        del g
        torch.cuda.empty_cache()
    return {"batch": batch, "output": {"class": "exact"}, "has_pf": ch.has_pf}


def mode_check(ch, gen):
    ok = True
    res = {}
    arms = [a for a in arm_list() if a[0] != "base"]
    arms.append(("default", "1"))
    for m in MS:
        inp = ch.inputs(m, gen)
        os.environ["TRTLLM_CSA2_QB_L2PF"] = "0"
        g_ref, o_ref = ch.capture(inp)
        cands = {}
        for name, env in arms:
            os.environ["TRTLLM_CSA2_QB_L2PF"] = env
            cands[name] = ch.capture(inp)
        bad = {n: 0 for n in cands}
        iters = WARMUP + 10
        for it in range(iters):
            # fresh decode-step inputs (static buffers), as each decode step brings a new token
            new = ch.inputs(m, gen)
            for k in inp:
                inp[k].copy_(new[k])
            g_ref.replay()
            torch.cuda.synchronize()
            ref = [[t.clone() for t in lay] for lay in o_ref]
            fin = all(torch.isfinite(t.float()).all().item() for lay in ref for t in lay)
            ok &= fin
            for n, (g, o) in cands.items():
                g.replay()
                torch.cuda.synchronize()
                for lr, lc in zip(ref, o):
                    for tr, tc in zip(lr, lc):
                        if not torch.equal(tr.view(torch.int16), tc.view(torch.int16)):
                            bad[n] += 1
                            break
        res[str(m)] = {"iterations": iters, "bad_layer_iters": bad, "finite": fin,
                       "tensors_per_layer": [len(l) for l in o_ref[:3]],
                       "default_setting": ch.L.l2pf_setting(m) if ch.has_pf else None,
                       "compiles_in_capture_so_far": COMPILES["in_capture"]}
        log(f"CHECK M={m}: {res[str(m)]}")
        ok &= all(v == 0 for v in bad.values())
        del g_ref, cands
        torch.cuda.empty_cache()
    os.environ["TRTLLM_CSA2_QB_L2PF"] = "1"
    ok &= COMPILES["in_capture"] == 0
    return {"passed": bool(ok), "per_m": res, "compiles_total": COMPILES["n"],
            "compiles_in_capture": COMPILES["in_capture"]}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "pair"
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    ch = Chain(gen)
    log(f"mode={mode} has_pf={ch.has_pf} index layers={sum(ch.index)} MS={MS} rounds={ROUNDS} "
        f"PDL env={os.environ.get('TRTLLM_ENABLE_PDL')}")
    for ix, n in ((False, QB_N), (True, QB_N + IDX_N)):
        wb = A_N * HIDDEN * (1 + 1 / 32) + n * Q_RANK * (1 + 1 / 32)
        log(f"{'indexer' if ix else 'plain'}: A+B weight+SF {wb / 1e6:.2f} MB floor {wb / (DRAM_GBPS * 1e3):.2f} us")
    if mode == "arms":
        out = mode_arms(ch, gen)
    elif mode == "s1check":
        # one job: S0 table, S1 arms table (incl. the shipped per-M default), then the untimed bitwise checks in
        # a child (PDL on, plus its TRTLLM_ENABLE_PDL=0 grandchild). passed = the checks.
        occ = occupancy_table(ch.L)
        log(f"S0 occupancy {json.dumps(occ)}")
        arms = mode_arms(ch, gen)
        env = dict(os.environ, L2PF_MS=os.environ.get("L2PF_CHECK_MS", "1,2,3,4,8"), L2PF_ARMS="PF0_f256_pl")
        env.pop("TRTLLM_CSA2_QB_L2PF", None)
        env.pop("L2PF_WITH_DEFAULT", None)
        log("child: check (PDL on, then its TRTLLM_ENABLE_PDL=0 child)")
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "check"], env=env, capture_output=True,
                           text=True, timeout=2400)
        sys.stdout.write(p.stdout[-8000:])
        sys.stdout.flush()
        try:
            child = json.loads(p.stdout.strip().splitlines()[-1])
        except Exception:  # noqa: BLE001
            child = {"passed": False, "rc": p.returncode, "stderr": p.stderr[-3000:]}
        out = {"passed": bool(child.get("passed")), "check": child, "s0_occupancy": occ, "s1": arms}
    elif mode == "pairchk":
        # candidate arm of the final pair, one job: S0 table, the timing (tree default), then the untimed bitwise
        # checks in children (PDL on with its own TRTLLM_ENABLE_PDL=0 grandchild), so a check failure cannot hide
        # the timing and vice versa.
        occ = occupancy_table(ch.L)
        log(f"S0 occupancy {json.dumps(occ)}")
        os.environ["TRTLLM_CSA2_QB_L2PF"] = "1"
        out = mode_pair(ch, gen)
        env = dict(os.environ, L2PF_MS=os.environ.get("L2PF_CHECK_MS", "1,2,3,4,8"),
                   L2PF_ARMS=os.environ.get("L2PF_ARMS", "PF0_f256_pl"))
        env.pop("TRTLLM_CSA2_QB_L2PF", None)
        log("child: check (PDL on, then its TRTLLM_ENABLE_PDL=0 child)")
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "check"], env=env, capture_output=True,
                           text=True, timeout=2400)
        sys.stdout.write(p.stdout[-8000:])
        sys.stdout.flush()
        try:
            child = json.loads(p.stdout.strip().splitlines()[-1])
        except Exception:  # noqa: BLE001
            child = {"passed": False, "rc": p.returncode, "stderr": p.stderr[-3000:]}
        out["passed"] = bool(child.get("passed"))
        out["check"] = child
        out["s0_occupancy"] = occ
    elif mode == "check":
        out = mode_check(ch, gen)
        if os.environ.get("TRTLLM_ENABLE_PDL") != "0":
            env = dict(os.environ, TRTLLM_ENABLE_PDL="0", L2PF_ARMS="PF0_f256_pl;PF1_f256_pl;PF2_f256_pl")
            log("child: TRTLLM_ENABLE_PDL=0")
            p = subprocess.run([sys.executable, os.path.abspath(__file__), "check"], env=env, capture_output=True,
                               text=True, timeout=1500)
            sys.stdout.write(p.stdout[-6000:])
            sys.stderr.write(p.stderr[-3000:])
            try:
                child = json.loads(p.stdout.strip().splitlines()[-1])
            except Exception:  # noqa: BLE001
                child = {"passed": False, "rc": p.returncode}
            out = {"passed": bool(out["passed"] and child.get("passed")), "pdl_on": out, "pdl_off": child}
    else:
        out = mode_pair(ch, gen)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
