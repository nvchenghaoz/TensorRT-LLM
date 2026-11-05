"""Idea 233 correctness for the built ops (trtllm::csa2_lowm_wo_a + csa2_mxfp8_low_m_gemm on wo_b), TP4 rank shapes.

Sections (all untimed; argv: <checkpoint> [sections, default ABCDE]):
  A  bytes + numerics at M=1..8: csa2_lowm_wo_a's e4m3 + 1x128 UE8M0 scale bytes vs b12x_wo._inv_rope_quant128 (K1)
     over stress inputs (positions 0..65535 incl. >= 35072); K3' quant bytes vs trtllm::mxfp8_quantize; o_lora and the
     final output vs FP64 beside main's b12x chain error; determinism; the production LowMWOProjection
     (_project_output through the module stand-in) bitwise vs the direct op chain.
  B  live-overlap PDL: a Triton producer triggers launch_dependents at once, sleeps ~5 us, then writes o (NaN-poisoned
     before). K2' (and K3' after it) must match the reference; the skip-wait control (debug_flags=1) must FAIL.
  C  W-AGG: graph = aggressor (mul256 = 4x mul_ on 256 MB fp32, or a 4x 256 MB stream copy) on a forked stream beside
     a chain of 40 K2'->K3' layers on 40 weight copies; outputs NaN-poisoned before every replay; 90 replays at M=1,
     30 at M=2/4/8, each bitwise vs the isolated eager chain. Positive control in the same job: main's b12x K2
     (wo_a_dense_gemm_mxfp8) chain under the same aggressors.
  D  decode window: _project_output captured in CUDA graphs at batch 1/2/4/8, 8 consecutive replays with fresh
     inputs and advancing positions, each bitwise vs eager and within main's (b12x chain) error vs FP64.
  E  repeated execution matching pair.py (5 warmup + 30 iterations, reset before every replay): every iteration
     compared with the reference; inputs, weights and outputs inspected before each reset.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "b12x_wo"))
sys.path.insert(0, HERE)
import common as C  # noqa: E402  (TRTLLM_ENABLE_PDL=1)
import torch  # noqa: E402

import tensorrt_llm._torch.custom_ops  # noqa: E402,F401
from tensorrt_llm._torch.attention.backends.sparse.csa2 import b12x_wo as B  # noqa: E402

results = {}
failures = []
MS = list(range(1, 9))


def log(*a):
    print(*a, flush=True)


def fail(msg):
    failures.append(msg)
    log("FAIL:", msg)


class Env:
    def __init__(self, ckpt):
        from b12x.gemm._shared import wo_mxfp8 as W
        self.W = W
        self.dims = C.dims_from_ckpt(ckpt)
        d = self.dims
        self.G, self.R, self.GW, self.H = d["groups_local"], d["rank"], d["group_width"], d["hidden"]
        self.HD, self.NOPE, self.ROPE = d["head_dim"], d["nope"], d["rope"]
        self.base = C.make_weights(d, seed=1234)
        self.cs = C.make_cos_sin(d)

    def wsc(self, w):
        return (torch.log2(w.sa_pow2.float()) + 127).round().to(torch.uint8).contiguous()

    def k2(self, w, wsc, o2, pos, M, flags=0, xq=None, xs=None):
        cfg = B._LOWM_WO_A_CONFIGS[M]
        return torch.ops.trtllm.csa2_lowm_wo_a(o2, pos, self.cs, w.wa_main, wsc, self.G, self.HD, self.NOPE, cfg[0],
                                               cfg[1], cfg[2], True, xq, xs, flags)

    def k3(self, w, lora, M, xq=None, xsf=None):
        cfg = B._LOWM_WO_B_CONFIGS[M]
        return torch.ops.trtllm.csa2_mxfp8_low_m_gemm(lora, w.wb_q, w.wb_sf.reshape(-1), cfg[0], cfg[1], cfg[2], 0,
                                                     True, None, None, xq, xsf)[0]

    def main_chain(self, w, o3, pos, sc, M):
        """main's b12x_wo route (K1 -> b12x wo_a -> b12x wo_b fused quant), returns (out, o_lora)."""
        W = self.W
        B.inv_rope_quant128(o3, pos, self.cs, sc.x_q, groups=self.G, group_width=self.GW, nope_dim=self.NOPE,
                            rope_dim=self.ROPE)
        W.wo_a_dense_gemm_mxfp8(sc.x_q, w.wa_b12x, out=sc.tmp, alpha=sc.alpha, expected_m=M, sfb_k_replicated=True)
        pipe = (False, True, 0) if M in B._wo_b_wide_rows() else (False, False, 0)
        W.wo_b_dense_gemm_fused_quant_mxfp8(sc.tmp, w.wb_b12x, out=sc.out, expected_m=M, sfb_k_replicated=False,
                                            fq_pipeline=pipe)
        return sc.out[:, :, 0], sc.tmp.permute(0, 2, 1).reshape(M, -1)


def unswizzle(sf, rows, k):
    cols = k // 32
    pc = (cols + 3) // 4 * 4
    r = torch.arange(rows, device=sf.device).view(-1, 1)
    c = torch.arange(cols, device=sf.device).view(1, -1)
    idx = (c % 4) + (c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + (r // 128) * (128 * pc)
    return sf.reshape(-1).view(torch.uint8)[idx]


def k1_ref(env, o3, pos, M):
    sc = C.Scratch(env.dims, M)
    B.inv_rope_quant128(o3, pos, env.cs, sc.x_q, groups=env.G, group_width=env.GW, nope_dim=env.NOPE,
                        rope_dim=env.ROPE)
    torch.cuda.synchronize()
    vals = sc.x_q.values.view(torch.uint8).permute(0, 2, 1).contiguous()
    sm = sc.x_q.scale_mma.view(torch.uint8)
    m = torch.arange(M, device="cuda")
    s = sm[m % 32, (m // 32) % 4, m // 128]
    return vals, s[:, 0].permute(0, 2, 1).contiguous(), bool((s == s[:, :1]).all())


def stress_inputs(env, M, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    d = env.dims
    o3 = torch.randn(M, d["heads_tp"], d["head_dim"], device="cuda", generator=g)
    scale = torch.exp2(torch.randint(-20, 8, (M, d["heads_tp"], d["head_dim"] // 128, 1), device="cuda",
                                     generator=g).float())
    o3 = (o3.view(M, d["heads_tp"], -1, 128) * scale).view_as(o3).to(torch.bfloat16)
    for m in range(M):
        h = (seed + m) % d["heads_tp"]
        o3[m, h, :128] = 0
        o3[m, (h + 1) % d["heads_tp"], 128:256] *= 1e-30
        for k in range(3):
            blk = o3[m, (h + 2 + k) % d["heads_tp"], 128 * k:128 * (k + 1)]
            blk.clamp_(-1, 1)
            blk[5] = 448.0 * 2.0 ** (k * 7 - 9)
    lo = 35072 if seed % 2 else 0
    pos = torch.randint(lo, 65536, (M,), device="cuda", generator=g, dtype=torch.int32)
    return o3, pos


# ------------------------------------------------------------------------------------------------------------------
def section_a(env):
    w = env.base
    wsc = env.wsc(w)
    out = {}
    nbad_q = 0
    ncases = 0
    for M in MS:
        for seed in range(5):
            o3, pos = stress_inputs(env, M, 1000 * M + seed)
            ref_q, ref_s, rep = k1_ref(env, o3, pos, M)
            xq = torch.zeros(M, env.G, env.GW, dtype=torch.uint8, device="cuda")
            xs = torch.zeros(M, env.G, env.GW // 128, dtype=torch.uint8, device="cuda")
            for p in (pos, pos.long()):
                xq.zero_()
                xs.zero_()
                env.k2(w, wsc, o3.view(M, -1), p, M, xq=xq, xs=xs)
                torch.cuda.synchronize()
                nbad_q += int((xq != ref_q).sum()) + int((xs != ref_s).sum()) + int(not rep)
                ncases += 1
    out["k2_quant_bytes_mismatch"] = nbad_q
    out["k2_quant_cases"] = ncases
    log(f"A K2' quant bytes vs K1: {nbad_q} mismatching bytes over {ncases} cases (int32 + int64 positions)")
    if nbad_q:
        fail(f"A K2' quant bytes differ from K1 ({nbad_q})")
    rows = {}
    for M in MS:
        o3, pos = C.make_inputs(env.dims, M, 11 + M)
        pos[-1] = 35072 + M
        o_orig = o3.clone()
        ref_out, ref_lora = C.fp64_reference(o_orig, pos, env.cs, w, env.dims)
        sc = C.Scratch(env.dims, M)
        main_out, main_lora = env.main_chain(w, o3, pos, sc, M)
        main_out, main_lora = main_out.clone(), main_lora.clone()
        ref_q, ref_s, _ = k1_ref(env, o3, pos, M)
        xq = torch.zeros(M, env.G, env.GW, dtype=torch.uint8, device="cuda")
        xs = torch.zeros(M, env.G, env.GW // 128, dtype=torch.uint8, device="cuda")
        loras = [env.k2(w, wsc, o3.view(M, -1), pos, M, xq=xq if i == 0 else None, xs=xs if i == 0 else None)
                 for i in range(3)]
        K = env.G * env.R
        q_ref, s_ref = torch.ops.trtllm.mxfp8_quantize(loras[0], True)
        xq3 = torch.zeros(M, K, dtype=torch.uint8, device="cuda")
        xsf3 = torch.zeros_like(s_ref.view(torch.uint8).reshape(-1))
        outs = [env.k3(w, loras[0], M, xq=xq3 if i == 0 else None, xsf=xsf3 if i == 0 else None) for i in range(3)]
        torch.cuda.synchronize()
        assert torch.equal(o3, o_orig), "o3 modified"
        r = {
            "k2_quant_bytes_equal": bool(torch.equal(xq, ref_q) and torch.equal(xs, ref_s)),
            "k3_quant_bytes_equal": bool(torch.equal(xq3, q_ref.view(torch.uint8))
                                         and torch.equal(unswizzle(xsf3, M, K), unswizzle(s_ref, M, K))),
            "deterministic": all(torch.equal(loras[0], x) for x in loras[1:]) and all(
                torch.equal(outs[0], x) for x in outs[1:]),
            "finite": bool(torch.isfinite(outs[0]).all()),
            "shape": list(outs[0].shape),
            "main_lora_vs_fp64": C.err(main_lora, ref_lora), "new_lora_vs_fp64": C.err(loras[0], ref_lora),
            "main_out_vs_fp64": C.err(main_out, ref_out), "new_out_vs_fp64": C.err(outs[0], ref_out),
            "new_vs_main_out": C.err(outs[0], main_out), "new_lora_vs_main_lora": C.err(loras[0], main_lora),
            "new_out_bitwise_main": bool(torch.equal(outs[0], main_out)),
        }
        ok = (r["k2_quant_bytes_equal"] and r["k3_quant_bytes_equal"] and r["deterministic"] and r["finite"]
              and r["shape"] == [M, env.H]
              and r["new_lora_vs_fp64"]["max_rel"] <= 1.25 * r["main_lora_vs_fp64"]["max_rel"] + 1e-3
              and r["new_out_vs_fp64"]["max_rel"] <= 1.25 * r["main_out_vs_fp64"]["max_rel"] + 1e-3
              and r["new_vs_main_out"]["max_rel"] < 1.6e-2)
        r["ok"] = ok
        rows[M] = r
        log(f"A M={M} ok={ok} quant k2/k3 {r['k2_quant_bytes_equal']}/{r['k3_quant_bytes_equal']} det "
            f"{r['deterministic']} | vs FP64 max_rel: main lora {r['main_lora_vs_fp64']['max_rel']:.4e} new lora "
            f"{r['new_lora_vs_fp64']['max_rel']:.4e} main out {r['main_out_vs_fp64']['max_rel']:.4e} new out "
            f"{r['new_out_vs_fp64']['max_rel']:.4e} | new vs main out max_abs {r['new_vs_main_out']['max_abs']:.3e} "
            f"bitwise {r['new_out_bitwise_main']}")
        if not ok:
            fail(f"A M={M}: {r}")
    out["rows"] = rows
    # Production route: module stand-in + DeepseekV41Attention._project_output.
    import harness as Hn
    mod = Hn.make_module(env.dims, w, env.cs)
    prod = {}
    for M in MS:
        o3, pos = C.make_inputs(env.dims, M, 51 + M)
        o2 = o3.view(M, -1)
        y = Hn.project(mod, o2.clone(), pos)
        lora = env.k2(w, wsc, o2, pos, M)
        y_ref = env.k3(w, lora, M)
        torch.cuda.synchronize()
        path = type(mod._b12x_wo).__name__ if mod._b12x_wo is not None else "main"
        direct = bool(getattr(mod._b12x_wo, "_use_direct_wo_b", lambda: False)())
        same = bool(torch.equal(y, y_ref))
        prod[M] = {"path": path, "direct_wo_b": direct, "bitwise_vs_ops": same}
        log(f"A production M={M}: {prod[M]}")
        if path != "LowMWOProjection" or not direct or not same:
            fail(f"A production route M={M}: {prod[M]}")
    out["production"] = prod
    results["A"] = out


# ------------------------------------------------------------------------------------------------------------------
def _producer():
    import triton
    import triton.language as tl

    @triton.jit
    def late_producer(src, dst, n, SPINS: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        t = tl.zeros([BLOCK], dtype=tl.int32) + pid
        t = tl.inline_asm_elementwise("griddepcontrol.launch_dependents;\n mov.b32 $0, $1;", "=r,r", [t],
                                      dtype=tl.int32, is_pure=False, pack=1)
        for _ in range(SPINS):
            t = tl.inline_asm_elementwise("nanosleep.u32 1000;\n mov.b32 $0, $1;", "=r,r", [t],
                                          dtype=tl.int32, is_pure=False, pack=1)
        mask = (offs < n) & (t >= 0)
        v = tl.load(src + offs, mask=mask)
        tl.store(dst + offs, v, mask=mask)

    return late_producer


def section_b(env):
    prod = _producer()
    w = env.base
    wsc = env.wsc(w)
    out = {}
    for M in (1, 2, 4, 8):
        o3, pos = C.make_inputs(env.dims, M, 71 + M)
        src = o3.view(M, -1).contiguous()
        ref_l = env.k2(w, wsc, src, pos, M).clone()
        ref_y = env.k3(w, ref_l, M).clone()
        dst = torch.empty_like(src)
        n = src.numel()

        def run(flags):
            dst.fill_(float("nan"))
            prod[((n + 1023) // 1024,)](src, dst, n, SPINS=5, BLOCK=1024)
            lora = env.k2(w, wsc, dst, pos, M, flags=flags)
            return lora, env.k3(w, lora, M)

        l0, y0 = run(0)
        lc, yc = run(1)
        torch.cuda.synchronize()
        eager_ok = torch.equal(l0, ref_l) and torch.equal(y0, ref_y)
        ctrl_wrong = not torch.equal(lc, ref_l)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            run(0)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                gl, gy = run(0)
            gc = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gc, stream=s):
                gcl, gcy = run(1)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g_bad = c_bad = 0
        for _ in range(20):
            gl.fill_(0)
            gy.fill_(0)
            g.replay()
            torch.cuda.synchronize()
            g_bad += int(not (torch.equal(gl, ref_l) and torch.equal(gy, ref_y)))
            gcl.fill_(0)
            gc.replay()
            torch.cuda.synchronize()
            c_bad += int(not torch.equal(gcl, ref_l))
        e = {"eager_ok": eager_ok, "eager_skipwait_control_wrong": ctrl_wrong, "graph_bad_of_20": g_bad,
             "graph_skipwait_control_bad_of_20": c_bad}
        out[M] = e
        log(f"B M={M} {json.dumps(e)}")
        if not eager_ok or g_bad:
            fail(f"B M={M}: K2'/K3' wrong behind a start-trigger producer {e}")
        if not ctrl_wrong and c_bad == 0:
            fail(f"B M={M}: skip-wait control did not fail (test not sensitive) {e}")
        del g, gc
    results["B"] = out


# ------------------------------------------------------------------------------------------------------------------
def _aggressors():
    big = torch.ones(64 * 1024 * 1024, dtype=torch.float32, device="cuda")
    dst = torch.empty_like(big)

    def mul256():
        for _ in range(4):
            big.mul_(1.0)

    def stream_copy():
        for _ in range(4):
            dst.copy_(big)

    return {"mul256": mul256, "stream_copy": stream_copy}


def _overlap(victims, agg, replays):
    iso = [v() for v in victims]
    torch.cuda.synchronize()
    iso = [t.clone() for t in iso]
    s0 = torch.cuda.Stream()
    s1 = torch.cuda.Stream()
    s0.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s0):
        for v in victims[:2]:
            v()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s0):
            s1.wait_stream(s0)
            with torch.cuda.stream(s1):
                agg()
            outs = [v() for v in victims]
            s0.wait_stream(s1)
    torch.cuda.current_stream().wait_stream(s0)
    torch.cuda.synchronize()
    bad = bad_el = 0
    for _ in range(replays):
        for o in outs:
            o.fill_(float("nan"))
        g.replay()
        torch.cuda.synchronize()
        wrong = [int((o.view(torch.int16) != r.view(torch.int16)).sum()) for o, r in zip(outs, iso)]
        bad += int(any(wrong))
        bad_el += sum(wrong)
    del g
    return bad, bad_el


def section_c(env, nl=40):
    aggs = _aggressors()
    copies = [env.base] + [C.clone_weights(env.base, env.dims) for _ in range(nl - 1)]
    wscs = [env.wsc(c) for c in copies]
    out = {}
    ctrl_total = 0
    for M in (1, 2, 4, 8):
        replays = 90 if M == 1 else 30
        o3, pos = C.make_inputs(env.dims, M, 91 + M)
        o2 = o3.view(M, -1)
        sc = C.Scratch(env.dims, M)
        W = env.W

        def new_victims():
            vs = []
            for c, s in zip(copies, wscs):
                def f(c=c, s=s):
                    return env.k3(c, env.k2(c, s, o2, pos, M), M)
                vs.append(f)
            return vs

        tmps = [env.W.empty_dense_gemm_mnl_view(M, env.R, env.G, device=torch.device("cuda"), dtype=torch.bfloat16)
                for _ in copies]
        B.inv_rope_quant128(o3, pos, env.cs, sc.x_q, groups=env.G, group_width=env.GW, nope_dim=env.NOPE,
                            rope_dim=env.ROPE)

        def b12x_k2_victims():
            vs = []
            for c, t in zip(copies, tmps):
                def f(c=c, t=t):
                    W.wo_a_dense_gemm_mxfp8(sc.x_q, c.wa_b12x, out=t, alpha=sc.alpha, expected_m=M,
                                            sfb_k_replicated=True)
                    return t
                vs.append(f)
            return vs

        e = {"replays": replays, "layers": nl}
        for an, agg in aggs.items():
            nb, ne = _overlap(new_victims(), agg, replays)
            cb, ce = _overlap(b12x_k2_victims(), agg, replays)
            ctrl_total += cb
            e[an] = {"new_bad_replays": nb, "new_bad_elems": ne, "b12x_k2_control_bad_replays": cb,
                     "b12x_k2_control_bad_elems": ce}
            log(f"C M={M} {an} {json.dumps(e[an])}")
            if nb:
                fail(f"C M={M} {an}: new chain wrong under the cross-stream aggressor {e[an]}")
        out[M] = e
        torch.cuda.empty_cache()
    out["b12x_k2_control_bad_total"] = ctrl_total
    if ctrl_total == 0:
        fail("C: main's b12x K2 control never faulted (W-AGG not sensitive in this job)")
    results["C"] = out


# ------------------------------------------------------------------------------------------------------------------
def section_d_e(env):
    import harness as Hn
    w = env.base
    mod = Hn.make_module(env.dims, w, env.cs)
    out_d, out_e = {}, {}
    for M in (1, 2, 4, 8):
        g0 = torch.Generator(device="cuda").manual_seed(500 + M)
        static_o = torch.randn(M, env.dims["heads_tp"] * env.HD, device="cuda", generator=g0).to(torch.bfloat16)
        static_pos = torch.randint(30000, 35000, (M,), device="cuda", generator=g0, dtype=torch.int32)
        Hn.project(mod, static_o, static_pos)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            Hn.project(mod, static_o, static_pos)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                gout = Hn.project(mod, static_o, static_pos)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        sc = C.Scratch(env.dims, M)
        steps = []
        for step in range(8):
            o_new = torch.randn(M, env.dims["heads_tp"] * env.HD, device="cuda", generator=g0).to(torch.bfloat16)
            pos_new = static_pos + step + 1  # positions advance every decode step
            eager = Hn.project(mod, o_new.clone(), pos_new).clone()
            main_out, _ = env.main_chain(w, o_new.view(M, env.dims["heads_tp"], env.HD).clone(), pos_new, sc, M)
            ref64, _ = C.fp64_reference(o_new.view(M, env.dims["heads_tp"], env.HD), pos_new, env.cs, w, env.dims)
            torch.cuda.synchronize()
            static_o.copy_(o_new)
            static_pos.copy_(pos_new)
            gout.fill_(float("nan"))
            g.replay()
            torch.cuda.synchronize()
            st = {"bitwise_vs_eager": bool(torch.equal(gout, eager)),
                  "new_vs_fp64": C.err(gout, ref64)["max_rel"], "main_vs_fp64": C.err(main_out, ref64)["max_rel"],
                  "new_vs_main": C.err(gout, main_out)["max_rel"]}
            steps.append(st)
            if not st["bitwise_vs_eager"] or st["new_vs_fp64"] > 1.25 * st["main_vs_fp64"] + 1e-3:
                fail(f"D M={M} step {step}: {st}")
        out_d[M] = {"path": type(mod._b12x_wo).__name__, "steps": steps}
        log(f"D M={M} graph decode window: {json.dumps(out_d[M])}")
        # E: the timed protocol of pair.py (5 warmup + 30 iterations, reset before every replay), untimed.
        o_p, pos_p = C.make_inputs(env.dims, M, seed=100 + M)
        pristine = o_p.view(M, -1).clone()
        static_o.copy_(pristine)
        static_pos.copy_(pos_p)
        ref = Hn.project(mod, pristine.clone(), pos_p).clone()
        torch.cuda.synchronize()
        wa0, wb0 = w.wa_main.clone(), w.wb_q.clone()
        per = []
        ok_state = True
        for it in range(5 + 30):
            # inspect the mutable state before the reset point
            ok_state &= bool(torch.equal(static_o, pristine)) and bool(torch.equal(static_pos, pos_p))
            static_o.copy_(pristine)  # reset point
            g.replay()
            torch.cuda.synchronize()
            d = (gout.float() - ref.float()).abs().max().item()
            per.append(d)
        ok_w = bool(torch.equal(wa0, w.wa_main) and torch.equal(wb0, w.wb_q))
        out_e[M] = {"iterations": len(per), "max_abs_per_iter": per, "inputs_unchanged_before_reset": ok_state,
                    "weights_unchanged": ok_w}
        log(f"E M={M} iterations={len(per)} worst_abs={max(per)} inputs_unchanged={ok_state} weights={ok_w}")
        if max(per) != 0 or not ok_state or not ok_w:
            fail(f"E M={M}: {out_e[M]}")
        del g
        torch.cuda.empty_cache()
    results["D"] = out_d
    results["E"] = out_e


def main():
    ckpt = sys.argv[1]
    secs = sys.argv[2] if len(sys.argv) > 2 else "ABCDE"
    t0 = time.time()
    env = Env(ckpt)
    log("ops:", hasattr(torch.ops.trtllm, "csa2_lowm_wo_a"), hasattr(torch.ops.trtllm, "csa2_mxfp8_low_m_gemm"),
        "TRTLLM_CSA2_LOWM_WO", os.environ.get("TRTLLM_CSA2_LOWM_WO", "<unset>"))
    for name, fn in (("A", section_a), ("B", section_b), ("C", section_c), ("D", section_d_e)):
        if name not in secs:
            continue
        log(f"=== section {name} ({time.time() - t0:.0f}s)")
        try:
            fn(env)
        except Exception as e:
            import traceback
            traceback.print_exc()
            fail(f"section {name} raised {e!r}")
    art = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
    os.makedirs(art, exist_ok=True)
    with open(os.path.join(art, "lowm_wo_check.json"), "w") as f:
        json.dump(results, f, indent=1, default=str)
    print(json.dumps({"passed": not failures, "sections": secs, "failures": failures[:20],
                      "b12x_k2_control_bad_total": results.get("C", {}).get("b12x_k2_control_bad_total"),
                      "compared_against": "Triton K1 bytes, trtllm::mxfp8_quantize bytes, FP64 reference, main's b12x "
                                          "chain, eager op chain, isolated eager outputs"}))


if __name__ == "__main__":
    main()
