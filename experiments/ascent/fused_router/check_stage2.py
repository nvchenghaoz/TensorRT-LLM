"""Idea 261 (Stage 2 of idea 67 as its own landing) correctness, untimed. Candidate tree only.

Reference: main's chain dsv3_router_gemm_op -> gate_forward -> FlashInfer mxfp8_q0_route_triton (decode branch:
route_assign_decode_kernel with expert_offset/zero_buf + _mxfp8_q0_route_direct_kernel), i.e. what
Sm12xW4a8DecodeState.run calls when nothing was prerouted.
1. bitwise: M=1..8 (M=3 = the ROUTE_MAX_M boundary + 1) x EP4 slot_start 0/96/192/288 and EP2xTP2 (192 local)
   0/192, random + adversarial x; both sides start from the same poisoned workspace/buffer; every buffer FC1/FC2
   (or 121's fc12 kernel) reads -- offsets, token_map, token_weights, dst_rows, scale_dst_rows, q_out, scale_out and
   the zero-filled q1/sf1/out buffer -- plus ids/weights/logits compared bitwise; counter 0 after every call.
2. repeated: 1000 calls with changing inputs/offsets/M, each compared; counter 0 after each.
3. graph + PDL poison: 8 layers, K1 stand-in (PDL, spins then writes x) before each fused route, x NaN-poisoned and
   the workspaces re-poisoned before each of 10 replays, every layer compared after every replay (M=1/2/4/8, EP4
   and EP2xTP2). TEST_SKIP_WAIT is read ONCE per process: (a) setting it in this process after the first launch must
   change nothing (graph still clean); (b) the negative control runs in its own subprocess with the variable set
   before the first launch and must be caught.
3b. no hang beside an all-SM aux kernel, both orders.
4. leaf: Sm12xW4a8DecodeState.run with the stash (route skipped; fc12 fused kernel of main) vs plain run(): MoE output
   within 2x main's own repeat spread; q1/sf1 of the stash run bitwise vs plain run; foreign stash ignored; module
   path DeepseekV4Gate.fused_forward(decode_state) takes Stage 2 exactly at M <= ROUTE_MAX_M (M=1,2 yes; 3,4,8 no).
5. full decode graph per batch 1/2/4/8 (module path: fused_forward(decode_state) -> routing_method.apply ->
   state.run), EP4 r0 and EP2xTP2 ep1: capture with no JIT (FlashInfer fc12 compile cache and Triton route caches
   unchanged by the capture), 20 replays with changing x each compared to eager plain path; fc12 sync workspace and
   router counter 0 after; then replays beside an all-SM aux kernel finish (no hang).
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (EXPERTS, HIDDEN, SCALE, SPIN_C1_NS, Aux, capture, gate_main, k1_standin, log,  # noqa: E402
                    make_decode_state, router_main)

import torch  # noqa: E402

E_LOCAL = 96
FIELDS = ("offsets", "token_map", "token_weights", "dst_rows", "scale_dst_rows", "q_out", "scale_out")
LAYOUTS = [(96, 0), (96, 96), (96, 192), (96, 288), (192, 0), (192, 192)]
BUF_BYTES = 300 * 1024


def bits(t):
    t = t.contiguous()
    return t.view(torch.uint8) if t.element_size() == 1 else t.view(torch.int32)


def same(a, b):
    return a.shape == b.shape and bool(torch.equal(bits(a), bits(b)))


def poison(ws, buf, val=0x5A):
    for f in FIELDS:
        getattr(ws, f).view(torch.uint8).fill_(val)
    buf.fill_(val)


class Ctx:
    def __init__(self):
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_mxfp8_q0_route_triton import (
            make_mxfp8_q0_route_workspace, mxfp8_q0_route_triton)
        self.mk, self.route = make_mxfp8_q0_route_workspace, mxfp8_q0_route_triton
        self.dev = "cuda"
        self.gen = torch.Generator(device=self.dev).manual_seed(99)
        self.counter = torch.zeros(1, dtype=torch.int32, device=self.dev)
        self.ws_w = [(torch.randn((EXPERTS, HIDDEN), generator=self.gen, device=self.dev) * HIDDEN ** -0.5)
                     .to(torch.bfloat16) for _ in range(4)]

    def bufs(self, m, e=E_LOCAL):
        ids_shape = torch.empty(m, 6, dtype=torch.int32, device=self.dev)
        x0 = torch.empty(m, HIDDEN, dtype=torch.bfloat16, device=self.dev)
        return self.mk(x0, ids_shape, e), torch.empty(BUF_BYTES, dtype=torch.uint8, device=self.dev)

    def run_ref(self, x, w, b, off, ws, buf, e=E_LOCAL):
        lg = router_main(x, w)
        ids, wts = gate_main(lg, b)
        self.route(x, ids, wts, e, workspace=ws, enable_pdl=True, expert_offset=off, zero_buf=buf)
        return ids, wts, lg

    def run_cand(self, x, w, b, off, ws, buf, e=E_LOCAL):
        return torch.ops.trtllm.dsv4_fused_router_route(
            x, w, b, self.counter, SCALE, off, e, ws.offsets, ws.token_map, ws.token_weights, ws.dst_rows,
            ws.scale_dst_rows, ws.q_out, ws.scale_out, buf)

    @staticmethod
    def compare(r, c, wr, wc, br, bc):
        out = {"ids": bool(torch.equal(r[0], c[0])), "weights": same(r[1], c[1]), "logits": same(r[2], c[2]),
               "buf": same(br, bc)}
        for f in FIELDS:
            out[f] = same(getattr(wr, f), getattr(wc, f))
        return out

    def make_x(self, m, kind):
        x = torch.randn((m, HIDDEN), generator=self.gen, device=self.dev)
        if kind == 1:
            x = x * 300.0
        elif kind == 2:
            x = x * 1e-6
        elif kind == 3:
            x = torch.sign(x) * 56.0
            x[:, ::7] *= 0.5
        elif kind == 4:
            x[0] = 0.0
        elif kind == 5:
            x = x * 448.0 / 2.5
        elif kind == 6:
            x[:, :128] = 1e4
        return x.to(torch.bfloat16).contiguous()

    def graph_check(self, m, e=E_LOCAL):
        nl = 8
        gen, dev = self.gen, self.dev
        lw = [self.ws_w[i % 4] for i in range(nl)]
        lb = [(torch.randn((EXPERTS,), generator=gen, device=dev) * 0.05).float() for _ in range(nl)]
        offs = [((0, 96, 192, 288) if e == 96 else (0, 192))[i % (4 if e == 96 else 2)] for i in range(nl)]
        srcs = [self.make_x(m, 0) for _ in range(nl)]
        xs = [s.clone() for s in srcs]
        cws = [self.bufs(m, e) for _ in range(nl)]

        def step():
            o = []
            for li in range(nl):
                k1_standin(srcs[li], xs[li], SPIN_C1_NS)
                o.append(self.run_cand(xs[li], lw[li], lb[li], offs[li], cws[li][0], cws[li][1], e))
            return o

        g, gout = capture(step)
        wr, br = self.bufs(m, e)
        bad = 0
        for s in range(10):
            for li in range(nl):
                srcs[li].copy_(self.make_x(m, s % 7))
                xs[li].fill_(float("nan"))
                poison(*cws[li], 0x33)
            torch.cuda.synchronize()
            g.replay()
            torch.cuda.synchronize()
            if int(self.counter.item()) != 0:
                bad += 1
                self.counter.zero_()
                continue
            for li in range(nl):
                poison(wr, br, 0x33)
                r = self.run_ref(srcs[li], lw[li], lb[li], offs[li], wr, br, e)
                if not all(self.compare(r, gout[li], wr, cws[li][0], br, cws[li][1]).values()):
                    bad += 1
        del g
        return {"m": m, "E": e, "steps": 10, "layers": nl, "bad": bad}


def negative_control():
    """Subprocess body: TRTLLM_DSV4_FUSED_ROUTER_TEST_SKIP_WAIT=1 set before the first launch."""
    import tensorrt_llm  # noqa: F401
    assert os.environ.get("TRTLLM_DSV4_FUSED_ROUTER_TEST_SKIP_WAIT") == "1"
    c = Ctx()
    r = c.graph_check(1)
    print(json.dumps({"negative_control": r}))


def triton_cache_size(fn):
    n = 0
    for attr in ("device_caches", "cache"):
        d = getattr(fn, attr, None)
        if isinstance(d, dict):
            for v in d.values():
                if isinstance(v, (tuple, list)) and v and isinstance(v[0], dict):
                    n += len(v[0])
                elif isinstance(v, dict):
                    n += len(v)
            return n
    return -1


def main():
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.models import modeling_deepseekv4 as mdl
    from tensorrt_llm._torch.models.modeling_deepseekv4 import DeepseekV4Gate
    assert hasattr(torch.ops.trtllm, "dsv4_fused_router_route")
    max_m = mdl._DSV4_FUSED_ROUTER_ROUTE_MAX_M
    c = Ctx()
    gen, dev = c.gen, c.dev
    res, fails = {"route_max_m": max_m}, []

    # ---------------- 1. bitwise ----------------
    n_ok = n_tot = 0
    per = {}
    for m in range(1, 9):
        wsl = {e: (c.bufs(m, e), c.bufs(m, e)) for e in (96, 192)}
        for trial in range(60):
            e, off = LAYOUTS[trial % len(LAYOUTS)]
            (wr, br), (wc, bc) = wsl[e]
            kind = (trial // len(LAYOUTS)) % 7
            x = c.make_x(m, kind)
            w = c.ws_w[trial % 4]
            b = (torch.randn((EXPERTS,), generator=gen, device=dev) * (0.05 if trial % 3 else 1.0)).float()
            if trial % 5 == 0:
                b[off:off + e] += 0.5  # many local pairs
            poison(wr, br)
            poison(wc, bc)
            r = c.run_ref(x, w, b, off, wr, br, e)
            cc = c.run_cand(x, w, b, off, wc, bc, e)
            cmp = c.compare(r, cc, wr, wc, br, bc)
            ok = all(cmp.values()) and int(c.counter.item()) == 0
            k = f"M{m}_E{e}_off{off}"
            per.setdefault(k, [0, 0])
            per[k][0] += ok
            per[k][1] += 1
            n_tot += 1
            n_ok += ok
            if not ok and len(fails) < 12:
                fails.append({"stage": "bitwise", "m": m, "E": e, "off": off, "kind": kind,
                              "bad": [kk for kk, v in cmp.items() if not v]})
    res["bitwise"] = {"ok": n_ok, "total": n_tot, "per_M_layout_ok_of": per}
    log("1 bitwise", n_ok, "/", n_tot)

    # ---------------- 2. repeated ----------------
    nbad = 0
    wsets = {(m, e): (c.bufs(m, e), c.bufs(m, e)) for m in range(1, 9) for e in (96, 192)}
    for i in range(1000):
        m = 1 + (i * 3) % 8
        e, off = LAYOUTS[i % len(LAYOUTS)]
        (wr, br), (wc, bc) = wsets[(m, e)]
        x = c.make_x(m, i % 7)
        b = (torch.randn((EXPERTS,), generator=gen, device=dev) * 0.05).float()
        poison(wr, br, i % 251)
        poison(wc, bc, i % 251)
        r = c.run_ref(x, c.ws_w[i % 4], b, off, wr, br, e)
        cc = c.run_cand(x, c.ws_w[i % 4], b, off, wc, bc, e)
        if not all(c.compare(r, cc, wr, wc, br, bc).values()) or int(c.counter.item()) != 0:
            nbad += 1
    res["repeated"] = {"calls": 1000, "bad": nbad, "counter_end": int(c.counter.item())}
    if nbad:
        fails.append({"stage": "repeated", "bad": nbad})
    log("2 repeated", res["repeated"])

    # ---------------- 3. graph + poison; skip-wait read once per process ----------------
    res["graph"] = []
    for m, e in ((1, 96), (2, 96), (3, 96), (4, 96), (8, 96), (1, 192), (2, 192), (4, 192)):
        gr = c.graph_check(m, e)
        res["graph"].append(gr)
        if gr["bad"]:
            fails.append({"stage": "graph", **gr})
    os.environ["TRTLLM_DSV4_FUSED_ROUTER_TEST_SKIP_WAIT"] = "1"  # after the first launch: must be ignored
    try:
        late = c.graph_check(1)
    finally:
        os.environ.pop("TRTLLM_DSV4_FUSED_ROUTER_TEST_SKIP_WAIT", None)
    res["skip_wait_set_after_first_launch_ignored"] = late
    if late["bad"]:
        fails.append({"stage": "skip_wait_not_cached", **late})
    env = dict(os.environ, TRTLLM_DSV4_FUSED_ROUTER_TEST_SKIP_WAIT="1")
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--negative-control"], env=env,
                       capture_output=True, text=True, timeout=900)
    neg = None
    for line in reversed(p.stdout.strip().splitlines()):
        if line.startswith("{"):
            neg = json.loads(line).get("negative_control")
            break
    res["negative_control_skip_wait_subprocess"] = {"returncode": p.returncode, "result": neg}
    if neg is None or neg["bad"] == 0:
        fails.append({"stage": "negative_control_not_detected", "rc": p.returncode,
                      "stderr_tail": p.stderr[-1500:]})
    c.counter.zero_()
    log("3 graph", res["graph"], "late-env", late["bad"], "negative control", neg)

    # ---------------- 3b. no hang beside an all-SM aux kernel ----------------
    import time as _time
    nsm = torch.cuda.get_device_properties(0).multi_processor_count
    aux = Aux(ncta=nsm, num_warps=32)
    side = torch.cuda.Stream()
    hang = {}
    for order in ("aux_first", "router_first"):
        m, e, off = 2, 96, 96
        (wr, br), (wc, bc) = c.bufs(m, e), c.bufs(m, e)
        x = c.make_x(m, 0)
        b = (torch.randn((EXPERTS,), generator=gen, device=dev) * 0.05).float()
        poison(wr, br)
        r = c.run_ref(x, c.ws_w[0], b, off, wr, br, e)
        torch.cuda.synchronize()
        t0 = _time.time()
        nb = 0
        if order == "aux_first":
            with torch.cuda.stream(side):
                aux.launch(20_000_000)
        for i in range(50):
            if order == "router_first" and i == 5:
                with torch.cuda.stream(side):
                    aux.launch(20_000_000)
            poison(wc, bc)
            cc = c.run_cand(x, c.ws_w[0], b, off, wc, bc, e)
            nb += int(not all(c.compare(r, cc, wr, wc, br, bc).values()))
        torch.cuda.synchronize()
        dt = _time.time() - t0
        hang[order] = {"seconds": round(dt, 4), "bad": nb, "counter": int(c.counter.item())}
        if nb or int(c.counter.item()) or dt > 5:
            fails.append({"stage": "no_hang", "order": order, **hang[order]})
    res["no_hang"] = hang
    log("3b no hang", hang)

    # ---------------- 4. leaf hand-off + 5. full decode graph ----------------
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import kernel_moe_mxfp8_mxfp4_fc12_fused as fc12m
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_mxfp8_q0_route_triton as rtm
    leaf, full = {}, {}
    for lname, (e_loc, inter, slot) in {"EP4_r0": (96, 2304, 0), "EP4_r3": (96, 2304, 288),
                                         "EP2TP2_ep0": (192, 1152, 0), "EP2TP2_ep1": (192, 1152, 192)}.items():
        state, dec = make_decode_state(e_loc, inter, slot)
        state.warmup(6)  # production: every decode shape JIT'd and its workspace made before any capture
        sync = fc12m.sync_workspace(torch.device("cuda", torch.cuda.current_device()))

        def bufviews(out, m, pairs):
            sf1_shape, n_q1, n_sf1, off_sf1, off_out, nbytes = state._buf_layout(m, pairs)
            buf = torch.empty(0, dtype=torch.uint8, device=dev).set_(out.untyped_storage(), 0, (nbytes,))
            assert out.data_ptr() == buf.data_ptr() + off_out
            return buf[:n_q1], buf[off_sf1:off_sf1 + n_sf1]

        for m in (1, 2, 3, 4, 8):
            x = c.make_x(m, 0)
            b = (torch.randn((EXPERTS,), generator=gen, device=dev) * 0.05).float()
            b[slot:slot + e_loc] += 0.5
            ids0, w0 = gate_main(router_main(x, c.ws_w[0]), b)
            bo = state.run(x, ids0, w0)
            bq1, bsf1 = (t.clone() for t in bufviews(bo, m, m * 6))
            base_out = bo.clone()
            del bo
            pre = state.preroute_buffers(x, 6)
            assert pre is not None
            ws, buf = pre
            buf.fill_(0x7F)
            ids, wts, _ = c.run_cand(x, c.ws_w[0], b, state.slot_start, ws, buf, state.num_experts)
            state.set_prerouted(x, ids, wts, ws, buf)
            fo = state.run(x, ids, wts)
            fused_out = fo.clone()
            fq1, fsf1 = bufviews(fo, m, m * 6)
            again = state.run(x, ids0, w0).clone()
            d = (fused_out.float() - base_out.float()).abs().max().item()
            nd = (again.float() - base_out.float()).abs().max().item()
            tol = max(2.0 * nd, base_out.float().abs().max().item() * 2.0 ** -6)
            key = f"{lname}_M{m}"
            local_pairs = int(((ids0 >= slot) & (ids0 < slot + e_loc)).sum())
            leaf[key] = {"q1_bitwise": same(fq1, bq1), "sf1_bitwise": same(fsf1, bsf1),
                         "used_stash_buf": fo.data_ptr() > buf.data_ptr() and fo.data_ptr() < buf.data_ptr()
                         + buf.numel(), "max_abs_diff": d, "main_repeat_diff": nd, "tol": tol,
                         "within_tol": d <= tol, "stash_cleared": state._prerouted is None,
                         "finite": bool(torch.isfinite(fused_out.float()).all()), "local_pairs": local_pairs}
            state.set_prerouted(c.make_x(m, 0), ids, wts, ws, buf)  # a stash for another x is ignored
            other = state.run(x, ids0, w0).clone()
            leaf[key]["foreign_stash_ignored"] = (other.float() - base_out.float()).abs().max().item() <= tol
            gm = DeepseekV4Gate(hidden_size=HIDDEN, num_experts=EXPERTS, top_k=6, n_group=1, topk_group=1,
                                routed_scaling_factor=SCALE, is_hashed=False, vocab_size=1024,
                                dtype=torch.bfloat16).to(dev)
            gm.weight.copy_(c.ws_w[0])
            gm.e_score_correction_bias.copy_(b)
            lg = gm.fused_forward(x, decode_state=state)
            took = state._prerouted is not None
            gi, gw = gm.routing_method.apply(lg)
            mod_out = state.run(x, gi, gw).clone()
            leaf[key]["module_stage2"] = took == (m <= max_m)
            leaf[key]["module_routing_bitwise"] = bool(torch.equal(gi, ids0)) and same(gw, w0)
            leaf[key]["module_out_ok"] = (mod_out.float() - base_out.float()).abs().max().item() <= tol
            leaf[key]["module_stash_cleared"] = state._prerouted is None and gm.routing_method.fused_routing is None
            v = leaf[key]
            if not (v["finite"] and v["stash_cleared"] and v["within_tol"] and v["q1_bitwise"] and v["sf1_bitwise"]
                    and v["used_stash_buf"] and v["foreign_stash_ignored"] and v["module_stage2"]
                    and v["module_routing_bitwise"] and v["module_out_ok"] and v["module_stash_cleared"]
                    and v["local_pairs"] > 0):
                fails.append({"stage": "leaf", "key": key, **v})

            # 5. full decode graph of this batch (module path), no JIT in capture, then beside an all-SM aux
            if m == 3:
                continue
            nl = 4
            gms, srcs, xs = [], [], []
            for li in range(nl):
                g_ = DeepseekV4Gate(hidden_size=HIDDEN, num_experts=EXPERTS, top_k=6, n_group=1, topk_group=1,
                                    routed_scaling_factor=SCALE, is_hashed=False, vocab_size=1024,
                                    dtype=torch.bfloat16).to(dev)
                g_.weight.copy_(c.ws_w[li % 4])
                bb = (torch.randn((EXPERTS,), generator=gen, device=dev) * 0.05).float()
                bb[slot:slot + e_loc] += 0.3
                g_.e_score_correction_bias.copy_(bb)
                gms.append(g_)
                srcs.append(c.make_x(m, 0))
                xs.append(srcs[-1].clone())
            hits = []
            orig_take = state._take_prerouted

            def take(*a):
                r_ = orig_take(*a)
                hits.append(r_ is not None)
                return r_
            state._take_prerouted = take

            def step():
                o = []
                for li in range(nl):
                    k1_standin(srcs[li], xs[li], SPIN_C1_NS)
                    lg_ = gms[li].fused_forward(xs[li], decode_state=state)
                    gi_, gw_ = gms[li].routing_method.apply(lg_)
                    o.append((gi_, gw_, state.run(xs[li], gi_, gw_)))
                return o

            jit0 = (len(fc12m._COMPILED), triton_cache_size(rtm._route_assign_decode_kernel),
                    triton_cache_size(rtm._mxfp8_q0_route_direct_kernel))
            # capture() runs step twice eagerly first; count JIT only across the capture itself
            s_ = torch.cuda.Stream()
            s_.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s_):
                step()
                torch.cuda.synchronize()
                jit1 = (len(fc12m._COMPILED), triton_cache_size(rtm._route_assign_decode_kernel),
                        triton_cache_size(rtm._mxfp8_q0_route_direct_kernel))
                hits.clear()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=s_):
                    gout = step()
            torch.cuda.current_stream().wait_stream(s_)
            torch.cuda.synchronize()
            jit2 = (len(fc12m._COMPILED), triton_cache_size(rtm._route_assign_decode_kernel),
                    triton_cache_size(rtm._mxfp8_q0_route_direct_kernel))
            state._take_prerouted = orig_take
            cap_hits = {"calls": len(hits), "taken": sum(hits)}
            nbad, worst = 0, 0.0
            for s in range(20):
                news = [c.make_x(m, s % 7) for _ in range(nl)]
                for li in range(nl):
                    srcs[li].copy_(news[li])
                    xs[li].fill_(float("nan"))
                torch.cuda.synchronize()
                g.replay()
                torch.cuda.synchronize()
                for li in range(nl):
                    gi_, gw_, o_ = gout[li]
                    ri, rw = gate_main(router_main(news[li], gms[li].weight), gms[li].e_score_correction_bias)
                    r1 = state.run(news[li], ri, rw).clone()
                    r2 = state.run(news[li], ri, rw).clone()
                    tl_ = max(2.0 * (r1.float() - r2.float()).abs().max().item(),
                              r1.float().abs().max().item() * 2.0 ** -6)
                    dd = (o_.float() - r1.float()).abs().max().item()
                    worst = max(worst, dd / max(tl_, 1e-30))
                    if not (torch.equal(gi_, ri) and same(gw_, rw) and dd <= tl_
                            and bool(torch.isfinite(o_.float()).all())):
                        nbad += 1
            torch.cuda.synchronize()
            cnt = [int(g_._fused_router_counter.item()) for g_ in gms]
            sync_nz = int((sync != 0).sum().item())
            # beside an all-SM aux kernel (aux first, then replays)
            t0 = _time.time()
            with torch.cuda.stream(side):
                aux.launch(20_000_000)
            for _ in range(5):
                g.replay()
            torch.cuda.synchronize()
            dt = _time.time() - t0
            sync_nz2 = int((sync != 0).sum().item())
            fk = f"{lname}_M{m}"
            full[fk] = {"capture_hits": cap_hits, "jit_before_capture": jit1, "jit_after_capture": jit2,
                        "jit_warm0": jit0, "replays": 20, "bad": nbad, "worst_d_over_tol": round(worst, 4),
                        "router_counters": cnt, "fc12_sync_nonzero": sync_nz, "aux_seconds": round(dt, 4),
                        "fc12_sync_nonzero_after_aux": sync_nz2}
            exp_taken = nl if m <= max_m else 0
            if (nbad or any(cnt) or sync_nz or sync_nz2 or jit1 != jit2 or dt > 5
                    or cap_hits["taken"] != exp_taken or cap_hits["calls"] != nl):
                fails.append({"stage": "full_graph", "key": fk, **full[fk]})
            del g, gout, gms
        del state
        torch.cuda.empty_cache()
        log("4/5", lname, "done")
    res["leaf"] = leaf
    res["full_graph"] = full
    log("4 leaf", leaf)
    log("5 full graph", full)
    print(json.dumps({"passed": not fails, "compared": "dsv4_fused_router_route (Stage 2) vs main's dsv3_router_gemm_op "
                      "+ gate_forward + mxfp8_q0_route_triton(expert_offset, zero_buf), every workspace buffer "
                      "bitwise; leaf (route skipped, main's fc12) vs plain run()", "results": res, "fails": fails}))


if __name__ == "__main__":
    with torch.no_grad():
        if "--negative-control" in sys.argv:
            negative_control()
        else:
            main()
