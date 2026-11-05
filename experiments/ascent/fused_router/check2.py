"""Stage 2 correctness of torch.ops.trtllm.dsv4_fused_router_route (idea 67) against main's chain
dsv3_router_gemm_op -> gate_forward -> FlashInfer mxfp8_q0_route_triton (decode branch: route_assign_decode_kernel
with expert_offset/zero_buf + _mxfp8_q0_route_direct_kernel), the route Sm12xW4a8DecodeState.run calls.

Both sides start from identically poisoned workspaces and buffers; every buffer FC1/FC2 read (offsets, token_map,
token_weights, q_out, scale_out, the zero-filled q1/sf1/out buffer) plus dst_rows/scale_dst_rows and the
ids/weights/logits are compared bitwise.
1. bitwise at M=1..8 x EP4 offsets 0/96/192/288 on random + adversarial x (big, tiny (amax floor), exact powers of
   two at the e4m3 max, zero rows, saturating values).
2. repeated: 1000 calls with changing inputs/offsets/M, each compared; counter 0 after each.
3. graph: 8 layers, K1 stand-in (PDL) before each fused route, x NaN-poisoned before each replay, 10 replays,
   every layer's buffers compared step by step (M=1/2/4/8). Negative control with the wait skipped must fail.
4. leaf: Sm12xW4a8DecodeState.run with set_prerouted (route skipped) vs plain run() -> FC2 output.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (EXPERTS, HIDDEN, SCALE, SPIN_C1_NS, capture, gate_main, k1_standin, log,  # noqa: E402
                    make_decode_state, router_main)

import torch  # noqa: E402

E_LOCAL = 96
FIELDS = ("offsets", "token_map", "token_weights", "dst_rows", "scale_dst_rows", "q_out", "scale_out")


def bits(t):
    t = t.contiguous()
    return t.view(torch.uint8) if t.element_size() == 1 else t.view(torch.int32)


def same(a, b):
    return a.shape == b.shape and bool(torch.equal(bits(a), bits(b)))


def poison(ws, buf, val=0x5A):
    for f in FIELDS:
        getattr(ws, f).view(torch.uint8).fill_(val)
    buf.fill_(val)


def main():
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.models.modeling_deepseekv4 import DeepseekV4Gate
    assert hasattr(torch.ops.trtllm, "dsv4_fused_router_route")
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_mxfp8_q0_route_triton import (
        make_mxfp8_q0_route_workspace, mxfp8_q0_route_triton)

    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(99)
    counter = torch.zeros(1, dtype=torch.int32, device=dev)
    ws_w = [(torch.randn((EXPERTS, HIDDEN), generator=gen, device=dev) * HIDDEN ** -0.5).to(torch.bfloat16)
            for _ in range(4)]
    res, fails = {}, []
    BUF_BYTES = 300 * 1024  # > the M=8 q1+sf1+out buffer

    # (local experts, slot_start): EP4 ranks 0..3, EP2xTP2 moe_ep_rank 0/1
    LAYOUTS = [(96, 0), (96, 96), (96, 192), (96, 288), (192, 0), (192, 192)]

    def bufs(m, e=E_LOCAL):
        ids_shape = torch.empty(m, 6, dtype=torch.int32, device=dev)
        x0 = torch.empty(m, HIDDEN, dtype=torch.bfloat16, device=dev)
        return make_mxfp8_q0_route_workspace(x0, ids_shape, e), torch.empty(BUF_BYTES, dtype=torch.uint8,
                                                                            device=dev)

    def run_ref(x, w, b, off, ws, buf, e=E_LOCAL):
        lg = router_main(x, w)
        ids, wts = gate_main(lg, b)
        mxfp8_q0_route_triton(x, ids, wts, e, workspace=ws, enable_pdl=True, expert_offset=off, zero_buf=buf)
        return ids, wts, lg

    def run_cand(x, w, b, off, ws, buf, e=E_LOCAL):
        return torch.ops.trtllm.dsv4_fused_router_route(
            x, w, b, counter, SCALE, off, e, ws.offsets, ws.token_map, ws.token_weights, ws.dst_rows,
            ws.scale_dst_rows, ws.q_out, ws.scale_out, buf)

    def compare(r, c, wr, wc, br, bc):
        out = {"ids": bool(torch.equal(r[0], c[0])), "weights": same(r[1], c[1]), "logits": same(r[2], c[2]),
               "buf": same(br, bc)}
        for f in FIELDS:
            out[f] = same(getattr(wr, f), getattr(wc, f))
        return out

    def make_x(m, kind):
        x = torch.randn((m, HIDDEN), generator=gen, device=dev)
        if kind == 1:
            x = x * 300.0
        elif kind == 2:
            x = x * 1e-6  # amax floor 1e-4
        elif kind == 3:
            x = torch.sign(x) * 56.0  # amax = 56 -> sf = 0.125 exactly (no round-up)
            x[:, ::7] *= 0.5
        elif kind == 4:
            x[0] = 0.0
        elif kind == 5:
            x = x * 448.0 / 2.5  # saturating e4m3 neighbourhood
        elif kind == 6:
            x[:, :128] = 1e4  # one hot tile
        return x.to(torch.bfloat16).contiguous()

    # ---------------- 1. bitwise ----------------
    n_ok = n_tot = 0
    per_layout = {f"E{e}_off{o}": 0 for e, o in LAYOUTS}
    for m in range(1, 9):
        wsl = {e: (bufs(m, e), bufs(m, e)) for e in (96, 192)}
        for trial in range(60):
            e, off = LAYOUTS[trial % len(LAYOUTS)]
            (wr, br), (wc, bc) = wsl[e]
            kind = (trial // len(LAYOUTS)) % 7
            x = make_x(m, kind)
            w = ws_w[trial % 4]
            b = (torch.randn((EXPERTS,), generator=gen, device=dev) * (0.05 if trial % 3 else 1.0)).float()
            poison(wr, br)
            poison(wc, bc)
            r = run_ref(x, w, b, off, wr, br, e)
            c = run_cand(x, w, b, off, wc, bc, e)
            per_layout[f"E{e}_off{off}"] += 1
            cmp = compare(r, c, wr, wc, br, bc)
            ok = all(cmp.values()) and int(counter.item()) == 0
            n_tot += 1
            n_ok += ok
            if not ok and len(fails) < 12:
                bad = {k: v for k, v in cmp.items() if not v}
                extra = {}
                if not cmp["q_out"]:
                    extra["q_out_bytes_diff"] = int((bits(wr.q_out) != bits(wc.q_out)).sum())
                if not cmp["scale_out"]:
                    extra["scale_words_diff"] = int((wr.scale_out != wc.scale_out).sum())
                fails.append({"stage": "bitwise", "m": m, "E": e, "off": off, "kind": kind, "bad": bad, **extra})
    res["bitwise"] = {"ok": n_ok, "total": n_tot, "cases_per_layout": per_layout}
    log("1 bitwise", res["bitwise"], fails[:4])

    # ---------------- 2. repeated ----------------
    nbad = 0
    wsets = {(m, e): (bufs(m, e), bufs(m, e)) for m in range(1, 9) for e in (96, 192)}
    for i in range(1000):
        m = 1 + (i * 3) % 8
        e, off = LAYOUTS[i % len(LAYOUTS)]
        (wr, br), (wc, bc) = wsets[(m, e)]
        x = make_x(m, i % 7)
        b = (torch.randn((EXPERTS,), generator=gen, device=dev) * 0.05).float()
        poison(wr, br, i % 251)
        poison(wc, bc, i % 251)
        r = run_ref(x, ws_w[i % 4], b, off, wr, br, e)
        c = run_cand(x, ws_w[i % 4], b, off, wc, bc, e)
        if not all(compare(r, c, wr, wc, br, bc).values()) or int(counter.item()) != 0:
            nbad += 1
    res["repeated"] = {"calls": 1000, "bad": nbad}
    if nbad:
        fails.append({"stage": "repeated", "bad": nbad})
    log("2 repeated", res["repeated"])

    # ---------------- 3. graph + poison ----------------
    def graph_check(m, skip_wait, e=E_LOCAL):
        nl = 8
        lw = [ws_w[i % 4] for i in range(nl)]
        lb = [(torch.randn((EXPERTS,), generator=gen, device=dev) * 0.05).float() for _ in range(nl)]
        offs = [((0, 96, 192, 288) if e == 96 else (0, 192))[i % (4 if e == 96 else 2)] for i in range(nl)]
        srcs = [make_x(m, 0) for _ in range(nl)]
        xs = [s.clone() for s in srcs]
        cws = [bufs(m, e) for _ in range(nl)]

        def step():
            o = []
            for li in range(nl):
                k1_standin(srcs[li], xs[li], SPIN_C1_NS)
                o.append(run_cand(xs[li], lw[li], lb[li], offs[li], cws[li][0], cws[li][1], e))
            return o

        if skip_wait:
            os.environ["TRTLLM_DSV4_FUSED_ROUTER_TEST_SKIP_WAIT"] = "1"
        try:
            g, gout = capture(step)
        finally:
            os.environ.pop("TRTLLM_DSV4_FUSED_ROUTER_TEST_SKIP_WAIT", None)
        wr, br = bufs(m, e)
        bad = 0
        for s in range(10):
            for li in range(nl):
                srcs[li].copy_(make_x(m, s % 7))
                xs[li].fill_(float("nan"))
                poison(*cws[li], 0x33)
            torch.cuda.synchronize()
            g.replay()
            torch.cuda.synchronize()
            if int(counter.item()) != 0:
                bad += 1
                counter.zero_()
                continue
            for li in range(nl):
                poison(wr, br, 0x33)
                r = run_ref(srcs[li], lw[li], lb[li], offs[li], wr, br, e)
                if not all(compare(r, gout[li], wr, cws[li][0], br, cws[li][1]).values()):
                    bad += 1
        del g
        return {"m": m, "E": e, "steps": 10, "layers": nl, "bad": bad}

    res["graph"] = []
    for m, e in ((1, 96), (2, 96), (4, 96), (8, 96), (1, 192), (4, 192), (8, 192)):
        gr = graph_check(m, False, e)
        res["graph"].append(gr)
        if gr["bad"]:
            fails.append({"stage": "graph", **gr})
    neg = graph_check(1, True)
    res["negative_control_skip_wait"] = neg
    if neg["bad"] == 0:
        fails.append({"stage": "negative_control_not_detected", **neg})
    counter.zero_()
    log("3 graph", res["graph"], "negative control bad", neg["bad"])

    # ---------------- 3b. no hang beside an all-SM aux kernel (Stage 2 variant) ----------------
    import time as _time

    from common import Aux
    nsm = torch.cuda.get_device_properties(0).multi_processor_count
    aux = Aux(ncta=nsm, num_warps=32)
    side = torch.cuda.Stream()
    hang = {}
    for order in ("aux_first", "router_first"):
        m, e, off = 4, 192, 192
        (wr, br), (wc, bc) = bufs(m, e), bufs(m, e)
        x = make_x(m, 0)
        b = (torch.randn((EXPERTS,), generator=gen, device=dev) * 0.05).float()
        poison(wr, br)
        r = run_ref(x, ws_w[0], b, off, wr, br, e)
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
            poison(wc, bc)  # same poison as the reference: unrouted tails must match too
            c = run_cand(x, ws_w[0], b, off, wc, bc, e)
            nb += int(not all(compare(r, c, wr, wc, br, bc).values()))
        torch.cuda.synchronize()
        dt = _time.time() - t0
        hang[order] = {"seconds": round(dt, 4), "bad": nb, "counter": int(counter.item())}
        if nb or int(counter.item()) or dt > 5:
            fails.append({"stage": "no_hang", "order": order, **hang[order]})
    res["no_hang"] = hang
    log("3b no hang", hang)

    # ---------------- 4. leaf hand-off ----------------
    leaf = {}
    for lname, (e_loc, inter, slot) in {"EP4_r0": (96, 2304, 0), "EP4_r3": (96, 2304, 288),
                                         "EP2TP2_ep0": (192, 1152, 0), "EP2TP2_ep1": (192, 1152, 192)}.items():
        state, dec = make_decode_state(e_loc, inter, slot)
        for m in (1, 2, 4, 8):
            x = make_x(m, 0)
            b = (torch.randn((EXPERTS,), generator=gen, device=dev) * 0.05).float()
            # bias the gate toward this rank's experts so the leaf has routed pairs at every offset
            b[slot:slot + e_loc] += 0.5
            ids0, w0 = gate_main(router_main(x, ws_w[0]), b)
            base_out = state.run(x, ids0, w0).clone()  # also creates the shape's shared workspace
            pre = state.preroute_buffers(x, 6)
            assert pre is not None
            ws, buf = pre
            buf.fill_(0x7F)
            ids, wts, _ = run_cand(x, ws_w[0], b, state.slot_start, ws, buf, state.num_experts)
            state.set_prerouted(x, ids, wts, ws, buf)
            fused_out = state.run(x, ids, wts).clone()
            again = state.run(x, ids0, w0).clone()  # no stash: plain route
            d = (fused_out.float() - base_out.float()).abs().max().item()
            # fc2_finalize scatter-adds the top-k rows with atomics: main's own output varies run to run (pitfall
            # 990). Tolerance = 2x main's run-to-run spread or 2^-6 of the output amplitude (a few bf16 ulps).
            nd = (again.float() - base_out.float()).abs().max().item()
            tol = max(2.0 * nd, base_out.float().abs().max().item() * 2.0 ** -6)
            key = f"{lname}_M{m}"
            local_pairs = int(((ids0 >= slot) & (ids0 < slot + e_loc)).sum())
            leaf[key] = {"bitwise_vs_plain": same(fused_out, base_out), "max_abs_diff": d, "main_repeat_diff": nd,
                         "tol": tol, "within_tol": d <= tol,
                         "plain_repeat_bitwise": same(again, base_out), "stash_cleared": state._prerouted is None,
                         "finite": bool(torch.isfinite(fused_out.float()).all()), "local_pairs": local_pairs}
            # a stash for another x is ignored
            state.set_prerouted(make_x(m, 0), ids, wts, ws, buf)
            other = state.run(x, ids0, w0).clone()
            leaf[key]["foreign_stash_ignored"] = (other.float() - base_out.float()).abs().max().item() <= tol
            # module path: DeepseekV4Gate.fused_forward(decode_state) -> routing_method.apply -> state.run
            gm = DeepseekV4Gate(hidden_size=HIDDEN, num_experts=EXPERTS, top_k=6, n_group=1, topk_group=1,
                                routed_scaling_factor=SCALE, is_hashed=False, vocab_size=1024,
                                dtype=torch.bfloat16).to(dev)
            gm.weight.copy_(ws_w[0])
            gm.e_score_correction_bias.copy_(b)
            lg = gm.fused_forward(x, decode_state=state)
            took_stage2 = state._prerouted is not None
            gi, gw = gm.routing_method.apply(lg)
            mod_out = state.run(x, gi, gw).clone()
            # fused_forward takes Stage 2 exactly at M <= TRTLLM_DSV4_FUSED_ROUTER_ROUTE_MAX_M (2), else Stage 1
            leaf[key]["module_stage2"] = took_stage2 == (m <= int(os.environ.get(
                "TRTLLM_DSV4_FUSED_ROUTER_ROUTE_MAX_M", "2")))
            leaf[key]["module_routing_bitwise"] = bool(torch.equal(gi, ids0)) and same(gw, w0)
            leaf[key]["module_out_ok"] = (mod_out.float() - base_out.float()).abs().max().item() <= tol
            leaf[key]["module_stash_cleared"] = state._prerouted is None and gm.routing_method.fused_routing is None
            del gm
        del state
        torch.cuda.empty_cache()
    res["leaf"] = leaf
    for m, v in leaf.items():
        if not (v["finite"] and v["stash_cleared"] and v["within_tol"]
                and v["foreign_stash_ignored"] and v["module_stage2"] and v["module_routing_bitwise"]
                and v["module_out_ok"] and v["module_stash_cleared"] and v["local_pairs"] > 0):
            fails.append({"stage": "leaf", "m": m, **v})
    log("4 leaf", leaf)
    print(json.dumps({"passed": not fails, "compared": "dsv4_fused_router_route vs dsv3_router_gemm_op + "
                      "gate_forward + mxfp8_q0_route_triton(expert_offset, zero_buf) on main, every workspace "
                      "buffer bitwise", "results": res, "fails": fails}))


if __name__ == "__main__":
    with torch.no_grad():
        main()
