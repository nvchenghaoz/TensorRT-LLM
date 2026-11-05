"""Correctness of torch.ops.trtllm.dsv4_fused_router (idea 67 Stage 1) against main's pair
dsv3_router_gemm_op (router_gemm_mma_kernel) + gate_forward (gate_forward_kernel<384,6,false,false>).

1. bitwise: logits, ids, weights at M=1..8 on random and adversarial inputs (large logits > 20, tied scores,
   zero rows, large bias), counter 0 after every call.
2. repeated: 1200 back-to-back calls with changing inputs and M (no sync), every call compared; counter 0 at sync
   points and at the end; then 100 calls with a sync + counter check after each.
3. graph: 8-layer graph with the K1 stand-in (PDL edge) before every fused router, M=1/2/4/8, 10 replays with new
   inputs each, compared step by step; x poisoned with NaN before every replay (the K1 stand-in writes the real x
   after a 4.9 us spin), so a read of x before griddepcontrol.wait shows up. Negative control: the same with
   TRTLLM_DSV4_FUSED_ROUTER_TEST_SKIP_WAIT=1 must FAIL.
4. no hang: fused router calls queued behind/alongside an all-SM aux kernel on a second stream.
5. integration: DeepseekV4Gate.fused_forward + DeepSeekV4MoeRoutingMethod.apply hand-off == gate()+apply();
   fallbacks (M=9, hashed gate, other tensor) keep main's path.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (EXPERTS, HIDDEN, SPIN_C1_NS, Aux, capture, fused, gate_main, has_fused, k1_standin,  # noqa: E402
                    log, router_main)

import torch  # noqa: E402


def bits(t):
    return t.contiguous().view(torch.int32)


def same(a, b):
    return a.shape == b.shape and bool(torch.equal(bits(a), bits(b)))


def ref(x, w, b):
    lg = router_main(x, w)
    ids, wts = gate_main(lg, b)
    return ids, wts, lg


def cmp(out, refv):
    ids, wts, lg = out
    rids, rwts, rlg = refv
    return {"logits": same(lg, rlg), "ids": bool(torch.equal(ids, rids)), "weights": same(wts, rwts)}


def main():
    import tensorrt_llm  # noqa: F401
    assert has_fused(), "torch.ops.trtllm.dsv4_fused_router missing (base tree?)"
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(777)
    counter = torch.zeros(1, dtype=torch.int32, device=dev)
    res = {}
    fails = []

    # ---------------- 1. bitwise ----------------
    ws = [(torch.randn((EXPERTS, HIDDEN), generator=gen, device=dev) * HIDDEN ** -0.5).to(torch.bfloat16)
          for _ in range(4)]
    n_ok = n_tot = 0
    for m in range(1, 9):
        for trial in range(160):
            kind = trial % 8
            w = ws[trial % 4]
            x = torch.randn((m, HIDDEN), generator=gen, device=dev)
            b = torch.randn((EXPERTS,), generator=gen, device=dev) * 0.05
            if kind == 1:
                x = x * 12.0  # logits well above the softplus threshold 20 for some experts
            elif kind == 2:
                b = torch.zeros(EXPERTS, device=dev)  # ties: duplicate rows of w below
                w = w.clone()
                w[EXPERTS // 2:] = w[: EXPERTS // 2]
            elif kind == 3:
                x[0] = 0.0  # a zero row: all logits 0 -> all scores equal, index tie-break
            elif kind == 4:
                b = torch.randn((EXPERTS,), generator=gen, device=dev) * 3.0
            elif kind == 5:
                x = x * 1e-3
            elif kind == 6:
                x = -x.abs() * 4.0
            x = x.to(torch.bfloat16).contiguous()
            b = b.float().contiguous()
            out = fused(x, w, b, counter)
            r = ref(x, w, b)
            c = cmp(out, r)
            cnt = int(counter.item())
            ok = all(c.values()) and cnt == 0
            n_tot += 1
            n_ok += ok
            if not ok and len(fails) < 10:
                d = (out[2] - r[2]).abs().max().item()
                fails.append({"stage": "bitwise", "m": m, "trial": trial, "kind": kind, **c, "counter": cnt,
                              "max_dlogit": d})
    res["bitwise"] = {"ok": n_ok, "total": n_tot}
    log("1 bitwise", res["bitwise"], fails[:3])

    # ---------------- 2. repeated back-to-back ----------------
    pool = []
    for i in range(64):
        m = 1 + i % 8
        x = torch.randn((m, HIDDEN), generator=gen, device=dev).to(torch.bfloat16)
        b = (torch.randn((EXPERTS,), generator=gen, device=dev) * 0.05).float()
        pool.append((x, ws[i % 4], b, ref(x, ws[i % 4], b)))
    torch.cuda.synchronize()
    outs = []
    for i in range(1200):
        x, w, b, _ = pool[(i * 7) % 64]
        outs.append(((i * 7) % 64, fused(x, w, b, counter)))
    torch.cuda.synchronize()
    cnt_end = int(counter.item())
    bad = 0
    for j, o in outs:
        if not all(cmp(o, pool[j][3]).values()):
            bad += 1
    sync_bad = 0
    for i in range(100):
        x, w, b, r = pool[(i * 5) % 64]
        o = fused(x, w, b, counter)
        torch.cuda.synchronize()
        if not all(cmp(o, r).values()) or int(counter.item()) != 0:
            sync_bad += 1
    res["repeated"] = {"calls": 1200, "mismatched": bad, "counter_end": cnt_end, "synced_calls": 100,
                       "synced_bad": sync_bad}
    if bad or cnt_end or sync_bad:
        fails.append({"stage": "repeated", **res["repeated"]})
    log("2 repeated", res["repeated"])

    # ---------------- 3. graph + PDL poison ----------------
    def graph_check(m, skip_wait):
        nl = 8
        lw = [ws[i % 4] for i in range(nl)]
        lb = [(torch.randn((EXPERTS,), generator=gen, device=dev) * 0.05).float() for _ in range(nl)]
        srcs = [torch.randn((m, HIDDEN), generator=gen, device=dev).to(torch.bfloat16) for _ in range(nl)]
        xs = [s.clone() for s in srcs]

        def step():
            o = []
            for li in range(nl):
                k1_standin(srcs[li], xs[li], SPIN_C1_NS)
                o.append(fused(xs[li], lw[li], lb[li], counter))
            return o

        if skip_wait:
            os.environ["TRTLLM_DSV4_FUSED_ROUTER_TEST_SKIP_WAIT"] = "1"
        try:
            g, gout = capture(step)
        finally:
            os.environ.pop("TRTLLM_DSV4_FUSED_ROUTER_TEST_SKIP_WAIT", None)
        nbad = 0
        steps = 10
        for s in range(steps):
            new = [torch.randn((m, HIDDEN), generator=gen, device=dev).to(torch.bfloat16) for _ in range(nl)]
            for li in range(nl):
                srcs[li].copy_(new[li])
                xs[li].fill_(float("nan"))  # poison: only K1 (after its spin) writes the real x
            torch.cuda.synchronize()
            g.replay()
            torch.cuda.synchronize()
            if int(counter.item()) != 0:
                nbad += 1
                continue
            for li in range(nl):
                r = ref(new[li], lw[li], lb[li])
                if not all(cmp(gout[li], r).values()):
                    nbad += 1
        del g
        return {"m": m, "steps": steps, "layers": nl, "bad": nbad}

    res["graph"] = []
    for m in (1, 2, 4, 8):
        gres = graph_check(m, False)
        res["graph"].append(gres)
        if gres["bad"]:
            fails.append({"stage": "graph", **gres})
    log("3 graph+poison", res["graph"])
    neg = graph_check(1, True)
    res["negative_control_skip_wait"] = neg
    neg_detected = neg["bad"] > 0
    log("3 negative control (skip wait) bad:", neg["bad"], "detected" if neg_detected else "NOT DETECTED")
    if not neg_detected:
        fails.append({"stage": "negative_control_not_detected", **neg})
    counter.zero_()
    torch.cuda.synchronize()

    # ---------------- 4. no hang beside an all-SM aux kernel ----------------
    nsm = torch.cuda.get_device_properties(0).multi_processor_count
    aux = Aux(ncta=nsm, num_warps=32)
    side = torch.cuda.Stream()
    hang = {}
    for order in ("aux_first", "router_first"):
        x, w, b, r = pool[3]
        t0 = time.time()
        if order == "aux_first":
            with torch.cuda.stream(side):
                aux.launch(20_000_000)
            outs = [fused(x, w, b, counter) for _ in range(50)]
        else:
            outs = [fused(x, w, b, counter) for _ in range(5)]
            with torch.cuda.stream(side):
                aux.launch(20_000_000)
            outs += [fused(x, w, b, counter) for _ in range(45)]
        torch.cuda.synchronize()
        dt = time.time() - t0
        nb = sum(0 if all(cmp(o, r).values()) else 1 for o in outs)
        hang[order] = {"seconds": round(dt, 4), "bad": nb, "counter": int(counter.item())}
        if nb or int(counter.item()) or dt > 5:
            fails.append({"stage": "no_hang", "order": order, **hang[order]})
    res["no_hang"] = hang
    log("4 no hang", hang)

    # ---------------- 5. integration: gate module + routing method ----------------
    from tensorrt_llm._torch.models.modeling_deepseekv4 import DeepseekV4Gate

    def mkgate(hashed):
        gm = DeepseekV4Gate(hidden_size=HIDDEN, num_experts=EXPERTS, top_k=6, n_group=1, topk_group=1,
                            routed_scaling_factor=1.5, is_hashed=hashed, vocab_size=1024,
                            dtype=torch.bfloat16).to(dev)
        with torch.no_grad():
            gm.weight.copy_(ws[0])
            if not hashed:
                gm.e_score_correction_bias.copy_(pool[0][2])
        return gm

    gm = mkgate(False)
    integ = {}
    for m in range(1, 10):
        x = torch.randn((m, HIDDEN), generator=gen, device=dev).to(torch.bfloat16)
        lg_f = gm.fused_forward(x)
        if m > 8:
            integ[f"m{m}_fallback"] = lg_f is None
            continue
        ids_f, w_f = gm.routing_method.apply(lg_f)
        lg_m = gm(x)
        ids_m, w_m = gm.routing_method.apply(lg_m)
        integ[f"m{m}"] = same(lg_f, lg_m) and bool(torch.equal(ids_f, ids_m)) and same(w_f, w_m)
        integ[f"m{m}_stash_cleared"] = gm.routing_method.fused_routing is None
    # a different tensor after fused_forward: routed from its own values
    x = torch.randn((2, HIDDEN), generator=gen, device=dev).to(torch.bfloat16)
    gm.fused_forward(x)
    other = gm(torch.randn((2, HIDDEN), generator=gen, device=dev).to(torch.bfloat16))
    ids_o, w_o = gm.routing_method.apply(other)
    gm.routing_method.fused_routing = None
    ids_r, w_r = gm.routing_method.apply(other)
    integ["other_tensor_routed_by_value"] = bool(torch.equal(ids_o, ids_r)) and same(w_o, w_r)
    try:
        gh = mkgate(True)
        integ["hashed_fallback"] = gh.fused_forward(x) is None
    except Exception as e:  # noqa: BLE001
        integ["hashed_fallback"] = f"construct failed: {e!r}"
    res["integration"] = integ
    if not all(v is True for v in integ.values()):
        fails.append({"stage": "integration", **{k: v for k, v in integ.items() if v is not True}})
    log("5 integration", integ)

    passed = not fails
    print(json.dumps({"passed": passed, "compared": "dsv4_fused_router vs dsv3_router_gemm_op + gate_forward "
                      "(main), bitwise logits/ids/weights", "results": res, "fails": fails}))


if __name__ == "__main__":
    with torch.no_grad():
        main()
