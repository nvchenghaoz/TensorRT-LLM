"""Idea 313 paired timing: K1 fold into dsv4FusedRouterKernel (run_gpu_pair: BASE = main tree, CAND = branch).

One GPU, one MoE rank (EP4 rank 0: 96 x 2304, slot 0), 40 layers in one CUDA graph, cold L2 (512 MB read-only
flush before every replay; events bracket the replay only). Per layer, distinct router weights / biases / mHC K1
inputs, the real Sm12xW4a8DecodeState.run (FlashInfer SM12x decode chain):

  stand-in attention AR = router_prewait/pdl_ext.py producer (plain launch, 4 CTAs x 256, spins SPIN_NS from its
  start, then writes x_attn; triggers its dependents at its END or at its start = EARLY)
  MAIN chain: producer -> K1 (_mhc_lagged_layer_input_kernel, PDL) -> fused router (Stage 2 at M <= 2, else Stage 1)
              -> run()
  FOLD chain: producer -> [aux stream: K1 into the shared expert's own buffer] -> fused router with K1 folded in
              (dsv4_fused_router[_route]_k1fold; writes layer_input) -> run() -> join aux

The BASE arm (main tree, no fold op) times MAIN-END (main's production schedule) and MAIN-EARLY. The CAND arm times
all four, ABBA-interleaved per iteration; its reported kernel_ms is the branch's production schedule: FOLD-EARLY at
M <= K1_FOLD_MAX_TOKENS (default 2), MAIN-EARLY above it.
Checks (CAND, after warmup and after the last timed replay, every layer): FOLD graph's layer_input, logits, ids,
weights bitwise == eager K1 -> dsv4_fused_router on the same inputs; aux K1 copy bitwise == layer_input; MoE output
vs MAIN graph's within max(2x MAIN repeat spread, amax * 2^-6) (fc2 scatter atomics); counters 0.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "fused_router"))
sys.path.insert(0, os.path.join(HERE, "..", "router_prewait"))
from common import HIDDEN, LAYERS, TOPK, biases, log, make_decode_state, router_weights  # noqa: E402

import torch  # noqa: E402

import pdl_ext  # noqa: E402

K1_P = int(os.environ.get("K1FOLD_P", "224"))
# Stand-in attention-AR window per M = W_crit (top-lead ruling 17:5x PST 11-05 (A)/(C)): the attention AR's own
# duration on the last-arriving rank. No profile row is reduced here, so the ruling's fallback is declared:
# 3.0 us at M=1, 3.5 us at M=2 (M=4 takes M=2's value; the fold is off at M=4 by default). Override with
# K1FOLD_SPIN_NS (one value for every M) or K1FOLD_SPIN_NS_BY_M ("1:3000,2:3500,4:3500").
W_CRIT_NS = {1: 3000, 2: 3500, 4: 3500}
if os.environ.get("K1FOLD_SPIN_NS"):
    SPIN_BY_M = {m: int(os.environ["K1FOLD_SPIN_NS"]) for m in W_CRIT_NS}
else:
    SPIN_BY_M = dict(W_CRIT_NS)
    for kv in filter(None, os.environ.get("K1FOLD_SPIN_NS_BY_M", "").split(",")):
        k, v = kv.split(":")
        SPIN_BY_M[int(k)] = int(v)
EPS = 1e-6
SCALE = 1.5
WARMUP = 5
CASES = [(1, 40, 3), (2, 40, 3), (4, 30, 3)]  # (M, samples, reps)


def make_k1_inputs(m, gen, dev):
    x = torch.randn(m, HIDDEN, generator=gen, device=dev).to(torch.bfloat16)
    res = torch.randn(m, 4, HIDDEN, generator=gen, device=dev).to(torch.bfloat16)
    post = (2.0 * torch.rand(m, 4, generator=gen, device=dev)).float().contiguous()
    comb = torch.rand(m, 4, 4, generator=gen, device=dev)
    comb = (comb / comb.sum(-2, keepdim=True)).float().contiguous()
    pre = (torch.rand(m, 4, generator=gen, device=dev) + 1e-6).float().contiguous()
    nw = (1.0 + 0.1 * torch.randn(HIDDEN, generator=gen, device=dev)).to(torch.bfloat16)
    return x, res, post, comb, pre, nw


def bits(t):
    t = t.contiguous()
    return t.view(torch.uint8) if t.element_size() == 1 else (
        t.view(torch.int16) if t.element_size() == 2 else t.view(torch.int32))


def same(a, b):
    return a.shape == b.shape and bool(torch.equal(bits(a), bits(b)))


def capture(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        fn()
        fn()
        torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=s):
            out = fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    return g, out


def stats(v):
    v = sorted(v)
    med = v[len(v) // 2]
    p10, p90 = v[int(0.1 * (len(v) - 1))], v[int(0.9 * (len(v) - 1))]
    return med, p10, p90, (p90 - p10) / med


def main():
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.modules.mhc.mhc_lagged_split import mhc_lagged_layer_input

    ops = torch.ops.trtllm
    has_fold = hasattr(ops, "dsv4_fused_router_k1fold")
    arm = "CAND" if has_fold else "BASE"
    dev = "cuda"
    ext = pdl_ext.ext()
    log("ARM", arm, torch.cuda.get_device_name(), "K1 P", K1_P, "AR stand-in window (W_crit) ns by M", SPIN_BY_M)
    gen = torch.Generator(device=dev).manual_seed(31301)
    rws = router_weights(gen, LAYERS)
    bs = biases(gen, LAYERS)
    counters = [torch.zeros(1, dtype=torch.int32, device=dev) for _ in range(LAYERS)]
    flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device=dev).fill_(1.0)
    state, dec = make_decode_state(96, 2304, 0)
    aux = torch.cuda.Stream()
    batch, secondary, extra = {}, {}, {}
    all_ok = True
    for m, samples, reps in CASES:
        route = m <= 2
        SPIN_NS = SPIN_BY_M[m]
        x0 = torch.zeros(m, HIDDEN, dtype=torch.bfloat16, device=dev)
        i0 = (torch.arange(m * TOPK, device=dev, dtype=torch.int32) % 96).view(m, TOPK)
        state.run(x0, i0, torch.full((m, TOPK), 1.0 / TOPK, device=dev))
        k1in = [make_k1_inputs(m, gen, dev) for _ in range(LAYERS)]
        srcs = [t[0] for t in k1in]
        xattn = [torch.empty_like(s) for s in srcs]
        li_main = [torch.empty(m, HIDDEN, dtype=torch.bfloat16, device=dev) for _ in range(LAYERS)]
        li_fold = [torch.empty(m, HIDDEN, dtype=torch.bfloat16, device=dev) for _ in range(LAYERS)]
        li_aux_by_arm = {}
        empty = pdl_ext.empty()

        def k1args(li):
            _, res, post, comb, pre, nw = k1in[li]
            return xattn[li], res, post, comb, pre, nw

        def router(li, x):
            if route:
                pre_b = state.preroute_buffers(x, TOPK)
                ws, buf = pre_b
                ids, wts, lg = ops.dsv4_fused_router_route(
                    x, rws[li], bs[li], counters[li], SCALE, 0, 96, ws.offsets, ws.token_map, ws.token_weights,
                    ws.dst_rows, ws.scale_dst_rows, ws.q_out, ws.scale_out, buf)
                state.set_prerouted(x, ids, wts, ws, buf)
            else:
                ids, wts, lg = ops.dsv4_fused_router(x, rws[li], bs[li], counters[li], SCALE)
            return ids, wts, lg

        def router_fold(li, x):
            a = (*k1args(li), EPS, K1_P)
            if route:
                pre_b = state.preroute_buffers(x, TOPK)
                ws, buf = pre_b
                ids, wts, lg = ops.dsv4_fused_router_route_k1fold(
                    x, *a, rws[li], bs[li], counters[li], SCALE, 0, 96, ws.offsets, ws.token_map,
                    ws.token_weights, ws.dst_rows, ws.scale_dst_rows, ws.q_out, ws.scale_out, buf)
                state.set_prerouted(x, ids, wts, ws, buf)
            else:
                ids, wts, lg = ops.dsv4_fused_router_k1fold(x, *a, rws[li], bs[li], counters[li], SCALE)
            return ids, wts, lg

        def make_step(fold, early, name):
            li_aux = li_aux_by_arm.setdefault(name, [None] * LAYERS)

            def step():
                outs = []
                for li in range(LAYERS):
                    ext.producer(srcs[li], xattn[li], SPIN_NS, 4, 256, empty, early)
                    if fold:
                        ev0 = torch.cuda.Event()
                        ev0.record()
                        with torch.cuda.stream(aux):
                            ev0.wait()
                            li_aux[li] = mhc_lagged_layer_input(*k1args(li), EPS, K1_P, enable_pdl=True)
                            ev1 = torch.cuda.Event()
                            ev1.record()
                        x = li_fold[li]
                        ids, wts, lg = router_fold(li, x)
                    else:
                        x = li_main[li]
                        mhc_lagged_layer_input(*k1args(li), EPS, K1_P, enable_pdl=True, out=x)
                        ids, wts, lg = router(li, x)
                    o = state.run(x, ids, wts)
                    if fold:
                        ev1.wait()
                    outs.append((lg, ids, wts, o))
                return outs
            return step

        arms = [("MAIN-END", False, False), ("MAIN-EARLY", False, True)]
        if has_fold:
            arms += [("FOLD-EARLY", True, True), ("FOLD-END", True, False)]
        graphs = {}
        for name, fold, early in arms:
            graphs[name] = capture(make_step(fold, early, name))
        assert getattr(state, "_prerouted", None) is None

        def check(tag):
            if not has_fold:
                return True, {}
            torch.cuda.synchronize()
            _, gout_f = graphs["FOLD-EARLY"]
            _, gout_m = graphs["MAIN-END"]
            graphs["MAIN-END"][0].replay()
            torch.cuda.synchronize()
            m_out = [o.clone() for (_, _, _, o) in gout_m]
            graphs["MAIN-END"][0].replay()
            torch.cuda.synchronize()
            m_out2 = [o.clone() for (_, _, _, o) in gout_m]
            for xa in xattn:
                xa.fill_(float("nan"))
            for t in li_fold:
                t.fill_(float("nan"))
            graphs["FOLD-EARLY"][0].replay()
            torch.cuda.synchronize()
            li_aux = li_aux_by_arm["FOLD-EARLY"]
            bad = {"li": 0, "aux_li": 0, "logits": 0, "ids": 0, "weights": 0, "moe_out": 0, "counters": 0}
            maxd = maxtol = 0.0
            for li in range(LAYERS):
                ref = mhc_lagged_layer_input(srcs[li], *k1args(li)[1:], EPS, K1_P, enable_pdl=False)
                rids, rwts, rlg = ops.dsv4_fused_router(ref, rws[li], bs[li],
                                                        torch.zeros(1, dtype=torch.int32, device=dev), SCALE)
                lg, ids, wts, o = gout_f[li]
                bad["li"] += int(not same(li_fold[li], ref))
                bad["aux_li"] += int(not same(li_aux[li], ref))
                bad["logits"] += int(not same(lg, rlg))
                bad["ids"] += int(not torch.equal(ids, rids))
                bad["weights"] += int(not same(wts, rwts))
                spread = (m_out[li].float() - m_out2[li].float()).abs().max().item()
                tol = max(2.0 * spread, m_out[li].float().abs().max().item() * 2.0 ** -6)
                d = (o.float() - m_out[li].float()).abs().max().item()
                fin = bool(torch.isfinite(o.float()).all())
                maxd, maxtol = max(maxd, d), max(maxtol, tol)
                bad["moe_out"] += int(not (fin and d <= tol))
            bad["counters"] = sum(int(c.item() != 0) for c in counters)
            ok = not any(bad.values())
            log(f"CHECK M={m} {tag}: {bad} (moe max |d| {maxd:.3g} tol {maxtol:.3g}) -> {'OK' if ok else 'FAIL'}")
            return ok, bad

        def timed(name):
            flush.sum()
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record()
            graphs[name][0].replay()
            b.record()
            torch.cuda.synchronize()
            return a.elapsed_time(b)

        names = [a[0] for a in arms]
        for _ in range(WARMUP):
            for n in names:
                timed(n)
        ok_w, bad_w = check(f"after {WARMUP} warmup replays")
        ts = {n: [] for n in names}
        for it in range(samples):
            order = names if it % 2 == 0 else names[::-1]  # ABBA
            for n in order:
                ts[n].append(sum(timed(n) for _ in range(reps)) / reps)
        ok_e, bad_e = check(f"after last of {samples}x{reps} timed replays")
        all_ok = all_ok and ok_w and ok_e
        res = {}
        for n in names:
            med, p10, p90, noise = stats(ts[n])
            res[n] = {"kernel_ms": med, "noise_floor": noise, "us_layer": med * 1000.0 / LAYERS,
                      "p10_us_layer": p10 * 1000.0 / LAYERS, "p90_us_layer": p90 * 1000.0 / LAYERS}
            log(f"{arm} M={m} {n}: median {med:.4f} ms = {med * 1000.0 / LAYERS:.3f} us/layer "
                f"[p10 {p10 * 1000.0 / LAYERS:.3f}, p90 {p90 * 1000.0 / LAYERS:.3f}] noise {noise:.4%}")
        # Each tree's production schedule: main = MAIN-END; branch = FOLD-EARLY at M <= K1_FOLD_MAX_TOKENS (default 2),
        # else MAIN-EARLY (the attention AR still triggers early at M <= 4).
        if has_fold:
            from tensorrt_llm._torch.modules.mhc.mhc_lagged_split import K1_FOLD_MAX_TOKENS
            head = "FOLD-EARLY" if m <= K1_FOLD_MAX_TOKENS else "MAIN-EARLY"
        else:
            head = "MAIN-END"
        batch[str(m)] = {"kernel_ms": res[head]["kernel_ms"], "noise_floor": res[head]["noise_floor"],
                         "case": head}
        for n in names:
            secondary[f"{m}:{n}"] = res[n]
        if has_fold:
            extra[str(m)] = {
                "fold_early_minus_main_end_us_layer": res["FOLD-EARLY"]["us_layer"] - res["MAIN-END"]["us_layer"],
                "fold_end_minus_main_end_us_layer": res["FOLD-END"]["us_layer"] - res["MAIN-END"]["us_layer"],
                "main_early_minus_main_end_us_layer": res["MAIN-EARLY"]["us_layer"] - res["MAIN-END"]["us_layer"],
                "check_after_warmup": bad_w, "check_after_timed": bad_e}
            log(f"M={m} deltas us/layer: {extra[str(m)]}")
        for g in graphs.values():
            del g
        graphs.clear()
    print(json.dumps({"batch": batch, "secondary": secondary, "output": {"class": "exact"}, "arm": arm,
                      "checks_passed": all_ok, "k1_p": K1_P, "spin_ns_by_m": SPIN_BY_M,
                      "w_crit": "fallback 3.0 us M1 / 3.5 us M2 (ruling 17:5x PST 11-05)", "deltas": extra}))
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    with torch.no_grad():
        main()
