"""Paired timing for idea 261 (idea 67 STAGE 2 as its own landing; run_gpu_pair: BASE = main tree, CAND = branch).

One GPU, one MoE rank (EP4 rank 0: 96 x 2304, slot 0): 40 distinct router weights [384,5120] bf16 + fp32 biases,
40 distinct sets of real mHC K1 inputs, the real Sm12xW4a8DecodeState.run (FlashInfer SM12x decode chain; on this
main: route + persistent fc12 kernel). Per layer, one 40-layer CUDA graph:
  P1-REAL chain: real K1 (_mhc_lagged_layer_input_kernel, PDL, early trigger; writes x) -> router row -> run()
  BASE: dsv4FusedRouterKernel<M,false> (Stage 1) -> run(): _route_assign_decode_kernel + _mxfp8_q0_route_direct_kernel
        + fc12 fused
  CAND: at M <= ROUTE_MAX_M (2): dsv4FusedRouterKernel<M,true> (Stage 2: also route + q0) -> run() skips both route
        kernels -> fc12 fused. Above it: exactly BASE.
Cold L2: read-only 512 MB flush before every replay (pitfall 1356). Each sample = mean of REPS individually-timed
(flush, replay) pairs; events bracket the replay only.
Checks in the same job output as the timings, after the warmup replays and after the last timed replay:
  * the graph's OWN logits/ids/weights of every layer bitwise vs dsv3_router_gemm_op + gate_forward recomputed on the
    graph's xs (which K1 wrote inside the graph);
  * the graph's OWN MoE output of every layer vs an eager plain run() (main's route) on the same x/ids/weights,
    within max(2x main's own repeat spread, amax * 2^-6) (fc2's scatter atomics reorder; pitfall 990);
  * the shared route workspace after the replay (= the last layer's route) bitwise vs FlashInfer's route on that
    layer's x/ids/weights (offsets, token_map/token_weights routed rows, dst_rows, scale_dst_rows, q_out routed rows,
    scale_out);
  * every layer's router arrival counter and the fc12 sync workspace read and == 0;
  * CAND at M <= 2: every layer's run() consumed the stash (hand-off taken 40/40 per call).
Targeted: EP4 M=1/2 (c1/c2). Secondary (declared before the pair; same code in both arms): EP4 M=4.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (HIDDEN, LAYERS, TOPK, biases, capture, fused, gate_main, has_fused, log,  # noqa: E402
                    make_decode_state, router_main, router_weights)

import torch  # noqa: E402

WARMUP = 5
K1_P = 224
# (M, samples, reps, targeted)
CASES = [(1, 60, 4, True), (2, 60, 4, True), (4, 40, 2, False)]
FIELDS = ("offsets", "token_map", "token_weights", "dst_rows", "scale_dst_rows", "q_out", "scale_out")


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
    return t.view(torch.uint8) if t.element_size() == 1 else t.view(torch.int32)


def same(a, b):
    return a.shape == b.shape and bool(torch.equal(bits(a), bits(b)))


def main():
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.modules.mhc.mhc_lagged_split import mhc_lagged_layer_input
    from tensorrt_llm._torch.moe.fused_moe.cutlass import w4a8_mxfp4_mxfp8_sm12x_decode as dec
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_mxfp8_q0_route_triton import (
        make_mxfp8_q0_route_workspace, mxfp8_q0_route_triton)

    assert has_fused()
    stage2 = hasattr(dec.Sm12xW4a8DecodeState, "preroute_buffers")
    arm = "CAND" if stage2 else "BASE"
    max_m = 0
    if stage2:
        from tensorrt_llm._torch.models import modeling_deepseekv4 as mdl
        max_m = mdl._DSV4_FUSED_ROUTER_ROUTE_MAX_M
    dev = "cuda"
    try:
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.kernel_moe_mxfp8_mxfp4_fc12_fused import sync_workspace
        fc12_sync = sync_workspace(torch.device(dev, torch.cuda.current_device()))
    except ImportError:
        fc12_sync = None
    log("ARM", arm, torch.cuda.get_device_name(), "ROUTE_MAX_M", max_m, "fc12 fused", bool(dec._fused_op()))
    gen = torch.Generator(device=dev).manual_seed(4242)
    rws = router_weights(gen, LAYERS)
    bs = biases(gen, LAYERS)
    counters = [torch.zeros(1, dtype=torch.int32, device=dev) for _ in range(LAYERS)]
    flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device=dev).fill_(1.0)
    state, _ = make_decode_state(96, 2304, 0)
    hits = []
    if stage2:
        orig_take = state._take_prerouted

        def take(*a):
            r = orig_take(*a)
            hits.append(r is not None)
            return r
        state._take_prerouted = take

    out, secondary, extra = {}, {}, {}
    all_ok = True
    for m, samples, reps, targeted in CASES:
        label = str(m)
        use_s2 = stage2 and m <= max_m
        kname = ("dsv4FusedRouterKernel<M,true> (Stage 2: GEMV+gate+route+q0) | fc12 fused" if use_s2 else
                 "dsv4FusedRouterKernel<M,false> (Stage 1) | _route_assign_decode_kernel+_mxfp8_q0_route_direct_kernel"
                 " | fc12 fused")
        # create this shape's shared route workspace (production: warmup) before capture
        x0 = torch.zeros(m, HIDDEN, dtype=torch.bfloat16, device=dev)
        i0 = (torch.arange(m * TOPK, device=dev, dtype=torch.int32) % 96).view(m, TOPK)
        state.run(x0, i0, torch.full((m, TOPK), 1.0 / TOPK, device=dev))

        def row(li, x):
            if use_s2:
                pre = state.preroute_buffers(x, TOPK)
                assert pre is not None
                ws, buf = pre
                ids, wts, lg = torch.ops.trtllm.dsv4_fused_router_route(
                    x, rws[li], bs[li], counters[li], 1.5, int(state.slot_start), int(state.num_experts),
                    ws.offsets, ws.token_map, ws.token_weights, ws.dst_rows, ws.scale_dst_rows, ws.q_out,
                    ws.scale_out, buf)
                state.set_prerouted(x, ids, wts, ws, buf)
            else:
                ids, wts, lg = fused(x, rws[li], bs[li], counters[li])
            o = state.run(x, ids, wts)
            return lg, ids, wts, o

        k1in = [make_k1_inputs(m, gen, dev) for _ in range(LAYERS)]
        xs = [torch.empty(m, HIDDEN, dtype=torch.bfloat16, device=dev) for _ in range(LAYERS)]

        def k1(li):
            xp, res, post, comb, pre, nw = k1in[li]
            mhc_lagged_layer_input(xp, res, post, comb, pre, nw, 1e-6, K1_P, enable_pdl=True, out=xs[li])

        def step():
            outs = []
            for li in range(LAYERS):
                k1(li)
                outs.append(row(li, xs[li]))
            return outs

        hits.clear()
        g, gout = capture(step)
        handoff = {"calls": len(hits), "taken": sum(hits)}
        assert state._prerouted is None
        ref_ws = make_mxfp8_q0_route_workspace(xs[0], torch.empty(m, TOPK, dtype=torch.int32, device=dev), 96)

        def check(tag):
            torch.cuda.synchronize()
            bad = {"logits": 0, "ids": 0, "weights": 0, "moe_out": 0, "counter": 0, "fc12_sync": 0, "ws": []}
            maxd, maxtol = 0.0, 0.0
            # workspace = last layer's route (snapshot before any eager call overwrites it)
            ws = dec._WORKSPACES[(xs[0].device, m, HIDDEN, TOPK, 96)]
            snap = {f: getattr(ws, f).clone() for f in FIELDS}
            li = LAYERS - 1
            lg, ids, wts, _ = gout[li]
            zb = torch.empty(64, dtype=torch.uint8, device=dev)
            mxfp8_q0_route_triton(xs[li], ids, wts, 96, workspace=ref_ws, enable_pdl=True, expert_offset=0,
                                  zero_buf=zb)
            n = int(ref_ws.offsets[96].item())
            for f in FIELDS:
                a, b = snap[f], getattr(ref_ws, f)
                if f in ("token_map", "token_weights", "q_out"):
                    a, b = a[:n], b[:n]
                if not same(a, b):
                    bad["ws"].append(f)
            cnt = [int(c.item()) for c in counters]
            bad["counter"] = sum(1 for c in cnt if c != 0)
            if fc12_sync is not None:
                bad["fc12_sync"] = int((fc12_sync != 0).sum().item())
            for li in range(LAYERS):
                lg, ids, wts, o = gout[li]
                rlg = router_main(xs[li], rws[li])
                rids, rwts = gate_main(rlg, bs[li])
                bad["logits"] += int(not same(lg, rlg))
                bad["ids"] += int(not torch.equal(ids, rids))
                bad["weights"] += int(not same(wts, rwts))
                r1 = state.run(xs[li], rids, rwts).clone()
                r2 = state.run(xs[li], rids, rwts).clone()
                spread = (r1.float() - r2.float()).abs().max().item()
                tol = max(2.0 * spread, r1.float().abs().max().item() * 2.0 ** -6)
                d = (o.float() - r1.float()).abs().max().item()
                fin = bool(torch.isfinite(o.float()).all())
                maxd, maxtol = max(maxd, d), max(maxtol, tol)
                bad["moe_out"] += int(not (fin and d <= tol))
            torch.cuda.synchronize()
            ok = not any(v for v in bad.values())
            log(f"CHECK {arm} M={label} {tag}: layers={LAYERS} mismatches logits={bad['logits']} ids={bad['ids']} "
                f"weights={bad['weights']} moe_out={bad['moe_out']} (max |d| {maxd:.3g}, tol {maxtol:.3g}) "
                f"ws_fields_bad={bad['ws']} (routed rows {n}) counters_nonzero={bad['counter']} "
                f"fc12_sync_nonzero={bad['fc12_sync']} hand-off {handoff} -> {'OK' if ok else 'FAIL'}")
            return ok, bad

        def replay_timed():
            flush.sum()
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record()
            g.replay()
            b.record()
            torch.cuda.synchronize()
            return a.elapsed_time(b)

        for _ in range(WARMUP):
            replay_timed()
        ok_w, bad_w = check(f"after {WARMUP} warmup replays")
        ts = []
        for _ in range(samples):
            ts.append(sum(replay_timed() for _ in range(reps)) / reps)
        ok_e, bad_e = check(f"after last of {samples}x{reps} timed replays")
        ok_h = handoff["taken"] == (handoff["calls"] if use_s2 else 0)
        all_ok = all_ok and ok_w and ok_e and ok_h
        del g, gout
        v = sorted(ts)
        med = v[len(v) // 2]
        p10, p90 = v[int(0.1 * (len(v) - 1))], v[int(0.9 * (len(v) - 1))]
        noise = (p90 - p10) / med
        log(f"{arm} M={label} EP4 r0 {'TARGET' if targeted else 'secondary'} kernels[K1 -> {kname}] 40-layer graph "
            f"median {med:.4f} ms = {med * 1000.0 / LAYERS:.3f} us/layer [p10 {p10 * 1000.0 / LAYERS:.3f}, "
            f"p90 {p90 * 1000.0 / LAYERS:.3f}] noise {noise:.4%} ({samples} samples x {reps} replays)")
        rec = {"kernel_ms": med, "noise_floor": noise}
        (out if targeted else secondary)[label] = rec
        extra[label] = {"us_layer": med * 1000.0 / LAYERS, "p10_us_layer": p10 * 1000.0 / LAYERS,
                        "p90_us_layer": p90 * 1000.0 / LAYERS, "samples": samples, "reps": reps, "kernels": kname,
                        "handoff": handoff, "check_after_warmup": bad_w, "check_after_timed": bad_e}
    print(json.dumps({"batch": out, "secondary": secondary, "output": {"class": "exact"}, "arm": arm,
                      "route_max_m": max_m, "checks_passed": all_ok, "chains": extra}))
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    with torch.no_grad():
        main()
