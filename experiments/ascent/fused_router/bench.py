"""Paired timing for idea 67 STAGE 1 (run_gpu_pair: BASE = main tree, CAND = branch tree, same script).

One GPU, one MoE rank: 40 distinct router weights [384,5120] bf16 + fp32 biases, 40 distinct sets of real mHC K1
inputs, the real Sm12xW4a8DecodeState.run (FlashInfer SM12x MoE decode chain). Per layer, one 40-layer CUDA graph:
  P1-REAL chain: real K1 (_mhc_lagged_layer_input_kernel, PDL, early trigger; writes x) -> router row -> run()
  BASE: router_gemm_mma_kernel -> gate_forward_kernel<384,6,false,false> -> run() (route_assign_decode + q0 + fc1 + fc2)
  CAND: dsv4FusedRouterKernel<M,false> (Stage 1: GEMV + gate, one launch) -> run() exactly as BASE
Cold L2: read-only 512 MB flush before every replay (pitfall 1356: no zero_() flush). Each sample = sum of REPS
individually-timed (flush, replay) pairs; events bracket the replay only.
Checks (same job output as the timings): after the warmup replays and after the last timed replay, the graph's OWN
output buffers (logits, ids, weights of every layer) are compared bitwise with main's dsv3_router_gemm_op +
gate_forward recomputed eagerly on the same xs (which K1 wrote inside the graph); every layer's arrival counter is
read and must be 0. A failed check makes the job exit non-zero.
Targeted: EP4 rank 0 (96 x 2304, slot 0) M=1/2/4 (main's layout, c1/c2/c4). Secondary (no loss beyond noise, not
targets): EP4 M=8, EP2xTP2 moe_ep_rank 1 (192 x 1152, slot 192) M=1/2/4, reported under "secondary".
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (HIDDEN, LAYERS, biases, capture, fused, gate_main, has_fused, log,  # noqa: E402
                    make_decode_state, router_main, router_weights)

import torch  # noqa: E402

WARMUP = 5
K1_P = 224  # fused_half_fma bigfuse 256 tactic's phase-2 threads (4 warps per K1 program)
# (layout, local experts, inter, slot_start, M, samples, reps per sample, targeted)
CASES = [("ep4", 96, 2304, 0, m, 60, 4, True) for m in (1, 2, 4)] + [("ep4", 96, 2304, 0, 8, 40, 2, False)] + \
        [("ep2tp2", 192, 1152, 192, m, 40, 2, False) for m in (1, 2, 4)]


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
    return t.contiguous().view(torch.int32)


def main():
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.modules.mhc.mhc_lagged_split import mhc_lagged_layer_input

    arm = "CAND" if has_fused() else "BASE"
    kname = {"BASE": "router_gemm_mma_kernel+gate_forward_kernel | route: _route_assign_decode_kernel+"
                     "_mxfp8_q0_route_direct_kernel | fc1_act_q1+fc2_finalize",
             "CAND": "dsv4FusedRouterKernel<M,false> (Stage 1) | route: _route_assign_decode_kernel+"
                     "_mxfp8_q0_route_direct_kernel | fc1_act_q1+fc2_finalize"}[arm]
    dev = "cuda"
    log("ARM", arm, torch.cuda.get_device_name(), kname)
    gen = torch.Generator(device=dev).manual_seed(4242)
    rws = router_weights(gen, LAYERS)
    bs = biases(gen, LAYERS)
    counters = [torch.zeros(1, dtype=torch.int32, device=dev) for _ in range(LAYERS)]
    flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device=dev).fill_(1.0)

    out, secondary, extra = {}, {}, {}
    all_ok = True
    state_key, state = None, None
    for lay, e, inter, slot, m, samples, reps, targeted in CASES:
        label = str(m) if lay == "ep4" else f"{lay}_{m}"
        if state_key != (e, inter, slot):
            state = None
            torch.cuda.empty_cache()
            state, _ = make_decode_state(e, inter, slot)
            state_key = (e, inter, slot)
            log(f"decode state E={state.num_experts} inter={state.inter} slot_start={state.slot_start}")

        def row(li, x):
            if arm == "CAND":
                ids, wts, lg = fused(x, rws[li], bs[li], counters[li])
            else:
                lg = router_main(x, rws[li])
                ids, wts = gate_main(lg, bs[li])
            state.run(x, ids, wts)
            return lg, ids, wts

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

        g, gout = capture(step)

        def check(tag):
            torch.cuda.synchronize()
            bad = {"logits": 0, "ids": 0, "weights": 0, "counter": 0}
            for li in range(LAYERS):
                lg, ids, wts = gout[li]
                rlg = router_main(xs[li], rws[li])
                rids, rwts = gate_main(rlg, bs[li])
                bad["logits"] += int(not torch.equal(bits(lg), bits(rlg)))
                bad["ids"] += int(not torch.equal(ids, rids))
                bad["weights"] += int(not torch.equal(bits(wts), bits(rwts)))
            torch.cuda.synchronize()
            cnt = [int(c.item()) for c in counters]
            bad["counter"] = sum(1 for c in cnt if c != 0)
            ok = not any(bad.values())
            log(f"CHECK {arm} {label} {tag}: layers={LAYERS} mismatches logits={bad['logits']} ids={bad['ids']} "
                f"weights={bad['weights']} counters_nonzero={bad['counter']} (max counter {max(cnt)}) "
                f"vs dsv3_router_gemm_op+gate_forward on the graph's xs -> {'OK' if ok else 'FAIL'}")
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
        all_ok = all_ok and ok_w and ok_e
        del g, gout
        v = sorted(ts)
        med = v[len(v) // 2]
        p10, p90 = v[int(0.1 * (len(v) - 1))], v[int(0.9 * (len(v) - 1))]
        noise = (p90 - p10) / med
        log(f"{arm} {label} E={e} inter={inter} slot={slot} {'TARGET' if targeted else 'secondary'} "
            f"kernels[K1=_mhc_lagged_layer_input_kernel -> {kname}] P1-REAL 40-layer graph "
            f"median {med:.4f} ms = {med * 1000.0 / LAYERS:.3f} us/layer [p10 {p10 * 1000.0 / LAYERS:.3f}, "
            f"p90 {p90 * 1000.0 / LAYERS:.3f}] noise {noise:.4%} ({samples} samples x {reps} replays)")
        rec = {"kernel_ms": med, "noise_floor": noise}
        (out if targeted else secondary)[label] = rec
        extra[label] = {"us_layer": med * 1000.0 / LAYERS, "p10_us_layer": p10 * 1000.0 / LAYERS,
                        "p90_us_layer": p90 * 1000.0 / LAYERS, "samples": samples, "reps": reps,
                        "check_after_warmup": bad_w, "check_after_timed": bad_e}
    print(json.dumps({"batch": out, "secondary": secondary, "output": {"class": "exact"}, "arm": arm,
                      "kernels": kname, "checks_passed": all_ok, "chains": extra}))
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    with torch.no_grad():
        main()
