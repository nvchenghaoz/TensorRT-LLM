"""router_gemm_mma_kernel<M,384,5120,8,640> weight loads before the PDL wait (idea 223): bench / check.

Shapes: DeepSeek-V4.1-Flash router, x [M,5120] bf16 x W [384,5120]^T -> fp32 [M,384], replicated on every EP4 rank.
40 distinct DRAM-resident weights (157 MB > 128 MB L2), as in the 40-layer decode step.

bench ARM [spin_ns grid traffic_mb trigger]:
  per M in (1,2,4,8), CUDA graphs of 40 layers, replays interleaved, each after a READ-ONLY 512 MB flush (sum):
   P0  R  = 40 x (router(x_l) -> gate_forward)      G  = 40 x gate_forward(scores_l)
   P1  PR = 40 x (producer_l -> router(dst_l) -> gate_forward)   PG = 40 x (producer_l -> gate_forward(scores_l))
       producer = stand-in for the in-server predecessor: plain launch, `grid` CTAs, triggers launch_dependents at
       its start (trigger=1) or end (0), streams `traffic_mb` MB of a read-only DRAM buffer, runs >= spin_ns, then
       writes the router input dst_l (copy of x_l).
  per-layer router cost = median over iterations of (R_i - G_i)/40 (P0) and (PR_i - PG_i)/40 (P1);
  noise = (p90-p10)/median of those per-iteration values. Also prints output hashes (eager M=1..16 x 40 weights;
  P1 graph outputs) that must be equal in the BASE and CAND arms.
check:
  (1) eager numerics M=1..16 vs FP64 and vs cuBLAS (trtllm::cublas_mm), deterministic repeats;
  (2) overlap-live: producer (24 CTAs, triggers at once, spins 5 us, then writes the router input that was
      NaN-poisoned) -> router, eager and in a CUDA graph, M=1..16 x 40 weights, bitwise vs the no-producer
      reference; negative control: a PDL consumer without griddepcontrol.wait after the same producer must read
      the poison; the same consumer with the wait must not;
  (3) repeated sequence (warmup 5 + 20 iterations) at M=1,2,4,8 under a graph of 40 x (producer -> router ->
      gate_forward): inputs re-created (fixed seed per reset) before every replay, router input poisoned,
      outputs poisoned; every iteration compared bitwise with the eager no-producer reference (router logits and
      gate outputs); weights/bias checked pristine; two graph instances (CAND vs CAND); per-iteration hash.
"""
import hashlib
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pdl_ext  # noqa: E402

HIDDEN, EXPERTS, LAYERS, TOPK, SCALE = 5120, 384, 40, 6, 1.5
SEED = 22300
MS = (1, 2, 4, 8)
WARMUP, ITERS = 10, 60


def log(*a):
    print(*a, flush=True)


EMPTY = {}


def empty_i():
    if "i" not in EMPTY:
        EMPTY["i"] = torch.empty(0, dtype=torch.int32, device="cuda")
    return EMPTY["i"]


def router(x, w):
    return torch.ops.trtllm.dsv3_router_gemm_op(x, w.t(), bias=None, out_dtype=torch.float32)


def gate(scores, bias, ow, oi):
    torch.ops.trtllm.gate_forward(scores, bias, empty_i(), empty_i(), ow, oi, TOPK, SCALE, False, None, None)


def kernel_names(fn):
    from torch.profiler import ProfilerActivity, profile
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sorted({e.name[:70] for e in prof.events() if e.device_type.name == "CUDA"})


def capture(step):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        step()
        step()
        torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=s):
            res = step()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    return g, res


def make_weights(gen):
    ws = [(torch.randn((EXPERTS, HIDDEN), generator=gen, device="cuda") * HIDDEN ** -0.5).to(torch.bfloat16)
          for _ in range(LAYERS)]
    bias = (torch.randn((EXPERTS,), generator=gen, device="cuda") * 0.05).float()
    return ws, bias


def make_x(gen, m):
    return [torch.randn((m, HIDDEN), generator=gen, device="cuda").to(torch.bfloat16) for _ in range(LAYERS)]


def stats(v):
    v = sorted(v)
    med = v[len(v) // 2]
    return med, (v[int(0.9 * (len(v) - 1))] - v[int(0.1 * (len(v) - 1))]) / abs(med)


def timed(g, flush):
    flush.sum()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    g.replay()
    e1.record()
    e1.synchronize()
    return e0.elapsed_time(e1) * 1000.0


def h_update(h, t):
    h.update(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())


def eager_ref_hash(ws):
    gen = torch.Generator(device="cuda").manual_seed(SEED + 5)
    h = hashlib.sha256()
    for m in range(1, 17):
        for w in ws:
            x = torch.randn((m, HIDDEN), generator=gen, device="cuda").to(torch.bfloat16)
            h_update(h, router(x, w))
    return h.hexdigest()[:16]


class Prod:
    def __init__(self, spin_ns, grid, traffic_mb, trigger, block=256):
        self.spin_ns, self.grid, self.trigger, self.block = int(spin_ns), int(grid), bool(trigger), int(block)
        n = int(traffic_mb * 1024 * 1024) // 16 * 16
        self.n = n
        # Distinct read-only slice per layer (as a DRAM-heavy predecessor would stream new bytes each layer).
        self.buf = torch.ones(n * LAYERS // 4, dtype=torch.float32, device="cuda") if n else None

    def __call__(self, li, src, dst):
        if self.n:
            sb = self.buf.view(torch.uint8)[li * self.n:(li + 1) * self.n]
        else:
            sb = pdl_ext.empty()
        pdl_ext.ext().producer(src, dst, self.spin_ns, self.grid, self.block, sb, self.trigger)


# Stage 0 (job 1065, server trace job 1021, ranks 0-3): the router's predecessor is _mhc_lagged_layer_input_kernel,
# grid [M] x 128 threads (1 CTA per token), not DRAM-bound; the router starts ~0.7 us after it at c1 (early trigger)
# and waits p50 4.1-4.6 us (c1) / 2.6-2.9 us (c4) for it to end. Stand-in: grid M x 128, trigger at start, spin so
# that pred_end - router_start matches those windows.
SERVER_SPIN_NS = {1: 4800, 2: 4800, 4: 3500, 8: 3500}


def server_prod(m):
    return Prod(SERVER_SPIN_NS[m], m, 0.0, 1, block=128)


def bench():
    arm = sys.argv[2] if len(sys.argv) > 2 else "ARM?"
    server = len(sys.argv) <= 3 or sys.argv[3] == "server"
    if not server:
        spin_ns = float(sys.argv[3])
        grid = int(sys.argv[4]) if len(sys.argv) > 4 else 24
        traffic_mb = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
        trigger = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    import tensorrt_llm  # noqa: F401
    ext = pdl_ext.ext()  # noqa: F841  build before timing
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    ws, bias = make_weights(gen)
    flush = torch.ones(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    batch = {}
    p1_hash = hashlib.sha256()
    for m in MS:
        prod = server_prod(m) if server else Prod(spin_ns, grid, traffic_mb, trigger)
        log(f"{arm} M={m} producer stand-in: grid {prod.grid} x {prod.block}, spin {prod.spin_ns} ns, traffic "
            f"{prod.n / 2**20:.2f} MB/layer, trigger {'start' if prod.trigger else 'end'}")
        xs = make_x(gen, m)
        dsts = [torch.empty_like(x) for x in xs]
        scs = [router(x, w) for x, w in zip(xs, ws)]
        ows = [torch.empty((m, TOPK), dtype=torch.float32, device="cuda") for _ in range(LAYERS)]
        ois = [torch.empty((m, TOPK), dtype=torch.int32, device="cuda") for _ in range(LAYERS)]
        names = kernel_names(lambda: router(xs[0], ws[0]))

        def r_chain():
            outs = []
            for li in range(LAYERS):
                o = router(xs[li], ws[li])
                gate(o, bias, ows[li], ois[li])
                outs.append(o)
            return outs

        def g_chain():
            for li in range(LAYERS):
                gate(scs[li], bias, ows[li], ois[li])

        def pr_chain():
            outs = []
            for li in range(LAYERS):
                prod(li, xs[li], dsts[li])
                o = router(dsts[li], ws[li])
                gate(o, bias, ows[li], ois[li])
                outs.append(o)
            return outs

        def pg_chain():
            for li in range(LAYERS):
                prod(li, xs[li], dsts[li])
                gate(scs[li], bias, ows[li], ois[li])

        gr, _ = capture(r_chain)
        gg, _ = capture(g_chain)
        gpr, pr_outs = capture(pr_chain)
        gpg, _ = capture(pg_chain)
        p0, p1, raw = [], [], {"R": [], "G": [], "PR": [], "PG": []}
        for it in range(WARMUP + ITERS):
            tr = timed(gr, flush)
            tg = timed(gg, flush)
            tpr = timed(gpr, flush)
            tpg = timed(gpg, flush)
            if it >= WARMUP:
                p0.append((tr - tg) / LAYERS)
                p1.append((tpr - tpg) / LAYERS)
                for k, v in (("R", tr), ("G", tg), ("PR", tpr), ("PG", tpg)):
                    raw[k].append(v / LAYERS)
        torch.cuda.synchronize()
        for o in pr_outs:
            h_update(p1_hash, o)
        m0, n0 = stats(p0)
        m1, n1 = stats(p1)
        rs = {k: round(stats(v)[0], 3) for k, v in raw.items()}
        log(f"{arm} M={m} P0 router-in-chain {m0:.3f} us/layer (noise {n0:.3f}) | P1 router-after-producer "
            f"{m1:.3f} us/layer (noise {n1:.3f}) | per-layer chain medians {rs} | {arm} kernels={names}")
        batch[str(m)] = {"kernel_ms": m1 / 1000.0, "noise_floor": n1, "per_layer_us": round(m1, 3), "case": "P1"}
        batch[f"p0_{m}"] = {"kernel_ms": m0 / 1000.0, "noise_floor": n0, "per_layer_us": round(m0, 3), "case": "P0"}
        batch[str(m)]["chain_medians_us_per_layer"] = rs
        batch[str(m)]["kernels"] = names
        del gr, gg, gpr, gpg
    eh = eager_ref_hash(ws)
    ph = p1_hash.hexdigest()[:16]
    log(f"{arm} output hashes: eager M=1..16 x 40 weights {eh}; P1 graph outputs {ph}")
    print(json.dumps({"arm": arm, "batch": batch, "output": {"class": "exact", "eager_hash": eh, "p1_hash": ph},
                      "producer": "server" if server else {"spin_ns": spin_ns, "grid": grid,
                                                           "traffic_mb": traffic_mb, "trigger": trigger}}))


def check():
    import tensorrt_llm  # noqa: F401
    ext = pdl_ext.ext()
    failures, report = [], {}
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    ws, bias = make_weights(gen)
    ws_p = [w.clone() for w in ws]
    bias_p = bias.clone()
    prod = Prod(5000, 24, 0.0, 1)

    # (1) numerics
    num = {}
    for m in range(1, 17):
        x = torch.randn((m, HIDDEN), generator=gen, device="cuda").to(torch.bfloat16)
        emax = cmax = 0.0
        det = True
        for w in ws[:8]:
            o = router(x, w)
            det = det and all(torch.equal(o, router(x, w)) for _ in range(3))
            ref = x.double() @ w.double().t()
            cub = torch.ops.trtllm.cublas_mm(x, w.t(), None, torch.float32)
            emax = max(emax, (o.double() - ref).abs().max().item())
            cmax = max(cmax, (o.double() - cub.double()).abs().max().item())
            if not (torch.isfinite(o).all() and o.shape == (m, EXPERTS)):
                failures.append(f"numerics M={m} non-finite/shape")
        num[m] = {"max_abs_vs_fp64": emax, "max_abs_vs_cublas": cmax, "deterministic": det}
        if not det or emax > 1e-3 or cmax > 1e-3:
            failures.append(f"numerics M={m} {num[m]}")
    report["numerics"] = num
    log("numerics", json.dumps(num))

    # (2) overlap-live, eager + graph, M=1..16 x 40 weights
    live = {}
    for m in range(1, 17):
        xs = make_x(gen, m)
        refs = [router(x, w) for x, w in zip(xs, ws)]
        dsts = [torch.empty_like(x) for x in xs]
        bad_e = 0
        for li in range(LAYERS):
            dsts[li].fill_(float("nan"))
            prod(li, xs[li], dsts[li])
            o = router(dsts[li], ws[li])
            torch.cuda.synchronize()
            bad_e += int(not torch.equal(o, refs[li]))

        def step():
            outs = []
            for li in range(LAYERS):
                prod(li, xs[li], dsts[li])
                outs.append(router(dsts[li], ws[li]))
            return outs

        g, gouts = capture(step)
        bad_g = 0
        for r in range(3):
            for d in dsts:
                d.fill_(float("nan"))
            for o in gouts:
                o.fill_(float("nan"))
            g.replay()
            torch.cuda.synchronize()
            bad_g += sum(int(not torch.equal(o, rf)) for o, rf in zip(gouts, refs))
        del g
        # negative / positive control with the same producer
        ctrl_bad = ctrl_wait_bad = 0
        for li in range(LAYERS):
            out = torch.empty_like(xs[li])
            dsts[li].fill_(float("nan"))
            prod(li, xs[li], dsts[li])
            ext.consumer_copy(dsts[li], out, False)
            torch.cuda.synchronize()
            ctrl_bad += int(not torch.equal(out, xs[li]))
            dsts[li].fill_(float("nan"))
            prod(li, xs[li], dsts[li])
            ext.consumer_copy(dsts[li], out, True)
            torch.cuda.synchronize()
            ctrl_wait_bad += int(not torch.equal(out, xs[li]))
        live[m] = {"eager_mismatch_layers": bad_e, "graph_mismatch_layer_replays": bad_g,
                   "nowait_control_wrong_layers": ctrl_bad, "wait_control_wrong_layers": ctrl_wait_bad}
        log(f"overlap-live M={m}", json.dumps(live[m]))
        if bad_e or bad_g or ctrl_wait_bad:
            failures.append(f"overlap-live M={m} {live[m]}")
        if ctrl_bad == 0:
            failures.append(f"overlap-live M={m}: no-wait negative control did not fail (overlap not live)")
    report["overlap_live"] = live

    # (3) repeated sequence: warmup 5 + 20, graph of 40 x (producer -> router -> gate), fresh inputs per reset
    seq = {}
    rw, ri = 5, 20
    seq_hash = hashlib.sha256()
    for m in MS:
        srcs = [torch.zeros((m, HIDDEN), dtype=torch.bfloat16, device="cuda") for _ in range(LAYERS)]
        dsts = [torch.empty_like(s) for s in srcs]
        ows = [torch.empty((m, TOPK), dtype=torch.float32, device="cuda") for _ in range(LAYERS)]
        ois = [torch.empty((m, TOPK), dtype=torch.int32, device="cuda") for _ in range(LAYERS)]

        def step():
            outs = []
            for li in range(LAYERS):
                prod(li, srcs[li], dsts[li])
                o = router(dsts[li], ws[li])
                gate(o, bias, ows[li], ois[li])
                outs.append(o)
            return outs

        insts = [capture(step) for _ in range(2)]
        per_iter = []
        nbad = 0
        for it in range(rw + ri):
            gi = torch.Generator(device="cuda").manual_seed(SEED + 1000 * m + it)
            for s in srcs:
                s.copy_(torch.randn((m, HIDDEN), generator=gi, device="cuda").to(torch.bfloat16))
            ref_l = [router(s, w) for s, w in zip(srcs, ws)]
            ref_g = []
            for li in range(LAYERS):
                a = torch.empty((m, TOPK), dtype=torch.float32, device="cuda")
                b = torch.empty((m, TOPK), dtype=torch.int32, device="cuda")
                gate(ref_l[li], bias, a, b)
                ref_g.append((a, b))
            torch.cuda.synchronize()
            res = []
            for k, (g, gouts) in enumerate(insts):
                for d in dsts:
                    d.fill_(float("nan"))
                for o in gouts:
                    o.fill_(float("nan"))
                for o in ows:
                    o.fill_(float("nan"))
                for o in ois:
                    o.fill_(-7)
                g.replay()
                torch.cuda.synchronize()
                bl = sum(int(not torch.equal(o, r)) for o, r in zip(gouts, ref_l))
                bg = sum(int(not (torch.equal(ows[li].view(torch.int32), ref_g[li][0].view(torch.int32))
                                  and torch.equal(ois[li], ref_g[li][1]))) for li in range(LAYERS))
                ea = max((o.double() - r.double()).abs().max().item() for o, r in zip(gouts, ref_l))
                hh = hashlib.sha256()
                for o in gouts:
                    h_update(hh, o)
                res.append({"inst": k, "logit_mismatch_layers": bl, "gate_mismatch_layers": bg, "max_abs": ea,
                            "hash": hh.hexdigest()[:12]})
                if k == 0:
                    for o in gouts:
                        h_update(seq_hash, o)
            pristine = all(torch.equal(w, p) for w, p in zip(ws, ws_p)) and torch.equal(bias, bias_p)
            cand_vs_cand = res[0]["hash"] == res[1]["hash"]
            ok = pristine and cand_vs_cand and all(r["logit_mismatch_layers"] == 0 and r["gate_mismatch_layers"] == 0
                                                   for r in res)
            nbad += int(not ok)
            per_iter.append({"iter": it, "warmup": it < rw, "pristine": pristine, "inst_equal": cand_vs_cand,
                             "max_abs": max(r["max_abs"] for r in res), "hash": res[0]["hash"]})
        seq[m] = {"iterations": rw + ri, "bad_iterations": nbad, "per_iter": per_iter}
        log(f"repeated-seq M={m}: bad {nbad}/{rw + ri}; per-iter max_abs "
            f"{[p['max_abs'] for p in per_iter]}; hashes {[p['hash'] for p in per_iter[:4]]}...")
        if nbad:
            failures.append(f"repeated-seq M={m}: {nbad} bad iterations")
        del insts
    report["repeated_sequence"] = seq
    sh = seq_hash.hexdigest()[:16]
    eh = eager_ref_hash(ws)
    log(f"hashes: eager M=1..16 x 40 {eh}; repeated-sequence {sh}")
    print(json.dumps({"passed": not failures, "failures": failures[:20], "eager_hash": eh, "sequence_hash": sh,
                      "compared": "router_gemm_mma<M,384,5120> under a live PDL producer (24 CTAs, early trigger, "
                                  "5 us spin, NaN-poisoned input) vs no-producer eager reference, bitwise; no-wait "
                                  "negative control; FP64/cuBLAS tolerance; repeated graph sequence 5+20 with fresh "
                                  "inputs and gate_forward outputs; hashes to compare with the BASE arm",
                      "summary": {"numerics": num, "overlap_live": live,
                                  "repeated": {m: {k: v for k, v in s.items() if k != "per_iter"}
                                               for m, s in seq.items()}}}, default=str))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "bench"
    torch.manual_seed(SEED)
    with torch.no_grad():
        {"bench": bench, "check": check}[mode]()
