"""DeepSeek-V4.1-Flash router GEMM at decode: trtllm::dsv3_router_gemm_op [M,5120] x [5120,384] -> fp32.

Base tree: the op falls back to cuBLAS (cublas_mm_out). Branch: router_gemm_mma_kernel<M, 384, 5120> (mma.sync).

Modes (argv[1]):
  bench - CUDA graph of 40 router calls, each on a distinct [384,5120] bf16 weight (157 MB > 128 MB L2, so the
          weights stream from DRAM as in the 40-layer decode step). kernel_ms = graph time / 40 (per layer).
  check - numerics vs FP64 and vs the unmodified cuBLAS path (trtllm::cublas_mm, the same cublas_mm_out call),
          top-6 routing agreement over 10k rows per M (gate_forward on both logits), graph capture/replay with the
          harness's warmup/iteration/reset points, determinism, M>16 stays on cuBLAS bitwise.
"""

import glob
import json
import math
import os
import sys

import torch

HIDDEN, EXPERTS, LAYERS = 5120, 384, 40
WARMUP, ITERS = 5, 30
SEED = 2468
MS = (1, 2, 4, 8, 16)


def log(*a):
    print(*a, flush=True)


def load_ops():
    import tensorrt_llm  # noqa: F401  registers torch.ops.trtllm


def make_weights(gen, n=LAYERS):
    return [(torch.randn((EXPERTS, HIDDEN), generator=gen, device="cuda") * HIDDEN ** -0.5).to(torch.bfloat16)
            for _ in range(n)]


def router(x, w):
    return torch.ops.trtllm.dsv3_router_gemm_op(x, w.t(), bias=None, out_dtype=torch.float32)


def cublas_ref(x, w):
    return torch.ops.trtllm.cublas_mm(x, w.t(), None, torch.float32)


def time_graph(fn, warm=WARMUP, iters=ITERS):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        fn()
        fn()
        torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=s):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    ts = []
    for i in range(warm + iters):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        if i >= warm:
            ts.append(a.elapsed_time(b) * 1000.0)
    ts.sort()
    med = ts[len(ts) // 2]
    p10, p90 = ts[int(0.1 * (len(ts) - 1))], ts[int(0.9 * (len(ts) - 1))]
    del g
    return med, (p90 - p10) / med


def kernel_names(fn):
    from torch.profiler import ProfilerActivity, profile
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sorted({e.name[:80] for e in prof.events() if e.device_type.name == "CUDA"})


def bench(warm=False):
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    ws = make_weights(gen, 1) * LAYERS if warm else make_weights(gen)  # warm: one L2-resident weight
    batch = {}
    for m in MS:
        xs = [torch.randn((m, HIDDEN), generator=gen, device="cuda").to(torch.bfloat16) for _ in range(LAYERS)]
        names = kernel_names(lambda: router(xs[0], ws[0]))

        def step():
            for x, w in zip(xs, ws):
                router(x, w)
        med, nf = time_graph(step)
        # Repeat once more for a within-arm spread across graph instances.
        med2, nf2 = time_graph(step)
        per_layer = min(med, med2) / LAYERS
        nfl = max(nf, nf2, abs(med - med2) / min(med, med2))
        log(f"M={m}: {med / LAYERS:.3f} / {med2 / LAYERS:.3f} us per layer (noise {nfl:.3f}) kernels={names}")
        batch[str(m)] = {"kernel_ms": per_layer / 1000.0, "noise_floor": nfl, "per_layer_us": round(per_layer, 3),
                         "kernels": names}
    print(json.dumps({"batch": batch, "output": {"class": "reordered"}}))


def real_gate(gen):
    """Layer-0 gate weight and bias from the checkpoint when readable (real scale), else synthetic."""
    ck = sys.argv[2] if len(sys.argv) > 2 else ""
    try:
        from safetensors import safe_open
        import re
        pat = re.compile(r"(^|\.)layers\.0\..*gate\.weight$")
        files = sorted(glob.glob(os.path.join(ck, "*.safetensors")))
        log("checkpoint shards:", len(files))
        for f in files:
            with safe_open(f, "pt", device="cpu") as h:
                if f == files[0]:
                    log("sample keys:", [k for k in h.keys() if "gate" in k][:8])
                keys = [k for k in h.keys() if pat.search(k) and "proj" not in k]
                if keys:
                    k = keys[0]
                    w = h.get_tensor(k).to(torch.bfloat16).cuda()
                    names = set(h.keys())
                    b = None
                    for bk in (k.replace(".weight", ".e_score_correction_bias"), k.replace("gate.weight", "gate.bias"),
                               k.replace(".weight", ".bias")):
                        if bk in names:
                            b = h.get_tensor(bk).float().cuda()
                            break
                    log("real gate:", k, tuple(w.shape), "bias:", None if b is None else tuple(b.shape))
                    if tuple(w.shape) == (EXPERTS, HIDDEN):
                        return w, b, k
    except Exception as e:
        log("real gate unavailable:", type(e).__name__, e)
    return None, None, None


def route(logits, bias):
    from tensorrt_llm._torch.moe.fused_moe.routing import DeepSeekV4MoeRoutingMethod
    rm = DeepSeekV4MoeRoutingMethod(top_k=6, n_group=1, topk_group=1, routed_scaling_factor=1.5,
                                    callable_e_score_correction_bias=lambda: bias,
                                    callable_tid2eid=lambda: None, is_hashed=False)
    return rm.apply(logits)


def check():
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    failures, report = [], {}
    w_syn = make_weights(gen, 1)[0]
    b_syn = (torch.randn((EXPERTS,), generator=gen, device="cuda") * 0.05).float()
    w_real, b_real, rk = real_gate(gen)
    gates = [("synthetic", w_syn, b_syn)]
    if w_real is not None:
        gates.append(("real_layer0", w_real, b_real if b_real is not None else b_syn))
    report["real_gate_key"] = rk
    # 1. Numerics vs FP64 and cuBLAS, M=1..16 plus fallback sizes.
    for gname, w, _ in gates:
        for m in list(range(1, 17)) + [17, 32, 2048]:
            x = torch.randn((m, HIDDEN), generator=gen, device="cuda").to(torch.bfloat16)
            out = router(x, w)
            cub = cublas_ref(x, w)
            ref = x.double() @ w.double().t()
            key = f"{gname}_M{m}"
            reps = [router(x, w) for _ in range(5)]
            det = all(torch.equal(r, out) for r in reps)
            scale = ref.abs().max().item()
            e_k = (out.double() - ref).abs().max().item()
            e_c = (cub.double() - ref).abs().max().item()
            d_kc = (out.double() - cub.double()).abs().max().item()
            fin = bool(torch.isfinite(out).all())
            shape_ok = out.shape == (m, EXPERTS) and out.dtype == torch.float32 and out.stride() == (EXPERTS, 1)
            rec = {"max_abs_vs_fp64": e_k, "cublas_max_abs_vs_fp64": e_c, "max_abs_vs_cublas": d_kc,
                   "max_abs_logit": scale, "bitwise_eq_cublas": bool(torch.equal(out, cub)),
                   "deterministic": det, "finite": fin, "shape_ok": shape_ok}
            ok = det and fin and shape_ok
            if m > 16:
                ok = ok and rec["bitwise_eq_cublas"]  # M>16 must stay on cuBLAS
            else:
                ok = ok and d_kc <= 1e-4 * max(scale, 1.0) and e_k <= 4 * e_c + 1e-6 * max(scale, 1.0)
            report[key] = rec
            if m in (1, 2, 4, 8, 16, 17, 2048):
                log(key, rec)
            if not ok:
                failures.append(key)
    # 2. Top-6 routing agreement on 10k rows per M=1..8 (gate_forward on kernel logits vs cuBLAS logits).
    rows_total = 10000
    topk = {}
    for gname, w, bias in gates:
        for m in range(1, 9):
            calls = math.ceil(rows_total / m)
            n_rows = idx_eq = w_bit = 0
            w_max = 0.0
            dl_max = 0.0
            ties = []
            for c in range(calls):
                x = torch.randn((m, HIDDEN), generator=gen, device="cuda").to(torch.bfloat16)
                lk = router(x, w)
                lc = cublas_ref(x, w)
                ik, wk = route(lk, bias)  # apply() returns (indices int32, weights fp32)
                ic, wc = route(lc, bias)
                assert ik.dtype == torch.int32 and wk.dtype == torch.float32
                same = (ik == ic).all(dim=1)
                n_rows += m
                idx_eq += int(same.sum())
                w_bit += int(((wk == wc).all(dim=1) & same).sum())
                if bool(same.any()):
                    w_max = max(w_max, (wk[same] - wc[same]).abs().max().item())
                dl_max = max(dl_max, (lk - lc).abs().max().item())
                if not bool(same.all()):
                    sc = torch.nn.functional.softplus(lc.double()).sqrt() + bias.double()
                    srt = sc.sort(dim=1, descending=True).values
                    for r in (~same).nonzero().flatten().tolist():
                        if len(ties) < 5:
                            ties.append({"call": c, "row": r, "gap_6th_7th": (srt[r, 5] - srt[r, 6]).item(),
                                         "idx_kernel": sorted(ik[r].tolist()), "idx_cublas": sorted(ic[r].tolist())})
            rate = idx_eq / n_rows
            topk[f"{gname}_M{m}"] = {"rows": n_rows, "top6_idx_identical_rate": rate,
                                     "top6_idx_and_weights_bitwise_rate": w_bit / n_rows,
                                     "max_abs_weight_diff_same_idx": w_max, "max_abs_dlogit": dl_max,
                                     "tie_cases": ties}
            log(f"topk {gname} M={m}: rows {n_rows} idx-identical {rate:.5f} "
                f"bitwise {w_bit / n_rows:.5f} wdiff {w_max:.3e} dlogit {dl_max:.3e} ties {len(ties)}")
            if rate < 0.999:
                failures.append(f"topk {gname} M={m} rate {rate}")
    report["topk"] = topk
    # 3. CUDA-graph capture/replay, harness warmup/iteration/reset points, M=1,2,4,8 (+16), 40 layers.
    ws = make_weights(gen)
    gq = {}
    for m in (1, 2, 4, 8, 16):
        static = [torch.zeros((m, HIDDEN), dtype=torch.bfloat16, device="cuda") for _ in range(LAYERS)]
        holder = {}

        def step():
            holder["o"] = [router(x, w) for x, w in zip(static, ws)]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        g = torch.cuda.CUDAGraph()
        with torch.cuda.stream(s):
            step()
            torch.cuda.synchronize()
            with torch.cuda.graph(g, stream=s):
                step()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        outs = holder["o"]
        errs, bad = [], 0
        for i in range(WARMUP + ITERS):
            gi = torch.Generator(device="cuda").manual_seed(SEED + 97 * i + m)
            for x in static:  # reset point: fresh inputs before every replay
                x.copy_(torch.randn((m, HIDDEN), generator=gi, device="cuda").to(torch.bfloat16))
            g.replay()
            torch.cuda.synchronize()
            e_it = 0.0
            for li in (0, 17, LAYERS - 1):
                eager = router(static[li], ws[li])
                cub = cublas_ref(static[li], ws[li])
                if not torch.equal(outs[li], eager):
                    bad += 1
                e_it = max(e_it, (outs[li].double() - cub.double()).abs().max().item())
            errs.append(e_it)
            # Inputs must be untouched by the replay (the op has no mutable state besides its output).
        gq[f"M{m}"] = {"iters": WARMUP + ITERS, "replay_vs_eager_mismatches": bad,
                       "max_abs_vs_cublas_per_iter_max": max(errs), "per_iter": [round(e, 9) for e in errs[:5]]}
        log("graph", m, gq[f"M{m}"])
        if bad:
            failures.append(f"graph M={m}")
        del g
    report["graph"] = gq
    print(json.dumps({"passed": not failures, "failures": failures[:20],
                      "compared": "dsv3_router_gemm_op [M,5120]x[5120,384] vs FP64 and vs the unmodified cuBLAS "
                                  "path (trtllm::cublas_mm); top-6 gate_forward routing on 10k rows per M=1..8; "
                                  "40-layer graph replays vs eager with fresh inputs per replay",
                      "report": report}, default=str))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "bench"
    torch.manual_seed(SEED)
    load_ops()
    with torch.no_grad():
        {"bench": bench, "warm": lambda: bench(True), "check": check}[mode]()
