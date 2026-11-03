"""DeepSeek-V4.1-Flash router GEMM at decode: trtllm::dsv3_router_gemm_op [M,5120] x [5120,384] -> fp32.

Base tree: the op falls back to cuBLAS (cublas_mm_out). Branch: router_gemm_mma_kernel<M, 384, 5120> (mma.sync).

Modes (argv[1]):
  bench - CUDA graph of 40 router calls, each on a distinct [384,5120] bf16 weight (157 MB > 128 MB L2, so the
          weights stream from DRAM as in the 40-layer decode step). kernel_ms = graph time / 40 (per layer).
          Sequence per M (SEQUENCE): kernel-name probe (2 eager calls), then twice: 2 eager calls, capture,
          WARMUP + ITERS replays with fixed inputs and no reset in between.
  check - (1) numerics vs FP64 and vs the unmodified cuBLAS path (trtllm::cublas_mm = the same cublas_mm_out call)
          at M=1..16 plus M=17/32/2048 (must stay on cuBLAS, bitwise); (2) top-6 routing (gate_forward) on kernel vs
          cuBLAS logits over 10k rows per M=1..8, ties reported; (3) the EXACT bench sequence (same inputs, seed,
          probe, two graph instances, WARMUP + ITERS replays each, no resets) with every replay's 40 outputs
          compared to the cuBLAS reference and to the eager branch output, and the inputs/weights checked
          unchanged after every replay; (4) a fresh-input graph sequence (inputs reset before every replay).
"""

import glob
import json
import math
import os
import re
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


def kernel_names(fn):
    from torch.profiler import ProfilerActivity, profile
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sorted({e.name[:80] for e in prof.events() if e.device_type.name == "CUDA"})


def graph_sequence(step, holder, on_replay=None):
    """One graph instance: 2 eager calls, capture, WARMUP + ITERS replays (no resets). Shared by bench and check."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        step()
        step()
        torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=s):
            step()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    outs = holder["o"]  # graph-pool outputs written by every replay
    ts = []
    for i in range(WARMUP + ITERS):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        if on_replay is None:
            if i >= WARMUP:
                ts.append(a.elapsed_time(b) * 1000.0)
        else:
            on_replay(i, outs)
    del g
    if not ts:
        return None, None
    ts.sort()
    med = ts[len(ts) // 2]
    p10, p90 = ts[int(0.1 * (len(ts) - 1))], ts[int(0.9 * (len(ts) - 1))]
    return med, (p90 - p10) / med


def setup_sequence():
    """Inputs exactly as the timed run makes them: weights first, then per M the 40 inputs."""
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    ws = make_weights(gen)
    per_m = {}
    for m in MS:
        per_m[m] = [torch.randn((m, HIDDEN), generator=gen, device="cuda").to(torch.bfloat16) for _ in range(LAYERS)]
    return ws, per_m


def make_step(xs, ws, holder):
    def step():
        holder["o"] = [router(x, w) for x, w in zip(xs, ws)]
    return step


def bench():
    arm = sys.argv[2] if len(sys.argv) > 2 else "ARM?"  # BASE or CAND, printed on every timing line
    ws, per_m = setup_sequence()
    batch = {}
    for m in MS:
        xs = per_m[m]
        names = kernel_names(lambda: router(xs[0], ws[0]))
        holder = {}
        step = make_step(xs, ws, holder)
        med, nf = graph_sequence(step, holder)
        med2, nf2 = graph_sequence(step, holder)  # second graph instance: within-arm spread
        per_layer = min(med, med2) / LAYERS
        nfl = max(nf, nf2, abs(med - med2) / min(med, med2))
        log(f"{arm} M={m}: {per_layer:.3f} us/layer (best of this arm's 2 graph instances: inst1 {med / LAYERS:.3f}, "
            f"inst2 {med2 / LAYERS:.3f}; noise {nfl:.3f}) {arm} kernels={names}")
        batch[str(m)] = {"kernel_ms": per_layer / 1000.0, "noise_floor": nfl, "per_layer_us": round(per_layer, 3),
                         "kernels": names}
    print(json.dumps({"arm": arm, "batch": batch, "output": {"class": "reordered"}}))


def real_gate():
    """Layer-0 gate weight and bias from the checkpoint when readable (real scale), else None."""
    ck = sys.argv[2] if len(sys.argv) > 2 else ""
    try:
        from safetensors import safe_open
        pat = re.compile(r"(^|\.)layers\.0\..*gate\.weight$")
        files = sorted(glob.glob(os.path.join(ck, "*.safetensors")))
        log("checkpoint", ck, "shards:", len(files))
        for f in files:
            with safe_open(f, "pt", device="cpu") as h:
                names = set(h.keys())
                keys = [k for k in names if pat.search(k) and "proj" not in k]
                for k in keys:
                    shape = tuple(h.get_slice(k).get_shape())
                    log("candidate gate:", k, shape)
                    if shape != (EXPERTS, HIDDEN):
                        continue
                    w = h.get_tensor(k).to(torch.bfloat16).cuda()
                    b = None
                    for bk in (k.replace(".weight", ".e_score_correction_bias"), k.replace("gate.weight", "gate.bias"),
                               k.replace(".weight", ".bias")):
                        if bk in names:
                            b = h.get_tensor(bk).float().cuda()
                            break
                    log("real gate:", k, tuple(w.shape), "bias:", None if b is None else bk)
                    return w, b, k
    except Exception as e:
        log("real gate unavailable:", type(e).__name__, e)
    return None, None, None


_RM = {}


def route(logits, bias):
    """gate_forward routing (the model's DeepSeekV4MoeRoutingMethod). Returns (int32 indices, fp32 weights)."""
    from tensorrt_llm._torch.moe.fused_moe.routing import DeepSeekV4MoeRoutingMethod
    key = id(bias)
    if key not in _RM:
        _RM[key] = DeepSeekV4MoeRoutingMethod(top_k=6, n_group=1, topk_group=1, routed_scaling_factor=1.5,
                                              callable_e_score_correction_bias=lambda: bias,
                                              callable_tid2eid=lambda: None, is_hashed=False)
    a, b = _RM[key].apply(logits)
    # Pick by dtype, never by position.
    if a.dtype.is_floating_point and not b.dtype.is_floating_point:
        a, b = b, a
    if a.dtype.is_floating_point or not b.dtype.is_floating_point:
        raise RuntimeError(f"unexpected routing output dtypes {a.dtype} {b.dtype}")
    return a, b


def check():
    gen = torch.Generator(device="cuda").manual_seed(SEED + 1)
    failures, report = [], {}
    w_syn = make_weights(gen, 1)[0]
    b_syn = (torch.randn((EXPERTS,), generator=gen, device="cuda") * 0.05).float()
    w_real, b_real, rk = real_gate()
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
            reps = [router(x, w) for _ in range(5)]
            det = all(torch.equal(r, out) for r in reps)
            scale = ref.abs().max().item()
            e_k = (out.double() - ref).abs().max().item()
            e_c = (cub.double() - ref).abs().max().item()
            d_kc = (out.double() - cub.double()).abs().max().item()
            rel_kc = ((out.double() - cub.double()).abs() / cub.double().abs().clamp_min(1e-6)).max().item()
            fin = bool(torch.isfinite(out).all())
            shape_ok = out.shape == (m, EXPERTS) and out.dtype == torch.float32 and out.stride() == (EXPERTS, 1)
            rec = {"max_abs_vs_fp64": e_k, "cublas_max_abs_vs_fp64": e_c, "max_abs_vs_cublas": d_kc,
                   "max_rel_vs_cublas": rel_kc, "max_abs_logit": scale, "bitwise_eq_cublas": bool(torch.equal(out, cub)),
                   "deterministic": det, "finite": fin, "shape_ok": shape_ok}
            ok = det and fin and shape_ok
            if m > 16:
                ok = ok and rec["bitwise_eq_cublas"]  # M>16 must stay on cuBLAS
            else:
                ok = ok and d_kc <= 1e-4 * max(scale, 1.0) and e_k <= 4 * e_c + 1e-6 * max(scale, 1.0)
            report[f"{gname}_M{m}"] = rec
            if m in (1, 2, 4, 8, 16, 17, 2048):
                log(f"{gname}_M{m}", rec)
            if not ok:
                failures.append(f"{gname}_M{m}")
    # 2. Top-6 routing agreement on 10k rows per M=1..8 (gate_forward on kernel logits vs cuBLAS logits).
    rows_total = 10000
    topk = {}
    for gname, w, bias in gates:
        for m in range(1, 9):
            calls = math.ceil(rows_total / m)
            n_rows = idx_eq = set_eq = w_bit = 0
            w_max = dl_max = 0.0
            ties = []
            for c in range(calls):
                x = torch.randn((m, HIDDEN), generator=gen, device="cuda").to(torch.bfloat16)
                lk = router(x, w)
                lc = cublas_ref(x, w)
                ik, wk = route(lk, bias)
                ic, wc = route(lc, bias)
                same = (ik == ic).all(dim=1)
                same_set = (ik.sort(dim=1).values == ic.sort(dim=1).values).all(dim=1)
                n_rows += m
                idx_eq += int(same.sum())
                set_eq += int(same_set.sum())
                w_bit += int(((wk == wc).all(dim=1) & same).sum())
                if bool(same.any()):
                    w_max = max(w_max, (wk[same] - wc[same]).abs().max().item())
                dl_max = max(dl_max, (lk - lc).abs().max().item())
                if not bool(same_set.all()):
                    sc = torch.nn.functional.softplus(x.double() @ w.double().t()).sqrt() + bias.double()
                    srt = sc.sort(dim=1, descending=True).values
                    for r in (~same_set).nonzero().flatten().tolist():
                        if len(ties) < 20:
                            ties.append({"call": c, "row": r, "fp64_gap_6th_7th": (srt[r, 5] - srt[r, 6]).item(),
                                         "idx_kernel": sorted(ik[r].tolist()), "idx_cublas": sorted(ic[r].tolist())})
            rate = set_eq / n_rows
            topk[f"{gname}_M{m}"] = {"rows": n_rows, "top6_set_identical_rate": rate,
                                     "top6_idx_order_identical_rate": idx_eq / n_rows,
                                     "top6_idx_and_weights_bitwise_rate": w_bit / n_rows,
                                     "max_abs_weight_diff_same_idx": w_max, "max_abs_dlogit": dl_max,
                                     "tie_cases": ties}
            log(f"topk {gname} M={m}: rows {n_rows} set-identical {rate:.5f} order-identical {idx_eq / n_rows:.5f} "
                f"bitwise {w_bit / n_rows:.5f} wdiff {w_max:.3e} dlogit {dl_max:.3e} ties {len(ties)} {ties[:3]}")
            if rate < 0.999:
                failures.append(f"topk {gname} M={m} rate {rate}")
    report["topk"] = topk
    # 3. The exact timed sequence (same inputs/seed/probe/graph instances/replay counts, no resets), every replay
    #    compared. Mutable state: only the graph-pool outputs; inputs and weights must stay pristine.
    ws, per_m = setup_sequence()
    ws_pristine = [w.clone() for w in ws]
    seq = {}
    for m in MS:
        xs = per_m[m]
        xs_pristine = [x.clone() for x in xs]
        ref_c = [cublas_ref(x, w) for x, w in zip(xs, ws)]
        ref_e = [router(x, w) for x, w in zip(xs, ws)]
        idx_ref = [route(r, b_syn)[0] for r in ref_c]
        kernel_names(lambda: router(xs[0], ws[0]))  # same probe as bench
        holder = {}
        step = make_step(xs, ws, holder)
        per_iter, bad = [], []

        def on_replay(i, outs, inst):
            torch.cuda.synchronize()
            ea = er = 0.0
            neq = ndiff_idx = 0
            for li in range(LAYERS):
                o = outs[li]
                d = (o.double() - ref_c[li].double()).abs()
                ea = max(ea, d.max().item())
                er = max(er, (d / ref_c[li].double().abs().clamp_min(1e-6)).max().item())
                if not torch.equal(o, ref_e[li]):
                    neq += 1
                if not torch.equal(route(o, b_syn)[0], idx_ref[li]):
                    ndiff_idx += 1
            inputs_ok = all(torch.equal(x, p) for x, p in zip(xs, xs_pristine)) and \
                all(torch.equal(w, p) for w, p in zip(ws, ws_pristine))
            per_iter.append({"inst": inst, "iter": i, "max_abs_vs_cublas": ea, "max_rel_vs_cublas": er,
                             "layers_neq_eager": neq, "layers_top6_diff_vs_cublas": ndiff_idx,
                             "inputs_pristine": inputs_ok})
            if neq or not inputs_ok or ea > 1e-4:
                bad.append((inst, i))

        for inst in range(2):
            graph_sequence(step, holder, lambda i, outs, inst=inst: on_replay(i, outs, inst))
        top6_diff = sum(p["layers_top6_diff_vs_cublas"] for p in per_iter)
        seq[f"M{m}"] = {"replays": len(per_iter), "bad_replays": len(bad),
                        "max_abs_vs_cublas": max(p["max_abs_vs_cublas"] for p in per_iter),
                        "max_rel_vs_cublas": max(p["max_rel_vs_cublas"] for p in per_iter),
                        "top6_diff_layer_replays": top6_diff,
                        "per_iter_max_abs": [p["max_abs_vs_cublas"] for p in per_iter],
                        "all_inputs_pristine": all(p["inputs_pristine"] for p in per_iter)}
        log("timed-sequence", m, {k: v for k, v in seq[f"M{m}"].items() if k != "per_iter_max_abs"})
        if bad or len(per_iter) != 2 * (WARMUP + ITERS):
            failures.append(f"timed-sequence M={m} bad {bad[:5]}")
    report["timed_sequence"] = seq
    # 4. Fresh-input graph replays (inputs reset before each replay) at M=1,2,4,8,16.
    fresh = {}
    for m in MS:
        static = [torch.zeros((m, HIDDEN), dtype=torch.bfloat16, device="cuda") for _ in range(LAYERS)]
        holder = {}
        step = make_step(static, ws, holder)
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
        nbad, emax = 0, 0.0
        for i in range(WARMUP + ITERS):
            gi = torch.Generator(device="cuda").manual_seed(SEED + 97 * i + m)
            for x in static:
                x.copy_(torch.randn((m, HIDDEN), generator=gi, device="cuda").to(torch.bfloat16))
            g.replay()
            torch.cuda.synchronize()
            for li in range(LAYERS):
                if not torch.equal(outs[li], router(static[li], ws[li])):
                    nbad += 1
                emax = max(emax, (outs[li].double() - cublas_ref(static[li], ws[li]).double()).abs().max().item())
        del g
        fresh[f"M{m}"] = {"replays": WARMUP + ITERS, "replay_vs_eager_mismatches": nbad, "max_abs_vs_cublas": emax}
        log("fresh-input graph", m, fresh[f"M{m}"])
        if nbad or emax > 1e-4:
            failures.append(f"fresh graph M={m}")
    report["fresh_input_graph"] = fresh
    print(json.dumps({"passed": not failures, "failures": failures[:20],
                      "compared": "dsv3_router_gemm_op [M,5120]x[5120,384] vs FP64 and vs the unmodified cuBLAS "
                                  "path (trtllm::cublas_mm); top-6 gate_forward routing on 10k rows per M=1..8; "
                                  "the exact bench sequence (2 graph instances x (5 warmup + 30) replays, no resets) "
                                  "with every replay's 40 outputs compared; fresh-input graph replays",
                      "summary": {"topk": {k: {kk: vv for kk, vv in v.items() if kk != "tie_cases"}
                                           for k, v in topk.items()},
                                  "timed_sequence": {k: {kk: vv for kk, vv in v.items() if kk != "per_iter_max_abs"}
                                                     for k, v in seq.items()},
                                  "fresh_input_graph": fresh,
                                  "numerics": {k: v for k, v in report.items() if k.endswith(("_M1", "_M2", "_M4",
                                                                                              "_M8", "_M16", "_M17",
                                                                                              "_M2048"))}},
                      "report": report}, default=str))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "bench"
    torch.manual_seed(SEED)
    load_ops()
    with torch.no_grad():
        {"bench": bench, "check": check}[mode]()
