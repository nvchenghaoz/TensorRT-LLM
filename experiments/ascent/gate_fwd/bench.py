"""gate_forward_kernel<384,6,false> body trim (idea 162): bench / check / ncu.

bench ARM  - per M in (1,2,4,8,32): CUDA graphs of 40 layers, each layer with its own [384,5120] bf16 router weight
             (157 MB > L2). A = 40 x router (dsv3_router_gemm_op), B = 40 x (router -> gate_forward), G = 40 x
             gate_forward on L2-hot scores (no flush, back to back). A and B replays are interleaved, each after a
             read-only 512 MB flush (flush.sum(), no dirty lines); per-node gate cost = median over iterations of
             (B_i - A_i)/40, noise = (p90-p10)/median of those per-iteration differences. kernel_ms = that per-node
             cost. Also prints an output hash on a fixed 64k-row set (compare BASE vs CAND arms: must be equal).
check      - bitwise new body vs the legacy body (TRTLLM_GATE_FORWARD_LEGACY_BODY=1, the original code) on 1M random
             rows over M=1..2048, crafted ties, s>20 / inf / huge negative, vision (image_mask) and hash paths; vs an
             FP32 torch reference (tolerance); the exact bench sequence (2 graph instances x WARMUP+ITERS replays,
             outputs poisoned before each replay, every replay compared, inputs checked pristine); graph capture and
             fresh-input replays at batch 1..32.
ncu        - warmup, then 2 legacy + 2 new launches at M=1 and M=4 plus 2 null launches each (profile_kernels with
             launch_skip=240, launch_count=12, regex 'gate_forward_kernel|_null_gate').
"""
import hashlib
import json
import os
import sys

import torch
import triton
import triton.language as tl

HIDDEN, EXPERTS, LAYERS, TOPK, SCALE = 5120, 384, 40, 6, 1.5
SEED = 9162
WARMUP, ITERS = 10, 60
MS = (1, 2, 4, 8, 32)
ENV = "TRTLLM_GATE_FORWARD_LEGACY_BODY"


def log(*a):
    print(*a, flush=True)


def set_legacy(on):
    if on:
        os.environ[ENV] = "1"
    else:
        os.environ.pop(ENV, None)


@triton.jit
def _null_gate(w_ptr, i_ptr, M, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M * 6
    tl.store(w_ptr + offs, (offs % 6).to(tl.float32) * 0.25, mask=mask)
    tl.store(i_ptr + offs, offs % 6, mask=mask)


def null_gate(m, w, i):
    _null_gate[(triton.cdiv(m, 4),)](w, i, m, BLOCK=32, num_warps=4)


EMPTY = {}


def empty_i():
    if "i" not in EMPTY:
        EMPTY["i"] = torch.empty(0, dtype=torch.int32, device="cuda")
    return EMPTY["i"]


def router(x, w):
    return torch.ops.trtllm.dsv3_router_gemm_op(x, w.t(), bias=None, out_dtype=torch.float32)


def gate(scores, bias, ow, oi, image_mask=None, vision_bias=None):
    torch.ops.trtllm.gate_forward(scores, bias, empty_i(), empty_i(), ow, oi, TOPK, SCALE, False, image_mask,
                                  vision_bias)


def gate_hash(scores, input_ids, tid2eid, ow, oi):
    torch.ops.trtllm.gate_forward(scores, torch.empty(0, dtype=torch.float32, device="cuda"), input_ids, tid2eid,
                                  ow, oi, TOPK, SCALE, True)


def run_gate(scores, bias, legacy, image_mask=None, vision_bias=None):
    m = scores.shape[0]
    ow = torch.full((m, TOPK), float("nan"), dtype=torch.float32, device="cuda")
    oi = torch.full((m, TOPK), -7, dtype=torch.int32, device="cuda")
    set_legacy(legacy)
    gate(scores, bias, ow, oi, image_mask, vision_bias)
    set_legacy(False)
    return ow, oi


def bits_equal(a, b):
    return a.shape == b.shape and torch.equal(a.view(torch.int32), b.view(torch.int32))


def capture(step):
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
    return g


def setup(gen):
    ws = [(torch.randn((EXPERTS, HIDDEN), generator=gen, device="cuda") * HIDDEN ** -0.5).to(torch.bfloat16)
          for _ in range(LAYERS)]
    bias = (torch.randn((EXPERTS,), generator=gen, device="cuda") * 0.05).float()
    per_m = {m: [torch.randn((m, HIDDEN), generator=gen, device="cuda").to(torch.bfloat16) for _ in range(LAYERS)]
             for m in MS}
    return ws, bias, per_m


def make_steps(m, xs, ws, bias, ows, ois, scs):
    def a():
        for li in range(LAYERS):
            router(xs[li], ws[li])

    def b():
        for li in range(LAYERS):
            gate(router(xs[li], ws[li]), bias, ows[li], ois[li])

    def gonly():
        for li in range(LAYERS):
            gate(scs[li], bias, ows[li], ois[li])

    return a, b, gonly


def timed(g, flush):
    if flush is not None:
        flush.sum()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    g.replay()
    e1.record()
    e1.synchronize()
    return e0.elapsed_time(e1) * 1000.0


def stats(v):
    v = sorted(v)
    med = v[len(v) // 2]
    return med, (v[int(0.9 * (len(v) - 1))] - v[int(0.1 * (len(v) - 1))]) / abs(med)


def output_hash(gen_seed=777):
    g = torch.Generator(device="cuda").manual_seed(gen_seed)
    bias = (torch.randn((EXPERTS,), generator=g, device="cuda") * 0.05).float()
    h = hashlib.sha256()
    for m in (1, 4, 2048, 64 * 1024 - 2053):
        sc = torch.randn((m, EXPERTS), generator=g, device="cuda") * 3.0
        ow = torch.empty((m, TOPK), dtype=torch.float32, device="cuda")
        oi = torch.empty((m, TOPK), dtype=torch.int32, device="cuda")
        gate(sc, bias, ow, oi)
        h.update(ow.cpu().numpy().tobytes())
        h.update(oi.cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def bench():
    arm = sys.argv[2] if len(sys.argv) > 2 else "ARM?"
    import tensorrt_llm  # noqa: F401
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    ws, bias, per_m = setup(gen)
    flush = torch.ones(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    batch = {}
    for m in MS:
        xs = per_m[m]
        scs = [router(x, w) for x, w in zip(xs, ws)]
        ows = [torch.empty((m, TOPK), dtype=torch.float32, device="cuda") for _ in range(LAYERS)]
        ois = [torch.empty((m, TOPK), dtype=torch.int32, device="cuda") for _ in range(LAYERS)]
        a, b, gonly = make_steps(m, xs, ws, bias, ows, ois, scs)
        ga, gb, gg = capture(a), capture(b), capture(gonly)
        diffs, gonlys = [], []
        for it in range(WARMUP + ITERS):
            ta = timed(ga, flush)
            tb = timed(gb, flush)
            tg = timed(gg, None)
            if it >= WARMUP:
                diffs.append((tb - ta) / LAYERS)
                gonlys.append(tg / LAYERS)
        node, nf = stats(diffs)
        go, gnf = stats(gonlys)
        log(f"{arm} M={m}: gate_forward per-node in router chain {node:.3f} us (noise {nf:.3f}); "
            f"gate-only chain {go:.3f} us/node (noise {gnf:.3f}) {arm}")
        batch[str(m)] = {"kernel_ms": node / 1000.0, "noise_floor": nf, "per_node_us": round(node, 3),
                         "gate_only_us": round(go, 3), "gate_only_noise": gnf}
        del ga, gb, gg
    hsh = output_hash()
    log(f"{arm} output hash {hsh}")
    print(json.dumps({"arm": arm, "batch": batch, "output": {"class": "exact", "hash": hsh}}))


def torch_ref(sc, bias):
    s = sc.double()
    sp = torch.where(s > 20, s, torch.log1p(torch.exp(s)))
    score = sp.sqrt()
    idx = torch.topk(score + bias.double(), k=TOPK, dim=-1).indices
    w = score.gather(1, idx)
    w = w / (w.sum(-1, keepdim=True) + 1e-20) * SCALE
    return idx.to(torch.int32), w.float()


def check():
    import tensorrt_llm  # noqa: F401
    gen = torch.Generator(device="cuda").manual_seed(SEED + 1)
    failures, report = [], {}

    def ab(name, sc, bias, image_mask=None, vision_bias=None, ref=True):
        wl, il = run_gate(sc, bias, True, image_mask, vision_bias)
        wn, in_ = run_gate(sc, bias, False, image_mask, vision_bias)
        wn2, in2 = run_gate(sc, bias, False, image_mask, vision_bias)
        ok = bits_equal(wl, wn) and torch.equal(il, in_) and bits_equal(wn, wn2) and torch.equal(in_, in2)
        rec = report.setdefault(name, {"rows": 0, "calls": 0, "bitwise_fail_calls": 0, "ref_idx_mismatch_rows": 0,
                                       "ref_max_abs_w": 0.0})
        rec["rows"] += sc.shape[0]
        rec["calls"] += 1
        if not ok:
            rec["bitwise_fail_calls"] += 1
            if len(failures) < 20:
                failures.append(f"{name} M={sc.shape[0]}")
        if ref and image_mask is None:
            ri, rw = torch_ref(sc, bias)
            same = (ri == in_).all(1)
            rec["ref_idx_mismatch_rows"] += int((~same).sum())
            if bool(same.any()):
                d = (rw[same] - wn[same]).abs()
                d = d[torch.isfinite(d)]
                if d.numel():
                    rec["ref_max_abs_w"] = max(rec["ref_max_abs_w"], d.max().item())
        return ok

    # 1. 1M random rows over many M, three logit scales, two bias scales.
    ms_cycle = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 31, 32, 33, 64, 100, 127, 128, 129, 255, 256, 511, 1024, 2047, 2048]
    rows, ci = 0, 0
    while rows < 1_000_000:
        m = ms_cycle[ci % len(ms_cycle)]
        sigma = (1.0, 3.0, 10.0)[ci % 3]
        bsc = (0.05, 1.0)[(ci // 3) % 2]
        bias = (torch.randn((EXPERTS,), generator=gen, device="cuda") * bsc).float()
        sc = torch.randn((m, EXPERTS), generator=gen, device="cuda") * sigma
        ab("random", sc, bias)
        rows += m
        ci += 1
    log("random", report["random"])
    # 2. Crafted ties: quantized logits, zero bias; all-equal rows; quantized bias; duplicated expert columns.
    for t in range(200):
        m = ms_cycle[t % len(ms_cycle)]
        q = (torch.randn((m, EXPERTS), generator=gen, device="cuda") * 2).round() / 2
        ab("ties_quantized_zero_bias", q, torch.zeros(EXPERTS, device="cuda"), ref=False)
        qb = ((torch.randn((EXPERTS,), generator=gen, device="cuda") * 4).round() / 8).float()
        ab("ties_quantized_quant_bias", q, qb, ref=False)
        eq = torch.full((m, EXPERTS), float(t % 7 - 3), device="cuda")
        ab("ties_all_equal", eq, torch.zeros(EXPERTS, device="cuda"), ref=False)
        dup = torch.randn((m, EXPERTS // 2), generator=gen, device="cuda").repeat(1, 2)
        ab("ties_dup_columns", dup, torch.zeros(EXPERTS, device="cuda"), ref=False)
    # 3. Softplus threshold and extremes: values around 20, >20, inf, huge negative.
    for t in range(100):
        m = ms_cycle[t % len(ms_cycle)]
        base = 20.0 + torch.randn((m, EXPERTS), generator=gen, device="cuda") * 2
        bias = (torch.randn((EXPERTS,), generator=gen, device="cuda") * 0.05).float()
        ab("around_20", base, bias)
        big = 20.0 + torch.rand((m, EXPERTS), generator=gen, device="cuda") * 80
        ab("above_20", big, bias)
        neg = -100.0 - torch.rand((m, EXPERTS), generator=gen, device="cuda") * 50
        ab("huge_negative", neg, bias, ref=False)
        mix = torch.randn((m, EXPERTS), generator=gen, device="cuda") * 3
        mix[:, (t * 37) % EXPERTS] = float("inf")
        ab("with_inf", mix, bias, ref=False)
    # 4. Vision path: image_mask + vision_bias.
    for t in range(150):
        m = ms_cycle[t % len(ms_cycle)]
        sc = torch.randn((m, EXPERTS), generator=gen, device="cuda") * 3
        bias = (torch.randn((EXPERTS,), generator=gen, device="cuda") * 0.5).float()
        vb = (torch.randn((EXPERTS,), generator=gen, device="cuda") * 0.5).float()
        mask = torch.rand((m,), generator=gen, device="cuda") < 0.5
        ab("vision", sc, bias, mask, vb, ref=False)
        # torch reference for vision: per-row bias
        wn, in_ = run_gate(sc, bias, False, mask, vb)
        rb = torch.where(mask.unsqueeze(-1), vb, bias)
        ri = torch.topk(torch.where(sc > 20, sc.double(), torch.log1p(torch.exp(sc.double()))).sqrt() + rb.double(),
                        k=TOPK, dim=-1).indices.to(torch.int32)
        report["vision"]["ref_idx_mismatch_rows"] += int((ri != in_).any(1).sum())
    # 5. Hash path (kernel body unchanged; checked against the reference gather).
    hrec = {"calls": 0, "fail": 0}
    vocab = 1000
    tid2eid = torch.stack([torch.randperm(EXPERTS, generator=torch.Generator().manual_seed(i))[:TOPK]
                           for i in range(vocab)]).to(torch.int32).cuda()
    for t in range(50):
        m = ms_cycle[t % len(ms_cycle)]
        sc = torch.randn((m, EXPERTS), generator=gen, device="cuda") * 3
        ids = torch.randint(0, vocab, (m,), generator=gen, device="cuda", dtype=torch.int32)
        outs = []
        for legacy in (True, False):
            ow = torch.full((m, TOPK), float("nan"), device="cuda")
            oi = torch.full((m, TOPK), -7, dtype=torch.int32, device="cuda")
            set_legacy(legacy)
            gate_hash(sc, ids, tid2eid, ow, oi)
            set_legacy(False)
            outs.append((ow, oi))
        ei = tid2eid[ids.long()]
        s = torch.where(sc > 20, sc, torch.log1p(torch.exp(sc))).sqrt().gather(1, ei.long())
        rw = s / (s.sum(-1, keepdim=True) + 1e-20) * SCALE
        ok = bits_equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1]) and \
            torch.equal(outs[1][1], ei) and torch.allclose(outs[1][0], rw, rtol=1e-5, atol=1e-6)
        hrec["calls"] += 1
        hrec["fail"] += 0 if ok else 1
    report["hash"] = hrec
    if hrec["fail"]:
        failures.append("hash")
    for k, v in report.items():
        log(k, v)
        if k != "hash" and v.get("bitwise_fail_calls", 0):
            failures.append(f"{k}: {v['bitwise_fail_calls']} bitwise-fail calls")
    rr = report["random"]
    if rr["ref_max_abs_w"] > 1e-5 or rr["ref_idx_mismatch_rows"] > 1e-4 * rr["rows"]:
        failures.append(f"random vs fp64 torch ref: {rr['ref_idx_mismatch_rows']} idx rows, w {rr['ref_max_abs_w']}")

    # 6. The exact bench sequence: same inputs/seed/graphs/replay counts; outputs poisoned before each replay and
    #    compared after it with the legacy eager reference; inputs checked unchanged.
    gen2 = torch.Generator(device="cuda").manual_seed(SEED)
    ws, bias, per_m = setup(gen2)
    ws_p = [w.clone() for w in ws]
    bias_p = bias.clone()
    seq = {}
    for m in MS:
        xs = per_m[m]
        xs_p = [x.clone() for x in xs]
        scs = [router(x, w) for x, w in zip(xs, ws)]
        refs = [run_gate(s, bias, True) for s in scs]
        ows = [torch.empty((m, TOPK), dtype=torch.float32, device="cuda") for _ in range(LAYERS)]
        ois = [torch.empty((m, TOPK), dtype=torch.int32, device="cuda") for _ in range(LAYERS)]
        a, b, gonly = make_steps(m, xs, ws, bias, ows, ois, scs)
        bad, n = [], 0
        for inst in range(2):
            gb, gg = capture(b), capture(gonly)
            for it in range(WARMUP + ITERS):
                for g in (gb, gg):
                    for o in ows:
                        o.fill_(float("nan"))
                    for o in ois:
                        o.fill_(-7)
                    g.replay()
                    torch.cuda.synchronize()
                    n += 1
                    maxerr = 0.0
                    for li in range(LAYERS):
                        if not (bits_equal(ows[li], refs[li][0]) and torch.equal(ois[li], refs[li][1])):
                            bad.append((inst, it, li))
                        maxerr = max(maxerr, (ows[li] - refs[li][0]).abs().max().item())
                    if maxerr > 0:
                        bad.append((inst, it, "maxerr", maxerr))
            pristine = all(torch.equal(x, p) for x, p in zip(xs, xs_p)) and \
                all(torch.equal(w, p) for w, p in zip(ws, ws_p)) and torch.equal(bias, bias_p)
            if not pristine:
                bad.append((inst, "inputs changed"))
            del gb, gg
        seq[f"M{m}"] = {"replays": n, "bad": len(bad), "max_abs_err": 0.0 if not bad else "see bad"}
        log("timed-sequence", m, seq[f"M{m}"], bad[:3])
        if bad:
            failures.append(f"timed-sequence M={m}: {bad[:3]}")
    report["timed_sequence"] = seq

    # 7. Graph capture + fresh-input replays at batch 1..32.
    fresh = {"batches": 0, "replays": 0, "mismatch": 0}
    bias = (torch.randn((EXPERTS,), generator=gen, device="cuda") * 0.05).float()
    for m in range(1, 33):
        st = torch.zeros((m, EXPERTS), device="cuda")
        ow = torch.empty((m, TOPK), dtype=torch.float32, device="cuda")
        oi = torch.empty((m, TOPK), dtype=torch.int32, device="cuda")
        g = capture(lambda: gate(st, bias, ow, oi))
        for i in range(8):
            st.copy_(torch.randn((m, EXPERTS), generator=gen, device="cuda") * 3)
            ow.fill_(float("nan"))
            oi.fill_(-7)
            g.replay()
            torch.cuda.synchronize()
            rw, ri = run_gate(st, bias, True)
            if not (bits_equal(ow, rw) and torch.equal(oi, ri)):
                fresh["mismatch"] += 1
            fresh["replays"] += 1
        fresh["batches"] += 1
        del g
    report["graph_batch_1_32"] = fresh
    log("graph batch 1..32", fresh)
    if fresh["mismatch"]:
        failures.append("graph batch 1..32")
    print(json.dumps({"passed": not failures, "failures": failures[:20],
                      "compared": "gate_forward<384,6,false> new body vs the legacy body (kill switch "
                                  "TRTLLM_GATE_FORWARD_LEGACY_BODY=1 = original code), bitwise out_weights/out_indices; "
                                  "1M random rows M=1..2048; crafted ties; s around/above 20, inf, -100; vision; hash; "
                                  "FP64 torch reference; exact bench sequence with poisoned outputs every replay; "
                                  "graph capture/replay batch 1..32",
                      "report": report}, default=str))


def ncu():
    import tensorrt_llm  # noqa: F401
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    ws = [(torch.randn((EXPERTS, HIDDEN), generator=gen, device="cuda") * HIDDEN ** -0.5).to(torch.bfloat16)
          for _ in range(LAYERS)]
    bias = (torch.randn((EXPERTS,), generator=gen, device="cuda") * 0.05).float()
    data = {}
    for m in (1, 4):
        xs = [torch.randn((m, HIDDEN), generator=gen, device="cuda").to(torch.bfloat16) for _ in range(LAYERS)]
        ow = torch.empty((m, TOPK), dtype=torch.float32, device="cuda")
        oi = torch.empty((m, TOPK), dtype=torch.int32, device="cuda")
        data[m] = (xs, ow, oi)
        for legacy in (True, False):
            set_legacy(legacy)
            for li in range(LAYERS):  # 40 matching launches
                gate(router(xs[li], ws[li]), bias, ow, oi)
        set_legacy(False)
        for li in range(LAYERS):  # 40 matching launches
            router(xs[li], ws[li])
            null_gate(m, ow, oi)
    torch.cuda.synchronize()
    log("warmup done (240 matching launches)")
    for m in (1, 4):
        xs, ow, oi = data[m]
        for kind in ("legacy", "legacy", "new", "new", "null", "null"):
            sc = router(xs[0], ws[0])
            if kind == "null":
                null_gate(m, ow, oi)
            else:
                set_legacy(kind == "legacy")
                gate(sc, bias, ow, oi)
                set_legacy(False)
            torch.cuda.synchronize()
            log("profiled", m, kind)


if __name__ == "__main__":
    torch.manual_seed(SEED)
    with torch.no_grad():
        {"bench": bench, "check": check, "ncu": ncu}[sys.argv[1] if len(sys.argv) > 1 else "bench"]()
