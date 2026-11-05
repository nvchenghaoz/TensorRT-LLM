"""Idea 279 correctness of the v2 default decode shared expert
(tensorrt_llm/_torch/modules/fp8_shared_decode.shared_decode_mlp -> shared_decode_mlp_v2).

One GPU, no model load; one rank's shared GatedMLP as DeepseekV4MoE builds it
(shared_row_split/common.py; the shared expert is laid out by tp_rank only, so it is the same
at EP4 and EP2xTP2). Reuses shared_decode_254/check_kernel.py's fp64 block reference.
 (1) x quant: _v2_xq_kernel output (e4m3 bytes and fp32 scales) byte-equal to
     per_token_quant_and_transform at M = 1..8 (rows >= M zero); h quant: the reference
     1x128 quant byte-equal to per_token_quant_and_transform on the reference h.
 (2) v2 vs the fp64 block reference (same quantized operands), BF16 rtol/atol 1.6e-2, at
     M = 1..8 over ranges empty/1/4/5/9 blocks (several positions); empty range = zeros;
     also vs v1 (shared_decode_mlp_v1, main's kernel) and vs main's GatedMLP forward.
 (3) CUDA graphs at M = 1..8 captured after the production warm-up (M 1, 2, 8 eager, as
     DeepseekV4MoE._prepare_shared_decode_split does), Triton caches unchanged by capture
     (no JIT inside capture); WARM warm-up replays then ITERS timed-like replays, before
     every replay the inputs (x, range) are rewritten from a pool and the output NaN-filled
     (reset point); every replay compared bitwise with the eager call on the same inputs,
     reported after warm-up and after the last replay. M = 9 is refused by the kernel
     (the decode split gate in modeling_deepseekv4.py, M <= 8, is unchanged).
Last line: {"passed": ..., ...}.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "shared_decode_254"))
if len(sys.argv) > 1 and not os.environ.get("CKPT"):
    os.environ["CKPT"] = sys.argv[1]

import torch  # noqa: E402

import check_kernel as ck  # noqa: E402  (imports shared_row_split/common.py as srs)

WARM = int(os.environ.get("CHK_WARM", "30"))
ITERS = int(os.environ.get("CHK_ITERS", "100"))
POOL = 5


def ncache(fn):
    try:
        return sum(len(v[0]) for v in fn.device_caches.values())
    except Exception:  # noqa: BLE001
        return None


def main():
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.modules import fp8_shared_decode as fsd
    from tensorrt_llm.quantization.utils.fp8_utils import per_token_quant_and_transform

    assert fsd._V2, "v2 must be the default"
    log = ck.log
    mlp, _tp, _scale, info = ck.srs.build_shared_mlp(rank=0)
    log("shared", info)
    gu, dn = mlp.gate_up_proj, mlp.down_proj
    w1, w2 = gu.weight, dn.weight
    s1 = fsd.block_scale_table(gu.weight_scale, *w1.shape)
    s2 = fsd.block_scale_table(dn.weight_scale, *w2.shape)
    limit = info["swiglu_limit"]
    hidden = w1.shape[1]
    nb = w1.shape[0] // 2 // 128
    ranges = [(0, 0), (3, 3), (0, 1), (8, 9), (4, 5), (0, 4), (5, 9), (0, 5), (2, 7), (0, nb)]
    g = torch.Generator(device="cpu").manual_seed(279)
    passed = True
    rows = []

    def mkx(m):
        gain = torch.exp2(torch.randint(-3, 4, (m, hidden // 128), generator=g).float())
        return (torch.randn(m, hidden, generator=g) * 0.5 *
                gain.repeat_interleave(128, 1)).to(torch.bfloat16).cuda()

    worst = {"ref": 0.0, "v1": 0.0, "main": 0.0}
    for m in range(1, fsd.MAX_TOKENS + 1):
        x = mkx(m)
        # (1) x quant by the v2 kernel itself
        xq = torch.full((fsd._TM, hidden), 0x7F, dtype=torch.uint8, device="cuda").view(
            torch.float8_e4m3fn)
        xs = torch.full((fsd._TM, hidden // 128), float("nan"), dtype=torch.float32,
                        device="cuda")
        fsd._v2_xq_kernel[(hidden // 128, )](x, xq, xs, m, x.stride(0), fsd._FP8_MAX,
                                             fsd._FP8_MIN, K=hidden, TM=fsd._TM, USE_PDL=False,
                                             num_warps=4)
        torch.cuda.synchronize()
        q_main, _sf = per_token_quant_and_transform(x)
        q_ref, s_ref = ck.ref_quant(x)
        xq_eq = bool(torch.equal(xq[:m].view(torch.uint8), q_main.view(torch.uint8)[:m]))
        xs_eq = bool(torch.equal(xs[:m], s_ref))
        pad_zero = bool((xq[m:].view(torch.uint8) == 0).all().item())
        row = {"M": m, "xq_byte_equal_main": xq_eq, "xs_equal_ref_scale": xs_eq,
               "xq_pad_rows_zero": pad_zero,
               "ref_quant_equal_main": bool(torch.equal(q_ref.view(torch.uint8),
                                                        q_main.view(torch.uint8)))}
        ok_m = xq_eq and xs_eq and pad_zero and row["ref_quant_equal_main"]
        main_full = mlp(x)
        rng = torch.zeros(2, dtype=torch.int32, device="cuda")
        outs = {}
        for (b0, b1) in ranges:
            rng.copy_(torch.tensor([b0, b1], dtype=torch.int32))
            out = fsd.shared_decode_mlp(x, w1, s1, w2, s2, rng, limit)
            out1 = fsd.shared_decode_mlp_v1(x, w1, s1, w2, s2, rng, limit)
            torch.cuda.synchronize()
            ref, h_ref = ck.ref_mlp(x, w1, s1, w2, s2, b0, b1, limit)
            e = ck.errs(out, ref)
            e["vs_v1"] = ck.errs(out, out1)
            e["finite"] = bool(torch.isfinite(out.float()).all().item())
            e["ok"] = e["ok"] and e["vs_v1"]["ok"] and e["finite"]
            if b1 == b0:
                e["zeros"] = bool((out == 0).all().item())
                e["ok"] = e["ok"] and e["zeros"]
            if (b0, b1) == (0, nb):
                hq_main, _ = per_token_quant_and_transform(h_ref.contiguous())
                hq_ref, _ = ck.ref_quant(h_ref)
                e["quant_h_byte_equal"] = bool(torch.equal(hq_main.view(torch.uint8),
                                                           hq_ref.view(torch.uint8)))
                e["ok"] = e["ok"] and e["quant_h_byte_equal"]
            worst["ref"] = max(worst["ref"], e["max_abs"])
            worst["v1"] = max(worst["v1"], e["vs_v1"]["max_abs"])
            row[f"[{b0},{b1})"] = e
            ok_m = ok_m and e["ok"]
            outs[(b0, b1)] = out.clone()
        row["full_vs_main"] = ck.errs(outs[(0, nb)], main_full)
        row["pair_sum_vs_main"] = ck.errs(outs[(0, 5)] + outs[(5, nb)], main_full,
                                          mag=outs[(0, 5)].float().abs()
                                          + outs[(5, nb)].float().abs())
        worst["main"] = max(worst["main"], row["full_vs_main"]["max_abs"])
        ok_m = ok_m and row["full_vs_main"]["ok"] and row["pair_sum_vs_main"]["ok"]
        row["ok"] = ok_m
        passed = passed and ok_m
        log(json.dumps(row))
        rows.append(row)

    # (3) CUDA graphs at M = 1..8 after the production warm-up
    kernels = [fsd._v2_xq_kernel, fsd._v2_gate_up_kernel, fsd._v2_down_kernel]
    rng_w = torch.tensor([0, 5], dtype=torch.int32, device="cuda")
    for m in (1, 2, 8):
        fsd.shared_decode_mlp(torch.zeros(m, hidden, dtype=torch.bfloat16, device="cuda"),
                              w1, s1, w2, s2, rng_w, limit)
    torch.cuda.synchronize()
    c0 = [ncache(k) for k in kernels]
    seq = [(0, 5), (5, nb), (0, 0), (0, nb), (4, 5)]
    graph_rows = []
    for m in range(1, fsd.MAX_TOKENS + 1):
        pool = [mkx(m) for _ in range(POOL)]
        xin = torch.empty(m, hidden, dtype=torch.bfloat16, device="cuda")
        rin = torch.zeros(2, dtype=torch.int32, device="cuda")
        refs = {}
        for p in range(POOL):
            for r in seq:
                rin.copy_(torch.tensor(r, dtype=torch.int32))
                refs[(p, r)] = fsd.shared_decode_mlp(pool[p], w1, s1, w2, s2, rin, limit).clone()
        xin.copy_(pool[0])
        rin.copy_(torch.tensor(seq[0], dtype=torch.int32))
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            gout = fsd.shared_decode_mlp(xin, w1, s1, w2, s2, rin, limit)
        torch.cuda.synchronize()
        c1 = [ncache(k) for k in kernels]
        no_jit = c1 == c0 and None not in c1
        diffs, ok_all = [], True
        for it in range(WARM + ITERS):
            p, r = it % POOL, seq[(it // POOL + it) % len(seq)]
            xin.copy_(pool[p])  # reset point: inputs rewritten, output poisoned
            rin.copy_(torch.tensor(r, dtype=torch.int32))
            gout.fill_(float("nan"))
            gr.replay()
            torch.cuda.synchronize()
            d = float((gout.float() - refs[(p, r)].float()).abs().max().nan_to_num(1e9).item())
            diffs.append(d)
            ok_all = ok_all and bool(torch.equal(gout, refs[(p, r)]))
        grow = {"M": m, "triton_cache_before_capture": c0, "after_capture": c1,
                "no_jit_in_capture": no_jit, "after_warmup_maxabs": max(diffs[:WARM]),
                "after_last_replay_maxabs": diffs[-1], "all_replays_maxabs": max(diffs),
                "all_replays_bitwise_vs_eager": ok_all, "replays": WARM + ITERS}
        grow["ok"] = bool(no_jit and ok_all)
        passed = passed and grow["ok"]
        log("graph", json.dumps(grow))
        graph_rows.append(grow)
        del gr
    try:
        fsd.shared_decode_mlp(mkx(9), w1, s1, w2, s2, rng_w, limit)
        m9_refused = False
    except AssertionError:
        m9_refused = True
    passed = passed and m9_refused
    print(json.dumps({"passed": passed, "impl": "v2 default", "m9_refused": m9_refused,
                      "worst_maxabs": worst, "graph_ok": all(r["ok"] for r in graph_rows),
                      "compared": "v2 shared_decode_mlp vs fp64 block reference, vs v1 "
                      "(main's kernel) and vs main GatedMLP; x quant byte-equal vs "
                      "per_token_quant_and_transform; graph replays bitwise vs eager "
                      "with inputs rewritten and outputs NaN-filled",
                      "M": list(range(1, 9)), "ranges": [list(r) for r in ranges]}))


if __name__ == "__main__":
    main()
