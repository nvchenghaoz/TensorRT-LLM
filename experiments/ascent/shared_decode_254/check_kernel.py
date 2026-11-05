"""Idea 254 correctness: tensorrt_llm/_torch/modules/fp8_shared_decode.shared_decode_mlp.

One GPU, no model load. One TP4 rank's shared GatedMLP built as DeepseekV4MoE builds it
(experiments/ascent/shared_row_split/common.py: checkpoint quant config, shared_tp 2,
swiglu_limit, seeded synthetic FP8 weights, post_load_weights -> SM120 packed UE8M0 scales).

Checks, for M = 1..8 and ranges empty / 1 / 4 / 5 / 9 blocks (several positions):
 (a) activation quant byte-equal to per_token_quant_and_transform (x and h);
 (b) kernel vs a torch fp64 reference of the same block math (same quantized operands),
     BF16 tolerance rtol/atol 1.6e-2 plus max abs/rel error;
 (c) kernel [0,9) vs main's GatedMLP forward (the original op), and [0,5) + [5,9) (the
     SHARD pair sum) vs main's full half;
 (d) repeated replays of one CUDA graph (the range tensor rewritten between replays,
     as the reset point) bitwise equal to the eager call, every replay, NREP replays.
Last line: {"passed": ..., ...}.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "shared_row_split"))
if len(sys.argv) > 1 and not os.environ.get("CKPT"):
    os.environ["CKPT"] = sys.argv[1]

import torch  # noqa: E402

import common as srs  # noqa: E402

NREP = int(os.environ.get("NREP", "20"))
TOL = 1.6e-2


def log(*a):
    print(*a, flush=True)


def ref_quant(v):
    """per-token 1x128 UE8M0 quant in torch fp32 (mirrors the Triton kernels)."""
    m, k = v.shape
    v2 = v.float().view(m, k // 128, 128)
    amax = v2.abs().amax(-1).clamp_min(1e-10)
    s = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    q = (v2 / s[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q.view(m, k), s


def ref_mlp(x, w1, s1, w2, s2, b0, b1, limit):
    m, k = x.shape
    i = w1.shape[0] // 2
    q, sx = ref_quant(x)
    qd = q.double().view(m, k // 128, 128)
    w1d = w1.double().view(2 * i, k // 128, 128)
    # partial per K block: [m, n, kb]
    part = torch.einsum("mbk,nbk->mnb", qd, w1d)
    scale = sx.double()[:, None, :] * s1.double().repeat_interleave(128, 0)[None, :, :]
    gu = (part * scale).sum(-1).to(torch.bfloat16).float()
    g, u = gu[:, :i], gu[:, i:]
    if limit is not None:
        g = torch.minimum(g, torch.tensor(limit))
        u = u.clamp(-limit, limit)
    h = (g * torch.sigmoid(g) * u).to(torch.bfloat16)
    out = torch.zeros(m, w2.shape[0], dtype=torch.float64, device=x.device)
    if b1 > b0:
        hq, sh = ref_quant(h)
        c0, c1 = b0 * 128, b1 * 128
        hd = hq.double()[:, c0:c1].reshape(m, b1 - b0, 128)
        w2d = w2.double()[:, c0:c1].reshape(w2.shape[0], b1 - b0, 128)
        part2 = torch.einsum("mbk,nbk->mnb", hd, w2d)
        sc2 = sh.double()[:, None, b0:b1] * s2.double().repeat_interleave(128, 0)[None, :, b0:b1]
        out = (part2 * sc2).sum(-1)
    return out.to(torch.bfloat16), h


def errs(a, b):
    a, b = a.float(), b.float()
    d = (a - b).abs()
    rel = d / b.abs().clamp_min(1e-6)
    ok = bool(torch.all(d <= TOL + TOL * b.abs()).item())
    return {"max_abs": float(d.max().item()), "max_rel_where_abs_gt_0.1":
            float(rel[b.abs() > 0.1].max().item()) if bool((b.abs() > 0.1).any()) else 0.0,
            "ok": ok}


def main():
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.modules import fp8_shared_decode as fsd
    from tensorrt_llm.quantization.utils.fp8_utils import per_token_quant_and_transform

    mlp, shared_tp, scale, info = srs.build_shared_mlp(rank=0)
    log("shared", info)
    gu, dn = mlp.gate_up_proj, mlp.down_proj
    w1, w2 = gu.weight, dn.weight
    log("gate_up", tuple(w1.shape), w1.dtype, gu.weight_scale.dtype, tuple(gu.weight_scale.shape),
        "down", tuple(w2.shape), dn.weight_scale.dtype, tuple(dn.weight_scale.shape))
    s1 = fsd.block_scale_table(gu.weight_scale, w1.shape[0], w1.shape[1])
    s2 = fsd.block_scale_table(dn.weight_scale, w2.shape[0], w2.shape[1])
    log("scale tables", tuple(s1.shape), tuple(s2.shape), "distinct s1", s1.unique().numel(),
        "distinct s2", s2.unique().numel())
    limit = info["swiglu_limit"]
    nb = w1.shape[0] // 2 // 128
    ranges = [(0, 0), (3, 3), (0, 1), (8, 9), (4, 5), (0, 4), (5, 9), (0, 5), (2, 7), (0, nb)]
    g = torch.Generator(device="cpu").manual_seed(5)
    passed = True
    results = []
    for m in range(1, fsd.MAX_TOKENS + 1):
        # vary activation block scales: per-128-group gains 2^[-3, 3]
        gain = torch.exp2(torch.randint(-3, 4, (m, w1.shape[1] // 128), generator=g).float())
        x = (torch.randn(m, w1.shape[1], generator=g) * 0.5 *
             gain.repeat_interleave(128, 1)).to(torch.bfloat16).cuda()
        # (a) activation quant byte-equal to main's
        q_main, _ = per_token_quant_and_transform(x)
        q_ref, _ = ref_quant(x)
        quant_x_eq = bool(torch.equal(q_main.view(torch.uint8), q_ref.view(torch.uint8)))
        # main's GatedMLP (the original op): the full half
        main_full = mlp(x)
        rng = torch.zeros(2, dtype=torch.int32, device="cuda")
        row = {"M": m, "quant_x_byte_equal": quant_x_eq}
        ok_m = quant_x_eq
        outs = {}
        for (b0, b1) in ranges:
            rng.copy_(torch.tensor([b0, b1], dtype=torch.int32))
            out = fsd.shared_decode_mlp(x, w1, s1, w2, s2, rng, limit)
            torch.cuda.synchronize()
            ref, h_ref = ref_mlp(x, w1, s1, w2, s2, b0, b1, limit)
            e = errs(out, ref)
            fin = bool(torch.isfinite(out.float()).all().item())
            if b1 == b0:
                e["zeros"] = bool((out == 0).all().item())
                e["ok"] = e["ok"] and e["zeros"]
            # (a) h quant byte-equal (main's quant on the reference h)
            if b0 == 0 and b1 == nb:
                hq_main, _ = per_token_quant_and_transform(h_ref.contiguous())
                hq_ref, _ = ref_quant(h_ref)
                e["quant_h_byte_equal"] = bool(torch.equal(hq_main.view(torch.uint8),
                                                           hq_ref.view(torch.uint8)))
                e["ok"] = e["ok"] and e["quant_h_byte_equal"]
            e["finite"] = fin
            e["ok"] = e["ok"] and fin
            row[f"[{b0},{b1})"] = e
            ok_m = ok_m and e["ok"]
            outs[(b0, b1)] = out.clone()
        # (c) vs the original op
        full = outs[(0, nb)]
        half = (outs[(0, 5)] + outs[(5, nb)])
        row["full_vs_main"] = errs(full, main_full)
        row["pair_sum_vs_main"] = errs(half, main_full)
        ok_m = ok_m and row["full_vs_main"]["ok"] and row["pair_sum_vs_main"]["ok"]
        # (d) graph replays with the range tensor rewritten between replays
        rng_g = torch.zeros(2, dtype=torch.int32, device="cuda")
        rng_g.copy_(torch.tensor([0, 5], dtype=torch.int32))
        fsd.shared_decode_mlp(x, w1, s1, w2, s2, rng_g, limit)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            gout = fsd.shared_decode_mlp(x, w1, s1, w2, s2, rng_g, limit)
        rep_ok = True
        seq = [(0, 5), (5, nb), (0, 0), (0, nb), (4, 5)]
        max_rep = 0.0
        for it in range(NREP):
            b0, b1 = seq[it % len(seq)]
            rng_g.copy_(torch.tensor([b0, b1], dtype=torch.int32))  # reset point
            gout.fill_(float("nan"))
            gr.replay()
            torch.cuda.synchronize()
            same = bool(torch.equal(gout, outs[(b0, b1)]))
            max_rep = max(max_rep, float((gout.float() - outs[(b0, b1)].float()).abs().max()
                                         .nan_to_num(1e9).item()))
            rep_ok = rep_ok and same
        row["graph_replays_bitwise"] = rep_ok
        row["graph_replay_maxdiff"] = max_rep
        ok_m = ok_m and rep_ok
        row["ok"] = ok_m
        passed = passed and ok_m
        log(json.dumps(row))
        results.append(row)
    print(json.dumps({"passed": passed, "compared": "shared_decode_mlp vs torch fp64 block-math "
                      "reference (same quantized operands) and vs main GatedMLP forward; quant "
                      "byte-equal vs per_token_quant_and_transform; CUDA graph replays bitwise",
                      "M": list(range(1, 9)), "ranges": [list(r) for r in ranges],
                      "worst_full_vs_main_abs": max(r["full_vs_main"]["max_abs"] for r in results),
                      "worst_pair_vs_main_abs": max(r["pair_sum_vs_main"]["max_abs"]
                                                    for r in results)}))


if __name__ == "__main__":
    main()
