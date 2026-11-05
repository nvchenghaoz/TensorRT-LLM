"""Branch 147 (idea 255), evaluator item 2: EP2xTP2 vs EP4 tolerance check on identical inputs.

One GPU, real checkpoint weights of MoE layers (default L20 and L39), no model load. Per layer:
  EP4    : 4 rank partials, 96 experts x inter 2304, slot_start 0/96/192/288
  EP2xTP2: 4 rank partials, 192 experts x inter 1152 (moe_ep_rank 0/1 x moe_tp_rank 0/1);
           w3/w1 rows and w2 columns sliced contiguously by moe_tp_rank, as the TP loader does.
Each layout's 4 partials are summed in BF16 (the allreduce dtype), rank order 0..3.
Decode M=1/2/4/8: Sm12xW4a8DecodeState.run (the FlashInfer decode chain, what serving runs).
Prefill M=2048: mxfp8_quantize + torch.ops.trtllm.fused_moe (CUTLASS), use_fused_finalize=False
(deterministic; the fold path is bitwise this, landing 2x leaf_check).
Reference: FP32 dequantized MXFP4 weights, BF16 x, SwiGLU limit 10, no activation quant.
Reports max_abs / max_rel(5% floor) / calc_diff of both layouts vs reference and EP2xTP2 vs EP4,
plus a repeat of every layout computation (determinism).
Pass per case: finite, and (EP2xTP2 max_abs vs ref <= 1.25 x EP4's + 1e-3, or EP2xTP2 vs EP4
max_abs <= 0.5, the served eager-vs-eager control).

  python experiments/ascent/b147_layout_tol.py --ckpt {checkpoint} [--layers 20,39]
Last stdout line: {"passed": bool, ...}
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
import torch  # noqa: E402

T0 = time.time()
TOPK, LIMIT, NE = 6, 10.0, 384
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def load_layer(ckpt, layer):
    from safetensors import safe_open
    idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    pat = re.compile(rf"^layers\.{layer}\.ffn\.experts\.(\d+)\.(w1|w2|w3)\.(weight|scale)$")
    by_file = defaultdict(list)
    for k, f in idx.items():
        mm = pat.match(k)
        if mm:
            by_file[f].append((k, int(mm.group(1)), mm.group(2), mm.group(3)))
    n = sum(len(v) for v in by_file.values())
    assert n == NE * 6, f"layer {layer}: found {n} expert tensors, expected {NE * 6}"
    t = {}
    for f, items in by_file.items():
        with safe_open(os.path.join(ckpt, f), "pt", device="cpu") as h:
            for k, e, w, kind in items:
                t[(e, w, kind)] = h.get_tensor(k)
    s = {kk: (tuple(v.shape), str(v.dtype)) for kk, v in t.items() if kk[0] == 0}
    log(f"layer {layer}: expert 0 tensors {s}")

    def stack(w, kind):
        return torch.stack([t[(e, w, kind)].view(torch.uint8) for e in range(NE)]).cuda()

    w1, w3, w2 = stack("w1", "weight"), stack("w3", "weight"), stack("w2", "weight")
    s1, s3, s2 = stack("w1", "scale"), stack("w3", "scale"), stack("w2", "scale")
    del t
    # TRT-LLM w3_w1 order: [w3 (up/linear); w1 (gate)]
    return w3, w1, s3, s1, w2, s2


def trt_layout(w31, s31, w2, s2):
    e, n1, _ = w31.shape
    _, n2, k2h = w2.shape
    k1 = w31.shape[2] * 2
    k2 = k2h * 2
    a = w31.contiguous().view(torch.int64)
    b = w2.contiguous().view(torch.int64)
    sa = torch.ops.trtllm.block_scale_interleave(s31.contiguous()).view(torch.int32).reshape(e, n1, k1 // 128)
    sb = torch.ops.trtllm.block_scale_interleave(s2.contiguous()).view(torch.int32).reshape(e, n2, k2 // 128)
    return a, sa, b, sb


def rank_weights(W, layout, r):
    w3, w1, s3, s1, w2, s2 = W
    inter = w3.shape[1]
    if layout == "ep4":
        sl = slice(96 * r, 96 * r + 96)
        w31 = torch.cat([w3[sl], w1[sl]], 1)
        s31 = torch.cat([s3[sl], s1[sl]], 1)
        return trt_layout(w31, s31, w2[sl], s2[sl]) + (96, 96 * r, 1, 0, 4, r)
    ep, tp = r // 2, r % 2
    half = inter // 2
    sl = slice(192 * ep, 192 * ep + 192)
    rs = slice(tp * half, (tp + 1) * half)
    w31 = torch.cat([w3[sl, rs], w1[sl, rs]], 1)
    s31 = torch.cat([s3[sl, rs], s1[sl, rs]], 1)
    w2r = w2[sl, :, tp * half // 2:(tp + 1) * half // 2]
    s2r = s2[sl, :, tp * half // 32:(tp + 1) * half // 32]
    return trt_layout(w31, s31, w2r, s2r) + (192, 192 * ep, 2, tp, 2, ep)


class Rank:
    def __init__(self, W, layout, r):
        from tensorrt_llm._torch.moe.fused_moe.cutlass.w4a8_mxfp4_mxfp8_sm12x_decode import Sm12xW4a8DecodeState
        from tensorrt_llm._torch.utils import ActivationType
        (self.w31, self.s31, self.w2, self.s2, self.e, self.slot, self.tp, self.tpr, self.ep,
         self.epr) = rank_weights(W, layout, r)
        one = torch.ones(self.e, dtype=torch.float32, device="cuda")
        self.qs = [self.s31, one, self.s2, one.clone()]
        self.limit = torch.full((self.e,), LIMIT, dtype=torch.float32, device="cuda")
        self.act = int(ActivationType.Swiglu)
        self.fi = Sm12xW4a8DecodeState(w3_w1_weight=self.w31, w3_w1_weight_scale=self.s31, w2_weight=self.w2,
                                       w2_weight_scale=self.s2, num_local_experts=self.e, slot_start=self.slot,
                                       swiglu_limit=LIMIT, enable_pdl=True)

    def cutlass(self, x, ids, w):
        xq, xsf = torch.ops.trtllm.mxfp8_quantize(x, True, alignment=128)
        return torch.ops.trtllm.fused_moe(
            xq, ids, w, self.w31, None, self.w2, None, torch.bfloat16,
            quant_scales=self.qs, input_sf=xsf, swizzled_input_sf=True,
            swiglu_alpha=None, swiglu_beta=None, swiglu_limit=self.limit, swiglu_clamp_after_silu=False,
            tp_size=self.tp, tp_rank=self.tpr, ep_size=self.ep, ep_rank=self.epr, cluster_size=1, cluster_rank=0,
            enable_alltoall=False, use_deepseek_fp8_block_scale=False, use_w4_group_scaling=False,
            use_int8_woq_per_channel=False, use_mxfp8_act_scaling=True, min_latency_mode=False,
            use_fused_finalize=False, tune_max_num_tokens=8192, activation_type=self.act,
            unpadded_hidden_size=x.shape[1],
        )[0]

    def __call__(self, x, ids, w):
        if x.shape[0] <= 32:
            return self.fi.run(x, ids, w).clone()
        return self.cutlass(x, ids, w)


def dequant(codes, sf):
    lut = torch.tensor(E2M1 + [-v for v in E2M1], dtype=torch.float32, device=codes.device)
    n, kh = codes.shape
    v = torch.empty(n, kh * 2, dtype=torch.float32, device=codes.device)
    v[:, 0::2] = lut[(codes & 0x0F).long()]
    v[:, 1::2] = lut[(codes >> 4).long()]
    return (v.view(n, -1, 32) * torch.pow(2.0, sf.float() - 127.0)[:, :, None]).view(n, -1)


def reference(W, x, ids, w):
    w3, w1, s3, s1, w2, s2 = W
    out = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32, device="cuda")
    xf = x.float()
    for e in torch.unique(ids).tolist():
        tok, slot = (ids == e).nonzero(as_tuple=True)
        lin = xf[tok] @ dequant(w3[e], s3[e]).t()
        gate = (xf[tok] @ dequant(w1[e], s1[e]).t()).clamp(max=LIMIT)
        act = gate * torch.sigmoid(gate) * lin.clamp(-LIMIT, LIMIT)
        out.index_add_(0, tok, (act @ dequant(w2[e], s2[e]).t()) * w[tok, slot][:, None])
    return out


def stats(o, ref):
    o, ref = o.float(), ref.float()
    d = (o - ref).abs()
    amax = ref.abs().max().item()
    den = (o.double() ** 2 + ref.double() ** 2).sum().item()
    return {"max_abs": d.max().item(), "mean_abs": d.mean().item(),
            "max_rel": (d / ref.abs().clamp_min(0.05 * max(amax, 1e-6))).max().item(),
            "calc_diff": 0.0 if den == 0 else 1 - 2 * (o.double() * ref.double()).sum().item() / den,
            "ref_absmax": amax, "finite": bool(torch.isfinite(o).all())}


def inputs(m, hidden, seed):
    g = torch.Generator(device="cuda").manual_seed(17 + 31 * m + 1000 * seed)
    x = torch.randn(m, hidden, device="cuda", generator=g).to(torch.bfloat16)
    gc = torch.Generator().manual_seed(5 + 13 * m + 1000 * seed)
    ids = torch.stack([torch.randperm(NE, generator=gc)[:TOPK] for _ in range(m)]).to(torch.int32)
    w = torch.rand(m, TOPK, generator=gc) + 0.1
    w = (w / w.sum(-1, keepdim=True) * 1.5).float()
    return x, ids.cuda(), w.cuda()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--layers", default="20,39")
    ap.add_argument("--ms", default="1,2,4,8,2048")
    args = ap.parse_args()
    import tensorrt_llm  # noqa: F401
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    ms = [int(v) for v in args.ms.split(",")]
    log("gpu", torch.cuda.get_device_name(), "layers", args.layers, "ms", ms)
    res, ok = {}, True
    for layer in [int(v) for v in args.layers.split(",")]:
        W = load_layer(args.ckpt, layer)
        hidden = W[0].shape[2] * 2
        log(f"layer {layer} loaded: hidden {hidden} inter {W[0].shape[1]}")
        cases = [(m, s) for m in ms for s in (0, 1)]
        ins = {c: inputs(c[0], hidden, c[1]) for c in cases}
        sums = {}
        for layout in ("ep4", "ep2tp2"):
            acc = {c: None for c in cases}
            det = {c: True for c in cases}
            for r in range(4):
                rk = Rank(W, layout, r)
                for c in cases:
                    x, ids, w = ins[c]
                    p = rk(x, ids, w)
                    p2 = rk(x, ids, w)
                    torch.cuda.synchronize()
                    det[c] = det[c] and bool(torch.equal(p, p2))
                    acc[c] = p.clone() if acc[c] is None else acc[c] + p  # BF16 sum, rank order 0..3
                del rk
                torch.cuda.empty_cache()
                log(f"layer {layer} {layout} rank {r} done")
            sums[layout] = (acc, det)
        for c in cases:
            x, ids, w = ins[c]
            ref = reference(W, x, ids, w)
            a, b = sums["ep4"][0][c], sums["ep2tp2"][0][c]
            se, st, d = stats(a, ref), stats(b, ref), stats(b, a.float())
            case_ok = (se["finite"] and st["finite"]
                       and (st["max_abs"] <= 1.25 * se["max_abs"] + 1e-3 or d["max_abs"] <= 0.5))
            key = f"L{layer}_M{c[0]}_seed{c[1]}"
            res[key] = {"path": "fi_decode_chain" if c[0] <= 32 else "cutlass_unfused_finalize",
                        "ep4_vs_ref": se, "ep2tp2_vs_ref": st, "ep2tp2_vs_ep4": d,
                        "ep4_repeat_bitwise": sums["ep4"][1][c], "ep2tp2_repeat_bitwise": sums["ep2tp2"][1][c],
                        "ok": bool(case_ok)}
            ok = ok and case_ok
            log(key, "OK" if case_ok else "FAIL", f"ep4 max_abs {se['max_abs']:.4g} mean {se['mean_abs']:.3g} "
                f"cd {se['calc_diff']:.3g} | ep2tp2 max_abs {st['max_abs']:.4g} mean {st['mean_abs']:.3g} "
                f"cd {st['calc_diff']:.3g} | tp-ep max_abs {d['max_abs']:.4g} | ref absmax {se['ref_absmax']:.3g} "
                f"| det {sums['ep4'][1][c]}/{sums['ep2tp2'][1][c]}")
        del W, sums
        torch.cuda.empty_cache()
    agg = {k: max(v[k2]["max_abs"] for v in res.values()) for k, k2 in
           (("ep4_vs_ref_max_abs", "ep4_vs_ref"), ("ep2tp2_vs_ref_max_abs", "ep2tp2_vs_ref"),
            ("ep2tp2_vs_ep4_max_abs", "ep2tp2_vs_ep4"))}
    agg.update({k: max(v[k2]["max_rel"] for v in res.values()) for k, k2 in
                (("ep4_vs_ref_max_rel", "ep4_vs_ref"), ("ep2tp2_vs_ref_max_rel", "ep2tp2_vs_ref"),
                 ("ep2tp2_vs_ep4_max_rel", "ep2tp2_vs_ep4"))})
    print(json.dumps({"passed": bool(ok), "output_class": "reordered", "aggregate": agg, "cases": res}))


if __name__ == "__main__":
    main()
