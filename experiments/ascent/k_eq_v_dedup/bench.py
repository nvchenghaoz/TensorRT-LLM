"""Gemma4 global (attention_k_eq_v) layer attention front: qkv GEMM + split +
v_norm + q/k norm + RoPE + convert_qkv, through the tree's Gemma4Attention.

Modes:
  bench  - time the front per M (CUDA events, cold L2 via 4 rotated weight
           copies), print per-M sha256 of the first call's q/k/v so base and
           candidate arms can be compared bitwise.
  check  - untimed sequence identical to the timed one (same warmup, iters,
           rotation); every call compared bitwise with the first call, with an
           in-script replica of main's [q|k|v=k] pipeline, and with an FP32
           reference.
Usage: bench.py <checkpoint> <bench|check>
"""

import hashlib
import json
import os
import sys

import torch
import torch.nn.functional as F

CKPT = sys.argv[1]
MODE = sys.argv[2] if len(sys.argv) > 2 else "bench"
MS = [300, 1000, 1200, 2000, 4000, 4800, 8000, 8100, 8192]
WARMUP = 5
ITERS = 25
NCOPIES = 4


def log(*a):
    print(*a, flush=True)


def main():
    import transformers

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_gemma4 import Gemma4Attention
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    cfg = transformers.AutoConfig.from_pretrained(CKPT)
    tcfg = getattr(cfg, "text_config", cfg)
    tcfg.torch_dtype = torch.bfloat16
    with open(os.path.join(CKPT, "config.json")) as f:
        raw = json.load(f)
    rtext = raw.get("text_config", raw)
    info = {
        "hidden_size": tcfg.hidden_size,
        "num_hidden_layers": tcfg.num_hidden_layers,
        "num_attention_heads": tcfg.num_attention_heads,
        "attention_k_eq_v": rtext.get("attention_k_eq_v"),
        "num_global_key_value_heads": rtext.get("num_global_key_value_heads"),
        "global_head_dim": rtext.get("global_head_dim"),
        "num_full_layers": sum(t == "full_attention" for t in tcfg.layer_types),
        "num_sliding_layers": sum(t == "sliding_attention" for t in tcfg.layer_types),
    }
    log("config:", json.dumps(info))
    layer_idx = tcfg.layer_types.index("full_attention")

    qc = QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8)
    mc = ModelConfig(pretrained_config=tcfg, quant_config=qc, attn_backend="FLASHINFER")
    torch.manual_seed(0)
    with torch.device("cuda"):
        attn = Gemma4Attention(mc, layer_idx=layer_idx, is_sliding=False)
    attn = attn.cuda()
    H = tcfg.hidden_size
    qs, kvs, hd = attn.q_size, attn.kv_size, attn.head_dim
    log(
        f"layer {layer_idx}: q_size={qs} kv_size={kvs} head_dim={hd} "
        f"qkv_proj type={type(attn.qkv_proj).__name__} "
        f"weight={tuple(attn.qkv_proj.weight.shape)} (N before dedup would be {qs + 2 * kvs})"
    )

    g = torch.Generator(device="cuda").manual_seed(1234)
    wq = (torch.randn(qs, H, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    wk = (torch.randn(kvs, H, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    qn = (1.0 + 0.1 * torch.randn(hd, device="cuda", generator=g)).to(attn.q_norm.weight.dtype)
    kn = (1.0 + 0.1 * torch.randn(hd, device="cuda", generator=g)).to(attn.k_norm.weight.dtype)
    attn.q_norm.weight.data.copy_(qn)
    attn.k_norm.weight.data.copy_(kn)
    # The checkpoint payload as the weight mapper builds it: v_proj := k_proj.
    attn.qkv_proj.load_weights([{"weight": wq}, {"weight": wk}, {"weight": wk.clone()}])
    copies = [attn.qkv_proj.weight.detach().clone() for _ in range(NCOPIES)]
    log(f"weight copies: {NCOPIES} x {copies[0].numel() * 2 / 2**20:.1f} MiB")

    def front(x, pos, i):
        attn.qkv_proj.weight.data = copies[i % NCOPIES]
        qkv = attn.qkv_proj(x)
        q, k, v, _, _ = attn.preprocess_qkv(qkv, pos)
        q, k, v = attn.convert_qkv(q, k, v)
        if k is None:
            q, k, v = q.split([qs, kvs, kvs], dim=-1)
        return q, k, v

    # Replica of main's pipeline: fused [q|k|v=k] GEMM, split, v_norm on the
    # raw V block, q/k norm, RoPE (Gemma4Attention.apply_rope on main).
    w_full = torch.cat([wq, wk, wk], dim=0)

    def main_front(x, pos):
        qkv = F.linear(x, w_full)
        q, k, v = qkv.split([qs, kvs, kvs], dim=-1)
        v = attn.v_norm(v.reshape(-1, hd)).reshape(-1, kvs)
        q, k = attn.apply_qk_norm(q, k)
        q, k = attn.rotary_emb(pos, [q, k])
        qkv = torch.concat([q, k, v], dim=-1)
        return qkv.split([qs, kvs, kvs], dim=-1)

    def fp32_ref(x, pos):
        eps = attn.q_norm.variance_epsilon
        xf = x.float()
        qf = xf @ wq.float().t()
        kf = xf @ wk.float().t()

        def rms(t, w=None):
            t = t.view(t.shape[0], -1, hd)
            o = t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + eps)
            if w is not None:
                o = o * w.float()
            return o

        v = rms(kf)
        q = rms(qf, qn)
        k = rms(kf, kn)
        cs = attn.rotary_emb.rotary_cos_sin[pos.view(-1).long()]  # [M, 2, hd/2]
        cos, sin = cs[:, 0, :].unsqueeze(1), cs[:, 1, :].unsqueeze(1)

        def rope(t):
            x1, x2 = t[..., : hd // 2], t[..., hd // 2 :]
            return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

        M = x.shape[0]
        return rope(q).reshape(M, -1), rope(k).reshape(M, -1), v.reshape(M, -1)

    def sha(ts):
        h = hashlib.sha256()
        for t in ts:
            h.update(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
        return h.hexdigest()[:16]

    def errs(out, ref):
        r = {}
        for name, o, rr in zip("qkv", out, ref):
            d = o.float() - rr
            r[name] = (d.abs().max().item(), (d.norm() / rr.norm()).item(), rr.abs().max().item())
        return r

    batch = {}
    passed = True
    details = {}
    for M in MS:
        torch.manual_seed(M)
        x = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)
        pos = torch.arange(M, device="cuda", dtype=torch.int32).unsqueeze(0)
        if MODE == "bench":
            times = []
            first = None
            for it in range(WARMUP + ITERS):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                out = front(x, pos, it)
                e.record()
                if it == 0:
                    first = sha(out)
                if it >= WARMUP:
                    e.synchronize()
                    times.append(s.elapsed_time(e))
            times.sort()
            n = len(times)
            med = times[n // 2]
            p10 = times[int(0.1 * (n - 1))]
            p90 = times[int(0.9 * (n - 1))]
            batch[str(M)] = {
                "kernel_ms": med,
                "p10_ms": p10,
                "p90_ms": p90,
                "noise_floor": (p90 - p10) / med,
                "qkv_sha": first,
            }
            log(f"M={M}: median {med:.4f} ms p10 {p10:.4f} p90 {p90:.4f} sha {first}")
        else:
            ref_a = main_front(x, pos)
            ref_b = fp32_ref(x, pos)
            ea = errs(ref_a, ref_b)
            first = None
            ok_rep = True
            ok_main = True
            ok_ref = True
            ec = None
            for it in range(WARMUP + ITERS):
                out = front(x, pos, it)
                torch.cuda.synchronize()
                if first is None:
                    first = [t.clone() for t in out]
                    ec = errs(out, ref_b)
                ok_rep &= all(torch.equal(a, b) for a, b in zip(out, first))
                ok_main &= all(torch.equal(a, b) for a, b in zip(out, ref_a))
            maxdiff_main = max((a.float() - b.float()).abs().max().item() for a, b in zip(first, ref_a))
            for name in "qkv":
                mag = ea[name][2]
                ulp = 2.0 ** (int(torch.tensor(mag).log2().floor().item()) - 7)
                ok_ref &= ec[name][0] <= 1.10 * ea[name][0] + ulp
                ok_ref &= ec[name][1] <= 1.10 * ea[name][1] + 2.0**-8
            finite = all(torch.isfinite(t).all().item() for t in first)
            shapes = [tuple(t.shape) for t in first] == [tuple(t.shape) for t in ref_a]
            ok = ok_rep and ok_main and ok_ref and finite and shapes
            passed &= ok
            details[str(M)] = {
                "repeat_bitwise": ok_rep,
                "bitwise_vs_main_replica": ok_main,
                "maxabs_vs_main": maxdiff_main,
                "cand_vs_fp32": {k: v[:2] for k, v in ec.items()},
                "main_vs_fp32": {k: v[:2] for k, v in ea.items()},
                "finite": finite,
                "shapes_ok": shapes,
                "sha": sha(first),
            }
            log(f"M={M}: {json.dumps(details[str(M)])}")
        del x
    if MODE == "bench":
        print(json.dumps({"batch": batch, "output": {"class": "exact"}, "config": info}))
    else:
        print(
            json.dumps(
                {
                    "passed": bool(passed),
                    "compared": "Gemma4 global k_eq_v attention front q/k/v vs main pipeline "
                    "replica (bitwise), vs first call over 30 calls (bitwise), and vs FP32 reference",
                    "qkv_weight_shape": list(attn.qkv_proj.weight.shape),
                    "config": info,
                    "details": details,
                }
            )
        )


if __name__ == "__main__":
    main()
