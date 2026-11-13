"""Probe + microbench: cuDNN sdpa_fp8 / sdpa_mxfp8 on SM120 at the MiniMax-H3 per-rank shape.

Shape: B=1, S=M (~37.9k packed video+audio tokens), H_local = num_attention_heads / 2 (tp 2),
D=128, non-causal, no mask. Q, K are contiguous NHD buffers; V is a strided view into a
[1, S, 3*H*D] qkv buffer, as in transformer_minimax_h3.py.

Modes:
  probe  : print cuDNN version, try to build bf16 / fp8 / mxfp8 graphs (guard bypassed), report.
  fa2    : time torch SDPA flash backend (VANILLA today) on HND views.
  fp8    : time CuDNNAttention fp8 (incl. the three quantize passes).
  mxfp8  : time CuDNNAttention mxfp8 (incl. quantize passes).
  cudnn_bf16 : time CuDNNAttention unquantized.
  check  : untimed repeated-execution check of a mode (--target) against FP32 reference.
"""

import argparse
import json
import math
import os
import sys
import time
import traceback

import torch
import torch.nn.functional as F


def log(*a):
    print(*a, flush=True)


def get_heads(ckpt):
    try:
        with open(os.path.join(ckpt, "transformer", "config.json")) as f:
            cfg = json.load(f)
        return int(cfg["num_attention_heads"]), int(cfg.get("attention_head_dim", 128)), cfg
    except Exception as e:  # noqa
        log("config read failed:", e)
        return 48, 128, {}


def _rope(x, pos, D):
    half = D // 2
    inv = 1.0 / (10000 ** (torch.arange(half, device=x.device, dtype=torch.float32) / half))
    ang = pos[:, None].float() * inv[None]  # [S, half]
    c, s = ang.cos()[None, :, None, :], ang.sin()[None, :, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], -1)


def make_inputs(S, H, D, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    # Q/K: per-head RMSNorm with a non-unit learned weight, then RoPE, as in H3.
    # Add a shared per-head mean direction so attention is not flat (realistic locality).
    wq = 0.5 + 1.5 * torch.rand(H, D, device="cuda", generator=g)
    wk = 0.5 + 1.5 * torch.rand(H, D, device="cuda", generator=g)
    wq[:, :4] *= 3.0
    wk[:, :4] *= 3.0
    base = torch.randn(1, S, H, D, device="cuda", generator=g)
    q = base + 0.8 * torch.randn(1, S, H, D, device="cuda", generator=g)
    k = base + 0.8 * torch.randn(1, S, H, D, device="cuda", generator=g)
    q = q / q.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt() * wq
    k = k / k.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt() * wk
    pos = torch.arange(S, device="cuda") % 4096
    q = _rope(q, pos, D).bfloat16().contiguous()
    k = _rope(k, pos, D).bfloat16().contiguous()
    qkv = (torch.randn(1, S, 3 * H * D, device="cuda", generator=g) * 2.0).bfloat16()
    v = qkv[..., 2 * H * D:].unflatten(-1, (H, D))  # strided view, row stride 3*H*D
    return q, k, v, qkv


def fi_runner(mode, q, k, v, H, D, scale_softmax=1.0):
    """FlashInfer SM120 per-tensor e4m3 kernels, quantize passes included."""
    import tensorrt_llm  # noqa: F401  (registers torch.ops.trtllm)

    S = q.shape[1]
    sm_scale = 1.0 / math.sqrt(D)
    out = torch.empty(1, S, H, D, dtype=torch.bfloat16, device=q.device)

    def quant(x):
        xq, ds = torch.ops.trtllm.quantize_e4m3_per_tensor(x.contiguous())
        return xq, ds.float().reshape(1)

    if mode == "fi_v2":
        from flashinfer.prefill import fmha_v2_prefill_sm120

        def fn():
            qq, dq = quant(q)
            kq, dk = quant(k)
            vq, dv = quant(v)
            s1 = dq * dk * sm_scale
            fmha_v2_prefill_sm120(qq, kq, vq, out, num_heads=H, head_dim=D, seq_len=S,
                                  scale_softmax=scale_softmax, scale_bmm1=1.0, scale_bmm2=1.0,
                                  causal=False, scale_bmm1_d=s1, scale_bmm2_d=dv)
            return out

        return fn
    if mode == "fi_cute":
        from flashinfer.attention.cute_dsl.sm120_fmha import sm120_fmha_fp8_ragged_prefill

        cu = torch.tensor([0, S], dtype=torch.int32, device=q.device)

        def fn():
            qq, dq = quant(q)
            kq, dk = quant(k)
            vq, dv = quant(v)
            sc = torch.stack([dq, dk, dv]).cpu()  # host sync: API takes host scalars
            sm = float(sc[0] * sc[1]) * sm_scale
            sm120_fmha_fp8_ragged_prefill(qq[0], kq[0], vq[0], out[0], cu, cu, max_seqlen_q=S,
                                          is_causal=False, sm_scale=sm, v_scale=float(sc[2]))
            return out

        return fn
    raise ValueError(mode)


def make_backend(mode, H, D):
    from tensorrt_llm._torch.visual_gen.attention.backends.cudnn import CuDNNAttention
    from tensorrt_llm.visual_gen.args import QuantAttentionConfig

    qac = None
    if mode == "fp8":
        qac = QuantAttentionConfig(qk_dtype="fp8", v_dtype="fp8")
    elif mode == "mxfp8":
        qac = QuantAttentionConfig(qk_dtype="mxfp8", v_dtype="mxfp8")
    return CuDNNAttention(num_heads=H, head_dim=D, dtype=torch.bfloat16, quant_attention_config=qac)


def bypass_guard():
    from tensorrt_llm._torch.visual_gen.attention.backends.cudnn import CuDNNAttention

    CuDNNAttention.check_hardware_compatibility = classmethod(lambda cls, device, quant_dtype=None: None)


def runner(mode, q, k, v, H, D):
    if mode == "be_fp8":
        # The branch's code path: FlashInferAttention with the fp8/fp8 recipe (NHD inputs,
        # V as the strided qkv view), exactly as the H3 attention module calls it.
        from tensorrt_llm._torch.visual_gen.attention.backends.flashinfer import FlashInferAttention
        from tensorrt_llm.visual_gen.args import QuantAttentionConfig

        be = FlashInferAttention(num_heads=H, head_dim=D, dtype=torch.bfloat16,
                                 quant_attention_config=QuantAttentionConfig(qk_dtype="fp8", v_dtype="fp8"))
        return lambda: be.forward(q, k, v, discard_padded_queries=True, valid_seq_len=None)
    if mode.startswith("fi_"):
        m, _, ss = mode.partition("@")
        return fi_runner(m, q, k, v, H, D, float(ss) if ss else 1.0)
    if mode == "fa2":
        from torch.nn.attention import SDPBackend, sdpa_kernel

        def fn():
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
            return o.transpose(1, 2)

        return fn
    be = make_backend({"cudnn_bf16": None}.get(mode, mode), H, D)
    return lambda: be.forward(q, k, v)


def reference_rows(q, k, v, rows):
    """FP32 SDPA (math) for a subset of query rows; rows are independent."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    qs = q[:, rows].float().transpose(1, 2)  # [1,H,r,D]
    kk = k.float().transpose(1, 2)
    vv = v.float().transpose(1, 2)
    outs = []
    with sdpa_kernel([SDPBackend.MATH]):
        for h in range(qs.shape[1]):
            outs.append(F.scaled_dot_product_attention(qs[:, h:h + 1], kk[:, h:h + 1], vv[:, h:h + 1]))
    return torch.cat(outs, 1).transpose(1, 2)  # [1,r,H,D]


def err(o, ref, rows):
    a = o[:, rows].float()
    d = (a - ref)
    return {"max_abs": d.abs().max().item(), "rel_l2": (d.norm() / ref.norm()).item()}


def time_fn(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for i in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    n = len(ts)
    med = ts[n // 2]
    p10, p90 = ts[int(0.1 * (n - 1))], ts[int(0.9 * (n - 1))]
    return med, (p90 - p10) / med, ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="probe")
    ap.add_argument("--target", default="fp8")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--seq", type=int, default=37966)
    ap.add_argument("--modes", default="cudnn_bf16,fp8,mxfp8,fa2")
    ap.add_argument("--heads", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--ref_rows", type=int, default=1024)
    ap.add_argument("--bypass", type=int, default=1)
    args = ap.parse_args()

    nh, D, cfg = get_heads(args.ckpt)
    H = args.heads or nh // 2
    S = args.seq
    log(f"config heads={nh} head_dim={D} H_local={H} S={S} cfg_keys={sorted(cfg)[:30]}")
    import cudnn

    log("cudnn frontend", getattr(cudnn, "__version__", "?"), "backend", cudnn.backend_version(),
        "torch cudnn", torch.backends.cudnn.version(), "cap", torch.cuda.get_device_capability())
    if args.bypass:
        bypass_guard()

    q, k, v, qkv = make_inputs(S, H, D)
    torch.manual_seed(1)
    rows = torch.randperm(S, device="cuda")[: args.ref_rows].sort().values

    if args.mode == "probe":
        res = {}
        ref = reference_rows(q, k, v, rows)
        for m in args.modes.split(","):
            log(f"--- {m}")
            try:
                fn = runner(m, q, k, v, H, D)
                t0 = time.time()
                o = fn()
                torch.cuda.synchronize()
                log(f"built+ran in {time.time() - t0:.1f}s")
                e = err(o, ref, rows)
                med, nf, _ = time_fn(fn, 3, 10)
                res[m] = {"ok": True, "ms": med, "noise": nf, **e}
                log(m, res[m])
            except Exception as ex:  # noqa
                tb = traceback.format_exc()
                log(tb)
                res[m] = {"ok": False, "error": f"{type(ex).__name__}: {str(ex)[:1500]}"}
            torch.cuda.empty_cache()
        print(json.dumps({"probe": res, "cudnn": cudnn.backend_version(), "S": S, "H": H}))
        return

    if args.mode == "decomp":
        import tensorrt_llm  # noqa: F401

        def fq(x):
            xq, ds = torch.ops.trtllm.quantize_e4m3_per_tensor(x.contiguous())
            return (xq.float() * ds.float()).bfloat16().view(x.shape) if False else (xq.float() * ds.float().reshape(-1)[0]).view(x.shape)

        ref = reference_rows(q, k, v, rows)
        qf, kf, vf = fq(q), fq(k), fq(v)
        out = {}
        out["inputs_all_fq"] = err(torch.zeros_like(q, dtype=torch.float32), ref, rows)  # placeholder
        r_all = reference_rows(qf, kf, vf, rows)
        out["inputs_all_fq"] = {"rel_l2": ((r_all - ref).norm() / ref.norm()).item()}
        r_qk = reference_rows(qf, kf, v.float(), rows)
        out["inputs_qk_fq"] = {"rel_l2": ((r_qk - ref).norm() / ref.norm()).item()}
        r_v = reference_rows(q.float(), k.float(), vf, rows)
        out["inputs_v_fq"] = {"rel_l2": ((r_v - ref).norm() / ref.norm()).item()}
        for m in args.modes.split(","):
            o = runner(m, q, k, v, H, D)()
            torch.cuda.synchronize()
            out[m] = {"vs_fp32": err(o, ref, rows), "vs_fq_inputs_ref": err(o, r_all, rows)}
        # logit statistics, to judge how realistic the synthetic inputs are
        lg = (q[:, rows[:64]].float().transpose(1, 2) @ k.float().transpose(1, 2).transpose(-1, -2)) / math.sqrt(D)
        out["logit_std"] = lg.std().item()
        out["logit_rowmax_mean"] = lg.amax(-1).mean().item()
        print(json.dumps(out))
        return

    if args.mode == "check":
        m = args.target
        ref = reference_rows(q, k, v, rows)
        fa = runner("fa2", q, k, v, H, D)()
        fa_err = err(fa, ref, rows)
        del fa
        pristine = [t.clone() for t in (q, k, qkv)]
        fn = runner(m, q, k, v, H, D)
        first = None
        diffs = []
        errs = []
        for i in range(args.warmup + args.iters):
            o = fn()
            torch.cuda.synchronize()
            for name, t, p in zip(("q", "k", "qkv"), (q, k, qkv), pristine):
                if not torch.equal(t, p):
                    diffs.append(f"iter {i}: input {name} modified")
            if first is None:
                first = o.clone()
            elif not torch.equal(o, first):
                diffs.append(f"iter {i}: output differs from first (max {(o.float()-first.float()).abs().max().item()})")
            errs.append(err(o, ref, rows))
        worst = max(errs, key=lambda e: e["rel_l2"])
        # Kernel-only error: FP32 SDPA on the dequantized per-tensor E4M3 inputs.
        import tensorrt_llm  # noqa: F401

        def fq(x):
            xq, ds = torch.ops.trtllm.quantize_e4m3_per_tensor(x.contiguous())
            return (xq.float() * ds.float().reshape(-1)[0]).view(x.shape)

        ref_fq = reference_rows(fq(q), fq(k), fq(v), rows)
        kern = err(first, ref_fq, rows)
        res = {
            "passed": not diffs and math.isfinite(worst["rel_l2"]) and kern["rel_l2"] < 0.02,
            "kernel_err_vs_fp32_on_dequantized_inputs": kern,
            "target": m,
            "compared": "every iteration vs FP32 SDPA math reference on %d query rows; bitwise vs first iteration; inputs vs pristine" % len(rows),
            "err_worst": worst,
            "fa2_bf16_err": fa_err,
            "deterministic": not any("output differs" in d for d in diffs),
            "diffs": diffs[:10],
            "iters": len(errs),
        }
        print(json.dumps(res))
        return

    fn = runner(args.mode, q, k, v, H, D)
    med, nf, ts = time_fn(fn, args.warmup, args.iters)
    log(f"{args.mode}: median {med:.3f} ms noise {nf:.4f} all {[round(t, 3) for t in ts]}")
    o = fn()
    ref = reference_rows(q, k, v, rows)
    e = err(o, ref, rows)
    cls = "exact" if args.mode in ("fa2",) else "mode_switch"
    print(json.dumps({"batch": {"1": {"kernel_ms": med, "noise_floor": nf}},
                      "output": {"class": cls, "err_vs_fp32": e}, "mode": args.mode, "S": S, "H": H}))


if __name__ == "__main__":
    main()
