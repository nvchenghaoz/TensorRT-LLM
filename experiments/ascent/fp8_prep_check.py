"""Untimed bitwise check: fp8_attn_prep.quantize_qkv_e4m3_per_tensor vs main's FP8 attention prep.

Reference = main's prep, still in the tree: FlashInferAttention._quantize_e4m3_per_tensor
(t.contiguous() + C++ torch.ops.trtllm.quantize_e4m3_per_tensor) on Q, K, V, descales via cat.

A. amax sweep: every positive finite bf16 value a (and 0) is the amax of one tensor (random values
   in [-a, a] plus +-a), three tensors per call, V a strided view of a packed buffer. Covers every
   amax/448 -> bf16 rounding boundary and the 448 satfinite edge (x = +-amax).
B. quantize sweep: for 600 amax values, Q holds every finite bf16 value with |x| <= amax.
C. real shape B=1 S=37966 H=28 D=128 on two datasets (gaussian; heavy-tailed with outliers),
   same warmup/iteration count as the timing (10 + 40): every iteration poisons the allocator's
   free blocks (0xFF) first, then compares E4M3 Q/K/V bytes and the 3 descales with the reference,
   and checks qkv, Q and K are bitwise unchanged.
D. FMHA output (with and without LSE) of FlashInferAttention._run_fp8_sm120 (branch prep) vs
   main's _run_fp8_sm120 body (main prep + the same sm120_fmha_fp8_ragged_prefill), 3 repeats.
Last stdout line: {"passed": bool, ...}.
"""

import json
import sys

import torch

S, H, D = 37966, 28, 128
WARMUP, ITERS = 10, 40


def log(*a):
    print(*a, flush=True)


def ref_prep(q, k, v):
    from tensorrt_llm._torch.visual_gen.attention.backends.flashinfer import FlashInferAttention

    qf, qd = FlashInferAttention._quantize_e4m3_per_tensor(q)
    kf, kd = FlashInferAttention._quantize_e4m3_per_tensor(k)
    vf, vd = FlashInferAttention._quantize_e4m3_per_tensor(v)
    return qf, kf, vf, torch.cat([qd, kd, vd])


def same(a, b):
    return a.shape == b.shape and torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def compare(new, ref):
    bad = []
    for name, a, b in zip("QKV", new[:3], ref[:3]):
        if not same(a, b):
            n = (a.view(torch.uint8) != b.view(torch.uint8)).sum().item() if a.shape == b.shape else -1
            bad.append(f"{name} fp8 differs ({n} bytes)")
    if not torch.equal(new[3], ref[3]):
        bad.append(f"descales {new[3].tolist()} vs {ref[3].tolist()}")
    return bad


def main():
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.visual_gen.attention.backends import fp8_attn_prep as prep

    fails = []
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(1234)

    # ---------------- A: exhaustive amax sweep
    bits = torch.arange(0, 0x7F80, dtype=torch.int32)  # +0 .. largest finite bf16
    amaxes = bits.to(torch.int16).view(torch.bfloat16).to(dev)
    n = amaxes.numel()
    log(f"A: {n} amax values")
    a_fail = 0
    rows, hh, dd = 4, 2, 64
    W = hh * dd
    for i0 in range(0, n, 3):
        a3 = amaxes[i0:i0 + 3]
        if a3.numel() < 3:
            a3 = torch.cat([a3, amaxes[:3 - a3.numel()]])
        u = torch.rand(3, rows * W, device=dev, generator=gen) * 2 - 1
        x = (u * a3.float()[:, None]).bfloat16()
        pos = torch.randint(0, rows * W, (3,), device=dev, generator=gen)
        sign = torch.where(torch.rand(3, device=dev, generator=gen) < 0.5, -1.0, 1.0)
        x[torch.arange(3, device=dev), pos] = (a3.float() * sign).bfloat16()
        q = x[0].view(1, rows, hh, dd)
        k = x[1].view(1, rows, hh, dd).clone()
        packed = torch.zeros(1, rows, 3 * W, device=dev, dtype=torch.bfloat16)
        packed[..., 2 * W:] = x[2].view(1, rows, W)
        v = packed[..., 2 * W:].unflatten(-1, (hh, dd))
        assert prep.fp8_qkv_prep_supported(q, k, v)
        bad = compare(prep.quantize_qkv_e4m3_per_tensor(q, k, v), ref_prep(q, k, v))
        if bad:
            a_fail += 1
            if a_fail <= 10:
                log(f"  A mismatch amax={a3.float().tolist()}: {bad}")
        if (i0 // 3) % 2000 == 0:
            log(f"  A progress {i0}/{n}, mismatches {a_fail}")
    log(f"A done: {a_fail} mismatching calls")
    if a_fail:
        fails.append(f"A: {a_fail} amax-sweep calls differ")

    # ---------------- B: quantize sweep, every bf16 value |x| <= amax
    allv = torch.arange(0, 0x7F80, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).to(dev)
    allv = torch.cat([allv, -allv])
    picks = torch.cat([
        torch.tensor([448.0, 1.0, 0.5, 3.0, 12.0, 100.0, 1e-3, 1e4, 65504.0], device=dev),
        torch.exp(torch.empty(591, device=dev).uniform_(-9.0, 11.0, generator=gen)),
    ]).bfloat16()
    b_fail = 0
    for j, a in enumerate(picks):
        vals = allv[allv.float().abs() <= a.float()]
        pad = (-vals.numel()) % W
        vals = torch.cat([vals, torch.zeros(pad, device=dev, dtype=torch.bfloat16)])
        r = vals.numel() // W
        q = vals.view(1, r, hh, dd)
        k = (torch.randn(1, r, hh, dd, device=dev, generator=gen) * 3).bfloat16()
        packed = (torch.randn(1, r, 3 * W, device=dev, generator=gen) * 7).bfloat16()
        v = packed[..., 2 * W:].unflatten(-1, (hh, dd))
        bad = compare(prep.quantize_qkv_e4m3_per_tensor(q, k, v), ref_prep(q, k, v))
        if bad:
            b_fail += 1
            if b_fail <= 10:
                log(f"  B mismatch amax={a.item()}: {bad}")
    log(f"B done: {b_fail}/{picks.numel()} mismatching amax values")
    if b_fail:
        fails.append(f"B: {b_fail} quantize-sweep cases differ")

    # ---------------- C: real shape, repeated sequence with poisoning
    Wf = H * D
    g = torch.Generator(device=dev).manual_seed(0)

    def dataset(kind):
        if kind == "gaussian":  # same as the timing script
            qkv = (torch.randn(1, S, 3 * Wf, device=dev, generator=g) * 2.5).bfloat16()
            q = (torch.randn(1, S, H, D, device=dev, generator=g) * 1.7).bfloat16()
            k = (torch.randn(1, S, H, D, device=dev, generator=g) * 3.1).bfloat16()
        else:  # heavy-tailed, per-channel spread, sparse outliers
            def ht(shape, s):
                x = torch.randn(shape, device=dev, generator=g)
                x = x * torch.exp(torch.randn(shape[-1], device=dev, generator=g) * 0.7) * s
                m = torch.rand(shape, device=dev, generator=g) < 1e-6
                return torch.where(m, x * 40, x).bfloat16()
            qkv = ht((1, S, 3 * Wf), 1.3)
            q = ht((1, S, Wf), 2.0).view(1, S, H, D)
            k = ht((1, S, Wf), 4.0).view(1, S, H, D)
        v = qkv[..., 2 * Wf:].unflatten(-1, (H, D))
        return q, k, v, qkv

    def poison(sizes):
        bufs = [torch.full((s,), 0xFF, dtype=torch.uint8, device=dev) for s in sizes]
        del bufs

    sizes = [S * Wf] * 3 + [4096, 64]
    for kind in ("gaussian", "heavy_tailed"):
        q, k, v, qkv = dataset(kind)
        pristine = [t.clone() for t in (q, k, qkv)]
        ref = ref_prep(q, k, v)
        log(f"C {kind}: ref descales {ref[3].tolist()}")
        c_fail = 0
        for it in range(WARMUP + ITERS):
            poison(sizes)
            out = prep.quantize_qkv_e4m3_per_tensor(q, k, v)
            bad = compare(out, ref)
            if bad:
                c_fail += 1
                log(f"  C {kind} iteration {it}: {bad}")
            for name, t, p in zip(("q", "k", "qkv"), (q, k, qkv), pristine):
                if not torch.equal(t, p):
                    c_fail += 1
                    log(f"  C {kind} iteration {it}: input {name} modified")
            del out
        log(f"C {kind}: {c_fail} failures over {WARMUP + ITERS} iterations")
        if c_fail:
            fails.append(f"C {kind}: {c_fail} failures")
        del pristine, ref

        # ---------------- D: FMHA output
        from flashinfer.attention.cute_dsl.sm120_fmha import sm120_fmha_fp8_ragged_prefill

        from tensorrt_llm._torch.visual_gen.attention.backends.flashinfer import FlashInferAttention
        from tensorrt_llm.visual_gen.args import QuantAttentionConfig

        attn = FlashInferAttention(layer_idx=1, num_heads=H, head_dim=D, dtype=torch.bfloat16,
                                   quant_attention_config=QuantAttentionConfig(qk_dtype="fp8",
                                                                               v_dtype="fp8"))

        def main_run(with_lse):
            qf, kf, vf, ds = ref_prep(q, k, v)
            ds = ds.tolist()
            o = torch.empty(S, H, D, dtype=q.dtype, device=dev)
            lse = torch.empty(S, H, dtype=torch.float32, device=dev) if with_lse else None
            cu = torch.arange(2, dtype=torch.int32, device=dev) * S
            sm120_fmha_fp8_ragged_prefill(qf.view(S, H, D), kf.view(S, H, D), vf.view(S, H, D), o,
                                          cu, cu, max_seqlen_q=S, is_causal=False,
                                          sm_scale=ds[0] * ds[1] * attn.scale, lse=lse,
                                          v_scale=ds[2])
            return o.view(1, S, H, D), lse

        for with_lse in (False, True):
            ro, rl = main_run(with_lse)
            for rep in range(3):
                poison([S * Wf * 2])
                no, nl = attn._run_fp8_sm120(q, k, v, is_causal=False, with_lse=with_lse)
                ok = torch.equal(no, ro)
                if with_lse:
                    ok = ok and torch.equal(nl.contiguous().view(-1),
                                            (rl.view(1, S, H).transpose(1, 2).contiguous()
                                             * __import__("math").log(2.0)).view(-1))
                if not ok:
                    fails.append(f"D {kind} with_lse={with_lse} rep {rep}: FMHA output differs")
                    log(f"  D {kind} with_lse={with_lse} rep {rep}: DIFFERS")
            log(f"D {kind} with_lse={with_lse}: done")
        del q, k, v, qkv

    # ---------------- E: layouts that take the fallback (cross-attention lengths, GQA, B=2 strided)
    e_cases = 0
    for (qs, ks) in (((1, 300, 4, 128), (1, 77, 4, 128)), ((1, 256, 8, 128), (1, 256, 2, 128)),
                     ((2, 256, 4, 128), (2, 256, 4, 128))):
        q = torch.randn(qs, device=dev, generator=gen).bfloat16() * 3
        k = torch.randn(ks, device=dev, generator=gen).bfloat16()
        packed = torch.randn(ks[0], ks[1], 3 * ks[2] * ks[3], device=dev, generator=gen).bfloat16()
        v = packed[..., 2 * ks[2] * ks[3]:].unflatten(-1, (ks[2], ks[3]))
        bad = compare(prep.quantize_qkv_e4m3_per_tensor(q, k, v), ref_prep(q, k, v))
        e_cases += 1
        log(f"E {qs} {ks}: supported={prep.fp8_qkv_prep_supported(q, k, v)} {bad or 'bitwise'}")
        if bad:
            fails.append(f"E {qs}/{ks}: {bad}")

    passed = not fails
    log("FAILURES:" if fails else "all bitwise", fails)
    print(json.dumps({
        "passed": passed,
        "compared": "fp8_attn_prep vs main prep (C++ quantize_e4m3_per_tensor on contiguous "
                    "Q/K/V): E4M3 bytes, 3 descales, FMHA output/LSE; amax sweep over all bf16, "
                    "quantize sweep, real shape 50 poisoned iterations x 2 datasets",
        "failures": fails,
    }))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
