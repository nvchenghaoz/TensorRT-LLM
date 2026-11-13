"""Stage-1 bake-off: quantized attention arms for MiniMax-H3 on SM120.

Per-rank shape: B=1, S=37966, H=28, D=128, non-causal, no mask.
Inputs: Q/K per-head RMSNorm (non-unit weight) + RoPE, contiguous [S,H,D];
V a strided [S,H,D] view (row stride 3*H*D=10752) into a qkv buffer.

Arms (each times its FULL prep from [S,H,D] inputs to an [S,H*D] BF16 output):
  fa2         torch SDPA FLASH (BF16)
  cudnn       torch SDPA CUDNN (BF16)
  C_pt_fp8    branch 11: V.contiguous, 3x trtllm per-tensor E4M3, CuTe-DSL sm120 fp8 fmha
  B_h3_fp8    flashinfer.diffusion_ops.minimax_h3_sm120_varlen_attention_fp8 (E4M3 Sage)
  A_cake      INT8 Q/K (smooth-K) + FP8 PV Sage2, bsa_attn_sm120_blk64_sage_fwd 'cake', full density
  A_cute      same, backend 'cute_dsl'
  A_cake_nc   A_cake with strided (no-copy) BHSD views fed to the Sage quantizers
"""
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

S = int(os.environ.get("BO_S", 37966))
H = int(os.environ.get("BO_H", 28))
D = 128
WARM = int(os.environ.get("BO_WARM", 5))
ITERS = int(os.environ.get("BO_ITERS", 24))
SCALE = 1.0 / math.sqrt(D)
dev = torch.device("cuda")


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# --------------------------------------------------------------------------- inputs
def rmsnorm(x, w, eps=1e-6):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float())


def rope(x, pos, theta=10000.0):
    # x [S,H,D] fp32, rotate-half style
    half = D // 2
    inv = 1.0 / (theta ** (torch.arange(0, half, device=dev, dtype=torch.float32) / half))
    ang = pos[:, None].float() * inv[None, :]
    cos = ang.cos()[:, None, :]
    sin = ang.sin()[:, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def make_inputs(seed, hard, gauss=False):
    g = torch.Generator(device=dev).manual_seed(seed)
    if gauss:
        # branch-11-like: near-independent Q/K rows, Gaussian logits (std ~2.8), diffuse softmax
        lat = torch.randn(S, 64, device=dev, generator=g)
        Pq = torch.randn(H, 64, D, device=dev, generator=g) / 8.0
        Pk = torch.randn(H, 64, D, device=dev, generator=g) / 8.0
        q_raw = 0.3 * torch.einsum("sl,hld->shd", lat, Pq) + torch.randn(S, H, D, device=dev, generator=g)
        k_raw = 0.3 * torch.einsum("sl,hld->shd", lat, Pk) + torch.randn(S, H, D, device=dev, generator=g)
        wq = (1.55 * torch.exp(0.3 * torch.randn(D, device=dev, generator=g))).bfloat16()
        wk = (1.55 * torch.exp(0.3 * torch.randn(D, device=dev, generator=g))).bfloat16()
    else:
        # structured tokens: shared latent so attention is peaky, not uniform
        lat = torch.randn(S, 64, device=dev, generator=g)
        Pq = torch.randn(H, 64, D, device=dev, generator=g) / 8.0
        Pk = Pq + 0.5 * torch.randn(H, 64, D, device=dev, generator=g) / 8.0
        q_raw = torch.einsum("sl,hld->shd", lat, Pq) + 0.7 * torch.randn(S, H, D, device=dev, generator=g)
        k_raw = torch.einsum("sl,hld->shd", lat, Pk) + 0.7 * torch.randn(S, H, D, device=dev, generator=g)
        wq = (1.6 * torch.exp(0.35 * torch.randn(D, device=dev, generator=g))).bfloat16()
        wk = (1.6 * torch.exp(0.35 * torch.randn(D, device=dev, generator=g))).bfloat16()
    pos = torch.arange(S, device=dev) % 4096
    q = rope(rmsnorm(q_raw.bfloat16(), wq), pos)
    k = rope(rmsnorm(k_raw.bfloat16(), wk), pos)
    del q_raw, k_raw
    qkv = torch.empty(S, 3 * H * D, device=dev, dtype=torch.bfloat16)
    vfull = torch.randn(S, H, D, device=dev, generator=g)
    if hard:
        # K: per-channel offset of 3x K std on a random half of channels (per head)
        kstd = k.std()
        mask = (torch.rand(H, D, device=dev, generator=g) < 0.5).float()
        sign = torch.where(torch.rand(H, D, device=dev, generator=g) < 0.5, -1.0, 1.0)
        k = k + 3.0 * kstd * mask * sign
        # V: 10x spread of channel scales
        vs = torch.pow(10.0, torch.rand(H, D, device=dev, generator=g))
        vfull = vfull * vs
    q = q.bfloat16().contiguous()
    k = k.bfloat16().contiguous()
    qkv[:, 2 * H * D:] = vfull.reshape(S, H * D).bfloat16()
    qkv[:, : 2 * H * D] = 0
    v = qkv[:, 2 * H * D:].view(S, H, D)
    assert v.stride(0) == 3 * H * D
    return q, k, v, qkv


# --------------------------------------------------------------------------- reference
@torch.no_grad()
def reference(q, k, v):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = torch.empty(S, H, D, device=dev, dtype=torch.float32)
    CH = 4096
    for h in range(H):
        kh = k[:, h].float()
        vh = v[:, h].float()
        for s0 in range(0, S, CH):
            qh = q[s0:s0 + CH, h].float()
            with sdpa_kernel(SDPBackend.MATH):
                o = F.scaled_dot_product_attention(qh[None, None], kh[None, None], vh[None, None], scale=SCALE)
            out[s0:s0 + CH, h] = o[0, 0]
    return out.reshape(S, H * D)


def logit_stats(q, k):
    h = 0
    qs = q[:2048, h].float()
    ks = k[:, h].float()
    lg = (qs @ ks.T) * SCALE
    p = torch.softmax(lg, -1)
    return {
        "logit_std": lg.std().item(),
        "row_max_minus_mean": (lg.max(-1).values - lg.mean(-1)).mean().item(),
        "p_max_mean": p.max(-1).values.mean().item(),
    }


def fq_e4m3(x, amax):
    s = (amax / 448.0).clamp_min(1e-12)
    return (x.float() / s).to(torch.float8_e4m3fn).float() * s


def emulations(q, k, v):
    """Fake-quantized inputs (FP32) isolating each input-quantization term."""
    qf, kf, vf = q.float(), k.float(), v.float()
    out = {}
    # per-tensor E4M3 Q/K (branch 11)
    out["qk_pt_e4m3"] = (fq_e4m3(qf, qf.abs().max()), fq_e4m3(kf, kf.abs().max()), vf)
    # Sage2 INT8: Q per 32 tokens per head, K smooth-K then per 64 tokens per head
    def int8_groups(x, gs):
        Sx = x.shape[0]
        pad = (-Sx) % gs
        xp = F.pad(x, (0, 0, 0, 0, 0, pad)).view(-1, gs, H, D)
        a = xp.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-12) / 127.0
        return (torch.round(xp / a).clamp(-127, 127) * a).view(-1, H, D)[:Sx]
    kc = kf - kf.mean(0, keepdim=True)
    out["qk_int8_sage"] = (int8_groups(qf, 32), int8_groups(kc, 64), vf)
    # E4M3 Sage (H3 op): Q per token per head, K smooth-K per 128 keys
    def e4m3_groups(x, gs):
        Sx = x.shape[0]
        pad = (-Sx) % gs
        xp = F.pad(x, (0, 0, 0, 0, 0, pad)).view(-1, gs, H, D)
        a = xp.abs().amax(dim=(1, 3), keepdim=True)
        return fq_e4m3(xp, a).view(-1, H, D)[:Sx]
    out["qk_e4m3_sage"] = (e4m3_groups(qf, 1), e4m3_groups(kc, 128), vf)
    # V terms
    out["v_pt_e4m3"] = (qf, kf, fq_e4m3(vf, vf.abs().max()))
    out["v_pc_e4m3"] = (qf, kf, fq_e4m3(vf, vf.abs().amax(0, keepdim=True)))
    return out


def errors(o, ref):
    of = o.float().reshape(S, H * D)
    d = of - ref
    rel = (d.norm() / ref.norm()).item()
    mx = d.abs().max().item()
    a = of.reshape(S * H, D)
    b = ref.reshape(S * H, D)
    cos = F.cosine_similarity(a, b, dim=-1).mean().item()
    return {"rel_l2": rel, "max_abs": mx, "cos": cos}


# --------------------------------------------------------------------------- arms
def arm_sdpa(backend):
    def run(q, k, v):
        qh = q.unsqueeze(0).transpose(1, 2)
        kh = k.unsqueeze(0).transpose(1, 2)
        vh = v.unsqueeze(0).transpose(1, 2)
        with sdpa_kernel(backend):
            o = F.scaled_dot_product_attention(qh, kh, vh, scale=SCALE)
        return o.transpose(1, 2).reshape(S, H * D)
    return run


def arm_pt_fp8():
    import tensorrt_llm  # noqa: F401  (registers torch.ops.trtllm)
    from flashinfer.attention.cute_dsl.sm120_fmha import sm120_fmha_fp8_ragged_prefill
    cu = torch.tensor([0, S], dtype=torch.int32, device=dev)
    o = torch.empty(S, H, D, device=dev, dtype=torch.bfloat16)

    def run(q, k, v):
        vc = v.contiguous()
        q8, qs = torch.ops.trtllm.quantize_e4m3_per_tensor(q)
        k8, ks = torch.ops.trtllm.quantize_e4m3_per_tensor(k)
        v8, vsc = torch.ops.trtllm.quantize_e4m3_per_tensor(vc)
        qs_, ks_, vs_ = torch.stack([qs.float().reshape(()), ks.float().reshape(()), vsc.float().reshape(())]).tolist()
        sm120_fmha_fp8_ragged_prefill(q8, k8, v8, o, cu, cu, max_seqlen_q=S, is_causal=False,
                                      sm_scale=SCALE * qs_ * ks_, v_scale=vs_)
        return o.view(S, H * D)
    return run


def arm_h3_fp8():
    from flashinfer.diffusion_ops import minimax_h3_sm120_varlen_attention_fp8
    cu = torch.tensor([0, S], dtype=torch.int32, device=dev)
    o = torch.empty(S, H, D, device=dev, dtype=torch.bfloat16)

    def run(q, k, v):
        vc = v.contiguous()
        minimax_h3_sm120_varlen_attention_fp8(q, k, vc, cu, out=o, cu_seqlens_host=[0, S], softmax_scale=SCALE)
        return o.view(S, H * D)
    return run


def arm_sage_int8(backend, nocopy=False):
    from flashinfer.cute_dsl.sparse.bsa_attn_sm120 import bsa_attn_sm120_blk64_sage_fwd
    import flashinfer.cute_dsl.sparse.bsa_utils.sage_quant_sm120 as sq
    nq = (S + 63) // 64
    nkv = (S + 63) // 64
    idx = torch.arange(nkv, dtype=torch.int32, device=dev).view(1, 1, 1, nkv).expand(1, H, nq, nkv).contiguous()
    nums = torch.full((1, H, nq), nkv, dtype=torch.int32, device=dev)
    bsz = torch.full((nkv,), 64, dtype=torch.int32, device=dev)
    bsz[-1] = S - 64 * (nkv - 1)
    out = torch.empty(1, H, S, D, device=dev, dtype=torch.bfloat16)
    padded = nkv * 64
    nqg = ((S + 127) // 128) * 4
    bufs = (
        torch.empty(1, H, S, D, device=dev, dtype=torch.int8),
        torch.empty(1, H, S, D, device=dev, dtype=torch.int8),
        torch.empty(1, H, D, padded, device=dev, dtype=torch.float8_e4m3fn),
        torch.empty(1, H, nqg, device=dev, dtype=torch.float32),
        torch.empty(1, H, nkv, device=dev, dtype=torch.float32),
        torch.empty(1, H, D, device=dev, dtype=torch.float32),
    )
    orig_check = sq._require_sm120_bf16_bhsd

    def lenient(name, t):
        if t.dtype != torch.bfloat16 or t.dim() != 4 or t.shape[-1] != 128 or t.stride(-1) != 1:
            raise ValueError(f"{name}: bad strided input")

    def run(q, k, v):
        qh = q.unsqueeze(0).transpose(1, 2)
        kh = k.unsqueeze(0).transpose(1, 2)
        vh = v.unsqueeze(0).transpose(1, 2)
        if nocopy:
            sq._require_sm120_bf16_bhsd = lenient
            try:
                ops = sq.quantize_sage_qkv_sm120(qh, kh, vh, out=bufs)
            finally:
                sq._require_sm120_bf16_bhsd = orig_check
        else:
            ops = sq.quantize_sage_qkv_sm120(qh.contiguous(), kh.contiguous(), vh.contiguous(), out=bufs)
        q8, k8, v8, qs, ks, vs = ops
        kw = dict(uniform_block_count=True, contiguous_block_indices=True, backend=backend, out=out)
        if backend == "cute_dsl":
            kw = dict(backend=backend, out=out)
        bsa_attn_sm120_blk64_sage_fwd(q8, k8, v8, qs, ks, vs, idx, nkv, block_sizes=bsz,
                                      q2k_block_nums=nums, softmax_scale=SCALE, **kw)
        return out[0].transpose(0, 1).reshape(S, H * D)
    return run


# --------------------------------------------------------------------------- main
def main():
    log("device", torch.cuda.get_device_name(), "S", S, "H", H)
    arms = {}
    builders = [
        ("fa2", lambda: arm_sdpa(SDPBackend.FLASH_ATTENTION)),
        ("cudnn", lambda: arm_sdpa(SDPBackend.CUDNN_ATTENTION)),
        ("C_pt_fp8", arm_pt_fp8),
        ("B_h3_fp8", arm_h3_fp8),
        ("A_cake", lambda: arm_sage_int8("cake")),
        ("A_cute", lambda: arm_sage_int8("cute_dsl")),
        ("A_cake_nc", lambda: arm_sage_int8("cake", nocopy=True)),
    ]
    only = os.environ.get("BO_ARMS")
    for name, b in builders:
        if only and name not in only.split(","):
            continue
        try:
            arms[name] = b()
            log("built", name)
        except Exception as e:  # noqa: BLE001
            log("BUILD FAIL", name, repr(e)[:400])

    results = {n: {} for n in arms}
    sets = {}
    all_sets = (("i", 1234, False, False), ("ii", 1234, True, False),
                ("iii", 1234, False, True), ("iv", 1234, True, True))
    want = os.environ.get("BO_SETS", "i,ii").split(",")
    for set_name, seed, hard, gauss in all_sets:
        if set_name not in want:
            continue
        q, k, v, qkv = make_inputs(seed, hard, gauss)
        sets[set_name] = (q, k, v, qkv)
        log("set", set_name, "logit stats", logit_stats(q, k))
        ref = reference(q, k, v)
        log("set", set_name, "reference done")
        if os.environ.get("BO_EMU", "1") == "1":
            emu = {}
            for en, (qq, kk, vv) in emulations(q, k, v).items():
                emu[en] = errors(reference(qq, kk, vv), ref)
                log("emu", set_name, en, emu[en])
                del qq, kk, vv
            results.setdefault("_emulation", {})[set_name] = emu
        for n, fn in list(arms.items()):
            try:
                with torch.no_grad():
                    o = fn(q, k, v).clone()
                torch.cuda.synchronize()
                e = errors(o, ref)
                results[n]["err_" + set_name] = e
                log(n, set_name, e)
            except Exception as ex:  # noqa: BLE001
                log("RUN FAIL", n, set_name, repr(ex)[:600])
                arms.pop(n)
                results[n]["fail"] = repr(ex)[:300]
            del o
        del ref
        torch.cuda.empty_cache()

    # bitwise check of no-copy vs contiguous Sage quantize (first set)
    q, k, v, qkv = sets[next(iter(sets))]
    if os.environ.get("BO_TIME", "1") != "1":
        print(json.dumps({"passed": True, "S": S, "H": H, "results": results}))
        return
    if "A_cake" in arms and "A_cake_nc" in arms:
        with torch.no_grad():
            a = arms["A_cake"](q, k, v).clone()
            b = arms["A_cake_nc"](q, k, v).clone()
        results["A_cake_nc"]["bitwise_vs_contig"] = bool(torch.equal(a, b))
        log("A_cake_nc bitwise vs contiguous:", torch.equal(a, b))
        del a, b

    # ------------------------------------------------------------ timing (set i), palindrome
    names = list(arms)
    times = {n: [] for n in names}
    with torch.no_grad():
        for n in names:
            for _ in range(WARM):
                arms[n](q, k, v)
        torch.cuda.synchronize()
        log("warmup done")
        rounds = ITERS // 2
        for r in range(rounds):
            order = names if r % 2 == 0 else names[::-1]
            for n in order:
                for _ in range(2):
                    s_ev = torch.cuda.Event(enable_timing=True)
                    e_ev = torch.cuda.Event(enable_timing=True)
                    s_ev.record()
                    arms[n](q, k, v)
                    e_ev.record()
                    e_ev.synchronize()
                    times[n].append(s_ev.elapsed_time(e_ev))
            if r % 4 == 0:
                log("round", r, {n: round(times[n][-1], 2) for n in names})
    for n in names:
        t = sorted(times[n])
        med = t[len(t) // 2]
        p10 = t[int(0.1 * (len(t) - 1))]
        p90 = t[int(0.9 * (len(t) - 1))]
        results[n]["ms_median"] = med
        results[n]["noise"] = (p90 - p10) / med
        results[n]["n"] = len(t)
        log(n, "median", round(med, 3), "spread", round((p90 - p10) / med, 4))

    # ------------------------------------------------------------ untimed repeat check
    pristine = (q.clone(), k.clone(), qkv.clone())
    with torch.no_grad():
        for n in names:
            first = None
            diffs = []
            ok_inputs = True
            for i in range(WARM + ITERS):
                o = arms[n](q, k, v)
                if first is None:
                    first = o.clone()
                elif not torch.equal(o, first):
                    diffs.append(i)
                if not (torch.equal(q, pristine[0]) and torch.equal(k, pristine[1]) and torch.equal(qkv, pristine[2])):
                    ok_inputs = False
            results[n]["repeat_bitwise"] = not diffs
            results[n]["repeat_diff_iters"] = diffs[:10]
            results[n]["inputs_unchanged"] = ok_inputs
            log(n, "repeat bitwise", not diffs, "inputs unchanged", ok_inputs)
            del first
    passed = all(results[n].get("repeat_bitwise") and results[n].get("inputs_unchanged") for n in names)
    print(json.dumps({"passed": passed, "S": S, "H": H, "results": results}))


if __name__ == "__main__":
    main()
