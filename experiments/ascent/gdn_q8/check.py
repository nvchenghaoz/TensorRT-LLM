# Correctness + repeated-execution check for idea 37 (8-bit GDN state pool).
#
# A. Round trip: q8_gather_dequant(q8_quant_scatter(x)) error <= 0.5 * scale
#    per element (int8-row), zero rows exact, negative indices untouched.
# B. Decode through Qwen3NextGatedDeltaNet.forward_decode in a CUDA graph of
#    STEPS sequential steps on one int8 pool (T1 at N=64/104/128, T1-norm at
#    N=32). WARMUP + ITERS iterations, each: restore the pool from a pristine
#    copy, replay. After every iteration: outputs and pool bitwise equal to
#    iteration 1; state after STEPS steps vs a STEPS-step FP32 reference for
#    every sequence/head/row within the format's round-trip bound
#    (row L2 error <= sqrt(K) * 0.5 * sum_t scale_t(row): the per-row update
#    map h -> exp(g) h (I - beta k k^T) is non-expansive in L2), outputs vs
#    the reference within that bound times ||q||; untouched slots bitwise
#    unchanged. Main's bf16 pool path error vs its own FP32 reference is
#    reported alongside.
# C. Mixed step: forward_extend with [2048, 2048, 2048, 1920] context tokens
#    from a snapshot state + 124 decode riders (riders -> T1, contexts ->
#    one-pass + FlashInfer chunk with dequant-gather / quant-scatter). Same
#    repeated-execution protocol; final states vs a sequential FP32 reference
#    from the dequantized initial states and the kernel's own q/k/v/g/beta.
import json
import math
import sys
import types

import torch

from tensorrt_llm._torch.modules.fla import gdn_state_q8 as q8mod
from tensorrt_llm._torch.modules.mamba import gdn_mixer
from tensorrt_llm._torch.modules.mamba.causal_conv1d import \
    causal_conv1d_update
from tensorrt_llm._torch.modules.mamba.gdn_prefill_onepass import \
    gdn_prefill_onepass

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

H, HV, K, V = 16, 48, 128, 128
KEY_DIM, VAL_DIM = H * K, HV * V
CONV_DIM = 2 * KEY_DIM + VAL_DIM
QKVZ_DIM = CONV_DIM + VAL_DIM
SLOT_BF16 = 817152
ROWS8 = V + 4
SLOT_Q8 = HV * ROWS8 * K + 61440
EPS = 1e-6
STEPS = 3
WARMUP, ITERS = 10, 50
FMT = q8mod.Q8_FORMAT
QMAX = 127.0 if FMT == 0 else 448.0
dev = torch.device("cuda")
results = dict(format="int8-row" if FMT == 0 else "e4m3-row")
ok_all = True


def log(*a):
    print(*a, flush=True)


def fail(msg):
    global ok_all
    ok_all = False
    log("FAIL:", msg)


def elem_err_bound(scale):
    # per-element quantization error bound for one row with this scale
    # (1 + 2^-10): q = rint(x * (qmax / amax)) and x' = q * (amax / qmax)
    # each round once in fp32.
    if FMT == 0:
        return 0.5 * scale * (1.0 + 2.0**-10)
    return scale * QMAX * 2.0**-4  # e4m3: 3 mantissa bits, RNE


def q8_pool(slots):
    flat = torch.zeros(slots * SLOT_Q8, dtype=torch.int8, device=dev)
    pool = flat.as_strided((slots, HV, ROWS8, K), (SLOT_Q8, ROWS8 * K, K, 1))
    return flat, pool


def bf16_pool(slots):
    flat = torch.zeros(slots * SLOT_BF16, dtype=torch.bfloat16, device=dev)
    pool = flat.as_strided((slots, HV, V, K), (SLOT_BF16, V * K, K, 1))
    return flat, pool


def scales_of(pool, idx):
    # [n, HV, V] fp32 row scales of slots idx
    raw = pool[idx.long()][:, :, V:, :].contiguous()
    return raw.view(torch.float32).reshape(idx.shape[0], HV, V)


def fill_q8(pool, slots, gen, std=0.2):
    for s0 in range(0, slots, 16):
        s1 = min(slots, s0 + 16)
        src = torch.randn(s1 - s0, HV, V, K, device=dev, generator=gen) * std
        q8mod.q8_quant_scatter(
            src, pool, torch.arange(s0, s1, dtype=torch.int32, device=dev))


def make_fake(gen):
    A = torch.exp(torch.rand(HV, device=dev, generator=gen) * 6.0 - 5.3)
    fake = types.SimpleNamespace(
        key_dim_per_tp=KEY_DIM, num_k_heads_per_tp=H, head_k_dim=K,
        num_v_heads_per_tp=HV, head_v_dim=V, A_log=torch.log(A).float(),
        dt_bias=(torch.randn(HV, device=dev, generator=gen) * 0.5 -
                 1.0).float(),
        conv1d=types.SimpleNamespace(
            weight=(torch.randn(CONV_DIM, 4, device=dev, generator=gen) *
                    0.3).to(torch.bfloat16), bias=None),
        activation="silu")
    cls = gdn_mixer.Qwen3NextGatedDeltaNet
    fake._forward_extend_q8 = types.MethodType(cls._forward_extend_q8, fake)
    return fake


def ref_gates(a, b, A_log, dt_bias):
    x = a.float() + dt_bias
    sp = torch.where(x <= 20.0, torch.log1p(torch.exp(x)), x)
    return -torch.exp(A_log) * sp, torch.sigmoid(b.float())


def ref_step(h, q, k, v, g, beta, qk_norm):
    # h [n,HV,V,K]; q,k [n,H,K]; v [n,HV,V]; g, beta [n,HV]
    q = q.float()
    k = k.float()
    if qk_norm:
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
    q = q * K**-0.5
    rep = HV // H
    q = q.repeat_interleave(rep, dim=1)
    k = k.repeat_interleave(rep, dim=1)
    h = h * torch.exp(g)[..., None, None]
    vp = (v.float() - torch.einsum("nhvk,nhk->nhv", h, k)) * beta[..., None]
    h = h + vp[..., :, None] * k[..., None, :]
    o = torch.einsum("nhvk,nhk->nhv", h, q)
    return h, o, q


def ref_norm(o, z, w):
    o = o.to(torch.bfloat16).float()
    rstd = torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + EPS)
    y = o * rstd * w.float()
    return y * (z.float() * torch.sigmoid(z.float()))


# ---------------------------------------------------------------- A
def check_round_trip():
    gen = torch.Generator(device=dev)
    gen.manual_seed(1)
    slots = 6
    flat, pool = q8_pool(slots)
    x = torch.randn(slots, HV, V, K, device=dev, generator=gen) * torch.exp(
        torch.randn(slots, HV, V, 1, device=dev, generator=gen) * 2)
    x[1, 3, 5] = 0.0
    idx = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=dev)
    q8mod.q8_quant_scatter(x[:4].contiguous(), pool, idx)
    before = flat.clone()
    neg = torch.tensor([-1], dtype=torch.int32, device=dev)
    q8mod.q8_quant_scatter(x[4:5].contiguous(), pool, neg)
    untouched = bool(torch.equal(before, flat))
    y = q8mod.q8_gather_dequant(pool, idx, V)
    s = scales_of(pool, idx)
    err = (y - x[:4]).abs()
    bound = elem_err_bound(s)[..., None] * (1 + 1e-6) + 1e-30
    within = bool((err <= bound).all())
    zero_ok = bool((y[1, 3, 5] == 0).all()) and float(s[1, 3, 5]) == 0.0
    yn = q8mod.q8_gather_dequant(pool, neg, V)
    neg_zero = bool((yn == 0).all())
    amax_ok = bool(torch.allclose(
        s, x[:4].abs().amax(-1) / QMAX, rtol=1e-6, atol=0))
    r = dict(within_bound=within, zero_row_exact=zero_ok,
             neg_index_untouched=untouched, neg_gather_zero=neg_zero,
             scale_is_amax_over_qmax=amax_ok,
             max_rel_err=float((err / (x[:4].abs().amax(-1, keepdim=True) +
                                       1e-30)).max()))
    log("A round trip", r)
    if not (within and zero_ok and untouched and neg_zero and amax_ok):
        fail("round trip")
    results["round_trip"] = r


# ---------------------------------------------------------------- B
def decode_inputs(n, gen):
    qkvz = (torch.randn(n, QKVZ_DIM, device=dev, generator=gen) * 0.7).to(
        torch.bfloat16)
    ba = (torch.randn(n, 2 * HV, device=dev, generator=gen) * 1.5).to(
        torch.bfloat16)
    return dict(qkvz=qkvz, a=ba[:, HV:], b=ba[:, :HV],
                z=qkvz[:, CONV_DIM:].view(n, HV, V))


def decode_call(fake, pool, inp, idx, cu, norm, norm_w):
    kw = dict(mixed_qkv=inp["qkvz"][:, :CONV_DIM], a=inp["a"], b=inp["b"],
              cache_indices=idx, num_decodes=idx.shape[0], conv_fused=True)
    if norm:
        kw["norm_fusion"] = (inp["z"], norm_w, EPS)
    return gdn_mixer.Qwen3NextGatedDeltaNet.forward_decode(
        fake, None, pool, cu, **kw)


def check_decode(n, norm):
    tag = ("t1norm_" if norm else "t1_") + str(n)
    gen = torch.Generator(device=dev)
    gen.manual_seed(100 + n + (1000 if norm else 0))
    slots = n + 4
    flat, pool = q8_pool(slots)
    fill_q8(pool, slots, gen)
    flat0 = flat.clone()
    fake = make_fake(gen)
    norm_w = (1.0 + 0.1 * torch.randn(V, device=dev, generator=gen)).to(
        torch.bfloat16)
    idx = torch.randperm(slots, device=dev, generator=gen)[:n].to(torch.int32)
    untouched = sorted(set(range(slots)) - set(idx.tolist()))
    cu = torch.arange(n + 1, dtype=torch.long, device=dev)
    inps = [decode_inputs(n, gen) for _ in range(STEPS)]

    # FP32 reference from the dequantized pristine state.
    h = q8mod.q8_gather_dequant(pool, idx, V)
    h_init = h.clone()
    ref_o, ref_q = [], []
    for inp in inps:
        mq = inp["qkvz"][:, :CONV_DIM]
        g, beta = ref_gates(inp["a"], inp["b"], fake.A_log, fake.dt_bias)
        h, o, qs = ref_step(h, mq[:, :KEY_DIM].view(n, H, K),
                            mq[:, KEY_DIM:2 * KEY_DIM].view(n, H, K),
                            mq[:, 2 * KEY_DIM:].view(n, HV, V), g, beta, True)
        ref_o.append(ref_norm(o, inp["z"], norm_w) if norm else o)
        ref_q.append(qs)
    ref_h = h

    # Eager step-by-step pass: per-step scales for the bound.
    step_scales = []
    eager_outs = []
    for inp in inps:
        eager_outs.append(
            decode_call(fake, pool, inp, idx, cu, norm, norm_w).clone())
        step_scales.append(scales_of(pool, idx))
    eager_pool = flat.clone()
    flat.copy_(flat0)

    # CUDA graph of STEPS sequential steps.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for inp in inps:
            decode_call(fake, pool, inp, idx, cu, norm, norm_w)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    flat.copy_(flat0)
    outs = []
    g_ = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_):
        for inp in inps:
            outs.append(decode_call(fake, pool, inp, idx, cu, norm, norm_w))
    torch.cuda.synchronize()

    sqrtk = math.sqrt(K)
    row_bound = sum(sqrtk * elem_err_bound(st) for st in step_scales)
    first = None
    per_iter = []
    det_ok = True
    for it in range(WARMUP + ITERS):
        flat.copy_(flat0)
        g_.replay()
        torch.cuda.synchronize()
        o_now = [o.clone() for o in outs]
        if first is None:
            first = ([o.clone() for o in o_now], flat.clone())
        same = all(torch.equal(a_, b_) for a_, b_ in zip(o_now, first[0])) \
            and torch.equal(flat, first[1])
        det_ok = det_ok and same
        # state vs k-step fp32 reference, before the next reset
        hq = q8mod.q8_gather_dequant(pool, idx, V)
        row_err = (hq - ref_h).norm(dim=-1)
        state_ok = bool((row_err <= row_bound * 1.001 + 1e-5).all())
        unt_ok = all(torch.equal(pool[u], flat0.as_strided(
            pool.shape, pool.stride())[u]) for u in untouched)
        o_last = o_now[-1].float().reshape(n, HV, V)
        o_err = (o_last - ref_o[-1]).abs()
        if not norm:
            # |e_row . q| <= ||e_row|| ||q|| ; + bf16 output rounding
            o_bound = (row_bound * ref_q[-1].norm(dim=-1)[..., None] * 1.001 +
                       ref_o[-1].abs() * 2.0**-8 + 1e-5)
            o_ok = bool((o_err <= o_bound).all())
        else:
            o_ok = bool(torch.isfinite(o_last).all())
        per_iter.append(dict(
            it=it, same_as_iter1=bool(same), state_ok=state_ok,
            untouched_ok=bool(unt_ok), out_ok=o_ok,
            max_state_err=float((hq - ref_h).abs().max()),
            max_out_err=float(o_err.max())))
        if not (same and state_ok and unt_ok and o_ok):
            fail("%s iteration %d same=%s state=%s untouched=%s out=%s" %
                 (tag, it, same, state_ok, unt_ok, o_ok))
    eager_same = torch.equal(eager_pool, first[1]) and all(
        torch.equal(a_, b_) for a_, b_ in zip(eager_outs, first[0]))

    # Main's bf16 path on the same (bf16-rounded) initial state.
    fb, pb = bf16_pool(slots)
    hb0 = torch.zeros(slots, HV, V, K, device=dev)
    hb0[idx.long()] = h_init
    pb.copy_(hb0.to(torch.bfloat16))
    hb = pb[idx.long()].float()
    for inp in inps:
        ob = decode_call(fake, pb, inp, idx, cu, norm, norm_w)
        mq = inp["qkvz"][:, :CONV_DIM]
        g, beta = ref_gates(inp["a"], inp["b"], fake.A_log, fake.dt_bias)
        hb, orb, _ = ref_step(hb, mq[:, :KEY_DIM].view(n, H, K),
                              mq[:, KEY_DIM:2 * KEY_DIM].view(n, H, K),
                              mq[:, 2 * KEY_DIM:].view(n, HV, V), g, beta,
                              True)
    if norm:
        orb = ref_norm(orb, inps[-1]["z"], norm_w)
    main_state_err = float((pb[idx.long()].float() - hb).abs().max())
    main_out_err = float((ob.float().reshape(n, HV, V) - orb).abs().max())
    hq = q8mod.q8_gather_dequant(pool, idx, V)
    r = dict(
        iterations=len(per_iter), deterministic=det_ok,
        graph_equals_eager=bool(eager_same),
        all_state_within_bound=all(p["state_ok"] for p in per_iter),
        all_untouched_ok=all(p["untouched_ok"] for p in per_iter),
        all_out_ok=all(p["out_ok"] for p in per_iter),
        q8_max_state_err=per_iter[-1]["max_state_err"],
        q8_max_out_err=per_iter[-1]["max_out_err"],
        q8_mean_state_err=float((hq - ref_h).abs().mean()),
        q8_max_rel_row_err=float(((hq - ref_h).norm(dim=-1) /
                                  (ref_h.norm(dim=-1) + 1e-30)).max()),
        max_row_bound_used=float(row_bound.max()),
        state_err_drift=per_iter[-1]["max_state_err"] -
        per_iter[0]["max_state_err"],
        main_bf16_max_state_err=main_state_err,
        main_bf16_max_out_err=main_out_err,
        ref_state_absmax=float(ref_h.abs().max()),
        ref_out_absmax=float(ref_o[-1].abs().max()))
    log("B", tag, r)
    if not (det_ok and eager_same):
        fail(tag + " determinism / graph==eager")
    results[tag] = r
    del g_


# ---------------------------------------------------------------- C
def check_mixed():
    P = [2048, 2048, 2048, 1920]
    R = 124
    nseq = len(P) + R
    slots = nseq + 8
    lens = P + [1] * R
    T = sum(lens)
    NP = sum(P)
    gen = torch.Generator(device=dev)
    gen.manual_seed(4242)
    flat, pool = q8_pool(slots)
    fill_q8(pool, slots, gen)
    flat0 = flat.clone()
    conv = (torch.randn(slots, CONV_DIM, 3, device=dev, generator=gen) *
            0.5).to(torch.bfloat16)
    conv0 = conv.clone()
    fake = make_fake(gen)
    qkvz = (torch.randn(T, QKVZ_DIM, device=dev, generator=gen) * 0.7).to(
        torch.bfloat16)
    # causal_conv1d_update convolves the decode rows of mixed_qkv in place:
    # a mutable input, restored before every iteration.
    qkvz0 = qkvz.clone()
    ba = (torch.randn(T, 2 * HV, device=dev, generator=gen) * 1.5).to(
        torch.bfloat16)
    a, b = ba[:, HV:], ba[:, :HV]
    qsl_l = [0]
    for ln in lens:
        qsl_l.append(qsl_l[-1] + ln)
    qsl = torch.tensor(qsl_l, dtype=torch.int32, device=dev)
    qsl_long = qsl.long()
    idx = torch.randperm(slots, device=dev, generator=gen)[:nseq].to(
        torch.int32)
    untouched = sorted(set(range(slots)) - set(idx.tolist()))
    has_init = torch.ones(nseq, dtype=torch.bool, device=dev)
    mixed_qkv = qkvz[:, :CONV_DIM]

    def step():
        kw = dict(
            mixed_qkv=mixed_qkv, a=a, b=b, batch_size=nseq,
            has_initial_states=has_init, cache_indices=idx,
            query_start_loc=qsl, query_start_loc_long=qsl_long,
            num_prefill_tokens=NP, num_decode_tokens=R,
            state_indices_p=idx[:len(P)], state_indices_d=idx[len(P):],
            num_prefill=len(P), num_decodes=R, state_workspace=None,
            intermediate_state_indices=None, use_replay=False)
        return gdn_mixer.Qwen3NextGatedDeltaNet.forward_extend(
            fake, conv, pool, **kw)

    # Reference inputs: the kernel path's own front-end outputs on clones.
    cconv = conv0.clone()
    mp, md = torch.split(qkvz0.clone()[:, :CONV_DIM], [NP, R], dim=0)
    md = causal_conv1d_update(md, cconv, fake.conv1d.weight, None,
                              activation="silu",
                              conv_state_indices=idx[len(P):])
    qp, kp, vp, gp, bp = gdn_prefill_onepass(
        mp, None, fake.conv1d.weight, None, cconv, qsl[:len(P) + 1],
        idx[:len(P)], has_init[:len(P)], a, b, fake.A_log, fake.dt_bias, H,
        K, HV, V, activation="silu")
    h0 = q8mod.q8_gather_dequant(pool, idx, V)
    # prefill: sequential fp32 recurrence per sequence
    ref_hp = []
    ref_op = torch.empty(NP, HV, V, device=dev)
    for i, ln in enumerate(P):
        h = h0[i:i + 1].clone()
        st = qsl_l[i]
        for t in range(st, st + ln):
            h, o, _ = ref_step(h, qp[0, t:t + 1], kp[0, t:t + 1],
                               vp[0, t:t + 1], gp[0, t:t + 1],
                               bp[0, t:t + 1], False)
            ref_op[t] = o[0]
        ref_hp.append(h[0])
    ref_hp = torch.stack(ref_hp)
    # riders: one T1-style step
    g_r, beta_r = ref_gates(a[NP:], b[NP:], fake.A_log, fake.dt_bias)
    ref_hd, ref_od, ref_qd = ref_step(
        h0[len(P):], md[:, :KEY_DIM].view(R, H, K),
        md[:, KEY_DIM:2 * KEY_DIM].view(R, H, K),
        md[:, 2 * KEY_DIM:].view(R, HV, V), g_r, beta_r, True)
    log("C reference built")

    first = None
    per_iter = []
    det_ok = True
    sqrtk = math.sqrt(K)
    for it in range(WARMUP + ITERS):
        flat.copy_(flat0)
        conv.copy_(conv0)
        qkvz.copy_(qkvz0)
        out = step()
        torch.cuda.synchronize()
        if first is None:
            first = (out.clone(), flat.clone(), conv.clone())
            sc = scales_of(pool, idx)
        same = (torch.equal(out, first[0]) and torch.equal(flat, first[1])
                and torch.equal(conv, first[2]))
        det_ok = det_ok and same
        hq = q8mod.q8_gather_dequant(pool, idx, V)
        # prefill: single final rounding + chunk-vs-sequential fp32 slack
        rerr_p = (hq[:len(P)] - ref_hp).norm(dim=-1)
        bnd_p = (sqrtk * elem_err_bound(sc[:len(P)]) * 1.001 +
                 1e-3 * ref_hp.norm(dim=-1) + 1e-5)
        p_ok = bool((rerr_p <= bnd_p).all())
        rerr_d = (hq[len(P):] - ref_hd).norm(dim=-1)
        bnd_d = sqrtk * elem_err_bound(sc[len(P):]) * 1.001 + 1e-5
        d_ok = bool((rerr_d <= bnd_d).all())
        unt_ok = all(torch.equal(pool[u], flat0.as_strided(
            pool.shape, pool.stride())[u]) for u in untouched)
        od = out[0, NP:].float()
        od_bound = (bnd_d * ref_qd.norm(dim=-1)[..., None] +
                    ref_od.abs() * 2.0**-8 + 1e-5)
        od_ok = bool(((od - ref_od).abs() <= od_bound).all())
        op_err = float((out[0, :NP].float() - ref_op).abs().max())
        per_iter.append(dict(it=it, same=bool(same), prefill_state_ok=p_ok,
                             rider_state_ok=d_ok, untouched_ok=bool(unt_ok),
                             rider_out_ok=od_ok, prefill_out_max_err=op_err))
        if not (same and p_ok and d_ok and unt_ok and od_ok):
            fail("mixed iteration %d same=%s p=%s d=%s unt=%s od=%s" %
                 (it, same, p_ok, d_ok, unt_ok, od_ok))
        if it % 20 == 0:
            log("C iteration", it, per_iter[-1])
    hq = q8mod.q8_gather_dequant(pool, idx, V)

    # Main's bf16 path on the same bf16-rounded initial state.
    fb, pb = bf16_pool(slots)
    full = torch.zeros(slots, HV, V, K, device=dev)
    full[idx.long()] = h0
    pb.copy_(full.to(torch.bfloat16))
    conv.copy_(conv0)
    qkvz.copy_(qkvz0)
    kw = dict(
        mixed_qkv=mixed_qkv, a=a, b=b, batch_size=nseq,
        has_initial_states=has_init, cache_indices=idx,
        query_start_loc=qsl, query_start_loc_long=qsl_long,
        num_prefill_tokens=NP, num_decode_tokens=R,
        state_indices_p=idx[:len(P)], state_indices_d=idx[len(P):],
        num_prefill=len(P), num_decodes=R, state_workspace=None,
        intermediate_state_indices=None, use_replay=False)
    outb = gdn_mixer.Qwen3NextGatedDeltaNet.forward_extend(
        fake, conv, pb, **kw)
    torch.cuda.synchronize()
    hb = pb[idx.long()].float()
    r = dict(
        iterations=len(per_iter), deterministic=det_ok,
        all_prefill_state_within_bound=all(p["prefill_state_ok"]
                                           for p in per_iter),
        all_rider_state_within_bound=all(p["rider_state_ok"]
                                         for p in per_iter),
        all_rider_out_ok=all(p["rider_out_ok"] for p in per_iter),
        all_untouched_ok=all(p["untouched_ok"] for p in per_iter),
        q8_prefill_state_max_err=float((hq[:len(P)] - ref_hp).abs().max()),
        q8_rider_state_max_err=float((hq[len(P):] - ref_hd).abs().max()),
        q8_prefill_out_max_err=per_iter[-1]["prefill_out_max_err"],
        q8_rider_out_max_err=float((out[0, NP:].float() -
                                    ref_od).abs().max()),
        main_bf16_prefill_state_max_err=float(
            (hb[:len(P)] - ref_hp).abs().max()),
        main_bf16_rider_state_max_err=float(
            (hb[len(P):] - ref_hd).abs().max()),
        main_bf16_prefill_out_max_err=float(
            (outb[0, :NP].float() - ref_op).abs().max()),
        main_bf16_rider_out_max_err=float(
            (outb[0, NP:].float() - ref_od).abs().max()),
        ref_prefill_state_absmax=float(ref_hp.abs().max()),
        ref_prefill_out_absmax=float(ref_op.abs().max()))
    log("C mixed", r)
    if not det_ok:
        fail("mixed determinism")
    results["mixed"] = r


def main():
    log("gpu", torch.cuda.get_device_name(), "format", results["format"])
    check_round_trip()
    for n in (64, 104, 128):
        check_decode(n, False)
    check_decode(32, True)
    check_mixed()
    print(json.dumps(dict(passed=ok_all, compared_against=(
        "FP32 sequential delta-rule reference from the dequantized initial "
        "state (k-step for decode, per-token for prefill); format "
        "round-trip bound; iteration 1 bitwise; main bf16 path error "
        "reported alongside"), results=results)), flush=True)


main()
