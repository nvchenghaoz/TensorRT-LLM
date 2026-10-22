# Correctness check for arm F (T == 1 GDN decode with the gated RMSNorm folded
# in), through the production entry Qwen3NextGatedDeltaNet.forward_decode.
# Base path (unchanged code in this tree, identical to main 7f09143a):
#   forward_decode(...) -> o, then rms_norm_gated_token_major(o, z, w, eps).
# Candidate path: forward_decode(..., norm_fusion=(z, w, eps)) -> y.
# Shapes: Qwen3.8-27B GDN (H=16, HV=48, K=V=128), q/k/v/z views of one
# [N, 16384] in_proj_qkvz row block, a/b column slices of [N, 96], bf16 state
# pool with the serving slot stride 817152 and a PAD (-1) slot at N >= 4.
# A. Repeated execution, 10 warmup + 50 iterations, pristine pool restore
#    before each: sha256 of y and of the whole pool every iteration, compared
#    with the base path's (also repeated) at N = 1/4/5/8/16/32. o is compared
#    via the kernel's debug store.
# B. Stateful 20-step evolution without reset (new q/k/v/a/b/z each step):
#    candidate pool/y bitwise vs base every step; both vs an independent FP32
#    reference (state rounded to bf16 each step, as the pool stores it).
# C. CUDA graph: capture the candidate forward_decode, replay with new inputs
#    and indices, compare bitwise with eager base.
# D. Routing: _decode_norm_fusion row gate is 5..32 (c1-c4 keep the old path).
import hashlib
import json
import os
import sys
import types

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

from tensorrt_llm._torch.modules.fla import gdn_decode_t1 as t1mod  # noqa
from tensorrt_llm._torch.modules.mamba import gdn_mixer  # noqa: E402
from tensorrt_llm._torch.modules.mamba.layernorm_gated import \
    rms_norm_gated_token_major  # noqa: E402

EPS = 1e-6
H, HV, K, V = common.H, common.HV, common.K, common.V
WARMUP, ITERS = 10, 50
failures = []
summary = {}


def log(*a):
    print(*a, flush=True)


def fail(msg):
    failures.append(msg)
    log("FAIL:", msg)


def sha(t):
    return hashlib.sha256(
        t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def fake_self(lay):
    return types.SimpleNamespace(key_dim_per_tp=H * K, num_k_heads_per_tp=H,
                                 head_k_dim=K, num_v_heads_per_tp=HV,
                                 head_v_dim=V, A_log=lay["A_log"],
                                 dt_bias=lay["dt_bias"])


def fwd(lay, fused):
    n = lay["n"]
    kw = dict(mixed_qkv=lay["qkvz"][:, :common.CONV_DIM], a=lay["a"],
              b=lay["b"], cache_indices=lay["idx"], num_decodes=n,
              conv_fused=True)
    cu = torch.arange(n + 1, dtype=torch.long, device="cuda")
    if fused:
        kw["norm_fusion"] = (lay["z"], lay["norm_w"], EPS)
    out = gdn_mixer.Qwen3NextGatedDeltaNet.forward_decode(
        fake_self(lay), None, lay["pool"], cu, **kw)
    if fused:
        return out
    return rms_norm_gated_token_major(out.reshape(-1, V), lay["z"],
                                      lay["norm_w"], EPS)


def f_debug_o(lay):
    o = torch.empty(lay["n"] * HV, V, dtype=torch.bfloat16, device="cuda")
    r = torch.empty(lay["n"] * HV, dtype=torch.float32, device="cuda")
    t1mod.fused_sigmoid_gating_delta_rule_update_t1_norm(
        A_log=lay["A_log"], a=lay["a"], dt_bias=lay["dt_bias"],
        softplus_beta=1.0, softplus_threshold=20.0, q=lay["q"], k=lay["k"],
        v=lay["v"], b=lay["b"], z=lay["z"], norm_weight=lay["norm_w"],
        eps=EPS, initial_state_source=lay["pool"],
        initial_state_indices=lay["idx"], scale=K**-0.5,
        use_qk_l2norm_in_kernel=True, o_debug=o, rstd_debug=r)
    return o


def base_o(lay):
    common.make_call(None, None)(lay)
    return lay["out"].reshape(-1, V)


def section_a(n):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(11000 + n)
    lay = common.make_layer(n, gen, pad=(n >= 4))
    res = {}
    for name, fused in (("base", False), ("cand", True)):
        ys, pools = set(), set()
        for it in range(WARMUP + ITERS):
            common.reset([lay])
            y = fwd(lay, fused)
            torch.cuda.synchronize()
            ys.add(sha(y))
            pools.add(sha(lay["flat"]))
        res[name] = (ys, pools)
        if len(ys) != 1 or len(pools) != 1:
            fail(f"A N={n} {name}: not deterministic over {WARMUP + ITERS} "
                 f"calls ({len(ys)} y shas, {len(pools)} pool shas)")
    # o: base kernel output vs F debug o.
    common.reset([lay])
    ob = base_o(lay).clone()
    common.reset([lay])
    of = f_debug_o(lay)
    torch.cuda.synchronize()
    o_eq = sha(ob) == sha(of)
    y_eq = res["base"][0] == res["cand"][0]
    p_eq = res["base"][1] == res["cand"][1]
    if not (o_eq and y_eq and p_eq):
        fail(f"A N={n}: o_eq {o_eq} y_eq {y_eq} pool_eq {p_eq}")
    summary[f"A_N{n}"] = dict(calls=WARMUP + ITERS, pad=(n >= 4), o_sha_eq=o_eq,
                              y_sha_eq=y_eq, pool_sha_eq=p_eq,
                              y_sha=sorted(res["cand"][0])[0][:16],
                              pool_sha=sorted(res["cand"][1])[0][:16])
    log(f"A N={n}: {summary[f'A_N{n}']}")


def reference_step(lay, state):
    n = lay["n"]
    q = lay["q"].reshape(n, H, K).float()
    k = lay["k"].reshape(n, H, K).float()
    v = lay["v"].reshape(n, HV, V).float()
    q = q / (torch.sqrt((q * q).sum(-1, keepdim=True)) + 1e-6) * K**-0.5
    k = k / (torch.sqrt((k * k).sum(-1, keepdim=True)) + 1e-6)
    q = q.repeat_interleave(HV // H, dim=1)
    k = k.repeat_interleave(HV // H, dim=1)
    g = -torch.exp(lay["A_log"]) * F.softplus(
        lay["a"].float() + lay["dt_bias"], beta=1.0, threshold=20.0)
    beta = torch.sigmoid(lay["b"].float())
    h = state.float() * torch.exp(g)[:, :, None, None]
    vn = (v - (h * k[:, :, None, :]).sum(-1)) * beta[:, :, None]
    h = h + vn[:, :, :, None] * k[:, :, None, :]
    o = (h * q[:, :, None, :]).sum(-1)
    x = o.to(torch.bfloat16).float()
    rstd = torch.rsqrt((x * x).mean(-1, keepdim=True) + EPS)
    zz = lay["z"].float()
    y = x * rstd * lay["norm_w"].float() * zz * torch.sigmoid(zz)
    return y.reshape(n * HV, V), h.to(torch.bfloat16)


def section_b(n):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(12000 + n)
    lb = common.make_layer(n, gen, pad=False)
    lc = dict(lb)
    lc["flat"] = lb["flat"].clone()
    lc["pool"] = lc["flat"].as_strided(lb["pool"].shape, lb["pool"].stride())
    idx = lb["idx"].long()
    ref_state = lb["pool"][idx].clone()
    worst = dict(y_err_base=0.0, y_err_cand=0.0, st_err_base=0.0,
                 st_err_cand=0.0)
    bit = 0
    for step in range(20):
        lb["qkvz"].copy_(torch.randn(lb["qkvz"].shape, device="cuda",
                                     generator=gen) * 0.7)
        lb["ba"].copy_(torch.randn(lb["ba"].shape, device="cuda",
                                   generator=gen) * 1.5)
        yb = fwd(lb, False)
        yc = fwd(lc, True)
        ry, ref_state = reference_step(lb, ref_state)
        torch.cuda.synchronize()
        same = (sha(yb) == sha(yc)) and torch.equal(
            lb["flat"].view(torch.int16), lc["flat"].view(torch.int16))
        bit += int(same)
        if not same:
            fail(f"B N={n} step {step}: candidate differs from base")
        worst["y_err_base"] = max(worst["y_err_base"],
                                  (yb.float() - ry).abs().max().item())
        worst["y_err_cand"] = max(worst["y_err_cand"],
                                  (yc.float() - ry).abs().max().item())
        worst["st_err_base"] = max(worst["st_err_base"], (
            lb["pool"][idx].float() - ref_state.float()).abs().max().item())
        worst["st_err_cand"] = max(worst["st_err_cand"], (
            lc["pool"][idx].float() - ref_state.float()).abs().max().item())
    worst["bitwise_steps"] = bit
    summary[f"B_N{n}"] = worst
    log(f"B N={n}: {worst}")


def section_c():
    n = 8
    gen = torch.Generator(device="cuda")
    gen.manual_seed(13000)
    lay = common.make_layer(n, gen, pad=True)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fwd(lay, True)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y_static = fwd(lay, True)
    torch.cuda.synchronize()
    ok = 0
    for trial in range(3):
        lay["qkvz"].copy_(torch.randn(lay["qkvz"].shape, device="cuda",
                                      generator=gen) * 0.7)
        lay["ba"].copy_(torch.randn(lay["ba"].shape, device="cuda",
                                    generator=gen) * 1.5)
        perm = torch.randperm(n + 4, device="cuda", generator=gen)[:n]
        perm[trial] = -1
        lay["idx"].copy_(perm.to(torch.int32))
        common.reset([lay])
        g.replay()
        torch.cuda.synchronize()
        yg, pg = sha(y_static), sha(lay["flat"])
        common.reset([lay])
        ye = fwd(lay, False)
        torch.cuda.synchronize()
        same = yg == sha(ye) and pg == sha(lay["flat"])
        ok += int(same)
        if not same:
            fail(f"C trial {trial}: graph replay differs from eager base")
    summary["C_graph"] = dict(trials=3, bitwise=ok)
    log(f"C: {summary['C_graph']}")


def section_d():
    summary["D_routing"] = dict(
        enabled=gdn_mixer._T1_NORM_FUSION,
        min_rows=gdn_mixer._T1_NORM_FUSION_MIN_ROWS,
        max_rows=gdn_mixer._T1_NORM_FUSION_MAX_ROWS)
    if gdn_mixer._T1_NORM_FUSION_MIN_ROWS < 5:
        fail("D: fusion routed at N <= 4 (c1-c4 must keep the old path)")
    log(f"D: {summary['D_routing']}")


for n in (1, 4, 5, 8, 16, 32):
    section_a(n)
for n in (8, 16, 32):
    section_b(n)
section_c()
section_d()
print(json.dumps({"passed": not failures, "failures": failures[:20],
                  "compared_against": "base path in-tree (T1 kernel + "
                  "rms_norm_gated_token_major, unchanged from main 7f09143a) "
                  "bitwise via sha256; FP32 reference errors reported",
                  "summary": summary}), flush=True)
