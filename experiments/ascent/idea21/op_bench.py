"""Idea 21 op microbenchmark / numerics check: persistent row softmax vs main's Inductor softmax.

usage: python op_bench.py <mode> [T]
  sweep : time candidate launch configs (decoder e4m3 and encoder bf16 kernels) vs main's
          Inductor kernel, cold L2, one process, AB/BA alternation.
  check : untimed numerics vs main's Inductor kernel (same process) under the declared
          tolerance + repeat-call determinism + inputs unchanged + fp64 reference for scale.
  time  : timing of the tree's CFG vs main (pair evidence style JSON).

S is the real producer's output: bmm(q^T, k, out_dtype=fp32) of seeded bf16 q, k [512, T].

DECLARED TOLERANCE (brief / plan 351, stated before any timed run, never widened):
  pq (decoder e4m3): bitwise on >= 99.99% of elements and never more than 1 e4m3 ulp.
  scale_a: max rel error <= 1e-5.
  bf16 P (encoder): max 1 bf16 ulp; report share differing.
"""

import json
import os
import sys
import time

import torch

from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae import row_softmax_triton as rs

MODE = sys.argv[1] if len(sys.argv) > 1 else "sweep"
T = int(sys.argv[2]) if len(sys.argv) > 2 else 65536
C = 512
SCALE = C**-0.5
T0 = time.time()
DEV = torch.device("cuda")


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def make_s(seed=0, amp=1.5):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = (torch.randn(1, C, T, device="cuda", generator=g) * amp).to(torch.bfloat16)
    k = (torch.randn(1, C, T, device="cuda", generator=g) * amp).to(torch.bfloat16)
    s = torch.bmm(q.transpose(1, 2), k, out_dtype=torch.float32)
    del q, k
    return s  # [1, T, T]


def f_dec(s2):
    z = s2 * SCALE
    e = torch.exp(z - z.amax(dim=-1, keepdim=True))
    scale_a = 1.0 / (448.0 * e.sum(dim=-1, keepdim=True))
    pq = (e * 448.0).to(torch.float8_e4m3fn)
    return pq, scale_a


def f_enc(s3):
    return torch.softmax(s3 * SCALE, dim=-1).to(torch.bfloat16)


base_dec = torch.compile(f_dec, dynamic=False,
                         options={"layout_optimization": False, "emulate_precision_casts": True})
base_enc = torch.compile(f_enc, dynamic=False, options={"layout_optimization": False})

FLUSH = torch.empty(512 * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)


def flush():
    FLUSH.fill_(1.0)


def timed(fn):
    flush()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    out = fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b), out


def stats(ts):
    ts = sorted(ts)
    n = len(ts)
    med = ts[n // 2]
    p10, p90 = ts[int(0.1 * n)], ts[max(int(0.9 * n) - 1, 0)]
    return {"median_ms": round(med, 4), "p10_ms": round(p10, 4), "p90_ms": round(p90, 4),
            "noise_floor": round((p90 - p10) / med, 5), "n": n}


def set_cfg(c):
    rs.CFG.update(c)


def cand_dec(s):
    return rs.row_softmax_e4m3(s[0], SCALE)


def cand_enc(s):
    return rs.row_softmax_bf16(s, SCALE)


def _ord8(t):
    i = t.view(torch.int8).to(torch.int32)
    return torch.where(i < 0, -(i & 0x7F), i)


def _ord16(t):
    i = t.contiguous().view(torch.int16).to(torch.int32)
    return torch.where(i < 0, -(i & 0x7FFF), i)


def cmp_chunked(a, b, ordf, rows=4096):
    """a, b same-shape low-precision tensors [..., T, N]; returns bitwise share, max ulp, max abs."""
    a2 = a.reshape(-1, a.shape[-1])
    b2 = b.reshape(-1, b.shape[-1])
    ndiff, mulp, mabs = 0, 0, 0.0
    for i in range(0, a2.shape[0], rows):
        x, y = a2[i:i + rows], b2[i:i + rows]
        u = (ordf(x) - ordf(y)).abs()
        ndiff += int((u > 0).sum().item())
        mulp = max(mulp, int(u.max().item()))
        mabs = max(mabs, (x.float() - y.float()).abs().max().item())
    tot = a2.numel()
    return {"bitwise": ndiff == 0, "share_differ": ndiff / tot, "max_ulp": mulp, "max_abs": mabs}


def ref_sum64(s, rows):
    """fp64 row sums of exp((s - max s) * scale) for the sampled rows."""
    x = s.reshape(-1, s.shape[-1])[rows].double()
    z = (x - x.amax(-1, keepdim=True)) * SCALE
    return torch.exp(z).sum(-1)


def run_check():
    s = make_s(0)
    s_ref = s.clone()
    res = {"T": T, "cfg": dict(rs.CFG)}
    gates = {}
    # ---- decoder e4m3
    pq_b, sa_b = base_dec(s[0])
    pq_b, sa_b = pq_b.clone(), sa_b.clone()
    pq_c, sa_c = cand_dec(s)
    c = cmp_chunked(pq_c, pq_b, _ord8)
    rel = ((sa_c - sa_b).abs() / sa_b.abs()).max().item()
    res["dec_pq_vs_main"] = c
    res["dec_scale_a_vs_main"] = {"max_rel": rel, "bitwise_share": (sa_c == sa_b).float().mean().item(),
                                  "max_abs": (sa_c - sa_b).abs().max().item()}
    rows = torch.arange(0, T, max(1, T // 2048), device="cuda")
    l64 = ref_sum64(s, rows)
    sa64 = 1.0 / (448.0 * l64)
    res["dec_scale_a_vs_fp64"] = {
        "cand_max_rel": ((sa_c[rows, 0].double() - sa64).abs() / sa64).max().item(),
        "main_max_rel": ((sa_b[rows, 0].double() - sa64).abs() / sa64).max().item()}
    gates["dec_pq"] = c["max_ulp"] <= 1 and (1.0 - c["share_differ"]) >= 0.9999
    gates["dec_scale_a"] = rel <= 1e-5
    # repeat sequence (same count as timing: 3 warm + ITERS) vs iteration 0
    iters = int(os.environ.get("ITERS", "20"))
    rep = True
    inp = True
    first = (pq_c.clone(), sa_c.clone())
    del pq_c, sa_c
    for i in range(3 + iters):
        a, b_ = cand_dec(s)
        rep &= bool(torch.equal(a.view(torch.uint8), first[0].view(torch.uint8)) and torch.equal(b_, first[1]))
        del a, b_
        if i % 5 == 0:
            inp &= bool(torch.equal(s, s_ref))
    inp &= bool(torch.equal(s, s_ref))
    gates["dec_repeat_bitwise"] = rep
    gates["dec_inputs_unchanged"] = inp
    del first, pq_b, sa_b
    torch.cuda.empty_cache()
    log("decoder", json.dumps(res), json.dumps(gates))
    # ---- encoder bf16
    p_b = base_enc(s).clone()
    p_c = cand_enc(s)
    c = cmp_chunked(p_c, p_b, _ord16)
    res["enc_p_vs_main"] = c
    gates["enc_p"] = c["max_ulp"] <= 1
    # P vs fp64 softmax on sampled rows: both arms
    x = s.reshape(-1, T)[rows].double()
    p64 = torch.softmax(x * SCALE, -1)
    res["enc_p_vs_fp64"] = {
        "cand_max_abs": (p_c.reshape(-1, T)[rows].double() - p64).abs().max().item(),
        "main_max_abs": (p_b.reshape(-1, T)[rows].double() - p64).abs().max().item()}
    del x, p64
    first = p_c.clone()
    del p_c, p_b
    torch.cuda.empty_cache()
    rep = True
    for i in range(3 + iters):
        a = cand_enc(s)
        rep &= bool(torch.equal(a, first))
        del a
    inp = bool(torch.equal(s, s_ref))
    gates["enc_repeat_bitwise"] = rep
    gates["enc_inputs_unchanged"] = inp
    # different input after, then back (no stale state)
    s2 = make_s(1)
    a2 = cand_enc(s2)
    del a2, s2
    a = cand_enc(s)
    gates["enc_after_other_input_bitwise"] = bool(torch.equal(a, first))
    del a
    res["gates"] = gates
    res["passed"] = bool(all(gates.values()))
    log("check", json.dumps(res))
    print(json.dumps({"passed": res["passed"], "output": {"class": "reordered"}, **res}))


def bench_pair(kind, s, cfgs, iters):
    """Alternate main + each candidate cfg (balanced: seeded random order per round)."""
    import random
    rnd = random.Random(0)
    arms = {"main": None}
    for i, c in enumerate(cfgs):
        arms[f"c{i}"] = c
    fns = {}

    def fn_for(name):
        if name == "main":
            return (lambda: base_dec(s[0])) if kind == "dec" else (lambda: base_enc(s))
        c = arms[name]

        def f():
            set_cfg(c)
            return cand_dec(s) if kind == "dec" else cand_enc(s)
        return f

    for n in arms:
        fns[n] = fn_for(n)
        for _ in range(2):
            o = fns[n]()
            del o
    torch.cuda.synchronize()
    ts = {n: [] for n in arms}
    names = list(arms)
    for it in range(iters):
        order = names[:]
        rnd.shuffle(order)
        for n in order:
            t, o = timed(fns[n])
            del o
            ts[n].append(t)
    return {n: stats(v) for n, v in ts.items()}


def run_sweep():
    s = make_s(0)
    iters = int(os.environ.get("ITERS", "8"))
    base_cfg = dict(rs.CFG)
    cfgs = []
    for blk, nw in ((2048, 4), (4096, 8), (8192, 8), (8192, 16), (4096, 4)):
        for cps in (1, 2):
            cfgs.append({**base_cfg, "BLOCK": blk, "num_warps": nw, "ctas_per_sm": cps})
    cfgs.append({**base_cfg, "store_cs": False})
    cfgs.append({**base_cfg, "reverse": False})
    cfgs.append({**base_cfg, "num_stages": 2})
    cfgs.append({**base_cfg, "ctas_per_sm": 3})
    for kind in ("dec", "enc"):
        r = bench_pair(kind, s, cfgs, iters)
        for n, st in r.items():
            log(kind, n, json.dumps(cfgs[int(n[1:])] if n != "main" else "main"), json.dumps(st))
        torch.cuda.empty_cache()
    print(json.dumps({"passed": True}))


def run_time():
    s = make_s(0)
    iters = int(os.environ.get("ITERS", "50"))
    out = {}
    for kind in ("dec", "enc"):
        r = bench_pair(kind, s, [dict(rs.CFG)], iters)
        out[kind] = r
        log(kind, json.dumps(r))
    print(json.dumps({"passed": True, "op": out}))


with torch.inference_mode():
    log("T", T, "cfg", rs.CFG)
    if MODE == "check":
        run_check()
    elif MODE == "sweep":
        run_sweep()
    elif MODE == "time":
        run_time()
