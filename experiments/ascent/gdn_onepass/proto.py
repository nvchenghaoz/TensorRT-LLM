"""Prototype diagnostics: one-pass GDN prefill front end vs main's chain (in-process).

Not evidence (reference computed in this tree); used to iterate on the reduce
order and to get a first timing. python -u proto.py
"""
import json
import time

import torch

from tensorrt_llm._torch.modules.mamba.causal_conv1d import causal_conv1d_fn
from tensorrt_llm._torch.modules.mamba.fuse_elementwise_ops import (
    extract_transpose_prefill_slice,
    fused_gdn_post_conv,
)
from tensorrt_llm._torch.modules.mamba.gdn_prefill_conv import (
    gdn_fused_transpose_conv_enabled,
    gdn_prefill_transpose_conv,
)
from tensorrt_llm._torch.modules.mamba.gdn_prefill_onepass import gdn_prefill_onepass

T0 = time.time()
dev = torch.device("cuda")
DIM, QKVZ, NB = 10240, 16384, 12
BLK = 817152  # production pool block stride (elements of bf16)


def log(*a):
    print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


def make(lens, ndec, seed=0, has_init=None):
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    T = sum(lens)
    qkvz = (torch.randn(T, QKVZ, device=dev, generator=g) * 1.5).to(torch.bfloat16)
    dec = (torch.randn(ndec, DIM, device=dev, generator=g)).to(torch.bfloat16) if ndec else None
    w = (torch.randn(DIM, 4, device=dev, generator=g) * 0.5).to(torch.bfloat16)
    pool = torch.zeros(NB * BLK, dtype=torch.bfloat16, device=dev)
    conv = pool.as_strided((NB, DIM, 3), (BLK, 3, 1))
    conv.copy_(torch.randn(NB, DIM, 3, device=dev, generator=g).to(torch.bfloat16))
    ba = torch.randn(T + ndec, 96, device=dev, generator=g).to(torch.bfloat16)
    A_log = torch.log(torch.rand(48, device=dev, generator=g) * 16)
    dt_bias = torch.rand(48, device=dev, generator=g) * 2 - 1
    n = len(lens)
    qsl = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int32,
                       device=dev)
    idx = torch.randperm(NB, device=dev, generator=g)[:n].to(torch.int32)
    hi = torch.ones(n, dtype=torch.bool, device=dev) if has_init is None else torch.tensor(
        has_init, dtype=torch.bool, device=dev)
    return dict(x=qkvz[:, :DIM], qkvz=qkvz, dec=dec, w=w, pool=pool, conv=conv,
                pool0=pool.clone(), a=ba[:, 48:], b=ba[:, :48], A_log=A_log, dt_bias=dt_bias,
                qsl=qsl, idx=idx, hi=hi, T=T)


def run_main(c):
    x = c["x"]
    T = x.shape[0]
    if gdn_fused_transpose_conv_enabled(T):
        xt = gdn_prefill_transpose_conv(x, c["w"], None, c["conv"], c["qsl"], c["idx"], c["hi"],
                                        activation="silu")
    else:
        xt = extract_transpose_prefill_slice(x, T, 0, x.shape[1])
        xt = causal_conv1d_fn(xt, c["w"], None, activation="silu", conv_states=c["conv"],
                              has_initial_state=c["hi"], cache_indices=c["idx"],
                              query_start_loc=c["qsl"])
    return fused_gdn_post_conv(xt, c["dec"], c["a"], c["b"], c["A_log"], c["dt_bias"], 16, 128,
                               48, 128)


def run_one(c, **kw):
    return gdn_prefill_onepass(c["x"], c["dec"], c["w"], None, c["conv"], c["qsl"], c["idx"],
                               c["hi"], c["a"], c["b"], c["A_log"], c["dt_bias"], 16, 128, 48,
                               128, **kw)


def ulp_diff(a, b):
    return (a.float() - b.float()).abs().max().item()


def compare(c, variants):
    c["pool"].copy_(c["pool0"])
    ref = run_main(c)
    ref_conv = c["pool"].clone()
    res = {}
    for name, kw in variants.items():
        c["pool"].copy_(c["pool0"])
        out = run_one(c, **kw)
        torch.cuda.synchronize()
        r = {}
        for nm, o, rr in zip("qkvgb", out, ref):
            r[nm] = (int((o.view(-1).view(torch.int16 if o.dtype != torch.float32 else torch.int32)
                          != rr.view(-1).view(torch.int16 if rr.dtype != torch.float32
                                              else torch.int32)).sum().item()),
                     ulp_diff(o, rr))
        r["conv"] = bool(torch.equal(c["pool"].view(torch.int16), ref_conv.view(torch.int16)))
        res[name] = r
    return res


def timeit(fn, c, cold, iters=50, warm=10):
    flush = timeit.flush
    ts = []
    for i in range(warm + iters):
        c["pool"].copy_(c["pool0"])
        if cold:
            flush.add_(1)
        else:
            # producer stand-in: re-write the qkv columns so they are L2-resident
            c["qkvz"][:, :DIM].copy_(c["qkvz"][:, :DIM])
        torch.cuda._sleep(40000)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn(c)
        e.record()
        if i >= warm:
            ts.append((s, e))
    torch.cuda.synchronize()
    v = sorted(s.elapsed_time(e) * 1e3 for s, e in ts)
    return dict(med=v[len(v) // 2], p10=v[len(v) // 10], p90=v[(9 * len(v)) // 10])


timeit.flush = torch.empty(64 * 1024 * 1024, dtype=torch.float32, device=dev)

variants = {}
for red in (1, 2, 3):
    for ws in (False, True):
        for tr in (False, True):
            variants["r%d_ws%d_tr%d" % (red, ws, tr)] = dict(reduce=red, warp_seq=ws, tree=tr)
variants["auto"] = {}

out = {}
cases = [("1x2048", [2048], 0), ("1x85", [85], 0), ("85+d3", [85], 3), ("2048+d1", [2048], 1),
         ("1x1", [1], 0), ("1x3", [3], 0), ("2x2048", [2048, 2048], 0), ("1x8192", [8192], 0),
         ("ragged", [1000, 37, 2], 2)]
for name, lens, nd in cases:
    c = make(lens, nd)
    r = compare(c, variants)
    out[name] = r
    good = [k for k, v in r.items() if all(v[x][0] == 0 for x in "qkvgb") and v["conv"]]
    log(name, "bitwise variants:", good)
    log("   auto:", r["auto"], " r1:", r["r1_ws0_tr0"], " r3:", r["r3_ws0_tr0"])
    del c
    torch.cuda.empty_cache()

for name, lens in [("1x2048", [2048]), ("2x2048", [2048, 2048]), ("1x8192", [8192])]:
    c = make(lens, 0)
    for _ in range(3):
        run_main(c)
        run_one(c)
    for cold in (False, True):
        tm = timeit(run_main, c, cold)
        to = timeit(run_one, c, cold)
        tm2 = timeit(run_main, c, cold)
        log("%s %s main %s onepass %s main2 %s" % (name, "cold" if cold else "warm", tm, to, tm2))
    del c
    torch.cuda.empty_cache()
print(json.dumps(dict(done=True)))
