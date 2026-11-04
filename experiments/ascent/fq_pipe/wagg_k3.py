"""Idea 163 stage 2: K3-only W-AGG on main's chain for the wo_b fused-quant arms.

For each M: main's K1 (E4M3 + SF) -> main's TMA-A wo_a (K2) run eagerly once
per layer (40 layers, weights rotated, no aggressor) give the K3 inputs tmp[i];
K3 = wo_b_dense_gemm_fused_quant_mxfp8 with each arm then forms a 40-layer
K3-only CUDA graph. Every replay is overlapped by 4 side streams doing mul_ on
256 MB each (W-AGG); after each replay every layer's K3 output must equal the
eager pinned ("off") K3 output bitwise and tmp[i] must be unchanged. Outputs are
NaN-poisoned before each replay. V0 control: main's full chain K1 -> TMA-A K2
-> K3 off as a graph under the same aggressor on the same GPU, vs its eager
output. Also eager K3 bitwise vs off on an adversarial group-major input.
Usage: wagg_k3.py <ckpt> [MS] [ARMS]
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "b12x_wo"))
import common as C  # noqa: E402
import torch  # noqa: E402

from b12x.gemm._shared import wo_mxfp8 as W  # noqa: E402
from tensorrt_llm._torch.attention.backends.sparse.csa2 import b12x_wo as B  # noqa: E402

ckpt = sys.argv[1]
MS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1,2,3,4,8").split(",")]
ALL = {"off": (False, False, 0), "w": (False, True, 0), "w_p2": (False, True, 2)}
ARMS = {k: ALL[k] for k in (sys.argv[3] if len(sys.argv) > 3 else "off,w").split(",")}
LAYERS, AGG_ITERS = 40, int(os.environ.get("AGG_ITERS", "8"))
N1, N = int(os.environ.get("N1", "90")), int(os.environ.get("N", "30"))
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


dims = C.dims_from_ckpt(ckpt)
G, R, GW, H = dims["groups_local"], dims["rank"], dims["group_width"], dims["hidden"]
w0 = C.make_weights(dims)
cs = C.make_cos_sin(dims)
ws = [w0] + [C.clone_weights(w0, dims) for _ in range(LAYERS - 1)]
alpha = W._cached_alpha_one(C.DEV)
agg_streams = [torch.cuda.Stream() for _ in range(4)]
agg_bufs = [torch.ones(64 * 1024 * 1024, device=C.DEV, dtype=torch.float32) for _ in range(4)]


def bitdiff(a, b):
    return int((a.view(torch.int16) != b.view(torch.int16)).sum())


def k3(tmp, w, out, M, arm):
    W.wo_b_dense_gemm_fused_quant_mxfp8(tmp, w.wb_b12x, out=out, expected_m=M,
                                        sfb_k_replicated=False, fq_pipeline=arm)


def capture(fns):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fns[:2]:
            f()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in fns:
            f()
    torch.cuda.synchronize()
    return g


def wagg(g, n, poison, count):
    bad, worst = 0, 0
    main = torch.cuda.current_stream()
    for _ in range(n):
        poison()
        ev = torch.cuda.Event()
        ev.record(main)
        for s, b in zip(agg_streams, agg_bufs):
            s.wait_event(ev)
            with torch.cuda.stream(s):
                for _ in range(AGG_ITERS):
                    b.mul_(1.0000001)
        g.replay()
        torch.cuda.synchronize()
        nb = count()
        if nb:
            bad += 1
            worst = max(worst, nb)
    for b in agg_bufs:
        b.fill_(1.0)
    return {"replays": n, "bad_replays": bad, "max_bad_elems": worst}


def adversarial_tmp(M, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(M, G * R, device=C.DEV, generator=g)
    xb = x.view(M, -1, 32)
    nb = xb.shape[1]
    for r in range(M):
        for j in range(10):
            b = (r * 13 + j * 37) % nb
            p = (r + j) % 32
            blk = torch.zeros(32, device=C.DEV)
            if j == 1:
                blk[p] = 3.3895313892515355e38
            elif j == 2:
                blk[p] = -1.1754943508222875e-38
            elif j == 3:
                blk = torch.rand(32, device=C.DEV, generator=g) * 200.0 * 2.0 ** -5
                blk[p] = 240.0 * 2.0 ** -5
            elif j == 4:
                blk = (torch.rand(32, device=C.DEV, generator=g) - 0.5) * 2.0 ** -10
                blk[p] = 448.0 * 2.0 ** -3
            elif j == 5:
                blk[p] = -1.0
            elif j == 6:
                blk = -torch.rand(32, device=C.DEV, generator=g) * 2.0 ** 90
                blk[p] = -1.875 * 2.0 ** 97
            elif j > 6:
                blk = torch.randn(32, device=C.DEV, generator=g) * 10.0 ** (j - 8)
            xb[r, b] = blk
    t = W.empty_dense_gemm_mnl_view(M, R, G, device=C.DEV, dtype=torch.bfloat16)
    t.permute(0, 2, 1).copy_(x.to(torch.bfloat16).view(M, G, R))
    return t


ok = True
res = {}
for M in MS:
    row = {}
    o3, pos = C.make_inputs(dims, M, seed=100 + M)
    xq = W.empty_mxfp8_rows_for_dense_gemm(M, GW, num_groups=G, device=C.DEV)
    tmps = [W.empty_dense_gemm_mnl_view(M, R, G, device=C.DEV, dtype=torch.bfloat16) for _ in range(LAYERS)]
    outs = [torch.empty((M, H, 1), device=C.DEV, dtype=torch.bfloat16) for _ in range(LAYERS)]

    def k1():
        B.inv_rope_quant128(o3, pos, cs, xq, groups=G, group_width=GW, nope_dim=dims["nope"],
                            rope_dim=dims["rope"])

    def k2(i):
        W.wo_a_dense_gemm_mxfp8(xq, ws[i].wa_b12x, out=tmps[i], alpha=alpha, expected_m=M,
                                sfb_k_replicated=True)

    # adversarial eager bitwise vs off
    ta = adversarial_tmp(M, 50 + M)
    ref_a = torch.empty((M, H, 1), device=C.DEV, dtype=torch.bfloat16)
    k3(ta, w0, ref_a, M, ALL["off"])
    for an, arm in ARMS.items():
        o = torch.empty_like(ref_a)
        k3(ta, w0, o, M, arm)
        torch.cuda.synchronize()
        row[f"adv_{an}_bad_vs_off"] = bitdiff(o, ref_a)
        ok &= row[f"adv_{an}_bad_vs_off"] == 0
    # main's K1 -> K2 eager, K3 off eager reference
    k1()
    for i in range(LAYERS):
        k2(i)
        k3(tmps[i], ws[i], outs[i], M, ALL["off"])
    ref_tmp = [t.clone() for t in tmps]
    ref_out = [o.clone() for o in outs]
    torch.cuda.synchronize()
    n = N1 if M == 1 else N

    def poison():
        for o in outs:
            o.fill_(float("nan"))

    def count():
        return (sum(bitdiff(outs[i], ref_out[i]) for i in range(LAYERS))
                + sum(bitdiff(tmps[i], ref_tmp[i]) for i in range(LAYERS)))

    for an, arm in ARMS.items():
        g = capture([lambda i=i, arm=arm: k3(tmps[i], ws[i], outs[i], M, arm) for i in range(LAYERS)])
        row[f"wagg_K3only_{an}"] = wagg(g, n, poison, count)
        ok &= row[f"wagg_K3only_{an}"]["bad_replays"] == 0
        log(f"M={M} K3-only {an}", row[f"wagg_K3only_{an}"])
        del g
    # V0 control: main's full chain
    g = capture([lambda i=i: (k1(), k2(i), k3(tmps[i], ws[i], outs[i], M, ALL["off"])) for i in range(LAYERS)])

    def poison_v0():
        for o in outs:
            o.fill_(float("nan"))
        for t in tmps:
            t.fill_(float("nan"))

    row["wagg_V0_main_chain_control"] = wagg(g, n, poison_v0, count)
    del g
    log(f"M={M} V0 control", row["wagg_V0_main_chain_control"])
    res[M] = row
    log(f"M={M}", json.dumps(row))

control_fails = all(r["wagg_V0_main_chain_control"]["bad_replays"] > 0 for r in res.values())
print(json.dumps({"passed": bool(ok and control_fails), "control_fails_every_M": control_fails,
                  "compared_against": "eager pinned K3 (fq_pipeline off) on main's K1->TMA-A K2 tmp, bitwise, "
                                      "every W-AGG replay; adversarial per-32 input bitwise vs off",
                  "results": res}, default=str))
