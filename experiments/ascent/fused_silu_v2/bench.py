# SwiGLU->NVFP4 permuted, decode rows 5..32: main kernel (A, A') vs V2 (C).
# Usage: python bench.py table|check
# Loads only fused_silu_mul_nvfp4.py (no tensorrt_llm import, no model).
import hashlib
import importlib.util
import json
import os
import sys

import torch

path = os.path.join(os.getcwd(), "tensorrt_llm/_torch/modules/fused_silu_mul_nvfp4.py")
spec = importlib.util.spec_from_file_location("fsm", path)
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)
print("module", path, "sha", hashlib.sha256(open(path, "rb").read()).hexdigest()[:12],
      "has_v2", hasattr(F, "_launch_permuted_v2"), flush=True)
mode = sys.argv[1] if len(sys.argv) > 1 else "table"
dev = "cuda"
I = 17408
N = 2 * I
NSF = I // 16
NKT = (NSF + 3) // 4
L, NIN, WARM, ITERS = 64, 4, 10, 50
gs = torch.tensor([448.0 * 6.0 / 24.0], dtype=torch.float32, device=dev)


def make_x(M, seed):
    gen = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(M, N, device=dev, generator=gen) * 2.0
    # a few special columns: zero blocks, large, tiny (subnormal-ish SF)
    x[:, 0:128] = 0.0
    x[:, 128:256] *= 1e-6
    x[:, 256:384] *= 40.0
    return x.to(torch.bfloat16).contiguous()


def outs(M):
    m_pad = (M + 127) // 128 * 128
    return (torch.empty((M, I // 2), dtype=torch.uint8, device=dev),
            torch.empty((m_pad * NKT * 4,), dtype=torch.uint8, device=dev))


def launcher(arm):
    if arm in ("A", "A2"):
        def f(x, q, s):
            F._V2_ENABLED = False
            F._launch(x, gs, q, s, permuted=True)
        return f
    nb, w = arm
    def f(x, q, s):
        F._launch_permuted_v2(x, gs, q, s, nb, w)
    return f


def capture(fn):
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * (len(v) - 1) + 0.5))]


def table():
    arms = ["A", "A2", (32, 1), (64, 2), (64, 1), (128, 4), (128, 2)]
    res = {}
    for M in [5, 8, 16, 24, 32]:
        xs = [make_x(M, 100 + k) for k in range(NIN)]
        os_ = [outs(M) for _ in range(L)]
        graphs = {}
        for arm in arms:
            fn = launcher(arm)
            graphs[str(arm)] = capture(
                lambda fn=fn: [fn(xs[l % NIN], os_[l][0], os_[l][1]) for l in range(L)])
        F._V2_ENABLED = True
        for _ in range(WARM):
            for g in graphs.values():
                g.replay()
        torch.cuda.synchronize()
        t = {k: [] for k in graphs}
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        for it in range(ITERS):
            ks = list(graphs)
            if it % 2:
                ks = ks[::-1]
            for k in ks:
                ev[0].record()
                graphs[k].replay()
                ev[1].record()
                torch.cuda.synchronize()
                t[k].append(ev[0].elapsed_time(ev[1]) * 1000.0 / L)
        r = {k: {"med_us": round(q(v, 0.5), 3), "p90_us": round(q(v, 0.9), 3),
                 "p10_us": round(q(v, 0.1), 3)} for k, v in t.items()}
        aa = abs(r["A"]["med_us"] - r["A2"]["med_us"]) / r["A"]["med_us"]
        r["AA_floor"] = round(aa, 4)
        res[M] = r
        print("M", M, json.dumps(r), flush=True)
        del graphs
    print(json.dumps({"table": res, "note": "per-call us, CUDA graph of 64 back-to-back calls, 4 rotating L2-resident inputs, arms interleaved"}))


def sha(t):
    return hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest()[:16]


def check():
    ok = True
    rep = {}
    nb, w = F._V2_NB, F._V2_WARPS
    env_v2 = F._V2_ENABLED
    print("env TRTLLM_FUSED_SILU_NVFP4_DECODE_V2 =",
          os.environ.get("TRTLLM_FUSED_SILU_NVFP4_DECODE_V2"), "-> _V2_ENABLED", env_v2, flush=True)
    calls = {"n": 0}
    orig_v2 = F._launch_permuted_v2

    def counted(*a, **k):
        calls["n"] += 1
        return orig_v2(*a, **k)

    F._launch_permuted_v2 = counted
    for M in range(1, 41):
        x = make_x(M, 7 + M)
        # reference: main's kernel (V2 off) and the unpermuted op on unpermuted input
        F._V2_ENABLED = False
        rq, rs = outs(M)
        rq.fill_(0x5A); rs.fill_(0xA5)
        F._launch(x, gs, rq, rs, permuted=True)
        perm = F.gate_up_swiglu_row_permutation(N).to(dev)
        xu = torch.empty_like(x)
        xu[:, perm] = x
        uq, us = F._fused_silu_mul_nvfp4_impl(xu.contiguous(), gs, permuted=False)
        torch.cuda.synchronize()
        ref_ok = torch.equal(rq, uq) and torch.equal(rs, us)
        F._V2_ENABLED = env_v2
        # Candidate through the registered custom op, captured in a CUDA graph
        # (the op allocates its outputs inside the graph pool).
        calls["n"] = 0
        holder = {}

        def run_op():
            holder["o"] = torch.ops.trtllm.fused_silu_mul_nvfp4_permuted(x, gs)

        g = capture(run_op)
        used_v2 = calls["n"] > 0
        cq, cs = holder["o"]
        bad = 0
        for it in range(WARM + ITERS):
            cq.fill_((it * 37) & 0xFF); cs.fill_((it * 91 + 3) & 0xFF)
            g.replay()
            torch.cuda.synchronize()
            if not (torch.equal(cq, rq) and torch.equal(cs, rs)):
                bad += 1
        v2 = used_v2
        expect_v2 = env_v2 and (5 <= M <= 32)
        if v2 != expect_v2:
            ok = False
        rep[M] = {"v2": v2, "expect_v2": expect_v2, "ref_vs_unpermuted": ref_ok, "bad_iters": bad,
                  "fp4_sha": sha(cq), "sf_sha": sha(cs), "ref_fp4_sha": sha(rq), "ref_sf_sha": sha(rs)}
        ok = ok and ref_ok and bad == 0
        print("M", M, json.dumps(rep[M]), flush=True)
    print(json.dumps({"passed": ok, "nb": nb, "warps": w, "v2_enabled": env_v2,
                      "v2_rows": [k for k, v in rep.items() if v["v2"]],
                      "graph_captured_rows": "1-40 (torch.ops.trtllm.fused_silu_mul_nvfp4_permuted)",
                      "compared": "V2 dispatch (rows 5-32) and main path (other rows) FP4 + full SF buffer (padding rows incl.) vs main's fused_silu_mul_nvfp4_permuted kernel (V2 off) and vs unpermuted fused_silu_mul_nvfp4, bitwise, 10+50 graph replays with buffers refilled with garbage before each replay",
                      "per_M": {str(k): v["bad_iters"] for k, v in rep.items()}}))


if mode == "table":
    table()
else:
    check()
