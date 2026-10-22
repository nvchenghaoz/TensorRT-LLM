"""Idea 10 (branch 18) untimed correctness, synthetic FP8 linears of the real
shapes with main's load-time copies (same construction as bench.py).

Per shape and rows M in 1,2,3,4,8,12,16,17,20,24,28,32,33,64:
  (a) dispatch: output of quant_method.apply == the kernel the route should
      run, bitwise, and the W4A4 call count. Rows <= 16 must take main's
      kernels (GEMV <= per-shape GEMV cap, else Marlin; 0 W4A4 calls). Rows
      17..32: Marlin for (14336,5120)/(5120,6144), W4A4 for (16384,5120).
      Rows > 32: W4A4.
  (b) repeated execution (rows 17,24,32): the bench's CUDA graph (R rotated
      modules through apply), 10 warmup + 50 replays, every replay's outputs
      NaN-poisoned before and compared bitwise to the first replay and to the
      eager call; inputs unchanged; Marlin persistent workspace all zero.
  (c) error vs the FP32 reference of the original op (x @ (W_fp8 * s)^T) for
      the new route (Marlin W4A16) and main's route (W4A4) at rows 17..32.
  (d) kill switch TRTLLM_FP8_NVFP4_DECODE_WIDE=0 rebuild: rows 17..32 take
      W4A4 again, bitwise == main's route.
Last stdout line: {"passed": ...}.
"""
import json
import os
import time

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
import torch  # noqa: E402
from torch import nn  # noqa: E402

from tensorrt_llm._torch.autotuner import autotune  # noqa: E402
from tensorrt_llm._torch.modules import fp8_prefill_nvfp4 as fpn  # noqa: E402
from tensorrt_llm._torch.modules import linear as L  # noqa: E402

T0 = time.time()
dev = "cuda"
WARM, ITERS = 10, 50
MS = [1, 2, 3, 4, 8, 12, 16, 17, 20, 24, 28, 32, 33, 64]
REP_MS = [17, 24, 32]
SHAPES = [("qkvz", 16384, 5120, 4, None), ("gdn_out", 5120, 6144, 8, None),
          ("qkv", 14336, 5120, 4, 12288)]
WIDE = {(14336, 5120), (5120, 6144)}


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


gen = torch.Generator(device=dev)
gen.manual_seed(4321)


def rnd(*shape, scale=1.0):
    return (torch.randn(*shape, device=dev, generator=gen) * scale).to(
        torch.bfloat16)


class Lin(nn.Module):

    def forward(self, x):
        return self.quant_method.apply(self, x, None)


def make_lin(n, k, in_scale, split):
    W = rnd(n, k, scale=0.02)
    ws = (W.float().abs().amax() / 448.0).reshape(())
    m = Lin()
    m.quant_method = L.FP8QDQLinearMethod()
    m.weight = nn.Parameter((W.float() / ws).to(torch.float8_e4m3fn),
                            requires_grad=False)
    m.weight_scale = nn.Parameter(ws.float().clone(), requires_grad=False)
    m.input_scale = nn.Parameter(torch.tensor(in_scale, device=dev),
                                 requires_grad=False)
    m.force_dynamic_quantization = False
    m.enable_cuda_core = True
    m.all_reduce = None
    m.mapping = None
    m.dtype = torch.bfloat16
    m.out_features, m.in_features = n, k
    m.bias = None
    m._nvfp4_decode_copy = True
    L._build_fp8_nvfp4_decode_copy(m)
    env = fpn.ENV_ATTN_QG if split else fpn.ENV_GDN_IN_PROJ_QKVZ
    assert fpn.maybe_build_prefill_nvfp4_copy(m, env)
    if split:
        fpn.get_prefill_nvfp4_copy(m).split = split
    del W
    return m


def marlin(m, x):
    return torch.ops.trtllm.marlin_nvfp4_gemm(
        x, m._nvfp4_decode_weight, scale_a=None,
        scale_b=m._nvfp4_decode_scale, alpha=None,
        weight_global_scale=m._nvfp4_decode_global_scale, bias=None,
        out_dtype=torch.bfloat16, size_n=m.out_features,
        size_k=m.in_features)


def gemv(m, x):
    return torch.ops.trtllm.nvfp4_w4a16_gemv(x, *m._nvfp4_decode_gemv)


def w4a4(m, x):
    return fpn.prefill_nvfp4_apply(fpn.get_prefill_nvfp4_copy(m), m, x,
                                   x.shape, None)


def calls():
    return fpn.prefill_nvfp4_stats()["calls"]


ok = True
fails = []


def fail(msg):
    global ok
    ok = False
    fails.append(msg)
    log("FAIL", msg)


def expected_route(m, M, wide=True):
    n, k = m.out_features, m.in_features
    gcap = m.__dict__.get("_nvfp4_decode_gemv_max_rows", 0)
    if M <= gcap:
        return "gemv"
    if M <= 16:
        return "marlin"
    if M <= 32 and wide and (n, k) in WIDE:
        return "marlin"
    return "w4a4"


KFN = {"gemv": gemv, "marlin": marlin, "w4a4": w4a4}
res = {"dispatch": {}, "repeat": {}, "err": {}, "kill": {}, "caps": {}}
for sname, n, k, R, split in SHAPES:
    xs = {M: rnd(M, k, scale=1.5) for M in MS}
    s_in = 1.1 * max(float(x.float().abs().amax()) for x in xs.values()) / 448
    mods = [make_lin(n, k, s_in, split) for _ in range(R)]
    m0 = mods[0]
    res["caps"][sname] = {"marlin": L.fp8_nvfp4_decode_max_rows(m0),
                          "gemv": m0.__dict__.get(
                              "_nvfp4_decode_gemv_max_rows", 0)}
    log(sname, "caps", res["caps"][sname])
    for M in MS:
        x = xs[M]
        with autotune():
            w4a4(m0, x)
        torch.cuda.synchronize()
    # (a) dispatch
    for M in MS:
        x = xs[M]
        route = expected_route(m0, M)
        c0 = calls()
        y = m0(x)
        dc = calls() - c0
        y_ref = KFN[route](m0, x)
        eq = bool(torch.equal(y, y_ref))
        want_calls = 1 if route == "w4a4" else 0
        res["dispatch"][f"{sname}_M{M}"] = {"route": route, "eq": eq,
                                            "w4a4_calls": dc}
        if not eq or dc != want_calls:
            fail(f"dispatch {sname} M={M} route={route} eq={eq} calls={dc}")
    torch.cuda.synchronize()
    log(sname, "dispatch", {kk: v for kk, v in res["dispatch"].items()
                            if kk.startswith(sname)})
    # (b) repeated graph replays
    for M in REP_MS:
        x = xs[M]
        x0 = x.clone()
        eager = [m(x).clone() for m in mods]
        s0 = torch.cuda.Stream()
        s0.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s0):
            for _ in range(2):
                for m in mods:
                    m(x)
        torch.cuda.current_stream().wait_stream(s0)
        torch.cuda.synchronize()
        c0 = calls()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            outs = [m(x) for m in mods]
        cap_calls = calls() - c0
        first = None
        nbad_first, nbad_eager, nnan, maxd = 0, 0, 0, 0.0
        for it in range(WARM + ITERS):
            for o in outs:
                o.fill_(float("nan"))
            g.replay()
            torch.cuda.synchronize()
            cur = [o.clone() for o in outs]
            if any(not bool(torch.isfinite(c).all()) for c in cur):
                nnan += 1
            if first is None:
                first = cur
            for c, f, e in zip(cur, first, eager):
                if not torch.equal(c, f):
                    nbad_first += 1
                    maxd = max(maxd, float((c.float() - f.float()).abs()
                                           .max()))
                if not torch.equal(c, e):
                    nbad_eager += 1
        x_same = bool(torch.equal(x, x0))
        route = expected_route(m0, M)
        r = {"route": route, "capture_w4a4_calls": cap_calls,
             "mismatch_vs_first": nbad_first, "mismatch_vs_eager": nbad_eager,
             "nonfinite": nnan, "max_drift": maxd, "input_unchanged": x_same,
             "replays": WARM + ITERS}
        res["repeat"][f"{sname}_M{M}"] = r
        want_cap = R if route == "w4a4" else 0
        if (nbad_first or nbad_eager or nnan or not x_same
                or cap_calls != want_cap):
            fail(f"repeat {sname} M={M} {r}")
        log(sname, f"M={M} repeat", r)
        del g, outs
    # (c) error vs FP32 reference of the original op
    Wd = m0.weight.float() * m0.weight_scale.float()
    for M in [17, 24, 32]:
        x = xs[M]
        ref = x.float() @ Wd.t()
        e = {}
        for nm, f in (("marlin_w4a16", marlin), ("w4a4_main", w4a4)):
            d = f(m0, x).float() - ref
            e[nm] = {"rel_rms": float(d.norm() / ref.norm()),
                     "max_abs": float(d.abs().max())}
        res["err"][f"{sname}_M{M}"] = e
    log(sname, "err vs FP32", {kk: v for kk, v in res["err"].items()
                               if kk.startswith(sname)})
    # (d) kill switch
    os.environ[L._FP8_NVFP4_DECODE_WIDE_ENV] = "0"
    mk = make_lin(n, k, s_in, split)
    os.environ.pop(L._FP8_NVFP4_DECODE_WIDE_ENV)
    kres = {"cap": L.fp8_nvfp4_decode_max_rows(mk)}
    for M in [16, 17, 32]:
        x = xs[M]
        c0 = calls()
        y = mk(x)
        dc = calls() - c0
        route = expected_route(mk, M, wide=False)
        eq = bool(torch.equal(y, KFN[route](mk, x)))
        kres[f"M{M}"] = {"route": route, "eq": eq, "w4a4_calls": dc}
        if not eq or dc != (1 if route == "w4a4" else 0):
            fail(f"kill {sname} M={M} {kres}")
    if kres["cap"] != 16:
        fail(f"kill cap {sname} {kres['cap']}")
    res["kill"][sname] = kres
    log(sname, "kill switch", kres)
    del mods, mk, xs, Wd
    torch.cuda.empty_cache()

torch.cuda.synchronize()
ws = None
if hasattr(torch.ops.trtllm, "marlin_nvfp4_persistent_workspace_snapshot"):
    snap = torch.ops.trtllm.marlin_nvfp4_persistent_workspace_snapshot(
        torch.empty(1, device=dev))
    ws = bool((snap == 0).all().item()) if snap.numel() else None
    if ws is False:
        fail("Marlin persistent workspace not zero")
res["workspace_all_zero"] = ws
print(json.dumps({"passed": ok, "fails": fails,
                  "compared": "apply() output vs the routed kernel bitwise "
                              "(rows 1-64, 3 shapes); 60 graph replays at "
                              "rows 17/24/32 bitwise vs first replay and "
                              "eager; error vs FP32 reference of the FP8 op; "
                              "kill switch",
                  **res}), flush=True)
