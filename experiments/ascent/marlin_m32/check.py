"""Idea 55 (branch 36) untimed correctness: kM32Table (Marlin rows 17-32) + qkvz cap 32.

Synthetic FP8 per-tensor linears of the real shapes with the tree's load-time copies (same
construction as pair.py), R rotated modules per shape (as pair.py's timed loop).
Per shape and rows M in 1,2,3,4,8,12,16,17,20,24,28,32,33,64:
  (a) dispatch: quant_method.apply output == the kernel the route should run, bitwise, and
      the W4A4 call count. Rows <= GEMV cap: GEMV; <= 16: Marlin (BF16 chain); 17..32: Marlin
      (FP32 reduce, kM32Table) for all three shapes; > 32: W4A4. At 17..32 the table launch
      must differ from TRTLLM_MARLIN_M32_TABLE=0 (generic launch) on the same input (active).
  (b) repeated execution (rows 17,24,32): the bench's CUDA graph (R modules through apply,
      PDL on), 10 warmup + 50 replays = 60, outputs NaN-poisoned before every replay and
      compared bitwise to replay 1 and to the eager call; inputs unchanged.
  (c) error vs the FP32 reference of the original FP8 op (x @ (W_fp8 * s)^T) at 17/24/32:
      table Marlin rel-RMS <= 1.02 x generic Marlin's (branch 18) + 1e-6; for qkvz also
      <= W4A4's (main's route at these rows).
  (d) kill switches: TRTLLM_FP8_NVFP4_DECODE_WIDE=0 -> cap 16, rows 17..32 W4A4 bitwise;
      TRTLLM_FP8_NVFP4_DECODE_WIDE_QKVZ=0 -> qkvz cap 16 (others 32);
      TRTLLM_MARLIN_M32_TABLE=0 -> apply == explicit generic launch.
  (e) Marlin persistent lock workspace all zero at the end.
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
SHAPES = [("qkvz", 16384, 5120, 9, None), ("out_oproj", 5120, 6144, 24, None),
          ("qkv", 14336, 5120, 10, 12288)]


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


gen = torch.Generator(device=dev)
gen.manual_seed(4321)


def rnd(*shape, scale=1.0):
    return (torch.randn(*shape, device=dev, generator=gen) * scale).to(torch.bfloat16)


class Lin(nn.Module):

    def forward(self, x):
        return self.quant_method.apply(self, x, None)


def make_lin(n, k, in_scale, split):
    W = rnd(n, k, scale=0.02)
    ws = (W.float().abs().amax() / 448.0).reshape(())
    m = Lin()
    m.quant_method = L.FP8QDQLinearMethod()
    m.weight = nn.Parameter((W.float() / ws).to(torch.float8_e4m3fn), requires_grad=False)
    m.weight_scale = nn.Parameter(ws.float().clone(), requires_grad=False)
    m.input_scale = nn.Parameter(torch.tensor(in_scale, device=dev), requires_grad=False)
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


def set_table(on):
    os.environ.pop("TRTLLM_MARLIN_M32_TABLE", None)
    if not on:
        os.environ["TRTLLM_MARLIN_M32_TABLE"] = "0"


def marlin(m, x, mode=None):
    if mode is None:  # the mode apply() passes for these rows
        mode = L._marlin_workspace_mode(m, x.shape[0])
    return torch.ops.trtllm.marlin_nvfp4_gemm(
        x, m._nvfp4_decode_weight, scale_a=None, scale_b=m._nvfp4_decode_scale, alpha=None,
        weight_global_scale=m._nvfp4_decode_global_scale, bias=None,
        out_dtype=torch.bfloat16, size_n=m.out_features, size_k=m.in_features, workspace_mode=mode)


def marlin_generic(m, x):
    set_table(False)
    try:
        return marlin(m, x)
    finally:
        set_table(True)


def gemv(m, x):
    return torch.ops.trtllm.nvfp4_w4a16_gemv(x, *m._nvfp4_decode_gemv)


def w4a4(m, x):
    return fpn.prefill_nvfp4_apply(fpn.get_prefill_nvfp4_copy(m), m, x, x.shape, None)


def calls():
    return fpn.prefill_nvfp4_stats()["calls"]


ok = True
fails = []


def fail(msg):
    global ok
    ok = False
    fails.append(msg)
    log("FAIL", msg)


def expected_route(m, M, cap=32):
    gcap = m.__dict__.get("_nvfp4_decode_gemv_max_rows", 0)
    if M <= gcap:
        return "gemv"
    if M <= cap:
        return "marlin"
    return "w4a4"


KFN = {"gemv": gemv, "marlin": marlin, "w4a4": w4a4}
res = {"dispatch": {}, "repeat": {}, "err": {}, "kill": {}, "caps": {}}
set_table(True)
for sname, n, k, R, split in SHAPES:
    xs = {M: rnd(M, k, scale=1.5) for M in MS}
    s_in = 1.1 * max(float(x.float().abs().amax()) for x in xs.values()) / 448
    mods = [make_lin(n, k, s_in, split) for _ in range(R)]
    m0 = mods[0]
    res["caps"][sname] = {"marlin": L.fp8_nvfp4_decode_max_rows(m0),
                          "gemv": m0.__dict__.get("_nvfp4_decode_gemv_max_rows", 0)}
    log(sname, "caps", res["caps"][sname])
    if res["caps"][sname]["marlin"] != 32:
        fail(f"cap {sname} {res['caps'][sname]}")
    for M in MS:
        with autotune():
            w4a4(m0, xs[M])
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
        d = {"route": route, "eq": eq, "w4a4_calls": dc}
        if 17 <= M <= 32:
            d["table_ne_generic"] = not bool(torch.equal(y, marlin_generic(m0, x)))
            if not d["table_ne_generic"]:
                fail(f"table not active {sname} M={M}")
        res["dispatch"][f"{sname}_M{M}"] = d
        if not eq or dc != want_calls:
            fail(f"dispatch {sname} M={M} {d}")
    torch.cuda.synchronize()
    log(sname, "dispatch", {kk: v for kk, v in res["dispatch"].items() if kk.startswith(sname)})
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
                    maxd = max(maxd, float((c.float() - f.float()).abs().max()))
                if not torch.equal(c, e):
                    nbad_eager += 1
        x_same = bool(torch.equal(x, x0))
        gen_eq = sum(int(torch.equal(f, marlin_generic(m, x))) for f, m in zip(first, mods))
        r = {"route": expected_route(m0, M), "capture_w4a4_calls": cap_calls,
             "mismatch_vs_first": nbad_first, "mismatch_vs_eager": nbad_eager,
             "nonfinite_replays": nnan, "max_drift": maxd, "input_unchanged": x_same,
             "replays": WARM + ITERS, "copies": R, "graph_eq_generic_launch": gen_eq}
        res["repeat"][f"{sname}_M{M}"] = r
        if (nbad_first or nbad_eager or nnan or not x_same or cap_calls != 0 or gen_eq != 0):
            fail(f"repeat {sname} M={M} {r}")
        log(sname, f"M={M} repeat", r)
        del g, outs
    # (c) error vs FP32 reference of the original op
    Wd = m0.weight.float() * m0.weight_scale.float()
    for M in REP_MS:
        x = xs[M]
        ref = x.float() @ Wd.t()
        e = {}
        for nm, f in (("marlin_table", marlin), ("marlin_generic_b18", marlin_generic),
                      ("w4a4", w4a4)):
            dd = f(m0, x).float() - ref
            e[nm] = {"rel_rms": float(dd.norm() / ref.norm()), "max_abs": float(dd.abs().max())}
        res["err"][f"{sname}_M{M}"] = e
        if e["marlin_table"]["rel_rms"] > 1.02 * e["marlin_generic_b18"]["rel_rms"] + 1e-6:
            fail(f"err table > generic {sname} M={M} {e}")
        if sname == "qkvz" and e["marlin_table"]["rel_rms"] > e["w4a4"]["rel_rms"]:
            fail(f"err qkvz table > w4a4 M={M} {e}")
    log(sname, "err vs FP32", {kk: v for kk, v in res["err"].items() if kk.startswith(sname)})
    # (d) kill switches
    kres = {}
    os.environ[L._FP8_NVFP4_DECODE_WIDE_ENV] = "0"
    mk = make_lin(n, k, s_in, split)
    os.environ.pop(L._FP8_NVFP4_DECODE_WIDE_ENV)
    kres["wide0_cap"] = L.fp8_nvfp4_decode_max_rows(mk)
    for M in [16, 17, 32]:
        x = xs[M]
        c0 = calls()
        y = mk(x)
        dc = calls() - c0
        route = expected_route(mk, M, cap=16)
        eq = bool(torch.equal(y, KFN[route](mk, x)))
        kres[f"wide0_M{M}"] = {"route": route, "eq": eq, "w4a4_calls": dc}
        if not eq or dc != (1 if route == "w4a4" else 0):
            fail(f"kill wide0 {sname} M={M} {kres}")
    if kres["wide0_cap"] != 16:
        fail(f"kill wide0 cap {sname} {kres['wide0_cap']}")
    del mk
    os.environ["TRTLLM_FP8_NVFP4_DECODE_WIDE_QKVZ"] = "0"
    mk = make_lin(n, k, s_in, split)
    os.environ.pop("TRTLLM_FP8_NVFP4_DECODE_WIDE_QKVZ")
    kres["wide_qkvz0_cap"] = L.fp8_nvfp4_decode_max_rows(mk)
    if kres["wide_qkvz0_cap"] != (16 if sname == "qkvz" else 32):
        fail(f"kill wide_qkvz0 cap {sname} {kres['wide_qkvz0_cap']}")
    if sname == "qkvz":
        c0 = calls()
        y = mk(xs[32])
        kres["wide_qkvz0_M32_w4a4"] = bool(torch.equal(y, w4a4(mk, xs[32]))) and calls() - c0 == 2
        if not kres["wide_qkvz0_M32_w4a4"]:
            fail(f"kill wide_qkvz0 route {sname}")
    del mk
    set_table(False)
    y = m0(xs[32])
    kres["table0_M32_eq_generic"] = bool(torch.equal(y, marlin(m0, xs[32])))
    set_table(True)
    kres["table0_M32_ne_table"] = not bool(torch.equal(y, m0(xs[32])))
    if not (kres["table0_M32_eq_generic"] and kres["table0_M32_ne_table"]):
        fail(f"kill table0 {sname} {kres}")
    res["kill"][sname] = kres
    log(sname, "kill switches", kres)
    del mods, xs, Wd
    torch.cuda.empty_cache()

torch.cuda.synchronize()
ws = None
if hasattr(torch.ops.trtllm, "marlin_nvfp4_persistent_workspace_snapshot"):
    snap = torch.ops.trtllm.marlin_nvfp4_persistent_workspace_snapshot(torch.empty(1, device=dev))
    ws = bool((snap == 0).all().item()) if snap.numel() else None
    if ws is False:
        fail("Marlin persistent workspace not zero")
res["workspace_all_zero"] = ws
print(json.dumps({"passed": ok, "fails": fails,
                  "compared": "apply() output vs the routed kernel bitwise (rows 1-64, 3 shapes); "
                              "60 NaN-poisoned graph replays (R rotated modules, PDL) at rows "
                              "17/24/32 bitwise vs replay 1 and eager; table != generic launch; "
                              "error vs FP32 reference of the FP8 op (table vs branch 18 generic "
                              "Marlin vs W4A4); kill switches; lock workspace all zero",
                  **res}), flush=True)
