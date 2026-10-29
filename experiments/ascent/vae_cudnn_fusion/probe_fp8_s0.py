"""Idea 172 STEP 0: FP8 (E4M3) 3x3 conv fprop at the 6 FLUX.3 VAE GN->conv shapes.

usage: python experiments/ascent/vae_cudnn_fusion/probe_fp8_s0.py <checkpoint_root>
env:   S0_SHAPES=S5,S6 (subset), S0_SKIP_P2=1, S0_ROUNDS (3), S0_ITERS (20)

Scale scheme (brief): W per output channel s_w[o] = amax|W[o]|/448 (saturating quant, once);
A per tensor, dynamic: a_c = gamma*rstd_g, b_c = beta - mean_g*a_c, hi_c = max(a_c*min_c+b_c,
a_c*max_c+b_c), amax = max(max_c swish(max(hi_c,0)), 0.27846), s_x = amax/448,
q = sat_e4m3(swish(a_c*x+b_c)/s_x) (fp8 NHWC). Conv epilogue: acc*s_x*s_w[o] (+bias)(+res) -> bf16.

P1  cuDNN FE FP8 graph per (shape, epilogue) from ONE builder (build_fp8_graph), plans from
    heur modes A + FALLBACK, every plan timed; refusals recorded verbatim.
P3  Triton T8 implicit-GEMM FP8 conv (no atomics, no split-K: deterministic), small config sweep;
    plus the bf16 Triton control T16 at S5/S6 (gate for an own GN-prologue conv, plan #1370(3)).
    B = aten F.conv2d(h, w, None) bf16 (main's conv), Bb = with bias.  All in one interleaved bench.
    Correctness: every FP8 arm vs fp64 conv of the SAME dequantized operands (declared tol:
    max_abs <= 1e-2 * max|ref|, fp32 accumulation + bf16 store), and vs main's bf16 conv
    (precision error, reported). 5x repeat bitwise; inputs restored from pristine copies before
    timing and before the correctness pass, and checked unchanged.
P4  chains through the next GN-stats kernel: U (main: compiled apply+swish -> conv(+b)(+res) ->
    var_mean), U8 (compiled apply->fp8 + best FP8 conv + compiled var_mean), Dstats (extra
    min/max reduction + a/b/s_x derivation vs main's stats + combine).
    S172 = sum_sites (U - U8) - sum_sites Dstats; unc = sum of p10-p90 half widths.
P2  emulated decode: real checkpoint VAE, bf16 trunk, all 28 GN->conv pairs replaced by the
    FP8 math above (eager, fp32 conv on exactly-dequantized operands, TF32 off), seeds 0-2,
    PSNR / u8 vs R = this tree's fp32 (checkpoint dtype) eager decoder, next to main's bf16
    decoder (eager and compiled) and an emulation control without quantization.
"""

import json
import math
import os
import sys
import time
import types

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

import cudnn

ART = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp/probe_fp8_s0")
os.makedirs(ART, exist_ok=True)
OUT = open(os.path.join(ART, "probe_fp8_s0.jsonl"), "w")
ROOT = sys.argv[1] if len(sys.argv) > 1 else None


def emit(d):
    s = json.dumps(d)
    print(s, flush=True)
    OUT.write(s + "\n")
    OUT.flush()


torch.manual_seed(0)
dev = "cuda"
torch._dynamo.config.cache_size_limit = 256
torch.backends.cudnn.benchmark = False
FP8 = torch.float8_e4m3fn
QMAX = 448.0
SWISH_MIN = 0.27846

emit({"versions": {"cudnn_backend": cudnn.backend_version(), "fe": getattr(cudnn, "__version__", "?"),
                   "torch": torch.__version__, "torch_cudnn": torch.backends.cudnn.version(),
                   "triton": triton.__version__, "gpu": torch.cuda.get_device_name()}})

# name, cin, cout, hw, n_conv1 (bias), n_conv2 (bias+res), n_tail (scale only)
SHAPES = [
    ("S5", 256, 128, 1024, 1, 0, 0),
    ("S6", 128, 128, 1024, 2, 2, 1),
    ("S3", 512, 256, 512, 1, 0, 0),
    ("S4", 256, 256, 512, 2, 3, 0),
    ("S2", 512, 512, 256, 3, 3, 0),
    ("S1", 512, 512, 128, 5, 5, 0),
]
only = os.environ.get("S0_SHAPES")
if only:
    SHAPES = [s for s in SHAPES if s[0] in only.split(",")]
ROUNDS = int(os.environ.get("S0_ROUNDS", "3"))
ITERS = int(os.environ.get("S0_ITERS", "20"))
OPTS = {"layout_optimization": False, "emulate_precision_casts": True}
G = 32
EPS = 1e-6

BF16 = cudnn.data_type.BFLOAT16
F32 = cudnn.data_type.FLOAT
E4M3 = cudnn.data_type.FP8_E4M3
handle = cudnn.create_handle()
cudnn.set_stream(handle=handle, stream=torch.cuda.current_stream().cuda_stream)


# ------------------------------------------------------------------ shared math
def comb_main(var_c, mean_c):
    c = mean_c.shape[1]
    mc = mean_c.view(1, G, c // G)
    vc = var_c.view(1, G, c // G)
    mg = mc.mean(-1, keepdim=True)
    vg = vc.mean(-1, keepdim=True) + (mc - mg).square().mean(-1, keepdim=True)
    rg = torch.rsqrt(vg + EPS)
    mean = mg.expand(1, G, c // G).reshape(1, c, 1, 1).float()
    rstd = rg.expand(1, G, c // G).reshape(1, c, 1, 1).float()
    return mean, rstd


def ab_sx(var_c, mean_c, mn_c, mx_c, gamma, beta):
    """Per-channel a, b and the per-tensor activation scale s_x (no host sync)."""
    mean, rstd = comb_main(var_c, mean_c)
    a = rstd * gamma.float().view(1, -1, 1, 1)
    b = beta.float().view(1, -1, 1, 1) - mean * a
    lo = a * mn_c.view(1, -1, 1, 1) + b
    hi = a * mx_c.view(1, -1, 1, 1) + b
    top = torch.maximum(lo, hi).clamp_min(0)
    amax = torch.clamp((top * torch.sigmoid(top)).amax(), min=SWISH_MIN)
    return a, b, amax / QMAX


def quant_w(w):
    """[Cout,Cin,3,3] -> (fp8 weights, s_w[Cout] fp32), per output channel, saturating."""
    wf = w.float()
    sw = wf.abs().amax(dim=(1, 2, 3)).clamp_min(1e-12) / QMAX
    q = (wf / sw.view(-1, 1, 1, 1)).clamp(-QMAX, QMAX).to(FP8)
    return q, sw


# main's GN apply (bf16 out) + swish, as group_norm_channel_stats math
def _apply(x, mean, rstd, gamma, beta):
    xf = x.float()
    y = (xf - mean) * rstd
    y = y * gamma.float().view(1, -1, 1, 1) + beta.float().view(1, -1, 1, 1)
    y = y.to(x.dtype)
    return y * torch.sigmoid(y)


def u_conv1(x, mean, rstd, gamma, beta, w, b):
    h = F.conv2d(_apply(x, mean, rstd, gamma, beta), w, b, padding=1)
    return torch.var_mean(h.float(), dim=(2, 3), correction=0)


def u_conv2(x, mean, rstd, gamma, beta, w, b, res):
    h = F.conv2d(_apply(x, mean, rstd, gamma, beta), w, b, padding=1)
    return torch.var_mean((res + h).float(), dim=(2, 3), correction=0)


def u_tail(x, mean, rstd, gamma, beta, w):
    return F.conv2d(_apply(x, mean, rstd, gamma, beta), w, None, padding=1)


def apply_fp8(x, a, b, sx):
    y = x.float() * a + b
    y = y * torch.sigmoid(y)
    return (y / sx).clamp(-QMAX, QMAX).to(FP8)


def c_stats(y):
    return torch.var_mean(y.float(), dim=(2, 3), correction=0)


def c_stats_mm(y):
    yf = y.float()
    v, m = torch.var_mean(yf, dim=(2, 3), correction=0)
    return v, m, yf.amin(dim=(2, 3)), yf.amax(dim=(2, 3))


def cc(fn):
    return torch.compile(fn, dynamic=False, options=OPTS)


# ------------------------------------------------------------------ cuDNN FE FP8 builder
def nhwc(c, h, w):
    return [c * h * w, 1, w * c, c]


def build_fp8_graph(key):
    """ONE per-key builder. key = (cin, cout, hw, epi), epi in {'s', 'sb', 'sbr'}."""
    cin, cout, hw, epi = key
    g = cudnn.pygraph(io_data_type=BF16, intermediate_data_type=F32, compute_data_type=F32, handle=handle)
    T = {}
    T["X"] = g.tensor(name="X", dim=[1, cin, hw, hw], stride=nhwc(cin, hw, hw), data_type=E4M3)
    T["W"] = g.tensor(name="W", dim=[cout, cin, 3, 3], stride=nhwc(cin, 3, 3), data_type=E4M3)
    T["SX"] = g.tensor(name="SX", dim=[1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=F32)
    T["SW"] = g.tensor(name="SW", dim=[1, cout, 1, 1], stride=[cout, 1, cout, cout], data_type=F32)
    y = g.conv_fprop(image=T["X"], weight=T["W"], padding=[1, 1], stride=[1, 1], dilation=[1, 1],
                     compute_data_type=F32, name="conv")
    y.set_data_type(F32)
    y = g.mul(a=y, b=T["SX"], compute_data_type=F32, name="sx")
    y = g.mul(a=y, b=T["SW"], compute_data_type=F32, name="sw")
    if "b" in epi:
        T["bias"] = g.tensor(name="bias", dim=[1, cout, 1, 1], stride=[cout, 1, cout, cout], data_type=BF16)
        y = g.add(a=y, b=T["bias"], compute_data_type=F32, name="bias_add")
    if "r" in epi:
        T["R"] = g.tensor(name="R", dim=[1, cout, hw, hw], stride=nhwc(cout, hw, hw), data_type=BF16)
        y = g.add(a=y, b=T["R"], compute_data_type=F32, name="res_add")
    y.set_output(True).set_data_type(BF16)
    T["Y"] = y
    info = {}
    t0 = time.perf_counter()
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    g.check_support()
    t1 = time.perf_counter()
    try:
        g.build_plans(cudnn.build_plan_policy.ALL)
    except Exception as e:  # noqa: BLE001
        info["build_all_error"] = repr(e)[:300]
    t2 = time.perf_counter()
    n = g.get_execution_plan_count()
    names, ws = [], []
    for i in range(n):
        try:
            names.append(g.get_plan_name_at_index(i))
        except Exception as e:  # noqa: BLE001
            names.append(f"?{type(e).__name__}")
        try:
            ws.append(int(g.get_workspace_size_plan_at_index(i)))
        except Exception:  # noqa: BLE001
            ws.append(int(g.get_workspace_size()))
    info.update({"graph_s": t1 - t0, "build_plans_s": t2 - t1, "n_plans": n, "names": names, "ws": ws})
    return g, T, info


class FEArm:
    def __init__(self, g, T, info, tensors):
        self.g, self.T, self.info = g, T, info
        self.vp = {T[k]: v for k, v in tensors.items() if k in T and k != "Y"}
        self.out = tensors["Y"]
        self.vp[T["Y"]] = self.out
        self.ws = torch.empty(max(info["ws"] + [1]), dtype=torch.uint8, device=dev)
        self.ok = []
        for i in range(info["n_plans"]):
            try:
                self.run(i)
                torch.cuda.synchronize()
                self.ok.append(i)
            except Exception as e:  # noqa: BLE001
                emit({"plan_exec_fail": i, "name": info["names"][i] if i < len(info["names"]) else "?",
                      "err": repr(e)[:200]})

    def run(self, i):
        self.g.execute_plan_at_index(self.vp, self.ws, i, handle=handle)


# ------------------------------------------------------------------ Triton T8 / T16
@triton.jit
def _conv3x3_nhwc(X, Wt, Y, SX, SW, BIAS, RES, H, WD,
                  CIN: tl.constexpr, COUT: tl.constexpr, SCALED: tl.constexpr,
                  HAS_BIAS: tl.constexpr, HAS_RES: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    NT: tl.constexpr = COUT // BN
    pid = tl.program_id(0)
    pid_n = pid % NT
    pid_m = pid // NT
    M = H * WD
    offs_m = pid_m * BM + tl.arange(0, BM)
    m_ok = offs_m < M
    ph = offs_m // WD
    pw = offs_m % WD
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kk in range(0, 9 * CIN, BK):
        tap = kk // CIN
        ci = kk - tap * CIN
        ih = ph + (tap // 3 - 1)
        iw = pw + (tap % 3 - 1)
        ok = m_ok & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < WD)
        a = tl.load(X + ((ih * WD + iw) * CIN + ci)[:, None] + offs_k[None, :], mask=ok[:, None], other=0.0)
        b = tl.load(Wt + ((kk + offs_k) * COUT)[:, None] + offs_n[None, :])
        acc = tl.dot(a, b, acc)
    if SCALED:
        acc = acc * tl.load(SX) * tl.load(SW + offs_n)[None, :]
    if HAS_BIAS:
        acc += tl.load(BIAS + offs_n).to(tl.float32)[None, :]
    yoff = offs_m[:, None] * COUT + offs_n[None, :]
    if HAS_RES:
        acc += tl.load(RES + yoff, mask=m_ok[:, None], other=0.0).to(tl.float32)
    tl.store(Y + yoff, acc.to(tl.bfloat16), mask=m_ok[:, None])


T8_CFGS = [(128, 128, 128, 8, 3), (128, 128, 64, 4, 4), (128, 64, 128, 4, 3), (256, 128, 64, 8, 3),
           (64, 128, 128, 4, 4), (128, 128, 128, 4, 2)]
T16_CFGS = [(128, 128, 64, 8, 3), (128, 64, 64, 4, 4), (256, 128, 32, 8, 3), (128, 128, 32, 4, 4),
            (64, 128, 64, 4, 4)]


def t_conv(xq, wt, y, sx, sw, bias, res, hw, cin, cout, cfg, scaled=True):
    bm, bn, bk, nw, ns = cfg
    grid = ((hw * hw // bm) * (cout // bn),)
    _conv3x3_nhwc[grid](xq, wt, y, sx, sw, bias if bias is not None else sw, res if res is not None else y,
                        hw, hw, CIN=cin, COUT=cout, SCALED=scaled, HAS_BIAS=bias is not None,
                        HAS_RES=res is not None, BM=bm, BN=bn, BK=bk, num_warps=nw, num_stages=ns)


# ------------------------------------------------------------------ timing
def pct(v, q):
    v = sorted(v)
    k = (len(v) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def bench(fns, rounds=ROUNDS, iters=ITERS, warm=5):
    res = {k: [] for k in fns}
    for f in fns.values():
        for _ in range(warm):
            f()
    torch.cuda.synchronize()
    for _ in range(rounds):
        for k, f in fns.items():
            evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
            for s, e in evs:
                s.record()
                f()
                e.record()
            torch.cuda.synchronize()
            res[k] += [s.elapsed_time(e) for s, e in evs]
    return {k: {"med": pct(v, 0.5), "p10": pct(v, 0.1), "p90": pct(v, 0.9), "n": len(v)} for k, v in res.items()}


def hw2(t):
    return (t["p90"] - t["p10"]) / 2


def emu_kernel_seq(f, n=5):
    torch.cuda.synchronize()
    outs = []
    for _ in range(n):
        outs.append(f().clone())
    torch.cuda.synchronize()
    return outs


# ================================================================== P1 / P3 / P4 per shape
S172, S172_unc, Sconv = 0.0, 0.0, 0.0
summary = {}
all_ok = True

for name, cin, cout, hw, n1, n2, nt in SHAPES:
    t_shape = time.time()
    print(f"=== {name} {cin}->{cout} @{hw}^2", flush=True)
    row = {"cin": cin, "cout": cout, "hw": hw, "n": [n1, n2, nt]}
    try:
        CL = torch.channels_last
        x = (torch.randn(1, cin, hw, hw, device=dev) * 1.5 + 0.3).to(torch.bfloat16).contiguous(memory_format=CL)
        gamma = (1 + 0.2 * torch.randn(cin, device=dev)).to(torch.bfloat16)
        beta = (0.2 * torch.randn(cin, device=dev)).to(torch.bfloat16)
        w = (torch.randn(cout, cin, 3, 3, device=dev) / math.sqrt(9 * cin)).to(torch.bfloat16).contiguous(memory_format=CL)
        bvec = (0.1 * torch.randn(cout, device=dev)).to(torch.bfloat16)
        res = torch.randn(1, cout, hw, hw, device=dev).to(torch.bfloat16).contiguous(memory_format=CL)
        var_c, mean_c = torch.var_mean(x.float(), dim=(2, 3), correction=0)
        mn_c, mx_c = x.float().amin(dim=(2, 3)), x.float().amax(dim=(2, 3))
        mean, rstd = comb_main(var_c, mean_c)
        a_s, b_s, sx = ab_sx(var_c, mean_c, mn_c, mx_c, gamma, beta)
        a_s, b_s, sx = a_s.contiguous(), b_s.contiguous(), sx.reshape(1).contiguous()
        h_bf = _apply(x, mean, rstd, gamma, beta).contiguous(memory_format=CL)
        xq = apply_fp8(x, a_s, b_s, sx).contiguous(memory_format=CL)
        wq, sw = quant_w(w)
        wq = wq.contiguous(memory_format=CL)
        wt = wq.permute(2, 3, 1, 0).reshape(9 * cin, cout).contiguous()
        wt16 = w.permute(2, 3, 1, 0).reshape(9 * cin, cout).contiguous()
        sx4, sw4, b4 = sx.view(1, 1, 1, 1), sw.view(1, cout, 1, 1).contiguous(), bvec.view(1, cout, 1, 1)
        row["sx"] = sx.item()
        row["xq_sat_frac"] = (xq.float().abs() >= QMAX).float().mean().item()
        # pristine copies of every input
        inputs = {"x": x, "h_bf": h_bf, "xq": xq, "wq": wq, "wt": wt, "wt16": wt16, "w": w, "sx": sx,
                  "sw": sw, "bvec": bvec, "res": res, "a_s": a_s, "b_s": b_s, "mean": mean, "rstd": rstd,
                  "gamma": gamma, "beta": beta}
        pristine = {k: v.clone() for k, v in inputs.items()}

        def restore():
            for k, v in inputs.items():
                v.copy_(pristine[k])

        def unchanged():
            return all(torch.equal(v, pristine[k]) for k, v in inputs.items())

        def yb():
            return torch.empty(1, cout, hw, hw, device=dev, dtype=torch.bfloat16).contiguous(memory_format=CL)

        epis = {"sb": (bvec, None)}
        if n2:
            epis["sbr"] = (bvec, res)
        if nt:
            epis["s"] = (None, None)

        # ---- P1: cuDNN FE FP8
        c8 = {}
        for epi in epis:
            key = (cin, cout, hw, epi)
            try:
                g, T, info = build_fp8_graph(key)
                tens = {"X": xq, "W": wq, "SX": sx4, "SW": sw4, "bias": b4, "R": res, "Y": yb()}
                arm = FEArm(g, T, info, tens)
                c8[epi] = arm
                emit({"shape": name, "P1": epi, **info, "ok_plans": arm.ok})
            except Exception as e:  # noqa: BLE001
                emit({"shape": name, "P1": epi, "refusal": repr(e)[:600]})
        row["C8"] = {e: {"n_plans": a.info["n_plans"], "ok": a.ok, "names": a.info["names"]} for e, a in c8.items()}

        # ---- P3: Triton T8 config sweep on the 'sb' epilogue, T16 control at S5/S6
        def sweep(cfgs, scaled, xin, wtin, tag):
            yy = yb()
            fns, errs = {}, {}
            for cfg in cfgs:
                if cout % cfg[1] or cin % cfg[2]:
                    continue
                f = (lambda cfg=cfg: t_conv(xin, wtin, yy, sx, sw, bvec, None, hw, cin, cout, cfg, scaled))
                try:
                    f()
                    torch.cuda.synchronize()
                    fns[str(cfg)] = f
                except Exception as e:  # noqa: BLE001
                    errs[str(cfg)] = repr(e)[:160]
            tt = bench(fns, rounds=1, iters=10, warm=2) if fns else {}
            best = min(tt, key=lambda k: tt[k]["med"]) if tt else None
            emit({"shape": name, "P3": tag, "sweep": {k: v["med"] for k, v in tt.items()}, "errors": errs, "best": best})
            return (eval(best) if best else None)

        t8cfg = sweep(T8_CFGS, True, xq, wt, "T8")
        t16cfg = sweep(T16_CFGS, False, h_bf, wt16, "T16") if name in ("S5", "S6") else None
        row["T8_cfg"], row["T16_cfg"] = t8cfg, t16cfg

        # ---- interleaved conv bench
        restore()
        y_b = yb()
        fns = {"B": lambda: F.conv2d(h_bf, w, None, padding=1), "Bb": lambda: F.conv2d(h_bf, w, bvec, padding=1)}
        for epi, arm in c8.items():
            for i in arm.ok:
                fns[f"C8_{epi}_p{i}"] = (lambda arm=arm, i=i: arm.run(i))
        if t8cfg:
            for epi, (bb, rr) in epis.items():
                fns[f"T8_{epi}"] = (lambda bb=bb, rr=rr: t_conv(xq, wt, y_b, sx, sw, bb, rr, hw, cin, cout, t8cfg))
        if t16cfg:
            fns["T16_sb"] = lambda: t_conv(h_bf, wt16, y_b, sx, sw, bvec, None, hw, cin, cout, t16cfg, False)
        tp = bench(fns)
        row["inputs_unchanged_after_timing"] = unchanged()
        emit({"shape": name, "conv_bench": tp})

        def best_c8(epi):
            ks = [k for k in tp if k.startswith(f"C8_{epi}_p")]
            return min(ks, key=lambda k: tp[k]["med"]) if ks else None

        eng = {}
        for epi in epis:
            kc = best_c8(epi)
            kt = f"T8_{epi}" if f"T8_{epi}" in tp else None
            cand = [k for k in (kc, kt) if k]
            bestk = min(cand, key=lambda k: tp[k]["med"]) if cand else None
            eng[epi] = {"C8": kc, "C8_plan": (c8[epi].info["names"][int(kc.rsplit("_p", 1)[1])] if kc else None),
                        "C8_ms": tp[kc] if kc else None, "T8_ms": tp[kt] if kt else None, "best": bestk}
        row["engine"] = eng
        row["B"], row["Bb"] = tp["B"], tp["Bb"]
        if "T16_sb" in tp:
            row["T16_sb"] = tp["T16_sb"]

        def run_best(epi, out=None):
            k = eng[epi]["best"]
            if k is None:
                return None
            if k.startswith("C8"):
                c8[epi].run(int(k.rsplit("_p", 1)[1]))
                return c8[epi].out
            bb, rr = epis[epi]
            yy = out if out is not None else y_b
            t_conv(xq, wt, yy, sx, sw, bb, rr, hw, cin, cout, t8cfg)
            return yy

        # ---- correctness: restore, 5x repeat bitwise per FP8 arm, vs fp64 of same operands
        restore()
        num = {}
        with torch.no_grad():
            xd = xq.double() * sx.double()
            wd = wq.double() * sw.double().view(-1, 1, 1, 1)
            c64 = F.conv2d(xd, wd, None, padding=1)
            u64 = F.conv2d(h_bf.double(), w.double(), None, padding=1)
            for epi, (bb, rr) in epis.items():
                ref = c64 + (bb.double().view(1, -1, 1, 1) if bb is not None else 0) + (rr.double() if rr is not None else 0)
                um = F.conv2d(h_bf, w, bb, padding=1)
                if rr is not None:
                    um = rr + um
                arms_ck = {}
                if best_c8(epi):
                    kc = best_c8(epi)
                    arms_ck["C8"] = lambda kc=kc, epi=epi: (c8[epi].run(int(kc.rsplit("_p", 1)[1])), c8[epi].out)[1]
                if t8cfg:
                    yt = yb()
                    arms_ck["T8"] = lambda bb=bb, rr=rr, yt=yt: (t_conv(xq, wt, yt, sx, sw, bb, rr, hw, cin, cout, t8cfg), yt)[1]
                for an, f in arms_ck.items():
                    outs = emu_kernel_seq(f, 5)
                    rep = all(torch.equal(outs[0], o) for o in outs[1:])
                    o0 = outs[0].double()
                    ra = ref.abs().max().item()
                    e64 = (o0 - ref).abs().max().item()
                    ok = rep and e64 <= 1e-2 * ra
                    all_ok = all_ok and ok
                    num[f"{an}_{epi}"] = {"repeat5_bitwise": rep, "max_abs_vs_fp64_same_operands": e64, "ref_absmax": ra,
                                          "tol": 1e-2 * ra, "passed": ok,
                                          "max_abs_vs_main_bf16": (o0 - um.double()).abs().max().item(),
                                          "rel_l2_vs_main_bf16": ((o0 - um.double()).norm() / um.double().norm()).item(),
                                          "main_bf16_vs_fp64_unquant_max": (um.double() - (u64 + (ref - c64))).abs().max().item()}
            del xd, wd, c64, u64
        num["inputs_unchanged"] = unchanged()
        all_ok = all_ok and num["inputs_unchanged"] and row["inputs_unchanged_after_timing"]
        row["numerics"] = num
        emit({"shape": name, "numerics": num})

        # ---- P4: chains through the next GN-stats kernel
        restore()
        cu1 = cc(u_conv1)
        cu2 = cc(u_conv2) if n2 else None
        cut = cc(u_tail) if nt else None
        cap = cc(apply_fp8)
        cst = cc(c_stats)
        cmm = cc(c_stats_mm)
        ccomb = cc(comb_main)
        cabsx = cc(ab_sx)
        q_t = cap(x, a_s, b_s, sx)
        row["apply_fp8_out_channels_last"] = q_t.is_contiguous(memory_format=torch.channels_last)
        row["apply_fp8_bitwise_vs_eager"] = torch.equal(q_t.view(torch.uint8), xq.view(torch.uint8))
        y8 = {e: yb() for e in epis}

        def u8(epi):
            q = cap(x, a_s, b_s, sx)
            k = eng[epi]["best"]
            if k.startswith("C8"):
                arm = c8[epi]
                arm.vp[arm.T["X"]] = q
                arm.run(int(k.rsplit("_p", 1)[1]))
                yy = arm.out
            else:
                bb, rr = epis[epi]
                yy = y8[epi]
                t_conv(q, wt, yy, sx, sw, bb, rr, hw, cin, cout, t8cfg)
            return yy if epi == "s" else cst(yy)

        chains = {"U_conv1": lambda: cu1(x, mean, rstd, gamma, beta, w, bvec),
                  "Dstats_main": lambda: (cst(x), ccomb(var_c, mean_c)),
                  "Dstats_fp8": lambda: (cmm(x), cabsx(var_c, mean_c, mn_c, mx_c, gamma, beta))}
        if eng["sb"]["best"]:
            chains["U8_conv1"] = lambda: u8("sb")
        if n2:
            chains["U_conv2"] = lambda: cu2(x, mean, rstd, gamma, beta, w, bvec, res)
            if eng["sbr"]["best"]:
                chains["U8_conv2"] = lambda: u8("sbr")
        if nt:
            chains["U_tail"] = lambda: cut(x, mean, rstd, gamma, beta, w)
            if eng["s"]["best"]:
                chains["U8_tail"] = lambda: u8("s")
        tc = bench(chains)
        row["chains"] = tc
        row["inputs_unchanged_after_chains"] = unchanged()
        emit({"shape": name, "chains": tc})
        dst = tc["Dstats_fp8"]["med"] - tc["Dstats_main"]["med"]
        dst_unc = hw2(tc["Dstats_fp8"]) + hw2(tc["Dstats_main"])
        rows = {}
        for kind, n, epi in (("conv1", n1, "sb"), ("conv2", n2, "sbr"), ("tail", nt, "s")):
            if not n or f"U8_{kind}" not in tc:
                continue
            U, U8 = tc[f"U_{kind}"], tc[f"U8_{kind}"]
            sav = U["med"] - U8["med"] - dst
            unc = hw2(U) + hw2(U8) + dst_unc
            bk = eng[epi]["best"]
            conv_sav = (tp["B"]["med"] - tp[bk]["med"]) if bk else 0.0
            rows[kind] = {"n": n, "U": U["med"], "U8": U8["med"], "Dstats": dst, "save_each": sav, "unc_each": unc,
                          "engine": bk, "conv_only_B_minus_best8": conv_sav}
            S172 += n * sav
            S172_unc += n * unc
            Sconv += n * conv_sav
        row["S_rows"] = rows
        emit({"shape": name, "S_rows": rows})
        del c8, fns, chains
    except Exception as e:  # noqa: BLE001
        import traceback

        row["error"] = traceback.format_exc()[-1500:]
        emit({"shape": name, "error": row["error"]})
        all_ok = False
    row["shape_s"] = time.time() - t_shape
    summary[name] = row
    torch.cuda.empty_cache()

emit({"S172_ms": S172, "S172_unc_ms": S172_unc, "Sconv_only_ms": Sconv})


# ================================================================== P2 emulated decode
p2 = {}
if os.environ.get("S0_SKIP_P2") != "1" and ROOT:
    try:
        sys.path.insert(0, os.path.join(os.getcwd(), "experiments", "ascent"))
        import gn_check  # noqa: E402

        from tensorrt_llm._torch.visual_gen.models.flux3.checkpoint import Flux3Checkpoint
        from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import set_vae_decoder_trunk_dtype

        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        DTYPE = getattr(torch, Flux3Checkpoint(ROOT, "nvfp4").config["vae"]["dtype"])

        def blocks(dec):
            bl = [dec.mid.block_1, dec.mid.block_2]
            for lvl in dec.up:
                bl += list(lvl.block)
            return bl

        WQ = {}

        def emu_conv(norm, conv, x, bias=True, quant=True):
            xf = x.float()
            var_c, mean_c = torch.var_mean(xf, dim=(2, 3), correction=0)
            mn, mx = xf.amin(dim=(2, 3)), xf.amax(dim=(2, 3))
            a, b, sx = ab_sx(var_c, mean_c, mn, mx, norm.weight, norm.bias)
            y = xf * a + b
            s = y * torch.sigmoid(y)
            if quant:
                xin = (s / sx).clamp(-QMAX, QMAX).to(FP8).float() * sx
                if id(conv) not in WQ:
                    q, sw_ = quant_w(conv.weight)
                    WQ[id(conv)] = q.float() * sw_.view(-1, 1, 1, 1)
                wgt = WQ[id(conv)]
            else:
                xin, wgt = s, conv.weight.float()
            out = F.conv2d(xin, wgt, conv.bias.float() if bias else None, padding=1)
            return out.to(x.dtype)

        def patch(dec, quant):
            def fwd(self, x):
                h = emu_conv(self.norm1, self.conv1, x, True, quant)
                h = emu_conv(self.norm2, self.conv2, h, True, quant)
                if self.in_channels != self.out_channels:
                    x = self.nin_shortcut(x)
                return x + h

            def parts(self, x):
                h = emu_conv(self.norm1, self.conv1, x, True, quant)
                h = emu_conv(self.norm2, self.conv2, h, False, quant)
                if self.in_channels != self.out_channels:
                    x = self.nin_shortcut(x)
                return x, h, self.conv2.bias

            bl = blocks(dec)
            for blk in bl:
                blk.forward = types.MethodType(fwd, blk)
                blk.forward_parts = types.MethodType(parts, blk)
            return 2 * len(bl)

        def mk(trunk=None, compiled=False, emu=None):
            vae, _ = gn_check.build_vae(ROOT)
            n = None
            if trunk is not None:
                set_vae_decoder_trunk_dtype(vae.decoder, trunk)
            if emu is not None:
                n = patch(vae.decoder, emu)
            if compiled:
                vae.compile_decoder()
            return vae, n

        def psnr(a, b):
            mse = ((a.float().clamp(-1, 1) - b.float().clamp(-1, 1)) ** 2).mean().item()
            return float("inf") if mse == 0 else 10 * math.log10(4.0 / mse)

        def diff(a, b):
            du = (gn_check.to_u8(a.float()).int() - gn_check.to_u8(b.float()).int()).abs()
            return {"psnr_db": psnr(a, b), "max_abs": (a.float() - b.float()).abs().max().item(),
                    "u8_max": du.max().item(), "u8_frac": (du > 0).float().mean().item(),
                    "u8_mean": du.float().mean().item()}

        zs = []
        for s in range(3):
            gg = torch.Generator(device="cuda").manual_seed(s)
            zs.append(torch.randn(1, 128, 64, 64, device="cuda", generator=gg).to(DTYPE))
        z0 = [z.clone() for z in zs]
        outs = {}
        arms = [("R_fp32_eager", dict()),
                ("main_bf16_eager", dict(trunk=torch.bfloat16)),
                ("main_bf16_compiled", dict(trunk=torch.bfloat16, compiled=True)),
                ("emu_noq_bf16", dict(trunk=torch.bfloat16, emu=False)),
                ("emu_fp8_bf16", dict(trunk=torch.bfloat16, emu=True))]
        for an, kw in arms:
            t0 = time.time()
            vae, n = mk(**kw)
            with torch.inference_mode():
                o = [vae.decode(z).float().clone() for z in zs]
                rep = all(torch.equal(vae.decode(z).float(), oo) for z, oo in zip(zs, o))
            outs[an] = o
            emit({"P2_arm": an, "patched_convs": n, "repeat_bitwise": rep, "s": time.time() - t0})
            del vae
            torch.cuda.empty_cache()
        R = outs["R_fp32_eager"]
        for an in outs:
            if an == "R_fp32_eager":
                continue
            per = [diff(outs[an][i], R[i]) for i in range(3)]
            p2[an] = {"vs_R_per_seed": per, "psnr_min": min(p["psnr_db"] for p in per),
                      "psnr_mean": sum(p["psnr_db"] for p in per) / 3, "u8_max": max(p["u8_max"] for p in per),
                      "u8_frac_max": max(p["u8_frac"] for p in per)}
        p2["emu_fp8_vs_main_bf16_compiled"] = [diff(outs["emu_fp8_bf16"][i], outs["main_bf16_compiled"][i]) for i in range(3)]
        p2["latents_unchanged"] = all(torch.equal(a, b) for a, b in zip(zs, z0))
        emit({"P2": p2})
    except Exception:  # noqa: BLE001
        import traceback

        p2["error"] = traceback.format_exc()[-1500:]
        emit({"P2_error": p2["error"]})

compact = {k: {"engine": {e: (v["best"], v["C8_plan"], v["C8_ms"]["med"] if v["C8_ms"] else None,
                              v["T8_ms"]["med"] if v["T8_ms"] else None) for e, v in r.get("engine", {}).items()},
               "B_med": r.get("B", {}).get("med"), "B_p10": r.get("B", {}).get("p10"), "B_p90": r.get("B", {}).get("p90"),
               "T16_sb": r.get("T16_sb", {}).get("med"), "S_rows": r.get("S_rows"), "err": r.get("error", "")[-300:]}
           for k, r in summary.items()}
with open(os.path.join(ART, "summary.json"), "w") as f:
    json.dump({"summary": summary, "P2": p2, "S172_ms": S172, "S172_unc_ms": S172_unc}, f)
print(json.dumps({"passed": bool(all_ok and "error" not in p2), "S172_ms": S172, "S172_unc_ms": S172_unc,
                  "Sconv_only_ms": Sconv, "shapes": compact,
                  "P2": {k: ({"psnr_min": v["psnr_min"], "u8_max": v["u8_max"]} if isinstance(v, dict) and "psnr_min" in v else None)
                         for k, v in p2.items()}}))
