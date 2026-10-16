# Correctness check for the small-N BF16 GEMV route (in_proj_ba, N=96, K=5120),
# through the real tensorrt_llm Linear (UnquantizedLinearMethod.apply).
#
# 1. Tolerance (class reordered): per M=1..4 (routed) and two activation
#    distributions, error vs an FP32 torch reference (TF32 off) on the same BF16
#    operands must satisfy max-abs <= 1.25 * cuBLAS max-abs + 1 BF16 ulp of the
#    output magnitude, same for mean-abs; no NaN/Inf.
# 2. Repeated execution: the timed loop's sequence (10 warmup + 50 timed, L2
#    flush before each) extended to 100 timed iterations, untimed, eager. After
#    every iteration the output is compared bitwise with iteration 1 and with
#    the tolerance; per-iteration max error and drift are reported.
# 3. CUDA graph: 48 layers captured, replayed 100 times (flush before each);
#    every replay bitwise equal to the eager outputs.
# 4. Routing: M=1..4 routed; M=5..16, prefill M=2048, N>4096 (lm_head shape
#    N=248320), bias, fp16, non-unit last-dim stride and kill-switch cases are
#    not routed and are bitwise equal to F.linear. Row-strided (sliced) and 3D
#    inputs are routed and correct.
import json
import os
import subprocess
import sys

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modules import small_n_gemv as sng
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev = torch.device("cuda")
N, K, LAYERS = 96, 5120, 48
WARMUP, ITERS = 10, 100
flush_buf = torch.ones(64 * 1024 * 1024, dtype=torch.float32, device=dev)
failures = []


def log(*a):
    print(*a, flush=True)


def fail(msg):
    failures.append(msg)
    log("FAIL:", msg)


def make_linear(w, bias=None):
    lin = Linear(K if w.shape[1] == K else w.shape[1], w.shape[0],
                 bias=bias is not None, dtype=w.dtype,
                 tensor_parallel_mode=TensorParallelMode.COLUMN).to(dev)
    lin.weight.data.copy_(w)
    if bias is not None:
        lin.bias.data.copy_(bias)
    return lin


def ulp_bf16(x):
    m = x.abs().max().item()
    if m == 0:
        return 0.0
    e = torch.floor(torch.log2(torch.tensor(m))).item()
    return 2.0**(e - 7)


def errs(out, ref):
    d = (out.float() - ref).abs()
    return d.max().item(), d.mean().item()


def make_x(m, kind, gen):
    x = torch.randn(m, K, device=dev, generator=gen)
    if kind == "outlier":
        # LLM hidden states: a few large channels.
        idx = torch.randperm(K, device=dev, generator=gen)[:16]
        x[:, idx] *= 30.0
    return x.to(torch.bfloat16)


def main():
    log("gpu", torch.cuda.get_device_name(), "sm",
        torch.cuda.get_device_capability())
    gen = torch.Generator(device=dev)
    gen.manual_seed(2025)
    ws = [(torch.randn(N, K, device=dev, generator=gen) * 0.02).to(
        torch.bfloat16) for _ in range(LAYERS)]
    lins = [make_linear(w) for w in ws]
    report = {"tolerance": {}, "repeat": {}, "graph": {}, "routing": {}}

    # 1. tolerance
    for m in (1, 2, 3, 4):
        for kind in ("normal", "outlier"):
            worst = None
            for li in range(0, LAYERS, 6):
                x = make_x(m, kind, gen)
                ref = x.float() @ ws[li].float().t()
                cub = F.linear(x, ws[li])
                assert sng.maybe_small_n_gemv(x, ws[li]) is not None
                out = lins[li](x)
                if not torch.isfinite(out).all():
                    fail("non-finite m=%d %s" % (m, kind))
                cmax, cmean = errs(cub, ref)
                omax, omean = errs(out, ref)
                u = ulp_bf16(ref)
                ok = (omax <= 1.25 * cmax + u) and (omean <= 1.25 * cmean + u)
                if not ok:
                    fail("tolerance m=%d %s layer %d: max %.3e (cublas %.3e) "
                         "mean %.3e (cublas %.3e) ulp %.3e" %
                         (m, kind, li, omax, cmax, omean, cmean, u))
                r = (omax / max(cmax, 1e-30), omean / max(cmean, 1e-30))
                if worst is None or r[0] > worst["max_ratio"]:
                    worst = {"layer": li, "max": omax, "mean": omean,
                             "cublas_max": cmax, "cublas_mean": cmean,
                             "ulp": u, "max_ratio": r[0], "mean_ratio": r[1],
                             "bitwise_eq_cublas_frac":
                             (out == cub).float().mean().item()}
            report["tolerance"]["%d_%s" % (m, kind)] = worst
            log("tolerance M=%d %s worst:" % (m, kind), worst)

    # 2. repeated execution, eager, same reset points as timing (flush)
    eager_ref = {}
    for m in (1, 2, 3, 4):
        xs = [make_x(m, "normal", gen) for _ in range(LAYERS)]
        refs = [x.float() @ w.float().t() for x, w in zip(xs, ws)]
        tols = []
        for x, w, ref in zip(xs, ws, refs):
            cmax, cmean = errs(F.linear(x, w), ref)
            tols.append((1.25 * cmax + ulp_bf16(ref),
                         1.25 * cmean + ulp_bf16(ref)))
        first = None
        per_iter = []
        drift = 0
        for it in range(WARMUP + ITERS):
            flush_buf.sum()
            outs = [lin(x) for lin, x in zip(lins, xs)]
            torch.cuda.synchronize()
            if first is None:
                first = [o.clone() for o in outs]
            mx = 0.0
            for li, (o, ref) in enumerate(zip(outs, refs)):
                if not torch.equal(o, first[li]):
                    drift += 1
                emax, emean = errs(o, ref)
                mx = max(mx, emax)
                if emax > tols[li][0] or emean > tols[li][1]:
                    fail("repeat tol m=%d iter %d layer %d" % (m, it, li))
            per_iter.append(mx)
        if drift:
            fail("repeat drift m=%d: %d layer-iterations differ" % (m, drift))
        eager_ref[m] = (xs, first)
        report["repeat"][str(m)] = {
            "iterations": WARMUP + ITERS,
            "bitwise_mismatches_vs_iter1": drift,
            "max_err_first": per_iter[0],
            "max_err_last": per_iter[-1],
            "max_err_range": [min(per_iter), max(per_iter)],
        }
        log("repeat M=%d:" % m, report["repeat"][str(m)])

    # 3. CUDA graph replay vs eager, bitwise
    for m in (1, 2, 3, 4):
        xs, first = eager_ref[m]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            [lin(x) for lin, x in zip(lins, xs)]
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            gouts = [lin(x) for lin, x in zip(lins, xs)]
        mism = 0
        for it in range(ITERS):
            flush_buf.sum()
            g.replay()
            torch.cuda.synchronize()
            for o, e in zip(gouts, first):
                if not torch.equal(o, e):
                    mism += 1
        if mism:
            fail("graph vs eager m=%d: %d mismatches" % (m, mism))
        report["graph"][str(m)] = {"replays": ITERS,
                                   "bitwise_mismatches_vs_eager": mism}
        log("graph M=%d:" % m, report["graph"][str(m)])
        del g

    # 4. routing
    routing = report["routing"]
    w0, lin0 = ws[0], lins[0]
    for m in (1, 2, 3, 4):
        routing["routed_M%d" % m] = sng.maybe_small_n_gemv(
            make_x(m, "normal", gen), w0) is not None
        if not routing["routed_M%d" % m]:
            fail("M=%d not routed" % m)
    for m in (5, 6, 8, 12, 16, 2048, 8192):
        x = make_x(m, "normal", gen)
        r = sng.maybe_small_n_gemv(x, w0) is not None
        eq = torch.equal(lin0(x), F.linear(x, w0))
        routing["M%d_routed" % m] = r
        routing["M%d_bitwise_eq_F_linear" % m] = eq
        if r or not eq:
            fail("M=%d should stay on cuBLAS (routed=%s eq=%s)" % (m, r, eq))
    # wide BF16 linears keep cuBLAS (lm_head shape and N just above table)
    for n in (129, 4097, 248320):
        w = (torch.randn(n, K, device=dev, generator=gen) * 0.02).to(
            torch.bfloat16)
        lin = make_linear(w)
        for m in (1, 4):
            x = make_x(m, "normal", gen)
            r = sng.maybe_small_n_gemv(x, w) is not None
            eq = torch.equal(lin(x), F.linear(x, w))
            routing["N%d_M%d_routed" % (n, m)] = r
            routing["N%d_M%d_bitwise_eq_F_linear" % (n, m)] = eq
            if r or not eq:
                fail("N=%d M=%d should stay on cuBLAS" % (n, m))
        del lin, w
        torch.cuda.empty_cache()
    # bias -> old path
    b = (torch.randn(N, device=dev, generator=gen)).to(torch.bfloat16)
    linb = make_linear(w0, b)
    x = make_x(2, "normal", gen)
    eq = torch.equal(linb(x), F.linear(x, w0, b))
    routing["bias_bitwise_eq_F_linear"] = eq
    if not eq:
        fail("bias path changed")
    # fp16 -> old path
    w16 = w0.to(torch.float16)
    lin16 = make_linear(w16)
    x16 = make_x(2, "normal", gen).to(torch.float16)
    r = sng.maybe_small_n_gemv(x16, w16) is not None
    eq = torch.equal(lin16(x16), F.linear(x16, w16))
    routing["fp16_routed"] = r
    routing["fp16_bitwise_eq_F_linear"] = eq
    if r or not eq:
        fail("fp16 should stay on cuBLAS")
    # non-unit last-dim stride -> old path, same values
    xt = make_x(K, "normal", gen)[:, :2].t()  # [2, K], stride (1, K)
    r = sng.maybe_small_n_gemv(xt, w0) is not None
    eq = torch.equal(lin0(xt), F.linear(xt, w0))
    routing["transposed_input_routed"] = r
    routing["transposed_input_bitwise_eq_F_linear"] = eq
    if r or not eq:
        fail("transposed input should stay on cuBLAS")
    # row-strided input (column slice of a wider tensor) -> routed, correct
    big = make_x(3, "normal", gen)
    wide = torch.cat([big, big], dim=1)  # [3, 2K]
    xs_ = wide[:, K:]
    r = sng.maybe_small_n_gemv(xs_, w0) is not None
    out_s = lin0(xs_)
    eq = torch.equal(out_s, lin0(big.contiguous()))
    routing["row_strided_routed"] = r
    routing["row_strided_bitwise_eq_contiguous"] = eq
    if not r or not eq:
        fail("row-strided input")
    # 3D input -> routed, shape kept, equal to 2D
    x3 = make_x(4, "normal", gen)
    o3 = lin0(x3.view(1, 4, K))
    ok3 = (tuple(o3.shape) == (1, 4, N)
           and torch.equal(o3.view(4, N), lin0(x3)))
    routing["3d_input_ok"] = ok3
    if not ok3:
        fail("3D input")
    # Kill switch: fresh process with TRTLLM_SMALL_N_BF16_GEMV=0.
    env = dict(os.environ, TRTLLM_SMALL_N_BF16_GEMV="0")
    code = ("import torch;from tensorrt_llm._torch.modules import "
            "small_n_gemv as s;x=torch.randn(2,5120,device='cuda')."
            "to(torch.bfloat16);w=torch.randn(96,5120,device='cuda')."
            "to(torch.bfloat16);print('KILL',s.maybe_small_n_gemv(x,w) is None)")
    p = subprocess.run([sys.executable, "-c", code], env=env,
                       capture_output=True, text=True)
    ks = "KILL True" in p.stdout
    routing["kill_switch_disables"] = ks
    if not ks:
        fail("kill switch: " + p.stdout[-300:] + p.stderr[-300:])
    log("routing:", routing)

    report["passed"] = not failures
    report["failures"] = failures
    report["compared"] = ("Linear(in_proj_ba N=96,K=5120) routed GEMV vs FP32 "
                          "torch reference and cuBLAS F.linear (reordered "
                          "tolerance), repeat/graph bitwise, routing")
    print(json.dumps(report), flush=True)


main()
