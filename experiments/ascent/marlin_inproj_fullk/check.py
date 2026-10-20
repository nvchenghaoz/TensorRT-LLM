# Idea 201 / branch 119: untimed correctness for the full-K Marlin decode entry of
# (16384,5120) qkvz and (14336,5120) qkv (TRTLLM_MARLIN_INPROJ_FULLK, read per call).
# Arms in one process: D = env unset (tree default), K = env "0" (kill switch = main's path).
#  1. Touched cases (qkvz/qkv, M = 1..8): no NaN/Inf; FP64-reference max-abs and mean-abs error
#     (dequantized NVFP4 weight, same BF16 activations) of D <= 1.25 x K's + 1 BF16 ulp of the
#     output magnitude, and D <= K + 1 ulp (acceptance item 2) at every M; 3 eager calls equal.
#  2. Untouched cases: (5120,6144) at M 1..8/16/64, rows > 8 (9/12/16/64/129) and prefill (2048)
#     on qkvz/qkv: D bitwise equal to K.
#  3. Repeated execution: per touched shape at M = 3, 4, 8, a graph of norm -> Marlin x 2 copies
#     -> norm (state restored by one in-graph copy), 10 warmup + 50 replays with every output and
#     state buffer NaN-poisoned before each replay, each compared bitwise with replay 1 and with the
#     eager arm-D reference; and the per-replay max error vs the FP64 reference of the raw Marlin
#     output (drift check).
#  4. Every case prints sha256 prefixes of D and K outputs, so the same script run on main
#     (where the env does not exist) shows K == main and untouched D == main across trees.
# Brace-free so it can run inline via python -c.
import hashlib
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
ENV = "TRTLLM_MARLIN_INPROJ_FULLK"
os.environ.pop(ENV, None)
import torch  # noqa: E402

import tensorrt_llm._torch.custom_ops  # noqa: E402,F401
from tensorrt_llm._torch.modules.linear import _pack_nvfp4_weight_for_marlin  # noqa: E402

T0 = time.time()


def log(*a):
    print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


for word in sys.argv[1:]:
    print(word, flush=True)
try:
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          timeout=30).stdout.strip()
    tree = subprocess.run(["git", "rev-parse", "HEAD^" + chr(123) + "tree" + chr(125)],
                          capture_output=True, text=True, timeout=30).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                           timeout=30).stdout.strip().splitlines()
    print("TREE_COMMIT=" + (head or "unknown") + " TREE=" + tree + " dirty_files=" + str(len(dirty)),
          flush=True)
except Exception as exc:  # noqa: BLE001
    print("TREE_COMMIT=unknown " + repr(exc), flush=True)

dev = torch.device("cuda")
EPS = 1e-6
WARM, ITERS = 10, 50
SHAPES = (("qkvz", 16384, 5120), ("qkv", 14336, 5120), ("outproj", 5120, 6144))
TOUCHED_M = (1, 2, 3, 4, 5, 6, 7, 8)
UNTOUCHED_M = (9, 12, 16, 64, 129, 2048)
GRAPH_M = (3, 4, 8)
log("gpu", torch.cuda.get_device_name(dev), "PDL", os.environ.get("TRTLLM_ENABLE_PDL"))
LUT = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
                   dtype=torch.float64, device=dev)


def set_env(v):
    if v is None:
        os.environ.pop(ENV, None)
    else:
        os.environ[ENV] = v


def build(n, k, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    w = torch.randn(n, k, generator=g, device=dev) * 0.02
    w = w * (torch.rand(n, 1, generator=g, device=dev) * 1.5 + 0.25)
    w[:, 3::211] *= 8.0
    w = w.to(torch.bfloat16)
    amax = w.float().abs().amax()
    gsc = torch.exp2(torch.ceil(torch.log2(amax / (448.0 * 6.0))))
    w_fp4, w_sf = torch.ops.trtllm.fp4_quantize(w, (1.0 / gsc).reshape(1).float(), 16, False, True)
    del w
    packed = _pack_nvfp4_weight_for_marlin(w_fp4, w_sf, gsc.reshape(1).float(), n, k, 16)
    sf = torch.ops.trtllm.block_scale_interleave_reverse(w_sf.view(n, -1))[:, :k // 16].contiguous()
    sf_f = sf.view(torch.float8_e4m3fn).double()
    p = w_fp4.view(torch.uint8).reshape(n, k // 2)
    vals = torch.stack([LUT[(p & 0xF).long()], LUT[(p >> 4).long()]], dim=-1).reshape(n, k)
    wref = (vals.view(n, k // 16, 16) * sf_f.unsqueeze(-1)).reshape(n, k) * gsc.double()
    del vals, p, sf_f, sf, w_fp4, w_sf
    return packed, wref


def marlin(x, wt, n, k):
    return torch.ops.trtllm.marlin_nvfp4_gemm(x, wt[0], scale_a=None, scale_b=wt[1], alpha=None,
                                              weight_global_scale=wt[2], bias=None,
                                              out_dtype=torch.bfloat16, size_n=n, size_k=k)


def norm(x, r, w):
    torch.ops.trtllm.flashinfer_gemma_fused_add_rmsnorm(x, r, w, EPS)


def h16(y):
    return hashlib.sha256(y.contiguous().view(torch.int16).cpu().numpy().tobytes()).hexdigest()[:16]


def errs(y, yref):
    d = (y.double() - yref).abs()
    return float(d.max()), float(d.mean())


def make_x(m, k, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    x = (torch.randn(m, k, generator=g, device=dev) * 0.5).to(torch.bfloat16)
    x[:, 5::97] *= 8.0
    return x


failures = []
cases = dict()
hashes = dict()
SIDE = torch.cuda.Stream()
n_replays = 0
for sname, n, k in SHAPES:
    packed, wref = build(n, k, 11 + n + k)
    touched = sname != "outproj"
    ms = list(TOUCHED_M) + list(UNTOUCHED_M) if sname != "outproj" else list(TOUCHED_M) + [16, 64]
    for m in ms:
        x = make_x(m, k, 100 + m)
        set_env("0")
        yk = marlin(x, packed, n, k).clone()
        set_env(None)
        yd = [marlin(x, packed, n, k).clone() for _ in range(3)]
        torch.cuda.synchronize()
        key = sname + "_M" + str(m)
        c = dict()
        c["D_repeat_equal"] = all(torch.equal(y, yd[0]) for y in yd)
        c["finite"] = bool(torch.isfinite(yd[0].float()).all())
        c["D_eq_K"] = bool(torch.equal(yd[0], yk))
        c["D_eq_K_frac"] = float((yd[0].view(torch.int16) == yk.view(torch.int16)).float().mean())
        hashes[key] = dict(D=h16(yd[0]), K=h16(yk))
        is_touched = touched and m <= 8
        c["touched"] = is_touched
        ok = c["D_repeat_equal"] and c["finite"]
        if is_touched:
            yref = x.double() @ wref.t()
            ymax = float(yref.abs().max())
            ulp = 2.0 ** (int(torch.floor(torch.log2(torch.tensor(ymax)))) - 7)
            dmax, dmean = errs(yd[0], yref)
            kmax, kmean = errs(yk, yref)
            c["err_D_max"], c["err_D_mean"], c["err_K_max"], c["err_K_mean"], c["ulp"] = dmax, dmean, kmax, kmean, ulp
            c["rule_1p25"] = dmax <= 1.25 * kmax + ulp and dmean <= 1.25 * kmean + ulp
            c["rule_main_plus_ulp"] = dmax <= kmax + ulp and dmean <= kmean + ulp
            ok = ok and c["rule_1p25"] and c["rule_main_plus_ulp"]
            log("%-7s M=%4d touched: err max D %.4g K %.4g mean D %.4g K %.4g ulp %.3g | rules %s %s | D==K frac %.3f"
                " | repeat %s finite %s | hash D %s K %s"
                % (sname, m, dmax, kmax, dmean, kmean, ulp, c["rule_1p25"], c["rule_main_plus_ulp"],
                   c["D_eq_K_frac"], c["D_repeat_equal"], c["finite"], hashes[key]["D"], hashes[key]["K"]))
        else:
            ok = ok and c["D_eq_K"]
            log("%-7s M=%4d untouched: D==K bitwise %s | repeat %s finite %s | hash D %s K %s"
                % (sname, m, c["D_eq_K"], c["D_repeat_equal"], c["finite"], hashes[key]["D"], hashes[key]["K"]))
        c["ok"] = ok
        if not ok:
            failures.append(key)
        cases[key] = c
        del yd, yk, x
    # 3. repeated execution in graphs (touched shapes only)
    if touched:
        ws = [packed, tuple(t.clone() for t in packed)]
        g = torch.Generator(device=dev).manual_seed(5)
        wn1 = (0.1 * torch.randn(k, generator=g, device=dev)).to(torch.bfloat16)
        wn2 = (0.1 * torch.randn(n, generator=g, device=dev)).to(torch.bfloat16)
        for m in GRAPH_M:
            per = 2 * m * k
            st0 = (torch.randn(len(ws) * per, generator=g, device=dev) * 0.5).to(torch.bfloat16)
            st = st0.clone()
            xs = [st[i * per: i * per + m * k].view(m, k) for i in range(len(ws))]
            rs = [st[i * per + m * k: (i + 1) * per].view(m, k) for i in range(len(ws))]
            outs = []

            def fn():
                st.copy_(st0)
                raw, ys = [], []
                for i in range(len(ws)):
                    norm(xs[i], rs[i], wn1)
                    y = marlin(xs[i], ws[i], n, k)
                    raw.append(y.clone())
                    r2 = torch.zeros_like(y)
                    norm(y, r2, wn2)
                    ys.append(y)
                return raw, ys

            set_env(None)
            eraw, eys = fn()
            ref_state = torch.cat([st.clone()] + [y.reshape(-1).clone() for y in eraw + eys])
            xin = [xx.clone() for xx in xs]
            yrefs = [xin[i].double() @ wref.t() for i in range(len(ws))]
            torch.cuda.synchronize()
            SIDE.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(SIDE):
                for _ in range(3):
                    fn()
            torch.cuda.current_stream().wait_stream(SIDE)
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                graw, gys = fn()
            first = None
            worst = 0.0
            errs_it = []
            okg = True
            for it in range(WARM + ITERS):
                st.fill_(float("nan"))
                for t in graw + gys:
                    t.fill_(float("nan"))
                gr.replay()
                torch.cuda.synchronize()
                out = torch.cat([st.clone()] + [y.reshape(-1).clone() for y in graw + gys])
                if first is None:
                    first = out
                eq1 = bool(torch.equal(out, first))
                eqr = bool(torch.equal(out, ref_state))
                e = max(errs(graw[i], yrefs[i])[0] for i in range(len(ws)))
                errs_it.append(e)
                n_replays += 1
                if not (eq1 and eqr):
                    okg = False
            key = sname + "_graph_M" + str(m)
            cases[key] = dict(ok=okg, replays=WARM + ITERS, err_max_first=errs_it[0], err_max_last=errs_it[-1],
                              err_drift=max(errs_it) - min(errs_it))
            log("%-7s graph M=%d: %d NaN-poisoned replays == replay 1 and == eager: %s | per-replay max err "
                "first %.4g last %.4g drift %.3g" % (sname, m, WARM + ITERS, okg, errs_it[0], errs_it[-1],
                                                     max(errs_it) - min(errs_it)))
            if not okg:
                failures.append(key)
            del gr, graw, gys, eraw, eys
            torch.cuda.synchronize()
        del ws
    del packed, wref
    torch.cuda.empty_cache()

os.environ.pop(ENV, None)
passed = not failures
print(json.dumps(dict(passed=passed, failures=failures, graph_replays_checked=n_replays, cases=cases,
                      hashes=hashes, warmup=WARM, iters=ITERS,
                      compared="D (default) vs K (TRTLLM_MARLIN_INPROJ_FULLK=0, main's path): touched = FP64-ref "
                               "error rules; untouched = bitwise; graph replays bitwise vs replay 1 and eager")))
