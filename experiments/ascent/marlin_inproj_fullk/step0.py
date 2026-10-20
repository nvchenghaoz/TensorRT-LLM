# Idea 201 / branch 119 STEP 0: in-tree table for the full-K Marlin decode entry of
# GDN in_proj_qkvz (16384,5120) and attn qkv_proj (14336,5120), M = 1..8.
# Arms in ONE process (env TRTLLM_MARLIN_INPROJ_FULLK read per call, at capture):
#   A  = "0"  : main's 188-CTA split-K launch (kill switch)
#   F6 = unset: 128 threads, {64k,128n}, 6 stages, grid N/128 (the branch default)
#   W  = "w"  : main's 256-thread {128k,128n} 4-stage tile on the N/128 grid
#   AA = "0"  : a second graph of arm A (A/A bias floor)
# Cold L2: each graph chains C distinct weight copies (C x bytes >= 600 MB) and a 512 MiB
# READ flush (sum) runs before every replay outside the timed events. Arm order cycles
# through all 24 permutations. 10 warmup + 100 timed replays, PDL on.
# Also reports per arm: FP64-reference max/mean abs error (dequantized NVFP4 weight, same BF16
# activations), NaN/Inf, all-copies-equal, graph == eager, and an output hash.
# Brace-free so it can run inline via python -c.
import hashlib
import itertools
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
WARM, ITERS = 10, 100
K = 5120
SHAPES = (("qkvz", 16384, 13), ("qkv", 14336, 15))
MS = (1, 2, 3, 4, 5, 6, 7, 8)
ARMS = (("A", "0"), ("F6", None), ("W", "w"), ("AA", "0"))
ARM_NAMES = tuple(a[0] for a in ARMS)
PERMS = list(itertools.permutations(ARM_NAMES))
log("gpu", torch.cuda.get_device_name(dev), "PDL", os.environ.get("TRTLLM_ENABLE_PDL"))


def set_env(v):
    if v is None:
        os.environ.pop(ENV, None)
    else:
        os.environ[ENV] = v


LUT = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
                   dtype=torch.float64, device=dev)


def build(n, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    w = torch.randn(n, K, generator=g, device=dev) * 0.02
    w = w * (torch.rand(n, 1, generator=g, device=dev) * 1.5 + 0.25)
    w[:, 3::211] *= 8.0
    w = w.to(torch.bfloat16)
    amax = w.float().abs().amax()
    gsc = torch.exp2(torch.ceil(torch.log2(amax / (448.0 * 6.0))))
    w_fp4, w_sf = torch.ops.trtllm.fp4_quantize(w, (1.0 / gsc).reshape(1).float(), 16, False, True)
    del w
    packed = _pack_nvfp4_weight_for_marlin(w_fp4, w_sf, gsc.reshape(1).float(), n, K, 16)
    sf = torch.ops.trtllm.block_scale_interleave_reverse(w_sf.view(n, -1))[:, :K // 16].contiguous()
    sf_f = sf.view(torch.float8_e4m3fn).double()
    p = w_fp4.view(torch.uint8).reshape(n, K // 2)
    vals = torch.stack([LUT[(p & 0xF).long()], LUT[(p >> 4).long()]], dim=-1).reshape(n, K)
    wref = (vals.view(n, K // 16, 16) * sf_f.unsqueeze(-1)).reshape(n, K) * gsc.double()
    del vals, p, sf_f, sf, w_fp4, w_sf
    return packed, wref


def marlin(x, wt, n):
    return torch.ops.trtllm.marlin_nvfp4_gemm(x, wt[0], scale_a=None, scale_b=wt[1], alpha=None,
                                              weight_global_scale=wt[2], bias=None,
                                              out_dtype=torch.bfloat16, size_n=n, size_k=K)


SIDE = torch.cuda.Stream()


def capture(fn):
    SIDE.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(SIDE):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(SIDE)
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        out = fn()
    torch.cuda.synchronize()
    return gr, out


def stats(v):
    t = sorted(v)
    n = len(t)
    return dict(med=t[n // 2], p10=t[n // 10], p90=t[(9 * n) // 10])


def h16(y):
    return hashlib.sha256(y.contiguous().view(torch.int16).cpu().numpy().tobytes()).hexdigest()[:16]


flush = torch.ones(512 * 1024 * 1024 // 4, dtype=torch.int32, device=dev)
results = dict()
gx = torch.Generator(device=dev).manual_seed(7)
for sname, n, C in SHAPES:
    packed, wref = build(n, 11 + n)
    ws = [packed] + [tuple(t.clone() for t in packed) for _ in range(C - 1)]
    mb = sum(t.numel() * t.element_size() for wt in ws for t in wt) / 1e6
    log(sname, "N", n, "copies", C, "weight MB %.1f" % mb)
    rs = dict()
    for m in MS:
        x = (torch.randn(m, K, generator=gx, device=dev) * 0.5).to(torch.bfloat16)
        x[:, 5::97] *= 8.0
        yref = x.double() @ wref.t()
        ymax = float(yref.abs().max())
        ulp = 2.0 ** (int(torch.floor(torch.log2(torch.tensor(ymax)))) - 7)
        graphs, outs, eager = dict(), dict(), dict()
        for name, env in ARMS:
            set_env(env)
            eager[name] = marlin(x, ws[0], n).clone()
            gr, out = capture(lambda: [marlin(x, wt, n) for wt in ws])
            graphs[name] = gr
            outs[name] = out
        os.environ.pop(ENV, None)
        times = dict((a, []) for a in ARM_NAMES)
        for it in range(WARM + ITERS):
            for a in PERMS[it % len(PERMS)]:
                flush.sum()
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                graphs[a].replay()
                e1.record()
                if it >= WARM:
                    times[a].append((e0, e1))
        torch.cuda.synchronize()
        r = dict()
        for a in ARM_NAMES:
            st = stats([e0.elapsed_time(e1) * 1e3 / C for e0, e1 in times[a]])
            y = outs[a][0]
            d = (y.double() - yref).abs()
            st["max_err"] = float(d.max())
            st["mean_err"] = float(d.mean())
            st["finite"] = bool(torch.isfinite(y.float()).all())
            st["copies_equal"] = all(torch.equal(o, y) for o in outs[a])
            st["graph_eq_eager"] = bool(torch.equal(y, eager[a]))
            st["hash"] = h16(y)
            st["tbps"] = (n * K / 2 + n * K / 16 + m * K * 2 + m * n * 2) / (st["med"] * 1e-6) / 1e12
            r[a] = st
        a, f, w, aa = r["A"], r["F6"], r["W"], r["AA"]
        r["F6_vs_A_pct"] = 100.0 * (f["med"] - a["med"]) / a["med"]
        r["W_vs_A_pct"] = 100.0 * (w["med"] - a["med"]) / a["med"]
        r["AA_vs_A_pct"] = 100.0 * (aa["med"] - a["med"]) / a["med"]
        r["F6_p90_lt_A_p10"] = f["p90"] < a["p10"] and f["p90"] < aa["p10"]
        r["A_eq_AA"] = bool(torch.equal(outs["A"][0], outs["AA"][0]))
        r["F6_eq_A_frac"] = float((outs["F6"][0].view(torch.int16) == outs["A"][0].view(torch.int16)).float().mean())
        r["bf16_ulp_at_max"] = ulp
        r["F6_err_ok"] = (f["max_err"] <= a["max_err"] + ulp) and (f["mean_err"] <= a["mean_err"] + ulp)
        rs[str(m)] = r
        log("%s M=%d us/call: A %.2f [%.2f-%.2f] %.3fTB/s | F6 %.2f [%.2f-%.2f] %.3fTB/s %+.2f%% | W %.2f [%.2f-%.2f] "
            "%+.2f%% | AA %.2f %+.2f%% | F6.p90<A.p10 %s | err max A %.4g F6 %.4g W %.4g mean A %.4g F6 %.4g "
            "W %.4g ulp %.3g ok %s | A==AA %s F6==A frac %.3f | finite %s/%s/%s copies_eq %s/%s/%s g==e %s/%s/%s"
            % (sname, m, a["med"], a["p10"], a["p90"], a["tbps"], f["med"], f["p10"], f["p90"], f["tbps"],
               r["F6_vs_A_pct"], w["med"], w["p10"], w["p90"], r["W_vs_A_pct"], aa["med"], r["AA_vs_A_pct"],
               r["F6_p90_lt_A_p10"], a["max_err"], f["max_err"], w["max_err"], a["mean_err"], f["mean_err"],
               w["mean_err"], ulp, r["F6_err_ok"], r["A_eq_AA"], r["F6_eq_A_frac"], a["finite"], f["finite"],
               w["finite"], a["copies_equal"], f["copies_equal"], w["copies_equal"], a["graph_eq_eager"],
               f["graph_eq_eager"], w["graph_eq_eager"]))
        del graphs, outs, eager
        torch.cuda.synchronize()
    results[sname] = rs
    del ws, packed, wref
    torch.cuda.empty_cache()

q4 = results["qkvz"]["4"]
gate = dict()
gate["qkvz_M4_ge_3pct"] = q4["F6_vs_A_pct"] <= -3.0
gate["qkvz_M4_beyond_AA"] = abs(q4["F6_vs_A_pct"]) > abs(q4["AA_vs_A_pct"])
gate["qkvz_M4_p90_lt_p10"] = q4["F6_p90_lt_A_p10"]
gate["GO"] = all(gate.values())
log("STEP 0 GATE:", gate)
batch = dict()
for m in MS:
    d = results["qkvz"][str(m)]["F6"]
    batch[str(m)] = dict(kernel_ms=d["med"] / 1e3, noise_floor=(d["p90"] - d["p10"]) / d["med"])
print(json.dumps(dict(batch=batch, output=dict(cls="reordered"), gate=gate, results=results,
                      warm=WARM, iters=ITERS)))
