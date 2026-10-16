"""Idea 86 kernel_time bench (branch 39): prefill rows > 16 of attn qkv_proj
(14336,5120), attn o_proj (5120,6144) and GDN out_proj (5120,6144, gated norm
included), FP8 per-tensor (main) vs NVFP4 W4A4 prefill copies.

Runs unchanged in either source tree (pass the file's text with python -c):
the tree's own fp8_prefill_nvfp4.build_fp8_prefill_nvfp4_copies walks a
synthetic model (model.model.layers of real Attention / Qwen3NextGatedDeltaNet
instances holding FP8QDQ linears with main's decode Marlin copies) and decides
which modules get copies; GDN out_proj runs through the tree's own
Qwen3NextGatedDeltaNet._postprocess_gdn_output. Arm "tree" = the tree as
built (base: FP8 everywhere); arm "main_fp8" = same process with the copies
removed (= kill switches). Kernel time: CUDA events after a GPU sleep
pre-roll (CPU dispatch hidden), weights rotated over R modules (>= 256 MB).
Also: rows <= 16 bitwise (tree vs main_fp8 arm + hashes for cross-tree
comparison), M=17 path check, 10 warmup + 50 NaN-poisoned repeated calls per
shape bitwise vs each module's first call, FP32-reference errors.
"""
import hashlib
import json
import os
import statistics
import time

import torch
from torch import nn

from tensorrt_llm._torch.attention.attention import Attention
from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.modules import fp8_prefill_nvfp4 as fpn
from tensorrt_llm._torch.modules.linear import (FP8QDQLinearMethod,
                                                _build_fp8_nvfp4_decode_copy)
from tensorrt_llm._torch.modules.mamba.gdn_mixer import Qwen3NextGatedDeltaNet
from tensorrt_llm._torch.modules.mamba.layernorm_gated import \
    rms_norm_gated_token_major

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


dev = "cuda"
R = int(os.environ.get("BENCH_ROT", "16"))
MS = [int(v) for v in os.environ.get("BENCH_MS", "2048,4096,8192").split(",")]
SMALL = [1, 2, 4, 8, 16, 17]
WARM, ITERS = 10, 50
SLEEP = int(os.environ.get("BENCH_SLEEP_CYCLES", "3000000"))
COUNT = {"qkv": 16, "o": 16, "out": 48}
HV, NVH = 128, 48
VD = HV * NVH  # 6144
CONV = 10240  # in_proj_qkvz conv part: 2 * 16 * 128 + 6144
H = 5120
ATTR = "_fp8_prefill_nvfp4"
gen = torch.Generator(device=dev)
gen.manual_seed(1234)


def rnd(*shape, scale=1.0):
    return (torch.randn(*shape, device=dev, generator=gen) * scale).to(
        torch.bfloat16)


class Lin(nn.Module):

    def forward(self, x, all_reduce_params=None):
        return self.quant_method.apply(self, x, None)


def make_lin(N, K, in_scale):
    W = rnd(N, K, scale=0.02)
    ws = (W.float().abs().amax() / 448.0).reshape(())
    m = Lin()
    m.quant_method = FP8QDQLinearMethod()
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
    m.out_features, m.in_features = N, K
    m.bias = None
    # main's decode path: rows <= 16 on the W4A16 Marlin copy
    m._nvfp4_decode_copy = True
    _build_fp8_nvfp4_decode_copy(m)
    del W
    return m


def h(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy()
                          .tobytes()).hexdigest()[:16]


# ---------------- inputs ----------------
allM = sorted(set(MS + SMALL))
xq = {M: rnd(M, H, scale=1.5) for M in allM}
xo = {M: rnd(M, VD, scale=0.5) for M in allM}
ao = {M: rnd(M, NVH, HV, scale=0.3) for M in allM}
qkvz = {M: rnd(M, CONV + VD) for M in allM}
zz = {M: qkvz[M][:, CONV:].view(M, NVH, HV) for M in allM}
norm_w = nn.Parameter((1.0 + 0.1 * torch.randn(
    HV, device=dev, generator=gen)).to(torch.bfloat16),
                      requires_grad=False)
EPS = 1e-6
s_q = 1.1 * max(float(x.float().abs().amax()) for x in xq.values()) / 448.0
s_o = 1.1 * max(float(x.float().abs().amax()) for x in xo.values()) / 448.0
s_out = 1.1 * max(
    float(
        rms_norm_gated_token_major(ao[M].reshape(-1, HV), zz[M], norm_w,
                                   EPS).float().abs().amax())
    for M in allM) / 448.0
log(f"input scales qkv {s_q:.4g} o {s_o:.4g} out {s_out:.4g}")

# ---------------- synthetic model ----------------
root, inner = nn.Module(), nn.Module()
root.model = inner
layers, qkv_m, o_m, out_m, gdn_m = [], [], [], [], []
for i in range(R):
    L = nn.Module()
    a = Attention.__new__(Attention)
    nn.Module.__init__(a)
    a.attn_output_gate = True
    a.qkv_proj = make_lin(14336, H, s_q)
    a.o_proj = make_lin(H, VD, s_o)
    L.self_attn = a
    layers.append(L)
    qkv_m.append(a.qkv_proj)
    o_m.append(a.o_proj)
for i in range(R):
    L = nn.Module()
    g = Qwen3NextGatedDeltaNet.__new__(Qwen3NextGatedDeltaNet)
    nn.Module.__init__(g)
    g.out_proj = make_lin(H, VD, s_out)
    nm = nn.Module()
    nm.weight = norm_w
    nm.eps = EPS
    nm.fp8_scale = g.out_proj.input_scale.detach()
    g.norm = nm
    g.head_v_dim = HV
    g.value_dim_per_tp = VD
    L.linear_attn = g
    layers.append(L)
    gdn_m.append(g)
    out_m.append(g.out_proj)
inner.layers = nn.ModuleList(layers)
inner.num_hidden_layers = len(layers)
log(f"built {len(layers)} synthetic layers (R={R}); "
    f"mem {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
nbuilt = fpn.build_fp8_prefill_nvfp4_copies(root)
MODS = {"qkv": qkv_m, "o": o_m, "out": out_m}
copies = {
    g: sum(1 for m in ms if fpn.get_prefill_nvfp4_copy(m) is not None)
    for g, ms in MODS.items()
}
log(f"build_fp8_prefill_nvfp4_copies -> {nbuilt}; copies per group {copies}; "
    f"stats {fpn.prefill_nvfp4_stats()}")
saved = {id(m): m.__dict__.get(ATTR) for ms in MODS.values() for m in ms}


def set_arm(arm):
    for ms in MODS.values():
        for m in ms:
            c = saved[id(m)]
            if arm == "tree" and c is not None:
                m.__dict__[ATTR] = c
            else:
                m.__dict__.pop(ATTR, None)


def fn(grp, M):
    if grp == "qkv":
        return lambda i: qkv_m[i](xq[M])
    if grp == "o":
        return lambda i: o_m[i](xo[M])
    return lambda i: Qwen3NextGatedDeltaNet._postprocess_gdn_output(
        gdn_m[i], ao[M], zz[M])


def timeit(f):
    for i in range(WARM):
        f(i % R)
    torch.cuda.synchronize()
    evs = []
    for i in range(ITERS):
        torch.cuda._sleep(SLEEP)
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        f(i % R)
        e1.record()
        evs.append((e0, e1))
    torch.cuda.synchronize()
    ts = sorted(a.elapsed_time(b) for a, b in evs)
    med = statistics.median(ts)
    p10, p90 = ts[int(0.1 * len(ts))], ts[int(0.9 * len(ts)) - 1]
    return {"median_ms": med, "p90_ms": p90, "nf": (p90 - p10) / med}


def poison(M, N, K):
    # Freed 0xFF (NaN) blocks of the output / activation sizes: the next
    # allocations of those sizes reuse them.
    ts = [torch.full((n, ), 255, dtype=torch.uint8, device=dev)
          for n in (M * N * 2, M * K * 2, M * K, M * K // 2)]
    del ts


NK = {"qkv": (14336, H), "o": (H, VD), "out": (H, VD)}
ok = True
fails = []
res = {"small": {}, "large": {}, "repeat": {}, "err": {}, "diag": {}}

# ---------------- rows <= 16 bitwise + M=17 path ----------------
for grp in ("qkv", "o", "out"):
    for M in SMALL:
        f = fn(grp, M)
        set_arm("main_fp8")
        y_main = f(0).clone()
        set_arm("tree")
        c0 = fpn.prefill_nvfp4_stats()["calls"]
        y_tree = f(0).clone()
        dc = fpn.prefill_nvfp4_stats()["calls"] - c0
        eq = torch.equal(y_main, y_tree)
        r = {"tree_eq_main": eq, "nvfp4_calls": dc, "hash_tree": h(y_tree)}
        if M <= 16:
            if not eq or dc != 0:
                ok = False
                fails.append(f"small {grp} M={M}")
        elif copies[grp]:
            r["rel_rms_vs_main"] = float((y_tree.float() - y_main.float()).norm()
                                         / y_main.float().norm())
            if dc != 1:
                ok = False
                fails.append(f"M=17 {grp} nvfp4 path not taken")
        res["small"][f"{grp}/{M}"] = r
    torch.cuda.synchronize()
log("rows<=16:", {k: (v["tree_eq_main"], v["nvfp4_calls"])
                  for k, v in res["small"].items()})

# ---------------- autotune + timing + checks per M ----------------
arms = ["main_fp8"] + (["tree"] if any(copies.values()) else [])
timing = {a: {} for a in arms}
for M in MS:
    set_arm("tree")
    with autotune():
        for grp in ("qkv", "o", "out"):
            fn(grp, M)(0)
    torch.cuda.synchronize()
    for grp in ("qkv", "o", "out"):
        N, K = NK[grp]
        f = fn(grp, M)
        for arm in arms:
            set_arm(arm)
            t = timeit(f)
            timing[arm][f"{grp}/{M}"] = t
            # untimed repeated sequence: same warmup / iteration counts and
            # module rotation as timing, outputs NaN-poisoned, each call
            # bitwise equal to that module's first call
            first, nbad, nnan = {}, 0, 0
            for it in range(WARM + ITERS):
                k = it % R
                poison(M, N, K)
                y = f(k)
                if not bool(torch.isfinite(y).all()):
                    nnan += 1
                if k not in first:
                    first[k] = y.clone()
                elif not torch.equal(y, first[k]):
                    nbad += 1
            res["repeat"][f"{arm}/{grp}/{M}"] = {"mismatch": nbad,
                                                 "nonfinite": nnan}
            if nbad or nnan:
                ok = False
                fails.append(f"repeat {arm} {grp} M={M}")
            res["large"][f"{arm}/{grp}/{M}"] = h(first[0])
            # error vs FP32 reference (module 0)
            m0 = MODS[grp][0]
            Wdq = m0.weight.float() * m0.weight_scale.float()
            if grp == "qkv":
                xin = xq[M].float()
            elif grp == "o":
                xin = xo[M].float()
            else:
                xin = rms_norm_gated_token_major(ao[M].reshape(-1, HV), zz[M],
                                                 norm_w, EPS).view(
                                                     M, VD).float()
            ref = xin @ Wdq.t()
            d = first[0].float() - ref
            res["err"][f"{arm}/{grp}/{M}"] = {
                "rel_rms": float(d.norm() / ref.norm()),
                "max_abs": float(d.abs().max()),
                "mean_abs": float(d.abs().mean())
            }
            del ref, d, Wdq, xin, first
            log(f"M={M} {grp:3s} {arm:8s} median {t['median_ms']*1e3:8.1f} us "
                f"p90 {t['p90_ms']*1e3:8.1f} nf {t['nf']:.3f} "
                f"err {res['err'][f'{arm}/{grp}/{M}']['rel_rms']:.4f} "
                f"rep {res['repeat'][f'{arm}/{grp}/{M}']}")
    # diagnostics (note d): norm / quant / GEMM split
    s8 = out_m[0].input_scale
    a2 = ao[M].reshape(-1, HV)
    y16 = rms_norm_gated_token_major(a2, zz[M], norm_w, EPS).view(M, VD)
    d = {}
    d["out_norm_fp8"] = timeit(lambda i: rms_norm_gated_token_major(
        a2, zz[M], norm_w, EPS, fp8_scale=s8))
    d["out_norm_bf16"] = timeit(lambda i: rms_norm_gated_token_major(
        a2, zz[M], norm_w, EPS))
    q8 = rms_norm_gated_token_major(a2, zz[M], norm_w, EPS,
                                    fp8_scale=s8).view(M, VD)
    d["out_gemm_fp8"] = timeit(lambda i: torch.ops.trtllm.cublas_scaled_mm(
        q8, out_m[i].weight.t(), scale_a=out_m[i].input_scale,
        scale_b=out_m[i].weight_scale, bias=None, out_dtype=torch.bfloat16))
    d["qkv_quant_fp8"] = timeit(
        lambda i: torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
            xq[M], qkv_m[i].input_scale))
    for grp, xin in (("qkv", xq[M]), ("o", xo[M]), ("out", y16)):
        if not copies[grp]:
            continue
        c0 = fpn.get_prefill_nvfp4_copy(MODS[grp][0])
        d[f"{grp}_quant_fp4"] = timeit(
            lambda i: torch.ops.trtllm.fp4_quantize(xin, c0.act_global, 16,
                                                    False))
        af, asf = torch.ops.trtllm.fp4_quantize(xin, c0.act_global, 16, False)
        cs = [fpn.get_prefill_nvfp4_copy(m) for m in MODS[grp]]
        d[f"{grp}_gemm_fp4"] = timeit(lambda i: torch.ops.trtllm.nvfp4_gemm(
            af, cs[i].w_fp4, asf, cs[i].w_sf, cs[i].alpha, torch.bfloat16,
            allowed_backends="cutlass,cublaslt,cuda_core"))
    res["diag"][str(M)] = {k: round(v["median_ms"] * 1e3, 1)
                           for k, v in d.items()}
    log(f"M={M} diag (us): {res['diag'][str(M)]}")
    del q8, y16

# ---------------- summary ----------------
arm = arms[-1]
batch, per_group = {}, {}
for M in MS:
    tot = sum(COUNT[g] * timing[arm][f"{g}/{M}"]["median_ms"] for g in COUNT)
    tot_main = sum(COUNT[g] * timing["main_fp8"][f"{g}/{M}"]["median_ms"]
                   for g in COUNT)
    nf = max(timing[arm][f"{g}/{M}"]["nf"] for g in COUNT)
    batch[str(M)] = {"kernel_ms": tot, "noise_floor": nf,
                     "main_fp8_same_process_ms": tot_main}
    for g in COUNT:
        tm = timing["main_fp8"][f"{g}/{M}"]["median_ms"]
        tt = timing[arm][f"{g}/{M}"]
        per_group[f"{g}/{M}"] = {
            "main_us": round(tm * 1e3, 1),
            "tree_us": round(tt["median_ms"] * 1e3, 1),
            "tree_p90_us": round(tt["p90_ms"] * 1e3, 1),
            "speedup_pct": round(100.0 * (1.0 - tt["median_ms"] / tm), 1)
        }
picks = []
try:
    for k, v in AutoTuner.get().profiling_cache.cache.items():
        if "nvfp4" in str(k):
            picks.append(f"{str(k)[:110]} -> {str(v)[:60]}")
except Exception as e:  # noqa: BLE001
    picks.append(repr(e))
for p in picks:
    log("autotuner pick:", p)
log("per group:", per_group)
log("errors:", res["err"])
log("large-M output hashes:", res["large"])
log("small-M tree hashes:", {k: v["hash_tree"] for k, v in res["small"].items()})
out = {
    "batch": batch,
    "output": {
        "class": "precision" if any(copies.values()) else "exact",
        "copies": copies,
        "per_group": per_group,
        "err_rel_rms": {k: round(v["rel_rms"], 5)
                        for k, v in res["err"].items()},
        "diag_us": res["diag"],
        "small_hash": {k: v["hash_tree"] for k, v in res["small"].items()},
        "large_hash": res["large"],
        "repeat_ok": all(v["mismatch"] == 0 and v["nonfinite"] == 0
                         for v in res["repeat"].values()),
    },
    "arm": arm,
    "fails": fails,
    "passed": ok,
}
print(json.dumps(out), flush=True)
