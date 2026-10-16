"""Idea 141 (branch 68) unit check + per-call kernel time: gated-attention
qkv_proj (14336,5120) prefill rows > 16.

Arms, same process, same GPU, weights rotated over R synthetic modules
(>= 256 MB FP8 weights, cold L2):
  main_fp8  copies removed = main's FP8 per-tensor path (static E4M3 quant +
            cublas_scaled_mm), identical code to the kill switch;
  tree      the tree as built by its own build_fp8_prefill_nvfp4_copies
            (base: no qkv copy -> FP8; branch 68: Q+gate W4A4 + K/V FP8);
  whole     (branch tree only) the same copy with split=None = branch 56's
            whole-qkv W4A4 output.
Checks (branch tree): rows 1..16 bitwise == main_fp8 with 0 W4A4 calls;
M=17 takes the QG path; Q+gate columns bitwise == whole; K/V columns vs
main_fp8 (bitwise flag + max-abs) and both vs the FP32 reference; 10 warmup
+ 50 NaN-poisoned repeated calls per arm bitwise == each module's first call;
kill switch (ATTR_QG=0 rebuild) builds no copy -> output == main_fp8.
Runs unchanged in either tree (pass the text with python -c). Last stdout
line: pair JSON ("batch" keys 1/2/4 = c1/c2/c4 prefill M 2048/4096/8192).
"""
import hashlib
import json
import os
import statistics
import subprocess
import time

import torch
from torch import nn

os.environ["TRTLLM_FP8_PREFILL_NVFP4_ATTN_O"] = "0"  # only qkv_proj here
os.environ["TRTLLM_FP8_PREFILL_NVFP4_ATTN_QKV"] = "0"

from tensorrt_llm._torch.attention.attention import Attention  # noqa: E402
from tensorrt_llm._torch.autotuner import autotune  # noqa: E402
from tensorrt_llm._torch.models.modeling_utils import \
    DecoderModel  # noqa: E402
from tensorrt_llm._torch.modules import fp8_prefill_nvfp4 as fpn  # noqa: E402
from tensorrt_llm._torch.modules.linear import (  # noqa: E402
    FP8QDQLinearMethod, _build_fp8_nvfp4_decode_copy)

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


try:
    tc = subprocess.run(["git", "rev-parse", "HEAD"],
                        cwd=os.path.dirname(fpn.__file__),
                        capture_output=True, text=True,
                        timeout=20).stdout.strip()
except Exception as e:  # noqa: BLE001
    tc = repr(e)
QG_ENV = getattr(fpn, "ENV_ATTN_QG", None)
log(f"TREE_COMMIT {tc} fpn={fpn.__file__} has_QG={QG_ENV is not None} "
    f"QG_env={os.environ.get('TRTLLM_FP8_PREFILL_NVFP4_ATTN_QG')}")

dev = "cuda"
R = int(os.environ.get("BENCH_ROT", "8"))
MS = [int(v) for v in os.environ.get("BENCH_MS", "2048,4096,8192").split(",")]
MKEY = {2048: "1", 4096: "2", 8192: "4"}
SMALL = [1, 2, 4, 8, 16, 17]
WARM, ITERS = 10, 50
SLEEP = int(os.environ.get("BENCH_SLEEP_CYCLES", "3000000"))
H, N, Q_SIZE, KV_SIZE = 5120, 14336, 6144, 1024
SPLIT = 2 * Q_SIZE
ATTR = "_fp8_prefill_nvfp4"
gen = torch.Generator(device=dev)
gen.manual_seed(1234)


def rnd(*shape, scale=1.0):
    return (torch.randn(*shape, device=dev, generator=gen) * scale).to(
        torch.bfloat16)


class Lin(nn.Module):

    def forward(self, x, all_reduce_params=None):
        return self.quant_method.apply(self, x, None)


def make_lin(n, k, in_scale):
    W = rnd(n, k, scale=0.02)
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
    m.out_features, m.in_features = n, k
    m.bias = None
    m._nvfp4_decode_copy = True  # main's decode Marlin W4A16 copy
    _build_fp8_nvfp4_decode_copy(m)
    del W
    return m


def h(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy()
                          .tobytes()).hexdigest()[:16]


allM = sorted(set(MS + SMALL))
xq = {M: rnd(M, H, scale=1.5) for M in allM}
s_q = 1.1 * max(float(x.float().abs().amax()) for x in xq.values()) / 448.0
log(f"input scale {s_q:.4g}")

root, llm = nn.Module(), nn.Module()
inner = DecoderModel.__new__(DecoderModel)
nn.Module.__init__(inner)
root.llm = llm
llm.model = inner
layers, mods = [], []
for i in range(R):
    L = nn.Module()
    a = Attention.__new__(Attention)
    nn.Module.__init__(a)
    a.attn_output_gate = True
    a.q_size, a.kv_size = Q_SIZE, KV_SIZE
    a.qkv_proj = make_lin(N, H, s_q)
    a.o_proj = nn.Identity()
    L.self_attn = a
    layers.append(L)
    mods.append(a.qkv_proj)
inner.layers = nn.ModuleList(layers)
inner.num_hidden_layers = len(layers)
log(f"built {R} synthetic attn layers; mem "
    f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB")
nbuilt = fpn.build_fp8_prefill_nvfp4_copies(root)
saved = [m.__dict__.get(ATTR) for m in mods]
ncopy = sum(c is not None for c in saved)
splits = sorted({getattr(c, "split", None) for c in saved if c is not None},
                key=str)
log(f"copies {ncopy}/{R} splits {splits} stats {fpn.prefill_nvfp4_stats()}")
has_qg = ncopy > 0 and splits == [SPLIT]

ok = True
fails = []


def fail(msg):
    global ok
    ok = False
    fails.append(msg)
    log("FAIL", msg)


def set_arm(arm):
    for m, c in zip(mods, saved):
        if arm in ("tree", "whole") and c is not None:
            m.__dict__[ATTR] = c
            c.split = SPLIT if arm == "tree" else None
        else:
            m.__dict__.pop(ATTR, None)


def calls():
    return fpn.prefill_nvfp4_stats()["calls"]


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


def poison(M):
    ts = [torch.full((n, ), 255, dtype=torch.uint8, device=dev)
          for n in (M * N * 2, M * 2048 * 2, M * H * 2, M * H, M * H // 2)]
    del ts


res = {"small": {}, "unit": {}, "repeat": {}, "err": {}, "diag": {},
       "hash_main": {}, "hash_tree": {}}
arms = ["main_fp8", "tree"] + (["whole"] if has_qg else [])

# ---------------- rows <= 16 bitwise + M=17 path ----------------
for M in SMALL:
    set_arm("main_fp8")
    y_main = mods[0](xq[M]).clone()
    set_arm("tree")
    c0 = calls()
    y_tree = mods[0](xq[M]).clone()
    dc = calls() - c0
    eq = torch.equal(y_main, y_tree)
    r = {"eq_main": eq, "w4a4_calls": dc, "hash_tree": h(y_tree)}
    if M <= 16:
        if not eq or dc != 0:
            fail(f"small M={M} eq={eq} calls={dc}")
    elif has_qg:
        r["kv_eq_main"] = torch.equal(y_main[:, SPLIT:], y_tree[:, SPLIT:])
        r["kv_maxabs"] = float((y_main[:, SPLIT:].float() -
                                y_tree[:, SPLIT:].float()).abs().max())
        if dc != 1:
            fail("M=17 QG path not taken")
    res["small"][str(M)] = r
torch.cuda.synchronize()
log("rows<=17:", res["small"])

# ---------------- per M: autotune, unit checks, timing ----------------
timing = {a: {} for a in arms}
for M in MS:
    set_arm("tree")
    with autotune():
        mods[0](xq[M])
    torch.cuda.synchronize()
    x = xq[M]
    f = lambda i: mods[i](x)  # noqa: E731
    outs = {}
    for arm in arms:
        set_arm(arm)
        timing[arm][M] = timeit(f)
        # untimed repeated sequence (same warmup/iters/rotation), outputs
        # NaN-poisoned, each call bitwise == that module's first call
        first, nbad, nnan, maxd = {}, 0, 0, 0.0
        for it in range(WARM + ITERS):
            k = it % R
            poison(M)
            y = f(k)
            if not bool(torch.isfinite(y).all()):
                nnan += 1
            if k not in first:
                first[k] = y.clone()
            elif not torch.equal(y, first[k]):
                nbad += 1
                maxd = max(maxd, float((y.float() -
                                        first[k].float()).abs().max()))
        res["repeat"][f"{arm}/{M}"] = {"mismatch": nbad, "nonfinite": nnan,
                                       "max_drift": maxd}
        if nbad or nnan:
            fail(f"repeat {arm} M={M} {res['repeat'][f'{arm}/{M}']}")
        outs[arm] = first[0]
        t = timing[arm][M]
        log(f"M={M} {arm:8s} median {t['median_ms'] * 1e3:7.1f} us p90 "
            f"{t['p90_ms'] * 1e3:7.1f} nf {t['nf']:.3f} "
            f"rep {res['repeat'][f'{arm}/{M}']}")
    res["hash_main"][str(M)] = h(outs["main_fp8"])
    res["hash_tree"][str(M)] = h(outs["tree"])
    # FP32 reference (module 0)
    m0 = mods[0]
    ref = x.float() @ (m0.weight.float() * m0.weight_scale.float()).t()
    err = {}
    for arm in arms:
        d = outs[arm].float() - ref
        err[arm] = {
            "qg_max_abs": float(d[:, :SPLIT].abs().max()),
            "qg_mean_abs": float(d[:, :SPLIT].abs().mean()),
            "kv_max_abs": float(d[:, SPLIT:].abs().max()),
            "kv_mean_abs": float(d[:, SPLIT:].abs().mean()),
            "rel_rms": float(d.norm() / ref.norm()),
        }
    res["err"][str(M)] = err
    del ref, d
    if has_qg:
        u = {}
        u["qg_eq_whole"] = torch.equal(outs["tree"][:, :SPLIT],
                                       outs["whole"][:, :SPLIT])
        u["kv_eq_main"] = torch.equal(outs["tree"][:, SPLIT:],
                                      outs["main_fp8"][:, SPLIT:])
        u["kv_maxabs_vs_main"] = float(
            (outs["tree"][:, SPLIT:].float() -
             outs["main_fp8"][:, SPLIT:].float()).abs().max())
        u["kv_maxabs_main_ref"] = err["main_fp8"]["kv_max_abs"]
        u["kv_maxabs_tree_ref"] = err["tree"]["kv_max_abs"]
        u["kv_mean_main_ref"] = err["main_fp8"]["kv_mean_abs"]
        u["kv_mean_tree_ref"] = err["tree"]["kv_mean_abs"]
        res["unit"][str(M)] = u
        if not u["qg_eq_whole"]:
            fail(f"M={M} Q+gate columns != whole-qkv W4A4")
        # K/V at FP8 precision: within 1.25x main's error vs FP32 ref
        if (u["kv_maxabs_tree_ref"] > 1.25 * u["kv_maxabs_main_ref"] + 1e-2
                or u["kv_mean_tree_ref"] > 1.25 * u["kv_mean_main_ref"]):
            fail(f"M={M} K/V columns not at FP8 precision {u}")
        log(f"M={M} unit {u}")
    log(f"M={M} err vs FP32 ref {err}")
    # component diagnostics (module rotation)
    if has_qg:
        c = saved
        dg = {}
        dg["fp8_quant"] = timeit(
            lambda i: torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
                x, mods[i].input_scale))
        q8, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
            x, mods[0].input_scale)
        dg["kv_gemm_fp8"] = timeit(lambda i: torch.ops.trtllm.cublas_scaled_mm(
            q8, mods[i].weight[SPLIT:].t(), scale_a=mods[i].input_scale,
            scale_b=mods[i].weight_scale, bias=None,
            out_dtype=torch.bfloat16))
        dg["full_gemm_fp8"] = timeit(
            lambda i: torch.ops.trtllm.cublas_scaled_mm(
                q8, mods[i].weight.t(), scale_a=mods[i].input_scale,
                scale_b=mods[i].weight_scale, bias=None,
                out_dtype=torch.bfloat16))
        dg["fp4_quant"] = timeit(lambda i: torch.ops.trtllm.fp4_quantize(
            x, c[i].act_global, 16, False))
        af, asf = torch.ops.trtllm.fp4_quantize(x, c[0].act_global, 16, False)
        dg["w4a4_gemm"] = timeit(lambda i: torch.ops.trtllm.nvfp4_gemm(
            af, c[i].w_fp4, asf, c[i].w_sf, c[i].alpha, torch.bfloat16,
            allowed_backends="cutlass,cublaslt,cuda_core"))
        ybuf = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
        kvb = torch.empty(M, N - SPLIT, dtype=torch.bfloat16, device=dev)
        dg["slice_copy"] = timeit(lambda i: ybuf[:, SPLIT:].copy_(kvb))
        res["diag"][str(M)] = {k: round(v["median_ms"] * 1e3, 1)
                               for k, v in dg.items()}
        log(f"M={M} diag (us) {res['diag'][str(M)]}")
        del q8, af, asf, ybuf, kvb
    del outs
    torch.cuda.empty_cache()

# ---------------- kill switch: QG=0 rebuild builds no copy ----------------
if has_qg:
    set_arm("main_fp8")
    os.environ[QG_ENV] = "0"
    nb = fpn.build_fp8_prefill_nvfp4_copies(root)
    nks = sum(m.__dict__.get(ATTR) is not None for m in mods)
    M = MS[0]
    y = mods[0](xq[M])
    ks = {"built": nb, "copies": nks, "hash": h(y),
          "eq_main_hash": h(y) == res["hash_main"][str(M)]}
    res["kill_switch"] = ks
    if nb or nks or not ks["eq_main_hash"]:
        fail(f"kill switch {ks}")
    log("kill switch QG=0:", ks)
    os.environ.pop(QG_ENV)
    set_arm("tree")

# ---------------- summary ----------------
batch, per_m = {}, {}
for M in MS:
    tt, tm = timing["tree"][M], timing["main_fp8"][M]
    per_m[str(M)] = {
        "main_fp8_us": round(tm["median_ms"] * 1e3, 1),
        "tree_us": round(tt["median_ms"] * 1e3, 1),
        "tree_p90_us": round(tt["p90_ms"] * 1e3, 1),
        "saved_us": round((tm["median_ms"] - tt["median_ms"]) * 1e3, 1),
        "speedup_pct": round(100.0 * (1.0 - tt["median_ms"] /
                                      tm["median_ms"]), 1),
    }
    if "whole" in timing:
        per_m[str(M)]["whole_w4a4_us"] = round(
            timing["whole"][M]["median_ms"] * 1e3, 1)
    if M in MKEY:
        batch[MKEY[M]] = {"kernel_ms": tt["median_ms"], "noise_floor": tt["nf"],
                          "M": M,
                          "main_fp8_same_process_ms": tm["median_ms"]}
if has_qg and MS[0] == 2048 and per_m["2048"]["saved_us"] < 100.0:
    log("NOTE: saved < 100 us/call at M=2048")
log("per M:", per_m)
log("repeat:", res["repeat"])
log("hash main_fp8:", res["hash_main"], "tree:", res["hash_tree"])
out = {
    "batch": batch,
    "output": {
        "class": "precision" if has_qg else "exact",
        "per_M": per_m,
        "unit": res["unit"],
        "err_vs_fp32_ref": res["err"],
        "diag_us": res["diag"],
        "small": res["small"],
        "kill_switch": res.get("kill_switch"),
        "hash_main": res["hash_main"],
        "hash_tree": res["hash_tree"],
        "repeat_ok": all(v["mismatch"] == 0 and v["nonfinite"] == 0
                         for v in res["repeat"].values()),
    },
    "tree_commit": tc,
    "has_qg": has_qg,
    "fails": fails,
    "passed": ok,
}
print(json.dumps(out), flush=True)
