# Idea 177 / branch 94: STEP 1 fork-window gate and paired kernel evidence for routing
# trtllm::nvfp4_w4a16_gemv to GDN in_proj_qkvz (16384x5120, rows <= 3) and attn qkv_proj
# (14336x5120, rows <= 2). Decode windows run through production routing: FP8QDQLinearMethod.apply
# on decode copies built by linear._build_fp8_nvfp4_decode_copy (Linear.load_weights +
# post_load_weights on an FP8 per-tensor weight).
# Graphs per arm and per M (CUDA graphs, PDL on):
#   qkvz : 48 GDN windows: gemma fused add+RMSNorm (PDL) -> ev0 -> in_proj_qkvz on the current
#          stream || aux: ev0.wait, in_proj_ba (maybe_small_n_gemv, 96x5120), ev1 -> ev1.wait ->
#          one non-PDL consumer z = y[:, :96] + ba (maybe_execute_in_parallel's order)
#   qkv  : 16 attention windows: norm -> qkv_proj -> consumer y[:, :96] + y[:, 96:192]
#   step : 64 windows in model order (3 GDN + 1 attention) x 16 = one decode step's qkvz + qkv
# Every window owns distinct weight copies (clones of one production-built module per shape), so
# every layer reads cold weights; a 256 MiB READ flush (sum) runs before every replay, outside the
# timed event pair. The x/r state reset (st.copy_(st0)) precedes the flush, also outside the events.
# Arms in ONE process; the order cycles through all 6 permutations, one per iteration:
#   tree   = the tree's default routing (env unset): GEMV at rows <= cap on the candidate,
#            Marlin on main
#   marlin = modules built with TRTLLM_NVFP4_DECODE_GEMV_INPROJ=0 (Marlin on both trees)
#   aa     = a second graph on the marlin arm's copies (A/A bias floor)
# kernel_ms = the tree arm's step graph time (one decode step's 48 qkvz + 16 qkv windows).
# Brace-free so run_gpu_pair's base arm can run it inline with python -c. Extra argv words are
# printed first.
import itertools
import json
import os
import subprocess
import sys
import time
import types

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
INPROJ_ENV = "TRTLLM_NVFP4_DECODE_GEMV_INPROJ"
os.environ.pop(INPROJ_ENV, None)
os.environ.pop("TRTLLM_NVFP4_DECODE_GEMV", None)
import torch  # noqa: E402

import tensorrt_llm  # noqa: E402,F401
import tensorrt_llm._torch.custom_ops  # noqa: E402,F401
from tensorrt_llm._torch.modules import linear as L  # noqa: E402
from tensorrt_llm._torch.modules.linear import Linear  # noqa: E402
from tensorrt_llm._torch.modules.small_n_gemv import maybe_small_n_gemv  # noqa: E402
from tensorrt_llm.models.modeling_utils import QuantConfig  # noqa: E402
from tensorrt_llm.quantization.mode import QuantAlgo  # noqa: E402

T0 = time.time()


def log(*a):
    print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


for word in sys.argv[1:]:
    print(word, flush=True)
try:
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          timeout=30).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                           timeout=30).stdout.strip().splitlines()
    print("TREE_COMMIT=" + (head or "unknown") + " dirty_files=" + str(len(dirty)), flush=True)
except Exception as exc:  # noqa: BLE001
    print("TREE_COMMIT=unknown " + repr(exc), flush=True)
log("linear.py:", L.__file__, "cwd:", os.getcwd(), "per-shape caps:",
    getattr(L, "_NVFP4_DECODE_GEMV_ROW_CAPS", None))

dev = torch.device("cuda")
EPS = 1e-6
K = 5120
NQKVZ, NQKV, NBA = 16384, 14336, 96
NGDN, NATT = 48, 16
WARM, ITERS = 10, 60
MS = (1, 2, 3, 4)
ARMS = ("tree", "marlin", "aa")
GRAPHS = ("qkvz", "qkv", "step")
log("gpu", torch.cuda.get_device_name(dev), "PDL", os.environ.get("TRTLLM_ENABLE_PDL"))


def make_lin(n, seed, inproj_env):
    if inproj_env is None:
        os.environ.pop(INPROJ_ENV, None)
    else:
        os.environ[INPROJ_ENV] = inproj_env
    gen = torch.Generator(device=dev).manual_seed(seed)
    w = torch.randn(n, K, generator=gen, device=dev) * 0.02
    w = w * (torch.rand(n, 1, generator=gen, device=dev) * 1.5 + 0.25)
    w[:, 3::211] *= 8.0
    s = (w.abs().amax() / 448.0).float()
    w8 = (w / s).to(torch.float8_e4m3fn)
    del w
    lin = Linear(K, n, bias=False, dtype=torch.bfloat16,
                 quant_config=QuantConfig(quant_algo=QuantAlgo.FP8)).to(dev)
    lin._nvfp4_decode_copy = True
    lin.load_weights([dict(weight=w8, weight_scale=s.reshape(()),
                           input_scale=torch.tensor(0.04, device=dev))])
    lin.post_load_weights()
    os.environ.pop(INPROJ_ENV, None)
    return lin


def clone_ns(lin):
    ns = types.SimpleNamespace()
    ns.out_features = lin.out_features
    ns.in_features = lin.in_features
    ns._nvfp4_decode_weight = lin._nvfp4_decode_weight.clone()
    ns._nvfp4_decode_scale = lin._nvfp4_decode_scale.clone()
    ns._nvfp4_decode_global_scale = lin._nvfp4_decode_global_scale.clone()
    gv = lin.__dict__.get("_nvfp4_decode_gemv")
    ns._nvfp4_decode_gemv = None if gv is None else (gv[0].clone(), gv[1].clone(), gv[2])
    if "_nvfp4_decode_gemv_max_rows" in lin.__dict__:
        ns._nvfp4_decode_gemv_max_rows = lin.__dict__["_nvfp4_decode_gemv_max_rows"]
    return ns


NS = dict(tree=dict(), marlin=dict())
COPY_INFO = dict()
METHOD = None
for name, n, count, seed in (("qkvz", NQKVZ, NGDN, 11), ("qkv", NQKV, NATT, 12)):
    for src, env in (("tree", None), ("marlin", "0")):
        lin = make_lin(n, seed, env)
        METHOD = lin.quant_method
        gv = lin.__dict__.get("_nvfp4_decode_gemv")
        COPY_INFO[name + "_" + src] = dict(
            method=type(lin.quant_method).__name__,
            marlin_copy=getattr(lin, "_nvfp4_decode_weight", None) is not None,
            gemv_copy=gv is not None,
            gemv_max_rows=lin.__dict__.get("_nvfp4_decode_gemv_max_rows", None))
        NS[src][name] = [clone_ns(lin) for _ in range(count)]
        del lin, gv
        torch.cuda.empty_cache()
        log("built", name, src, COPY_INFO[name + "_" + src])
log("GPU memory allocated GB %.2f" % (torch.cuda.memory_allocated() / 1e9))

g = torch.Generator(device=dev).manual_seed(5)
wba = [(torch.randn(NBA, K, generator=g, device=dev) * 0.02).to(torch.bfloat16)
       for _ in range(NGDN)]
wn = (0.1 * torch.randn(K, generator=g, device=dev)).to(torch.bfloat16)
flush = torch.ones(256 * 1024 * 1024 // 4, dtype=torch.int32, device=dev)

LAYERS = dict()
LAYERS["qkvz"] = [("g", i) for i in range(NGDN)]
LAYERS["qkv"] = [("a", i) for i in range(NATT)]
step_layers = []
for b in range(NATT):
    step_layers += [("g", 3 * b), ("g", 3 * b + 1), ("g", 3 * b + 2), ("a", b)]
LAYERS["step"] = step_layers

SIDE = torch.cuda.Stream()
AUX = torch.cuda.Stream()


def norm(x, r):
    torch.ops.trtllm.flashinfer_gemma_fused_add_rmsnorm(x, r, wn, EPS)


def in_proj_ba(x, w):
    y = maybe_small_n_gemv(x, w)
    return torch.nn.functional.linear(x, w) if y is None else y


def make_fn(arm, kind, xs, rs):
    src = "marlin" if arm == "aa" else arm
    qkvz_ns = NS[src]["qkvz"]
    qkv_ns = NS[src]["qkv"]
    layers = LAYERS[kind]
    ev0 = [torch.cuda.Event() for _ in layers]
    ev1 = [torch.cuda.Event() for _ in layers]

    def fn():
        zs = []
        for j, (t, idx) in enumerate(layers):
            x = xs[j]
            norm(x, rs[j])
            if t == "g":
                ev0[j].record()
                y = METHOD.apply(qkvz_ns[idx], x, None)
                with torch.cuda.stream(AUX):
                    ev0[j].wait()
                    ba = in_proj_ba(x, wba[idx])
                    ev1[j].record()
                ev1[j].wait()
                zs.append(y[:, :NBA] + ba)
            else:
                y = METHOD.apply(qkv_ns[idx], x, None)
                zs.append(y[:, :NBA] + y[:, NBA:2 * NBA])
        return zs

    return fn


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


def flush_read():
    return flush.sum()


PERMS = list(itertools.permutations(ARMS))
results = dict()
for m in MS:
    per = 2 * m * K
    st0 = (torch.randn(len(LAYERS["step"]) * per, generator=g, device=dev) * 0.5).to(torch.bfloat16)
    rm = dict()
    for kind in GRAPHS:
        nl = len(LAYERS[kind])
        graphs = dict()
        sts = dict()
        outs = dict()
        for arm in ARMS:
            st = st0.clone()
            xs = [st[i * per: i * per + m * K].view(m, K) for i in range(nl)]
            rs = [st[i * per + m * K: (i + 1) * per].view(m, K) for i in range(nl)]
            gr, out = capture(make_fn(arm, kind, xs, rs))
            graphs[arm] = gr
            sts[arm] = st
            outs[arm] = out
        # One replay per arm from the same state: marlin == aa bitwise; tree vs marlin share.
        for arm in ARMS:
            sts[arm].copy_(st0)
            graphs[arm].replay()
        torch.cuda.synchronize()
        aa_equal = all(torch.equal(a, b) for a, b in zip(outs["marlin"], outs["aa"]))
        eq = 0
        tot = 0
        maxdiff = 0.0
        finite = True
        for a, b in zip(outs["tree"], outs["marlin"]):
            eq += int((a.view(torch.int16) == b.view(torch.int16)).sum().item())
            tot += a.numel()
            maxdiff = max(maxdiff, (a.float() - b.float()).abs().max().item())
            finite &= bool(torch.isfinite(a.float()).all().item())
        times = dict((arm, []) for arm in ARMS)
        for it in range(WARM + ITERS):
            for arm in PERMS[it % len(PERMS)]:
                sts[arm].copy_(st0)
                flush_read()
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                graphs[arm].replay()
                e1.record()
                if it >= WARM:
                    times[arm].append((e0, e1))
        torch.cuda.synchronize()
        r = dict()
        for arm in ARMS:
            tot_us = [e0.elapsed_time(e1) * 1e3 for e0, e1 in times[arm]]
            r[arm] = stats(tot_us)
            r[arm + "_per_layer"] = stats([v / nl for v in tot_us])
        a, b, c = r["tree"], r["marlin"], r["aa"]
        r["tree_vs_marlin_pct"] = 100.0 * (a["med"] - b["med"]) / b["med"]
        r["aa_vs_marlin_pct"] = 100.0 * (c["med"] - b["med"]) / b["med"]
        r["marlin_aa_bitwise"] = aa_equal
        r["tree_marlin_bitwise_frac"] = eq / max(tot, 1)
        r["tree_marlin_max_abs_diff"] = maxdiff
        r["tree_finite"] = finite
        r["layers"] = nl
        rm[kind] = r
        pa, pb, pc = r["tree_per_layer"], r["marlin_per_layer"], r["aa_per_layer"]
        log("M=%d %-4s per-layer us: tree %.2f [%.2f-%.2f] | marlin %.2f [%.2f-%.2f] | aa %.2f "
            "[%.2f-%.2f] | tree-marlin %+.2f%% | aa-marlin %+.2f%% | step total tree %.1f marlin %.1f us"
            " | bitwise tree==marlin %.3f maxdiff %.4g | marlin==aa %s"
            % (m, kind, pa["med"], pa["p10"], pa["p90"], pb["med"], pb["p10"], pb["p90"], pc["med"],
               pc["p10"], pc["p90"], r["tree_vs_marlin_pct"], r["aa_vs_marlin_pct"], a["med"], b["med"],
               r["tree_marlin_bitwise_frac"], maxdiff, aa_equal))
        del graphs, sts, outs
        torch.cuda.synchronize()
    results[str(m)] = rm


def faster(r):
    a, b, c = r["tree_per_layer"], r["marlin_per_layer"], r["aa_per_layer"]
    gap = abs(a["med"] - b["med"])
    aagap = abs(c["med"] - b["med"])
    return a["p90"] < b["p10"] and a["p90"] < c["p10"] and a["med"] < b["med"] and gap > aagap


def same(r):
    a, b = r["tree_per_layer"], r["marlin_per_layer"]
    return b["p10"] <= a["med"] <= b["p90"] or a["p10"] <= b["med"] <= a["p90"]


# STEP 1 gate (fixed before measuring; meaningful on the candidate tree).
gate = dict()
q = dict((m, results[str(m)]["qkvz"]) for m in MS)
gate["qkvz_M1_faster"] = faster(q[1])
gate["qkvz_M2_faster"] = faster(q[2])
gate["qkvz_M3_not_slower"] = q[3]["tree_per_layer"]["med"] <= q[3]["marlin_per_layer"]["med"]
gate["qkvz_M4_identical"] = same(q[4])
gate["GO"] = all(gate.values())
info = dict()
a2 = dict((m, results[str(m)]["qkv"]) for m in MS)
info["qkv_M1_faster"] = faster(a2[1])
info["qkv_M2_faster"] = faster(a2[2])
info["qkv_M3_identical"] = same(a2[3])
info["qkv_M4_identical"] = same(a2[4])
log("GATE (STEP 1, qkvz fork window):", gate)
log("qkv window:", info)

batch = dict()
for m in MS:
    d = results[str(m)]["step"]["tree"]
    batch[str(m)] = dict(kernel_ms=d["med"] / 1e3, noise_floor=(d["p90"] - d["p10"]) / d["med"])
output = dict()
output["class"] = "reordered"
print(json.dumps(dict(batch=batch, output=output, results=results, gate=gate, qkv_info=info,
                      copies=COPY_INFO, warm=WARM, iters=ITERS,
                      note="kernel_ms = tree-default arm, one decode step's 48 qkvz fork windows + "
                           "16 qkv windows (norm -> proj -> consumer), cold weights, READ flush")))
