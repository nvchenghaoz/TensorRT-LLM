# Idea 201 / branch 119: fork-window pair for the full-K Marlin decode entry of GDN in_proj_qkvz
# (16384x5120) and attn qkv_proj (14336x5120). Production routing: FP8QDQLinearMethod.apply on
# decode copies built by Linear.load_weights + post_load_weights (GEMV at rows <= 3 / 2, Marlin
# above). Graphs per arm and per M (CUDA graphs, PDL on):
#   qkvz : 48 GDN windows: gemma fused add+RMSNorm (PDL) -> ev0 -> in_proj_qkvz on the current
#          stream || aux: ev0.wait, in_proj_ba (maybe_small_n_gemv, 96x5120), ev1 -> ev1.wait ->
#          one non-PDL consumer z = y[:, :96] + ba (maybe_execute_in_parallel's order)
#   qkv  : 16 attention windows: norm -> qkv_proj -> consumer y[:, :96] + y[:, 96:192]
#   step : 64 windows in model order (3 GDN + 1 attention) x 16 = one decode step's qkvz + qkv
# Every window owns distinct weight copies; a 256 MiB READ flush (sum) runs before every replay,
# outside the timed event pair, after the x/r state reset (st.copy_(st0)).
# Arms in ONE process (TRTLLM_MARLIN_INPROJ_FULLK is read per call, i.e. at capture); the order
# cycles through all 6 permutations, one per iteration:
#   tree = env unset (the tree's default: full-K entry on the candidate, main's launch on main)
#   kill = env "0" (main's 188-CTA split-K launch on both trees)
#   aa   = a second graph of the kill arm (A/A bias floor)
# Fork timing (area-lead request): one profiled replay set per arm and M of the qkvz graph
# (torch.profiler, CUPTI kernel records of the uninstrumented graph, 3 replays = 144 windows):
# per window qkvz end, in_proj_ba end, consumer start -> ba_end - qkvz_end and
# join-to-consumer = consumer start - max(qkvz end, ba end); medians.
# kernel_ms = the tree arm's step graph time. Brace-free (runs inline via python -c).
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
import types

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
ENV = "TRTLLM_MARLIN_INPROJ_FULLK"
os.environ.pop(ENV, None)
os.environ.pop("TRTLLM_NVFP4_DECODE_GEMV_INPROJ", None)
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
    tree = subprocess.run(["git", "rev-parse", "HEAD^" + chr(123) + "tree" + chr(125)],
                          capture_output=True, text=True, timeout=30).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                           timeout=30).stdout.strip().splitlines()
    print("TREE_COMMIT=" + (head or "unknown") + " TREE=" + tree + " dirty_files=" + str(len(dirty)),
          flush=True)
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
MS = (1, 2, 3, 4, 8)
ARMS = ("tree", "kill", "aa")
ARM_ENV = dict(tree=None, kill="0", aa="0")
GRAPHS = ("qkvz", "qkv", "step")
PROF_REPLAYS = 3
log("gpu", torch.cuda.get_device_name(dev), "PDL", os.environ.get("TRTLLM_ENABLE_PDL"))


def set_env(v):
    if v is None:
        os.environ.pop(ENV, None)
    else:
        os.environ[ENV] = v


def make_lin(n, seed):
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


NS = dict()
COPY_INFO = dict()
METHOD = None
for name, n, count, seed in (("qkvz", NQKVZ, NGDN, 11), ("qkv", NQKV, NATT, 12)):
    lin = make_lin(n, seed)
    METHOD = lin.quant_method
    gv = lin.__dict__.get("_nvfp4_decode_gemv")
    COPY_INFO[name] = dict(method=type(lin.quant_method).__name__,
                           marlin_copy=getattr(lin, "_nvfp4_decode_weight", None) is not None,
                           gemv_copy=gv is not None,
                           gemv_max_rows=lin.__dict__.get("_nvfp4_decode_gemv_max_rows", None))
    NS[name] = [clone_ns(lin) for _ in range(count)]
    del lin, gv
    torch.cuda.empty_cache()
    log("built", name, COPY_INFO[name])
log("GPU memory allocated GB %.2f" % (torch.cuda.memory_allocated() / 1e9))

g = torch.Generator(device=dev).manual_seed(5)
wba = [(torch.randn(NBA, K, generator=g, device=dev) * 0.02).to(torch.bfloat16) for _ in range(NGDN)]
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


def make_fn(kind, xs, rs):
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
                y = METHOD.apply(NS["qkvz"][idx], x, None)
                with torch.cuda.stream(AUX):
                    ev0[j].wait()
                    ba = in_proj_ba(x, wba[idx])
                    ev1[j].record()
                ev1[j].wait()
                zs.append(y[:, :NBA] + ba)
            else:
                y = METHOD.apply(NS["qkv"][idx], x, None)
                zs.append(y[:, :NBA] + y[:, NBA:2 * NBA])
        return zs

    return fn


def capture(fn, env):
    set_env(env)
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
    os.environ.pop(ENV, None)
    return gr, out


def stats(v):
    t = sorted(v)
    n = len(t)
    return dict(med=t[n // 2], p10=t[n // 10], p90=t[(9 * n) // 10])


def med(v):
    t = sorted(v)
    return t[len(t) // 2] if t else None


def h16(ts):
    hh = hashlib.sha256()
    for t in ts:
        hh.update(t.contiguous().view(torch.int16).cpu().numpy().tobytes())
    return hh.hexdigest()[:16]


def fork_profile(gr, st, st0, tag):
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(PROF_REPLAYS):
            st.copy_(st0)
            flush.sum()
            gr.replay()
        torch.cuda.synchronize()
    path = os.path.join(os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp"), "fork_" + tag + ".json")
    prof.export_chrome_trace(path)
    with open(path) as fh:
        tr = json.load(fh)
    evs = tr.get("traceEvents", []) if isinstance(tr, dict) else tr
    ks = []
    for e in evs:
        if e.get("cat") != "kernel":
            continue
        ks.append((float(e["ts"]), float(e.get("dur", 0.0)), e.get("name", ""),
                   e.get("args", dict()).get("grid")))
    ks.sort()
    # qkvz = Marlin / nvfp4 W4A16 GEMV; consumer = the elementwise add; norm and the flush
    # reduce are skipped; anything else after the first qkvz is in_proj_ba (small_n_gemv, or
    # the cuBLAS fallback at M > 6).
    q, ba, cons = [], [], []
    qgrid = set()
    ba_names = set()
    started = False
    for ts, dur, name, grid in ks:
        if "Marlin" in name or "nvfp4W4a16Gemv" in name:
            q.append((ts, ts + dur))
            qgrid.add(str(grid))
            started = True
        elif "Norm" in name or "norm" in name or "reduce" in name:
            continue
        elif "elementwise" in name:
            if started:
                cons.append((ts, ts + dur))
        elif started:
            ba.append((ts, ts + dur))
            ba_names.add(name.split("(")[0][-50:])
    res = dict(n_q=len(q), n_ba=len(ba), n_cons=len(cons), qkvz_grids=sorted(qgrid), ba_kernels=sorted(ba_names))
    nw = NGDN * PROF_REPLAYS
    if len(q) == nw and len(ba) == nw and len(cons) >= nw:
        # per window: in_proj_ba = first aux kernel starting after this qkvz starts, consumer =
        # first elementwise kernel starting after this qkvz starts
        bsel, csel = [], []
        jb, jc = 0, 0
        for a in q:
            while jb < len(ba) - 1 and ba[jb][0] < a[0] - 5.0:
                jb += 1
            while jc < len(cons) - 1 and cons[jc][0] < a[0]:
                jc += 1
            bsel.append(ba[jb])
            csel.append(cons[jc])
        ba, cons = bsel, csel
        d_ba_q = [b[1] - a[1] for a, b in zip(q, ba)]
        join = [c[0] - max(a[1], b[1]) for a, b, c in zip(q, ba, cons)]
        win = [c[1] - a[0] for a, c in zip(q, cons)]
        res.update(ba_end_minus_qkvz_end_us=med(d_ba_q), join_to_consumer_us=med(join),
                   qkvz_dur_us=med([a[1] - a[0] for a in q]), ba_dur_us=med([b[1] - b[0] for b in ba]),
                   ba_start_minus_qkvz_start_us=med([b[0] - a[0] for a, b in zip(q, ba)]),
                   frac_ba_inside_qkvz=sum(1 for v in d_ba_q if v <= 0) / float(nw),
                   qkvz_start_to_consumer_end_us=med(win))
    else:
        res["mismatch"] = True
    try:
        os.remove(path)
    except OSError:
        pass
    return res


PERMS = list(itertools.permutations(ARMS))
results = dict()
forks = dict()
hashes = dict()
for m in MS:
    per = 2 * m * K
    st0 = (torch.randn(len(LAYERS["step"]) * per, generator=g, device=dev) * 0.5).to(torch.bfloat16)
    rm = dict()
    for kind in GRAPHS:
        nl = len(LAYERS[kind])
        graphs, sts, outs = dict(), dict(), dict()
        for arm in ARMS:
            st = st0.clone()
            xs = [st[i * per: i * per + m * K].view(m, K) for i in range(nl)]
            rs = [st[i * per + m * K: (i + 1) * per].view(m, K) for i in range(nl)]
            gr, out = capture(make_fn(kind, xs, rs), ARM_ENV[arm])
            graphs[arm], sts[arm], outs[arm] = gr, st, out
        for arm in ARMS:
            sts[arm].copy_(st0)
            graphs[arm].replay()
        torch.cuda.synchronize()
        kill_aa_equal = all(torch.equal(a, b) for a, b in zip(outs["kill"], outs["aa"]))
        eq, tot, finite = 0, 0, True
        for a, b in zip(outs["tree"], outs["kill"]):
            eq += int((a.view(torch.int16) == b.view(torch.int16)).sum().item())
            tot += a.numel()
            finite &= bool(torch.isfinite(a.float()).all().item())
        hk = kind + "_M" + str(m)
        hashes[hk] = dict(tree=h16(outs["tree"]), kill=h16(outs["kill"]))
        times = dict((arm, []) for arm in ARMS)
        for it in range(WARM + ITERS):
            for arm in PERMS[it % len(PERMS)]:
                sts[arm].copy_(st0)
                flush.sum()
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
        a, b, c = r["tree"], r["kill"], r["aa"]
        r["tree_vs_kill_pct"] = 100.0 * (a["med"] - b["med"]) / b["med"]
        r["aa_vs_kill_pct"] = 100.0 * (c["med"] - b["med"]) / b["med"]
        r["kill_aa_bitwise"] = kill_aa_equal
        r["tree_kill_bitwise_frac"] = eq / max(tot, 1)
        r["tree_finite"] = finite
        r["layers"] = nl
        rm[kind] = r
        pa, pb, pc = r["tree_per_layer"], r["kill_per_layer"], r["aa_per_layer"]
        log("M=%d %-4s per-window us: tree %.2f [%.2f-%.2f] | kill %.2f [%.2f-%.2f] | aa %.2f [%.2f-%.2f] | "
            "tree-kill %+.2f%% | aa-kill %+.2f%% | graph total tree %.1f kill %.1f us | tree==kill frac %.3f | "
            "kill==aa %s | hash tree %s kill %s"
            % (m, kind, pa["med"], pa["p10"], pa["p90"], pb["med"], pb["p10"], pb["p90"], pc["med"], pc["p10"],
               pc["p90"], r["tree_vs_kill_pct"], r["aa_vs_kill_pct"], a["med"], b["med"],
               r["tree_kill_bitwise_frac"], kill_aa_equal, hashes[hk]["tree"], hashes[hk]["kill"]))
        if kind == "qkvz":
            for arm in ("tree", "kill"):
                fr = fork_profile(graphs[arm], sts[arm], st0, arm + "_M" + str(m))
                forks[arm + "_M" + str(m)] = fr
                log("   fork %s M=%d: %s" % (arm, m, json.dumps(fr)))
        del graphs, sts, outs
        torch.cuda.synchronize()
    results[str(m)] = rm


def faster(r):
    a, b, c = r["tree_per_layer"], r["kill_per_layer"], r["aa_per_layer"]
    gap = abs(a["med"] - b["med"])
    aagap = abs(c["med"] - b["med"])
    return a["p90"] < b["p10"] and a["p90"] < c["p10"] and a["med"] < b["med"] and gap > aagap


def same(r):
    a, b = r["tree_per_layer"], r["kill_per_layer"]
    return b["p10"] <= a["med"] <= b["p90"] or a["p10"] <= b["med"] <= a["p90"]


gate = dict()
gate["qkvz_M4_faster"] = faster(results["4"]["qkvz"])
gate["step_M4_faster"] = faster(results["4"]["step"])
gate["step_M1_same"] = same(results["1"]["step"])
gate["step_M2_same"] = same(results["2"]["step"])
gate["GO"] = all(gate.values())
info = dict()
info["qkv_M3_faster"] = faster(results["3"]["qkv"])
info["qkv_M4_faster"] = faster(results["4"]["qkv"])
info["step_M8_faster"] = faster(results["8"]["step"])
log("GATE (in-process, meaningful on the candidate tree):", gate, info)

batch = dict()
for m in MS:
    d = results[str(m)]["step"]["tree"]
    batch[str(m)] = dict(kernel_ms=d["med"] / 1e3, noise_floor=(d["p90"] - d["p10"]) / d["med"])
output = dict()
output["class"] = "reordered"
print(json.dumps(dict(batch=batch, output=output, results=results, forks=forks, hashes=hashes, gate=gate,
                      info=info, copies=COPY_INFO, warm=WARM, iters=ITERS,
                      note="kernel_ms = tree-default arm, one decode step's 48 qkvz fork windows + 16 qkv "
                           "windows (norm -> proj -> consumer), cold weights, READ flush")))
