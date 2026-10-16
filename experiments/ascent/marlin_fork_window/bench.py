# Kernel-window gate for idea 137: TRTLLM_MARLIN_FORK_FRESH_WORKSPACE default "1" -> "0".
# Window per layer, production enqueue order, 48 layers in one CUDA graph:
#   1. gemma fused add+RMSNorm (PDL; torch.ops.trtllm.flashinfer_gemma_fused_add_rmsnorm)
#   2. ev0.record
#   3. in_proj_qkvz Marlin 16384x5120 on the current stream
#      || aux: ev0.wait, in_proj_ba via maybe_small_n_gemv (96x5120), ev1.record
#   4. ev1.wait
#   5. one non-PDL elementwise consumer of both outputs: z = y[:, :96] + ba
# Arms in ONE process, alternating replays (order flipped every iteration):
#   env1    = TRTLLM_MARLIN_FORK_FRESH_WORKSPACE=1 (main's marking: mode 1 at rows >= 2)
#   default = env unset (the tree's default: mode 1 at rows >= 2 on main, mode 0 on the candidate)
# The Marlin workspace mode is taken from production code: a fresh module per arm whose children are
# in_proj_qkvz (with _nvfp4_decode_weight) and in_proj_ba, marked by linear.mark_marlin_fork_sites,
# mode = linear._marlin_workspace_mode(qkvz, M).
# Cold L2: 8 distinct qkvz weight copies (~377 MB) cycled across the 48 layers, plus a 256 MiB flush
# before every replay outside the timed event pair. Input/residual reset: one st.copy_(st0) per
# replay, before the flush, outside the timed event pair (identical in both arms).
# One CUDA-event pair around the whole replay; per layer = total / 48.
# Brace-free so run_gpu_pair's base arm can run it inline with python -c. Extra argv words
# (e.g. BASE_COMMIT=... CAND_COMMIT=...) are printed first.
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
os.environ.pop("TRTLLM_MARLIN_FORK_FRESH_WORKSPACE", None)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import tensorrt_llm._torch.custom_ops  # noqa: E402,F401
from tensorrt_llm._torch.modules import linear as L  # noqa: E402
from tensorrt_llm._torch.modules.small_n_gemv import maybe_small_n_gemv  # noqa: E402

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
log("linear.py:", L.__file__, "cwd:", os.getcwd())

dev = torch.device("cuda")
EPS = 1e-6
N, K, NBA = 16384, 5120, 96
C = 8
LAYERS = 48
WARM, ITERS = 20, 100
MS = (1, 2, 4, 8)
ENV = L._MARLIN_FORK_FRESH_WORKSPACE_ENV
ARMS = ("env1", "default")
log("gpu", torch.cuda.get_device_name(dev), "PDL", os.environ.get("TRTLLM_ENABLE_PDL"))


def build(seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    w = (torch.randn(N, K, generator=g, device=dev) * 0.02).to(torch.bfloat16)
    gs = (448.0 * 6.0 / w.float().abs().amax()).float().reshape(1)
    w_fp4, w_sf = torch.ops.trtllm.fp4_quantize(w, gs, 16, False, True)
    del w
    return L._pack_nvfp4_weight_for_marlin(w_fp4, w_sf, (1.0 / gs).float(), N, K, 16)


base = build(11 + N + K)
ws = [tuple(t.clone() for t in base) for _ in range(C)]
del base
wbytes = sum(t.numel() * t.element_size() for wt in ws for t in wt)
g = torch.Generator(device=dev).manual_seed(5)
wba = [(torch.randn(NBA, K, generator=g, device=dev) * 0.02).to(torch.bfloat16) for _ in range(C)]
wn = (0.1 * torch.randn(K, generator=g, device=dev)).to(torch.bfloat16)
flush = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device=dev)
log("qkvz copies", C, "total MB %.1f" % (wbytes / 1e6), "flush MiB 256")


def arm_module(arm):
    if arm == "env1":
        os.environ[ENV] = "1"
    else:
        os.environ.pop(ENV, None)
    root = torch.nn.Module()
    root.in_proj_qkvz = torch.nn.Module()
    root.in_proj_ba = torch.nn.Module()
    root.in_proj_qkvz._nvfp4_decode_weight = ws[0][0]
    n_marked = L.mark_marlin_fork_sites(root)
    os.environ.pop(ENV, None)
    return root.in_proj_qkvz, n_marked


QKVZ = dict()
MODES = dict()
for arm in ARMS:
    q, n_marked = arm_module(arm)
    QKVZ[arm] = q
    MODES[arm] = [L._marlin_workspace_mode(q, m) for m in MS]
    log("arm", arm, "marked", n_marked, "modes at M=1/2/4/8:", "/".join(str(x) for x in MODES[arm]))


def marlin(x, wt, mode):
    return torch.ops.trtllm.marlin_nvfp4_gemm(x, wt[0], scale_a=None, scale_b=wt[1], alpha=None,
                                              weight_global_scale=wt[2], bias=None,
                                              out_dtype=torch.bfloat16, size_n=N, size_k=K,
                                              workspace_mode=mode)


def in_proj_ba(x, w):
    y = maybe_small_n_gemv(x, w)
    return F.linear(x, w) if y is None else y


def norm(x, r, w):
    torch.ops.trtllm.flashinfer_gemma_fused_add_rmsnorm(x, r, w, EPS)


SIDE = torch.cuda.Stream()
AUX = torch.cuda.Stream()


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


def profile_once(gr, st, st0, tag):
    from torch.profiler import ProfilerActivity, profile
    st.copy_(st0)
    flush.fill_(7)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        gr.replay()
        torch.cuda.synchronize()
    path = os.path.join(os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp"), "trace_" + tag + ".json")
    prof.export_chrome_trace(path)
    with open(path) as fh:
        tr = json.load(fh)
    evs = tr.get("traceEvents", []) if isinstance(tr, dict) else tr
    per_stream = dict()
    marlin_grids = set()
    n_fill = 0
    n_k = 0
    for e in evs:
        if e.get("cat") != "kernel":
            continue
        n_k += 1
        args = e.get("args", dict())
        s = str(args.get("stream"))
        name = e.get("name", "")
        short = name.split("(")[0][-60:]
        d = per_stream.setdefault(s, dict())
        d[short] = d.get(short, 0) + 1
        if "arlin" in name:
            marlin_grids.add(str(args.get("grid")) + " block " + str(args.get("block")))
        if "Fill" in name or "fill" in name:
            n_fill += 1
    log("profile", tag, "kernels", n_k, "fill kernels", n_fill, "marlin grid", sorted(marlin_grids))
    for s, d in per_stream.items():
        log("   stream", s, sorted(d.items()))
    return dict(kernels=n_k, fill_kernels=n_fill, marlin_grids=sorted(marlin_grids),
                streams=len(per_stream))


results = dict()
profiles = dict()
for m in MS:
    per = 2 * m * K
    st0 = (torch.randn(LAYERS * per, generator=g, device=dev) * 0.5).to(torch.bfloat16)
    graphs = dict()
    sts = dict()
    outs = dict()
    for arm in ARMS:
        mode = L._marlin_workspace_mode(QKVZ[arm], m)
        st = st0.clone()
        xs = [st[i * per: i * per + m * K].view(m, K) for i in range(LAYERS)]
        rs = [st[i * per + m * K: (i + 1) * per].view(m, K) for i in range(LAYERS)]
        ev0 = [torch.cuda.Event() for _ in range(LAYERS)]
        ev1 = [torch.cuda.Event() for _ in range(LAYERS)]

        def fn(xs=xs, rs=rs, ev0=ev0, ev1=ev1, mode=mode):
            zs = []
            for i in range(LAYERS):
                c = i % C
                norm(xs[i], rs[i], wn)
                ev0[i].record()
                y = marlin(xs[i], ws[c], mode)
                with torch.cuda.stream(AUX):
                    ev0[i].wait()
                    ba = in_proj_ba(xs[i], wba[c])
                    ev1[i].record()
                ev1[i].wait()
                zs.append(y[:, :NBA] + ba)
            return zs

        gr, out = capture(fn)
        graphs[arm] = gr
        sts[arm] = st
        outs[arm] = out
    # both arms' outputs bitwise equal after one replay from the same state
    for arm in ARMS:
        sts[arm].copy_(st0)
        graphs[arm].replay()
    torch.cuda.synchronize()
    same = all(torch.equal(a, b) for a, b in zip(outs[ARMS[0]], outs[ARMS[1]]))
    times = dict((arm, []) for arm in ARMS)
    for it in range(WARM + ITERS):
        order = ARMS if it % 2 == 0 else ARMS[::-1]
        for arm in order:
            sts[arm].copy_(st0)
            flush.fill_(it)
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            graphs[arm].replay()
            e1.record()
            if it >= WARM:
                times[arm].append((e0, e1))
        if it % 40 == 39:
            torch.cuda.synchronize()
            log("M=%d iteration %d" % (m, it + 1))
    torch.cuda.synchronize()
    r = dict()
    for arm in ARMS:
        r[arm] = stats([e0.elapsed_time(e1) * 1e3 / LAYERS for e0, e1 in times[arm]])
        r[arm]["mode"] = L._marlin_workspace_mode(QKVZ[arm], m)
    a, b = r["env1"], r["default"]
    r["default_minus_env1_us"] = b["med"] - a["med"]
    r["default_vs_env1_pct"] = 100.0 * (b["med"] - a["med"]) / a["med"]
    r["outputs_bitwise_equal"] = same
    results[str(m)] = r
    log("M=%d per-layer us: env1(mode %d) med %.2f p10 %.2f p90 %.2f | default(mode %d) med %.2f "
        "p10 %.2f p90 %.2f | default-env1 %+.2f us (%+.2f%%) | outputs equal %s"
        % (m, a["mode"], a["med"], a["p10"], a["p90"], b["mode"], b["med"], b["p10"], b["p90"],
           r["default_minus_env1_us"], r["default_vs_env1_pct"], same))
    if m in (2, 4):
        for arm in ARMS:
            profiles[arm + "_M" + str(m)] = profile_once(graphs[arm], sts[arm], st0, arm + "_M" + str(m))
    del graphs, sts, outs
    torch.cuda.synchronize()

# Gate (fixed before measuring; cand = default arm on the candidate tree, main = env1 arm).
r1, r2, r4, r8 = (results[str(m)] for m in MS)
gate = dict()
gate["M4_ge_4pct"] = r4["default_vs_env1_pct"] <= -4.0
gate["M4_cand_p90_lt_main_p10"] = r4["default"]["p90"] < r4["env1"]["p10"]
gate["M2_cand_p90_lt_main_p10"] = r2["default"]["p90"] < r2["env1"]["p10"]
gate["M1_unchanged"] = (r1["env1"]["p10"] <= r1["default"]["med"] <= r1["env1"]["p90"]
                        or r1["default"]["p10"] <= r1["env1"]["med"] <= r1["default"]["p90"])
gate["M8_not_slower"] = r8["default"]["med"] <= r8["env1"]["p90"]
gate["GO"] = all(gate.values())
log("modes env1", MODES["env1"], "default", MODES["default"])
log("GATE (meaningful on the candidate tree):", gate)

batch = dict()
for m in MS:
    d = results[str(m)]["default"]
    batch[str(m)] = dict(kernel_ms=d["med"] / 1e3, noise_floor=(d["p90"] - d["p10"]) / d["med"])
output = dict()
output["class"] = "exact"
print(json.dumps(dict(batch=batch, output=output, per_layer_us=results, modes=MODES, gate=gate,
                      profiles=profiles, layers=LAYERS, warm=WARM, iters=ITERS,
                      qkvz_copies=C, qkvz_mb=wbytes / 1e6,
                      note="kernel_ms = per-layer window time of the tree-default arm (env unset)")))
