import json, os, statistics, subprocess, sys
# Dense Marlin NVFP4 W4A16 decode GEMMs: cold-L2 CUDA-graph + PDL pair bench, NPROC independent processes per tree,
# median of per-process medians; noise floor = max(median within-run (p90-p10)/median, cross-process range).
# Brace-free so it can be passed via python -c to both trees (the base tree does not have this file).
# Arms per process, captured separately and replayed in rotated order: default = this tree's selection;
# kill = TRTLLM_MARLIN_M16_TABLE=0 (main's selection); killA = A/A copy of kill.
# Shapes (N,K,copies,calls/step): (5120,6144) x24, 64 (GDN out_proj + attn o_proj); (16384,5120) x9, 48 (GDN
# in_proj_qkvz); (14336,5120) x10, 16 (attn qkv). Each graph = all copies of one shape (>= 400 MB, cold L2).
# After the timed loop each process checks with torch.equal on int16 views, per (arm, shape, M), every timed-graph
# output against (a) a fresh eager gemm on the same copy under the same arm's env and (b) the kill arm's graph
# output, and hashes each arm's own timed-graph outputs. Process 0 also records, from a torch.profiler trace of one
# eager call per (arm, shape, M), the Marlin template that ran (threads, m_blocks, n_blocks, k_blocks, m8, stages,
# group_blocks, full_k) and its grid/block.
BENCH = r"""
import os, sys, json, tempfile
os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
import torch
from tensorrt_llm._torch.modules import linear as L
SH = [(5120, 6144, 24, 64), (16384, 5120, 9, 48), (14336, 5120, 10, 16)]
MS = [8, 12, 16]
WARM, ITERS = 10, 100
PROFILE = os.environ.get("MARLIN_M16_PROFILE", "0") == "1"
dev = torch.device("cuda")
def log(*a):
    print(*a, flush=True)
def mk(n, k, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    w = torch.randn(n, k, generator=g, device=dev) * 0.02
    w = w * (torch.rand(n, 1, generator=g, device=dev) * 1.5 + 0.25)
    s = (w.abs().amax() / 448.0).float()
    w8 = (w / s).to(torch.float8_e4m3fn)
    del w
    wv = w8.to(torch.bfloat16)
    amax = wv.abs().amax().float() * s
    gg = torch.exp2(torch.ceil(torch.log2(amax / 2688.0)))
    fp4, sf = torch.ops.trtllm.fp4_quantize(wv, (s / gg).reshape(1), 16, False, True)
    return L._pack_nvfp4_weight_for_marlin(fp4, sf, gg.reshape(1).to(torch.float32), n, k, 16)
def gemm(x, c, n, k):
    return torch.ops.trtllm.marlin_nvfp4_gemm(x, c[0], scale_a=None, scale_b=c[1], alpha=None,
        weight_global_scale=c[2], bias=None, out_dtype=torch.bfloat16, size_n=n, size_k=k)
def h(ts):
    acc = 0
    for t in ts:
        v = t.contiguous().view(torch.int16).view(-1).to(torch.int64)
        idx = torch.arange(v.numel(), device=v.device, dtype=torch.int64)
        acc = (acc * 1000003 + int(((v + 40000) * (idx % 65521 + 1)).sum().item())) % (2 ** 61 - 1)
    return str(acc)
def eqv(a, b):
    return bool(torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16)))
def setarm(arm):
    os.environ.pop("TRTLLM_MARLIN_M16_CFG", None)
    if arm.startswith("kill"):
        os.environ["TRTLLM_MARLIN_M16_TABLE"] = "0"
    else:
        os.environ.pop("TRTLLM_MARLIN_M16_TABLE", None)
def stats(v):
    t = sorted(v)
    med, p10, p90 = t[len(t) // 2], t[len(t) // 10], t[(9 * len(t)) // 10]
    return med, p10, p90
ARMS = ["default", "kill", "killA"]
copies = dict()
for n, k, C, w in SH:
    copies[(n, k)] = [mk(n, k, 1000 + i) for i in range(C)]
    torch.cuda.synchronize()
    log("built", n, k, "copies", C, "mem GB %.2f" % (torch.cuda.memory_allocated() / 1e9))
res = dict()
hashes = dict()
direct = dict()
step = dict()
for m in MS:
    graphs = dict()
    outs = dict()
    xs = dict()
    for n, k, C, w in SH:
        g = torch.Generator(device=dev).manual_seed(7 + m)
        x = torch.randn(m, k, generator=g, device=dev).to(torch.bfloat16)
        xs[(n, k)] = x
        for arm in ARMS:
            setarm(arm)
            s0 = torch.cuda.Stream()
            s0.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s0):
                for _ in range(2):
                    o = [gemm(x, c, n, k) for c in copies[(n, k)]]
            torch.cuda.current_stream().wait_stream(s0)
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                o = [gemm(x, c, n, k) for c in copies[(n, k)]]
            graphs[(arm, n, k)] = gr
            outs[(arm, n, k)] = o
    setarm("default")
    keys = list(graphs)
    for key in keys:
        for _ in range(WARM):
            graphs[key].replay()
    torch.cuda.synchronize()
    evs = dict((key, []) for key in keys)
    for it in range(ITERS):
        r = it % len(keys)
        order = keys[r:] + keys[:r]
        if it % 2:
            order = order[::-1]
        for key in order:
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            graphs[key].replay()
            e1.record()
            evs[key].append((e0, e1))
    torch.cuda.synchronize()
    for n, k, C, w in SH:
        key = "%d,%d" % (n, k)
        x = xs[(n, k)]
        eager = dict()
        for arm in ARMS:
            setarm(arm)
            eager[arm] = [gemm(x, c, n, k) for c in copies[(n, k)]]
        torch.cuda.synchronize()
        setarm("default")
        for arm in ARMS:
            a_ok = all(eqv(o, e) for o, e in zip(outs[(arm, n, k)], eager[arm]))
            b_ok = all(eqv(o, e) for o, e in zip(outs[(arm, n, k)], outs[("kill", n, k)]))
            direct.setdefault(arm, dict()).setdefault(key, dict())[str(m)] = dict(
                graph_eq_fresh_eager=a_ok, graph_eq_kill_graph=b_ok, copies=C)
            hashes.setdefault(arm, dict()).setdefault(key, dict())[str(m)] = h(outs[(arm, n, k)])
        del eager
    for arm in ARMS:
        per_it = [0.0] * ITERS
        for n, k, C, w in SH:
            ts = [e0.elapsed_time(e1) / C for e0, e1 in evs[(arm, n, k)]]
            for i in range(ITERS):
                per_it[i] += ts[i] * w
            med, p10, p90 = stats(ts)
            res.setdefault(arm, dict()).setdefault("%d,%d" % (n, k), dict())[str(m)] = dict(
                us=med * 1e3, p10_us=p10 * 1e3, p90_us=p90 * 1e3, noise_floor=(p90 - p10) / med)
        med, p10, p90 = stats(per_it)
        step.setdefault(arm, dict())[str(m)] = dict(kernel_ms=med, p10_ms=p10, p90_ms=p90,
                                                     noise_floor=(p90 - p10) / med)
    for n, k, C, w in SH:
        key = "%d,%d" % (n, k)
        d = res["default"][key][str(m)]
        kk = res["kill"][key][str(m)]
        ka = res["killA"][key][str(m)]
        dd = direct["default"][key][str(m)]
        log("(%d,%d) M=%d default %.2f us kill %.2f killA %.2f d=%+.1f%% | default graph==fresh eager %s, "
            "default graph==kill graph %s" % (n, k, m, d["us"], kk["us"], ka["us"], 100.0 * (d["us"] / kk["us"] - 1),
            dd["graph_eq_fresh_eager"], dd["graph_eq_kill_graph"]))
    log("M=%d weighted step Marlin ms: default %.4f kill %.4f killA %.4f" % (
        m, step["default"][str(m)]["kernel_ms"], step["kill"][str(m)]["kernel_ms"], step["killA"][str(m)]["kernel_ms"]))
    del graphs, outs
    torch.cuda.synchronize()
kcfg = dict()
if PROFILE:
    try:
        from torch.profiler import profile, ProfilerActivity
        path = os.path.join(tempfile.gettempdir(), "marlin_m16_trace_%d.json" % os.getpid())
        for m in MS:
            for n, k, C, w in SH:
                g = torch.Generator(device=dev).manual_seed(7 + m)
                x = torch.randn(m, k, generator=g, device=dev).to(torch.bfloat16)
                for arm in ["default", "kill"]:
                    setarm(arm)
                    gemm(x, copies[(n, k)][0], n, k)
                    torch.cuda.synchronize()
                    with profile(activities=[ProfilerActivity.CUDA]) as prof:
                        gemm(x, copies[(n, k)][0], n, k)
                        torch.cuda.synchronize()
                    prof.export_chrome_trace(path)
                    with open(path) as f:
                        tr = json.load(f)
                    ks = []
                    for ev in tr.get("traceEvents", []):
                        nm = ev.get("name", "")
                        if ev.get("cat") == "kernel" and "Marlin" in nm:
                            a = ev.get("args", dict())
                            i0 = nm.find("Marlin<")
                            tmpl = nm[i0:nm.find(">", i0) + 1] if i0 >= 0 else nm
                            ks.append(dict(kernel=tmpl, grid=a.get("grid"), block=a.get("block"),
                                           dyn_smem=a.get("shared memory")))
                    kcfg.setdefault(arm, dict()).setdefault("%d,%d" % (n, k), dict())[str(m)] = ks
                    log("kernel (%d,%d) M=%d arm=%s %s" % (n, k, m, arm, json.dumps(ks)))
        setarm("default")
    except Exception as e:
        log("profile failed", repr(e))
print(json.dumps(dict(shapes=res, step=step, hashes=hashes, direct=direct, kernels=kcfg)))
"""
NPROC = int(os.environ.get("BENCH_NPROC", "5"))
ARMS = ["default", "kill", "killA"]
runs = []
for r in range(NPROC):
    env = dict(os.environ)
    env["MARLIN_M16_PROFILE"] = "1" if r == 0 else "0"
    p = subprocess.Popen([sys.executable, "-c", BENCH], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, env=env)
    last = None
    for line in p.stdout:
        if line.startswith(chr(123)):
            last = line
        else:
            print("[proc %d] %s" % (r, line.rstrip()), flush=True)
    p.wait()
    if p.returncode != 0 or last is None:
        raise SystemExit("bench process %d failed (rc %s)" % (r, p.returncode))
    runs.append(json.loads(last))
def agg(vals_med, vals_noise):
    mm = statistics.median(vals_med)
    within = statistics.median(vals_noise)
    across = (max(vals_med) - min(vals_med)) / mm
    return mm, within, across
step_mm = dict()
shapes_mm = dict()
for arm in ARMS:
    for m in runs[0]["step"][arm]:
        mm, within, across = agg([x["step"][arm][m]["kernel_ms"] for x in runs],
                                 [x["step"][arm][m]["noise_floor"] for x in runs])
        step_mm.setdefault(arm, dict())[m] = dict(kernel_ms=mm, noise_floor=max(within, across),
            within_run_p90_p10=within, across_process_range=across,
            process_medians_ms=[x["step"][arm][m]["kernel_ms"] for x in runs])
    for key in runs[0]["shapes"][arm]:
        for m in runs[0]["shapes"][arm][key]:
            mm, within, across = agg([x["shapes"][arm][key][m]["us"] for x in runs],
                                     [x["shapes"][arm][key][m]["noise_floor"] for x in runs])
            shapes_mm.setdefault(arm, dict()).setdefault(key, dict())[m] = dict(
                us=mm, noise_floor=max(within, across), process_medians_us=[x["shapes"][arm][key][m]["us"] for x in runs])
direct_all = dict()
hash_stable = True
for arm in ARMS:
    for key in runs[0]["direct"][arm]:
        for m in runs[0]["direct"][arm][key]:
            a = all(x["direct"][arm][key][m]["graph_eq_fresh_eager"] for x in runs)
            b = all(x["direct"][arm][key][m]["graph_eq_kill_graph"] for x in runs)
            bn = any(x["direct"][arm][key][m]["graph_eq_kill_graph"] for x in runs)
            direct_all.setdefault(arm, dict()).setdefault(key, dict())[m] = dict(
                graph_eq_fresh_eager_all_procs=a, graph_eq_kill_graph_all_procs=b, graph_eq_kill_graph_any_proc=bn)
            hs = [x["hashes"][arm][key][m] for x in runs]
            if len(set(hs)) != 1:
                hash_stable = False
for key in shapes_mm["default"]:
    for m in shapes_mm["default"][key]:
        d = shapes_mm["default"][key][m]
        kk = shapes_mm["kill"][key][m]
        dd = direct_all["default"][key][m]
        print("SUMMARY (%s) M=%s default %.2f us kill %.2f us (%+.1f%%) noise %.3f | graph==fresh eager %s | "
              "graph==kill %s" % (key, m, d["us"], kk["us"], 100.0 * (d["us"] / kk["us"] - 1), max(d["noise_floor"],
              kk["noise_floor"]), dd["graph_eq_fresh_eager_all_procs"], dd["graph_eq_kill_graph_all_procs"]), flush=True)
batch = dict()
for m in ["8", "16"]:
    batch[m] = step_mm["default"][m]
out = dict(batch=batch,
           measure="per-step weighted dense Marlin decode time: 64 x (5120,6144) + 48 x (16384,5120) + 16 x "
                   "(14336,5120), ms; median over %d processes of each process's median of 100 rotated replays" % NPROC,
           step_by_arm=step_mm, shapes=shapes_mm, direct_checks=direct_all,
           output_hashes=runs[0]["hashes"], output_hashes_identical_across_processes=hash_stable,
           kernels=runs[0]["kernels"], nproc=NPROC, warmup=10, iters=100)
out["output"] = dict(note="rows 9-16 reordered (check job); rows 8 and the kill arm bitwise equal to main; "
                          "direct_checks = torch.equal on int16 views of every timed-graph output")
out["output"]["class"] = "reordered"
print(json.dumps(out))
