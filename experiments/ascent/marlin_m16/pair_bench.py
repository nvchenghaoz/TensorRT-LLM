import os, sys, json
os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
import torch
from tensorrt_llm._torch.modules import linear as L
# Dense Marlin NVFP4 W4A16 decode GEMMs, cold-L2 CUDA-graph + PDL pair bench (brace-free: passed via python -c).
# Shapes (N,K,copies,calls/step): (5120,6144) x24 copies, 64 calls/step (GDN out_proj + attn o_proj);
# (16384,5120) x9, 48 (GDN in_proj_qkvz); (14336,5120) x10, 16 (attn qkv). Each graph = all copies of a shape
# (>= 400 MB, cold L2). Arms per tree, captured separately, replayed in rotated order in one process:
# default = this tree's selection; kill = TRTLLM_MARLIN_M16_TABLE=0 (main's selection); killA = A/A copy of kill.
# kernel_ms per batch = weighted per-step Marlin time sum(calls * us/call) for the default arm.
SH = [(5120, 6144, 24, 64), (16384, 5120, 9, 48), (14336, 5120, 10, 16)]
MS = [8, 12, 16]
WARM, ITERS = 10, 100
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
    # Direct bitwise checks after the timed loop (int16 view, torch.equal), per (arm, shape, M):
    # (a) every graph output == a fresh eager gemm on the same copy under the same arm's env;
    # (b) every graph output == the kill arm's graph output (expected False for default at M>8 in the candidate);
    # (c) diagnostic: fresh eager default == fresh eager kill.
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
            c_ok = all(eqv(o, e) for o, e in zip(eager[arm], eager["kill"]))
            direct.setdefault(arm, dict()).setdefault(key, dict())[str(m)] = dict(
                graph_eq_fresh_eager=a_ok, graph_eq_kill_graph=b_ok, eager_eq_kill_eager=c_ok, copies=C)
            log("direct (%d,%d) M=%d arm=%s graph==fresh_eager %s graph==kill_graph %s eager==kill_eager %s (%d copies)"
                % (n, k, m, arm, a_ok, b_ok, c_ok, C))
        del eager
    for arm in ARMS:
        per_it = [0.0] * ITERS
        for n, k, C, w in SH:
            ts = [e0.elapsed_time(e1) / C for e0, e1 in evs[(arm, n, k)]]
            for i in range(ITERS):
                per_it[i] += ts[i] * w
            t = sorted(ts)
            med, p10, p90 = t[len(t) // 2], t[len(t) // 10], t[(9 * len(t)) // 10]
            key = "%d,%d" % (n, k)
            res.setdefault(arm, dict()).setdefault(key, dict())[str(m)] = dict(
                us=med * 1e3, p10_us=p10 * 1e3, p90_us=p90 * 1e3, noise_floor=(p90 - p10) / med)
            hashes.setdefault(arm, dict()).setdefault(key, dict())[str(m)] = h(outs[(arm, n, k)])
        t = sorted(per_it)
        med, p10, p90 = t[len(t) // 2], t[len(t) // 10], t[(9 * len(t)) // 10]
        step.setdefault(arm, dict())[str(m)] = dict(kernel_ms=med, p10_ms=p10, p90_ms=p90,
                                                     noise_floor=(p90 - p10) / med)
    for n, k, C, w in SH:
        key = "%d,%d" % (n, k)
        d = res["default"][key][str(m)]
        kk = res["kill"][key][str(m)]
        ka = res["killA"][key][str(m)]
        log("(%d,%d) M=%d default %.2f us (p10 %.2f p90 %.2f) kill %.2f us killA %.2f us d=%+.2f us (%+.1f%%) "
            "default==kill bitwise %s" % (n, k, m, d["us"], d["p10_us"], d["p90_us"], kk["us"], ka["us"],
            d["us"] - kk["us"], 100.0 * (d["us"] / kk["us"] - 1),
            hashes["default"][key][str(m)] == hashes["kill"][key][str(m)]))
    log("M=%d weighted step Marlin ms: default %.4f kill %.4f killA %.4f" % (
        m, step["default"][str(m)]["kernel_ms"], step["kill"][str(m)]["kernel_ms"], step["killA"][str(m)]["kernel_ms"]))
    del graphs, outs
    torch.cuda.synchronize()
batch = dict()
for mm in ["8", "16"]:
    batch[mm] = step["default"][mm]
out = dict(batch=batch, measure="per-step weighted dense Marlin decode time: 64 x (5120,6144) + 48 x (16384,5120) + 16 x (14336,5120), ms; median of 100 rotated replays",
           step_by_arm=step, shapes=res, output_hashes=hashes, direct_checks=direct, warmup=WARM, iters=ITERS)
out["output"] = dict(note="rows 9-16 reordered (see check job); rows 8 and the kill arm bitwise equal to base via output_hashes; "
                     "direct_checks: torch.equal on int16 views of every graph output vs fresh eager (same arm env) and vs the kill arm")
out["output"]["class"] = "reordered"
print(json.dumps(out))
