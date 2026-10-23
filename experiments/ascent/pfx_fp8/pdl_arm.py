import json, os, sys
sys.path.insert(0, os.path.join(os.getcwd(), "experiments/ascent/cascade"))
import cascade_lib as cl
import numpy as np
import torch
# Pair arm for idea 22, case set of the top-lead ruling (run inline with python -c in each tree): the WHOLE cascade
# decode attention op (kernel_mha_prefix + kernel_mha suffix walk + fold), both launched with
# CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION as in serving, timed from the first launch to the last kernel's
# end with CUDA events: B=64 and B=128 as CUDA-graph replays, B=60 and B=124 (mixed-step generation batches) as eager
# launches. Serving shape: 24q/4kv d256 e4m3 paged KV, 32768 shared + 2176 unique tokens; grids as the runner picks
# them. Cold L2 (384 MiB read flush) + NaN poison before every call; 20 warmup, then 3 rounds x 100 timed with the
# cases alternated; 10-90% trimmed mean, median, p90. prefix_only_us (graph, PDL) and prefix TFLOP/s are context.
cl.init_cuda()
k = cl.Kernels(os.path.join(os.getcwd(), "cpp/kernels/xqa"))
inp = cl.build_inputs(128)
fl = cl.xe.Flusher(384)
CASES = [(64, True), (128, True), (60, False), (124, False)]


def mk(b):
    m = cl.main_nb_sub_seq(b)
    c = cl.Call(k, inp, b, m, cascade=True)
    pre = lambda: cl.xe.launch_ex(k.prefix, (c.c, c.g, cl.NB_K), (k.pcta, 1, 1), k.psmem, c.pfx_args.ptr, True)
    mha = lambda: cl.xe.launch_ex(k.mha, (m, cl.NB_K, b), (128, 1, 2), k.smem, c.mha_args.ptr, True)
    return c, m, pre, mha


runs = {}
for b, graph in CASES:
    c, m, pre, mha = mk(b)
    def both(pre=pre, mha=mha):
        pre()
        mha()
    both()
    torch.cuda.synchronize()
    if graph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            both()
        torch.cuda.synchronize()
        gp = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gp):
            pre()
        torch.cuda.synchronize()
        runs[b] = dict(c=c, m=m, fn=g.replay, pfn=gp.replay, mode="cuda_graph_pdl")
    else:
        runs[b] = dict(c=c, m=m, fn=both, pfn=None, mode="eager_pdl")


def time_fn(fn, n):
    st = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    for i in range(n):
        inp["out"].fill_(float("nan"))
        fl()
        st[i].record()
        fn()
        en[i].record()
    torch.cuda.synchronize()
    return np.array([st[i].elapsed_time(en[i]) for i in range(n)])


for b, _ in CASES:
    time_fn(runs[b]["fn"], 20)
    if runs[b]["pfn"] is not None:
        time_fn(runs[b]["pfn"], 20)
ts = {b: [] for b, _ in CASES}
tp = {b: [] for b, _ in CASES}
for rnd in range(3):
    order = CASES if rnd % 2 == 0 else CASES[::-1]
    for b, _ in order:
        ts[b].append(time_fn(runs[b]["fn"], 100))
        if runs[b]["pfn"] is not None:
            tp[b].append(time_fn(runs[b]["pfn"], 100))
res = {"batch": {}, "output": {"class": "precision"}, "prefix_regs": k.pregs, "prefix_local_bytes": k.plmem,
       "source_sha256": k.src_sha}
for b, graph in CASES:
    s = cl.stats(np.concatenate(ts[b]))
    r = runs[b]
    row = dict(kernel_ms=s["kernel_ms"], noise_floor=s["noise_floor"], median_ms=s["median_ms"], p10_ms=s["p10_ms"],
               p90_ms=s["p90_ms"], samples=s["samples"], mode=r["mode"], prefix_chunks=r["c"].c,
               row_groups=r["c"].g, nb_sub_seq=r["m"])
    if tp[b]:
        sp = cl.stats(np.concatenate(tp[b]))
        row.update(prefix_only_us=sp["kernel_ms"] * 1e3, prefix_noise_floor=sp["noise_floor"],
                   prefix_tflops=4.0 * b * cl.NB_Q * 32768 * cl.HD / (sp["kernel_ms"] * 1e-3) / 1e12)
    res["batch"][str(b)] = row
    print(f"B={b} {r['mode']} (C={r['c'].c} G={r['c'].g} m={r['m']}): cascade {s['kernel_ms'] * 1e3:.2f} us "
          f"(median {s['median_ms'] * 1e3:.2f}, p90 {s['p90_ms'] * 1e3:.2f}, noise {s['noise_floor']:.3f})"
          + (f", prefix only {row['prefix_only_us']:.2f} us ({row['prefix_tflops']:.0f} TFLOP/s)" if tp[b] else ""),
          flush=True)
print(json.dumps(res), flush=True)
