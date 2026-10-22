import json, os, sys
sys.path.insert(0, os.path.join(os.getcwd(), "experiments/ascent/cascade"))
import cascade_lib as cl
import numpy as np
import torch
# Pair arm for idea 76 (run inline with python -c in each tree): cwd's kernel_mha_prefix alone (kernel_ms) and prefix +
# kernel_mha (total_ms), B=8/16/24/32, serving shapes, CUDA graph replay, cold L2 (384 MiB read flush) + NaN poison
# before every replay, prefix/total alternated per round, 4 rounds x (20 warmup + 100 timed), 10-90% trimmed mean.
cl.init_cuda()
k = cl.Kernels(os.path.join(os.getcwd(), "cpp/kernels/xqa"))
inp = cl.build_inputs(32)
fl = cl.xe.Flusher(384)
res = {"batch": {}, "output": {"class": "exact"}, "prefix_regs": k.pregs, "prefix_local_bytes": k.plmem,
       "source_sha256": k.src_sha}
for b in (8, 16, 24, 32):
    m = cl.main_nb_sub_seq(b)
    pf = cl.Call(k, inp, b, m, cascade=True)
    lp = lambda: cl.Kernels._launch(k.prefix, (pf.c, pf.g, cl.NB_K), (k.pcta, 1, 1), k.psmem, pf.pfx_args)
    lp()
    torch.cuda.synchronize()
    gp = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gp):
        lp()
    gt = cl.Call(k, inp, b, m, cascade=True).capture().graph
    ts = {"p": [], "t": []}
    for rnd in range(4):
        for name in (("p", "t") if rnd % 2 == 0 else ("t", "p")):
            g = gp if name == "p" else gt
            n = 120
            st = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
            en = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
            for i in range(n):
                inp["out"].fill_(float("nan"))
                fl()
                st[i].record()
                g.replay()
                en[i].record()
            torch.cuda.synchronize()
            ts[name].append(np.array([st[i].elapsed_time(en[i]) for i in range(20, n)]))
    sp, stt = cl.stats(np.concatenate(ts["p"])), cl.stats(np.concatenate(ts["t"]))
    res["batch"][str(b)] = dict(kernel_ms=sp["kernel_ms"], noise_floor=sp["noise_floor"], median_ms=sp["median_ms"],
                                p90_ms=sp["p90_ms"], total_ms=stt["kernel_ms"], total_median_ms=stt["median_ms"],
                                total_p90_ms=stt["p90_ms"], total_noise_floor=stt["noise_floor"], chunks=pf.c)
    print(f"B={b}: prefix {sp['kernel_ms'] * 1e3:.2f} us (noise {sp['noise_floor']:.3f}), total "
          f"{stt['kernel_ms'] * 1e3:.2f} us", flush=True)
print(json.dumps(res), flush=True)
