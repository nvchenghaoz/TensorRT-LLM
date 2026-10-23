import json, os, sys
sys.path.insert(0, os.path.join(os.getcwd(), "experiments/ascent/cascade"))
import cascade_lib as cl
import numpy as np
import torch
# Pair arm for idea 22 (run inline with python -c in each tree, so the base arm needs no file of this branch): the
# whole cascade decode attention op (kernel_mha_prefix + kernel_mha suffix/merge) -> kernel_ms, and kernel_mha_prefix
# alone -> prefix_only_us, at B = 64 / 100 / 128, serving shape (24q/4kv d256 e4m3 paged KV, 32768 shared + 2176
# unique tokens), kernel_mha sub-sequence count and prefix grid as the runner picks them. CUDA graph replay, cold L2
# (384 MiB read flush) + NaN poison before every replay, prefix/total alternated per round, 4 rounds x (20 warmup +
# 100 timed), 10-90% trimmed mean. prefix_tflops counts 4 * B * 24 * 32768 * 256 FLOPs (QK^T + PV).
cl.init_cuda()
k = cl.Kernels(os.path.join(os.getcwd(), "cpp/kernels/xqa"))
inp = cl.build_inputs(128)
fl = cl.xe.Flusher(384)
res = {"batch": {}, "output": {"class": "precision"}, "prefix_regs": k.pregs, "prefix_local_bytes": k.plmem,
       "source_sha256": k.src_sha}
for b in (64, 100, 128):
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
    flops = 4.0 * b * cl.NB_Q * 32768 * cl.HD
    res["batch"][str(b)] = dict(kernel_ms=stt["kernel_ms"], noise_floor=stt["noise_floor"],
                                median_ms=stt["median_ms"], p90_ms=stt["p90_ms"],
                                prefix_only_us=sp["kernel_ms"] * 1e3, prefix_median_us=sp["median_ms"] * 1e3,
                                prefix_p90_us=sp["p90_ms"] * 1e3, prefix_noise_floor=sp["noise_floor"],
                                prefix_tflops=flops / (sp["kernel_ms"] * 1e-3) / 1e12, chunks=pf.c, row_groups=pf.g,
                                nb_sub_seq=m)
    print(f"B={b} (C={pf.c} G={pf.g} m={m}): prefix {sp['kernel_ms'] * 1e3:.2f} us "
          f"({res['batch'][str(b)]['prefix_tflops']:.0f} TFLOP/s, noise {sp['noise_floor']:.3f}), total "
          f"{stt['kernel_ms'] * 1e3:.2f} us (noise {stt['noise_floor']:.3f})", flush=True)
print(json.dumps(res), flush=True)
