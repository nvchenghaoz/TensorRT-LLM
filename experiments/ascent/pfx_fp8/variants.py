#!/usr/bin/env python3
# Idea 22 step 0: kernel_mha_prefix variants of this tree's mha.cu compiled in one process (XQA_PFX_FP8 / XQA_PFX_Q_TERMS
# macros): FP8 m16n8k32 with two-term Q (default), FP8 with one-term Q, and the BF16 m16n8k16 path (= main's code).
# Prefix-only and prefix + kernel_mha CUDA-graph replays (cold L2: 384 MiB read flush + NaN poison before each replay),
# B = 64 / 100 / 128 at the serving shape, plus output error vs the FP32 reference (q scaled x1 and x4 for peakier
# softmax) per variant.
import json
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "experiments/ascent/cascade"))
import cascade_lib as cl  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

VARIANTS = {"fp8_q2": {"XQA_PFX_FP8": "1", "XQA_PFX_Q_TERMS": "2"},
            "fp8_q1": {"XQA_PFX_FP8": "1", "XQA_PFX_Q_TERMS": "1"},
            "bf16": {"XQA_PFX_FP8": "0"}}
BATCHES = [int(x) for x in os.environ.get("PFX_BATCHES", "64,100,128").split(",")]
ROUNDS = int(os.environ.get("PFX_ROUNDS", "3"))


def build(macros):
    saved = dict(cl.xe.MACROS)
    cl.xe.MACROS.update(macros)
    try:
        return cl.Kernels(os.path.join(os.getcwd(), "cpp/kernels/xqa"))
    finally:
        cl.xe.MACROS.clear()
        cl.xe.MACROS.update(saved)


def graphs(k, inp, b):
    m = cl.main_nb_sub_seq(b)
    pf = cl.Call(k, inp, b, m, cascade=True)

    def lp():
        cl.Kernels._launch(k.prefix, (pf.c, pf.g, cl.NB_K), (k.pcta, 1, 1), k.psmem, pf.pfx_args)

    lp()
    torch.cuda.synchronize()
    gp = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gp):
        lp()
    full = cl.Call(k, inp, b, m, cascade=True).capture()
    return gp, full, pf


def time_g(g, inp, fl, n=120, warm=20):
    st = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    for i in range(n):
        inp["out"].fill_(float("nan"))
        fl()
        st[i].record()
        g.replay()
        en[i].record()
    torch.cuda.synchronize()
    return np.array([st[i].elapsed_time(en[i]) for i in range(warm, n)])


def main():
    cl.init_cuda()
    kerns = {name: build(mc) for name, mc in VARIANTS.items()}
    for name, k in kerns.items():
        cl.log(f"{name}: prefix regs {k.pregs}, local {k.plmem} B, smem {k.psmem}")
    inp = cl.build_inputs(max(BATCHES))
    fl = cl.xe.Flusher(384)
    res = {"regs": {n: k.pregs for n, k in kerns.items()}, "local": {n: k.plmem for n, k in kerns.items()},
           "batch": {}, "err": {}}
    for b in BATCHES:
        gs = {n: graphs(k, inp, b) for n, k in kerns.items()}
        ts = {n: {"p": [], "t": []} for n in kerns}
        for rnd in range(ROUNDS):
            order = list(kerns) if rnd % 2 == 0 else list(kerns)[::-1]
            for n in order:
                ts[n]["p"].append(time_g(gs[n][0], inp, fl))
                ts[n]["t"].append(time_g(gs[n][1].graph, inp, fl))
        row = {}
        c, g = gs["bf16"][2].c, gs["bf16"][2].g
        flops = 4.0 * b * cl.NB_Q * 32768 * cl.HD
        for n in kerns:
            sp = cl.stats(np.concatenate(ts[n]["p"]))
            stt = cl.stats(np.concatenate(ts[n]["t"]))
            row[n] = dict(prefix_us=sp["kernel_ms"] * 1e3, prefix_median_us=sp["median_ms"] * 1e3,
                          prefix_noise=sp["noise_floor"], total_us=stt["kernel_ms"] * 1e3,
                          total_noise=stt["noise_floor"], prefix_tflops=flops / (sp["kernel_ms"] * 1e-3) / 1e12)
            cl.log(f"B={b} (C={c} G={g}) {n}: prefix {row[n]['prefix_us']:.1f} us ({row[n]['prefix_tflops']:.0f} "
                   f"TFLOP/s, noise {sp['noise_floor']:.3f}), prefix+suffix {row[n]['total_us']:.1f} us")
        res["batch"][str(b)] = row
        del gs
    # errors vs FP32 reference, B = 64, q x1 and x4
    q0 = inp["q"].clone()
    for qs in (1.0, 4.0):
        inp["q"].copy_((q0.float() * qs).to(torch.bfloat16))
        b = 64
        ref = cl.reference(inp, b)
        e = {}
        outs = {}
        for n, k in kerns.items():
            call = cl.Call(k, inp, b, cl.main_nb_sub_seq(b), cascade=True).capture()
            inp["out"].fill_(float("nan"))
            call.graph.replay()
            torch.cuda.synchronize()
            o = call.out().float().clone()
            outs[n] = o
            d = (o - ref).abs()
            e[n] = dict(max_abs=float(d.max()), mean_abs=float(d.mean()),
                        rel_l2=float((o - ref).norm() / ref.norm()), finite=bool(torch.isfinite(o).all()))
        for n in kerns:
            e[n]["mean_ratio_vs_bf16"] = e[n]["mean_abs"] / max(e["bf16"]["mean_abs"], 1e-30)
            e[n]["max_ratio_vs_bf16"] = e[n]["max_abs"] / max(e["bf16"]["max_abs"], 1e-30)
            e[n]["max_abs_vs_bf16_out"] = float((outs[n] - outs["bf16"]).abs().max())
        res["err"][f"q_x{qs:g}"] = e
        cl.log(f"err q x{qs:g}: {json.dumps(e)}")
    inp["q"].copy_(q0)
    res["output"] = {"class": "precision"}
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
