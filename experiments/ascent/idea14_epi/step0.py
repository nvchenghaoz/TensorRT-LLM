"""Idea 14 STEP 0: at the served 2048^2 M, time main's QKV3 GEMM alone vs GEMM + qknorm_rope_fp8
(static mode 1 on V + mode 2), graph replay, interleaved. Also dumps the VEC kernel PTX (mode 2)
so the epilogue can replicate its arithmetic forms.
"""

import glob
import json
import os
import sys
import time

import torch

N = 3072
K = 3072


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def sf_rows(m):
    return (m + 127) // 128 * 128


def make_inputs(m, dev, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed + m)
    a = torch.randint(0, 256, (m, K // 2), generator=g, dtype=torch.uint8).to(dev)
    w = torch.randint(0, 256, (3 * N, K // 2), generator=g, dtype=torch.uint8).to(dev)
    sfa = torch.randint(0x30, 0x40, (sf_rows(m) * (K // 16),), generator=g, dtype=torch.uint8).to(dev)
    sfb = torch.randint(0x30, 0x40, (3 * N * (K // 16),), generator=g, dtype=torch.uint8).to(dev)
    alpha = torch.tensor([0.75, 1.25, 0.5], dtype=torch.float32, device=dev)
    qn = (1.0 + 0.3 * torch.randn(128, generator=g)).to(torch.bfloat16).to(dev)
    kn = (1.0 + 0.3 * torch.randn(128, generator=g)).to(torch.bfloat16).to(dev)
    ang = torch.rand(m, 64, generator=g) * 6.28
    cs = torch.cat([ang.cos(), ang.sin()], -1).float().contiguous().to(dev)
    pos = torch.arange(m, dtype=torch.int64, device=dev)
    return a, w, sfa, sfb, alpha, qn, kn, cs, pos


def main():
    dump = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp/idea14")
    os.makedirs(dump, exist_ok=True)
    os.environ.setdefault("CUTE_DSL_KEEP_PTX", "1")
    os.environ.setdefault("CUTE_DSL_DUMP_DIR", dump)
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import (
        flux3_sm120_qknorm_rope_fp8 as qr8,
    )
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_strided_gemm as strided

    dev = torch.device("cuda")
    strided.preload(dev)
    log("VEC", qr8.VEC, "STATIC", qr8.STATIC_QK, "L2D", qr8.L2DISCARD)
    R = 4
    ITERS = int(os.environ.get("S0_ITERS", "40"))
    res = {}
    for m in [int(x) for x in os.environ.get("S0_MS", "16464,32848").split(",")]:
        a, w, sfa, sfb, alpha, qn, kn, cs, pos = make_inputs(m, dev)
        out = torch.empty((m, 3 * N), dtype=torch.bfloat16, device=dev)
        planes = out.view(3, m, N)

        def gemm():
            strided.gemm_qkv3(a, w, sfa, sfb, alpha, out)

        def full(keep_v=True):
            gemm()
            return qr8.qknorm_rope_fp8(planes[0], planes[1], planes[2], qn, kn, cs, pos,
                                       128 ** -0.5, discard_qk=True, keep_v=keep_v)

        def qk_only():
            return qr8.qknorm_rope_fp8(planes[0], planes[1], planes[2], qn, kn, cs, pos,
                                       128 ** -0.5, discard_qk=False, keep_v=True)

        fns = {"gemm": gemm, "gemm+pass": full, "gemm+pass_nov": lambda: full(False),
               "pass_only": qk_only}
        for f in fns.values():
            f()
        torch.cuda.synchronize()
        graphs = {}
        for name, f in fns.items():
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(R):
                    f()
            graphs[name] = g
        for g in graphs.values():
            g.replay()
        torch.cuda.synchronize()
        times = {n: [] for n in graphs}
        for _ in range(ITERS):
            for name, g in graphs.items():
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                g.replay()
                e1.record()
                e1.synchronize()
                times[name].append(e0.elapsed_time(e1) * 1000.0 / R)
        row = {}
        for name, t in times.items():
            t = sorted(t)
            med = t[len(t) // 2]
            row[name] = {"us": round(med, 1),
                         "nf": round((t[int(0.9 * len(t)) - 1] - t[int(0.1 * len(t))]) / med, 4)}
        log(f"M{m}", json.dumps(row))
        res[m] = row
        del graphs
        torch.cuda.empty_cache()
    ptx = glob.glob(os.path.join(dump, "**", "*.ptx"), recursive=True)
    log("ptx files", ptx[:20])
    print(json.dumps({"step0": res, "ptx": ptx[:20]}))


if __name__ == "__main__":
    sys.exit(main())
