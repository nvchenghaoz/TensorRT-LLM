"""Branch 54 / idea 84 STEP 0: where the fused QKV epilogue's ~0.10 ms (16K) / ~0.23 ms (32848)
over the no-op floor goes. One process, one GPU, synthetic NVFP4 inputs at the served shapes
(N = K = 3072), CUDA-graph replays of R back-to-back GEMM launches, arms interleaved in a seeded
random order every iteration.

Arms (GEMM launch only, v_quant excluded unless named):
  fused / fused_aa  the served fused GEMM (two graphs of the same call: the A/A arm)
  op                the served op (fused GEMM + v_quant, keep_v=False)
  noop              FLUX3_EPI_DIAG_NOOP (mainloop floor)
  nocs              no cos_sin loads (row-dependent register values)
  nox               no cross-warp smem exchange (2 bar.sync gone)
  novst             V plane computed, not stored
  noqkst            Q/K E4M3 computed, not stored
  nosf              no per-tile static factor (constant)
Env: S0_MS, S0_R, S0_ITERS, S0_ARMS. Last stdout line: JSON.
"""

import json
import os
import random
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "idea14_epi"))
from op import SM, N, make_inputs  # noqa: E402
from b54_variants import build_all  # noqa: E402


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def stats(t):
    t = sorted(t)
    med = t[len(t) // 2]
    return med, (t[int(0.9 * len(t)) - 1] - t[int(0.1 * len(t))]) / med


def main():
    from tensorrt_llm._torch.visual_gen.attention.backends.fp8_sm12x import _MIN_SCALING_FACTOR
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_qkv_epi_fp8 as epi
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_strided_gemm as strided

    dev = torch.device("cuda")
    strided.preload(dev)
    epi.preload(dev)
    mod = epi.module()
    log("served stages", int(mod.qkv3_epi_stages()), "smem", int(mod.qkv3_epi_smem()))
    arms_env = os.environ.get("S0_ARMS")
    names = arms_env.split(",") if arms_env else None
    mods = build_all(names, log)
    for n, m in mods.items():
        log(f"{n}: stages {int(m.qkv3_epi_stages())} smem {int(m.qkv3_epi_smem())}")
    ms = [int(x) for x in os.environ.get("S0_MS", "16384,16464,32848").split(",")]
    R = int(os.environ.get("S0_R", "4"))
    ITERS = int(os.environ.get("S0_ITERS", "40"))
    rng = random.Random(54)
    minsf = float(_MIN_SCALING_FACTOR)
    result = {}
    for m in ms:
        a, w, sfa, sfb, alpha, qn, kn, cs, pos = make_inputs(m, dev)
        qkv8 = torch.empty((3, m, N), dtype=torch.uint8, device=dev)
        gout = torch.empty((m, 3 * N), dtype=torch.bfloat16, device=dev)
        v = gout.view(3, m, N)[2]
        n_part = ((m + 127) // 128) * (N // 128)
        parts = torch.zeros(n_part + 8, dtype=torch.float32, device=dev)
        ws = strided.workspace(dev, 0)

        def gemm(md):
            return lambda: md.qkv3_epi_fp8(a, w, sfa, sfb, alpha, qn, kn, cs, qkv8, v, parts, ws,
                                           minsf)

        fns = {"fused": gemm(mod), "fused_aa": gemm(mod),
               "op": lambda: epi.qkv_rope_fp8(a, w, sfa, sfb, alpha, qn, kn, cs, SM, False)}
        for n, md in mods.items():
            fns[n] = gemm(md)
        for f in fns.values():
            f()
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
        order = list(graphs)
        for it in range(ITERS):
            rng.shuffle(order)
            for name in order:
                g = graphs[name]
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                g.replay()
                e1.record()
                e1.synchronize()
                times[name].append(e0.elapsed_time(e1) / R)
        row = {}
        for name, t in times.items():
            med, nf = stats(t)
            row[name] = {"ms": round(med, 4), "nf": round(nf, 4)}
        base = row["fused"]["ms"]
        if "noop" in row:
            row["epi_over_noop_ms"] = round(base - row["noop"]["ms"], 4)
        for n in ("nocs", "nox", "novst", "noqkst", "nosf"):
            if n in row:
                row[f"save_{n}_ms"] = round(base - row[n]["ms"], 4)
        log(f"M{m}", json.dumps(row))
        result[str(m)] = row
        del graphs
        torch.cuda.empty_cache()
    print(json.dumps({"step0": result}))


if __name__ == "__main__":
    sys.exit(main())
