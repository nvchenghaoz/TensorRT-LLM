#!/usr/bin/env python3
# Idea 22: single-process A/B of the whole cascade op, main (mha.cu of the branch base commit, git archive) vs this
# tree, arms alternated per round in one process. Same cases and method as pdl_arm.py: B=64 / B=128 CUDA-graph replay
# and B=60 / B=124 eager, prefix + kernel_mha both launched with PDL, cold L2 (384 MiB read flush) + NaN poison before
# every call, 20 warmup per arm and case, then 3 rounds x 100 timed per arm and case, 10-90% trimmed mean.
import json
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "experiments/ascent/cascade"))
import cascade_lib as cl  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

BASE_COMMIT = "0674e6a60795a3ad9a1324cf64948749b44c721a"
CASES = [(64, True), (128, True), (60, False), (124, False)]


def main():
    cl.init_cuda()
    arms = {"main": cl.Kernels(cl.extract_tree(BASE_COMMIT)),
            "candidate": cl.Kernels(os.path.join(os.getcwd(), "cpp/kernels/xqa"))}
    inp = cl.build_inputs(128)
    fl = cl.xe.Flusher(384)
    fns = {}
    for an, k in arms.items():
        for b, graph in CASES:
            m = cl.main_nb_sub_seq(b)
            c = cl.Call(k, inp, b, m, cascade=True)

            def both(k=k, c=c, m=m, b=b):
                cl.xe.launch_ex(k.prefix, (c.c, c.g, cl.NB_K), (k.pcta, 1, 1), k.psmem, c.pfx_args.ptr, True)
                cl.xe.launch_ex(k.mha, (m, cl.NB_K, b), (128, 1, 2), k.smem, c.mha_args.ptr, True)

            both()
            torch.cuda.synchronize()
            if graph:
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    both()
                torch.cuda.synchronize()
                fns[(an, b)] = (g.replay, g)
            else:
                fns[(an, b)] = (both, None)

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

    for key in fns:
        time_fn(fns[key][0], 20)
    ts = dict((key, []) for key in fns)
    for rnd in range(3):
        for b, _ in (CASES if rnd % 2 == 0 else CASES[::-1]):
            for an in (("main", "candidate") if rnd % 2 == 0 else ("candidate", "main")):
                ts[(an, b)].append(time_fn(fns[(an, b)][0], 100))
    res = {"batch": {}, "output": {"class": "precision"}, "arms_alternated_in_one_process": True,
           "main_source_sha256": arms["main"].src_sha, "candidate_source_sha256": arms["candidate"].src_sha}
    for b, graph in CASES:
        row = {"mode": "cuda_graph_pdl" if graph else "eager_pdl"}
        for an in arms:
            s = cl.stats(np.concatenate(ts[(an, b)]))
            row[an] = dict(kernel_ms=s["kernel_ms"], median_ms=s["median_ms"], p90_ms=s["p90_ms"],
                           noise_floor=s["noise_floor"])
        row["gain"] = 1.0 - row["candidate"]["kernel_ms"] / row["main"]["kernel_ms"]
        res["batch"][str(b)] = row
        cl.log(f"B={b} {row['mode']}: main {row['main']['kernel_ms'] * 1e3:.2f} us, candidate "
               f"{row['candidate']['kernel_ms'] * 1e3:.2f} us, gain {row['gain'] * 100:.2f}%")
    res["mean_gain"] = float(np.mean([r["gain"] for r in res["batch"].values()]))
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
