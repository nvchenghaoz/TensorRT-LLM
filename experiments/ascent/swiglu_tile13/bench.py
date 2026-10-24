"""Branch 13: SwiGLU-epilogue gate_up GEMM (trtllm::nvfp4_gate_up_swiglu_nvfp4)
at decode rows, op level, synthetic Qwen3.8-27B shapes (N=34816, K=5120).

L distinct weight copies (94.7 MB each, L=4 -> 380 MB) chained in one CUDA graph
(PDL on), plus a 256 MB L2 scrub before every timed replay: every call reads a
cold weight. us/call = replay time / L.

MODE=diag : arms = BENCH_TILES (e.g. "1,2"), order rotated each iteration, in
            one process; prints us/call, % of byte floor, bitwise agreement.
MODE=pair : the tree's own dispatch tile (GatedMLP._swiglu_epi_tile(m)); prints
            the evidence JSON.
"""
import json
import os
import statistics

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
import torch  # noqa: E402

import tensorrt_llm  # noqa: E402,F401
from tensorrt_llm._torch.custom_ops import torch_custom_ops  # noqa: E402,F401
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP  # noqa: E402

dev = "cuda"
N, K = 34816, 5120
L = int(os.environ.get("BENCH_L", "4"))
MS = [int(v) for v in os.environ.get("BENCH_MS", "48,64,96,128").split(",")]
MODE = os.environ.get("MODE", "pair")
TILES = [int(v) for v in os.environ.get("BENCH_TILES", "1,2").split(",")]
WARM, ITERS = int(os.environ.get("BENCH_WARM", "10")), int(os.environ.get("BENCH_ITERS", "60"))
DRAM = 1508.57e9
W_BYTES = N * K // 2 + N * K // 16


def make_weights(seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    w = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev, generator=g)
    sf2 = torch.randint(0x28, 0x38, (N, K // 16), dtype=torch.uint8, device=dev, generator=g)
    sf = torch.ops.trtllm.block_scale_interleave(sf2)
    return w, sf


def make_act(m, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    a = torch.randint(0, 256, (m, K // 2), dtype=torch.uint8, device=dev, generator=g)
    mp = (m + 127) // 128 * 128
    sf2 = torch.randint(0x30, 0x3c, (mp, K // 16), dtype=torch.uint8, device=dev, generator=g)
    sf2[m:] = 0
    sf = torch.ops.trtllm.block_scale_interleave(sf2)
    return a, sf


def main():
    ws = [make_weights(100 + i) for i in range(L)]
    alpha = torch.tensor([0.0005], dtype=torch.float32, device=dev)
    osc = torch.tensor([1.0], dtype=torch.float32, device=dev)
    scrub = torch.empty(64 * 1024 * 1024, dtype=torch.float32, device=dev)
    print("floor_us", round(W_BYTES / DRAM * 1e6, 2), flush=True)
    res = {}
    for m in MS:
        acts = [make_act(m, 7 + 31 * m + i) for i in range(L)]
        arms = TILES if MODE == "diag" else [GatedMLP._swiglu_epi_tile(m)]

        def chain(t):
            return [torch.ops.trtllm.nvfp4_gate_up_swiglu_nvfp4(
                acts[i][0], ws[i][0], acts[i][1], ws[i][1], alpha, osc, t) for i in range(L)]

        graphs, gouts, refs = {}, {}, {}
        for t in arms:
            eager = chain(t)
            torch.cuda.synchronize()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    chain(t)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                go = chain(t)
            g.replay()
            torch.cuda.synchronize()
            for (f, sfo), (fe, sfe) in zip(go, eager):
                assert torch.equal(f, fe) and torch.equal(sfo, sfe), "graph != eager"
            graphs[t], gouts[t] = g, go
            refs[t] = [(f.clone(), s_.clone()) for f, s_ in go]
        t0 = arms[0]
        exact = {str(t): all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
                             for a, b in zip(refs[t], refs[t0])) for t in arms}
        ts = {t: [] for t in arms}
        for it in range(WARM + ITERS):
            order = arms[it % len(arms):] + arms[:it % len(arms)]
            for t in order:
                scrub.zero_()
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                graphs[t].replay()
                e1.record()
                e1.synchronize()
                if it >= WARM:
                    ts[t].append(e0.elapsed_time(e1) * 1e3 / L)
        stable = {str(t): all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
                              for a, b in zip(gouts[t], refs[t])) for t in arms}
        row = {}
        for t in arms:
            v = sorted(ts[t])
            n = len(v)
            med = statistics.median(v)
            p10, p90 = v[int(0.1 * (n - 1))], v[int(0.9 * (n - 1))]
            row[str(t)] = {"med_us": round(med, 2), "p10": round(p10, 2), "p90": round(p90, 2),
                           "pct_floor": round(100 * W_BYTES / DRAM * 1e6 / med, 1),
                           "noise_floor": round((p90 - p10) / med, 4)}
        row["exact_vs_" + str(t0)] = exact
        row["stable_after_timing"] = stable
        res[str(m)] = row
        print("M", m, json.dumps(row), flush=True)
        del graphs, gouts
        torch.cuda.empty_cache()
    if MODE == "pair":
        batch = {}
        for m, row in res.items():
            t = str(GatedMLP._swiglu_epi_tile(int(m)))
            batch[m] = {"kernel_ms": row[t]["med_us"] / 1e3, "noise_floor": row[t]["noise_floor"],
                        "tile": int(t), "p10_us": row[t]["p10"], "p90_us": row[t]["p90"],
                        "pct_floor": row[t]["pct_floor"]}
        print(json.dumps({"batch": batch, "output": {"class": "exact"}}))
    else:
        print(json.dumps({"diag": res}))


if __name__ == "__main__":
    main()
