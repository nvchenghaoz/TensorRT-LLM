"""Paired microbenchmark for trtllm::flashinfer_mm_mxfp8 (serving's SM120 MXFP8 linear GEMM).

One "layer" = the five CSA2 MXFP8 projections at DeepSeek-V4.1 TP4 per-rank shapes
(wq_a [1280,5120], wkv [512,5120], wq_b [8192,1280], o_b_proj [5120,2048],
index_wq_b [4096,1280]). Weights rotate over L layer copies (>128 MB L2) inside one
CUDA graph, so every GEMM streams its weight cold from DRAM, as in decode.
Before timing, every shape/M is run once under flashinfer.autotune() like the
serving warmup (flashinfer_mxfp8_autotune), so the baseline uses the tuned CUTLASS
tactic serving uses. Reports kernel_ms = time per layer (5 GEMMs) per M.
"""
import json
import math
import os

import torch

import tensorrt_llm  # noqa: F401
from flashinfer import autotune

SHAPES = [(1280, 5120), (512, 5120), (8192, 1280), (5120, 2048), (4096, 1280)]
MS = [int(m) for m in os.environ.get("MS", "1,2,4,8,2048").split(",")]
REPLAYS = int(os.environ.get("REPLAYS", "30"))
ROT_BYTES = 300 * 1024 * 1024
dev = torch.device("cuda")
op = torch.ops.trtllm.flashinfer_mm_mxfp8

torch.manual_seed(0)
layer_bytes = sum(n * k * (1 + 1 / 32) for n, k in SHAPES)
L = max(2, math.ceil(ROT_BYTES / layer_bytes))
weights = []
for n, k in SHAPES:
    w = torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.05
    wq, wsf = torch.ops.trtllm.mxfp8_quantize(w, True)
    weights.append([(wq, wsf)] + [(wq.clone(), wsf.clone()) for _ in range(L - 1)])
print(f"layers={L} layer_MB={layer_bytes/1e6:.1f}", flush=True)

res = {}
for M in MS:
    torch.manual_seed(100 + M)
    acts = []
    for n, k in SHAPES:
        x = torch.randn(M, k, device=dev, dtype=torch.bfloat16)
        acts.append(torch.ops.trtllm.mxfp8_quantize(x, True))
    # serving-like warmup: autotune pass, then eager call
    with autotune():
        for (aq, asf), ws in zip(acts, weights):
            op(aq, asf, ws[0][0], ws[0][1], torch.bfloat16)
    for (aq, asf), ws in zip(acts, weights):
        op(aq, asf, ws[0][0], ws[0][1], torch.bfloat16)
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for (aq, asf), ws in zip(acts, weights):
            op(aq, asf, ws[1][0], ws[1][1], torch.bfloat16)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for li in range(L):
            for (aq, asf), ws in zip(acts, weights):
                op(aq, asf, ws[li][0], ws[li][1], torch.bfloat16)
    for _ in range(3):
        g.replay()
    ts = []
    for _ in range(REPLAYS):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        e1.synchronize()
        ts.append(e0.elapsed_time(e1) / L)
    ts.sort()
    med = ts[len(ts) // 2]
    p10 = ts[int(0.1 * len(ts))]
    p90 = ts[min(len(ts) - 1, int(0.9 * len(ts)))]
    res[str(M)] = {"kernel_ms": round(med, 5), "noise_floor": round((p90 - p10) / med, 4)}
    print(f"M={M} per-layer {med*1000:.2f} us noise {(p90-p10)/med:.4f}", flush=True)
    del g

print(json.dumps({"batch": res, "output": {"class": "exact"}}))
