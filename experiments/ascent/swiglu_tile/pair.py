"""Branch 11 paired microbenchmark (identical in base and candidate trees; run
inline with `python -c "$(cat pair.py)"`-style argv so the base tree needs no
copy). Qwen3.8-27B dense MLP chain with the tree's own GatedMLP dispatch,
L=4 distinct layers (~570 MB weights) + 256 MB L2 scrub before every timed run.
Decode rows (< 1024): BF16 input, CUDA graph, PDL on. Prefill rows: eager,
fused_gemma_add_rmsnorm_nvfp4 producer as in serving. One autotune pass per M.
10 warmup + 100 timed. kernel_ms = median per MLP layer.
"""
import json
import os
import statistics
import sys

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.getcwd(), "experiments", "ascent", "swiglu_epi"))
from mlp_common import EPS, GS_X, H, build_mlp, dev  # noqa: E402

from tensorrt_llm._torch.autotuner import autotune  # noqa: E402
from tensorrt_llm._torch.utils import Fp4QuantizedTensor  # noqa: E402

L = 4
MS = [int(v) for v in os.environ.get("BENCH_MS", "8,16,24,32,512,2079,4127").split(",")]
WARM, ITERS = 10, 100
mlps = [build_mlp(100 + i, 1.0) for i in range(L)]
print("permuted", [bool(getattr(m, "_swiglu_epi_permuted", False)) for m in mlps], flush=True)
gammas = [(torch.randn(H, device=dev) * 0.3).to(torch.bfloat16) for _ in range(L)]
gs = torch.tensor([GS_X], dtype=torch.float32, device=dev)
scrub = torch.empty(64 * 1024 * 1024, dtype=torch.float32, device=dev)
out = {}
for m in MS:
    gen = torch.Generator(device=dev).manual_seed(7 + m)
    x0 = (torch.randn((L, m, H), generator=gen, device=dev) * 0.7).to(torch.bfloat16)
    r0 = (torch.randn((L, m, H), generator=gen, device=dev) * 3.0).to(torch.bfloat16)
    xp, rp = x0.clone(), r0.clone()
    prefill = m >= 1024

    def layer(i):
        if prefill:
            q, sf = torch.ops.trtllm.fused_gemma_add_rmsnorm_nvfp4(xp[i], rp[i], gammas[i], EPS, gs)
            return mlps[i](Fp4QuantizedTensor(q, sf))
        return mlps[i](xp[i])

    with autotune():
        layer(0)
    torch.cuda.synchronize()
    if prefill:
        def runner():
            return [layer(i) for i in range(L)]
    else:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                [layer(i) for i in range(L)]
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            gouts = [layer(i) for i in range(L)]

        def runner(g=g):
            g.replay()
            return gouts
    ts = []
    chk = None
    for it in range(WARM + ITERS):
        xp.copy_(x0)
        rp.copy_(r0)
        scrub.zero_()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        o = runner()
        e1.record()
        e1.synchronize()
        if it >= WARM:
            ts.append(e0.elapsed_time(e1) * 1e3 / L)
        if it == 0:
            chk = torch.stack(o).view(torch.int16).sum(dtype=torch.int64).item()
    v = sorted(ts)
    n = len(v)
    med = statistics.median(v)
    p10, p90 = v[int(0.1 * (n - 1))], v[int(0.9 * (n - 1))]
    out[str(m)] = {"kernel_ms": med / 1e3, "noise_floor": round((p90 - p10) / med, 4),
                   "median_us_per_layer": round(med, 2), "p10_us": round(p10, 2), "p90_us": round(p90, 2),
                   "n": n, "out_checksum": chk}
    print("M", m, json.dumps(out[str(m)]), flush=True)
    torch.cuda.synchronize()
    if not prefill:
        del g, gouts
    torch.cuda.empty_cache()
print(json.dumps({"batch": out, "output": {"class": "exact"}}))
