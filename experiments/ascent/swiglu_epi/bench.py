"""Branch 128 paired microbenchmark: Qwen3.8-27B dense MLP layer chain in one
CUDA graph, identical script in the base and candidate trees.

Per layer (L distinct layers, ~535 MB of weights, > 128 MB L2):
  prefill M >= 17: fused_gemma_add_rmsnorm_nvfp4 (real producer) -> GatedMLP
                   (gate_up -> SwiGLU+quant -> down_proj, the real consumer)
  decode  M <= 16: BF16 input -> GatedMLP (gate_up input quant inside)
Residuals restored and a 256 MB L2 scrub before every replay, outside the
timed window. One autotune pass per M. 10 warmup + 50 timed replays.
Reports us per layer (median, p10/p90).
"""
import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlp_common import EPS, GS_X, H, build_mlp, dev, tree_info  # noqa: E402

from tensorrt_llm._torch.autotuner import autotune  # noqa: E402
from tensorrt_llm._torch.utils import Fp4QuantizedTensor  # noqa: E402

L = int(os.environ.get("BENCH_L", "4"))
MS = [int(v) for v in os.environ.get("BENCH_MS", "1,2,4,2048,4096,8192").split(",")]
WARM, ITERS = 10, 50


def main():
    print("TREE", json.dumps(tree_info()), flush=True)
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
        prefill = m >= 17

        def layer(i):
            if prefill:
                q, sf = torch.ops.trtllm.fused_gemma_add_rmsnorm_nvfp4(xp[i], rp[i], gammas[i], EPS, gs)
                return mlps[i](Fp4QuantizedTensor(q, sf))
            return mlps[i](xp[i])

        with autotune():
            xp.copy_(x0)
            rp.copy_(r0)
            layer(0)
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                xp.copy_(x0)
                rp.copy_(r0)
                [layer(i) for i in range(L)]
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            outs = [layer(i) for i in range(L)]
        torch.cuda.synchronize()
        ts = []
        for it in range(WARM + ITERS):
            xp.copy_(x0)
            rp.copy_(r0)
            scrub.zero_()
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            g.replay()
            e1.record()
            e1.synchronize()
            if it >= WARM:
                ts.append(e0.elapsed_time(e1) * 1e3 / L)
        v = sorted(ts)
        n = len(v)
        med = statistics.median(v)
        p10, p90 = v[int(0.1 * (n - 1))], v[int(0.9 * (n - 1))]
        h = torch.stack(outs).view(torch.int16).sum(dtype=torch.int64).item()
        out[str(m)] = {"kernel_ms": med / 1e3, "noise_floor": round((p90 - p10) / med, 4),
                       "median_us_per_layer": round(med, 3), "p10_us": round(p10, 3), "p90_us": round(p90, 3),
                       "n": n, "out_checksum": h}
        print("M", m, json.dumps(out[str(m)]), flush=True)
        del g, outs
        torch.cuda.empty_cache()
    print(json.dumps({"batch": out, "output": {"class": "exact"}}))


if __name__ == "__main__":
    main()
