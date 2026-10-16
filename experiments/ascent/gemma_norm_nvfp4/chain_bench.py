"""Branch 57 (idea 129) one-tree chain guard + SwiGLU PDL on/off timing.

Per decode layer (M = 1/2/4 rows), 12 distinct layers in one CUDA graph,
every weight used once per replay (12 x 134 MB >> 128 MB L2, so cold):
  norm+quant (post_attention_layernorm + gate_up input quant)
  -> gate_up nvfp4_gemm (34816 x 5120) -> SwiGLU+NVFP4 -> down nvfp4_gemm
     (5120 x 17408).

Four arms, captured once each and replayed alternately in the same process
(rotating start order), residual and x restored from pristine copies and a
256 MB L2 scrub before every replay (all outside the timed window):
  main        : FlashInfer gemma add+RMSNorm + fp4_quantize, SwiGLU PDL off
  fused       : fused Triton norm+quant,                      SwiGLU PDL off
  pdl         : FlashInfer + fp4_quantize,                    SwiGLU PDL on
  cand        : fused Triton norm+quant,                      SwiGLU PDL on
'main' is main's code path (SwiGLU PDL off == main's kernel; the norm ops are
main's ops); 'cand' is the branch default. Same autotuned GEMM tactics in all
arms (one autotune pass, shared in-process cache).
"""
import json
import statistics
import subprocess
import sys

import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.autotuner import autotune
from tensorrt_llm._torch.custom_ops import torch_custom_ops  # noqa: F401
from tensorrt_llm._torch.modules import fused_gemma_add_rmsnorm_nvfp4  # noqa
from tensorrt_llm._torch.modules import fused_silu_mul_nvfp4 as fs

H, I = 5120, 17408
L = 12
EPS = 1e-6
WARM, ITERS = 10, 50
dev = "cuda"
ARMS = ("main", "fused", "pdl", "cand")


def rand_w(n, k, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=dev,
                      generator=g)
    wsf = torch.randint(0x28, 0x38, (n * k // 16, ), dtype=torch.uint8,
                        device=dev, generator=g)
    return w, wsf


def main():
    c = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                       text=True).stdout.strip()
    print(f"TREE_COMMIT=a4c9a32fb01f8bf3e18a0805092222a51bacb0e8 head={c}",
          flush=True)
    ms = [int(a) for a in sys.argv[1:]] or [1, 2, 4]
    alpha = torch.tensor([0.002], dtype=torch.float32, device=dev)
    gs_x = torch.tensor([448.0 * 6.0 / 20.0], dtype=torch.float32,
                        device=dev)
    gs_h = torch.tensor([30.0], dtype=torch.float32, device=dev)
    wg = [rand_w(2 * I, H, 10 + i) for i in range(L)]
    wd = [rand_w(H, I, 100 + i) for i in range(L)]
    gamma = [(torch.randn(H, device=dev) * 0.3).to(torch.bfloat16)
             for _ in range(L)]
    scrub = torch.empty(64 * 1024 * 1024, dtype=torch.float32, device=dev)
    orig_pdl = fs._PDL_ENABLED
    results = {}
    for m in ms:
        gen = torch.Generator(device=dev).manual_seed(7 + m)
        x0 = (torch.randn((L, m, H), generator=gen, device=dev) *
              0.7).to(torch.bfloat16)
        r0 = (torch.randn((L, m, H), generator=gen, device=dev) *
              3.0).to(torch.bfloat16)
        xp, rp = x0.clone(), r0.clone()

        def layer(i, fused):
            x, r = xp[i], rp[i]
            if fused:
                q, sf = torch.ops.trtllm.fused_gemma_add_rmsnorm_nvfp4(
                    x, r, gamma[i], EPS, gs_x)
            else:
                torch.ops.trtllm.flashinfer_gemma_fused_add_rmsnorm(
                    x, r, gamma[i], EPS)
                q, sf = torch.ops.trtllm.fp4_quantize(x, gs_x, 16, False)
            h = torch.ops.trtllm.nvfp4_gemm(q, wg[i][0], sf, wg[i][1], alpha,
                                            torch.bfloat16)
            hq, hsf = torch.ops.trtllm.fused_silu_mul_nvfp4(h, gs_h)
            return torch.ops.trtllm.nvfp4_gemm(hq, wd[i][0], hsf, wd[i][1],
                                               alpha, torch.bfloat16)

        # One autotune pass (shared tactic cache for every arm).
        with autotune():
            xp.copy_(x0)
            rp.copy_(r0)
            layer(0, False)
        torch.cuda.synchronize()

        graphs = {}
        for arm in ARMS:
            fused = arm in ("fused", "cand")
            fs._PDL_ENABLED = arm in ("pdl", "cand")

            def body(fused=fused):
                return [layer(i, fused) for i in range(L)]

            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    xp.copy_(x0)
                    rp.copy_(r0)
                    body()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                outs = body()
            torch.cuda.synchronize()
            graphs[arm] = (g, outs)
        fs._PDL_ENABLED = orig_pdl

        ts = {a: [] for a in ARMS}
        for it in range(WARM + ITERS):
            order = ARMS[it % 4:] + ARMS[:it % 4]
            for arm in order:
                xp.copy_(x0)
                rp.copy_(r0)
                scrub.zero_()
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                graphs[arm][0].replay()
                e1.record()
                e1.synchronize()
                if it >= WARM:
                    ts[arm].append(e0.elapsed_time(e1) * 1e3 / L)
        # Outputs of every arm must agree bitwise (same math).
        last = {}
        for arm in ARMS:
            xp.copy_(x0)
            rp.copy_(r0)
            graphs[arm][0].replay()
            torch.cuda.synchronize()
            last[arm] = (torch.stack(graphs[arm][1]).clone(), rp.clone())
        same = all(
            torch.equal(last[a][0], last["main"][0])
            and torch.equal(last[a][1], last["main"][1]) for a in ARMS)
        row = {}
        for arm in ARMS:
            v = sorted(ts[arm])
            n = len(v)
            med = statistics.median(v)
            row[arm] = {
                "median_us_per_layer": round(med, 3),
                "p10_us": round(v[int(0.1 * (n - 1))], 3),
                "p90_us": round(v[int(0.9 * (n - 1))], 3),
                "noise_floor": round(
                    (v[int(0.9 * (n - 1))] - v[int(0.1 * (n - 1))]) / med,
                    4),
                "n": n,
            }
        row["outputs_bitwise_equal_all_arms"] = same
        for a in ("fused", "pdl", "cand"):
            row[f"{a}_minus_main_us_per_layer"] = round(
                row[a]["median_us_per_layer"] -
                row["main"]["median_us_per_layer"], 3)
        row["silu_pdl_on_minus_off_us_with_fused_norm"] = round(
            row["cand"]["median_us_per_layer"] -
            row["fused"]["median_us_per_layer"], 3)
        results[str(m)] = row
        print("M", m, json.dumps(row), flush=True)
        del graphs
        torch.cuda.empty_cache()
    print(
        json.dumps({
            "batch": {
                k: {
                    "kernel_ms": v["cand"]["median_us_per_layer"] / 1e3,
                    "noise_floor": v["cand"]["noise_floor"],
                }
                for k, v in results.items()
            },
            "output": {"class": "exact"},
            "chain": results,
        }))


if __name__ == "__main__":
    main()
