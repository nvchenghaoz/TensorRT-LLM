"""Decode chain microbenchmark for the post_attention gemma add+RMSNorm with a
fused NVFP4 epilogue (idea 23). No model load; synthetic tensors, real shapes.

Per layer (M decode rows), L distinct layers in one CUDA graph, every weight
used once per replay (L x ~105 MB >> 128 MB L2, so cold):
  producer nvfp4_gemm (N=5120, K=6144; writes x, as out_proj/o_proj does)
  -> post_attention norm (+ gate_up input quant)
  -> gate_up nvfp4_gemm (N=34816, K=5120)
Arms (captured once each, replayed alternately with rotating order; x and the
residual restored from pristine copies and a 256 MB L2 READ scrub before every
replay, all outside the timed window):
  main  : FlashInfer gemma_fused_add_rmsnorm + fp4_quantize   (main's ops)
  main2 : identical copy of main (A/A floor)
  fused : trtllm::flashinfer_gemma_fused_add_rmsnorm_nvfp4      (candidate)
Mode 'base' (any tree, e.g. the base arm of run_gpu_pair via python -c) runs
main/main2 only and reports main; mode 'cand' runs all three and reports fused.
Self-contained.
"""
import json
import statistics
import sys

import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.autotuner import autotune
from tensorrt_llm._torch.custom_ops import torch_custom_ops  # noqa: F401

H, N_UP, K_PROD = 5120, 34816, 6144
L = 16
EPS = 1e-6
WARM, ITERS = 10, 50
dev = "cuda"


def rand_w(n, k, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=dev,
                      generator=g)
    wsf = torch.randint(0x28, 0x38, (n * k // 16, ), dtype=torch.uint8,
                        device=dev, generator=g)
    return w, wsf


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "cand"
    ms = [int(a) for a in sys.argv[2:]] or [8, 16, 24, 32]
    if mode == "cand":
        from tensorrt_llm._torch.modules import \
            fused_gemma_add_rmsnorm_nvfp4  # noqa: F401
        arms = ("main", "main2", "fused")
    else:
        arms = ("main", "main2")
    print(f"mode={mode} arms={arms} ms={ms}", flush=True)
    alpha = torch.tensor([0.002], dtype=torch.float32, device=dev)
    gs_in = torch.tensor([20.0], dtype=torch.float32, device=dev)
    gs_x = torch.tensor([448.0 * 6.0 / 20.0], dtype=torch.float32, device=dev)
    wp = [rand_w(H, K_PROD, 1000 + i) for i in range(L)]
    wg = [rand_w(N_UP, H, 10 + i) for i in range(L)]
    gamma = [(torch.randn(H, device=dev) * 0.3).to(torch.bfloat16)
             for _ in range(L)]
    scrub = torch.empty(64 * 1024 * 1024, dtype=torch.float32, device=dev)
    scrub.normal_()
    sink = torch.zeros(1, dtype=torch.float32, device=dev)
    results = {}
    for m in ms:
        gen = torch.Generator(device=dev).manual_seed(7 + m)
        a = (torch.randn((L, m, K_PROD), generator=gen, device=dev) *
             0.5).to(torch.bfloat16)
        aq = []
        for i in range(L):
            q, sf = torch.ops.trtllm.fp4_quantize(a[i].contiguous(), gs_in, 16,
                                                  False)
            aq.append((q, sf))
        r0 = (torch.randn((L, m, H), generator=gen, device=dev) *
              3.0).to(torch.bfloat16)
        rp = r0.clone()

        def layer(i, arm):
            x = torch.ops.trtllm.nvfp4_gemm(aq[i][0], wp[i][0], aq[i][1],
                                            wp[i][1], alpha, torch.bfloat16)
            r = rp[i]
            if arm == "fused":
                q, sf = torch.ops.trtllm.flashinfer_gemma_fused_add_rmsnorm_nvfp4(
                    x, r, gamma[i], EPS, gs_x)
            else:
                torch.ops.trtllm.flashinfer_gemma_fused_add_rmsnorm(
                    x, r, gamma[i], EPS)
                q, sf = torch.ops.trtllm.fp4_quantize(x, gs_x, 16, False)
            return torch.ops.trtllm.nvfp4_gemm(q, wg[i][0], sf, wg[i][1],
                                               alpha, torch.bfloat16)

        with autotune():
            rp.copy_(r0)
            layer(0, "main")
        torch.cuda.synchronize()
        print(f"M={m} autotuned", flush=True)

        graphs = {}
        for arm in arms:

            def body(arm=arm):
                return [layer(i, arm) for i in range(L)]

            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    rp.copy_(r0)
                    body()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                outs = body()
            torch.cuda.synchronize()
            graphs[arm] = (g, outs)
        print(f"M={m} captured", flush=True)

        ts = {a_: [] for a_ in arms}
        na = len(arms)
        for it in range(WARM + ITERS):
            order = arms[it % na:] + arms[:it % na]
            for arm in order:
                rp.copy_(r0)
                sink += scrub.sum()  # 256 MB read: cold L2
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                graphs[arm][0].replay()
                e1.record()
                e1.synchronize()
                if it >= WARM:
                    ts[arm].append(e0.elapsed_time(e1) * 1e3 / L)
        last = {}
        for arm in arms:
            rp.copy_(r0)
            graphs[arm][0].replay()
            torch.cuda.synchronize()
            last[arm] = (torch.stack(graphs[arm][1]).clone(), rp.clone())
        same = all(
            torch.equal(last[a_][0], last["main"][0])
            and torch.equal(last[a_][1], last["main"][1]) for a_ in arms)
        row = {}
        for arm in arms:
            v = sorted(ts[arm])
            n = len(v)
            med = statistics.median(v)
            p10, p90 = v[int(0.1 * (n - 1))], v[int(0.9 * (n - 1))]
            row[arm] = {"median_us_per_layer": round(med, 3),
                        "p10_us": round(p10, 3), "p90_us": round(p90, 3),
                        "noise_floor": round((p90 - p10) / med, 4), "n": n}
        row["gate_up_out_and_residual_bitwise_all_arms"] = same
        row["aa_main2_minus_main_us"] = round(
            row["main2"]["median_us_per_layer"] -
            row["main"]["median_us_per_layer"], 3)
        if "fused" in arms:
            row["fused_minus_main_us"] = round(
                row["fused"]["median_us_per_layer"] -
                row["main"]["median_us_per_layer"], 3)
        results[str(m)] = row
        print("M", m, json.dumps(row), flush=True)
        del graphs
        torch.cuda.empty_cache()
    rep = "fused" if mode == "cand" else "main"
    print(json.dumps({
        "batch": {k: {"kernel_ms": v[rep]["median_us_per_layer"] / 1e3,
                      "noise_floor": max(v[rep]["noise_floor"],
                                         v["main"]["noise_floor"])}
                  for k, v in results.items()},
        "output": {"class": "exact"},
        "reported_arm": rep,
        "unit_note": "kernel_ms = median per-layer time of the chain "
                     "producer GEMM -> norm(+quant) -> gate_up GEMM",
        "chain": results,
    }))


if __name__ == "__main__":
    main()
