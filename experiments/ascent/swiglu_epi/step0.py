"""STEP 0 (branch 128 / idea 226): fused gate_up + SwiGLU + NVFP4 epilogue
prototype vs main's gate_up nvfp4_gemm + fused_silu_mul_nvfp4.

1. Bitwise: FP4 bytes + swizzled SF (padding included) of the fused kernel vs
   main's stock two-kernel output on the same inputs, M = 17/2048/4096/8192,
   random and outlier rows.
2. Real-shape chain in one CUDA graph per arm, L distinct layers:
   fused_gemma_add_rmsnorm_nvfp4 -> gate_up -> (silu) -> down nvfp4_gemm.
   Arms replayed alternately; residual restored outside the timed window.
   Also the standalone gate_up(+silu) vs fused kernel time.
"""
import json
import os
import statistics
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tensorrt_llm  # noqa: F401,E402
from tensorrt_llm._torch.autotuner import autotune  # noqa: E402
from tensorrt_llm._torch.custom_ops import torch_custom_ops  # noqa: F401,E402
from tensorrt_llm._torch.modules import fused_gemma_add_rmsnorm_nvfp4  # noqa: F401,E402
from tensorrt_llm._torch.modules import fused_silu_mul_nvfp4  # noqa: F401,E402

from ext import load_ext, make_perm  # noqa: E402

H, I = 5120, 17408
N = 2 * I
EPS = 1e-6
dev = "cuda"
L = int(os.environ.get("STEP0_L", "4"))
WARM, ITERS = 10, 50


def rand_w(n, k, seed, perm=None):
    g = torch.Generator(device=dev).manual_seed(seed)
    w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=dev, generator=g)
    sf2 = torch.randint(0x28, 0x38, (n, k // 16), dtype=torch.uint8, device=dev, generator=g)
    sw = torch.ops.trtllm.block_scale_interleave(sf2)
    if perm is None:
        return w, sw
    return w, sw, w[perm].contiguous(), torch.ops.trtllm.block_scale_interleave(sf2[perm].contiguous())


def stats(v):
    v = sorted(v)
    n = len(v)
    med = statistics.median(v)
    p10, p90 = v[int(0.1 * (n - 1))], v[int(0.9 * (n - 1))]
    return {"median_us": round(med, 3), "p10_us": round(p10, 3), "p90_us": round(p90, 3),
            "noise_floor": round((p90 - p10) / med, 4), "n": n}


def main():
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    print("TREE_COMMIT", head, flush=True)
    ext = load_ext(check_layout=os.environ.get("STEP0_CHECK_LAYOUT", "0") == "1")
    perm = make_perm(N, dev)
    alpha = torch.tensor([0.0005], dtype=torch.float32, device=dev)
    gs_x = torch.tensor([448.0 * 6.0 / 12.0], dtype=torch.float32, device=dev)
    wg = [rand_w(N, H, 10 + i, perm) for i in range(L)]
    wd = [rand_w(H, I, 100 + i) for i in range(L)]
    gamma = [(torch.randn(H, device=dev) * 0.3).to(torch.bfloat16) for _ in range(L)]
    out = {"bitwise": {}, "chain": {}}

    # ---------------- calibrate down's input scale on the stock path ----------
    x = (torch.randn((256, H), device=dev) * 0.7).to(torch.bfloat16)
    r = (torch.randn((256, H), device=dev) * 3.0).to(torch.bfloat16)
    q, sf = torch.ops.trtllm.fused_gemma_add_rmsnorm_nvfp4(x, r, gamma[0], EPS, gs_x)
    h = torch.ops.trtllm.nvfp4_gemm(q, wg[0][0], sf, wg[0][1], alpha, torch.bfloat16)
    a, b = h[:, :I].float(), h[:, I:].float()
    amax = (a * torch.sigmoid(a) * b).abs().max().item()
    gs_h = torch.tensor([448.0 * 6.0 / amax], dtype=torch.float32, device=dev)
    print("h amax", h.abs().max().item(), "swiglu amax", amax, "gs_h", gs_h.item(), flush=True)

    # ---------------- bitwise check --------------------------------------------
    all_ok = True
    for m in [int(v) for v in os.environ.get("STEP0_BITWISE_MS", "17,2048,4096,8192").split(",") if v]:
        for kind in ("random", "outlier"):
            gen = torch.Generator(device=dev).manual_seed(1000 + m)
            xx = (torch.randn((m, H), generator=gen, device=dev) * 0.7)
            rr = (torch.randn((m, H), generator=gen, device=dev) * 3.0)
            if kind == "outlier":
                idx = torch.randint(0, m, (max(1, m // 64), ), generator=gen, device=dev)
                rr[idx] *= 40.0
                xx[:, ::97] *= 25.0
            xx, rr = xx.to(torch.bfloat16), rr.to(torch.bfloat16)
            q, sfa = torch.ops.trtllm.fused_gemma_add_rmsnorm_nvfp4(xx, rr.clone(), gamma[0], EPS, gs_x)
            with autotune():
                h = torch.ops.trtllm.nvfp4_gemm(q, wg[0][0], sfa, wg[0][1], alpha, torch.bfloat16)
            h = torch.ops.trtllm.nvfp4_gemm(q, wg[0][0], sfa, wg[0][1], alpha, torch.bfloat16)
            bq, bsf = torch.ops.trtllm.fused_silu_mul_nvfp4(h, gs_h)
            res = {}
            for tile in (0, 1):
                fq, fsf = ext.fused_gate_up_swiglu_nvfp4(q, wg[0][2], sfa, wg[0][3], alpha, gs_h, tile, True)
                torch.cuda.synchronize()
                eq_q = torch.equal(fq, bq)
                eq_sf = torch.equal(fsf, bsf)
                nd = int((fq != bq).sum().item())
                nsf = int((fsf != bsf).sum().item())
                res[f"tile{tile}"] = {"fp4_equal": eq_q, "sf_equal": eq_sf, "fp4_diff_bytes": nd,
                                      "sf_diff_bytes": nsf}
                all_ok &= eq_q and eq_sf
                if not (eq_q and eq_sf) and nd > 0:
                    w = (fq != bq).nonzero()[:4].tolist()
                    res[f"tile{tile}"]["first_diffs"] = [[i, j, int(fq[i, j]), int(bq[i, j])] for i, j in w]
            out["bitwise"][f"{m}_{kind}"] = res
            print("BITWISE", m, kind, json.dumps(res), flush=True)
    out["bitwise_all_equal"] = all_ok

    # ---------------- chain timing ---------------------------------------------
    scrub = torch.empty(64 * 1024 * 1024, dtype=torch.float32, device=dev)
    for m in [int(v) for v in os.environ.get("STEP0_MS", "2048,4096,8192").split(",")]:
        t0 = time.time()
        gen = torch.Generator(device=dev).manual_seed(7 + m)
        x0 = (torch.randn((L, m, H), generator=gen, device=dev) * 0.7).to(torch.bfloat16)
        r0 = (torch.randn((L, m, H), generator=gen, device=dev) * 3.0).to(torch.bfloat16)
        xp, rp = x0.clone(), r0.clone()
        # Pre-quantized gate_up inputs for the standalone arms.
        pre = [torch.ops.trtllm.fused_gemma_add_rmsnorm_nvfp4(x0[i], r0[i].clone(), gamma[i], EPS, gs_x)
               for i in range(L)]

        def layer(i, arm):
            if arm in ("base_gemm", "fused_gemm", "base_gemm_silu", "fused_gemm_t1"):
                q, sfa = pre[i]
            else:
                q, sfa = torch.ops.trtllm.fused_gemma_add_rmsnorm_nvfp4(xp[i], rp[i], gamma[i], EPS, gs_x)
            if arm in ("base", "base_gemm", "base_gemm_silu"):
                h = torch.ops.trtllm.nvfp4_gemm(q, wg[i][0], sfa, wg[i][1], alpha, torch.bfloat16)
                if arm == "base_gemm":
                    return h
                hq, hsf = torch.ops.trtllm.fused_silu_mul_nvfp4(h, gs_h)
                if arm == "base_gemm_silu":
                    return hq
            else:
                tile = 1 if arm in ("fused_t1", "fused_gemm_t1") else 0
                hq, hsf = ext.fused_gate_up_swiglu_nvfp4(q, wg[i][2], sfa, wg[i][3], alpha, gs_h, tile, True)
                if arm in ("fused_gemm", "fused_gemm_t1"):
                    return hq
            return torch.ops.trtllm.nvfp4_gemm(hq, wd[i][0], hsf, wd[i][1], alpha, torch.bfloat16)

        with autotune():
            xp.copy_(x0)
            rp.copy_(r0)
            layer(0, "base")
        torch.cuda.synchronize()
        arms = ("base", "fused", "fused_t1", "base_gemm_silu", "fused_gemm", "fused_gemm_t1", "base_gemm")
        graphs = {}
        for arm in arms:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    xp.copy_(x0)
                    rp.copy_(r0)
                    [layer(i, arm) for i in range(L)]
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                outs = [layer(i, arm) for i in range(L)]
            torch.cuda.synchronize()
            graphs[arm] = (g, outs)
        ts = {a: [] for a in arms}
        for it in range(WARM + ITERS):
            k = it % len(arms)
            for arm in arms[k:] + arms[:k]:
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
        # chain outputs equal?
        last = {}
        for arm in ("base", "fused"):
            xp.copy_(x0)
            rp.copy_(r0)
            graphs[arm][0].replay()
            torch.cuda.synchronize()
            last[arm] = torch.stack(graphs[arm][1]).clone()
        row = {a: stats(ts[a]) for a in arms}
        row["chain_outputs_bitwise_equal"] = bool(torch.equal(last["base"], last["fused"]))
        row["chain_saving_us_per_layer"] = round(row["base"]["median_us"] - row["fused"]["median_us"], 3)
        row["kernel_saving_us_per_layer"] = round(
            row["base_gemm_silu"]["median_us"] - row["fused_gemm"]["median_us"], 3)
        row["fused_epilogue_cost_us"] = round(row["fused_gemm"]["median_us"] - row["base_gemm"]["median_us"], 3)
        out["chain"][str(m)] = row
        print("CHAIN", m, json.dumps(row), f"({time.time() - t0:.0f}s)", flush=True)
        del graphs, last, pre, x0, r0, xp, rp
        torch.cuda.empty_cache()

    c = out["chain"]
    print(json.dumps({
        "batch": {k: {"kernel_ms": v["fused"]["median_us"] / 1e3, "noise_floor": v["fused"]["noise_floor"]}
                  for k, v in c.items()},
        "output": {"class": "exact" if out["bitwise_all_equal"] else "reordered"},
        "step0": out,
    }))


if __name__ == "__main__":
    main()
