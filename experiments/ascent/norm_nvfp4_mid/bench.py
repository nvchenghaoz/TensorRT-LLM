"""Idea 41: post_attention gemma add+RMSNorm emitting gate_up's NVFP4 input at
decode rows 33..R. No model load; synthetic tensors, real shapes.

Per layer (M rows), L distinct layers in one CUDA graph (L x ~105 MB weights
>> 128 MB L2, plus a 256 MB L2 read scrub before each replay):
  producer nvfp4_gemm (N=5120, K=6144; writes x like out_proj/o_proj)
  -> post_attention norm (+ gate_up input quant)
  -> gate_up GEMM as GatedMLP runs it (M>=41: nvfp4_gate_up_swiglu_nvfp4
     tile 1 (0 from 448), else nvfp4_gemm N=34816)
Arms:
  disp   : THIS TREE's modeling_qwen3_next._post_attention_norm on a stub
           layer (eligibility forced True), then fp4_quantize if it returned
           BF16 (what NVFP4LinearMethod._input_prepare does). Reported arm.
  main   : stock flashinfer gemma_fused_add_rmsnorm + fp4_quantize
  cute   : trtllm::flashinfer_gemma_fused_add_rmsnorm_nvfp4
  triton : trtllm::fused_gemma_add_rmsnorm_nvfp4
Arms replayed alternately with rotating order; residual restored from a
pristine copy before every replay (outside the timed window). Bitwise: every
arm's residual, FP4, SF (whole padded buffer) and gate_up output vs main.
argv: [arms_csv] M...
"""
import json
import os
import statistics
import sys
import types

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")

import torch  # noqa: E402

import tensorrt_llm  # noqa: E402,F401
from tensorrt_llm._torch.autotuner import autotune  # noqa: E402
from tensorrt_llm._torch.custom_ops import torch_custom_ops  # noqa: E402,F401
from tensorrt_llm._torch.models import modeling_qwen3_next as mq  # noqa: E402
from tensorrt_llm._torch.modules import \
    fused_gemma_add_rmsnorm_nvfp4  # noqa: E402,F401
from tensorrt_llm._torch.modules.rms_norm import RMSNorm  # noqa: E402
from tensorrt_llm._torch.utils import Fp4QuantizedTensor  # noqa: E402

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
    args = sys.argv[1:]
    arms = ("disp", "main", "cute", "triton")
    op_only = False
    if args and args[0] == "op":
        # Time only the post-attn norm(+quant) ops (no GEMMs); x and the
        # residual are restored from pristine copies before every replay.
        op_only = True
        args = args[1:]
    if args and not args[0].isdigit():
        arms = tuple(args[0].split(","))
        args = args[1:]
    ms = [int(a) for a in args] or [33, 48, 64, 96, 128]
    if "main" not in arms:
        arms = ("main", ) + arms
    print(f"arms={arms} ms={ms}", flush=True)
    alpha = torch.tensor([0.002], dtype=torch.float32, device=dev)
    gs_in = torch.tensor([20.0], dtype=torch.float32, device=dev)
    gs_x = torch.tensor([448.0 * 6.0 / 20.0], dtype=torch.float32, device=dev)
    gs_down = torch.tensor([448.0 * 6.0 / 8.0], dtype=torch.float32,
                           device=dev)
    wp = [rand_w(H, K_PROD, 1000 + i) for i in range(L)]
    wg = [rand_w(N_UP, H, 10 + i) for i in range(L)]
    layers = []
    for i in range(L):
        norm = RMSNorm(hidden_size=H, eps=EPS, dtype=torch.bfloat16,
                       use_gemma=True).to(dev)
        with torch.no_grad():
            norm.weight.copy_((torch.randn(H, device=dev) * 0.3).to(
                torch.bfloat16))
        lay = types.SimpleNamespace(
            post_attention_layernorm=norm,
            mlp=types.SimpleNamespace(mlp=types.SimpleNamespace(
                gate_up_proj=types.SimpleNamespace(input_scale=gs_x))))
        lay._post_attn_nvfp4_ok = True
        layers.append(lay)
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
        tile = 0 if m >= 448 else 1
        if op_only:
            x0 = (torch.randn((L, m, H), generator=gen, device=dev) *
                  0.7).to(torch.bfloat16)
            xp = x0.clone()
        else:
            x0 = xp = None

        def restore():
            rp.copy_(r0)
            if xp is not None:
                xp.copy_(x0)

        def layer(i, arm):
            if op_only:
                x = xp[i]
            else:
                x = torch.ops.trtllm.nvfp4_gemm(aq[i][0], wp[i][0], aq[i][1],
                                                wp[i][1], alpha,
                                                torch.bfloat16)
            r = rp[i]
            norm = layers[i].post_attention_layernorm
            if arm == "disp":
                out, _ = mq._post_attention_norm(layers[i], x, r, None)
                if isinstance(out, Fp4QuantizedTensor):
                    q, sf = out.fp4_tensor, out.scaling_factor
                else:
                    q, sf = torch.ops.trtllm.fp4_quantize(out, gs_x, 16, False)
            elif arm == "cute":
                q, sf = torch.ops.trtllm.flashinfer_gemma_fused_add_rmsnorm_nvfp4(
                    x, r, norm.weight, EPS, gs_x)
            elif arm == "triton":
                q, sf = torch.ops.trtllm.fused_gemma_add_rmsnorm_nvfp4(
                    x, r, norm.weight, EPS, gs_x)
            else:
                torch.ops.trtllm.flashinfer_gemma_fused_add_rmsnorm(
                    x, r, norm.weight, EPS)
                q, sf = torch.ops.trtllm.fp4_quantize(x, gs_x, 16, False)
            if op_only:
                return (q.view(torch.uint8).reshape(-1),
                        sf.view(torch.uint8).reshape(-1))
            if m >= 41:
                o = torch.ops.trtllm.nvfp4_gate_up_swiglu_nvfp4(
                    q, wg[i][0], sf, wg[i][1], alpha, gs_down, tile)
                o = (o[0].view(torch.uint8).reshape(-1),
                     o[1].view(torch.uint8).reshape(-1))
            else:
                o = (torch.ops.trtllm.nvfp4_gemm(q, wg[i][0], sf, wg[i][1],
                                                 alpha, torch.bfloat16), )
            return (q.view(torch.uint8).reshape(-1),
                    sf.view(torch.uint8).reshape(-1)) + o

        with autotune():
            restore()
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
                    restore()
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
                restore()
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
            restore()
            graphs[arm][0].replay()
            torch.cuda.synchronize()
            last[arm] = ([tuple(t.clone() for t in lo)
                          for lo in graphs[arm][1]], rp.clone())
        bitwise = {}
        for arm in arms:
            lo, res = last[arm]
            lm, rm = last["main"]
            ok = torch.equal(res, rm) and all(
                len(a_) == len(b_) and all(
                    x_.shape == y_.shape and torch.equal(x_, y_)
                    for x_, y_ in zip(a_, b_)) for a_, b_ in zip(lo, lm))
            bitwise[arm] = ok
        row = {}
        for arm in arms:
            v = sorted(ts[arm])
            n = len(v)
            med = statistics.median(v)
            p10, p90 = v[int(0.1 * (n - 1))], v[int(0.9 * (n - 1))]
            row[arm] = {"median_us_per_layer": round(med, 3),
                        "p10_us": round(p10, 3), "p90_us": round(p90, 3),
                        "noise_floor": round((p90 - p10) / med, 4), "n": n}
        row["bitwise_vs_main"] = bitwise
        for arm in arms:
            if arm != "main":
                row[f"{arm}_minus_main_us"] = round(
                    row[arm]["median_us_per_layer"] -
                    row["main"]["median_us_per_layer"], 3)
        results[str(m)] = row
        print("M", m, json.dumps(row), flush=True)
        del graphs, last
        torch.cuda.empty_cache()
    rep = "disp" if "disp" in arms else arms[-1]
    allbit = all(all(v["bitwise_vs_main"].values()) for v in results.values())
    print(json.dumps({
        "batch": {k: {"kernel_ms": v[rep]["median_us_per_layer"] / 1e3,
                      "noise_floor": max(v[rep]["noise_floor"],
                                         v["main"]["noise_floor"])}
                  for k, v in results.items()},
        "output": {"class": "exact"},
        "bitwise_all_arms": allbit,
        "reported_arm": rep,
        "mode": "op" if op_only else "chain",
        "unit_note": ("kernel_ms = median per-layer time of post-attn "
                      "norm(+gate_up input quant) only" if op_only else
                      "kernel_ms = median per-layer time of the chain "
                      "producer GEMM -> post-attn norm(+quant) -> gate_up "
                      "GEMM"),
        "chain": results,
    }))


if __name__ == "__main__":
    main()
