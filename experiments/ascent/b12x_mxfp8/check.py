"""Correctness + repeated-execution check for trtllm::flashinfer_mm_mxfp8 b12x dispatch.

Protected references:
  (a) FlashInfer mm_mxfp8(backend='cutlass') on the same quantized inputs (the
      original op) -> must agree BITWISE;
  (b) FP32 dequantized reference (x_q * 2^sa) @ (w_q * 2^sw)^T -> BF16 tolerance.
Also: which backend actually ran (b12x for M<=32 on SM12x), no b12x JIT during
CUDA-graph capture, decode graphs at batch 1/2/4/8 with quantize+GEMM, replayed
over 8 iterations with fresh inputs, each iteration compared bitwise.
"""
import json
import traceback

import torch

import tensorrt_llm  # noqa: F401
from flashinfer import autotune, mm_mxfp8
from flashinfer.gemm import gemm_mm_mxfp8_cute_dsl as b12mod

dev = torch.device("cuda")
op = torch.ops.trtllm.flashinfer_mm_mxfp8
SHAPES = [("wq_a", 1280, 5120), ("wkv", 512, 5120), ("wq_b", 8192, 1280),
          ("o_b_proj", 5120, 2048), ("index_wq_b", 4096, 1280),
          ("fused_wqa_wkv", 1792, 5120), ("fused_wqb_idx", 12288, 1280)]
MS = [1, 2, 3, 4, 8, 16, 32, 33, 64, 2048]


def unswizzle(sf, rows, kblk):
    """128x4 swizzled UE8M0 -> [rows, kblk] uint8."""
    kpad = (kblk + 3) // 4 * 4
    r = torch.arange(rows, device=sf.device)[:, None]
    c = torch.arange(kblk, device=sf.device)[None, :]
    idx = ((r // 128) * (kpad // 4) + c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + (c % 4)
    return sf[idx]


def dequant(q, sf, rows, k):
    s = unswizzle(sf, rows, k // 32).float()
    scale = torch.exp2(s - 127.0).repeat_interleave(32, dim=1)
    return q.float() * scale


def cutlass_ref(aq, asf, wq, wsf):
    return mm_mxfp8(aq, wq.t(), asf, wsf, out_dtype=torch.bfloat16, use_8x4_sf_layout=False,
                    backend="cutlass")


passed = True
report = {"cases": {}, "graphs": {}}
calls = {"b12x": 0}
orig_fwd = None


def count_b12x():
    # wrap the runner class forward to count b12x launches made by the op
    ents = [e for e in __import__("tensorrt_llm._torch.custom_ops.flashinfer_custom_ops",
                                  fromlist=["x"]).__dict__.values()]
    return ents


try:
    torch.manual_seed(0)
    weights = {}
    for name, n, k in SHAPES:
        w = torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.05
        weights[name] = torch.ops.trtllm.mxfp8_quantize(w, True)
    # serving warmup: autotune pass over all shapes/M
    with autotune():
        for name, n, k in SHAPES:
            wq, wsf = weights[name]
            for M in MS:
                x = torch.randn(M, k, device=dev, dtype=torch.bfloat16)
                aq, asf = torch.ops.trtllm.mxfp8_quantize(x, True)
                op(aq, asf, wq, wsf, torch.bfloat16)
    torch.cuda.synchronize()
    report["b12x_kernel_cache_after_warmup"] = len(b12mod._B12X_MM_MXFP8_KERNEL_CACHE)

    for name, n, k in SHAPES:
        wq, wsf = weights[name]
        wdq = dequant(wq, wsf, n, k)
        for M in MS:
            torch.manual_seed(1000 * M + n)
            x = torch.randn(M, k, device=dev, dtype=torch.bfloat16)
            aq, asf = torch.ops.trtllm.mxfp8_quantize(x, True)
            nb = len(b12mod._B12X_MM_MXFP8_KERNEL_CACHE)
            out = op(aq, asf, wq, wsf, torch.bfloat16)
            out2 = op(aq, asf, wq, wsf, torch.bfloat16)
            ref = cutlass_ref(aq, asf, wq, wsf)
            fp32 = dequant(aq, asf, M, k) @ wdq.t()
            bit = bool(torch.equal(out, ref))
            det = bool(torch.equal(out, out2))
            err = (out.float() - fp32).abs()
            tol_ok = bool(torch.allclose(out.float(), fp32, rtol=1.6e-2, atol=1.6e-2 * float(fp32.abs().max())))
            ref_err = float((ref.float() - fp32).abs().max())
            ok = bit and det and tol_ok and bool(torch.isfinite(out).all()) and out.shape == (M, n) \
                and out.is_contiguous() and out.dtype == torch.bfloat16
            passed &= ok
            report["cases"][f"{name}:M{M}"] = {
                "bitwise_vs_cutlass": bit, "deterministic": det, "fp32_tol_ok": tol_ok,
                "max_abs_vs_fp32": float(err.max()), "cutlass_max_abs_vs_fp32": ref_err,
                "new_compiles": len(b12mod._B12X_MM_MXFP8_KERNEL_CACHE) - nb, "ok": ok}
            if not ok:
                print("FAIL", name, M, report["cases"][f"{name}:M{M}"], flush=True)
        print(f"{name}: done", flush=True)

    # backend actually used: patch runner class forward counter
    entry = None
    from tensorrt_llm._torch.custom_ops import flashinfer_custom_ops as fco
    runners = fco._b12x_runners if hasattr(fco, "_b12x_runners") else None
    report["b12x_runner_present"] = None
    if runners is None:
        # nested in the IS_FLASHINFER_AVAILABLE block -> module globals
        runners = getattr(fco, "_b12x_runners", None)
    if runners:
        entry = list(runners.values())[0]
        report["b12x_runner_present"] = entry is not None
    if entry is not None:
        cls = type(entry[0])
        orig = cls.forward
        def counted(self, *a, **kw):
            calls["b12x"] += 1
            return orig(self, *a, **kw)
        cls.forward = counted

    # decode graphs: quantize + 5 CSA2 GEMMs, batch 1/2/4/8, 8 replays with fresh inputs
    for B in (1, 2, 4, 8):
        names = ["wq_a", "wkv", "wq_b", "o_b_proj", "index_wq_b"]
        static_x = {nm: torch.zeros(B, weights[nm][0].shape[1], device=dev, dtype=torch.bfloat16) for nm in names}

        def step():
            outs = []
            for nm in names:
                aq, asf = torch.ops.trtllm.mxfp8_quantize(static_x[nm], True)
                outs.append(op(aq, asf, weights[nm][0], weights[nm][1], torch.bfloat16))
            return outs
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            step()
        torch.cuda.current_stream().wait_stream(s)
        nb = len(b12mod._B12X_MM_MXFP8_KERNEL_CACHE)
        calls["b12x"] = 0
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            gouts = step()
        captured_b12x = calls["b12x"]
        compiles_in_capture = len(b12mod._B12X_MM_MXFP8_KERNEL_CACHE) - nb
        it_res = []
        for it in range(8):
            torch.manual_seed(5000 + 31 * B + it)
            for nm in names:
                static_x[nm].copy_(torch.randn_like(static_x[nm]))
            g.replay()
            torch.cuda.synchronize()
            per = []
            for nm, go in zip(names, gouts):
                aq, asf = torch.ops.trtllm.mxfp8_quantize(static_x[nm], True)
                ref = cutlass_ref(aq, asf, weights[nm][0], weights[nm][1])
                d = (go.float() - ref.float()).abs().max()
                per.append({"bitwise": bool(torch.equal(go, ref)), "max_abs": float(d),
                            "max_rel": float(d / ref.float().abs().max())})
            it_ok = all(p["bitwise"] for p in per)
            passed &= it_ok
            it_res.append({"it": it, "all_bitwise": it_ok,
                           "max_abs": max(p["max_abs"] for p in per),
                           "max_rel": max(p["max_rel"] for p in per)})
        ok_g = compiles_in_capture == 0 and captured_b12x == len(names)
        passed &= ok_g
        report["graphs"][str(B)] = {"b12x_launches_captured": captured_b12x,
                                     "compiles_in_capture": compiles_in_capture, "iters": it_res}
        print(f"graph B={B}: captured b12x={captured_b12x} compiles={compiles_in_capture} "
              f"iters_ok={[r['all_bitwise'] for r in it_res]}", flush=True)
        del g
    # prefill M=2048 must not route to b12x
    calls["b12x"] = 0
    x = torch.randn(2048, 5120, device=dev, dtype=torch.bfloat16)
    aq, asf = torch.ops.trtllm.mxfp8_quantize(x, True)
    op(aq, asf, weights["wq_a"][0], weights["wq_a"][1], torch.bfloat16)
    report["prefill_2048_b12x_calls"] = calls["b12x"]
    passed &= calls["b12x"] == 0
except Exception:
    passed = False
    report["error"] = traceback.format_exc()[-3000:]

n_cases = len(report["cases"])
report["n_cases"] = n_cases
report["n_cases_ok"] = sum(1 for v in report["cases"].values() if v["ok"])
print(json.dumps({"passed": bool(passed) and n_cases == len(SHAPES) * len(MS),
                  "compared": "trtllm::flashinfer_mm_mxfp8 vs flashinfer mm_mxfp8 backend=cutlass (bitwise) and FP32 dequant reference (rtol 1.6e-2)",
                  **{k: v for k, v in report.items() if k != "cases"},
                  "cases_summary": {k: [v["bitwise_vs_cutlass"], round(v["max_abs_vs_fp32"], 5), v["new_compiles"]]
                                    for k, v in report["cases"].items()}}))
