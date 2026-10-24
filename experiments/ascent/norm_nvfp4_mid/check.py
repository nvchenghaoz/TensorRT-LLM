"""Idea 41 correctness: this tree's modeling_qwen3_next._post_attention_norm
(stub layer, eligibility forced) followed by the gate_up GEMM GatedMLP runs
(M>=41 nvfp4_gate_up_swiglu_nvfp4, tile 1 below 448 rows, else nvfp4_gemm),
against main's ops on the same inputs: FlashInfer gemma_fused_add_rmsnorm +
trtllm fp4_quantize + the same GEMM. Bitwise on the residual (mutated in
place), FP4 bytes, the whole swizzled SF buffer (zeroed padding rows) and the
GEMM output.

Repeated execution: for every M in 33..128 one CUDA graph of the dispatch
chain is captured and replayed 10 warmup + 50 times; before every replay the
residual and x are restored from pristine copies and the graph's FP4/SF/GEMM
output buffers are poisoned (0xFF); after every replay all outputs are
compared with replay 1 and with main's reference. Prefill-sized M (129, 160,
2047, 2048, 4096, 8181, 8192) run the same 60-iteration sequence eagerly.
"""
import json
import os
import sys
import types

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")

import torch  # noqa: E402

import tensorrt_llm  # noqa: E402,F401
from tensorrt_llm._torch.custom_ops import torch_custom_ops  # noqa: E402,F401
from tensorrt_llm._torch.models import modeling_qwen3_next as mq  # noqa: E402
from tensorrt_llm._torch.modules import \
    fused_gemma_add_rmsnorm_nvfp4  # noqa: E402,F401
from tensorrt_llm._torch.modules.rms_norm import RMSNorm  # noqa: E402
from tensorrt_llm._torch.utils import Fp4QuantizedTensor  # noqa: E402

H, N_UP = 5120, 34816
EPS = 1e-6
WARM, ITERS = 10, 50
dev = "cuda"


def main():
    ms_graph = list(range(33, 129))
    ms_eager = [129, 160, 2047, 2048, 4096, 8181, 8192]
    if len(sys.argv) > 1:
        ms_graph = [int(a) for a in sys.argv[1:] if int(a) <= 128]
        ms_eager = [int(a) for a in sys.argv[1:] if int(a) > 128]
    alpha = torch.tensor([0.002], dtype=torch.float32, device=dev)
    gs_down = torch.tensor([448.0 * 6.0 / 8.0], dtype=torch.float32,
                           device=dev)
    g = torch.Generator(device=dev).manual_seed(123)
    w = torch.randint(0, 256, (N_UP, H // 2), dtype=torch.uint8, device=dev,
                      generator=g)
    wsf = torch.randint(0x28, 0x38, (N_UP * H // 16, ), dtype=torch.uint8,
                        device=dev, generator=g)
    norm = RMSNorm(hidden_size=H, eps=EPS, dtype=torch.bfloat16,
                   use_gemma=True).to(dev)
    with torch.no_grad():
        wt = torch.randn(H, device=dev, generator=g) * 0.3
        u = torch.rand(H, device=dev, generator=g)
        wt = torch.where(u < 0.01, torch.full_like(wt, -1.0), wt)
        norm.weight.copy_(wt.to(torch.bfloat16))
    passed = True
    report = {}
    for m in ms_graph + ms_eager:
        sv = [448.0 * 6.0 / 20.0, 100.0, 0.05][m % 3]
        gs_x = torch.tensor([sv], dtype=torch.float32, device=dev)
        lay = types.SimpleNamespace(
            post_attention_layernorm=norm,
            mlp=types.SimpleNamespace(mlp=types.SimpleNamespace(
                gate_up_proj=types.SimpleNamespace(input_scale=gs_x))))
        lay._post_attn_nvfp4_ok = True
        gen = torch.Generator(device=dev).manual_seed(1000 + m)
        sc = torch.exp(torch.empty((m, 1), device=dev).uniform_(
            -6.0, 5.0, generator=gen))
        x0 = (torch.randn((m, H), generator=gen, device=dev) * sc).to(
            torch.bfloat16)
        r0 = (torch.randn((m, H), generator=gen, device=dev) * sc * 3).to(
            torch.bfloat16)
        x0[0, :16] = 0.0
        r0[0, :16] = 0.0
        tile = 0 if m >= 448 else 1

        def gemm(q, sf):
            if m >= 41:
                o = torch.ops.trtllm.nvfp4_gate_up_swiglu_nvfp4(
                    q, w, sf, wsf, alpha, gs_down, tile)
                return [o[0].view(torch.uint8), o[1].view(torch.uint8)]
            return [torch.ops.trtllm.nvfp4_gemm(q, w, sf, wsf, alpha,
                                                torch.bfloat16)]

        # main's reference
        xr, rr = x0.clone(), r0.clone()
        torch.ops.trtllm.flashinfer_gemma_fused_add_rmsnorm(
            xr, rr, norm.weight, EPS)
        qr, sfr = torch.ops.trtllm.fp4_quantize(xr, gs_x, 16, False)
        ref = [rr, qr.view(torch.uint8).reshape(-1),
               sfr.view(torch.uint8).reshape(-1)] + gemm(qr, sfr)
        torch.cuda.synchronize()

        x, r = x0.clone(), r0.clone()
        fused_seen = []

        def chain():
            out, _ = mq._post_attention_norm(lay, x, r, None)
            fused_seen.append(isinstance(out, Fp4QuantizedTensor))
            if isinstance(out, Fp4QuantizedTensor):
                q, sf = out.fp4_tensor, out.scaling_factor
            else:
                q, sf = torch.ops.trtllm.fp4_quantize(out, gs_x, 16, False)
            return [q.view(torch.uint8).reshape(-1),
                    sf.view(torch.uint8).reshape(-1)] + gemm(q, sf)

        use_graph = m in ms_graph
        if use_graph:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                x.copy_(x0)
                r.copy_(r0)
                chain()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outs = chain()
            torch.cuda.synchronize()
        first, bad_first, bad_ref, maxd = None, 0, 0, 0
        for it in range(WARM + ITERS):
            x.copy_(x0)
            r.copy_(r0)
            if use_graph:
                for t in outs:
                    t.view(torch.uint8).fill_(0xFF)
                graph.replay()
                cur_outs = outs
            else:
                cur_outs = chain()
            torch.cuda.synchronize()
            cur = [r.clone()] + [t.clone() for t in cur_outs]
            if first is None:
                first = cur
            d1 = sum(int((a.view(torch.uint8) != b.view(torch.uint8)).sum())
                     if a.shape == b.shape else -1
                     for a, b in zip(cur, first))
            d2 = sum(int((a.view(torch.uint8) != b.view(torch.uint8)).sum())
                     if a.shape == b.shape else 10**9
                     for a, b in zip(cur, ref))
            bad_first += int(d1 != 0)
            bad_ref += int(d2 != 0)
            maxd = max(maxd, abs(d1), d2)
        if use_graph:
            del graph
        ok = bad_first == 0 and bad_ref == 0
        passed = passed and ok
        report[str(m)] = {"graph": use_graph, "fused_path": fused_seen[-1],
                          "bad_vs_iter1": bad_first, "bad_vs_main": bad_ref,
                          "max_ndiff": maxd, "ok": ok}
        print("M", m, json.dumps(report[str(m)]), flush=True)
    fused_ms = [k for k, v in report.items() if v["fused_path"]]
    print(json.dumps({
        "passed": passed,
        "compared": "dispatch (_post_attention_norm + gate_up GEMM) vs main's "
                    "gemma_fused_add_rmsnorm + fp4_quantize + same GEMM: "
                    "residual, FP4, full swizzled SF incl. pad rows, GEMM "
                    "output; bitwise, every one of 60 iterations vs iter 1 "
                    "and vs main; CUDA-graph replays for M 33-128",
        "n_cases": len(report), "fused_path_ms": fused_ms,
        "iterations": WARM + ITERS, "per_m": report,
    }))


if __name__ == "__main__":
    main()
