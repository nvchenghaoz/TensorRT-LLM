"""Diagnostic (no model load): where does the unfused-on-permuted-rows chain lose time vs main's
when the gate_up pick is not ('cutlass', 2)?  Per M, CUDA events, 5 warmup + 30 timed, 3 rotating
inputs/weight copies (cold-ish L2):
  gemm_orig / gemm_perm : NVFP4GemmUnifiedRunner tactic T on original / permuted gate_up rows
  act_main / act_perm   : gelu_tanh_mul_fp4_quant / gelu_tanh_mul_fp4_quant_permuted on the GEMM output
  chain_main / chain_perm
T in ('cublaslt', 1), ('cublaslt', 2), ('cutlass', 0).
usage: act_diag.py <checkpoint>
"""
import json
import os
import statistics
import sys
import time

T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def main():
    ck = sys.argv[1]
    import torch
    cfg = json.load(open(os.path.join(ck, "config.json")))
    tc = cfg.get("text_config", cfg)
    H, I = tc["hidden_size"], tc["intermediate_size"]
    N, K = 2 * I, H
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import NVFP4GemmUnifiedRunner
    from tensorrt_llm._torch.kernels.fused_ops import nvfp4_gelu_gemm_sm120 as ge
    from tensorrt_llm._torch.kernels.fused_ops.gelu_tanh_mul_fp4_quant import gelu_tanh_mul_fp4_quant
    dev = torch.device("cuda")
    g = torch.Generator(device=dev)
    g.manual_seed(5)
    copies = []
    for _ in range(3):
        w = torch.randn(N, K, device=dev, generator=g).to(torch.bfloat16)
        gs_w = (448 * 6 / w.float().abs().max()).reshape(1).float()
        wf, ws = torch.ops.trtllm.fp4_quantize(w, gs_w, 16, False)
        del w
        wp, wsp = ge.permute_gate_up_rows(wf, ws, K)
        copies.append((wf, ws, wp, wsp, gs_w))
    runner = NVFP4GemmUnifiedRunner(0, torch.bfloat16, ["cutlass", "cublaslt"])
    gs_d = torch.tensor([3.07], device=dev)

    def timeit(fn):
        for i in range(5):
            fn(i)
        torch.cuda.synchronize()
        ts = []
        for i in range(30):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            fn(i)
            e.record()
            e.synchronize()
            ts.append(s.elapsed_time(e) * 1e3)
        ts.sort()
        return {"med_us": round(statistics.median(ts), 1), "p10": round(ts[3], 1), "p90": round(ts[26], 1)}

    out = {}
    for M in (1000, 1200, 1500, 2000):
        a = torch.randn(M, K, device=dev, generator=g).to(torch.bfloat16)
        gs_a = (448 * 6 / a.float().abs().max()).reshape(1).float()
        af, asf = torch.ops.trtllm.fp4_quantize(a, gs_a, 16, False)
        alphas = [(1.0 / (gs_a * c[4])).float() for c in copies]
        for T in (("cublaslt", 1), ("cublaslt", 2), ("cutlass", 0)):
            try:
                ys = [runner(inputs=[af, c[0], asf, c[1], alphas[j]], tactic=T) for j, c in enumerate(copies)]
                yps = [runner(inputs=[af, c[2], asf, c[3], alphas[j]], tactic=T) for j, c in enumerate(copies)]
            except Exception as e:  # noqa: BLE001
                out[f"{M}|{T}"] = {"error": str(e)[:200]}
                continue
            r = {
                "gemm_orig": timeit(lambda i: runner(inputs=[af, copies[i % 3][0], asf, copies[i % 3][1],
                                                             alphas[i % 3]], tactic=T)),
                "gemm_perm": timeit(lambda i: runner(inputs=[af, copies[i % 3][2], asf, copies[i % 3][3],
                                                             alphas[i % 3]], tactic=T)),
                "act_main": timeit(lambda i: gelu_tanh_mul_fp4_quant(ys[i % 3], gs_d)),
                "act_perm": timeit(lambda i: ge.gelu_tanh_mul_fp4_quant_permuted(yps[i % 3], gs_d)),
                "chain_main": timeit(lambda i: gelu_tanh_mul_fp4_quant(
                    runner(inputs=[af, copies[i % 3][0], asf, copies[i % 3][1], alphas[i % 3]], tactic=T), gs_d)),
                "chain_perm": timeit(lambda i: ge.gelu_tanh_mul_fp4_quant_permuted(
                    runner(inputs=[af, copies[i % 3][2], asf, copies[i % 3][3], alphas[i % 3]], tactic=T), gs_d)),
            }
            # bitwise: permuted chain == main chain
            fm, sm = gelu_tanh_mul_fp4_quant(ys[0], gs_d)
            fp_, sp_ = ge.gelu_tanh_mul_fp4_quant_permuted(yps[0], gs_d)
            r["bitwise"] = bool(torch.equal(fm, fp_) and torch.equal(sm, sp_))
            out[f"{M}|{T[0]}{T[1]}"] = r
            log(f"M={M} T={T} {r}")
            del ys, yps
            torch.cuda.empty_cache()
    print(json.dumps({"passed": True, "rows": out}), flush=True)


if __name__ == "__main__":
    main()
