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
    Ms = [int(x) for x in os.environ.get("ACT_DIAG_MS", "1000,1200,1500,2000").split(",")]
    Ts = [tuple([t.split(":")[0], int(t.split(":")[1])])
          for t in os.environ.get("ACT_DIAG_TACTICS", "cublaslt:1,cublaslt:2,cutlass:0").split(",")]
    if os.environ.get("ACT_DIAG_HOST"):
        # host-only enqueue time (no sync inside the loop) of main's vs the candidate's dispatch,
        # the same calls bench.py times (tuned nvfp4_gemm + act vs gate_up_gelu_fp4_permuted)
        from tensorrt_llm._torch.autotuner import autotune
        bk = "cutlass,cublaslt,cutedsl,cuda_core"
        for M in Ms:
            a = torch.randn(M, K, device=dev, generator=g).to(torch.bfloat16)
            gs_a = (448 * 6 / a.float().abs().max()).reshape(1).float()
            af, asf = torch.ops.trtllm.fp4_quantize(a, gs_a, 16, False)
            al = [(1.0 / (gs_a * c[4])).float() for c in copies]
            with autotune():
                torch.ops.trtllm.nvfp4_gemm(af, copies[0][0], asf, copies[0][1], al[0], torch.bfloat16,
                                            allowed_backends=bk)

            def f_main(i):
                c = copies[i % 3]
                return gelu_tanh_mul_fp4_quant(torch.ops.trtllm.nvfp4_gemm(
                    af, c[0], asf, c[1], al[i % 3], torch.bfloat16, allowed_backends=bk), gs_d)

            def f_cand(i):
                c = copies[i % 3]
                return ge.gate_up_gelu_fp4_permuted(
                    af, asf, c[2], c[3], al[i % 3], torch.bfloat16, bk.split(","), gs_d, None,
                    lambda: torch.ops.trtllm.nvfp4_gemm(af, c[2], asf, c[3], al[i % 3], torch.bfloat16,
                                                        allowed_backends=bk))
            row = {}
            for name, fn in (("main", f_main), ("cand", f_cand), ("main2", f_main), ("cand2", f_cand)):
                for i in range(5):
                    fn(i)
                torch.cuda.synchronize()
                hs = []
                for i in range(30):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    fn(i)
                    hs.append((time.perf_counter() - t0) * 1e6)
                hs.sort()
                row[name] = {"host_us_med": round(hs[15], 1), "p10": round(hs[3], 1), "p90": round(hs[26], 1),
                             "gpu": timeit(fn)}
            out[f"host|{M}"] = row
            log(f"host M={M} {row}")
    for M in Ms:
        a = torch.randn(M, K, device=dev, generator=g).to(torch.bfloat16)
        gs_a = (448 * 6 / a.float().abs().max()).reshape(1).float()
        af, asf = torch.ops.trtllm.fp4_quantize(a, gs_a, 16, False)
        alphas = [(1.0 / (gs_a * c[4])).float() for c in copies]
        for T in Ts:
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
