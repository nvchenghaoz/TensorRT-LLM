"""Stage 1 kernel check for the fused gate_up + gelu + NVFP4-quant epilogue (no model load).

Per M: main chain = FP4GemmRunner tactic 2 (256x128x128, main's tuned gate_up pick at every M in
job 290) on the original rows -> gelu_tanh_mul_fp4_quant; fused = nvfp4_gelu_gemm_fp4 on the
permuted rows; also the permuted unfused path (tactic 2 on permuted rows ->
gelu_tanh_mul_fp4_quant_permuted). Payload and FULL SF buffers are byte-compared.
Data: randn seeds 0/1 and heavy-tailed randn*2^U(-8,8) per row. Then a rough cold-L2 timing.
usage: kernel_test.py <checkpoint> [quick]
"""
import json
import os
import sys
import threading
import time

T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def _hb():
    while True:
        time.sleep(45)
        log("heartbeat")


def cmpbytes(a, b):
    import torch
    a = a.reshape(-1).view(torch.uint8)
    b = b.reshape(-1).view(torch.uint8)
    if a.numel() != b.numel():
        return {"equal": False, "size": [a.numel(), b.numel()]}
    ne = a != b
    n = int(ne.sum().item())
    r = {"equal": n == 0, "mismatch": n}
    if n:
        i = int(ne.nonzero()[0].item())
        r["first"] = {"byte": i, "a": int(a[i]), "b": int(b[i])}
    return r


def main():
    threading.Thread(target=_hb, daemon=True).start()
    ck = sys.argv[1]
    quick = len(sys.argv) > 2 and sys.argv[2] == "quick"
    import torch
    cfg = json.load(open(os.path.join(ck, "config.json")))
    tc = cfg.get("text_config", cfg)
    hidden, inter = tc["hidden_size"], tc["intermediate_size"]
    N, K = 2 * inter, hidden
    log(f"gate_up N={N} K={K}")
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import NVFP4GemmUnifiedRunner
    from tensorrt_llm._torch.kernels.fused_ops import nvfp4_gelu_gemm_sm120 as ge
    from tensorrt_llm._torch.kernels.fused_ops.gelu_tanh_mul_fp4_quant import gelu_tanh_mul_fp4_quant

    t = time.time()
    mod = ge.get_gelu_gemm_module()
    log(f"module build/load {time.time() - t:.1f}s err={ge._module_error}")
    if mod is None:
        print(json.dumps({"passed": False, "error": ge._module_error}))
        return
    info = {"stages": int(mod.nvfp4_gelu_gemm_info(0)), "smem": int(mod.nvfp4_gelu_gemm_info(1))}
    log(f"kernel info {info}")
    dev = torch.device("cuda")
    runner = NVFP4GemmUnifiedRunner(0, torch.bfloat16, ["cutlass"])
    res = {"info": info, "rows": []}
    Ms = [300, 1000, 8000] if quick else [1, 37, 300, 301, 1000, 1023, 1200, 2000, 4000, 4800, 8000, 8100, 8192]
    ok = True
    for ds in ("seed0", "seed1", "heavy"):
        g = torch.Generator(device=dev)
        g.manual_seed({"seed0": 0, "seed1": 1, "heavy": 2}[ds])
        w = torch.randn(N, K, device=dev, generator=g)
        if ds == "heavy":
            w = w * torch.exp2(torch.empty(N, 1, device=dev).uniform_(-8, 8, generator=g))
        w = w.to(torch.bfloat16)
        gs_w = (448 * 6 / w.float().abs().max()).reshape(1).float()
        w_fp4, w_sf = torch.ops.trtllm.fp4_quantize(w, gs_w, 16, False)
        del w
        wp, wsp = ge.permute_gate_up_rows(w_fp4, w_sf, K)
        a_all = torch.randn(max(Ms), K, device=dev, generator=g)
        if ds == "heavy":
            a_all = a_all * torch.exp2(torch.empty(max(Ms), 1, device=dev).uniform_(-8, 8, generator=g))
        a_all = a_all.to(torch.bfloat16)
        for M in Ms:
            a = a_all[:M].contiguous()
            gs_a = (448 * 6 / a.float().abs().max()).reshape(1).float()
            a_fp4, a_sf = torch.ops.trtllm.fp4_quantize(a, gs_a, 16, False)
            alpha = (1.0 / (gs_a * gs_w)).float()
            y = runner(inputs=[a_fp4, w_fp4, a_sf, w_sf, alpha], tactic=("cutlass", 2))
            gg, uu = y[:, :inter].float(), y[:, inter:].float()
            amax = (torch.nn.functional.gelu(gg, approximate="tanh") * uu).abs().max().clamp_min(1e-30)
            del gg, uu
            gs_d = (448 * 6 / amax).reshape(1).float()
            fp4_ref, sf_ref = gelu_tanh_mul_fp4_quant(y, gs_d)
            del y
            out = torch.full((M, inter // 2), 0xA5, dtype=torch.uint8, device=dev)
            sfo = torch.full_like(sf_ref, 0xA5)
            fp4_f, sf_f = ge.nvfp4_gelu_gemm_fp4(a_fp4, a_sf, wp, wsp, alpha, gs_d, out=out, sf=sfo)
            yp = runner(inputs=[a_fp4, wp, a_sf, wsp, alpha], tactic=("cutlass", 2))
            fp4_u, sf_u = ge.gelu_tanh_mul_fp4_quant_permuted(yp, gs_d)
            del yp
            r = {"data": ds, "M": M, "fused_fp4": cmpbytes(fp4_f, fp4_ref), "fused_sf": cmpbytes(sf_f, sf_ref),
                 "perm_fp4": cmpbytes(fp4_u, fp4_ref), "perm_sf": cmpbytes(sf_u, sf_ref)}
            good = all(r[k]["equal"] for k in ("fused_fp4", "fused_sf", "perm_fp4", "perm_sf"))
            ok &= good
            res["rows"].append(r)
            log(f"{ds} M={M} ok={good} {json.dumps(r) if not good else ''}")
        del a_all, w_fp4, w_sf, wp, wsp
        torch.cuda.empty_cache()

    # rough timing, cold weights: 3 rotating weight copies (3 x 123 MB > 2 x L2)
    timing = {}
    copies = []
    g = torch.Generator(device=dev)
    g.manual_seed(7)
    for c in range(3):
        w = torch.randn(N, K, device=dev, generator=g).to(torch.bfloat16)
        gs_w = (448 * 6 / w.float().abs().max()).reshape(1).float()
        w_fp4, w_sf = torch.ops.trtllm.fp4_quantize(w, gs_w, 16, False)
        del w
        wp, wsp = ge.permute_gate_up_rows(w_fp4, w_sf, K)
        copies.append((w_fp4, w_sf, wp, wsp, gs_w))
    for M in ([300, 1000, 2000, 8000] if quick else [300, 1000, 1200, 2000, 4000, 4800, 8000, 8192]):
        a = torch.randn(M, K, device=dev, generator=g).to(torch.bfloat16)
        gs_a = (448 * 6 / a.float().abs().max()).reshape(1).float()
        a_fp4, a_sf = torch.ops.trtllm.fp4_quantize(a, gs_a, 16, False)
        alphas = [(1.0 / (gs_a * c[4])).float() for c in copies]
        gs_d = torch.tensor([1.0], device=dev)

        def base(i):
            c = copies[i % 3]
            y = runner(inputs=[a_fp4, c[0], a_sf, c[1], alphas[i % 3]], tactic=("cutlass", 2))
            return gelu_tanh_mul_fp4_quant(y, gs_d)

        def fused(i):
            c = copies[i % 3]
            return ge.nvfp4_gelu_gemm_fp4(a_fp4, a_sf, c[2], c[3], alphas[i % 3], gs_d)

        for name, fn in (("base", base), ("fused", fused)):
            for i in range(5):
                fn(i)
            torch.cuda.synchronize()
            ts = []
            for i in range(21):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                fn(i)
                e.record()
                e.synchronize()
                ts.append(s.elapsed_time(e) * 1e3)
            ts.sort()
            timing.setdefault(str(M), {})[name] = {"med_us": round(ts[10], 1), "p10": round(ts[2], 1),
                                                   "p90": round(ts[18], 1)}
        log(f"M={M} {timing[str(M)]}")
    res["timing"] = timing
    print(json.dumps({"passed": bool(ok), "info": info, "timing": timing,
                      "n_rows": len(res["rows"])}), flush=True)


if __name__ == "__main__":
    main()
