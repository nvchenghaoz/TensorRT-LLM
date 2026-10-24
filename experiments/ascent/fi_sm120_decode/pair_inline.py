import json, os, sys
os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
import torch
sys.path.insert(0, os.path.join(os.getcwd(), "experiments", "ascent"))
import nvfp4_coldtune as ct
import tensorrt_llm
from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.custom_ops import torch_custom_ops as tco
# Idea 43 pair bench (run inline in base and candidate). Per decode step at M=64/128 the
# four W4A4 GEMM shapes x layers; each call = fp4_quantize (producer) -> trtllm::nvfp4_gemm
# with the tree's own AutoTuner pick; nb calls over distinct weight copies (>= 400 MB, cold
# L2) in one CUDA graph, PDL on; 10 warmup + 50 timed replays; kernel_ms = layer-weighted
# sum of per-call medians.
SH = {"qkvz": (16384, 5120, 48), "qkv": (14336, 5120, 16), "o_out": (5120, 6144, 64),
      "down": (5120, 17408, 64)}
MS = (64, 128)
ct.MS = MS
has_fi = hasattr(tco, "FlashInferCutlassNVFP4Runner")
shapes = [ct.Shape(k, n, kk) for k, (n, kk, _) in SH.items()]
tuner = AutoTuner.get()
tuner.clear_cache()
for s in shapes:
    with autotune():
        torch.ops.trtllm.nvfp4_gemm(*s.inputs(128), torch.bfloat16, allowed_backends=ct.BACKENDS)
torch.cuda.synchronize()
r = ct.runner()
batch, detail = {}, {}
for m in MS:
    tot, noise = 0.0, 0.0
    for s in shapes:
        hit, rid, tac, t = tuner.profiling_cache.search_cache(
            ct.OP, [r], tuple(tuner._get_input_sizes(s.inputs(m))),
            tco.NVFP4GemmUnifiedRunner.tuning_config, apply_map_to_tuning_buckets=True)
        g0 = torch.Generator(device="cuda").manual_seed(4321 + m + s.n)
        x = torch.randn(m, s.k, generator=g0, device="cuda")
        sc = torch.exp(torch.randn(m, s.k // 16, 1, generator=g0, device="cuda") * 2.0)
        x = (x.view(m, s.k // 16, 16) * sc).reshape(m, s.k).to(torch.bfloat16)
        gs = (448.0 * 6.0 / x.float().abs().max()).reshape(1).float()
        alpha = (1.0 / (gs * s.gs_w)).float()

        def call(i):
            a, sf = ct.quant(x, gs)
            w, wsf = s.copies[i % s.nb]
            return torch.ops.trtllm.nvfp4_gemm(a, w, sf, wsf, alpha, torch.bfloat16,
                                               allowed_backends=ct.BACKENDS)
        for i in range(s.nb):
            call(i)
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for i in range(s.nb):
                call(i)
        torch.cuda.current_stream().wait_stream(st)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            outs = [call(i) for i in range(s.nb)]
        res = ct.stats(ct.time_graph(g, s.nb, 10, 50))
        del g, outs
        key = "%s/%d" % (s.name, m)
        detail[key] = dict(res, tactic=list(tac) if isinstance(tac, tuple) else tac,
                           us=res["median_ms"] * 1e3)
        tot += SH[s.name][2] * res["median_ms"]
        noise = max(noise, res["noise"])
        print("  %s pick %s: %.2f us/call (p90 %.2f) noise %.3f" % (
            key, detail[key]["tactic"], detail[key]["us"], res["p90_ms"] * 1e3, res["noise"]),
            flush=True)
    batch[str(m)] = {"kernel_ms": round(tot, 5), "noise_floor": round(noise, 4)}
    print("M=%d per-step W4A4 GEMM chain total %.4f ms" % (m, tot), flush=True)
print(json.dumps({"batch": batch, "output": {"class": "reordered" if has_fi else "exact"},
                  "detail": detail}))
