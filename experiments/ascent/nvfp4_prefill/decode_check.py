"""Item 4: rows <= 16 keep main's FP8 path bitwise, eager and CUDA-graph.

For M in {1,2,3,4,8,16} on (16384,5120): FP8QDQLinearMethod.apply with the
NVFP4 prefill copy built (this branch) vs main's FP8 ops called directly
(static_quantize_e4m3_per_tensor + cuda_scaled_mm for M<=8, cublas_scaled_mm
otherwise), and vs apply with the copy removed. Also captures apply in a
CUDA graph per M and replays it 60 times (10 warmup + 50, same counts as the
timing), checking every replay bitwise against the eager main output. Also
checks M=17 takes the NVFP4 path (differs from FP8) as a sanity check.
"""
import json

import torch

from tensorrt_llm._torch.modules import fp8_prefill_nvfp4 as fpn
from tensorrt_llm._torch.modules.linear import FP8QDQLinearMethod

torch.manual_seed(0)
dev = "cuda"
N, K = 16384, 5120
W = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
ws = (W.float().abs().amax() / 448.0).reshape(())
wq = (W.float() / ws).to(torch.float8_e4m3fn)


class M(torch.nn.Module):
    pass


m = M()
m.quant_method = FP8QDQLinearMethod()
m.weight = torch.nn.Parameter(wq, requires_grad=False)
m.weight_scale = torch.nn.Parameter(ws.float().clone(), requires_grad=False)
m.input_scale = torch.nn.Parameter(torch.tensor(8.0 / 448.0, device=dev),
                                   requires_grad=False)
m.force_dynamic_quantization = False
m.enable_cuda_core = True
m.all_reduce = None
m.mapping = None
m.dtype = torch.bfloat16
built = fpn.maybe_build_prefill_nvfp4_copy(m)
print(f"copy built: {built}", flush=True)


def main_fp8(x):
    q, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
        x, m.input_scale)
    op = (torch.ops.trtllm.cuda_scaled_mm
          if x.shape[0] <= 8 else torch.ops.trtllm.cublas_scaled_mm)
    return op(q, m.weight.t(), scale_a=m.input_scale, scale_b=m.weight_scale,
              bias=None, out_dtype=torch.bfloat16)


res = {}
ok = built
for Mr in (1, 2, 3, 4, 8, 16):
    x = (torch.randn(Mr, K, device=dev) * 1.5).to(torch.bfloat16)
    ref = main_fp8(x)
    y = m.quant_method.apply(m, x, None)
    copy = m.__dict__.pop("_fp8_prefill_nvfp4")
    y_nocopy = m.quant_method.apply(m, x, None)
    m.__dict__["_fp8_prefill_nvfp4"] = copy
    eager_eq = torch.equal(y, ref) and torch.equal(y_nocopy, ref)
    # CUDA graph capture of apply at this M
    xs = x.clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            m.quant_method.apply(m, xs, None)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        yg = m.quant_method.apply(m, xs, None)
    nbad = 0
    for i in range(60):
        xs.copy_(x)  # reset input each iteration
        g.replay()
        torch.cuda.synchronize()
        if not torch.equal(yg, ref):
            nbad += 1
    res[str(Mr)] = {"eager_bitwise_eq_main": eager_eq,
                    "graph_replays": 60, "graph_mismatches": nbad}
    ok = ok and eager_eq and nbad == 0
    print(f"M={Mr}: {res[str(Mr)]}", flush=True)

# sanity: M=17 takes the NVFP4 prefill path
x = (torch.randn(17, K, device=dev) * 1.5).to(torch.bfloat16)
calls0 = fpn.prefill_nvfp4_stats()["calls"]
y17 = m.quant_method.apply(m, x, None)
used = fpn.prefill_nvfp4_stats()["calls"] == calls0 + 1
ref17 = main_fp8(x)
rel = float((y17.float() - ref17.float()).norm() / ref17.float().norm())
res["17"] = {"nvfp4_path_used": used, "rel_rms_vs_fp8": rel}
print(f"M=17: {res['17']}", flush=True)
ok = ok and used
print(json.dumps({"passed": bool(ok), "compared": "apply() with NVFP4 copy "
                  "vs main FP8 ops (static quant + cuda/cublas_scaled_mm), "
                  "eager and CUDA-graph replays, rows<=16", "results": res}))
