"""GatedMLP end-to-end: fused path vs unfused path (same module, same weights).

Builds an NVFP4 W4A4 GatedMLP (hidden 5120, intermediate 17408) with random
weights, runs forward with the fused SwiGLU+NVFP4 path on and off, compares
outputs bitwise (eager and CUDA-graph replay, repeated), and lists the kernels
launched between gate_up and down_proj via torch.profiler.
"""
import json

import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

H, I = 5120, 17408
dev = "cuda"
torch.manual_seed(0)
qc = QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=16)
mlp = GatedMLP(hidden_size=H, intermediate_size=I, bias=False,
               dtype=torch.bfloat16, config=ModelConfig(quant_config=qc),
               layer_idx=0).to(dev)
with torch.no_grad():
    for lin in (mlp.gate_up_proj, mlp.down_proj):
        lin.weight.data.copy_(torch.randint(0, 256, lin.weight.shape,
                                            dtype=torch.uint8))
        lin.weight_scale.data.fill_(0x30)
        lin.input_scale.data.fill_(30.0 if lin is mlp.down_proj else 1.0)
        lin.alpha.data.fill_(0.002)
        lin.weight_scale_2.data.fill_(1.0)
print("down quant_method", type(mlp.down_proj.quant_method).__name__,
      flush=True)


def run(x, fused):
    mlp._fused_swiglu_nvfp4_ok = None if fused else False
    return mlp(x)


passed = True
res = {"gate_open": None}
for M in [1, 2, 4, 8, 512, 2048]:
    x = (torch.randn(M, H, device=dev) * 0.5).to(torch.bfloat16)
    ref = run(x, False)
    out = run(x, True)
    res["gate_open"] = bool(mlp._fused_swiglu_nvfp4_ok)
    eq = torch.equal(out, ref)
    bad = 0
    for _ in range(60):
        if not torch.equal(run(x, True), ref):
            bad += 1
    gbad = None
    if M <= 8:
        mlp._fused_swiglu_nvfp4_ok = None
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                mlp(x)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            gout = mlp(x)
        gbad = 0
        for _ in range(60):
            gout.fill_(float("nan"))
            g.replay()
            torch.cuda.synchronize()
            if not torch.equal(gout, ref):
                gbad += 1
        del g
    res[f"M{M}"] = {"eager_eq": eq, "eager_repeat_bad": bad,
                    "graph_replay_bad": gbad,
                    "nan": bool(torch.isnan(ref.float()).any())}
    passed = passed and eq and bad == 0 and (gbad in (None, 0))
    print(M, res[f"M{M}"], flush=True)


def kernels(fused, M=1):
    x = (torch.randn(M, H, device=dev) * 0.5).to(torch.bfloat16)
    run(x, fused)
    torch.cuda.synchronize()
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        run(x, fused)
        torch.cuda.synchronize()
    evs = [e for e in prof.events() if e.device_type ==
           torch.autograd.DeviceType.CUDA]
    evs.sort(key=lambda e: e.time_range.start)
    return [e.name[:90] for e in evs]


kf, ku = kernels(True), kernels(False)
res["kernels_fused_M1"] = kf
res["kernels_unfused_M1"] = ku
print("fused kernels", kf, flush=True)
print("unfused kernels", ku, flush=True)

def between_gemms(ks):
    # Kernels between the gate_up GEMM and the down_proj GEMM (the first
    # quantize_with_block_size is gate_up's own input quantize).
    gi = [i for i, k in enumerate(ks) if "Gemm" in k or "gemm" in k]
    return ks[gi[0] + 1:gi[-1]] if len(gi) >= 2 else ks


bf, bu = between_gemms(kf), between_gemms(ku)
res["between_gemms_fused"] = bf
res["between_gemms_unfused"] = bu
res["fused_list_ok"] = (len(bf) == 1 and "fused_silu_mul_nvfp4" in bf[0]
                        and len(bu) == 2)
passed = passed and res["fused_list_ok"] and res["gate_open"]
print(json.dumps({"passed": passed,
                  "compared": "GatedMLP NVFP4 W4A4 output, fused path vs "
                              "unfused silu_and_mul+fp4_quantize path, bitwise",
                  "results": res}))
