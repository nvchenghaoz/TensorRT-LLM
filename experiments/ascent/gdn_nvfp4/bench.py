import json, os, torch
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo
# GDN in_proj_qkvz (16384 x 5120) FP8 per-tensor Linear, flagged for the
# decode NVFP4 copy (ignored by trees without it). Cold L2: one CUDA graph
# replays C distinct Linear copies back to back (>= 256 MB streamed per pass).
N, K, C, WARM, ITERS = 16384, 5120, 8, 10, 60
MS = [1, 2, 4, 8, 16]
dev = torch.device("cuda")
def log(*a): print(*a, flush=True)
def make(seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    w = torch.randn(N, K, generator=g, device=dev) * 0.02
    w = w * (torch.rand(N, 1, generator=g, device=dev) * 1.5 + 0.25)
    s = (w.abs().amax() / 448.0).float()
    w8 = (w / s).to(torch.float8_e4m3fn)
    lin = Linear(K, N, bias=False, dtype=torch.bfloat16,
                 quant_config=QuantConfig(quant_algo=QuantAlgo.FP8)).to(dev)
    lin._nvfp4_decode_copy = True
    lin.load_weights([{"weight": w8, "weight_scale": s.reshape(()),
                       "input_scale": torch.tensor(0.05, device=dev)}])
    lin.post_load_weights()
    return lin, w8, s
lins = []
for i in range(C):
    lins.append(make(1000 + i))
    log("built copy", i)
active = getattr(lins[0][0], "_nvfp4_decode_weight", None) is not None
log("nvfp4 decode copy active:", active, "mem GB", torch.cuda.memory_allocated() / 1e9)
res, errs = {}, {}
for m in MS:
    x = (torch.randn(m, K, device=dev, generator=torch.Generator(device=dev).manual_seed(m)) * 2.0).to(torch.bfloat16)
    s0 = torch.cuda.Stream()
    s0.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s0):
        for _ in range(3):
            outs = [l(x) for l, _, _ in lins]
    torch.cuda.current_stream().wait_stream(s0)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = [l(x) for l, _, _ in lins]
    for _ in range(WARM):
        g.replay()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(ITERS)]
    for a, b in ev:
        a.record(); g.replay(); b.record()
    torch.cuda.synchronize()
    t = sorted(a.elapsed_time(b) / C for a, b in ev)
    med, p10, p90 = t[len(t) // 2], t[len(t) // 10], t[(9 * len(t)) // 10]
    bytes_per_call = (N * K // 2 + N * K // 16) if active else N * K
    # error vs FP32 reference of the original math (FP8 weight * scale, BF16 x)
    l0, w8, s = lins[0]
    ref = x.float() @ (w8.float() * s).t()
    d = (outs[0].float() - ref).abs()
    errs[str(m)] = {"max_abs": d.max().item(), "mean_abs": d.mean().item(),
                    "rel_rms": ((outs[0].float() - ref).norm() / ref.norm()).item()}
    res[str(m)] = {"kernel_ms": med, "noise_floor": (p90 - p10) / med, "p90_ms": p90,
                   "p10_ms": p10, "tb_s": bytes_per_call / (med * 1e-3) / 1e12, "samples": ITERS}
    log(f"M={m} median {med*1e3:.2f} us p90 {p90*1e3:.2f} us TB/s {res[str(m)]['tb_s']:.3f} err {errs[str(m)]}")
    del g
cls = "precision" if active else "exact"
print(json.dumps({"batch": res, "output": {"class": cls, "nvfp4_decode_copy": active,
                  "error_vs_fp32_fp8_weight_reference": errs,
                  "shape": [N, K], "copies": C, "warmup": WARM, "iters": ITERS}}))
