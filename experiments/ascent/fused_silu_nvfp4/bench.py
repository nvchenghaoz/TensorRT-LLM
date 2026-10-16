import sys, json, torch
import tensorrt_llm
from tensorrt_llm._torch.custom_ops import torch_custom_ops
from tensorrt_llm._torch.autotuner import autotune
arm = sys.argv[1] if len(sys.argv) > 1 else "baseline"
fused = None
if arm == "candidate":
    from tensorrt_llm._torch.modules.fused_silu_mul_nvfp4 import _fused_silu_mul_nvfp4_impl as fused
I, H = 17408, 5120
N = 2 * I
dev = "cuda"
torch.manual_seed(0)
NC, L, WARM, ITERS = 4, 8, 10, 50
ws = [torch.randint(0, 256, (N, H // 2), dtype=torch.uint8, device=dev) for _ in range(NC)]
wsf = [torch.full((N * (H // 16),), 0x30, dtype=torch.uint8, device=dev) for _ in range(NC)]
alpha = torch.tensor([0.01], dtype=torch.float32, device=dev)
gs = torch.tensor([1.0], dtype=torch.float32, device=dev)
sc = torch.tensor([30.0], dtype=torch.float32, device=dev)
print("arm", arm, "weights MB", NC * N * H // 2 / 2**20, flush=True)
def act(h1):
    if fused is not None:
        return fused(h1, sc)
    return torch.ops.trtllm.fp4_quantize(torch.ops.trtllm.silu_and_mul(h1), sc, 16, False)
def gemm(xq, xsf, k):
    return torch.ops.trtllm.nvfp4_gemm(xq, ws[k], xsf, wsf[k], alpha, torch.bfloat16)
def capture(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    return g, out
def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * (len(v) - 1) + 0.5))]
res, detail = {}, {}
for M in [1, 2, 4]:
    x = torch.randn(M, H, device=dev, dtype=torch.bfloat16)
    xq, xsf = torch.ops.trtllm.fp4_quantize(x, gs, 16, False)
    with autotune():
        for k in range(NC):
            gemm(xq, xsf, k)
    for k in range(NC):
        act(gemm(xq, xsf, k))
    torch.cuda.synchronize()
    chain = lambda: [act(gemm(xq, xsf, l % NC)) for l in range(L)]
    gonly = lambda: [gemm(xq, xsf, l % NC) for l in range(L)]
    gc, oc = capture(chain)
    gg, og = capture(gonly)
    for _ in range(WARM):
        gc.replay(); gg.replay()
    torch.cuda.synchronize()
    e = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    d, tcs, tgs = [], [], []
    for _ in range(ITERS):
        e[0].record(); gc.replay(); e[1].record(); gg.replay(); e[2].record()
        torch.cuda.synchronize()
        tc = e[0].elapsed_time(e[1]) / L
        tg = e[1].elapsed_time(e[2]) / L
        tcs.append(tc); tgs.append(tg); d.append(tc - tg)
    med = q(d, 0.5)
    res[str(M)] = {"kernel_ms": med, "noise_floor": (q(d, 0.9) - q(d, 0.1)) / max(abs(med), 1e-9)}
    detail[str(M)] = {"act_exposed_ms_median": med, "act_exposed_ms_p90": q(d, 0.9),
                      "chain_per_layer_ms_median": q(tcs, 0.5), "chain_per_layer_ms_p90": q(tcs, 0.9),
                      "gemm_only_per_layer_ms_median": q(tgs, 0.5), "samples": len(d)}
    print("M", M, detail[str(M)], flush=True)
    del gc, gg, oc, og
M = 2048
h1s = [torch.randn(M, N, device=dev, dtype=torch.bfloat16) for _ in range(2)]
for i in range(WARM):
    act(h1s[i % 2])
torch.cuda.synchronize()
t = []
for i in range(ITERS):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record(); act(h1s[i % 2]); b.record(); torch.cuda.synchronize()
    t.append(a.elapsed_time(b))
med = q(t, 0.5)
res[str(M)] = {"kernel_ms": med, "noise_floor": (q(t, 0.9) - q(t, 0.1)) / med}
detail[str(M)] = {"median_ms": med, "p90_ms": q(t, 0.9), "mode": "eager, 2 rotated 142MB inputs"}
print("M", M, detail[str(M)], flush=True)
print(json.dumps({"batch": res, "output": {"class": "exact"}, "arm": arm, "detail": detail,
                  "note": "M<=4: per-layer exposed time of the SwiGLU->NVFP4 step = (graph[gate_up GEMM->act]x8 - graph[gate_up GEMM]x8)/8, 4x89MB distinct gate_up weights rotated (cold L2), paired per iteration; M=2048 eager act only"}))
