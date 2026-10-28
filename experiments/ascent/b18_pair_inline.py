import json, torch
import tensorrt_llm
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp
H, K = 3072, 3072
def mk(M, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, generator=g)
    ws = [torch.randn(H, K, device="cuda", dtype=torch.bfloat16, generator=g) * s for s in (0.02, 0.035, 0.012)]
    def q(t):
        s = float(t.abs().amax().float() / 2688.0)
        d, sf = fp.nvfp4_quantize(t, torch.tensor([1.0 / s], device="cuda"))
        return d, sf, s
    a, asf, ags = q(x)
    qw = [q(w) for w in ws]
    w = torch.cat([t[0] for t in qw]).contiguous()
    wsf = torch.cat([t[1] for t in qw]).contiguous()
    qs = torch.tensor([ags * t[2] for t in qw], device="cuda", dtype=torch.float32)
    qn = (1 + 0.1 * torch.randn(128, device="cuda", generator=g)).bfloat16()
    kn = (1 + 0.1 * torch.randn(128, device="cuda", generator=g)).bfloat16()
    th = torch.rand(M, 64, device="cuda", generator=g) * 6.28
    cs = torch.cat([th.cos(), th.sin()], -1).float().contiguous()
    return (a, w, asf, wsf, ags, qw[0][2], qn, kn, cs, None, qs)
op = torch.ops.trtllm.flux3_primsts_nvfp4_qkv_rope
Ms = (4096, 4176, 4256)
cases = {M: mk(M, M) for M in Ms}
for M in Ms:
    for _ in range(5):
        op(*cases[M])
torch.cuda.synchronize()
N = 60
ev = {M: [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(N)] for M in Ms}
for i in range(N):
    for M in Ms:
        ev[M][i][0].record(); op(*cases[M]); ev[M][i][1].record()
torch.cuda.synchronize()
res = {}
for M in Ms:
    t = sorted(s.elapsed_time(e) for s, e in ev[M])
    med = t[N // 2]
    res[M] = (med, (t[(N * 9) // 10] - t[N // 10]) / med)
    print(M, "%.1f us" % (med * 1e3), "noise %.4f" % res[M][1], flush=True)
print(json.dumps({"batch": {"1": {"kernel_ms": res[4176][0], "noise_floor": res[4176][1]}},
                  "per_M_ms": {str(M): list(res[M]) for M in Ms}, "output": {"class": "exact"}}))
