import json
import torch
import triton
from tensorrt_llm._torch.kernels.fused_ops import rmsnorm_residual_add as rra
from tensorrt_llm._torch.kernels.fused_ops.rmsnorm_fp4_quant import rmsnorm_fp4_quant

H, EPS = 5376, 1e-6
dev = "cuda"
flush = torch.empty(64 * 1024 * 1024, dtype=torch.float32, device=dev)


def launch(x, r, w, w2, gs, nw):
    m, h = x.shape
    nkt = triton.cdiv(h // 16, 4)
    mp = triton.cdiv(m, 128) * 128
    o = torch.empty_like(x)
    q = torch.empty((m, h // 2), dtype=torch.uint8, device=dev)
    sf = torch.empty((mp * nkt * 4,), dtype=torch.uint8, device=dev)
    rra._rmsnorm_residual_add_fp4_kernel[(mp,)](
        x, r, w, o, w2, q, sf, gs, m, x.stride(0), r.stride(0), o.stride(0), EPS, EPS, nkt,
        H=h, BH=triton.next_power_of_2(h), DIV_RN=True, num_warps=nw)
    return o, q, sf


def t(fn, n=30):
    for _ in range(5):
        fn()
    ts = []
    for _ in range(n):
        flush.fill_(1.0)
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000)
    ts.sort()
    return ts[n // 2]


res = {}
for m in (300, 2000, 8000):
    g = torch.Generator(device=dev).manual_seed(m)
    x = (torch.randn(m, H, device=dev, generator=g) * 3).to(torch.bfloat16)
    r = (torch.randn(m, H, device=dev, generator=g) * 8).to(torch.bfloat16)
    w = (torch.rand(H, device=dev, generator=g) * 2 + .05).to(torch.bfloat16)
    w2 = (torch.rand(H, device=dev, generator=g) * 2 + .05).to(torch.bfloat16)
    gs = torch.tensor([448 * 6 / 40.0], device=dev)
    hb = rra.rmsnorm_residual_add(x, r, w, EPS)
    qb, sb = rmsnorm_fp4_quant(hb, w2, EPS, gs)
    tb = t(lambda: rra.rmsnorm_residual_add(x, r, w, EPS))
    tq = t(lambda: rmsnorm_fp4_quant(hb, w2, EPS, gs))
    print(f"M={m} main norm_add {tb:.1f}us cute norm_quant (h L2-cold) {tq:.1f}us", flush=True)
    for nw in (4, 8, 16):
        o, q, sf = launch(x, r, w, w2, gs, nw)
        torch.cuda.synchronize()
        hd = (o != hb).sum().item()
        qd = (q != qb).sum().item()
        tt = t(lambda: launch(x, r, w, w2, gs, nw))
        print(f"  nw={nw} {tt:.1f}us h_diff={hd} q_diff={qd} sf_eq={torch.equal(sf, sb)}", flush=True)
        res[f"{m}_nw{nw}"] = dict(us=tt, h_diff=hd, q_diff=qd)
print(json.dumps({"passed": True, "res": res}))
