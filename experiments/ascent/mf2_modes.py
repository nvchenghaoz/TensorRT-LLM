"""Time and byte-check the fused junction kernel per SSQ mode vs main's chain.
usage: mf2_modes.py <checkpoint> [modes=0,2,3]"""
import json
import os
import sys

import torch

CK = sys.argv[1]
MODES = [int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "0,2,3").split(",")]
cfg = json.load(open(os.path.join(CK, "config.json")))
tc = cfg.get("text_config", cfg)
H, EPS = int(tc["hidden_size"]), float(tc.get("rms_norm_eps", 1e-6))
print("H", H, flush=True)

from tensorrt_llm._torch.kernels.fused_ops import rmsnorm_residual_add as rra  # noqa: E402
from tensorrt_llm._torch.kernels.fused_ops.rmsnorm_fp4_quant import rmsnorm_fp4_quant  # noqa: E402

dev = "cuda"
flush = torch.empty(64 * 1024 * 1024, dtype=torch.float32, device=dev)


def make(m, seed=5):
    g = torch.Generator(device=dev).manual_seed(seed + m)
    chs = torch.exp(torch.randn(H, device=dev, generator=g))
    x = (torch.randn(m, H, device=dev, generator=g) * 4.0 * chs).to(torch.bfloat16)
    r = (torch.randn(m, H, device=dev, generator=g) * 20.0 * chs.flip(0)).to(torch.bfloat16)
    w = (torch.rand(H, device=dev, generator=g) * 2 + 0.05).to(torch.bfloat16)
    w2 = (torch.rand(H, device=dev, generator=g) * 4 + 0.05).to(torch.bfloat16)
    gs = torch.tensor([10.0], device=dev)
    return x, r, w, w2, gs


def chain(x, r, w, w2, gs):
    h = rra.rmsnorm_residual_add(x, r, w, EPS)
    return (h,) + tuple(rmsnorm_fp4_quant(h, w2, EPS, gs))


def fused(x, r, w, w2, gs):
    return rra.rmsnorm_residual_add_fp4_quant(x, r, w, EPS, w2, EPS, gs)


def tm(fn, a, it=20):
    for _ in range(3):
        fn(*a)
    ts = []
    for _ in range(it):
        flush.fill_(1.0)
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn(*a)
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


res = {}
for m in (300, 1000, 8000):
    a = make(m)
    res[f"chain_{m}"] = tm(chain, a)
    print(f"chain M={m}: {res[f'chain_{m}']:.1f} us", flush=True)
    ref = chain(*a)
    for mode in MODES:
        rra._FP4_SSQ_MODE = mode
        t = tm(fused, a)
        o = fused(*a)
        d = [(o[0] != ref[0]).sum().item(), (o[1] != ref[1]).sum().item(), (o[2] != ref[2]).sum().item()]
        res[f"mode{mode}_{m}"] = {"us": t, "diff_h_fp4_sf": d}
        print(f"mode {mode} M={m}: {t:.1f} us diffs h/fp4/sf={d}", flush=True)
try:
    k = rra._rmsnorm_residual_add_fp4_kernel
    for dc in k.device_caches.values():
        for ck in dc[0].values():
            print("compiled:", ck.metadata.num_warps if hasattr(ck, "metadata") else "",
                  "n_regs", getattr(ck, "n_regs", None), "n_spills", getattr(ck, "n_spills", None),
                  "shared", getattr(ck.metadata, "shared", None), flush=True)
except Exception as e:  # noqa: BLE001
    print("no compiled stats:", repr(e), flush=True)
print(json.dumps({"passed": True, "res": res}))
