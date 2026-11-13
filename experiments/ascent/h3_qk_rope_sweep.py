"""Sweep the launch shape of _minimax_h3_qk_norm_rope_kernel at the TP2 per-rank H3 shape (SM120).

Each shape is checked bitwise against the current default launch and timed (CUDA events, median of
>=20 after >=5 warmup). Prints one JSON line at the end.
"""

import json
import os
import sys

import torch

from tensorrt_llm._torch.visual_gen.models.minimax_h3 import fused_rope as fr

CKPT = sys.argv[1] if len(sys.argv) > 1 else None
M = int(os.environ.get("H3_M", "37966"))
WARM, ITERS = 5, 25


def cfg():
    heads, hd, rfd = 56, 128, 16
    if CKPT:
        with open(os.path.join(CKPT, "transformer", "config.json")) as f:
            c = json.load(f)
        heads, hd, rfd = c["num_attention_heads"], c["attention_head_dim"], c["rope_freq_dim"]
    return heads // 2, hd, rfd


def time_fn(fn):
    for _ in range(WARM):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(ITERS):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    med = ts[len(ts) // 2]
    return med, (ts[int(0.9 * len(ts))] - ts[int(0.1 * len(ts))]) / med


def main():
    torch.manual_seed(0)
    H, hd, rfd = cfg()
    rot = 6 * rfd
    print(f"H_local={H} hd={hd} rotary_dim={rot} M={M} cap={torch.cuda.get_device_capability()}", flush=True)
    dev = "cuda"
    qkv = torch.randn(1, M, 3 * H * hd, device=dev, dtype=torch.bfloat16)
    wq = (1 + 0.1 * torch.randn(hd, device=dev)).to(torch.bfloat16)
    wk = (1 + 0.1 * torch.randn(hd, device=dev)).to(torch.bfloat16)
    pos = torch.stack(
        [torch.arange(M, device=dev) // 4096, (torch.arange(M, device=dev) // 64) % 64, torch.arange(M, device=dev) % 64], -1
    )
    inv = 1.0 / (10000.0 ** (torch.arange(0, 2 * rfd, 2, dtype=torch.float32, device=dev) / (2 * rfd)))
    fq = pos.float().unsqueeze(-1) * inv.view(1, 1, -1)
    fq = torch.cat(fq.unbind(dim=1), dim=-1)
    fq = torch.cat((fq, fq), dim=-1)
    cos, sin = fq.cos(), fq.sin()
    eps = 1e-6
    q = torch.empty(1, M, H * hd, device=dev, dtype=torch.bfloat16)
    k = torch.empty_like(q)

    default_hpp = next(h for h in (8, 4, 2, 1) if H % h == 0)
    fr.launch_minimax_h3_qk_norm_rope(qkv, q, k, wq, wk, cos, sin, eps, H, hd)
    q0, k0 = q.clone(), k.clone()
    res = {}
    for T in (1, 2, 4, 8):
        for HP in (1, 2, 4):
            for W in (1, 2, 4):
                if H % HP or (T * HP) & (T * HP - 1):
                    continue
                q.zero_()
                k.zero_()

                def fn(T=T, HP=HP, W=W):
                    fr.launch_minimax_h3_qk_norm_rope(
                        qkv, q, k, wq, wk, cos, sin, eps, H, hd,
                        tokens_per_program=T, heads_per_program=HP, num_warps=W,
                    )

                fn()
                torch.cuda.synchronize()
                exact = bool(torch.equal(q, q0) and torch.equal(k, k0))
                med, spread = time_fn(fn)
                key = f"T{T}_H{HP}_W{W}"
                res[key] = {"ms": round(med, 4), "spread": round(spread, 4), "bitwise": exact}
                print(key, res[key], flush=True)
    base = res[f"T1_H{default_hpp}_W1"]
    best = min((v["ms"], k_) for k_, v in res.items() if v["bitwise"])
    print(json.dumps({"default": base, "best_bitwise": best, "all": res}))


if __name__ == "__main__":
    main()
