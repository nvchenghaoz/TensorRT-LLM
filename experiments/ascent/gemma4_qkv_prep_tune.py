"""Tile sweep for _gemma4_qkv_norm_rope_quant_kernel on this GPU (BF16 Q, FP8 K/V)."""
import json
import sys

import torch
import triton

from tensorrt_llm._torch.modules.gemma4.fused_qkv import _gemma4_qkv_norm_rope_quant_kernel


def t(fn, it=50, wu=10):
    for _ in range(wu):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(it):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def main():
    dev = "cuda"
    res = {}
    for name, nq, nk, hd in (("sliding", 32, 16, 256), ("global", 32, 4, 512)):
        for n in (2000, 8000):
            w = (nq + 2 * nk) * hd
            qkv = torch.randn(n, w, device=dev).to(torch.bfloat16)
            pos = torch.arange(n, device=dev, dtype=torch.int32)
            cs = torch.randn(16384, 2, hd // 2, device=dev)
            qw = torch.ones(hd, device=dev, dtype=torch.bfloat16)
            q = torch.empty(n, nq * hd, device=dev, dtype=torch.bfloat16)
            k = torch.empty(n, nk * hd, device=dev, dtype=torch.float8_e4m3fn)
            v = torch.empty(n, nk * hd, device=dev, dtype=torch.float8_e4m3fn)
            nbytes = qkv.numel() * 2 + q.numel() * 2 + k.numel() + v.numel()
            src = torch.empty(nbytes // 2, device=dev, dtype=torch.uint8)
            dst = torch.empty_like(src)
            cp = t(lambda: dst.copy_(src))
            print(f"{name} n={n} bytes={nbytes/1e6:.0f}MB copy-ceiling {cp*1e3:.1f}us "
                  f"({nbytes/cp/1e9:.0f} GB/s)", flush=True)
            for bn in (2, 4, 8, 16, 32):
                for nw in (2, 4, 8, 16):
                    if bn * hd // 2 < 32 * nw:
                        continue
                    grid = (nq + 2 * nk, triton.cdiv(n, bn))

                    def f():
                        _gemma4_qkv_norm_rope_quant_kernel[grid](
                            qkv, q, k, v, pos, cs, qw, qw, qkv.stride(0), q.stride(0), k.stride(0),
                            v.stride(0), nq, nk, 1e-6, n, HD=hd, HALF=hd // 2, Q_OUT_FP8=False,
                            KV_OUT_FP8=True, BLOCK_N=bn, num_warps=nw)
                    ms = t(f)
                    res[f"{name}_{n}_bn{bn}_w{nw}"] = ms
                    print(f"  bn={bn} warps={nw}: {ms*1e3:.1f}us {nbytes/ms/1e9:.0f} GB/s", flush=True)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
