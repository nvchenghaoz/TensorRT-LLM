"""Dump the reduce-operand layout of main's _fused_gdn_post_conv_kernel per specialization."""
import re

import torch

from tensorrt_llm._torch.modules.mamba.fuse_elementwise_ops import (
    _fused_gdn_post_conv_kernel,
    fused_gdn_post_conv,
)

dev = torch.device("cuda")
A_log = torch.randn(48, device=dev)
dt = torch.randn(48, device=dev)
cases = [(2048, 0), (2048, 1), (2048, 3), (85, 0), (85, 3), (1, 0), (2, 0), (3, 0), (8, 0),
         (24, 0), (2056, 0), (2050, 0), (4100, 0), (8192, 0), (8192, 3), (6144, 2)]
for T, D in cases:
    x = torch.randn(10240, T, device=dev).to(torch.bfloat16)
    ab = torch.randn(T + D, 96, device=dev).to(torch.bfloat16)
    dec = torch.randn(D, 10240, device=dev).to(torch.bfloat16) if D else None
    before = sum(len(c[0]) for c in _fused_gdn_post_conv_kernel.device_caches.values())
    fused_gdn_post_conv(x, dec, ab[:, 48:], ab[:, :48], A_log, dt, 16, 128, 48, 128)
    torch.cuda.synchronize()
    after = sum(len(c[0]) for c in _fused_gdn_post_conv_kernel.device_caches.values())
    print("case T=%d D=%d new_kernels=%d" % (T, D, after - before), flush=True)
    if after > before:
        for c in _fused_gdn_post_conv_kernel.device_caches.values():
            items = list(c[0].items())
            k, kern = items[-1]
            tt = kern.asm["ttgir"]
            ks = re.sub(r"\s+", "", str(k))
            print("   KEY", ks[:700])
            for line in tt.splitlines():
                if line.startswith("#blocked") or line.startswith("#linear") or line.startswith("#shared"):
                    print("   ", line[:300])
            for line in tt.splitlines():
                if "tt.reduce\"" in line and "q_inv_norm" in line:
                    pass
                if "}) : (tensor<" in line and "-> tensor<" in line:
                    print("    REDUCE", line.strip()[:200])
            for line in tt.splitlines():
                if "tt.load" in line:
                    m = re.search(r"%(\w+) = tt.load .*: (tensor<[^>]*>)", line.strip())
                    if m:
                        print("    LOAD", m.group(1), m.group(2)[:120])
print("DONE")
