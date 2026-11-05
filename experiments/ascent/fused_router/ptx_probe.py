"""Dump the arithmetic instructions of FlashInfer's _mxfp8_q0_route_direct_kernel and _route_assign_decode_kernel
(Triton) PTX, so the fused router's Stage 2 quant matches them instruction for instruction."""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import E_LOCAL, HIDDEN, log  # noqa: E402

import torch  # noqa: E402


def main():
    import tensorrt_llm  # noqa: F401
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_mxfp8_q0_route_triton as mod
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x._moe_utils import moe_route_meta as meta

    x = torch.randn(2, HIDDEN, device="cuda").to(torch.bfloat16)
    ids = torch.randint(0, 384, (2, 6), device="cuda", dtype=torch.int32)
    w = torch.rand(2, 6, device="cuda")
    ws = mod.make_mxfp8_q0_route_workspace(x, ids, E_LOCAL)
    buf = torch.empty(4096, dtype=torch.uint8, device="cuda")
    mod.mxfp8_q0_route_triton(x, ids, w, E_LOCAL, workspace=ws, enable_pdl=True, expert_offset=0, zero_buf=buf)
    torch.cuda.synchronize()
    log("scale_out", tuple(ws.scale_out.shape), ws.scale_out.stride(), "q_out", tuple(ws.q_out.shape))
    for name, kern in (("q0_direct", mod._mxfp8_q0_route_direct_kernel), ("route_decode", meta.route_assign_decode_kernel)):
        caches = getattr(kern, "device_caches", None) or getattr(kern, "cache", None)
        found = False
        if isinstance(caches, dict):
            for dev, c in caches.items():
                entries = c[0] if isinstance(c, tuple) else c
                for key, ck in (entries.items() if isinstance(entries, dict) else []):
                    ptx = ck.asm.get("ptx", "")
                    found = True
                    ops = sorted(set(re.findall(r"\b(div\.[a-z0-9.]+|rcp\.[a-z0-9.]+|cvt\.[a-z0-9.]+|max\.[a-z0-9.]+|"
                                                r"mul\.[a-z0-9.]+|griddepcontrol\.[a-z_]+|fma\.[a-z0-9.]+)", ptx)))
                    log(f"{name}: {ops}")
                    for ln in ptx.splitlines():
                        if re.search(r"div\.|rcp\.|cvt\.rn|e4m3|griddepcontrol", ln):
                            print("   ", ln.strip(), flush=True)
                    break
        if not found:
            log(f"{name}: no cache found; attrs {[a for a in dir(kern) if 'cache' in a]}")
    print('{"passed": true}')


if __name__ == "__main__":
    main()
