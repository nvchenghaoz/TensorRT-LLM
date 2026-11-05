"""Dump the PTX ops of b12x_wo._inv_rope_quant128_kernel (K1) that decide its quant bytes (idea 233 S1).

K2' must reproduce K1's inverse RoPE + 1x128 UE8M0 quant byte for byte, so it needs the exact
division / log2 / exp2 / ceil / convert instructions Triton emits.
"""
import json
import re

import torch

from tensorrt_llm._torch.attention.backends.sparse.csa2 import b12x_wo as B


def main():
    from b12x.gemm._shared import wo_mxfp8 as W

    M, G, GW, H, HD = 2, 2, 4096, 16, 512
    o3 = torch.randn(M, H, HD, device="cuda").to(torch.bfloat16)
    pos = torch.zeros(M, dtype=torch.int32, device="cuda")
    cs = torch.randn(1024, 2, 32, device="cuda").float()
    x_q = W.empty_mxfp8_rows_for_dense_gemm(M, GW, num_groups=G, device=torch.device("cuda"))
    sm = x_q.scale_mma.view(torch.uint8)
    vals = x_q.values
    print("values", tuple(vals.shape), vals.stride(), vals.dtype, flush=True)
    print("scale_mma", tuple(sm.shape), sm.stride(), flush=True)
    k = B._inv_rope_quant128_kernel[(M, G, GW // 128 // 4)](
        o3, pos, cs, vals, sm, o3.stride(0), cs.stride(0), vals.stride(0), vals.stride(1),
        vals.stride(2) if vals.ndim == 3 else 0, *sm.stride(), 448.0, -448.0, o3,
        GROUP_WIDTH=GW, HEAD_DIM=HD, NOPE_DIM=448, HALF_ROPE=32, NB=4, ROPE_VARIANT=0, STORE_ROPE=False,
        num_warps=4)
    torch.cuda.synchronize()
    ptx = k.asm["ptx"]
    ops = {}
    for line in ptx.splitlines():
        s = line.strip()
        m = re.match(r"([a-z][a-z0-9_.]*)", s)
        if not m:
            continue
        op = m.group(1)
        if any(op.startswith(p) for p in ("div", "lg2", "ex2", "cvt", "max", "min", "fma", "mul", "neg", "rcp",
                                          "abs", "sub", "add.f", "and", "shr", "selp", "setp", "call")):
            ops.setdefault(op, 0)
            ops[op] += 1
    for op, n in sorted(ops.items()):
        print(f"{op:40s} {n}", flush=True)
    # Print the float-op neighbourhood (div/lg2/ex2/cvt.rpi) verbatim.
    lines = ptx.splitlines()
    for i, line in enumerate(lines):
        if re.search(r"\b(div|lg2|ex2|cvt\.rpi|cvt\.rn\.satfinite|cvt\.rn\.bf16|max\.f32|min\.f32)", line):
            print("PTX", i, line.strip(), flush=True)
    # Verbatim body from the RoPE math through the e4m3 convert (the log2 is a libdevice routine).
    first = next(i for i, l in enumerate(lines) if "fma.rn" in l or "mul.f32" in l)
    last = max(i for i, l in enumerate(lines) if "cvt.rn.satfinite.e4m3x2" in l)
    for i in range(max(0, first - 30), last + 2):
        print("RAW %4d %s" % (i, lines[i].rstrip()), flush=True)
    print(json.dumps({"passed": True, "n_ptx_lines": len(lines)}))


if __name__ == "__main__":
    main()
