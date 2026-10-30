"""Branch 57 ncu harness (job-642 shape, fact 900): routed modules, 3 warm calls per shape,
cudaProfilerStart/Stop around: cand nfo(16464), cand nfo(32848), cand bf16(16464), base nfo(16464).
Run under profile_kernels capture=region kernel_regex fmha_v2_flux3."""

from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common as c  # noqa: E402

b = c.b


def main():
    lib = c.lib_mod()
    routed, levs = c.arm_levers(lib, b.make_qkv(4176, 7, False))
    mods = c.build(lib, levs, ("base", "cand"))
    gscale = torch.tensor([336.0], dtype=torch.float32, device="cuda")
    inps = {S: b.make_qkv(S, 900 + S, False) for S in (16464, 32848)}
    outs = {S: b.NfoOut(S, post=4096) for S in inps}
    o = torch.empty((1, 16464, b.H, b.D), device="cuda", dtype=torch.bfloat16)
    launches = [("cand_nfo_16464", lambda: b.call_nfo(lib, mods["cand"]["nfo"], inps[16464],
                                                       outs[16464], gscale)),
                ("cand_nfo_32848", lambda: b.call_nfo(lib, mods["cand"]["nfo"], inps[32848],
                                                       outs[32848], gscale)),
                ("cand_bf16_16464", lambda: b.call_bf16(lib, mods["cand"]["bf16"], inps[16464], o)),
                ("base_nfo_16464", lambda: b.call_nfo(lib, mods["base"]["nfo"], inps[16464],
                                                       outs[16464], gscale))]
    for _ in range(3):
        for _, f in launches:
            f()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for name, f in launches:
        f()
        torch.cuda.synchronize()
        print(f"[launch] {name}", flush=True)
    torch.cuda.cudart().cudaProfilerStop()
    print(json.dumps({"order": [n for n, _ in launches], "levers": {k: list(v) for k, v in
                                                                     levs.items()}}))


if __name__ == "__main__":
    main()
