"""Run selected configs once each at M=9600 gate_up (plain) for ncu."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage0 as S  # noqa: E402

names = sys.argv[1].split(",")
fused = len(sys.argv) > 2 and sys.argv[2] == "fused"
m, n, k = 9600, 14336, 5376
x = S.make_inputs(m, k, 0)
W = S.make_weight(n, k)
for name in names:
    S.G.gate_up_sm120(x, W, fused=fused, cfg=S.CFGS[name])
    torch.cuda.synchronize()
    print("ran", name, flush=True)
