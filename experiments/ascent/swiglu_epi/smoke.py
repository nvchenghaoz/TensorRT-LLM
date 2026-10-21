"""Quick in-tree smoke: op present; fused module path == kill-switch module path
(bitwise) at a few M. Not the protected evidence (that is check.py vs BASE files)."""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlp_common import DownInputTap, build_mlp, make_input, tree_info  # noqa: E402

print("TREE", json.dumps(tree_info()), flush=True)
mlp = build_mlp(11, 1.0)
tap = DownInputTap(mlp)
os.environ["TRTLLM_GATEUP_SWIGLU_EPI"] = "0"
off = build_mlp(11, 1.0)
os.environ.pop("TRTLLM_GATEUP_SWIGLU_EPI")
tap_off = DownInputTap(off)
print("permuted", mlp._swiglu_epi_permuted, off._swiglu_epi_permuted, flush=True)
ok = mlp._swiglu_epi_permuted and not off._swiglu_epi_permuted
res = {}
for m in (1, 4, 16, 17, 2048):
    x = make_input(m, 5 + m, quantized=m >= 2048)
    a = mlp(x)
    fa = tap.last
    b = off(x)
    fb = tap_off.last
    torch.cuda.synchronize()
    e = (torch.equal(a, b), torch.equal(fa[0], fb[0]), torch.equal(fa[1], fb[1]) if fa[1] is not None else None)
    res[m] = e
    ok &= all(v is not False for v in e)
    print("M", m, e, flush=True)
print(json.dumps({"passed": bool(ok), "res": {str(k): v for k, v in res.items()}}))
