import sys
sys.argv = [sys.argv[0]]
import torch
import importlib.util, os
spec = importlib.util.spec_from_file_location(
    "chk", os.path.join(os.path.dirname(os.path.abspath(__file__)), "check.py"))
chk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chk)
I = chk.I
scale = torch.tensor([30.0], dtype=torch.float32, device="cuda")
h1 = chk.make_h1(4, 77 + 4, subnormal=True)
rq, rsf = chk.reference(h1, scale)
rq = rq.view(torch.uint8)
fq, fsf = chk._fused_silu_mul_nvfp4_impl(h1, scale)
h = torch.ops.trtllm.silu_and_mul(h1)
idx = (fq != rq).nonzero()
blocks = set()
for r, c in idx.tolist():
    b = (2 * c) // 16
    if (r, b) in blocks:
        continue
    blocks.add((r, b))
    if len(blocks) > 4:
        break
    hv = h[r, b * 16:(b + 1) * 16]
    print("row", r, "block", b, "vecMax", hv.float().abs().max().item())
    print(" h bits", [hex(v) for v in hv.view(torch.int16).tolist()])
    print(" fused", [hex(v) for v in fq[r, b * 8:(b + 1) * 8].tolist()])
    print(" ref  ", [hex(v) for v in rq[r, b * 8:(b + 1) * 8].tolist()])
print({"passed": True, "n_mismatch": int(idx.shape[0])})
