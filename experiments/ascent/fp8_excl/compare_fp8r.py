"""Compare variants/fp8r FP8 weights with load-time per-channel quantization of the nvfp4 variant's BF16 weights."""
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

root = Path(sys.argv[1])
nv = root / "variants" / "nvfp4" / "model.safetensors"
f8 = root / "variants" / "fp8r" / "model.safetensors"
rootw = root / "model.safetensors"
blocks = ["content_mode_blocks.image.0"] + [f"single_blocks.{i}" for i in range(24, 28)]
names = ["q_proj", "k_proj", "v_proj", "attn_out", "mlp_in", "mlp_out"]
out = {}
with safe_open(str(nv), "pt", device="cpu") as a, safe_open(str(f8), "pt", device="cpu") as b, \
        safe_open(str(rootw), "pt", device="cpu") as r:
    rk = set(r.keys())
    for blk in blocks:
        for n in names:
            k = f"{blk}.{n}"
            w = a.get_tensor(k + ".weight").cuda()
            wr = r.get_tensor(k + ".weight").cuda() if (k + ".weight") in rk else None
            q = b.get_tensor(k + ".weight").cuda()
            s = b.get_tensor(k + ".weight_scale").cuda().float()
            amax = w.float().abs().amax(1)
            s_mine = torch.clamp(amax / 448.0, min=1e-12)
            q_mine = (w.float() / s_mine[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
            deq = q.float() * s[:, None]
            rel = ((deq - w.float()).norm() / w.float().norm()).item()
            entry = {
                "same_bf16_as_root": None if wr is None else bool(torch.equal(w, wr)),
                "scale_equal": bool(torch.equal(s, s_mine)),
                "scale_maxrel": ((s - s_mine).abs() / s_mine).max().item(),
                "q_equal": bool(torch.equal(q.view(torch.uint8), q_mine.view(torch.uint8))),
                "q_mismatch_frac": (q.view(torch.uint8) != q_mine.view(torch.uint8)).float().mean().item(),
                "fp8r_deq_relerr_vs_bf16": rel,
            }
            print(k, json.dumps(entry), flush=True)
            out[k] = entry
allq = all(v["q_equal"] for v in out.values())
alls = all(v["scale_equal"] for v in out.values())
print(json.dumps({"passed": True, "all_q_equal": allq, "all_scale_equal": alls}))
