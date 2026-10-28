"""Branch 32 diag: Inductor code + bits for mode-block LN+modulate, materialized vs broadcast forms."""

import json
import sys

import torch
import torch.nn.functional as F
from torch._inductor.utils import run_and_get_code

D = 3072
M = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
opts = {"emulate_precision_casts": True}
g = torch.Generator(device="cuda").manual_seed(0)


def rnd(*s, scale=1.0):
    return (torch.randn(*s, device="cuda", generator=g) * scale).to(torch.bfloat16)


def adaln(x, shift, scale):
    return (1 + scale) * F.layer_norm(x, (D,), eps=1e-6) + shift


def adaln_where(x, shift, scale, select):
    shift = torch.where(select, shift[:, 1:2], shift[:, 0:1])
    scale = torch.where(select, scale[:, 1:2], scale[:, 0:1])
    return (1 + scale) * F.layer_norm(x, (D,), eps=1e-6) + shift


def adaln_expand(x, shift, scale):
    return adaln(x, shift.expand_as(x), scale.expand_as(x))


x = rnd(1, M, D)
raw = rnd(1, 1, 3 * D, scale=0.3)
shift_r, scale_r, _ = raw.chunk(3, dim=-1)
shift_m = torch.cat([shift_r.expand(-1, M, -1)], dim=1)
scale_m = torch.cat([scale_r.expand(-1, M, -1)], dim=1)
variants = {
    "materialized": (adaln, (x, shift_m, scale_m)),
    "broadcast_view": (adaln, (x, shift_r, scale_r)),
    "broadcast_contig": (adaln, (x, shift_r.contiguous(), scale_r.contiguous())),
    "expand_in_graph": (adaln_expand, (x, shift_r, scale_r)),
    "where_dup": (
        adaln_where,
        (x, torch.cat([shift_r, shift_r], 1), torch.cat([scale_r, scale_r], 1),
         torch.zeros(1, M, 1, dtype=torch.bool, device="cuda")),
    ),
}
outs = {}
for name, (fn, args) in variants.items():
    torch._dynamo.reset()
    out, codes = run_and_get_code(torch.compile(fn, dynamic=False, options=opts), *args)
    outs[name] = out
    print(f"======== {name} ========", flush=True)
    for c in codes:
        # print only the triton kernel bodies and their heuristics
        keep = []
        on = False
        for line in c.splitlines():
            if "@triton_heuristics" in line or "@triton.jit" in line:
                on = True
            if on:
                keep.append(line)
            if on and line.strip().startswith("''', device_str"):
                on = False
        print("\n".join(keep), flush=True)
ref = outs["materialized"]
res = {k: {"bitwise": torch.equal(v, ref),
           "max_abs": (v.float() - ref.float()).abs().max().item()} for k, v in outs.items()}
fp32 = ((1 + scale_m.float()) * F.layer_norm(x.float(), (D,), eps=1e-6) + shift_m.float())
res["materialized_vs_fp32_max_abs"] = (ref.float() - fp32).abs().max().item()
print(json.dumps({"M": M, "results": res}))
