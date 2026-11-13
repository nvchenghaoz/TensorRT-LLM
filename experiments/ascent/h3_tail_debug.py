"""Which rounding step of the norm_out modulation differs between eager and h3_mod_out?"""

import json

import torch

from tensorrt_llm._torch.visual_gen.models.minimax_h3 import fused_adaln as fa

torch.manual_seed(0)
M, D, NT = 4096, 5376, 2
n = (torch.randn(1, M, D, device="cuda") * 2).to(torch.bfloat16)
table = (torch.randn(NT, 2 * D, device="cuda") * 0.5).to(torch.bfloat16)
tidx = torch.randint(0, NT, (M,), device="cuda")
shift, scale = table.chunk(2, dim=-1)
with torch.inference_mode():
    s_sel = scale.index_select(0, tidx)
    h_sel = shift.index_select(0, tidx)
    s1 = 1.0 + s_sel
    p = n * s1
    o = p + h_sel
    eager = o.float()
    # emulations
    s1e = (1.0 + s_sel.float()).bfloat16()
    pe = (n.float() * s1e.float()).bfloat16()
    oe = (pe.float() + h_sel.float()).bfloat16().float()
    no_round_p = (n.float() * s1e.float() + h_sel.float()).bfloat16().float()
    fma_like = torch.addcmul(h_sel.float(), n.float(), s1e.float()).bfloat16().float()
    no_round_s1 = ((n.float() * (1.0 + s_sel.float())).bfloat16().float() + h_sel.float()).bfloat16().float()
    k32 = fa.h3_mod_out(n, table, tidx, 1, 0, torch.float32)
    k16 = fa.h3_mod_out(n, table, tidx, 1, 0, torch.bfloat16).float()

    def ne(a, b):
        return int((a != b).sum())

    print(json.dumps({
        "s1_eager_vs_emul": ne(s1.float(), s1e.float()),
        "p_eager_vs_emul": ne(p.float(), pe.float()),
        "eager_vs_emul": ne(eager, oe),
        "eager_vs_no_round_p": ne(eager, no_round_p),
        "k32_vs_eager": ne(k32, eager),
        "k16_vs_eager": ne(k16, eager),
        "k32_vs_no_round_p": ne(k32, no_round_p),
        "k32_vs_fma_like": ne(k32, fma_like),
        "k32_vs_no_round_s1": ne(k32, no_round_s1),
        "k32_vs_k16": ne(k32, k16),
        "total": eager.numel(),
    }))
