"""torch.compile probe for the CUDNN visual-gen attention backend (H3-like call site).

A module does qkv-proj-like slicing (V a strided view of a fused buffer), calls
CuDNNAttention through Attention-style NHD views and a projection, and is compiled with
torch.compile(dynamic=None) as the pipeline does per block. It runs repeated calls at one
sequence length (like the 49 denoising forwards) and then at a second length, and reports
graph breaks, frame compiles and recompiles from torch._dynamo counters. Output is checked
against eager.
"""

import json
import math

import torch
import torch._dynamo
from torch._dynamo.utils import counters

from tensorrt_llm._torch.visual_gen.attention.backends.cudnn import CuDNNAttention

H, D = 28, 128


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = CuDNNAttention(num_heads=H, head_dim=D, dtype=torch.bfloat16)
        self.proj = torch.nn.Linear(H * D, H * D, bias=False, dtype=torch.bfloat16)

    def forward(self, qkv, q, k):
        B, S, _ = qkv.shape
        v = qkv.split([H * D, H * D, H * D], dim=-1)[2]
        out = self.attn.forward(
            q.view(B, S, H, D), k.view(B, S, H, D), v.view(B, S, H, D),
            discard_padded_queries=True, valid_seq_len=None, timestep=None,
        )
        return self.proj(out.flatten(2))


def main():
    torch.manual_seed(0)
    blk = Block().cuda()
    with torch.inference_mode():
        S0 = 4096
        qkv = torch.randn(1, S0, 3 * H * D, device="cuda", dtype=torch.bfloat16)
        q = torch.randn(1, S0, H * D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, S0, H * D, device="cuda", dtype=torch.bfloat16)
        ex = torch._dynamo.explain(blk)(qkv, q, k)
        print(f"explain: graphs {ex.graph_count} breaks {ex.graph_break_count} "
              f"ops {ex.op_count}", flush=True)
        for br in ex.break_reasons:
            loc = br.user_stack[-1] if br.user_stack else None
            print(f"  break: {str(br.reason)[:200]} @ {loc}", flush=True)
        torch._dynamo.reset()
        counters.clear()
    cblk = torch.compile(blk, mode="default", dynamic=None)
    res = {}
    with torch.inference_mode():
        for S in (4096, 4096 + 300):
            qkv = torch.randn(1, S, 3 * H * D, device="cuda", dtype=torch.bfloat16)
            q = torch.randn(1, S, H * D, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(1, S, H * D, device="cuda", dtype=torch.bfloat16)
            ref = blk(qkv, q, k)
            before = {k2: dict(v2) for k2, v2 in counters.items()}
            outs = [cblk(qkv, q, k) for _ in range(6)]
            torch.cuda.synchronize()
            after = {k2: dict(v2) for k2, v2 in counters.items()}
            maxdiff = max(float((o.float() - ref.float()).abs().max()) for o in outs)
            bitwise = all(torch.equal(o, outs[0]) for o in outs)
            res[S] = dict(maxdiff_vs_eager=maxdiff, repeat_bitwise=bitwise)
            print(f"S={S}: maxdiff vs eager {maxdiff:.3e}, repeats bitwise {bitwise}", flush=True)
            print("  counters delta:", json.dumps(
                {k2: {kk: after[k2].get(kk, 0) - before.get(k2, {}).get(kk, 0)
                      for kk in after[k2]} for k2 in after}, default=str)[:3000], flush=True)
    gb = dict(counters.get("graph_break", {}))
    stats = dict(counters.get("stats", {}))
    print("graph_breaks:", json.dumps(gb, default=str)[:3000])
    print("stats:", json.dumps(stats, default=str))
    passed = all(r["repeat_bitwise"] and r["maxdiff_vs_eager"] < 5e-2 for r in res.values())
    print(json.dumps({"passed": passed, "results": res, "graph_breaks": len(gb),
                      "unique_graphs": stats.get("unique_graphs"),
                      "calls_captured": stats.get("calls_captured")}))


if __name__ == "__main__":
    main()
