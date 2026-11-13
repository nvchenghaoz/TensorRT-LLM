"""Idea 113 fix-A gate: is row 0 of the AdaLN time path bitwise equal at U=2 (row 1 = copy) and U=1?

Path (real dims from <ckpt>/transformer/config.json, random weights, seed 0):
  time_proj (diffusers Timesteps, flip_sin_to_cos) -> time_embedder (diffusers TimestepEmbedding,
  dtype fp32 and bf16 variants) -> per-block AdaLN table silu->Linear(time_embed_dim, 18*hidden) BF16
  (compiled like the block: torch.compile mode=default, dynamic=None; separate compiled instances for
  U=1 and U=2, plus one instance fed U=1 then U=2 the way the server sees it) -> norm_out
  silu->Linear(time_embed_dim, 2*hidden) BF16 (eager).
Usage: python h3_temb_pad_gate.py <checkpoint>. Last line JSON {"passed": ...}.
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers.models.embeddings import TimestepEmbedding, Timesteps


def main():
    cfg = json.loads((Path(sys.argv[1]) / "transformer" / "config.json").read_text())
    hidden = cfg.get("hidden_size") or cfg["num_attention_heads"] * cfg["attention_head_dim"]
    freq, ted_h, ted = cfg["freq_dim"], cfg["time_embed_hidden_dim"], cfg["time_embed_dim"]
    print(f"[gate] hidden={hidden} freq={freq} time_embed_hidden={ted_h} time_embed={ted}", flush=True)
    torch.manual_seed(0)
    dev = "cuda"
    tp = Timesteps(num_channels=freq, flip_sin_to_cos=True, downscale_freq_shift=0)
    w_ada = (torch.randn(18 * hidden, ted, device=dev) * 0.02).to(torch.bfloat16)
    b_ada = (torch.randn(18 * hidden, device=dev) * 0.02).to(torch.bfloat16)
    w_out = (torch.randn(2 * hidden, ted, device=dev) * 0.02).to(torch.bfloat16)
    b_out = (torch.randn(2 * hidden, device=dev) * 0.02).to(torch.bfloat16)

    def table(temb):
        return F.linear(F.silu(temb).to(torch.bfloat16), w_ada, b_ada).view(-1, 6 * hidden)

    c_u1 = torch.compile(table, mode="default", dynamic=None)
    c_u2 = torch.compile(lambda t: table(t), mode="default", dynamic=None)
    c_seq = torch.compile(lambda t: table(t) + 0, mode="default", dynamic=None)
    results = {}
    ok = True
    for te_dtype in (torch.float32, torch.bfloat16):
        te = TimestepEmbedding(in_channels=freq, time_embed_dim=ted_h, out_dim=ted).to(dev, te_dtype)
        for t in (1000.0, 999.0, 1.0, 0.0, 517.3):
            u1 = torch.tensor([t], device=dev, dtype=torch.float32)
            u2 = torch.tensor([t, t], device=dev, dtype=torch.float32)
            with torch.inference_mode():
                e1 = te(tp(u1).to(te_dtype))
                e2 = te(tp(u2).to(te_dtype))
                rec = {"temb": bool(torch.equal(e1[0], e2[0]))}
                t1, t2 = table(e1), table(e2)
                rec["table_eager"] = bool(torch.equal(t1[:3], t2[:3]))
                ct1, ct2 = c_u1(e1), c_u2(e2)
                rec["table_compiled"] = bool(torch.equal(ct1[:3], ct2[:3]))
                rec["compiled_vs_eager_u2"] = bool(torch.equal(ct2, t2))
                s1 = c_seq(e1)
                s2 = c_seq(e2)
                rec["table_compiled_same_fn"] = bool(torch.equal(s1[:3], s2[:3]))
                o1 = F.linear(F.silu(e1).to(torch.bfloat16), w_out, b_out)
                o2 = F.linear(F.silu(e2).to(torch.bfloat16), w_out, b_out)
                rec["norm_out"] = bool(torch.equal(o1[0], o2[0]))
                rec["max_abs_table_compiled"] = float((ct1[:3].float() - ct2[:3].float()).abs().max())
            key = f"{str(te_dtype).split('.')[-1]}_t{t}"
            results[key] = rec
            flags = [v for k, v in rec.items() if isinstance(v, bool) and k != "compiled_vs_eager_u2"]
            ok = ok and all(flags)
            print(f"[gate] {key}: {rec}", flush=True)
    print(json.dumps({"passed": ok, "compared": "row 0 (rows 0-2 of the 3U table) at U=2 vs U=1, bitwise",
                      "results": results}), flush=True)


if __name__ == "__main__":
    main()
