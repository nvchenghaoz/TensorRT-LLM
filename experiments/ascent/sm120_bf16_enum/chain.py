"""The Gemma-4-31B BF16 attention-projection GEMM chain of one forward: for each of the 60 decoder layers in
config order (layer_types), qkv_proj then o_proj, every layer with its OWN weights (cold, as in the model).
Sliding: qkv [16384, 5376], o [5376, 8192]; global ([q | k] after the k_eq_v dedup): qkv [18432, 5376],
o [5376, 16384]. Inputs are fixed buffers (x_h [M, hidden], x_o [M, q_heads*head_dim]); the op is what this tree's
Gemma4 attention runs on SM120 (trtllm::cublas_mm where _SM120_BF16_CUBLAS_MM_ROUTES covers M, else F.linear).
"""
import json
import os

import torch


def layer_plan(ckpt):
    cfg = json.load(open(os.path.join(ckpt, "config.json")))
    tc = cfg.get("text_config", cfg)
    H = tc["hidden_size"]
    nq, hd, nkv = tc["num_attention_heads"], tc["head_dim"], tc["num_key_value_heads"]
    ghd = tc.get("global_head_dim", hd)
    gkv = tc.get("num_global_key_value_heads", nkv) or nkv
    keqv = tc.get("attention_k_eq_v", False)
    types = tc["layer_types"]
    sl = {"qkv": (nq * hd + 2 * nkv * hd, H), "o": (H, nq * hd)}
    gl = {"qkv": (nq * ghd + (1 if keqv else 2) * gkv * ghd, H), "o": (H, nq * ghd)}
    plan = [(i, t, sl if t == "sliding_attention" else gl) for i, t in enumerate(types)]
    info = dict(hidden=H, layers=len(types), sliding=sum(t == "sliding_attention" for t in types),
                sl=sl, gl=gl, k_eq_v=keqv)
    return plan, info


class Chain:

    def __init__(self, ckpt, max_m, seed=0, heavy=False):
        self.plan, self.info = layer_plan(ckpt)
        g = torch.Generator(device="cuda").manual_seed(seed)

        def mk(r, c, scale):
            t = torch.randn(r, c, device="cuda", dtype=torch.float32, generator=g)
            if heavy:
                t = t * torch.exp2(torch.empty(r, 1, device="cuda").uniform_(-8, 8, generator=g))
            return (t * scale).to(torch.bfloat16)

        self.W = []
        for i, t, shp in self.plan:
            self.W.append((mk(*shp["qkv"], 0.02), mk(*shp["o"], 0.02)))
        H = self.info["hidden"]
        self.xh = mk(max_m, H, 1.0)
        self.xo = {k: mk(max_m, k, 1.0) for k in {shp["o"][1] for _, _, shp in self.plan}}

    def run(self, op, M, keep=False):
        outs = []
        last = None
        for (i, t, shp), (wq, wo) in zip(self.plan, self.W):
            q = op(self.xh[:M], wq)
            o = op(self.xo[wo.shape[1]][:M], wo)
            if keep:
                outs += [q, o]
            last = (q, o)
        return outs if keep else last
