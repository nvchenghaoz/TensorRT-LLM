"""Branch 15 (idea 24) shared op-chain builders. Synthetic tensors at the served shapes.

The chain is the non-GEMM part of one NVFP4 single block as each tree routes it:
  head:     LN + adaLN modulate + NVFP4 quantize of x (feeds QKV / gate_up)
  residual: x + gate * (attn_y + mlp_y)          (Inductor, emulate_precision_casts)
Routes:
  'table'    : [1, K, D] tables + select (bool for K=2, u8 index for K=3) -> the fused
               torch.ops.trtllm.flux3_adaln_nvfp4_quant head; residual gate via
               transformer_flux3._flux3_select_mod (fused by Inductor).
  'pertoken' : main's i2i route: materialized [1, S, D] shift/scale/gate (joint() cats),
               Inductor LN+modulate, the CuTe-DSL NVFP4 quantize, per-token-gate residual.
"""

import torch

D = 3072
EPS = 1e-6
CASES = {
    # case label -> chunk lengths in joint order (pack(): sorted channel names)
    "i2i-2048": (16384, 16384, 80),  # image, image_cond, text -> S=32848
    "t2i-2048": (16384, 80),  # image, text -> S=16464
}


def log(*a):
    import time

    print(f"[b15 {time.strftime('%H:%M:%S')}]", *a, flush=True)


def make_inputs(case, seed=1234):
    lens = CASES[case]
    m = sum(lens)
    k = len(lens)
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"

    def rn(*shape, s=1.0):
        return (torch.randn(*shape, generator=g, device=dev, dtype=torch.float32) * s).to(
            torch.bfloat16
        )

    x = rn(1, m, D, s=1.5)
    # sprinkle outliers so NVFP4 block scales vary
    x[0, ::97, ::31] *= 8
    tabs = {f: rn(1, k, D, s=0.5) for f in ("shift", "scale", "gate")}
    attn_y = rn(1, m, D, s=0.7)
    mlp_y = rn(1, m, D, s=0.7)
    starts, acc = [], 0
    for n in lens:
        starts.append(acc)
        acc += n
    ar = torch.arange(m, device=dev)
    index = torch.zeros(m, dtype=torch.uint8, device=dev)
    for b in starts[1:]:
        index += (ar >= b).to(torch.uint8)
    index = index.view(1, m, 1)
    select = (index != 0) if k == 2 else index
    pertoken = {
        f: torch.cat(
            [tabs[f][:, j : j + 1].expand(-1, n, -1) for j, n in enumerate(lens)], dim=1
        )
        for f in tabs
    }
    weight = torch.ones(D, dtype=torch.bfloat16, device=dev)
    bias = torch.zeros(D, dtype=torch.bfloat16, device=dev)
    # inverse global scale of the activation (checkpoint-like magnitude)
    gs = torch.tensor([448.0 * 6.0 / 16.0], dtype=torch.float32, device=dev)
    return dict(
        x=x, tabs=tabs, attn_y=attn_y, mlp_y=mlp_y, index=index, select=select,
        pertoken=pertoken, weight=weight, bias=bias, gs=gs, lens=lens, m=m, k=k,
    )


def table_route_ok(inp):
    """Does THIS tree's fused head accept the case's table form?"""
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_adaln_nvfp4 as warp

    return warp.supported(
        inp["x"], inp["select"], inp["tabs"]["scale"], inp["tabs"]["shift"], inp["weight"],
        inp["bias"],
    )


class Chains:
    def __init__(self, inp):
        from tensorrt_llm._torch.modules.layer_norm import LayerNorm
        from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_adaln_quant  # noqa: F401
        from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_adaln_nvfp4 as warp
        from tensorrt_llm._torch.visual_gen.models.flux3.flux3_primsts import nvfp4_quantize
        from tensorrt_llm._torch.visual_gen.models.flux3.transformer_flux3 import (
            _flux3_select_mod,
        )

        self.inp = inp
        self.warp = warp
        warp.register_identity_affine(inp["weight"], inp["bias"])
        self.norm = LayerNorm(
            hidden_size=D, eps=EPS, has_weights=False, has_bias=False, dtype=torch.bfloat16
        ).cuda()
        self.quant = nvfp4_quantize
        norm = self.norm

        def ln_mod(x, shift, scale):
            return (1 + scale) * norm(x) + shift

        def res_pt(x, gate, a, b):
            return x + gate * (a + b)

        def res_tab(x, sh, sc, gt, sel, a, b):
            gate = _flux3_select_mod((sh, sc, gt, sel))[2]
            return x + gate * (a + b)

        opts = {"emulate_precision_casts": True}
        self.ln_mod = torch.compile(ln_mod, dynamic=False, options=opts)
        self.res_pt = torch.compile(res_pt, dynamic=False, options=opts)
        self.res_tab = torch.compile(res_tab, dynamic=False, options=opts)

    def head_table(self, store_x_mod=False):
        i = self.inp
        return torch.ops.trtllm.flux3_adaln_nvfp4_quant(
            i["x"], i["select"], i["tabs"]["scale"], i["tabs"]["shift"], i["weight"],
            i["bias"], i["gs"], EPS, store_x_mod,
        )

    def head_pertoken(self):
        i = self.inp
        xm = self.ln_mod(i["x"], i["pertoken"]["shift"], i["pertoken"]["scale"])
        data, sf = self.quant(xm.reshape(-1, D), i["gs"])
        return xm, data, sf

    def resid_table(self):
        i = self.inp
        t = i["tabs"]
        return self.res_tab(i["x"], t["shift"], t["scale"], t["gate"], i["select"],
                            i["attn_y"], i["mlp_y"])

    def resid_pertoken(self):
        i = self.inp
        return self.res_pt(i["x"], i["pertoken"]["gate"], i["attn_y"], i["mlp_y"])

    def chain(self, route):
        if route == "table":
            _, data, sf = self.head_table(store_x_mod=False)
            out = self.resid_table()
        else:
            _, data, sf = self.head_pertoken()
            out = self.resid_pertoken()
        return data, sf, out


def snapshot(inp):
    out = {}
    for k, v in inp.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.clone()
        elif isinstance(v, dict):
            out[k] = {kk: vv.clone() for kk, vv in v.items()}
    return out


def same_as(inp, snap):
    for k, v in snap.items():
        if isinstance(v, dict):
            if not all(torch.equal(inp[k][kk], vv) for kk, vv in v.items()):
                return False
        elif not torch.equal(inp[k], v):
            return False
    return True


def restore(inp, snap):
    for k, v in snap.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                inp[k][kk].copy_(vv)
        else:
            inp[k].copy_(v)
