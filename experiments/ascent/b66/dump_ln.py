"""Branch 66 step 1: dump Inductor's LN+adaLN-modulate kernel (served math, standalone).

Single block: [1,2,D] tables + [1,S,1] bool select (torch.where), M=4176 (static and dynamic S).
Mode block: [1,1,D] rows, M=4096.
LayerNorm is tensorrt_llm's module (weight/bias buffers of ones/zeros), compiled per block with
dynamic=None and emulate_precision_casts=True as pipeline_flux3.torch_compile does.
"""

import json
import sys

import torch
from torch._inductor.utils import run_and_get_code

from tensorrt_llm._torch.modules.layer_norm import LayerNorm

D = 3072
opts = {"emulate_precision_casts": True}
g = torch.Generator(device="cuda").manual_seed(0)
ln = LayerNorm(hidden_size=D, eps=1e-6, has_weights=False, has_bias=False, dtype=torch.bfloat16).cuda()


def rnd(*s, scale=1.0):
    return (torch.randn(*s, device="cuda", generator=g) * scale).to(torch.bfloat16)


class Single(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_norm = ln

    def forward(self, x, shift_t, scale_t, select):
        shift = torch.where(select, shift_t[:, 1:2], shift_t[:, 0:1])
        scale = torch.where(select, scale_t[:, 1:2], scale_t[:, 0:1])
        return (1 + scale) * self.pre_norm(x) + shift


class Mode(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_norm = ln

    def forward(self, x, shift, scale):
        return (1 + scale) * self.pre_norm(x) + shift


def kernels(code):
    keep, on = [], False
    for line in code.splitlines():
        if "@triton_heuristics" in line or "@triton.jit" in line:
            on = True
        if on:
            keep.append(line)
        if on and line.strip().startswith("''', device_str"):
            on = False
    return "\n".join(keep)


def configs():
    import gc

    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    res = []
    for obj in gc.get_objects():
        try:
            hit = type(obj) is CachingAutotuner or isinstance(obj, CachingAutotuner)
        except ReferenceError:
            continue
        if hit:
            res.append(
                (
                    obj.fn.__name__,
                    [str(launcher.config) for launcher in getattr(obj, "launchers", [])],
                    [str(c) for c in obj.configs],
                )
            )
    return res


def dump(name, mod, args, dynamic_dim=None):
    torch._dynamo.reset()
    import gc

    gc.collect()
    if dynamic_dim is not None:
        torch._dynamo.maybe_mark_dynamic(args[0], dynamic_dim)
    fn = torch.compile(mod, dynamic=None, options=opts)
    with torch.inference_mode():
        out, codes = run_and_get_code(fn, *args)
    print(f"======== {name} ========", flush=True)
    for c in codes:
        print(kernels(c), flush=True)
    for c in configs():
        print("CONFIG", name, c, flush=True)
    return out


S = 4176
x = rnd(1, S, D)
tab = rnd(1, 2, 3 * D, scale=0.3)
shift_t, scale_t = tab[..., :D], tab[..., D : 2 * D]
select = torch.zeros(1, S, 1, dtype=torch.bool, device="cuda")
select[:, :80] = True
outs = {}
outs["single_static"] = dump("single_static", Single(), (x, shift_t, scale_t, select))
outs["single_dynamic"] = dump("single_dynamic", Single(), (x.clone(), shift_t, scale_t, select), 1)
xm = rnd(1, 4096, D)
table = rnd(1, 16, 2 * D, scale=0.3)
row = table[:, 5:6]
outs["mode_static"] = dump("mode_static", Mode(), (xm, row[..., :D], row[..., D : 2 * D]))
outs["mode_dynamic"] = dump("mode_dynamic", Mode(), (xm.clone(), row[..., :D], row[..., D : 2 * D]), 1)
print(
    json.dumps(
        {
            "single_static_eq_dynamic": torch.equal(outs["single_static"], outs["single_dynamic"]),
            "mode_static_eq_dynamic": torch.equal(outs["mode_static"], outs["mode_dynamic"]),
        }
    )
)
