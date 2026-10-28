"""Branch 62 (idea 112): isolate the history dependence of fp8_sm12x attention fed strided
packed q/k/v (views of the FP8R qkv_qknorm_rope [M, 9216] output).

Chain per call: trtllm::flux3_sm120_fp8r_qkv_qknorm_rope -> split(3072) -> [optional
.contiguous()] -> view [1, M, 24, 128] -> trtllm::visual_gen_fp8_attention_sm120.
Lengths: A = 4096 + 80 = 4176 rows, B = 4096 + 160 = 4256 rows. Sequence A, B, A.

Scenarios (each for variant strided and contiguous):
  eager      : plain calls.
  graph      : CUDAGraphRunner-like capture (2 eager warmups + gc + empty_cache, then capture
               with a pool shared across keys), A captured, B captured, A replayed.
  compile    : torch.compile(dynamic=None) -> static A graph, B recompiles dynamic, A again.
  compile_graph: compiled fn under the shared-pool graph capture (as served).
Every A output (attention output and packed producer output) is compared bitwise with the
eager first-call reference.
"""
import gc
import json
import sys
import time

import torch

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


from tensorrt_llm._torch.visual_gen.attention.backends.fp8_sm12x import (  # noqa: E402
    visual_gen_fp8_attention_sm120,
)
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_fp8r_qkv import (  # noqa: E402
    fp8r_qkv_qknorm_rope,
)

DEV = "cuda"
H, D = 24, 128
W = H * D
K = 3072
SCALE = 1.0 / D ** 0.5
LENS = {"A": 4096 + 80, "B": 4096 + 160}


def make_inputs(m, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    act = (torch.randn(m, K, device=DEV, generator=g) * 2).to(torch.float8_e4m3fn)
    act_s = (torch.rand(m, device=DEV, generator=g) * 0.02 + 0.01).float()
    pos = torch.arange(m, device=DEV, dtype=torch.float32)[:, None]
    freq = torch.arange(64, device=DEV, dtype=torch.float32)[None, :]
    ang = pos * (10000.0 ** (-freq / 64))
    cos_sin = torch.cat((ang.cos(), ang.sin()), -1).contiguous()
    return act, act_s, cos_sin


gw = torch.Generator(device=DEV).manual_seed(1234)
WEIGHT = (torch.randn(3 * W, K, device=DEV, generator=gw) * 2).to(torch.float8_e4m3fn)
WSCALE = (torch.rand(3 * W, device=DEV, generator=gw) * 0.02 + 0.01).float()
QN = (torch.rand(D, device=DEV, generator=gw) + 0.5).to(torch.bfloat16)
KN = (torch.rand(D, device=DEV, generator=gw) + 0.5).to(torch.bfloat16)

INPUTS = {k: make_inputs(m, 7 + i) for i, (k, m) in enumerate(LENS.items())}


def chain(act, act_s, cos_sin, contiguous: bool):
    out = fp8r_qkv_qknorm_rope(act, WEIGHT, act_s, WSCALE, QN, KN, cos_sin)
    q, k, v = out.split(W, -1)
    if contiguous:
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    m = out.shape[0]
    q, k, v = (t.reshape(1, m, H, D) for t in (q, k, v))
    attn = visual_gen_fp8_attention_sm120(q, k, v, SCALE, False)
    return attn, out


def clone_inputs(key):
    return tuple(t.clone() for t in INPUTS[key])


class Runner:
    """Shared-pool capture/replay as tensorrt_llm/_torch/visual_gen/cuda_graph_runner.py."""

    def __init__(self, fn):
        self.fn, self.graphs, self.static, self.outs, self.pool = fn, {}, {}, {}, None

    def __call__(self, key):
        args = INPUTS[key]
        if key not in self.graphs:
            static = tuple(a.clone() for a in args)
            for _ in range(2):
                self.fn(*static)
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                out = self.fn(*static)
            self.graphs[key], self.static[key], self.outs[key] = g, static, out
            self.pool = g.pool()
        for s, a in zip(self.static[key], args):
            s.copy_(a)
        self.graphs[key].replay()
        return self.outs[key]


def diff(a, b):
    if a.shape != b.shape:
        return {"bitwise": False, "shape": [list(a.shape), list(b.shape)]}
    eq = torch.equal(a, b)
    d = (a.float() - b.float()).abs()
    nz = int((d != 0).sum())
    rows = (d.reshape(d.shape[0] if d.dim() == 2 else d.shape[1], -1) != 0).any(-1) if d.dim() in (2, 4) else None
    first_rows = rows.nonzero().flatten()[:8].tolist() if rows is not None else []
    return {"bitwise": bool(eq), "max_abs": float(d.max()), "n_diff": nz, "first_rows": first_rows,
            "n_rows": int(rows.sum()) if rows is not None else None}


def run_scenario(name, variant, make_fn):
    contiguous = variant == "contiguous"
    torch._dynamo.reset()
    fn = make_fn(lambda a, s, c: chain(a, s, c, contiguous))
    seq = ["A", "A", "B", "B", "A", "A"]
    outs = []
    for key in seq:
        attn, packed = fn(key)
        torch.cuda.synchronize()
        outs.append((key, attn.clone(), packed.clone()))
    res = {}
    ref_attn, ref_packed = REF[variant]
    for i, (key, attn, packed) in enumerate(outs):
        if key != "A":
            continue
        res[f"call{i}"] = {"attn_vs_ref": diff(attn, ref_attn), "packed_vs_ref": diff(packed, ref_packed)}
    ok = all(v["attn_vs_ref"]["bitwise"] and v["packed_vs_ref"]["bitwise"] for v in res.values())
    log(f"{name}/{variant}: {'OK' if ok else 'MISMATCH'} {json.dumps(res)}")
    return ok, res


def eager_fn(f):
    return lambda key: f(*INPUTS[key])


def graph_fn(f):
    return Runner(f)


def compile_fn(f):
    cf = torch.compile(f)
    return lambda key: cf(*INPUTS[key])


def compile_graph_fn(f):
    return Runner(torch.compile(f))


def main():
    global REF
    log("torch", torch.__version__, "gpu", torch.cuda.get_device_name())
    REF = {}
    for variant in ("strided", "contiguous"):
        attn, packed = chain(*clone_inputs("A"), variant == "contiguous")
        torch.cuda.synchronize()
        REF[variant] = (attn.clone(), packed.clone())
    d = diff(REF["strided"][0], REF["contiguous"][0])
    log("eager first-call strided vs contiguous attn:", d)
    scen = sys.argv[1:] or ["eager", "graph", "compile", "compile_graph"]
    makers = {"eager": eager_fn, "graph": graph_fn, "compile": compile_fn,
              "compile_graph": compile_graph_fn}
    summary = {"strided_vs_contiguous_first_call": d["bitwise"]}
    for s in scen:
        for variant in ("strided", "contiguous"):
            ok, _ = run_scenario(s, variant, makers[s])
            summary[f"{s}/{variant}"] = ok
    print(json.dumps({"passed": all(v for k, v in summary.items()), **summary}), flush=True)


if __name__ == "__main__":
    main()
