"""Branch 3 (idea 12) STEP 0: is the block-tail residual pointwise worth fusing into the LPT pair?

Synthetic NVFP4 tensors at the served shapes (no model load). Per M, CUDA graphs of:
  pair   : flux3_sm120_lpt_pair.run (down K=9216 + attn_out K=3072, N=3072), preallocated outs
  pairpw : pair + the compiled (emulate_precision_casts) residual: res + gate * (attn_y + mlp_y)
           with the single-block [1,2,D] table + [1,S,1] select layout
  pw     : the residual pointwise alone
Two input sets cycled per graph so the working set exceeds L2. Last stdout line: JSON.
"""

import json
import os
import time

import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_lpt_pair as lpt

H, FF = 3072, 9216
SAMPLES = int(os.environ.get("S0_SAMPLES", "60"))
REPLAYS = 3


def log(*a):
    print(*a, flush=True)


def _inv(s):
    return torch.tensor([1.0 / s], device="cuda", dtype=torch.float32)


def make(M, seed, T):
    g = torch.Generator(device="cuda").manual_seed(seed)
    hid = torch.randn(M, FF, device="cuda", dtype=torch.bfloat16, generator=g)
    att = torch.randn(M, H, device="cuda", dtype=torch.bfloat16, generator=g)
    wd = torch.randn(H, FF, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    wo = torch.randn(H, H, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    c = {}
    c["a0"], c["sfa0"] = fp.nvfp4_quantize(hid, _inv(float(hid.abs().amax()) / 2688.0))
    c["a1"], c["sfa1"] = fp.nvfp4_quantize(att, _inv(float(att.abs().amax()) / 2688.0))
    c["b0"], c["sfb0"] = fp.nvfp4_quantize(wd, _inv(float(wd.abs().amax()) / 2688.0))
    c["b1"], c["sfb1"] = fp.nvfp4_quantize(wo, _inv(float(wo.abs().amax()) / 2688.0))
    c["al0"] = torch.tensor([1e-3], device="cuda")
    c["al1"] = torch.tensor([2e-3], device="cuda")
    c["d0"] = torch.empty(M, H, device="cuda", dtype=torch.bfloat16)
    c["d1"] = torch.empty(M, H, device="cuda", dtype=torch.bfloat16)
    c["res"] = torch.randn(1, M, H, device="cuda", dtype=torch.bfloat16, generator=g)
    c["gate"] = torch.randn(1, 2, H, device="cuda", dtype=torch.bfloat16, generator=g)
    sel = torch.zeros(1, M, 1, device="cuda", dtype=torch.bool)
    sel[:, M - T:] = True
    c["sel"] = sel
    del hid, att, wd, wo
    return c


def pw_fn(res, a, m, gate, sel):
    gate = torch.where(sel, gate[:, 1:2], gate[:, 0:1])
    out = a.reshape(1, -1, H) + m.reshape(1, -1, H)
    return res + gate * out


pw = torch.compile(pw_fn, dynamic=False, fullgraph=True, options={"emulate_precision_casts": True})


def pair(c):
    lpt.run(c["a0"], c["b0"], c["sfa0"], c["sfb0"], c["al0"], c["d0"],
            c["a1"], c["b1"], c["sfa1"], c["sfb1"], c["al1"], c["d1"])


def arm(sets, kind):
    def f():
        for c in sets:
            if kind in ("pair", "pairpw"):
                pair(c)
            if kind in ("pairpw", "pw"):
                pw(c["res"], c["d1"], c["d0"], c["gate"], c["sel"])
    return f


def capture(fn):
    s = torch.cuda.Stream()
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        fn()
    torch.cuda.synchronize()
    return g


def main():
    res = {}
    for M, T in ((16896, 512), (33280, 512)):
        sets = [make(M, 1, T), make(M, 2, T)]
        graphs = {k: capture(arm(sets, k)) for k in ("pair", "pairpw", "pw")}
        times = {k: [] for k in graphs}
        for r in range(SAMPLES):
            for k, g in graphs.items():
                torch.cuda.synchronize()
                a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                torch.cuda._sleep(2_000_000)
                a.record()
                for _ in range(REPLAYS):
                    g.replay()
                b.record()
                b.synchronize()
                times[k].append(a.elapsed_time(b) / REPLAYS / len(sets))
        med = {k: sorted(v)[len(v) // 2] for k, v in times.items()}
        res[str(M)] = {"median_ms_per_call": med,
                       "pw_exposed_ms": med["pairpw"] - med["pair"]}
        log(M, res[str(M)])
        del sets, graphs
        torch.cuda.empty_cache()
    print(json.dumps(res))


if __name__ == "__main__":
    t0 = time.time()
    main()
