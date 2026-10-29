"""Branch 15 (top-lead ruling 16:30): the i2i single-block tail at M=32848 with idea 24's
[1,3,D] gate table + u8 chunk index (served layout image [0,16384) | image_cond [16384,32768) |
text [32768,32848)), in ONE process on one GPU, arms interleaved (CUDA graphs of 20 calls each):
  pair_only   : main's LPT pair (t256 at this M) alone
  base        : pair + the branch's Inductor residual  x + _flux3_select_mod(gate)*(attn_y+mlp_y)
                (the route with TRTLLM_FLUX3_SM120_PAIR_RESIDUAL=0)
  base_AA     : the same as base, a second graph (A/A noise)
  fused       : landing 7's fused 128x128 pair + residual (PAIR_RESIDUAL=1 route)
Untimed check: fused vs base bitwise over WARMUP+ITERS calls, inputs unchanged.
Reuses experiments/ascent/pair_residual/bench.py's synthetic NVFP4 case builder (fixed seeds,
3 input sets cycled). Last line: {"passed": ..., "table": {...}}.
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pair_residual"))
import bench as pr  # noqa: E402

from tensorrt_llm._torch.visual_gen.models.flux3.transformer_flux3 import (  # noqa: E402
    _flux3_select_mod,
)

H = pr.H


def served_index(m, lens):
    ar = torch.arange(m, device="cuda")
    idx = torch.zeros(m, dtype=torch.uint8, device="cuda")
    acc = 0
    for n in lens[:-1]:
        acc += n
        idx += (ar >= acc).to(torch.uint8)
    return idx.view(1, m, 1)


def _res(x, table, index, attn_y, mlp_y):
    gate = _flux3_select_mod((table, table, table, index))[2]
    output = attn_y.reshape(*x.shape[:-1], -1) + mlp_y.reshape(*x.shape[:-1], -1)
    return x + gate * output


res = torch.compile(_res, dynamic=False, options={"emulate_precision_casts": True})


def run_base(c):
    attn_y, mlp_y = pr.run_pair(c)
    return res(c["x"], c["gate"], c["index"], attn_y, mlp_y)


def main():
    pr.preload()
    calls = 20
    out = {"table": {}, "check": {}}
    passed = True
    for m, lens, form in ((32848, (16384, 16384, 80), "u8x3"), (16464, (16384, 80), "select")):
        cs = []
        for i in range(pr.SETS):
            c = pr.make_case(m, 7 * m + i, form)
            idx = served_index(m, lens)
            c["index"] = idx if form == "u8x3" else (idx != 0)
            cs.append(c)
        # untimed bitwise check fused vs base, inputs unchanged
        c = cs[0]
        pristine = [t.clone() for t in pr.inputs(c)]
        ref = run_base(c).reshape(c["x"].shape)
        n_eq = 0
        for _ in range(pr.WARMUP + pr.ITERS):
            o = pr.run_fused(c).reshape(c["x"].shape)
            n_eq += bool(torch.equal(o, ref))
        torch.cuda.synchronize()
        inputs_ok = all(torch.equal(a, b) for a, b in zip(pristine, pr.inputs(c)))
        ok = n_eq == pr.WARMUP + pr.ITERS and inputs_ok
        passed &= ok
        from tensorrt_llm._torch.visual_gen.models.flux3 import (
            flux3_sm120_lpt_pair_residual as lptr,
        )

        out["check"][f"M{m}_{form}"] = {"fused_vs_base_bitwise": f"{n_eq}/{pr.WARMUP + pr.ITERS}",
                                        "inputs_unchanged": inputs_ok,
                                        "route_gate_form_takes_fused": bool(lptr.gate_form(
                                            c["gate"], c["index"], m, H))}
        s = torch.cuda.Stream()

        def seq(f):
            return [lambda c=cs[i % pr.SETS]: f(c) for i in range(calls)]

        graphs = {}
        graphs["pair_only"], _ = pr.capture(seq(pr.run_pair), s)
        graphs["base"], _ = pr.capture(seq(run_base), s)
        graphs["fused"], _ = pr.capture(seq(pr.run_fused), s)
        graphs["base_AA"], _ = pr.capture(seq(run_base), s)
        t = pr.time_graphs(graphs, samples=60, warm=10, calls=calls)
        row = {}
        for n, v in t.items():
            med, p10, p90 = pr.stat(v)
            row[n] = {"us": round(med, 2), "p10": round(p10, 2), "p90": round(p90, 2)}
        row["residual_us"] = round(row["base"]["us"] - row["pair_only"]["us"], 2)
        row["fused_minus_base_us"] = round(row["fused"]["us"] - row["base"]["us"], 2)
        row["AA_diff_us"] = round(row["base_AA"]["us"] - row["base"]["us"], 2)
        out["table"][f"M{m}_{form}"] = row
        pr.log(f"M={m} {form}: {json.dumps(row)} check {json.dumps(out['check'][f'M{m}_{form}'])}")
        del graphs, cs
        torch.cuda.synchronize()
    out["passed"] = bool(passed)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
