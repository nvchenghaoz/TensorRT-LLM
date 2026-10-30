"""Branch 3 (idea 12): one denoise step's 32 NVFP4 block tails through the SERVED custom ops,
synthetic NVFP4 tensors at the served shapes (no model load), CUDA-graph replays.

  base (main's tail): torch.ops.trtllm.flux3_sm120_nvfp4_out_pair_prequant (main's routing of
        the down + attn_out pair, whatever module main serves at that M) followed by the block's
        residual pointwise compiled as the block is (torch.compile, emulate_precision_casts):
        x + gate * (attn_y + mlp_y).
  fused (branch): torch.ops.trtllm.flux3_sm120_nvfp4_tail_one_launch (one launch); the timed
        branch arm ("served") uses it for calls in its M-range bucket and main's tail otherwise,
        as the block forward routes; check runs the op at every M.
Per case, one graph = 28 single-block tails at the served single-block M + 4 image mode-block
tails at M=16384: t2i M=16464 with the [1,2,D] gate table + bool select (T=80 text rows last),
i2i M=32848 with main's per-token [1,S,D] gate; mode blocks use the [1,1,D] gate row. Two input
sets per M are cycled so the working set exceeds L2.

  pair  : run_gpu_pair unit. Each tree runs its served arm (fused when the op exists). Per case:
          capture, WARMUP replays, SAMPLES timed replays (CUDA events, one replay each).
  check : branch only, untimed, same plans / capture / warmup / replay count as pair, cases in
          the order t2i -> i2i -> t2i. Before every replay all outputs are NaN-poisoned; after it
          every fused output is compared bitwise with the base arm's output for the same call
          (base graph replayed in the same sequence) and with replay 0; inputs are checked
          unchanged after every replay.
Last stdout line: JSON.
"""

import json
import math
import os
import sys
import time

import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp
from tensorrt_llm._torch.visual_gen.models.flux3.primsts import PreparedNVFP4

H, FF, T_TEXT = 3072, 9216, 80
M_T2I, M_I2I, M_MODE = 16464, 32848, 16384
WARMUP = int(os.environ.get("B3_WARMUP", "3"))
SAMPLES = int(os.environ.get("B3_SAMPLES", "50"))
HAS_FUSED = hasattr(fp, "nvfp4_tail_one_launch")


def log(*a):
    print(*a, flush=True)


def _gs(t):
    return float(t.abs().amax().float() / (448.0 * 6.0))


def _inv(s):
    return torch.tensor([1.0 / s], device="cuda", dtype=torch.float32)


def make(M, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    hid = torch.randn(M, FF, device=dev, dtype=torch.bfloat16, generator=g)
    att = torch.randn(M, H, device=dev, dtype=torch.bfloat16, generator=g)
    w_down = torch.randn(H, FF, device=dev, dtype=torch.bfloat16, generator=g) * 0.02
    w_out = torch.randn(H, H, device=dev, dtype=torch.bfloat16, generator=g) * 0.02
    s_hid, s_down, s_out = _gs(hid), _gs(w_down), _gs(w_out)
    s_att = 1.0  # attn_out keeps a unit input global scale as served
    c = {"M": M}
    c["a_down"], c["sf_down"] = fp.nvfp4_quantize(hid, _inv(s_hid))
    c["a_attn"], c["sf_attn"] = fp.nvfp4_quantize(att, _inv(s_att))
    q_down, sf_wd = fp.nvfp4_quantize(w_down, _inv(s_down))
    q_out, sf_wo = fp.nvfp4_quantize(w_out, _inv(s_out))
    c["down"] = PreparedNVFP4(q_down, sf_wd, s_hid, s_down)
    c["out"] = PreparedNVFP4(q_out, sf_wo, s_att, s_out)
    c["al_down"] = torch.tensor([s_hid * s_down], device=dev, dtype=torch.float32)
    c["al_attn"] = torch.tensor([s_att * s_out], device=dev, dtype=torch.float32)
    c["res"] = torch.randn(1, M, H, device=dev, dtype=torch.bfloat16, generator=g)
    c["table"] = torch.randn(1, 2, H, device=dev, dtype=torch.bfloat16, generator=g)
    c["row"] = torch.randn(1, 1, H, device=dev, dtype=torch.bfloat16, generator=g)
    c["pertok"] = torch.randn(1, M, H, device=dev, dtype=torch.bfloat16, generator=g)
    sel = torch.zeros(1, M, 1, device=dev, dtype=torch.bool)
    sel[:, M - T_TEXT:] = True
    c["sel"] = sel
    del hid, att, w_down, w_out
    torch.cuda.synchronize()
    return c


INPUTS = ("a_down", "sf_down", "a_attn", "sf_attn", "res", "table", "row", "pertok", "sel")


def pw_single(res, a, m, table, sel):
    gate = torch.where(sel, table[:, 1:2], table[:, 0:1])
    return res + gate * (a.reshape(1, -1, H) + m.reshape(1, -1, H))


def pw_gate(res, a, m, gate):
    return res + gate * (a.reshape(1, -1, H) + m.reshape(1, -1, H))


_OPT = {"emulate_precision_casts": True}
pw_single_c = torch.compile(pw_single, dynamic=False, fullgraph=True, options=_OPT)
pw_gate_c = torch.compile(pw_gate, dynamic=False, fullgraph=True, options=_OPT)


def gate_of(c, layout):
    """(gate, select, gate_mode) exactly as the block forward hands them to the fused op."""
    if layout == "single":
        return c["table"], c["sel"], 1
    if layout == "pertok":
        return c["pertok"], None, 2
    return c["row"], None, 0


def base_call(c, layout):
    attn_y, mlp_y = torch.ops.trtllm.flux3_sm120_nvfp4_out_pair_prequant(
        c["a_down"], c["down"].weight, c["sf_down"], c["down"].scale,
        c["a_attn"], c["sf_attn"], c["out"].weight, c["out"].scale,
    )
    if layout == "single":
        return pw_single_c(c["res"], attn_y, mlp_y, c["table"], c["sel"])
    gate = c["pertok"] if layout == "pertok" else c["row"]
    return pw_gate_c(c["res"], attn_y, mlp_y, gate)


def served_call(c, layout):
    """The branch tree's served tail: the one-launch op in its M-range bucket, else main's."""
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_tail_one_launch as tol

    if tol.enabled() and c["M"] >= tol.min_rows():
        return fused_call(c, layout)
    return base_call(c, layout)


def fused_call(c, layout):
    gate, sel, mode = gate_of(c, layout)
    return torch.ops.trtllm.flux3_sm120_nvfp4_tail_one_launch(
        c["a_down"], c["down"].weight, c["sf_down"], c["down"].scale,
        c["a_attn"], c["sf_attn"], c["out"].weight, c["out"].scale,
        c["res"], gate, sel, mode,
    )


def pairA_pw_call(c, layout):
    """Diagnostics arm (b): this kernel as a plain 128-tile AlongN pair (mode 1: writes attn_y
    and mlp_y, residual off) followed by the compiled Inductor residual."""
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_tail_one_launch as tol

    M = c["M"]
    attn_y = torch.empty(M, H, device="cuda", dtype=torch.bfloat16)
    mlp_y = torch.empty(M, H, device="cuda", dtype=torch.bfloat16)
    gate, sel, mode = gate_of(c, layout)
    tol.run(c["a_attn"], c["out"].weight, c["sf_attn"], c["out"].scale, c["al_attn"],
            c["a_down"], c["down"].weight, c["sf_down"], c["down"].scale, c["al_down"],
            c["res"], gate.reshape(-1, H), None if sel is None else sel.reshape(-1),
            attn_y, gate_mode=mode, mode=1, dbg_attn=attn_y, dbg_mlp=mlp_y)
    if layout == "single":
        return pw_single_c(c["res"], attn_y, mlp_y, c["table"], c["sel"])
    g = c["pertok"] if layout == "pertok" else c["row"]
    return pw_gate_c(c["res"], attn_y, mlp_y, g)


CASES = (("t2i-2048", M_T2I, "single"), ("i2i-2048", M_I2I, "pertok"))


def make_plans():
    mode_sets = [make(M_MODE, 5), make(M_MODE, 6)]
    plans = {}
    for i, (label, M, layout) in enumerate(CASES):
        sets = [make(M, 1 + 2 * i), make(M, 2 + 2 * i)]
        plans[label] = [(sets[j % 2], layout) for j in range(28)] + \
            [(mode_sets[j % 2], "mode") for j in range(4)]
    return plans


_STREAM = []


def capture(plan, fn):
    if not _STREAM:
        _STREAM.append(torch.cuda.Stream())
    s = _STREAM[0]
    with torch.cuda.stream(s):
        for _ in range(2):
            for c, layout in plan:
                fn(c, layout)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        outs = [fn(c, layout) for c, layout in plan]
    torch.cuda.synchronize()
    return g, outs


def stat(v):
    v = sorted(v)
    n = len(v)
    med = v[n // 2]
    return med, (v[(9 * n) // 10] - v[n // 10]) / med, v[n // 10], v[(9 * n) // 10]


def pair():
    arm = os.environ.get("B3_ARM", "served" if HAS_FUSED else "base")
    fn = {"served": served_call, "fused": fused_call, "base": base_call}[arm]
    plans = make_plans()
    out = {"batch": {}, "arm": arm}
    for label, M, layout in CASES:
        g, _ = capture(plans[label], fn)
        for _ in range(WARMUP):
            g.replay()
        torch.cuda.synchronize()
        times = []
        for i in range(SAMPLES):
            torch.cuda._sleep(2_000_000)
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record()
            g.replay()
            b.record()
            b.synchronize()
            times.append(a.elapsed_time(b))
            if i % 10 == 0:
                log(f"  {label} {arm} sample {i}: {times[-1]:.3f} ms")
        med, noise, p10, p90 = stat(times)
        out["batch"][label] = {"kernel_ms": med, "noise_floor": noise, "p10_ms": p10,
                               "p90_ms": p90, "single_M": M, "mode_M": M_MODE}
        log(label, arm, json.dumps(out["batch"][label]))
        del g
        torch.cuda.empty_cache()
    out["output"] = {"class": "exact"}
    out["unit"] = ("one CUDA-graph replay = one denoise step's 32 NVFP4 block tails (28 single "
                   "blocks: t2i M16464 [1,2,D] table + select, i2i M32848 per-token gate; 4 image "
                   "mode blocks M16384 [1,1,D] row). base = served "
                   "trtllm::flux3_sm120_nvfp4_out_pair_prequant + compiled residual pointwise; "
                   "served (branch) = trtllm::flux3_sm120_nvfp4_tail_one_launch for calls in its "
                   "M-range bucket (default M >= 24576), main's tail otherwise")
    out["samples"], out["warmup"] = SAMPLES, WARMUP
    return out


def check():
    assert HAS_FUSED, "check runs on the branch tree"
    plans = make_plans()
    pristine = {}
    for plan in plans.values():
        for c, _ in plan:
            if id(c) not in pristine:
                pristine[id(c)] = (c, {k: c[k].clone() for k in INPUTS})
    fails, summary = [], {}
    for label in ("t2i-2048", "i2i-2048", "t2i-2048"):
        plan = plans[label]
        gb, outs_b = capture(plan, base_call)
        gf, outs_f = capture(plan, fused_call)
        first = None
        n_eq = 0
        for it in range(WARMUP + SAMPLES):
            for o in outs_b + outs_f:
                o.fill_(float("nan"))
            gb.replay()
            gf.replay()
            torch.cuda.synchronize()
            for j, (ob, of) in enumerate(zip(outs_b, outs_f)):
                if torch.isnan(ob).any():
                    fails.append(f"{label} it {it} call {j}: base output still NaN")
                if not torch.equal(of, ob):
                    fails.append(f"{label} it {it} call {j}: fused != base "
                                 f"({int((of != ob).sum())} elems)")
                else:
                    n_eq += 1
            if first is None:
                first = [o.clone() for o in outs_f]
            elif not all(torch.equal(a, b) for a, b in zip(outs_f, first)):
                fails.append(f"{label} it {it}: fused != iteration 0")
            for c, p in pristine.values():
                if not all(torch.equal(c[k], p[k]) for k in INPUTS):
                    fails.append(f"{label} it {it}: inputs of M={c['M']} changed")
                    break
            if len(fails) > 20:
                break
        summary.setdefault(label, []).append({"replays": WARMUP + SAMPLES, "calls_equal": n_eq,
                                              "calls": (WARMUP + SAMPLES) * len(plan)})
        log(label, json.dumps(summary[label][-1]), "failures", len(fails))
        del gb, gf, outs_b, outs_f, first
        torch.cuda.empty_cache()
    return {"passed": not fails, "failures": fails[:20], "summary": summary,
            "compared": "trtllm::flux3_sm120_nvfp4_tail_one_launch vs served "
                        "trtllm::flux3_sm120_nvfp4_out_pair_prequant + compiled residual, the "
                        "timed graphs' plans, warmup and replay count, cases t2i -> i2i -> t2i, "
                        "NaN-poisoned outputs, bitwise per call and vs iteration 0; inputs "
                        "unchanged"}


def inter():
    """Branch only. Per served bucket, one process, interleaved graph replays of base (main's
    served tail), base2 (A/A copy) and fused (the op); each graph = 8 tail calls (2 sets cycled).
    Buckets: 32848 per-token gate (i2i single), 16464 table + select (t2i single), 16384 row
    (mode blocks)."""
    assert HAS_FUSED
    rounds = int(os.environ.get("B3_ROUNDS", "40"))
    out = {}
    for M, layout in ((M_I2I, "pertok"), (M_T2I, "single"), (M_MODE, "mode")):
        sets = [make(M, 11), make(M, 12)]
        plan = [(sets[j % 2], layout) for j in range(8)]
        graphs = {"base": capture(plan, base_call)[0], "fused": capture(plan, fused_call)[0],
                  "base2": capture(plan, base_call)[0],
                  "pairA_alongN_pw": capture(plan, pairA_pw_call)[0]}
        for g in graphs.values():
            g.replay()
        torch.cuda.synchronize()
        times = {k: [] for k in graphs}
        keys = list(graphs)
        for r in range(rounds):
            order = keys if r % 2 == 0 else keys[::-1]
            for k in order:
                torch.cuda._sleep(2_000_000)
                a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                a.record()
                graphs[k].replay()
                b.record()
                b.synchronize()
                times[k].append(a.elapsed_time(b) / len(plan))
        med = {k: stat(v)[0] for k, v in times.items()}
        ratio = sorted(f / b for f, b in zip(times["fused"], times["base"]))
        aa = sorted(b2 / b for b2, b in zip(times["base2"], times["base"]))
        n = len(ratio)
        out[f"{M}_{layout}"] = {
            "per_call_ms": med, "fused_minus_base_us": (med["fused"] - med["base"]) * 1e3,
            "fused_over_base": {"median": ratio[n // 2], "p10": ratio[n // 10],
                                "p90": ratio[(9 * n) // 10]},
            "aa_base2_over_base": {"median": aa[n // 2], "p10": aa[n // 10],
                                   "p90": aa[(9 * n) // 10]}}
        log(M, layout, json.dumps(out[f"{M}_{layout}"]))
        del sets, graphs
        torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    t0 = time.time()
    res = {"pair": pair, "check": check, "inter": inter}[sys.argv[1]]()
    res["wall_s"] = round(time.time() - t0, 1)
    print(json.dumps(res))
