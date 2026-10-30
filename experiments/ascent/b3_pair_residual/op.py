"""Branch 3 (idea 12): op-level check and timing of the one-launch block tail
(flux3_sm120_pair_residual.run) vs main's flux3_sm120_lpt_pair.run + compiled residual pointwise.

Synthetic NVFP4 tensors at served shapes (no model load).
  check : untimed. Per M in (4176, 16896, 33280) and per mod layout (single: [2,N] table +
          select; mode: [1,N] row): mode-1 debug outputs (attn_y, mlp_y) vs lpt.run bitwise;
          x_new vs lpt.run + compiled pointwise (emulate_precision_casts) bitwise, over 3 repeated
          calls each compared to the reference and call 0; inputs unchanged; CUDA-graph replay x3
          bitwise equal to eager.
  bench : per M, CUDA graphs of base (pair + pointwise, 2 input sets) vs fused, interleaved.
Last stdout line: JSON.
"""

import json
import os
import sys
import time

import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp

H, FF = 3072, 9216
T_TEXT = 80  # served text length (every frozen prompt pads to the first 80-token bucket)
M_T2I, M_I2I, M_MODE = 16464, 32848, 16384  # served single-block (t2i, i2i) and mode-block M


def log(*a):
    print(*a, flush=True)


def _inv(s):
    return torch.tensor([1.0 / s], device="cuda", dtype=torch.float32)


def make(M, seed, T=T_TEXT):
    g = torch.Generator(device="cuda").manual_seed(seed)
    hid = torch.randn(M, FF, device="cuda", dtype=torch.bfloat16, generator=g)
    att = torch.randn(M, H, device="cuda", dtype=torch.bfloat16, generator=g)
    wd = torch.randn(H, FF, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    wo = torch.randn(H, H, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    c = {"M": M}
    c["a0"], c["sfa0"] = fp.nvfp4_quantize(hid, _inv(float(hid.abs().amax()) / 2688.0))
    c["a1"], c["sfa1"] = fp.nvfp4_quantize(att, _inv(float(att.abs().amax()) / 2688.0))
    c["b0"], c["sfb0"] = fp.nvfp4_quantize(wd, _inv(float(wd.abs().amax()) / 2688.0))
    c["b1"], c["sfb1"] = fp.nvfp4_quantize(wo, _inv(float(wo.abs().amax()) / 2688.0))
    c["al0"] = torch.tensor([3.1e-3], device="cuda")
    c["al1"] = torch.tensor([1.7e-3], device="cuda")
    c["res"] = torch.randn(1, M, H, device="cuda", dtype=torch.bfloat16, generator=g)
    c["table"] = torch.randn(1, 2, H, device="cuda", dtype=torch.bfloat16, generator=g)
    c["row"] = torch.randn(1, 1, H, device="cuda", dtype=torch.bfloat16, generator=g)
    sel = torch.zeros(1, M, 1, device="cuda", dtype=torch.bool)
    sel[:, max(M - T, 0):] = True
    c["sel"] = sel
    c["pertok"] = torch.randn(1, M, H, device="cuda", dtype=torch.bfloat16, generator=g)
    c["table3"] = torch.randn(1, 3, H, device="cuda", dtype=torch.bfloat16, generator=g)
    idx = torch.zeros(1, M, 1, device="cuda", dtype=torch.uint8)
    idx[:, M // 2:] = 1
    idx[:, max(M - T, 0):] = 2
    c["idx3"] = idx
    c["d0"] = torch.empty(M, H, device="cuda", dtype=torch.bfloat16)
    c["d1"] = torch.empty(M, H, device="cuda", dtype=torch.bfloat16)
    c["out"] = torch.empty(1, M, H, device="cuda", dtype=torch.bfloat16)
    del hid, att, wd, wo
    return c


def pw_single(res, a, m, table, sel):
    gate = torch.where(sel, table[:, 1:2], table[:, 0:1])
    out = a.reshape(1, -1, H) + m.reshape(1, -1, H)
    return res + gate * out


def pw_mode(res, a, m, row):
    out = a.reshape(1, -1, H) + m.reshape(1, -1, H)
    return res + row * out


def pw_k3(res, a, m, table3, idx):
    gate = table3[0][idx.reshape(-1).long()].unsqueeze(0)
    out = a.reshape(1, -1, H) + m.reshape(1, -1, H)
    return res + gate * out


_OPT = {"emulate_precision_casts": True}
pw_single_c = torch.compile(pw_single, dynamic=False, fullgraph=True, options=_OPT)
pw_mode_c = torch.compile(pw_mode, dynamic=False, fullgraph=True, options=_OPT)
pw_k3_c = torch.compile(pw_k3, dynamic=False, fullgraph=True, options=_OPT)


def lpt_pair(c):
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_lpt_pair as lpt

    lpt.run(c["a0"], c["b0"], c["sfa0"], c["sfb0"], c["al0"], c["d0"],
            c["a1"], c["b1"], c["sfa1"], c["sfb1"], c["al1"], c["d1"])


def base(c, layout):
    lpt_pair(c)
    if layout == "single":
        return pw_single_c(c["res"], c["d1"], c["d0"], c["table"], c["sel"])
    if layout == "pertok":
        return pw_mode_c(c["res"], c["d1"], c["d0"], c["pertok"])
    if layout == "k3":
        return pw_k3_c(c["res"], c["d1"], c["d0"], c["table3"], c["idx3"])
    return pw_mode_c(c["res"], c["d1"], c["d0"], c["row"])


def fused(c, layout, out=None, mode=0, dbg=None, prefetch=None, early=None):
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_tail_one_launch as pr

    out = c["out"] if out is None else out
    gate_mode = None
    if layout == "single":
        gate, sel = c["table"].reshape(2, H), c["sel"].reshape(-1)
    elif layout == "pertok":
        gate, sel, gate_mode = c["pertok"], None, 2
    elif layout == "k3":
        gate, sel = c["table3"].reshape(3, H), c["idx3"].reshape(-1)
    else:
        gate, sel = c["row"].reshape(1, H), None
    pr.run(c["a1"], c["b1"], c["sfa1"], c["sfb1"], c["al1"],
           c["a0"], c["b0"], c["sfa0"], c["sfb0"], c["al0"],
           c["res"], gate, sel, out, gate_mode=gate_mode, mode=mode,
           dbg_attn=None if dbg is None else dbg[0], dbg_mlp=None if dbg is None else dbg[1],
           prefetch=prefetch, early=early)
    return out


INPUT_KEYS = ("a0", "sfa0", "b0", "sfb0", "a1", "sfa1", "b1", "sfb1", "al0", "al1", "res",
              "table", "row", "sel", "pertok", "table3", "idx3")


def check():
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_tail_one_launch as pr

    log("pair_residual info", pr.module().pair_residual_info(), "param bytes",
        pr.module().pair_residual_param_bytes())
    res = {}
    ok_all = True
    for M in (4176, M_MODE, M_T2I, M_I2I):
        c = make(M, 11)
        pristine = {k: c[k].clone() for k in INPUT_KEYS}
        # debug mode: attn_y / mlp_y vs LPT pair
        lpt_pair(c)
        torch.cuda.synchronize()
        ref_attn, ref_mlp = c["d1"].clone(), c["d0"].clone()
        da = torch.full_like(ref_attn, float("nan"))
        dm = torch.full_like(ref_mlp, float("nan"))
        fused(c, "mode", mode=1, dbg=(da, dm))
        torch.cuda.synchronize()
        r = {"attn_y_bitwise": bool(torch.equal(da, ref_attn)),
             "mlp_y_bitwise": bool(torch.equal(dm, ref_mlp))}
        if not r["attn_y_bitwise"]:
            r["attn_y_mismatch"] = int((da != ref_attn).sum())
        if not r["mlp_y_bitwise"]:
            r["mlp_y_mismatch"] = int((dm != ref_mlp).sum())
        for layout in ("single", "mode", "pertok", "k3"):
            for k in INPUT_KEYS:
                c[k].copy_(pristine[k])
            ref = base(c, layout).clone()
            torch.cuda.synchronize()
            outs = []
            for prefetch, early in ((1, 1), (0, 0), (1, 0), (1, 1)):
                out = torch.full_like(c["out"], float("nan"))
                fused(c, layout, out=out, prefetch=prefetch, early=early)
                torch.cuda.synchronize()
                outs.append(out)
            eq_ref = [bool(torch.equal(o, ref)) for o in outs]
            eq_0 = [bool(torch.equal(o, outs[0])) for o in outs]
            mism = int((outs[0] != ref).sum())
            inputs_ok = all(torch.equal(c[k], pristine[k]) for k in INPUT_KEYS)
            # CUDA graph replay
            gout = torch.full_like(c["out"], float("nan"))
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                fused(c, layout, out=gout)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                fused(c, layout, out=gout)
            graph_ok = []
            for _ in range(3):
                gout.fill_(float("nan"))
                g.replay()
                torch.cuda.synchronize()
                graph_ok.append(bool(torch.equal(gout, ref)))
            maxdiff = float((outs[0].float() - ref.float()).abs().max())
            r[layout] = {"eq_ref": eq_ref, "eq_call0": eq_0, "mismatch": mism,
                         "max_abs_diff": maxdiff, "inputs_unchanged": inputs_ok,
                         "graph_replay_eq_ref": graph_ok}
            ok_all &= all(eq_ref) and all(eq_0) and inputs_ok and all(graph_ok)
        ok_all &= r["attn_y_bitwise"] and r["mlp_y_bitwise"]
        res[str(M)] = r
        log(M, json.dumps(r))
        del c, pristine
        torch.cuda.empty_cache()
    return {"passed": bool(ok_all), "compared": "x_new vs flux3_sm120_lpt_pair.run + "
            "torch.compile(emulate_precision_casts) residual pointwise (single [1,2,D]+select and "
            "mode [1,1,D] layouts); debug attn_y/mlp_y vs lpt.run; bitwise", "cases": res}


def capture(fn):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        fn()
    torch.cuda.synchronize()
    return g


def stat(v):
    v = sorted(v)
    n = len(v)
    return v[n // 2], (v[(9 * n) // 10] - v[n // 10]) / v[n // 2]


def bench():
    """run_gpu_pair unit: the tree's served tail for one call (base tree: pair + pointwise;
    branch: fused when present). Per M, 2 input sets, single layout (28 of 32 blocks)."""
    have = os.path.exists(os.path.join(os.path.dirname(fp.__file__),
                                       "flux3_sm120_tail_one_launch.py"))
    arm = os.environ.get("B3_ARM", "fused" if have else "base")
    samples = int(os.environ.get("B3_SAMPLES", "50"))
    out = {"batch": {}, "arm": arm}
    mode_sets = [make(M_MODE, 5), make(M_MODE, 6)]
    # One denoise step's 32 NVFP4 block tails per case: 28 single blocks (t2i: M16464 with the
    # [1,2,D] table + select; i2i: M32848 with main's per-token [1,S,D] gate) + 4 image mode
    # blocks (M16384, [1,1,D] row).
    for label, M, layout in (("t2i-2048", M_T2I, "single"), ("i2i-2048", M_I2I, "pertok")):
        sets = [make(M, 1), make(M, 2)]
        plan = [(sets[i % 2], layout) for i in range(28)] + \
            [(mode_sets[i % 2], "mode") for i in range(4)]

        def f():
            for c, lay in plan:
                if arm == "fused":
                    fused(c, lay)
                else:
                    base(c, lay)

        g = capture(f)
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        times = []
        for i in range(samples):
            torch.cuda._sleep(2_000_000)
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record()
            g.replay()
            b.record()
            b.synchronize()
            times.append(a.elapsed_time(b))
            if i % 10 == 0:
                log(f"  {label} sample {i}: {times[-1]:.3f} ms")
        med, noise = stat(times)
        out["batch"][label] = {"kernel_ms": med, "noise_floor": noise, "M": M,
                               "p10_ms": sorted(times)[len(times) // 10],
                               "p90_ms": sorted(times)[(9 * len(times)) // 10]}
        log(label, M, arm, med, noise)
        del sets, g, plan
        torch.cuda.empty_cache()
    out["output"] = {"class": "exact"}
    out["unit"] = ("one CUDA-graph replay = one denoise step's 32 NVFP4 block tails (28 single "
                   "blocks at the case's served M (t2i 16464 table+select, i2i 32848 per-token "
                   "gate) + 4 image mode blocks at M16384 [1,1,D] row): base = "
                   "flux3_sm120_lpt_pair.run + torch.compile(emulate_precision_casts) residual "
                   "pointwise; fused = flux3_sm120_pair_residual.run (one launch)")
    return out


def inter():
    """Diagnostics: base / fused (prefetch on/off) interleaved in one process."""
    samples = int(os.environ.get("B3_SAMPLES", "40"))
    out = {}
    for M in (M_T2I, M_I2I):
        sets = [make(M, 1), make(M, 2)]
        arms = {
            "base": lambda: [base(c, "single") for c in sets],
            "pair_only": lambda: [lpt_pair(c) for c in sets],
            "fused_pf1_e0": lambda: [fused(c, "single", prefetch=1, early=0) for c in sets],
            "fused_pf1_e1": lambda: [fused(c, "single", prefetch=1, early=1) for c in sets],
            "fused_pf0_e1": lambda: [fused(c, "single", prefetch=0, early=1) for c in sets],
            "fused_mode_e1": lambda: [fused(c, "mode", early=1) for c in sets],
            "base_pertok": lambda: [base(c, "pertok") for c in sets],
            "fused_pertok_e1": lambda: [fused(c, "pertok", early=1) for c in sets],
        }
        graphs = {k: capture(v) for k, v in arms.items()}
        times = {k: [] for k in graphs}
        for r in range(samples):
            for k, g in graphs.items():
                torch.cuda._sleep(2_000_000)
                a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                a.record()
                for _ in range(3):
                    g.replay()
                b.record()
                b.synchronize()
                times[k].append(a.elapsed_time(b) / 3 / len(sets))
        out[str(M)] = {k: stat(v) for k, v in times.items()}
        log(M, json.dumps(out[str(M)]))
        del sets, graphs
        torch.cuda.empty_cache()
    return out


def diag():
    """Diagnostics at both Ms: fused vs its mainloops alone (mode 2) vs the LPT pair alone."""
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_tail_one_launch as pr

    samples = int(os.environ.get("B3_SAMPLES", "30"))
    out = {"stages_smem": pr.module().pair_residual_info()}
    for M in (M_T2I, M_I2I):
        sets = [make(M, 1), make(M, 2)]
        dummy = torch.empty(1, device="cuda", dtype=torch.bfloat16)

        def ml_only(c):
            pr.run(c["a1"], c["b1"], c["sfa1"], c["sfb1"], c["al1"],
                   c["a0"], c["b0"], c["sfa0"], c["sfb0"], c["al0"],
                   c["res"], c["row"].reshape(1, H), None, c["out"], mode=2)

        arms = {
            "pair_only": lambda: [lpt_pair(c) for c in sets],
            "fused": lambda: [fused(c, "single") for c in sets],
            "mainloops_only": lambda: [ml_only(c) for c in sets],
        }
        del dummy
        graphs = {k: capture(v) for k, v in arms.items()}
        times = {k: [] for k in graphs}
        for r in range(samples):
            for k, g in graphs.items():
                torch.cuda._sleep(2_000_000)
                a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                a.record()
                for _ in range(3):
                    g.replay()
                b.record()
                b.synchronize()
                times[k].append(a.elapsed_time(b) / 3 / len(sets))
        out[str(M)] = {k: stat(v) for k, v in times.items()}
        log(M, json.dumps(out[str(M)]))
        del sets, graphs
        torch.cuda.empty_cache()
    return out


def alt():
    """Untimed repeat check: ROUNDS x (t2i graph -> eager M=4176 -> i2i graph -> t2i graph).
    t2i graph: 28 single-layout calls at M=16896 + 4 mode-layout calls at 16384 (2 sets cycled);
    i2i graph: 28 per-token-gate calls at 33280 + 4 mode calls at 16384. Outputs are NaN-poisoned
    before every replay and compared bitwise with main's pair + compiled residual (computed once
    per call) and with round 0; inputs are checked unchanged after every round."""
    rounds = int(os.environ.get("B3_ROUNDS", "50"))
    sets = {M_T2I: [make(M_T2I, 1), make(M_T2I, 2)], M_I2I: [make(M_I2I, 3), make(M_I2I, 4)],
            M_MODE: [make(M_MODE, 5), make(M_MODE, 6)], 4176: [make(4176, 7)]}
    plans = {
        "t2i": [(sets[M_T2I][i % 2], "single") for i in range(28)]
        + [(sets[M_MODE][i % 2], "mode") for i in range(4)],
        "i2i": [(sets[M_I2I][i % 2], "pertok") for i in range(28)]
        + [(sets[M_MODE][i % 2], "mode") for i in range(4)],
    }
    pristine = {(M, i): {k: c[k].clone() for k in INPUT_KEYS}
                for M, ss in sets.items() for i, c in enumerate(ss)}
    # One output buffer per call slot (the graph's static outputs), refs per (set, layout).
    outs = {name: [torch.empty(1, c["M"], H, device="cuda", dtype=torch.bfloat16)
                   for c, _ in plan] for name, plan in plans.items()}
    refs = {}
    for name, plan in plans.items():
        for c, layout in plan:
            key = (id(c), layout)
            if key not in refs:
                refs[key] = base(c, layout).clone()
    e_c = sets[4176][0]
    e_ref = base(e_c, "single").clone()
    torch.cuda.synchronize()

    def run_plan(name):
        for (c, layout), o in zip(plans[name], outs[name]):
            fused(c, layout, out=o)

    graphs = {}
    s = torch.cuda.Stream()
    for name in plans:
        with torch.cuda.stream(s):
            run_plan(name)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            run_plan(name)
        graphs[name] = g
    first = {}
    fails = []
    order = ("t2i", "eager", "i2i", "t2i2")
    for r in range(rounds):
        for step in order:
            if step == "eager":
                o = torch.full_like(e_c["out"], float("nan"))
                fused(e_c, "single", out=o)
                torch.cuda.synchronize()
                got = [o]
                want = [e_ref]
            else:
                name = "t2i" if step.startswith("t2i") else "i2i"
                for o in outs[name]:
                    o.fill_(float("nan"))
                graphs[name].replay()
                torch.cuda.synchronize()
                got = outs[name]
                want = [refs[(id(c), layout)] for c, layout in plans[name]]
            for j, (gt, wt) in enumerate(zip(got, want)):
                if not torch.equal(gt, wt):
                    fails.append(f"round {r} {step} call {j} != reference")
                key = (step, j)
                if r == 0:
                    first[key] = gt.clone()
                elif not torch.equal(gt, first[key]):
                    fails.append(f"round {r} {step} call {j} != round 0")
        for (M, i), p in pristine.items():
            c = sets[M][i]
            if not all(torch.equal(c[k], p[k]) for k in INPUT_KEYS):
                fails.append(f"round {r}: inputs of M={M} set {i} changed")
        if r % 10 == 0:
            log(f"alt round {r}: failures so far {len(fails)}")
        if len(fails) > 20:
            break
    return {"passed": not fails, "rounds": rounds, "failures": fails[:20],
            "compared": "flux3_sm120_pair_residual graphs (t2i: 28 x M16464 single + 4 x M16384 "
                        "mode; i2i: 28 x M32848 per-token + 4 x M16384 mode) and an eager M4176 "
                        "call, alternated t2i->eager->i2i->t2i, NaN-poisoned outputs, bitwise vs "
                        "lpt pair + compiled residual and vs round 0; inputs unchanged"}


def prof():
    """ncu target: 6 fused calls (single layout) at M = B3_M (default 16896)."""
    M = int(os.environ.get("B3_M", str(M_T2I)))
    c = make(M, 1)
    for i in range(6):
        fused(c, os.environ.get("B3_LAYOUT", "single"))
    torch.cuda.synchronize()
    return {"M": M}


if __name__ == "__main__":
    torch.manual_seed(0)
    t0 = time.time()
    res = {"check": check, "bench": bench, "inter": inter, "diag": diag, "prof": prof,
           "alt": alt}[sys.argv[1]]()
    res["wall_s"] = round(time.time() - t0, 1)
    print(json.dumps(res))
