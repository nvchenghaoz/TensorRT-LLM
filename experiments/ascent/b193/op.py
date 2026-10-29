"""Branch 193 (idea 283): LN_ID variant of the served warp adaLN+NVFP4 kernel.

  op.py check  -> HEAD probe; LN_ID=1 vs LN_ID=0 (affine path, same identity tensors) BITWISE on fp4,
                  full poisoned SF and forced-store x_mod at M=80/4096/4176/4256, select + mode,
                  normal + 4x saturating gs, store on/off, plus an inf/NaN-row case; checks which
                  variant was compiled per call; non-identity controls (one 0x3F81 weight, one -0.0
                  bias, unregistered clone, in-place-bumped version) must take the affine path;
                  untimed 23-iteration repeat (resets every 8) vs iteration 0 and the A reference.
  op.py aa     -> ONE process: CUDA graphs of 28 M4176 select calls each after a gated-residual
                  addcmul writing x; arms LN_ID 0 / 1 / 0' (A/A) + residual-only, >=30 rotated
                  rounds; then the untimed repeated-replay sequence (same warmup / replay counts)
                  bitwise vs iteration 0 and the eager LN_ID=0 reference, inputs vs pristine.
  op.py pair   -> same graph through wk.launch with identity weights registered when the tree has
                  the registry (base tree: affine path). Prints the run_gpu_pair evidence JSON.
"""

import json
import os
import statistics
import subprocess
import sys
import types

import torch

from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_adaln_quant as faq
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_adaln_nvfp4 as wk

D = 3072
EPS = 1e-6
dev = "cuda"
MODE = sys.argv[1] if len(sys.argv) > 1 else "check"
ROUNDS = int(os.environ.get("B193_ROUNDS", "60"))
CALLS = 28
REPLAYS = 3
g = torch.Generator(device=dev).manual_seed(1234)
HAS_REG = hasattr(wk, "register_identity_affine")

# Identity affine exactly as LayerNorm(has_weights=False, has_bias=False) builds it.
W1 = torch.ones(D, dtype=torch.bfloat16, device=dev)
B0 = torch.zeros(D, dtype=torch.bfloat16, device=dev)
W_V = torch.ones(D, dtype=torch.bfloat16, device=dev)
B_V = torch.zeros(D, dtype=torch.bfloat16, device=dev)
if HAS_REG:
    faq._register_identity_ln(types.SimpleNamespace(weight=W1, bias=B0))

DEFS_SEEN = []
_orig_compile = wk._compile


def _spy_compile(device, warps, use_pdl, extra_defines=()):
    DEFS_SEEN.append(tuple(extra_defines))
    return _orig_compile(device, warps, use_pdl, extra_defines)


wk._compile = _spy_compile


def last_ln_id():
    return "LN_ID=1" in DEFS_SEEN[-1]


def heavy(*shape):
    base = torch.randn(*shape, device=dev, generator=g)
    mag = torch.exp(torch.randn(*shape[:-1], 1, device=dev, generator=g) * 2.0)
    chan = torch.exp(torch.randn(shape[-1], device=dev, generator=g) * 1.5)
    return (base * mag * chan).to(torch.bfloat16)


def make_cases():
    cases = []
    for s in (80, 4096, 4176, 4256):
        x = heavy(1, s, D)
        mod = (torch.randn(1, 2, 9 * D, device=dev, generator=g) * 0.5).to(torch.bfloat16)
        shift_t, scale_t = mod[..., D: 2 * D], mod[..., 4 * D: 5 * D]
        select = torch.zeros(1, s, 1, dtype=torch.bool, device=dev)
        select[:, : (40 if s == 80 else 80)] = True
        cases.append((f"select_M{s}", (x, select, scale_t, shift_t)))
        table = (torch.randn(1, 16, 3 * D, device=dev, generator=g) * 0.5).to(torch.bfloat16)
        row = table[:, 5:6]
        cases.append((f"mode_M{s}", (x, None, row[..., D: 2 * D], row[..., :D])))
    # inf / NaN rows plus constant rows (x == mu: zero numerators) at the served shape
    x = heavy(1, 4176, D)
    x[0, 3, 7] = float("inf")
    x[0, 9, 100] = float("nan")
    x[0, 11, :] = -float("inf")
    x[0, 20, :] = 1.5
    x[0, 21, :] = -0.0
    mod = (torch.randn(1, 2, 9 * D, device=dev, generator=g) * 0.5).to(torch.bfloat16)
    select = torch.zeros(1, 4176, 1, dtype=torch.bool, device=dev)
    select[:, :80] = True
    cases.append(("special_select_M4176", (x, select, mod[..., D: 2 * D], mod[..., 4 * D: 5 * D])))
    return cases


def gs_for(x):
    xf = x.float().abs()
    xf = torch.where(torch.isfinite(xf), xf, torch.zeros_like(xf))
    return float(torch.tensor(2688.0 / xf.max().item(), dtype=torch.float32))


def poisoned(x, byte=0xCD):
    m = x.numel() // D
    xm = torch.empty_like(x)
    xm.view(torch.uint8).fill_(byte)
    return (
        xm,
        torch.full((m, D // 2), byte, dtype=torch.uint8, device=dev),
        torch.full((faq._scale_size(m, D),), byte, dtype=torch.uint8, device=dev),
    )


def same(a, b):
    return (torch.equal(a[0].view(torch.int16), b[0].view(torch.int16))
            and torch.equal(a[1], b[1]) and torch.equal(a[2], b[2]))


def call(args, gs_t, store, w=W1, b=B0, ln_id=True, outs=None):
    x, sel, sc, sh = args
    saved = getattr(wk, "_LN_ID", None)
    if saved is not None:
        wk._LN_ID = ln_id
    try:
        return wk.launch(x, sel, sc, sh, w, b, gs_t, EPS, store_x_mod=store,
                         outs=outs if outs is not None else poisoned(x))
    finally:
        if saved is not None:
            wk._LN_ID = saved


def head_info():
    wt = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(wt, "..", "..", ".."))

    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, cwd=root, capture_output=True, text=True,
                                  timeout=30).stdout.strip()
        except Exception as e:  # noqa
            return f"ERR {e}"

    info = {
        "head": sh("git rev-parse HEAD"),
        "log": sh("git log --oneline -5"),
        "anc_9e499b22": sh("git merge-base --is-ancestor 9e499b22 HEAD; echo rc=$?"),
        "anc_69eac46e": sh("git merge-base --is-ancestor 69eac46e HEAD; echo rc=$?"),
        "gpu": sh("nvidia-smi --query-gpu=name,uuid,clocks.sm --format=csv,noheader"),
        "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "host": sh("hostname"),
        "grep_defaults": sh("grep -n '_DEFAULT_DEFINES = \\|_DEFAULT_WARPS = \\|X_POLICY\\|LN_ID' "
                            "tensorrt_llm/_torch/visual_gen/models/flux3/flux3_sm120_adaln_nvfp4.py"
                            " | head -30"),
        "grep_xpolicy_env": sh("grep -rn 'TRTLLM_FLUX3_ADALN_WARP_X_POLICY' tensorrt_llm/ | head -5"),
        "has_registry": HAS_REG,
    }
    for k, val in info.items():
        print("INFO", k, val, flush=True)
    return info


def run_check():
    info = head_info()
    ok_all = True
    rep = {}
    bad = []
    path_bad = []
    with torch.inference_mode():
        cases = make_cases()
        for name, args in cases:
            x = args[0]
            gs = gs_for(x)
            pristine = [a.clone() if a is not None else None for a in args]
            for gname, gsv in (("gs", gs), ("gs_sat4x", float(torch.tensor(gs * 4.0, dtype=torch.float32)))):
                gs_t = torch.tensor([gsv], dtype=torch.float32, device=dev)
                for store in (True, False):
                    ref = call(args, gs_t, store, ln_id=False)
                    if last_ln_id():
                        path_bad.append(f"{name}|ref_took_LN_ID")
                    out = call(args, gs_t, store, ln_id=True)
                    if not last_ln_id():
                        path_bad.append(f"{name}|cand_missed_LN_ID")
                    if not same(out, ref):
                        bad.append(f"{name}|{gname}|store{int(store)}")
                    # cross-check against random-affine-free Triton reference path is not the
                    # subject; the A arm (affine warp kernel) is the protected reference.
            inp_ok = all(a is None or torch.equal(a.view(torch.int16) if a.dtype == torch.bfloat16 else a,
                                                  p.view(torch.int16) if p.dtype == torch.bfloat16 else p)
                         for a, p in zip(args, pristine))
            ok_all &= inp_ok
            print(name, "bitwise:", not any(b.startswith(name + "|") for b in bad), "inputs_unchanged", inp_ok,
                  flush=True)
            rep[f"{name}|inputs_unchanged"] = inp_ok
        rep["bitwise_failures"] = bad
        rep["path_failures"] = path_bad
        ok_all &= not bad and not path_bad

        # Non-identity controls: must not register / must take the affine path.
        ctrl = {}
        args = cases[4][1]  # select_M4176
        gs_t = torch.tensor([gs_for(args[0])], dtype=torch.float32, device=dev)
        w_bad = W1.clone()
        w_bad.view(torch.int16)[1234] = 0x3F81
        b_neg = B0.clone()
        b_neg.view(torch.int16)[77] = -32768  # 0x8000 = -0.0
        w_copy, b_copy = W1.clone(), B0.clone()  # bitwise identity but never registered
        for cname, (w, b, do_reg) in {
            "weight_0x3F81": (w_bad, B0, True),
            "bias_neg0": (W1, b_neg, True),
            "unregistered_copy": (w_copy, b_copy, False),
        }.items():
            if do_reg:
                faq._register_identity_ln(types.SimpleNamespace(weight=w, bias=b))
            out = call(args, gs_t, True, w=w, b=b, ln_id=True)
            took = last_ln_id()
            ref = call(args, gs_t, True, w=w, b=b, ln_id=False)
            ctrl[cname] = {"took_LN_ID": took, "eq_affine": same(out, ref)}
            ok_all &= (not took) and same(out, ref)
        # the earlier registration of W1/B0 must still hold
        if True:
            call(args, gs_t, True, ln_id=True)
            ctrl["identity_still_registered"] = last_ln_id()
            ok_all &= last_ln_id()
        # version bump after registration -> miss
        w_v, b_v = W_V, B_V  # normal (non-inference) tensors: they carry a version counter
        faq._register_identity_ln(types.SimpleNamespace(weight=w_v, bias=b_v))
        call(args, gs_t, True, w=w_v, b=b_v, ln_id=True)
        hit_before = last_ln_id()
        w_v.mul_(1.0)
        call(args, gs_t, True, w=w_v, b=b_v, ln_id=True)
        hit_after = last_ln_id()
        ctrl["version_bump"] = {"hit_before": hit_before, "hit_after": hit_after}
        ok_all &= hit_before and not hit_after
        # env off arm == base launch: LN_ID never compiled with _LN_ID False
        rep["controls"] = ctrl
        print("CONTROLS", json.dumps(ctrl), flush=True)

        # Untimed repeat: warmup 3 + 20 iterations, resets (re-poison) every 8, store forced on.
        rep_ok = True
        for name, args in cases:
            if name not in ("select_M4176", "mode_M4096", "mode_M4256", "special_select_M4176"):
                continue
            gs_t = torch.tensor([gs_for(args[0])], dtype=torch.float32, device=dev)
            ref = call(args, gs_t, True, ln_id=False)
            pristine = [a.clone() if a is not None else None for a in args]
            outs = poisoned(args[0])
            first = None
            eq0 = eqref = inp = True
            for it in range(23):
                if it % 8 == 0:
                    inp &= all(a is None or torch.equal(a.view(torch.uint8), p.view(torch.uint8))
                               for a, p in zip(args, pristine))
                    for o in outs:
                        o.view(torch.uint8).fill_(0xCD)
                o = call(args, gs_t, True, ln_id=True, outs=outs)
                cur = tuple(t.clone() for t in o)
                if first is None:
                    first = cur
                eq0 &= same(cur, first)
                eqref &= same(cur, ref)
            inp &= all(a is None or torch.equal(a.view(torch.uint8), p.view(torch.uint8))
                       for a, p in zip(args, pristine))
            r = eq0 and eqref and inp
            rep[f"repeat|{name}"] = {"eq_iter0": eq0, "eq_ref": eqref, "inputs_pristine": inp}
            rep_ok &= r
            print("REPEAT", name, eq0, eqref, inp, flush=True)
        rep["repeat_passed"] = rep_ok
        ok_all &= rep_ok

        # Public op: identity tensors through torch.ops (dispatch auto, M4176 -> warp, LN_ID hit)
        args = cases[4][1]
        gs_t = torch.tensor([gs_for(args[0])], dtype=torch.float32, device=dev)
        wk._LN_ID = False
        a = tuple(t.clone() for t in torch.ops.trtllm.flux3_adaln_nvfp4_quant(
            args[0], args[1], args[2], args[3], W1, B0, gs_t, EPS, True))
        wk._LN_ID = True
        b = tuple(t.clone() for t in torch.ops.trtllm.flux3_adaln_nvfp4_quant(
            args[0], args[1], args[2], args[3], W1, B0, gs_t, EPS, True))
        rep["public_op_on_eq_off"] = same(a, b) and last_ln_id()
        ok_all &= rep["public_op_on_eq_off"]
    regs = {str(k): v for k, v in wk.NUM_REGS.items()}
    rep["regs"] = regs
    print("REGS", regs, flush=True)
    print(json.dumps({"passed": bool(ok_all), "info": info, **rep}))


def make_sets(nsets=4, s=4176):
    sets = []
    for _ in range(nsets):
        x = heavy(1, s, D)
        res = heavy(1, s, D)
        y = heavy(1, s, D)
        gate = (torch.randn(1, 1, D, device=dev, generator=g) * 0.3).to(torch.bfloat16)
        mod = (torch.randn(1, 2, 9 * D, device=dev, generator=g) * 0.5).to(torch.bfloat16)
        sel = torch.zeros(1, s, 1, dtype=torch.bool, device=dev)
        sel[:, :80] = True
        sc, sh = mod[..., D: 2 * D], mod[..., 4 * D: 5 * D]
        gs_t = torch.tensor([gs_for(x)], dtype=torch.float32, device=dev)
        outs = poisoned(x)
        sets.append([x, res, y, gate, sel, sc, sh, gs_t, outs])
    return sets


def build_graphs(sets, arms):
    """arms: name -> ln_id flag (None = residual only)."""

    def body(ln_id):
        for i in range(CALLS):
            x, res, y, gate, sel, sc, sh, gs_t, outs = sets[i % len(sets)]
            torch.addcmul(res, gate, y, out=x)
            if ln_id is None:
                continue
            call((x, sel, sc, sh), gs_t, False, ln_id=ln_id, outs=outs)

    graphs = {}
    for name, flag in arms.items():
        for _ in range(2):
            body(flag)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        n0 = len(DEFS_SEEN)
        with torch.cuda.graph(gr):
            body(flag)
        defs = DEFS_SEEN[n0:]
        n_ln = sum(1 for d in defs if "LN_ID=1" in d)
        print("GRAPH", name, "calls", len(defs), "LN_ID calls", n_ln, flush=True)
        graphs[name] = (gr, len(defs), n_ln)
    return graphs


def time_graphs(graphs, rounds):
    names = list(graphs)
    for _ in range(20):
        for n in names:
            graphs[n][0].replay()
    torch.cuda.synchronize()
    samples = {k: [] for k in names}
    for r in range(rounds):
        order = names[r % len(names):] + names[: r % len(names)]  # rotate arm order
        for name in order:
            gr = graphs[name][0]
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(REPLAYS):
                gr.replay()
            e.record()
            e.synchronize()
            samples[name].append(s.elapsed_time(e) / REPLAYS)
        if r % 10 == 0:
            print("round", r, {k: round(v[-1], 4) for k, v in samples.items()}, flush=True)
    return samples


def stats(v):
    st = sorted(v)
    med = statistics.median(st)
    p10, p90 = st[len(st) // 10], st[(9 * len(st)) // 10]
    return med, (p90 - p10) / abs(med) if med else float("inf")


def repeat_sequence(graphs, sets, pristine, ref_sets):
    """Untimed: same warmup (20) and replay count (rounds x REPLAYS) per arm, reset to pristine
    before; after every replay compare fp4/sf of every set with iteration 0 and the reference."""
    res = {}
    for name, (gr, ncalls, nln) in graphs.items():
        if ncalls == 0:
            continue
        restore(sets, pristine)
        first = None
        eq0 = eqref = True
        n_iter = 20 + ROUNDS * REPLAYS
        for it in range(n_iter):
            gr.replay()
            cur = [(s[8][1].clone(), s[8][2].clone()) for s in sets]
            if first is None:
                first = cur
            for (f4, sf), (f40, sf0), (rf4, rsf) in zip(cur, first, ref_sets):
                eq0 &= torch.equal(f4, f40) and torch.equal(sf, sf0)
                eqref &= torch.equal(f4, rf4) and torch.equal(sf, rsf)
        # inputs other than x (x is rewritten by the addcmul by design) unchanged
        inp = all(torch.equal(s[k].view(torch.uint8), p[k].view(torch.uint8))
                  for s, p in zip(sets, pristine) for k in (1, 2, 3, 4, 5, 6, 7))
        res[name] = {"eq_iter0": eq0, "eq_ref": eqref, "inputs_pristine": inp, "iters": n_iter}
        print("REPEAT", name, json.dumps(res[name]), flush=True)
    return res


def snapshot(sets):
    return [[t.clone() if isinstance(t, torch.Tensor) else tuple(o.clone() for o in t) for t in s]
            for s in sets]


def restore(sets, pristine):
    for s, p in zip(sets, pristine):
        for k in range(8):
            if s[k] is not None:
                s[k].copy_(p[k])
        for o, po in zip(s[8], p[8]):
            o.copy_(po)


def reference(sets):
    """Eager A reference (LN_ID 0) of the graph's per-set final state: x = res + gate*y."""
    out = []
    for x, res, y, gate, sel, sc, sh, gs_t, outs in sets:
        xx = torch.addcmul(res, gate, y)
        _, f4, sf = call((xx, sel, sc, sh), gs_t, False, ln_id=False)
        out.append((f4.clone(), sf.clone()))
    return out


def run_aa():
    info = head_info()
    with torch.inference_mode():
        sets = make_sets()
        pristine = snapshot(sets)
        ref_sets = reference(sets)
        arms = {"ln0": False, "ln1": True, "ln0p": False, "residual_only": None}
        restore(sets, pristine)
        graphs = build_graphs(sets, arms)
        restore(sets, pristine)
        samples = time_graphs(graphs, ROUNDS)
        ro = samples["residual_only"]
        out = {}
        for name in ("ln0", "ln1", "ln0p"):
            per = [(a - b) / CALLS * 1000.0 for a, b in zip(samples[name], ro)]
            med, nz = stats(samples[name])
            pm, pn = stats(per)
            out[name] = {"graph_ms": med, "graph_noise": nz, "op_us": pm, "op_noise": pn,
                         "ln_id_calls": graphs[name][2], "calls": graphs[name][1]}
        for a, b in (("ln1", "ln0"), ("ln0p", "ln0"), ("ln1", "ln0p")):
            d = [(x - y) / CALLS * 1000.0 for x, y in zip(samples[a], samples[b])]
            out[f"delta_{a}_minus_{b}_us_per_call"] = {
                "median": statistics.median(d), "neg_frac": sum(z < 0 for z in d) / len(d),
                "p10": sorted(d)[len(d) // 10], "p90": sorted(d)[(9 * len(d)) // 10]}
        print("AA", json.dumps(out), flush=True)
        rep = repeat_sequence(graphs, sets, pristine, ref_sets)
    ok = all(r["eq_iter0"] and r["eq_ref"] and r["inputs_pristine"] for r in rep.values())
    ok &= out["ln1"]["ln_id_calls"] == CALLS and out["ln0"]["ln_id_calls"] == 0 \
        and out["ln0p"]["ln_id_calls"] == 0
    print(json.dumps({"passed": bool(ok), "rounds": ROUNDS, "info": info, "aa": out, "repeat": rep}))


def run_pair():
    info = head_info()
    with torch.inference_mode():
        sets = make_sets()
        pristine = snapshot(sets)
        arms = {"served": True, "residual_only": None}
        graphs = build_graphs(sets, arms)
        restore(sets, pristine)
        samples = time_graphs(graphs, ROUNDS)
    per = [(a - b) / CALLS for a, b in zip(samples["served"], samples["residual_only"])]
    med, nz = stats(per)
    gm, gn = stats(samples["served"])
    print("PAIR", json.dumps({"op_ms": med, "op_noise": nz, "graph_ms": gm, "graph_noise": gn,
                              "ln_id_calls": graphs["served"][2], "has_registry": HAS_REG}), flush=True)
    print(json.dumps({"batch": {"1": {"kernel_ms": med, "noise_floor": nz}},
                      "output": {"class": "exact"},
                      "graph_ms_per_28": gm, "graph_noise": gn,
                      "ln_id_calls": graphs["served"][2], "head": info["head"],
                      "gpu": info["gpu"]}))


if __name__ == "__main__":
    {"check": run_check, "aa": run_aa, "pair": run_pair}[MODE]()
