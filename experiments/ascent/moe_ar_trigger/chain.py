"""Idea 208: MoE-AR early trigger + hc_attn K1 trigger-first, 1-GPU stand-in chain (SM120 decode M<=8).

Per boundary (36 per step, layers with / without index_wq_b interleaved as in the checkpoint):
  P  = stand-in MoE all-reduce: experiments/ascent/router_prewait/pdl_ext.py producer (plain launch, globaltimer
       spin of W us, then src -> x_prev copy; trigger at END or EARLY). Grid 4*M x 160 threads = the SM120 plain
       one-shot lamport AR launch shape at hidden 5120 (allReduceFusionKernels.cu kSm120DefaultCtasPerToken = 4).
  K1 = the real hc_attn K1 (mhc_lagged_split.mhc_lagged_layer_input, PDL), trigger_first per arm.
  A  = real wq_a||wkv low-M GEMM (trigger at start, bit 1), B = real fused q_norm + quant + wq_b(||index_wq_b),
       both through the tree's DeepseekV41Attention._project_inputs (csa2_shared_proj_bench.Probe, own weights per
       layer, ~25 MB/layer so every layer streams cold from DRAM).
  S  = non-PDL separator (the rest of the layer).
Arms: base (END, K1 wait-first) | a (EARLY, K1 wait-first) | ab (EARLY, K1 trigger-first). A tree without
trigger_first (the branch base) runs base and a only.

  --mode bench : per M and W, ABBA-interleaved CUDA-graph replays of every arm; us/boundary median, p10, p90.
                 The evidence JSON reports --pair-arm at W = --pair-w us (kernel_ms = one step of 36 boundaries).
  --mode check : candidate tree. Untimed repeated sequence: per iteration new inputs, NaN-poisoned x_prev and
                 outputs, every arm/W graph replayed and every boundary's K1 layer_input and A/B outputs (qr, q, swa,
                 index_q) compared bitwise with an eager no-overlap reference AND with the base graph. Arms also
                 include PDL0 (TRTLLM_ENABLE_PDL=0). No Triton JIT during capture. 1000 back-to-back replays at
                 M=1 and M=8. Negative controls each in a subprocess (--mode negctl --ctl k1nowait|lowmskip).
"""

import argparse
import inspect
import json
import os
import statistics
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "router_prewait"))
import csa2_shared_proj_bench as PB  # noqa: E402
import pdl_ext  # noqa: E402

from tensorrt_llm._torch.modules.mhc import mhc_lagged_split as S  # noqa: E402

HIDDEN = 5120
HAS_TF = "trigger_first" in inspect.signature(S.mhc_lagged_layer_input).parameters
HAS_GATE = hasattr(S, "K1_TRIGGER_FIRST_MIN_TOKENS")
TREE_MIN = getattr(S, "K1_TRIGGER_FIRST_MIN_TOKENS", 1)
ARMS = {"base": (False, False), "a": (True, False), "ab": (True, True)}
try:
    from tensorrt_llm._torch.models.modeling_deepseekv4 import _moe_ar_trigger_at_end as _TREE_AR_END
except Exception:  # noqa: BLE001  (branch base: the MoE AR always triggers at its end)
    _TREE_AR_END = None
try:  # idea 386: hc_attn K1 folded into A (csa2_mxfp8_low_m_gemm_k1fold); absent on the branch base
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import low_m_gemm as LMG

    HAS_FOLD = hasattr(LMG, "low_m_mm_k1fold")
except Exception:  # noqa: BLE001
    LMG, HAS_FOLD = None, False


def arm_params(arm, M):
    """(early AR trigger, K1 trigger_first kwarg, K1 min-tokens gate in force at capture).

    base / a / ab are forced arms (ab = trigger-first at every M, gate off); tree = what the tree's model runs
    at the hc_attn boundary: the MoE AR rule of modeling_deepseekv4 and trigger_first=True with the tree's own
    per-M gate (branch base: END trigger, wait-first K1)."""
    if arm in ("tree", "tree_fast0", "fold", "tree_k1"):
        early = _TREE_AR_END is not None and not _TREE_AR_END(M)
        return early, HAS_TF, TREE_MIN
    if arm == "ideal":  # idea 352: producer EARLY, no K1 (A is the producer's direct PDL dependent)
        return True, False, TREE_MIN
    early, tf = ARMS[arm]
    return early, tf, 1


def arm_k1(arm):
    """tree = the tree's production hc_attn boundary ('auto': the K1 fold where the tree's low_m_gemm admits it,
    else K1 -> A); fold / tree_k1 force one of the two (head tree only)."""
    return {"tree_fast0": "fast0", "ideal": "none", "tree": "auto", "fold": "fold", "tree_k1": "k1"}.get(arm, "k1")


def set_gate(v):
    if HAS_GATE:
        S.K1_TRIGGER_FIRST_MIN_TOKENS = v


def log(*a):
    print(*a, flush=True)


def layer_order(n_index, n_swa):
    order, acc = [], 0.0
    total = n_index + n_swa
    for _ in range(total):
        acc += n_index / total
        if acc >= 1.0 - 1e-9:
            order.append(True)
            acc -= 1.0
        else:
            order.append(False)
    return order


class Chain:
    def __init__(self, n_bnd, ckpt, P):
        import tensorrt_llm._torch.custom_ops  # noqa: F401

        argv = sys.argv
        sys.argv = [argv[0], "x", ckpt]  # csa2_shared_proj_bench.layer_counts reads argv[2]
        n_index, n_swa = PB.layer_counts()
        sys.argv = argv
        tot = n_index + n_swa
        # Scale the checkpoint's index/no-index split to n_bnd boundaries.
        ni = round(n_index * n_bnd / tot)
        self.order = layer_order(ni, n_bnd - ni)
        gen = torch.Generator(device="cuda").manual_seed(PB.SEED)
        self.probes = []
        for wi in self.order:
            p = PB.Probe(wi, gen)
            assert PB.attach_branch(p), "DeepseekV41Attention._project_inputs missing"
            self.probes.append(p)
        self.nw = [(1.0 + 0.1 * torch.randn(HIDDEN, generator=gen, device="cuda")).to(torch.bfloat16)
                   for _ in range(n_bnd)]
        self.n = n_bnd
        self.P = P
        self.scratch = torch.empty((8, 8), dtype=torch.bfloat16, device="cuda")
        if HAS_FOLD:
            LMG.k1fold_warmup(*self.probes[0]._wq_a_wkv)  # eager, before any capture
        log(f"chain: {n_bnd} boundaries ({ni} with index_wq_b), checkpoint split {n_index}/{n_swa}, P={P}, "
            f"trigger_first in tree: {HAS_TF}")

    def bufs(self, M):
        d = "cuda"
        return {
            "src": [torch.empty(M, HIDDEN, dtype=torch.bfloat16, device=d) for _ in range(self.n)],
            "x": [torch.empty(M, HIDDEN, dtype=torch.bfloat16, device=d) for _ in range(self.n)],
            "res": [torch.empty(M, 4, HIDDEN, dtype=torch.bfloat16, device=d) for _ in range(self.n)],
            "post": [torch.empty(M, 4, dtype=torch.float32, device=d) for _ in range(self.n)],
            "comb": [torch.empty(M, 16, dtype=torch.float32, device=d) for _ in range(self.n)],
            "pre": [torch.empty(M, 4, dtype=torch.float32, device=d) for _ in range(self.n)],
        }

    @staticmethod
    def fill(b, seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        for i in range(len(b["src"])):
            b["src"][i].copy_(torch.randn(b["src"][i].shape, generator=g, device="cuda"))
            b["res"][i].copy_(torch.randn(b["res"][i].shape, generator=g, device="cuda"))
            b["post"][i].copy_(torch.rand(b["post"][i].shape, generator=g, device="cuda") * 2)
            b["comb"][i].copy_(torch.rand(b["comb"][i].shape, generator=g, device="cuda") * 0.5)
            b["pre"][i].copy_(torch.rand(b["pre"][i].shape, generator=g, device="cuda") + 0.05)

    def precompute(self, b):
        """Idea 352 arm 'ideal': main's K1 (tree default) run eagerly on x_prev = src into per-boundary buffers,
        outside any capture; the ideal graph's A reads these instead of a K1 output."""
        if "li_pre" not in b:
            b["li_pre"] = [torch.empty_like(s) for s in b["src"]]
        for i in range(self.n):
            S.mhc_lagged_layer_input(b["src"][i], b["res"][i], b["post"][i], b["comb"][i], b["pre"][i],
                                     self.nw[i], 1e-6, self.P, out=b["li_pre"][i])
        torch.cuda.synchronize()

    def run(self, b, M, w_us, early, tf, k1_skip_wait=False, direct_x=False, k1="k1"):
        """One step. direct_x: eager reference (x_prev = src copied by torch, no producer overlap).
        k1: 'k1' (tree default K1), 'fast0' (main's pre-206 K1, fast=False), 'none' (idea 352 'ideal': no K1, A on
        b['li_pre'] from precompute())."""
        ext = pdl_ext.ext()
        outs = []
        for i in range(self.n):
            if direct_x:
                b["x"][i].copy_(b["src"][i])
            else:
                ext.producer(b["src"][i], b["x"][i], int(w_us * 1000), 4 * M, 160, pdl_ext.empty(), bool(early))
            kw = {}
            if HAS_TF:
                kw["trigger_first"] = bool(tf)
                if k1_skip_wait:
                    kw["_test_skip_wait"] = True
            if k1 == "fast0":
                kw["fast"] = False
            # 'fold' and 'auto' both fold exactly where the tree's gate admits it (M <= 2); elsewhere K1 -> A.
            use_fold = k1 in ("fold", "auto") and HAS_FOLD and LMG.k1fold_applies(
                M, self.P, self.probes[i]._wq_a_wkv[0])
            if use_fold:
                # Idea 386: no K1; A computes layer_input (pend.out) after its PDL wait on the producer.
                pend = S.K1FoldPending(torch.empty_like(b["x"][i]),
                                       (b["x"][i], b["res"][i], b["post"][i], b["comb"][i], b["pre"][i], self.nw[i]),
                                       1e-6, self.P)
                p = self.probes[i]
                qr, q, swa, idx = p._project_inputs(pend.out, k1_fold=pend)
                if p.with_index and idx is None:
                    idx = p.index_wq_b(qr)
                self.scratch[:M].copy_(q[:, :8])
                outs.append((pend.out, qr, q, swa, idx))
                continue
            if k1 == "none":
                li = b["li_pre"][i]
            else:
                li = S.mhc_lagged_layer_input(b["x"][i], b["res"][i], b["post"][i], b["comb"][i], b["pre"][i],
                                              self.nw[i], 1e-6, self.P, **kw)
            qr, q, swa, idx = PB.branch(self.probes[i], li)
            self.scratch[:M].copy_(q[:, :8])  # non-PDL separator: the rest of the layer
            outs.append((li, qr, q, swa, idx))
        return outs


def capture(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    c0 = k1_cache_size()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        out = fn()
    torch.cuda.synchronize()
    return g, out, k1_cache_size() - c0


def k1_cache_size():
    n = 0
    for kern in (S._mhc_lagged_layer_input_kernel, getattr(S, "_mhc_lagged_layer_input_fast_kernel", None)):
        for c in getattr(kern, "device_caches", {}).values():
            try:
                n += len(c[0])
            except Exception:  # noqa: BLE001
                pass
    return n


def pct(v, q):
    v = sorted(v)
    return v[int(q * (len(v) - 1))]


# ------------------------------------------------------------------------------------------------- bench
def beyond(win, lose, table, key):
    """True if arm `win` is faster than arm `lose` at table key prefix beyond the larger arm noise."""
    w, l_ = table[f"{key}_{win}"], table[f"{key}_{lose}"]
    gain = (l_["us_per_boundary"] - w["us_per_boundary"]) / l_["us_per_boundary"]
    return gain > max(w["noise"], l_["noise"]), gain


def bench(args):
    import wcrit

    # (i) W_crit: trace part (CPU) + stand-in overhead, printed before any arm is timed.
    wc = wcrit.compute(args.batches) if args.wcrit else {"W_crit": {M: {"W_crit_p10": args.pair_w,
                                                                          "W_crit_p50": args.pair_w,
                                                                          "source": "--pair-w"} for M in args.batches}}
    wtab = wc["W_crit"]
    log("DECLARED judged points (W_crit, us): " + json.dumps(wtab))
    log("DECLARED GATE: (a) on at M if a beats base beyond noise at W_crit p50; (b) on at M only if ab beats a "
        "beyond noise at W_crit p50 AND ab is not worse than a beyond noise at W_crit p10. Pair point: arm 'tree' "
        f"at W_crit p50 per M. Tree gate in force: K1 trigger-first from M >= {TREE_MIN} (HAS_GATE {HAS_GATE}).")
    ch = Chain(args.boundaries, args.ckpt, args.P)
    if args.arms:
        arms = list(args.arms)
    else:
        arms = [a for a in ("base", "a", "ab") if HAS_TF or a != "ab"] + ["tree"]
    table = {}
    for M in args.batches:
        b = ch.bufs(M)
        Chain.fill(b, 1)
        ch.precompute(b)
        P = ch.P
        W_ = P // 32
        log(f"M={M}: K1 P={P} W={W_} W_PAD={max(1 << (W_ - 1).bit_length(), 1)} K1_FAST={S.K1_FAST} "
            f"trigger-first min tokens {TREE_MIN} -> hc_attn K1 {'trigger-first' if M >= TREE_MIN else 'wait-first'}")
        graphs = {}
        ws = sorted(set([float(w) for w in args.ws] + [wtab[M]["W_crit_p10"], wtab[M]["W_crit_p50"]]))
        for w in ws:
            for arm in arms:
                early, tf, gate = arm_params(arm, M)
                set_gate(gate)
                graphs[(w, arm)] = capture(lambda: ch.run(b, M, w, early, tf, k1=arm_k1(arm)))
        set_gate(TREE_MIN)
        jit = {f"W{k[0]:g}_{k[1]}": v[2] for k, v in graphs.items() if v[2]}
        log(f"M={M}: captured {len(graphs)} graphs, ws {ws}, arm params "
            f"{ {a: arm_params(a, M) for a in arms} }, K1 JIT growth during capture {jit or 'none'}")
        for w in ws:
            for arm in arms:
                for _ in range(5):
                    graphs[(w, arm)][0].replay()
            torch.cuda.synchronize()
            times = {a: [] for a in arms}
            for r in range(args.reps):
                seq = arms if r % 2 == 0 else list(reversed(arms))
                for arm in seq:
                    g = graphs[(w, arm)][0]
                    e0 = torch.cuda.Event(enable_timing=True)
                    e1 = torch.cuda.Event(enable_timing=True)
                    e0.record()
                    for _ in range(args.inner):
                        g.replay()
                    e1.record()
                    e1.synchronize()
                    times[arm].append(e0.elapsed_time(e1) / args.inner)
            for arm in arms:
                t = times[arm]
                med = statistics.median(t)
                table[f"M{M}_W{w:g}_{arm}"] = {
                    "step_ms": med, "us_per_boundary": med * 1000 / ch.n,
                    "p10_us_per_boundary": pct(t, 0.1) * 1000 / ch.n, "p90_us_per_boundary": pct(t, 0.9) * 1000 / ch.n,
                    "noise": (pct(t, 0.9) - pct(t, 0.1)) / med}
            line = "  ".join(f"{a}: {table[f'M{M}_W{w:g}_{a}']['us_per_boundary']:.2f}"
                             f" [{table[f'M{M}_W{w:g}_{a}']['p10_us_per_boundary']:.2f},"
                             f"{table[f'M{M}_W{w:g}_{a}']['p90_us_per_boundary']:.2f}]" for a in arms)
            log(f"M={M} W={w:g}us us/boundary {line}")
        del graphs
        torch.cuda.empty_cache()
    pair_arm = args.pair_arm if args.pair_arm in arms else "tree"
    batch, gate, pair_w = {}, {}, {}
    for M in args.batches:
        w50, w10 = wtab[M]["W_crit_p50"], wtab[M]["W_crit_p10"]
        pw = w50 if args.wcrit else args.pair_w
        pair_w[str(M)] = pw
        e = table[f"M{M}_W{pw:g}_{pair_arm}"]
        batch[str(M)] = {"kernel_ms": e["step_ms"], "noise_floor": e["noise"],
                         "us_per_boundary": e["us_per_boundary"], "W_us": pw}
        g = {}
        if "base" in arms:
            eb = table[f"M{M}_W{pw:g}_base"]
            batch[str(M)]["in_job_base_us_per_boundary"] = eb["us_per_boundary"]
            batch[str(M)]["in_job_gain_vs_base"] = (eb["us_per_boundary"] - e["us_per_boundary"]) / eb["us_per_boundary"]
        if "a" in arms and "base" in arms:
            g["a_on"], g["a_gain_vs_base_p50"] = beyond("a", "base", table, f"M{M}_W{w50:g}")
        for wl, wv in (("p50", w50), ("p10", w10)):
            key = f"M{M}_W{wv:g}"
            t = table.get(f"{key}_tree")
            for other in ("ideal", "tree_fast0", "fold", "tree_k1"):
                o = table.get(f"{key}_{other}")
                if t and o:
                    d = t["us_per_boundary"] - o["us_per_boundary"]
                    g[f"tree_minus_{other}_us_{wl}"] = round(d, 3)
                    g[f"tree_minus_{other}_us_{wl}_p10p90"] = [
                        round(t["p10_us_per_boundary"] - o["p90_us_per_boundary"], 3),
                        round(t["p90_us_per_boundary"] - o["p10_us_per_boundary"], 3)]
                    g[f"tree_vs_{other}_beyond_noise_{wl}"] = abs(d) / t["us_per_boundary"] > max(t["noise"], o["noise"])
        if "ab" in arms and "a" in arms:
            b50, g["ab_gain_vs_a_p50"] = beyond("ab", "a", table, f"M{M}_W{w50:g}")
            worse10, g["a_gain_vs_ab_p10"] = beyond("a", "ab", table, f"M{M}_W{w10:g}")
            g["b_on"] = bool(b50 and not worse10)
            g["tree_runs_b"] = M >= TREE_MIN
            g["tree_matches_gate"] = g["tree_runs_b"] == g["b_on"]
        gate[str(M)] = g
        log(f"GATE M={M} at W_crit p50 {w50} / p10 {w10}: {g}")
    print(json.dumps({"batch": batch, "output": {"class": "exact"}, "pair_arm": pair_arm, "pair_w_us": pair_w,
                      "boundaries": ch.n, "trigger_first_in_tree": HAS_TF, "tree_k1_min_tokens": TREE_MIN,
                      "gate": gate, "W_crit": wtab, "wcrit_detail": {k: v for k, v in wc.items() if k != "W_crit"},
                      "table": table}))


# ------------------------------------------------------------------------------------------------- check
def flat(outs):
    r = []
    for t in outs:
        r.extend(x for x in t if x is not None)
    return r


def nbad(a, b):
    return sum(int((x.view(torch.uint8) != y.view(torch.uint8)).sum().item()) for x, y in zip(a, b))


def nonfinite(vals):
    return sum(int((~torch.isfinite(v.float())).sum().item()) for v in vals)


def poison(b, outs):
    for x in b["x"]:
        x.fill_(float("nan"))
    for t in flat(outs):
        t.fill_(float("nan"))


def check(args):
    assert HAS_TF, "check needs the candidate tree"
    ch = Chain(args.boundaries, args.ckpt, args.P)
    failures = []
    report = {}
    for M in args.batches:
        b = ch.bufs(M)
        Chain.fill(b, 1)
        cfgs = []
        for w in args.ws:
            for arm in ("base", "a", "ab", "tree"):
                cfgs.append((f"{arm}_W{w:g}", w, *arm_params(arm, M), "1"))
        cfgs.append((f"ab_W{args.ws[-1]:g}_PDL0", args.ws[-1], True, True, 1, "0"))
        cfgs.append((f"tree_W{args.ws[-1]:g}_PDL0", args.ws[-1], *arm_params("tree", M), "0"))
        graphs = {}
        jit = {}
        for name, w, early, tf, gate, pdl in cfgs:
            os.environ["TRTLLM_ENABLE_PDL"] = pdl
            set_gate(gate)
            ch.run(b, M, w, early, tf)  # eager warm (JIT) first, as the CUDA-graph runner does
            torch.cuda.synchronize()
            g, outs, grew = capture(lambda: ch.run(b, M, w, early, tf))
            graphs[name] = (g, outs)
            jit[name] = grew
        set_gate(TREE_MIN)
        os.environ["TRTLLM_ENABLE_PDL"] = "1"
        report_tree = {"tree_arm_params": arm_params("tree", M), "tree_k1_trigger_first": HAS_TF and M >= TREE_MIN}
        jit_bad = {k: v for k, v in jit.items() if v}
        if jit_bad:
            failures.append(f"M={M} K1 JIT during capture {jit_bad}")
        mism = {}
        n_cmp = 0
        for it in range(args.iters):
            seed = 100 + it
            Chain.fill(b, seed)
            ref = [t.clone() for t in flat(ch.run(b, M, 0, False, False, direct_x=True))]
            torch.cuda.synchronize()
            n_cmp = len(ref)
            base_vals = None
            for name, *_ in cfgs:
                g, outs = graphs[name]
                Chain.fill(b, seed)
                poison(b, outs)
                g.replay()
                torch.cuda.synchronize()
                vals = flat(outs)
                bad_ref = nbad(vals, ref)
                if base_vals is None:  # first cfg is base_W<first>
                    base_vals = [v.clone() for v in vals]
                bad_base = nbad(vals, base_vals)
                nf = nonfinite(vals)
                mism.setdefault(name, []).append(bad_ref + bad_base + nf)
                if bad_ref or bad_base or nf:
                    failures.append(f"M={M} {name} iter {it}: bad_ref {bad_ref} bad_base {bad_base} nonfinite {nf}")
        report[f"M{M}"] = {"max_mismatch_bytes_over_iters": {k: max(v) for k, v in mism.items()},
                           "iters": args.iters, "jit_in_capture": jit, "graphs": len(graphs),
                           "tensors_compared_per_replay": n_cmp, **report_tree}
        log(f"M={M}: {json.dumps(report[f'M{M}'])}")
        for la in (("ab", "tree") if M in (1, 8) else ()):
            g, outs = graphs[f"{la}_W{args.ws[0]:g}"]
            Chain.fill(b, 7)
            t0 = time.time()
            for _ in range(args.long_replays):
                g.replay()
            torch.cuda.synchronize()
            dt = time.time() - t0
            vals = [v.clone() for v in flat(outs)]
            ref = flat(ch.run(b, M, 0, False, False, direct_x=True))
            bad = nbad(vals, ref)
            report[f"M{M}"][f"long_replays_{la}"] = {"n": args.long_replays, "s": round(dt, 2), "final_bad_bytes": bad}
            log(f"M={M} {la}: {args.long_replays} back-to-back replays in {dt:.2f}s, final bad {bad}")
            if bad:
                failures.append(f"M={M} {la} long replays final mismatch {bad}")
        del graphs
        torch.cuda.empty_cache()
    neg = {}
    for ctl in ("k1nowait", "lowmskip"):
        cmd = [sys.executable, os.path.abspath(__file__), "--mode", "negctl", "--ctl", ctl, "--ckpt", args.ckpt,
               "--boundaries", "8", "--batches", "1", "4"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        last = (r.stdout.strip().splitlines() or ["{}"])[-1]
        try:
            res = json.loads(last)
        except Exception:  # noqa: BLE001
            res = {"parse_error": last[-500:], "stderr": r.stderr[-2000:]}
        res["rc"] = r.returncode
        neg[ctl] = res
        log(f"negctl {ctl}: {json.dumps(res)[:600]}")
        if not res.get("caught"):
            failures.append(f"negative control {ctl} not caught")
    report["negative_controls"] = neg
    print(json.dumps({"passed": not failures, "failures": failures[:20],
                      "compared": "every boundary's K1 layer_input + A/B outputs (qr, q, swa, index_q), graphs of "
                                  "arms base/a/ab(forced)/tree(per-M gate) x W and ab/tree PDL0, NaN-poisoned per replay, bitwise vs eager "
                                  "no-overlap reference and vs the base graph, every iteration",
                      "report": report}))


def check352(args):
    """Idea 352: arms tree / tree_fast0 / ideal, NaN-poisoned, >= args.iters replays per M and W, new inputs every
    iteration. Compares bitwise: each arm's A/B outputs (qr, q, swa, index_q) with the eager no-overlap reference
    (main's default K1 -> A -> B on x_prev = src) and with the tree graph of the same iteration; the K1 layer_input
    of tree and tree_fast0 with the reference; ideal's precomputed layer_input with the reference."""
    ch = Chain(args.boundaries, args.ckpt, args.P)
    arms = ("tree", "tree_fast0", "ideal") + (("fold", "tree_k1") if HAS_FOLD else ())
    failures, report = [], {}
    for M in args.batches:
        b = ch.bufs(M)
        Chain.fill(b, 1)
        ch.precompute(b)
        graphs, jit = {}, {}
        for w in args.ws:
            for arm in arms:
                early, tf, gate = arm_params(arm, M)
                set_gate(gate)
                ch.run(b, M, w, early, tf, k1=arm_k1(arm))  # eager warm (JIT) first
                torch.cuda.synchronize()
                g, outs, grew = capture(lambda: ch.run(b, M, w, early, tf, k1=arm_k1(arm)))
                graphs[(w, arm)] = (g, outs)
                jit[f"W{w:g}_{arm}"] = grew
        set_gate(TREE_MIN)
        bad_jit = {k: v for k, v in jit.items() if v}
        if bad_jit:
            failures.append(f"M={M} K1 JIT during capture {bad_jit}")
        mism = {}
        for it in range(args.iters):
            seed = 500 + it
            Chain.fill(b, seed)
            ref_t = ch.run(b, M, 0, False, False, direct_x=True)
            ref_li = [t[0].clone() for t in ref_t]
            ref_ab = [x.clone() for t in ref_t for x in t[1:] if x is not None]
            torch.cuda.synchronize()
            Chain.fill(b, seed)
            ch.precompute(b)
            mism.setdefault("li_pre_vs_ref", []).append(nbad(b["li_pre"], ref_li))
            for w in args.ws:
                tree_ab = None
                for arm in arms:
                    g, outs = graphs[(w, arm)]
                    Chain.fill(b, seed)
                    for x in b["x"]:
                        x.fill_(float("nan"))
                    for t in outs:
                        for k, x in enumerate(t):
                            if x is not None and not (k == 0 and arm == "ideal"):  # ideal's li is the input
                                x.fill_(float("nan"))
                    g.replay()
                    torch.cuda.synchronize()
                    ab = [x for t in outs for x in t[1:] if x is not None]
                    if tree_ab is None:
                        tree_ab = [x.clone() for x in ab]
                    r = {"ab_vs_ref": nbad(ab, ref_ab), "ab_vs_tree": nbad(ab, tree_ab), "nonfinite": nonfinite(ab)}
                    if arm != "ideal":
                        r["li_vs_ref"] = nbad([t[0] for t in outs], ref_li)
                    for k, v in r.items():
                        mism.setdefault(f"W{w:g}_{arm}_{k}", []).append(v)
                    if any(r.values()):
                        failures.append(f"M={M} W={w:g} {arm} iter {it}: {r}")
        report[f"M{M}"] = {"iters": args.iters, "max_mismatch_over_iters": {k: max(v) for k, v in mism.items()},
                           "jit_in_capture": jit, "tensors_ab_per_replay": len(ref_ab),
                           "arm_params": {a: arm_params(a, M) for a in arms},
                           "tree_runs_fold": bool(HAS_FOLD and LMG.k1fold_applies(
                               M, ch.P, ch.probes[0]._wq_a_wkv[0]))}
        log(f"M={M}: {json.dumps(report[f'M{M}'])}")
        del graphs
        torch.cuda.empty_cache()
    print(json.dumps({"passed": not failures, "failures": failures[:20],
                      "compared": "idea 352 arms tree (fast K1) / tree_fast0 (TRTLLM_MHC_K1_FAST=0 kernel) / ideal "
                                  "(no K1, A on precomputed layer_input): A/B outputs qr,q,swa,index_q of every "
                                  "boundary bitwise vs eager no-overlap reference and vs tree, K1 layer_input vs "
                                  "reference, NaN-poisoned x_prev and outputs, new inputs every iteration",
                      "report": report}))


def negctl(args):
    """A control that must be CAUGHT: K1 triggers first and skips its wait, or A (low-M) skips its wait."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import low_m_gemm as LM

    ch = Chain(args.boundaries, args.ckpt, args.P)
    set_gate(1)  # the controls run K1 trigger-first at every M, as before the per-M gate
    caught = {}
    for M in args.batches:
        b = ch.bufs(M)
        Chain.fill(b, 1)
        if args.ctl == "k1nowait":
            fn = lambda: ch.run(b, M, 10, True, True, k1_skip_wait=True)  # noqa: E731
        else:
            LM._TRIGGER_AT_START = 3  # bit 1 trigger at start + bit 0 skip griddepcontrol.wait (test-only)
            fn = lambda: ch.run(b, M, 10, True, True)  # noqa: E731
        g, outs, _ = capture(fn)
        LM._TRIGGER_AT_START = 2
        bad_total = 0
        for it in range(4):
            Chain.fill(b, 300 + it)
            ref = [t.clone() for t in flat(ch.run(b, M, 0, False, False, direct_x=True))]
            Chain.fill(b, 300 + it)
            poison(b, outs)
            g.replay()
            torch.cuda.synchronize()
            vals = flat(outs)
            bad_total += nbad(vals, ref) + nonfinite(vals)
        caught[str(M)] = bad_total
        log(f"negctl {args.ctl} M={M}: mismatching bytes+nonfinite over 4 replays = {bad_total}")
        del g
    print(json.dumps({"ctl": args.ctl, "caught": all(v > 0 for v in caught.values()), "per_M": caught}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="bench")
    p.add_argument("--ckpt", default="")
    p.add_argument("--boundaries", type=int, default=36)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--ws", type=float, nargs="+", default=[2, 5, 10])
    p.add_argument("--P", type=int, default=480)
    p.add_argument("--reps", type=int, default=40)
    p.add_argument("--inner", type=int, default=5)
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--long-replays", type=int, default=1000)
    p.add_argument("--pair-arm", default="tree")
    p.add_argument("--pair-w", type=float, default=5)
    p.add_argument("--wcrit", type=int, default=1, help="1: judge/pair at the W_crit table (wcrit.py)")
    p.add_argument("--ctl", default="k1nowait")
    p.add_argument("--arms", nargs="+", default=None, help="bench arms (default base a ab tree)")
    args = p.parse_args()
    torch.cuda.set_device(0)
    pdl_ext.ext()
    if args.mode == "bench":
        bench(args)
    elif args.mode == "check":
        check(args)
    elif args.mode == "check352":
        check352(args)
    elif args.mode == "both352":
        check352(args)
        bench(args)
    else:
        negctl(args)


if __name__ == "__main__":
    main()
